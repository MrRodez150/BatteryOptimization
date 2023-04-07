import timeit

import jax
import jax.numpy as jnp
from jax import vmap
from jax.numpy.linalg import norm
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix
from scipy.linalg import solve_banded
import numpy as np

from config import div_x_elec, div_x_sep, div_x_cc, div_r, delta_t, tol, maxit
from globalValues import T_ref, cutOff
from coeffs import sDiffCoeff
from mainAux import reorder_tot, unpack_vars
from derivatives import partials
from compute import compute_jac, p2d_init

pdr = div_r
ndr = div_r
pd = div_x_elec
nd = div_x_elec
od = div_x_sep
ad = div_x_cc
zd = div_x_cc




@jax.jit
def reorder_vec(v, idx):
    return v[idx]

def newton(fn, jac_fn, Umat, cs_pe1, cs_ne1, gamma_p, gamma_n, idx, re_idx):
    res = 100
    fail = False
    Uold = Umat

    J = jac_fn(Umat, Uold, cs_pe1, cs_ne1).block_until_ready()
    y = fn(Umat, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
    # Initial J and f evaluation

    y = reorder_vec(y, idx).block_until_ready()
    # Reordering overhead

    delta = solve_banded((11, 11), J, y)

    delta_reordered = reorder_vec(delta, re_idx).block_until_ready()
    Umat = Umat - delta_reordered

    count = 1

    while (count <= maxit and res > tol):

        J = jac_fn(Umat, Uold, cs_pe1, cs_ne1).block_until_ready()
        y = fn(Umat, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()

        y = reorder_vec(y, idx).block_until_ready()

        res = norm(y / norm(Umat, jnp.inf), jnp.inf)

        delta = solve_banded((11, 11), J, y)

        delta_reordered = reorder_vec(delta, re_idx).block_until_ready()

        Umat = Umat - delta_reordered

        count = count + 1

    if jnp.any(jnp.isnan(delta)):
        fail = True
        print("nan solution")

    elif max(abs(jnp.imag(delta))) > 0:
        fail = True
        print("solution complex")

    elif res > tol:
        fail = True
        print('Newton fail: no convergence')
        print(res,' greater than tolerance: ',tol)
        print('Iterations made: ',count)
    else:
        fail = False
    

    return Umat, fail





def p2d_fn_solver(p_eqn, n_eqn, o_eqn, a_eqn, z_eqn, Icell):
    
    start_t = timeit.default_timer()


    @jax.jit
    def c_p_mat_format(cmat):
        val = cmat.at[0 : pd*(pdr+2) : pdr+2].set(0)
        val = val.at[pdr+1 : pd*(pdr+2) : pdr+2].set(0)
        return val

    @jax.jit
    def c_n_mat_format(cmat):
        val = cmat.at[0 : nd*(ndr+2) : ndr+2].set(0)
        val = val.at[ndr+1 : nd*(ndr+2) : ndr+2].set(0)
        return val

    @jax.jit
    def form_c2_p_jit(temp, j, T):
        Deff_vec = vmap(sDiffCoeff)(p_eqn.Ds * jnp.ones(pd), p_eqn.ED * jnp.ones(pd), T[1:pd + 1])
        fn = lambda j, temp, Deff: -(j * temp / Deff)
        #        val=vmap(fn,(0,1,0),1)(j,temp,Deff_vec)
        val = vmap(fn, (0, None, 0), 1)(j, temp, Deff_vec)

        return val

    @jax.jit
    def form_c2_n_jit(temp, j, T):
        Deff_vec = vmap(sDiffCoeff)(n_eqn.Ds * jnp.ones(nd), n_eqn.ED * jnp.ones(nd), T[1:nd + 1])
        fn = lambda j, temp, Deff: -(j * temp / Deff)
        #        val=vmap(fn,(0,1,0),1)(j,temp,Deff_vec)
        val = vmap(fn, (0, None, 0), 1)(j, temp, Deff_vec)
        return val








    #Create initial conditions matrix

    Umat = jnp.hstack(
        [
            #ce
            p_eqn.ce_0 + jnp.zeros(pd + 2),
            o_eqn.ce_0 + jnp.zeros(od + 2),
            n_eqn.ce_0 + jnp.zeros(nd + 2),
            #j
            jnp.zeros(pd),
            jnp.zeros(nd),
            #eta
            jnp.zeros(pd),
            jnp.zeros(nd),
            #phis
            jnp.zeros(pd + 2) + p_eqn.openCircPot_start(),
            jnp.zeros(nd + 2) + n_eqn.openCircPot_start(),
            #phie
            jnp.zeros(pd + 2),
            jnp.zeros(od + 2),
            jnp.zeros(nd + 2),
            #T
            T_ref + jnp.zeros(ad + 2),
            T_ref + jnp.zeros(pd + 2),
            T_ref + jnp.zeros(od + 2),
            T_ref + jnp.zeros(nd + 2),
            T_ref + jnp.zeros(zd + 2)

        ])

    fn = p2d_init(p_eqn,n_eqn,o_eqn,a_eqn,z_eqn,Icell)

    part_fn = partials(a_eqn, p_eqn, o_eqn, n_eqn, z_eqn)

    cs_p_g_vec = p_eqn.gamma * jnp.ones(pd)
    cs_n_g_vec = n_eqn.gamma * jnp.ones(nd)

    jac_fn = compute_jac(cs_p_g_vec,cs_n_g_vec,part_fn,Icell)

    # cs solve

    cs_p_lu = splu(csc_matrix(p_eqn.A))
    cs_p_temp = p_eqn.temp_sol
    cmat_p = p_eqn.cavg * jnp.ones(pd * (pdr + 2))

    cs_n_lu = splu(csc_matrix(n_eqn.A))
    cs_n_temp = n_eqn.temp_sol
    cmat_n = n_eqn.cavg * jnp.ones(nd * (ndr + 2))

    # order and reorder indexes

    idx_tot = reorder_tot()
    re_idx = jnp.argsort(idx_tot)

    # solve for each time interval

    voltages = []
    tempMax = []
    flux = []
    overPots = []
    tempN = []
    times = []
    t = 0

    while True:
#    for i in range(0, int(div_t)+1):
        
        cmat_rhs_p = c_p_mat_format(cmat_p).block_until_ready()
        cmat_rhs_n = c_n_mat_format(cmat_n).block_until_ready()
        
        cI_pe_vec = cs_p_lu.solve(np.asarray(cmat_rhs_p))
        cI_ne_vec = cs_n_lu.solve(np.asarray(cmat_rhs_n))
        
        cs_pe1 = (cI_pe_vec[pdr:pd * (pdr + 2):pdr + 2] + cI_pe_vec[pdr + 1:pd * (pdr + 2):pdr + 2]) / 2
        cs_ne1 = (cI_ne_vec[ndr:nd * (ndr + 2):ndr + 2] + cI_ne_vec[ndr + 1:nd * (ndr + 2):ndr + 2]) / 2
        
        try:
            Umat, fail = newton(fn, jac_fn, Umat, cs_pe1, cs_ne1, cs_p_g_vec, cs_n_g_vec, idx_tot, re_idx)

        except (ValueError):
            fail = True
            print("nan/inf solution")
        
        Tvec, Tvec_p, Tvec_n, phis_p, phis_n, jvec_p, jvec_n, etavec_n = unpack_vars(Umat)

        cII_p = form_c2_p_jit(cs_p_temp, jvec_p, Tvec_p).block_until_ready()
        cII_n = form_c2_n_jit(cs_n_temp, jvec_n, Tvec_n).block_until_ready()

        cmat_p = jnp.reshape(cII_p, [pd * (pdr + 2)], order="F").block_until_ready() + cI_pe_vec
        cmat_n = jnp.reshape(cII_n, [nd * (ndr + 2)], order="F").block_until_ready() + cI_ne_vec

        v = float(phis_p[1] - phis_n[nd])
        voltages.append(v)
        tempMax.append(float(jnp.max(Tvec)))
        flux.append(float(jvec_n[1]))
        overPots.append(float(etavec_n[1]))
        tempN.append(float(Tvec_n[1]))
        times.append(t)
        t += delta_t

        if (v <= cutOff):
            break

        if (fail):
            print('Premature end of run')
            print("timestep:", len(times),'\n')
            break




    end_t = timeit.default_timer()

    sim_time = end_t - start_t

    print("Simulation done\n")

    return np.array(voltages), np.array(tempMax), np.array(flux), np.array(overPots), np.array(tempN), np.array(times), sim_time, fail
