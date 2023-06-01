import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np

import coeffs as coeffs
from p2dSolver_Newton import newton
from reorder import reorder_tot
from unpack import unpack_vars
from globalValues import T_ref
from settings import dxA, dxP, dxO, dxN, dxZ, drP, drN, delta_t, cutOff


def p2d_reorder_fn(peq, oeq, neq, lu_pe, lu_ne, temp_p, temp_n, gamma_p_vec, gamma_n_vec, fn_fast, jac_fn, tol=1e-7, verbose=False):
    
    @jax.jit
    def cmat_format_p(cmat):
        val=cmat.at[0:dxP * (drP + 2):drP + 2].set(0)

        val=val.at[drP + 1:dxP * (drP + 2):drP + 2].set(0)
        
        return val

    @jax.jit
    def cmat_format_n(cmat):
        val=cmat.at[0:dxN * (drN + 2):drN + 2].set(0)

        val=val.at[drN + 1:dxN * (drN + 2):drN + 2].set(0)

        return val

    @jax.jit
    def form_c2_p_jit(temp, j, T):
        Deff_vec = vmap(coeffs.solidDiffCoeff)(peq.Ds * jnp.ones(dxP), peq.ED * jnp.ones(dxP), T[1:dxP + 1])
        fn = lambda j, temp, Deff: -(j * temp / Deff)
        val = vmap(fn, (0, None, 0), 1)(j, temp, Deff_vec)

        return val

    @jax.jit
    def form_c2_n_jit(temp, j, T):
        Deff_vec = vmap(coeffs.solidDiffCoeff)(neq.Ds * jnp.ones(dxN), neq.ED * jnp.ones(dxN), T[1:dxN + 1])
        fn = lambda j, temp, Deff: -(j * temp / Deff)
        val = vmap(fn, (0, None, 0), 1)(j, temp, Deff_vec)
        return val

    U_fast = jnp.hstack(
        [

            peq.ce_0 + jnp.zeros(dxP + 2),
            oeq.ce_0 + jnp.zeros(dxO + 2),
            neq.ce_0 + jnp.zeros(dxN + 2),

            jnp.zeros(dxP),
            jnp.zeros(dxN),

            jnp.zeros(dxP),
            jnp.zeros(dxN),

            jnp.zeros(dxP + 2) + peq.open_circuit_poten(peq.cavg, peq.cavg, T_ref, peq.cmax),
            jnp.zeros(dxN + 2) + neq.open_circuit_poten(neq.cavg, neq.cavg, T_ref, neq.cmax),

            jnp.zeros(dxP + 2) + 0,
            jnp.zeros(dxO + 2) + 0,
            jnp.zeros(dxN + 2) + 0,

            T_ref + jnp.zeros(dxA + 2),
            T_ref + jnp.zeros(dxP + 2),
            T_ref + jnp.zeros(dxO + 2),
            T_ref + jnp.zeros(dxN + 2),
            T_ref + jnp.zeros(dxZ + 2)

        ])

    cmat_pe = peq.cavg * jnp.ones(dxP * (drP + 2))
    cmat_ne = neq.cavg * jnp.ones(dxN * (drN + 2))

    idx_tot = reorder_tot(dxP, dxN, dxO, dxA, dxZ)
    re_idx = jnp.argsort(idx_tot)


    voltages = []
    tempMax = []
    flux = []
    overPots = []
    tempN = []
    times = []
    t=0

    
    while True:
    # for i in range(0, int(steps)):
        
        cmat_rhs_pe = cmat_format_p(cmat_pe).block_until_ready()
        cmat_rhs_ne = cmat_format_n(cmat_ne).block_until_ready()

        cI_pe_vec = lu_pe.solve(np.asarray(cmat_rhs_pe))
        cI_ne_vec = lu_ne.solve(np.asarray(cmat_rhs_ne))

        cs_pe1 = (cI_pe_vec[drP:dxP * (drP + 2):drP + 2] + cI_pe_vec[drP + 1:dxP * (drP + 2):drP + 2]) / 2
        cs_ne1 = (cI_ne_vec[drN:dxN * (drN + 2):drN + 2] + cI_ne_vec[drN + 1:dxN * (drN + 2):drN + 2]) / 2

        try:
            U_fast, fail = newton(fn_fast, jac_fn, U_fast, cs_pe1, cs_ne1, gamma_p_vec, gamma_n_vec, idx_tot,re_idx, tol, verbose)

        except (ValueError):
            fail = 'Nan'
            if verbose:
                print("nan/inf solution")

        Tvec, Tvec_pe, Tvec_ne, phis_pe, phis_ne, jvec_pe, jvec_ne, etavec_ne = unpack_vars(U_fast)

        cII_p = form_c2_p_jit(temp_p, jvec_pe, Tvec_pe).block_until_ready()
        cII_n = form_c2_n_jit(temp_n, jvec_ne, Tvec_ne).block_until_ready()
        cmat_pe = jnp.reshape(cII_p, [dxP * (drP + 2)], order="F").block_until_ready() + cI_pe_vec
        cmat_ne = jnp.reshape(cII_n, [dxN * (drN + 2)], order="F").block_until_ready() + cI_ne_vec
        
        v = float(phis_pe[1] - phis_ne[dxN])
        voltages.append(v)
        tempMax.append(float(jnp.max(Tvec)))
        flux.append(float(jvec_ne[1]))
        overPots.append(float(etavec_ne[1]))
        tempN.append(float(Tvec_ne[1]))
        times.append(t)
        t += delta_t
        
        if v < cutOff:
            if verbose:
                print("Done reordered simulation\n")
            break
        
        if (fail == ''):
            pass

        else:
            if verbose: 
                print('Premature end of run: ' + fail)
                print("time point:", t, "Last V:", v,"\n")
            break

    return U_fast, cmat_pe, cmat_ne, np.array(voltages), np.array(tempMax), np.array(flux), np.array(overPots), np.array(tempN), np.array(times), [fail,t,v]
