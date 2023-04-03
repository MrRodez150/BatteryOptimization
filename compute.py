from jax import jit
import jax.numpy as jnp

from config import div_x_cc, div_x_elec, div_x_sep
from mainAux import unpack
from residualFunc import residualFunction
from builder import *

pd = div_x_elec
nd = div_x_elec
od = div_x_sep
ad = div_x_cc
zd = div_x_cc




def p2d_init(p_eqn,n_eqn,o_eqn,a_eqn,z_eqn,Icell):

    res_fn = residualFunction(p_eqn,n_eqn,o_eqn,a_eqn,z_eqn,Icell)

    def solver(U, U_old, cs_p1, cs_ne1, gamma_p, gamma_n):
        
        val = jnp.zeros(res_fn.spaces)

        ce_vec_p, ce_vec_o, ce_vec_n,\
        T_vec_a, T_vec_p, T_vec_o, T_vec_n, T_vec_z,\
        phie_p, phie_o, phie_n,\
        phis_p, phis_n,\
        j_vec_p, j_vec_n,\
        eta_p, eta_n = unpack(U)

        ce_vec_p_old, ce_vec_o_old, ce_vec_n_old,\
        T_vec_a_old, T_vec_p_old, T_vec_o_old, T_vec_n_old, T_vec_z_old,\
        _, _, _,\
        _, _,\
        _, _,\
        _, _ = unpack(U_old)

        val = res_fn.res_ce_p(val, ce_vec_p, T_vec_p, j_vec_p, ce_vec_p_old, ce_vec_o, T_vec_o)
        val = res_fn.res_ce_o(val, ce_vec_o, T_vec_o, ce_vec_o_old, ce_vec_p, T_vec_p, ce_vec_n, T_vec_n)
        val = res_fn.res_ce_n(val, ce_vec_n, T_vec_n, j_vec_n, ce_vec_n_old, ce_vec_o, T_vec_o)
        
        val = res_fn.res_T_a(val, T_vec_a, T_vec_a_old, T_vec_p)
        val = res_fn.res_T_p(val, T_vec_p, ce_vec_p, phie_p, phis_p, j_vec_p, eta_p, cs_p1, gamma_p, T_vec_p_old, T_vec_a, T_vec_o)
        val = res_fn.res_T_o(val, T_vec_o, ce_vec_o, phie_o, T_vec_o_old, T_vec_p, T_vec_n )
        val = res_fn.res_T_n(val, T_vec_n, ce_vec_n, phie_n, phis_n, j_vec_n, eta_n, cs_ne1, gamma_n, T_vec_n_old, T_vec_z, T_vec_o)
        val = res_fn.res_T_z(val, T_vec_z, T_vec_z_old, T_vec_n)
        
        val = res_fn.res_phie_p(val, ce_vec_p, phie_p, T_vec_p, j_vec_p, ce_vec_o,phie_o, T_vec_o)
        val = res_fn.res_phie_o(val, ce_vec_o, phie_o, T_vec_o, phie_p, phie_n)
        val = res_fn.res_phie_n(val, ce_vec_n, phie_n, T_vec_n, j_vec_n, ce_vec_o, phie_o, T_vec_o)
    
        val = res_fn.res_phis(val, phis_p, j_vec_p, phis_n, j_vec_n)
    
        val = res_fn.res_j(val, j_vec_p, ce_vec_p, T_vec_p, eta_p, cs_p1, gamma_p, j_vec_n, ce_vec_n, T_vec_n, eta_n, cs_ne1, gamma_n)
        val = res_fn.res_eta(val, eta_p, phis_p, phie_p, T_vec_p, j_vec_p, cs_p1, gamma_p, eta_n, phis_n, phie_n, T_vec_n, j_vec_n, cs_ne1, gamma_n)

        return val
    
    jit_solver = jit(solver)
    
    return jit_solver



# @partial(jax.jit, static_argnums=(4,5,6))
def compute_jac(gamma_p_vec, gamma_n_vec, d, Iapp):
    @jit
    def jacfn(U, Uold, cs_pe1, cs_ne1):
        
        Jnew = jnp.zeros([23, len(U)])

        ce_vec_p, ce_vec_o, ce_vec_n,\
        T_vec_a, T_vec_p, T_vec_o, T_vec_n, T_vec_z,\
        phie_p, phie_o, phie_n,\
        phis_p, phis_n,\
        j_vec_p, j_vec_n,\
        eta_p, eta_n = unpack(U)

        ce_vec_p_old, ce_vec_o_old, ce_vec_n_old,\
        T_vec_a_old, T_vec_p_old, T_vec_o_old, T_vec_n_old, T_vec_z_old,\
        _, _, _,\
        _, _,\
        _, _,\
        _, _ = unpack(Uold)

        #build positive electrode

        arg_ce_p = [ce_vec_p[0:pd], ce_vec_p[1:pd + 1], ce_vec_p[2:pd + 2],
                    T_vec_p[0:pd], T_vec_p[1:pd + 1], T_vec_p[2:pd + 2],
                    j_vec_p[0:pd],
                    ce_vec_p_old[1:pd + 1]]
        
        Jnew = build_dce_p(Jnew, d['dce_p'](*arg_ce_p), ad, pd)


        arg_j_p = [j_vec_p[0:pd],
                   ce_vec_p[1:pd + 1],
                   T_vec_p[1:pd + 1],
                   eta_p[0:pd],
                   cs_pe1,
                   gamma_p_vec]
        
        Jnew = build_dj_p(Jnew, d['dj_p'](*arg_j_p), ad, pd)


        arg_eta_p = [eta_p[0:pd],
                     phis_p[1:pd + 1],
                     phie_p[1:pd + 1],
                     T_vec_p[1:pd + 1],
                     j_vec_p[0:pd],
                     cs_pe1,
                     gamma_p_vec]
        
        Jnew = build_deta_p(Jnew, d['deta_p'](*arg_eta_p), ad, pd)


        arg_phis_p = [phis_p[0:pd], phis_p[1:pd + 1], phis_p[2:pd + 2],
                      j_vec_p[0:pd]]
        
        Jnew = build_dphis_p(Jnew, d['dphis_p'](*arg_phis_p), ad, pd)


        arg_phie_p = [ce_vec_p[0:pd], ce_vec_p[1:pd + 1], ce_vec_p[2:pd + 2],
                      phie_p[0:pd], phie_p[1:pd + 1], phie_p[2:pd + 2],
                      T_vec_p[0:pd], T_vec_p[1:pd + 1], T_vec_p[2:pd + 2],
                      j_vec_p[0:pd]]
        
        Jnew = build_dphie_p(Jnew, d['dphie_p'](*arg_phie_p), ad, pd)


        arg_T_p = [ce_vec_p[0:pd], ce_vec_p[1:pd + 1], ce_vec_p[2:pd + 2],
                   phie_p[0:pd], phie_p[2:pd + 2],
                   phis_p[0:pd], phis_p[2:pd + 2],
                   T_vec_p[0:pd], T_vec_p[1:pd + 1], T_vec_p[2:pd + 2],
                   j_vec_p[0:pd],
                   eta_p[0:pd],
                   cs_pe1,
                   gamma_p_vec,
                   T_vec_p_old[1:pd + 1]]

        Jnew = build_dT_p(Jnew, d['dT_p'](*arg_T_p), ad, pd)

        dce_p_bc1 = d['dce_p_bc1'](ce_vec_p[0], ce_vec_p[1], 0)
        dce_p_bc2 = d['dce_p_bc2'](ce_vec_p[pd], ce_vec_p[pd + 1],
                                   T_vec_p[pd], T_vec_p[pd + 1],
                                   ce_vec_o[0], ce_vec_o[1],
                                   T_vec_o[0], T_vec_o[1])

        dphis_p_bc1 = d['dphis_p_bc1'](phis_p[0], phis_p[1], Iapp)
        dphis_p_bc2 = d['dphis_p_bc2'](phis_p[pd], phis_p[pd + 1], 0)

        dphie_p_bc1 = d['dphie_p_bc1'](phie_p[0], phie_p[1], 0)
        dphie_p_bc2 = d['dphie_p_bc2'](phie_p[pd], phie_p[pd + 1], phie_o[0], phie_o[1],
                                       ce_vec_p[pd], ce_vec_p[pd + 1], ce_vec_o[0], ce_vec_o[1],
                                       T_vec_p[pd], T_vec_p[pd + 1], T_vec_o[0], T_vec_o[1])

        dT_p_bc1 = d['dT_p_bc1'](T_vec_a[ad], T_vec_a[ad + 1], T_vec_p[0], T_vec_p[1])
        dT_p_bc2 = d['dT_p_bc2'](T_vec_p[pd], T_vec_p[pd + 1], T_vec_o[0], T_vec_o[1])

        bc_p = dict([('ce', jnp.concatenate((jnp.array(dce_p_bc1), jnp.array(dce_p_bc2)))),
                     ('phis', jnp.concatenate((jnp.array(dphis_p_bc1), jnp.array(dphis_p_bc2)))),
                     ('phie', jnp.concatenate((jnp.array(dphie_p_bc1), jnp.array(dphie_p_bc2)))),
                     ('T', jnp.concatenate((jnp.array(dT_p_bc1), jnp.array(dT_p_bc2))))])

        Jnew = build_bc_p(Jnew, bc_p, ad, pd)

        #build separator

        bc_o = d['bc_o'](ce_vec_o[0], ce_vec_o[1], ce_vec_p[pd], ce_vec_p[pd + 1])
        
        Jnew = build_bc_s(Jnew, jnp.concatenate((jnp.array(bc_o), jnp.array(bc_o))), ad, pd, od)
        

        dce_o = d['dce_o'](ce_vec_o[0:od], ce_vec_o[1:od + 1],ce_vec_o[2:od + 2],
                           T_vec_o[0:od],T_vec_o[1:od + 1], T_vec_o[2:od + 2],
                           ce_vec_o_old[1:od + 1])

        Jnew = build_dce_o(Jnew, dce_o, ad, pd, od)


        dphie_o = d['dphie_o'](ce_vec_o[0:od], ce_vec_o[1:od + 1], ce_vec_o[2:od + 2],
                               phie_o[0:od], phie_o[1:od + 1], phie_o[2:od + 2],
                               T_vec_o[0:od], T_vec_o[1:od + 1], T_vec_o[2:od + 2])
        
        Jnew = build_dphie_o(Jnew, dphie_o, ad, pd, od)


        dT_o = d['dT_o'](ce_vec_o[0:od], ce_vec_o[1:od + 1], ce_vec_o[2:od + 2],
                         phie_o[0:od], phie_o[2:od + 2],
                         T_vec_o[0:od], T_vec_o[1:od + 1], T_vec_o[2:od + 2],
                         T_vec_o_old[1:od + 1])

        Jnew = build_dT_o(Jnew, dT_o, ad, pd, od)

        #build negative electrode
        
        dce_n = d['dce_n'](ce_vec_n[0:nd], ce_vec_n[1:nd + 1], ce_vec_n[2:nd + 2],
                           T_vec_n[0:nd], T_vec_n[1:nd + 1], T_vec_n[2:nd + 2],
                           j_vec_n[0:nd],
                           ce_vec_n_old[1:nd + 1])

        Jnew = build_dce_n(Jnew, dce_n, ad, pd, od, nd)


        arg_j_n = [j_vec_n[0:nd],
                   ce_vec_n[1:nd + 1],
                   T_vec_n[1:nd + 1],
                   eta_n[0:pd],
                   cs_ne1,
                   gamma_n_vec]

        Jnew = build_dj_n(Jnew, d['dj_n'](*arg_j_n), ad, pd, od, nd)


        arg_eta_n = [eta_n[0:nd],
                     phis_p[1:nd + 1],
                     phie_n[1:nd + 1],
                     T_vec_n[1:nd + 1],
                     j_vec_n[0:nd],
                     cs_ne1,
                     gamma_n_vec]

        Jnew = build_deta_n(Jnew, d['deta_n'](*arg_eta_n), ad, pd, od, nd)


        arg_phis_n = [phis_n[0:nd], phis_n[1:nd + 1], phis_n[2:nd + 2],
                      j_vec_n[0:nd]]

        Jnew = build_dphis_n(Jnew, d['dphis_n'](*arg_phis_n), ad, pd, od, nd)


        arg_phie_n = [ce_vec_n[0:nd], ce_vec_n[1:nd + 1], ce_vec_n[2:nd + 2],
                      phie_n[0:nd], phie_n[1:nd + 1], phie_n[2:nd + 2],
                      T_vec_n[0:nd], T_vec_n[1:nd + 1], T_vec_n[2:nd + 2],
                      j_vec_n[0:nd]]

        Jnew = build_dphie_n(Jnew, d['dphie_n'](*arg_phie_n), ad, pd, od, nd)


        arg_T_n = [ce_vec_n[0:nd], ce_vec_n[1:nd + 1], ce_vec_n[2:nd + 2],
                   phie_n[0:nd], phie_n[2:nd + 2],
                   phis_n[0:nd], phis_n[2:nd + 2],
                   T_vec_n[0:nd], T_vec_n[1:nd + 1], T_vec_n[2:nd + 2],
                   j_vec_n[0:nd],
                   eta_n[0:nd],
                   cs_ne1,
                   gamma_n_vec,
                   T_vec_n_old[1:nd + 1]]

        Jnew = build_dT_n(Jnew, d['dT_n'](*arg_T_n), ad, pd, od, nd)

        
        dce_n_bc1 = d['dce_n_bc1'](ce_vec_n[0], ce_vec_n[1],
                                   T_vec_n[0], T_vec_n[1],
                                   ce_vec_o[od], ce_vec_o[od + 1],
                                   T_vec_o[od], T_vec_o[od + 1])
        dce_n_bc2 = d['dce_n_bc2'](ce_vec_n[nd], ce_vec_n[nd + 1], 0)

        dphis_n_bc1 = d['dphis_n_bc1'](phis_n[0], phis_n[1], 0)
        dphis_n_bc2 = d['dphis_n_bc2'](phis_n[nd], phis_n[nd + 1], Iapp)

        dphie_n_bc1 = d['dphie_n_bc1'](phie_n[0], phie_n[1], phie_o[od], phie_o[od + 1],
                                      ce_vec_n[0], ce_vec_n[1], ce_vec_o[od], ce_vec_o[od + 1],
                                      T_vec_n[0], T_vec_n[1], T_vec_o[od], T_vec_o[od + 1])
        dphie_n_bc2 = d['dphie_n_bc2'](phie_n[nd], phie_n[nd + 1], 0)

        dT_n_bc1 = d['dT_n_bc1'](T_vec_o[od], T_vec_o[od + 1], T_vec_n[0], T_vec_n[1])
        dT_n_bc2 = d['dT_n_bc2'](T_vec_n[nd], T_vec_n[nd + 1], T_vec_z[0], T_vec_z[1])

        bc_n = dict([('ce', jnp.concatenate((jnp.array(dce_n_bc1), jnp.array(dce_n_bc2)))),
                     ('phis', jnp.concatenate((jnp.array(dphis_n_bc1), jnp.array(dphis_n_bc2)))),
                     ('phie', jnp.concatenate((jnp.array(dphie_n_bc1), jnp.array(dphie_n_bc2)))),
                     ('T', jnp.concatenate((jnp.array(dT_n_bc1), jnp.array(dT_n_bc2))))
                     ])
        
        Jnew = build_bc_n(Jnew, bc_n, ad, pd, od, nd)

        #build current collectors

        dT_a = d['dT_a'](T_vec_a[0:ad], T_vec_a[1:ad + 1],T_vec_a[2:ad + 2],T_vec_a_old[1:ad + 1])

        Jnew = build_dT_a(Jnew, dT_a, ad)


        dT_z = d['dT_z'](T_vec_z[0:zd], T_vec_z[1:zd + 1],T_vec_z[2:zd + 2],T_vec_z_old[1:zd + 1])

        Jnew = build_dT_z(Jnew, dT_z, ad, pd, od, nd, zd)


        dT_a_bc1 = d['dT_a_bc1'](T_vec_a[0], T_vec_a[1])
        dT_a_bc2 = d['dT_a_bc2'](T_vec_a[ad], T_vec_a[ad + 1], T_vec_p[0], T_vec_p[1])
        
        dT_z_bc1 = d['dT_z_bc1'](T_vec_n[nd], T_vec_n[nd + 1], T_vec_z[0], T_vec_z[1])
        dT_z_bc2 = d['dT_z_bc2'](T_vec_z[zd], T_vec_z[zd + 1])
        
        bc_az = dict([
            ('acc', jnp.concatenate((jnp.array(dT_a_bc1), jnp.array(dT_a_bc2)))),
            ('zcc', jnp.concatenate((jnp.array(dT_z_bc1), jnp.array(dT_z_bc2))))
        ])

        Jnew = build_bc_cc(Jnew, bc_az, ad, pd, od, nd, zd)


        return Jnew
    
    return jacfn

