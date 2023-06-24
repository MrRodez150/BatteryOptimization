from jax import jit
import jax.numpy as np

from resFunc import ResidualFunction
from unpack import unpack

def p2d_init_fast(eqn_p, eqn_n, eqn_o, eqn_a, eqn_z, Iapp):
    solver = ResidualFunction(eqn_p, eqn_n, eqn_o, eqn_a, eqn_z, Iapp)
   
    def fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n, delta_t):

        val= np.zeros(solver.Ntot)

        ce_p, ce_o, ce_n, \
        T_a, T_p, T_o, T_n, T_z, \
        phie_p, phie_o, phie_n, \
        phis_p, phis_n, \
        j_p, j_n, \
        eta_p, eta_n = unpack(U)
    
        
        ce_p_old, ce_o_old, ce_n_old,\
        T_a_old, T_p_old, T_o_old, T_n_old, T_z_old,\
        _,_,_,_,_,_,_,_,_= unpack(Uold)
        
      
        val = solver.res_ce_p(val, ce_p, T_p, j_p, ce_p_old, ce_o, T_o, delta_t)
        val = solver.res_ce_o(val, ce_o, T_o, ce_o_old, ce_p, T_p, ce_n, T_n, delta_t)
        val = solver.res_ce_n(val, ce_n, T_n, j_n, ce_n_old, ce_o, T_o, delta_t)
        
        val = solver.res_T_a(val, T_a, T_a_old, T_p, delta_t)
        val = solver.res_T_p(val, T_p, ce_p, phie_p, phis_p, j_p, eta_p, cs_pe1, gamma_p, T_p_old, T_a, T_o, delta_t)
        val = solver.res_T_o(val, T_o, ce_o, phie_o, T_o_old, T_p, T_n, delta_t)
        val = solver.res_T_n(val, T_n, ce_n, phie_n, phis_n, j_n, eta_n, cs_ne1, gamma_n, T_n_old, T_z, T_o, delta_t)
        val = solver.res_T_z(val, T_z, T_z_old, T_n, delta_t)
        
        val = solver.res_phie_p(val, ce_p, phie_p, T_p, j_p, ce_o,phie_o, T_o)
        val = solver.res_phie_o(val, ce_o, phie_o, T_o, phie_p, phie_n)
        val = solver.res_phie_n(val, ce_n, phie_n, T_n, j_n, ce_o, phie_o, T_o)
    
        val = solver.res_phis(val, phis_p, j_p, phis_n, j_n)
    
        val = solver.res_j(val, j_p, ce_p, T_p, eta_p, cs_pe1, gamma_p, j_n, ce_n, T_n, eta_n, cs_ne1, gamma_n)
        val = solver.res_eta(val, eta_p, phis_p, phie_p, T_p, j_p, cs_pe1, gamma_p, eta_n, phis_n, phie_n, T_n, j_n, cs_ne1, gamma_n)
        
        return val
    
    fn_fast=jit(fn_fast)
        
    return fn_fast
    
    