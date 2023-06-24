from jax import jit
from jax.lax import dynamic_slice
from functools import partial
from settings import dxA, dxP, dxO, dxN, dxZ


ce_p0 =  0
ce_o0 = ce_p0 + dxP + 2
ce_n0 = ce_o0 + dxO +2

j_p0 = ce_n0 + dxN + 2
j_n0 = j_p0 + dxP

eta_p0 = j_n0 + dxN 
eta_n0 = eta_p0 + dxP

phis_p0 = eta_n0 + dxN
phis_n0 = phis_p0 + dxP + 2

phie_p0 = phis_n0 + dxN +2
phie_o0 = phie_p0 + dxP + 2
phie_n0 = phie_o0 + dxO + 2

t_a0 = phie_n0 + dxN + 2
t_p0 = t_a0 + dxA +2
t_o0 = t_p0 + dxP + 2
t_n0 = t_o0 + dxO + 2
t_z0 = t_n0 + dxN + 2





@partial(jit)
def unpack_vars(U):

    Tvec_p= dynamic_slice(U, [t_p0], [t_o0 - t_p0])
    Tvec_n = dynamic_slice(U, [t_n0], [t_z0 - t_n0])

    Tvec = dynamic_slice(U, [t_a0], [t_z0+dxZ+2 - t_a0])

    phis_p = dynamic_slice(U, [phis_p0], [phis_n0 - phis_p0])
    phis_n = dynamic_slice(U, [phis_n0], [phis_n0 + dxN + 2 - phis_n0])

    j_p = dynamic_slice(U,[j_p0],[j_n0 - j_p0])
    j_n = dynamic_slice(U, [j_n0], [j_n0 + dxN - j_n0])

    eta_n =dynamic_slice(U,[eta_n0],[eta_n0+dxN - eta_n0])

    return Tvec, Tvec_p, Tvec_n, phis_p, phis_n, j_p, j_n, eta_n



@partial(jit)
def unpack(U):
    
    ce_vec_p = dynamic_slice(U, [ce_p0], [ce_o0 - ce_p0])
    ce_vec_o = dynamic_slice(U, [ce_o0], [ce_n0 - ce_o0])
    ce_vec_n = dynamic_slice(U, [ce_n0], [ce_n0+dxN+2 - ce_n0])

    T_vec_a = dynamic_slice(U, [t_a0], [t_p0 - t_a0])
    T_vec_p= dynamic_slice(U, [t_p0], [t_o0 - t_p0])
    T_vec_o = dynamic_slice(U, [t_o0], [t_n0 - t_o0])
    T_vec_n = dynamic_slice(U, [t_n0], [t_z0 - t_n0])
    T_vec_z = dynamic_slice(U, [t_z0], [t_z0+dxZ+2 - t_z0])
    
    phie_p =dynamic_slice(U, [phie_p0],[phie_o0 - phie_p0])
    phie_o = dynamic_slice(U, [phie_o0],[phie_n0 - phie_o0])
    phie_n = dynamic_slice(U, [phie_n0],[phie_n0+dxN+2 - phie_n0])
    
    phis_p = dynamic_slice(U,[phis_p0],[phis_n0 - phis_p0])
    phis_n = dynamic_slice(U,[phis_n0],[phis_n0+dxN+2 - phis_n0])
    
    j_p = dynamic_slice(U,[j_p0],[j_n0 - j_p0])
    j_n = dynamic_slice(U,[j_n0],[j_n0+dxN - j_n0])
    
    eta_p =dynamic_slice(U,[eta_p0],[eta_n0 - eta_p0])
    eta_n =dynamic_slice(U,[eta_n0],[eta_n0+dxN - eta_n0])
    
    return ce_vec_p, ce_vec_o, ce_vec_n,\
           T_vec_a, T_vec_p, T_vec_o, T_vec_n, T_vec_z,\
           phie_p, phie_o, phie_n,\
           phis_p, phis_n,\
           j_p, j_n,\
           eta_p, eta_n
