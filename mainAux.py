from jax import jit
import jax.numpy as jnp
import numpy as np
from jax.lax import dynamic_slice
from functools import partial
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import spsolve

from config import delta_t, div_x_cc, div_x_elec, div_x_sep
from electrodeEqn import electrodeEquations
from separatorEqn import separatorEquations
from currentCollectorEqn import currentCollectorEquations
from batterySections import negative_electrode_data, \
LCO_positive_electrode_data, LFP_positive_electrode_data, \
separator_data, electrolyte_data, Cu_collector_data, Al_collector_data




pd = div_x_elec
nd = div_x_elec
od = div_x_sep
ad = div_x_cc
zd = div_x_cc



def get_sections_data(la,ve_p,lp,Rp,ve_o,lo,ve_n,ln,Rn,lz,L,material):

    a_data = Al_collector_data(la)
    
    if material=='LCO':
        p_data = LCO_positive_electrode_data(ve_p,lp,Rp,L)
    elif material=='LFP':
        p_data = LFP_positive_electrode_data(ve_p,lp,Rp,L)
    else:
        raise ValueError("Undefined material value for positive electrode")

    o_data = separator_data(ve_o,lo,L)
    
    n_data = negative_electrode_data(ve_n,ln,Rn,L)

    z_data = Cu_collector_data(lz)

    e_data = electrolyte_data(p_data.eps,o_data.eps,n_data.eps)

    return a_data, p_data, o_data, n_data, z_data, e_data
    


def get_battery_equations(a_data, p_data, o_data, n_data, z_data, iapp):

    p_eqn = electrodeEquations(p_data,o_data,a_data,z_data,'p',delta_t)

    n_eqn = electrodeEquations(n_data,o_data,a_data,z_data,'n',delta_t)

    o_eqn = separatorEquations(o_data,p_data,n_data,delta_t)

    a_eqn = currentCollectorEquations(a_data,delta_t,iapp)

    z_eqn = currentCollectorEquations(z_data,delta_t,iapp)
    
    return p_eqn, n_eqn, o_eqn, a_eqn, z_eqn






ce_p0 =  0
ce_o0 = ce_p0 + pd + 2
ce_n0 = ce_o0 + od +2

j_p0 = ce_n0 + nd + 2
j_n0 = j_p0 + pd

eta_p0 = j_n0 + nd 
eta_n0 = eta_p0 + pd

phis_p0 = eta_n0 + nd
phis_n0 = phis_p0 + pd + 2

phie_p0 = phis_n0 + nd +2
phie_o0 = phie_p0 + pd + 2
phie_n0 = phie_o0 + od + 2

t_a0 = phie_n0 + nd + 2
t_p0 = t_a0 + ad +2
t_o0 = t_p0 + pd + 2
t_n0 = t_o0 + od + 2
t_z0 = t_n0 + nd + 2





@partial(jit)
def unpack_vars(U):

    Tvec_p= dynamic_slice(U, [t_p0], [t_o0 - t_p0])
    Tvec_n = dynamic_slice(U, [t_n0], [t_z0 - t_n0])

    Tvec = dynamic_slice(U, [t_a0], [t_z0+zd+2 - t_a0])

    phis_p = dynamic_slice(U, [phis_p0], [phis_n0 - phis_p0])
    phis_n = dynamic_slice(U, [phis_n0], [phis_n0 + nd + 2 - phis_n0])

    j_p = dynamic_slice(U,[j_p0],[j_n0 - j_p0])
    j_n = dynamic_slice(U, [j_n0], [j_n0 + nd - j_n0])

    eta_n =dynamic_slice(U,[eta_n0],[eta_n0+nd - eta_n0])

    return Tvec, Tvec_p, Tvec_n, phis_p, phis_n, j_p, j_n, eta_n






@partial(jit)
def unpack(U):
    
    ce_vec_p = dynamic_slice(U, [ce_p0], [ce_o0 - ce_p0])
    ce_vec_o = dynamic_slice(U, [ce_o0], [ce_n0 - ce_o0])
    ce_vec_n = dynamic_slice(U, [ce_n0], [ce_n0+nd+2 - ce_n0])

    T_vec_a = dynamic_slice(U, [t_a0], [t_p0 - t_a0])
    T_vec_p= dynamic_slice(U, [t_p0], [t_o0 - t_p0])
    T_vec_o = dynamic_slice(U, [t_o0], [t_n0 - t_o0])
    T_vec_n = dynamic_slice(U, [t_n0], [t_z0 - t_n0])
    T_vec_z = dynamic_slice(U, [t_z0], [t_z0+zd+2 - t_z0])
    
    phie_p =dynamic_slice(U, [phie_p0],[phie_o0 - phie_p0])
    phie_o = dynamic_slice(U, [phie_o0],[phie_n0 - phie_o0])
    phie_n = dynamic_slice(U, [phie_n0],[phie_n0+nd+2 - phie_n0])
    
    phis_p = dynamic_slice(U,[phis_p0],[phis_n0 - phis_p0])
    phis_n = dynamic_slice(U,[phis_n0],[phis_n0+nd+2 - phis_n0])
    
    j_p = dynamic_slice(U,[j_p0],[j_n0 - j_p0])
    j_n = dynamic_slice(U,[j_n0],[j_n0+nd - j_n0])
    
    eta_p =dynamic_slice(U,[eta_p0],[eta_n0 - eta_p0])
    eta_n =dynamic_slice(U,[eta_n0],[eta_n0+nd - eta_n0])
    
    return ce_vec_p, ce_vec_o, ce_vec_n,\
           T_vec_a, T_vec_p, T_vec_o, T_vec_n, T_vec_z,\
           phie_p, phie_o, phie_n,\
           phis_p, phis_n,\
           j_p, j_n,\
           eta_p, eta_n







def reorder_tot():
    
    order = np.arange(0, 4*(pd+2) + 2*pd + 2*nd + 4*(nd+2) + 3*(od+2) + ad+2 + zd+2)
    
    for i in range(0,ad+1):
        order[t_a0+i]= i
    # ce0
    p0 = ad + 1
    order[0] = p0
     #phis_p0
    order[phis_p0] = 1 + p0
     #Phie0
    order[phie_p0] = 2 + p0
    #T_p0   
    order[t_p0] = 3+ p0
    order[t_a0 + ad + 1] = 4+ p0

    # ce pd+1
    order[pd+1] = 5 + 6*pd+ p0
    # phis pd+1    
    order[phis_p0 + pd + 1] = 5 + 6*pd + 1 + p0
            
    #Phiep pd+1
    order[phie_p0+pd+1] = 5 + 6*pd + 2 + p0
            
    # Tp0 + pd+1    
    order[t_p0+pd+1] = 5 + 6*pd + 3 + p0

    # u
    for i in range(1,pd+1):
        order[i]= 5 + 6*(i-1) + p0
    # j
    for i in range(0, pd):
        order[i + j_p0] = 5 + 6*i + 1+ p0
    
    #eta
    for i in range(0, pd):
        order[i + eta_p0 ] = 5 + 6*i + 2+ p0

    
    # phis
    for i in range(1, pd+1):
        order[i + phis_p0] = 5 + 6*(i-1) + 3+ p0

    
    #phie    
    for i in range(1, pd+1):
        order[i + phie_p0] =5 + 6*(i-1) +4+ p0

    # T    
    for i in range(1, pd+1):
        order[i + t_p0] = 5 + 6*(i-1) + 5+ p0
        
    #separator
    
    #ce_o0
    sep0 = 4*(pd+2) + 2*pd + 1 + p0
    order[ce_o0] = sep0
    order[phie_o0] = sep0 + 1
    order[t_o0] = sep0 + 2
    
    
    for i in range(1, od+1):
        order[i + ce_o0] = sep0 + 3*i
        order[i + phie_o0] = sep0 + 3*i + 1
        order[i + t_o0] = sep0 + 3*i + 2
        
    order[ce_o0+od+1] = sep0 + 3*od + 3
    order[phie_o0+od+1] = sep0 + 3*od + 4
    order[t_o0 + od + 1] = sep0 + 3*od + 5
        
        
    n0 = p0 + 4*(pd+2) + 2*pd + 3*(od+2) + 1
        # u0 
    # order[ce_n0] = n0 + 1
    order[ce_n0] = n0
    #phisp0
    order[phis_n0] = n0 + 1  
    #Phie0
    order[phie_n0] = n0 + 2
        #Tp0    
    order[t_n0] = n0 + 3
    
    
    for i in range(1,nd+1):
        order[ce_n0 + i]= n0 + 4 + 6*(i-1)
 
    # j
    for i in range(0, nd):
        order[i + j_n0] = n0 + 4 + 6*i + 1

    #eta
    for i in range(0, nd):
        order[i + eta_n0 ] = n0 + 4 + 6*i + 2

    
    # phis
    for i in range(1, nd+1):
        order[i + phis_n0] = n0 + 4 + 6*(i-1) + 3

    
    #phie    
    for i in range(1, nd+1):
        order[i + phie_n0] = n0 + 4 + 6*(i-1) + 4
    
    # T    
    for i in range(1, nd+1):
        order[i + t_n0] = n0 + 4 + 6*(i-1) + 5
        
        # u nd+1
    order[ce_n0 + nd + 1] = n0 + 4 + 6*nd
    # phis pd+1    
    order[phis_n0 + nd + 1] = n0 + 4 + 6*nd + 1  
    #Phiep pd+1
    order[phie_n0+nd+1] = n0 + 4 + 6*nd + 2      
    # Tp0 + pd+1    
    order[t_n0+nd+1] = n0 + 4 + 6*nd + 3
    order[t_z0] = n0 + 4 + 6*nd + 4
    
    for i in range(1,zd+2):
        order[t_z0+i]= n0 + 4 + 6*nd + 4 + i 
    
#    for i in range(0,ad+1):
#        order[ta0+i]= n0 + 4 + 6*nd + 4 + (i+1) + zd+1
#        
    
    sort_index = np.argsort(order)
        
    return sort_index







"""Unused"""

def precompute(eq_p,eq_n):

    row_p = eq_p.row
    col_p = eq_p.col
    dat_p = eq_p.dat
    pdr = eq_p.div_r

    row_n = eq_n.row
    col_n = eq_n.col
    dat_n = eq_n.dat
    ndr = eq_n.div_r

    Ap = csr_matrix((dat_p, (row_p, col_p)))
    An = csr_matrix((dat_n, (row_n, col_n)))

    A_p = kron(identity(pd), Ap)
    A_n = kron(identity(nd), An)

    vec_p = jnp.hstack([jnp.zeros(pdr+1), 1])
    vec_n = jnp.hstack([jnp.zeros(ndr+1), 1])

    temp_p = spsolve(Ap,vec_p)
    gamma_p = (temp_p[pdr] + temp_p[pdr+1])/2

    temp_n = spsolve(An,vec_n)
    gamma_n = (temp_n[ndr] + temp_n[ndr+1])/2

    return A_p, A_n, gamma_p, gamma_n, temp_p, temp_n
