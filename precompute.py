import jax.numpy as jnp
from scipy.sparse import csr_matrix, csc_matrix, kron, identity
from scipy.sparse.linalg import spsolve, splu
from settings import delta_t

def precompute(peq,neq):
    Mp = peq.M
    Mn = neq.M
    Np = peq.N
    Nn = neq.N
    dr_p = peq.dr
    Ds_p = peq.Ds
    dr_n = neq.dr
    Ds_n = neq.Ds
    
    Rp = peq.Rp*(jnp.linspace(1,Np,Np)-(1/2))/Np
    rp = peq.Rp*(jnp.linspace(0,Np, Np+1))/Np

    lambda1_p= delta_t*rp[0:Np]**2/(Rp**2*dr_p**2)
    lambda2_p = delta_t*rp[1:Np+1]**2/(Rp**2*dr_p**2)

    Rn = neq.Rp*(jnp.linspace(1,Nn,Nn)-(1/2))/Nn
    rn = neq.Rp*(jnp.linspace(0,Nn, Nn+1))/Nn

    lambda1_n = delta_t*rn[0:Nn]**2/(Rn**2*dr_n**2)
    lambda2_n = delta_t*rn[1:Nn+1]**2/(Rn**2*dr_n**2)

    row_p = jnp.hstack([0,0,jnp.arange(1,Np+1,1),jnp.arange(1,Np+1,1),jnp.arange(1,Np+1,1),Np+1,Np+1])
    col_p=jnp.hstack([0,1,jnp.arange(1,Np+1,1),jnp.arange(1,Np+1,1)-1,jnp.arange(1,Np+1,1)+1,Np,Np+1])
    data_p = jnp.hstack([-1,1,
            1+Ds_p*(lambda1_p+lambda2_p),
            -Ds_p*lambda1_p,
            -Ds_p*lambda2_p,
            -1/dr_p,1/dr_p])
        
    row_n = jnp.hstack([0,0,jnp.arange(1,Nn+1,1),jnp.arange(1,Nn+1,1),jnp.arange(1,Nn+1,1),Nn+1,Nn+1])
    col_n=jnp.hstack([0,1,jnp.arange(1,Nn+1,1),jnp.arange(1,Nn+1,1)-1,jnp.arange(1,Nn+1,1)+1,Nn,Nn+1])
    data_n = jnp.hstack([-1,1,
             1+Ds_n*(lambda1_n+lambda2_n),
             -Ds_n*lambda1_n,
            -Ds_n*lambda2_n,
            -1/dr_n,1/dr_n])

    Ape = csr_matrix((data_p, (row_p, col_p)))
    Ane = csr_matrix((data_n, (row_n, col_n)))

    Ap = kron(identity(Mp), Ape)
    An = kron(identity(Mn), Ane)

    vec_p = jnp.hstack([jnp.zeros(Np+1), 1])
    vec_n = jnp.hstack([jnp.zeros(Nn+1), 1])

    temp_p = spsolve(Ape,vec_p)
    gamma_p = (temp_p[Np] + temp_p[Np+1])/2

    temp_n = spsolve(Ane,vec_n)
    gamma_n = (temp_n[Nn] + temp_n[Nn+1])/2

    gamma_p_vec = gamma_p * jnp.ones(Mp)
    gamma_n_vec = gamma_n * jnp.ones(Mn)

    lu_p = splu(csc_matrix(Ap))
    lu_n = splu(csc_matrix(An))
    
    return lu_p, lu_n, gamma_n_vec, gamma_p_vec, temp_p, temp_n