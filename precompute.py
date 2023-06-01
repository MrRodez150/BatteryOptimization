import jax.numpy as jnp
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import spsolve

def precompute(peq,neq):
    Mp = peq.M; Mn = neq.M; 
    Np = peq.N; Nn = neq.N
    hy = peq.hy
    Ds = peq.Ds;
    delta_t=peq.delta_t
    
    #R = jnp.arange(0,peq.Rp + peq.hy, peq.hy) + peq.hy/2 ; R = R[0:-1]
    R = peq.Rp*(jnp.linspace(1,Np,Np)-(1/2))/Np;
    r = peq.Rp*(jnp.linspace(0,Np, Np+1))/Np
    lambda1 = delta_t*r[0:Np]**2/(R**2*peq.hy**2);
    lambda2 = delta_t*r[1:Np+1]**2/(R**2*peq.hy**2);
            
    hy_n = neq.hy
    Ds_n = neq.Ds
    Rn = neq.Rp*(jnp.linspace(1,Nn,Nn)-(1/2))/Nn;
    rn = neq.Rp*(jnp.linspace(0,Nn, Nn+1))/Nn
    lambda1_n = delta_t*rn[0:Nn]**2/(Rn**2*neq.hy**2);
    lambda2_n = delta_t*rn[1:Nn+1]**2/(Rn**2*neq.hy**2);

    row = jnp.hstack([0,0,jnp.arange(1,Np+1,1),jnp.arange(1,Np+1,1),jnp.arange(1,Np+1,1),Np+1,Np+1])
    col=jnp.hstack([0,1,jnp.arange(1,Np+1,1),jnp.arange(1,Np+1,1)-1,jnp.arange(1,Np+1,1)+1,Np,Np+1])
    data = jnp.hstack([-1,1,
            1+Ds*(lambda1+lambda2),
            -Ds*lambda1,
            -Ds*lambda2,
            -1/hy,1/hy]);
        
    row_n = jnp.hstack([0,0,jnp.arange(1,Nn+1,1),jnp.arange(1,Nn+1,1),jnp.arange(1,Nn+1,1),Nn+1,Nn+1])
    col_n=jnp.hstack([0,1,jnp.arange(1,Nn+1,1),jnp.arange(1,Nn+1,1)-1,jnp.arange(1,Nn+1,1)+1,Nn,Nn+1])
    data_n = jnp.hstack([-1,1,
             1+Ds_n*(lambda1_n+lambda2_n),
             -Ds_n*lambda1_n,
            -Ds_n*lambda2_n,
            -1/hy_n,1/hy_n]);
        
    #J1 = J[0:Np+2, 0:Np+2]
    Ape = csr_matrix((data, (row, col)))
    Ane = csr_matrix((data_n, (row_n, col_n)))
    Ap = kron( identity(Mp), Ape)
    An = kron(identity(Mn), Ane)
    vec_p = jnp.hstack([jnp.zeros(Np+1), 1])
    vec_n = jnp.hstack([jnp.zeros(Nn+1), 1])
    temp_p = spsolve(Ape,vec_p); gamma_p = (temp_p[Np] + temp_p[Np+1])/2
    temp_n = spsolve(Ane,vec_n); gamma_n = (temp_n[Nn] + temp_n[Nn+1])/2
    
    return Ap, An, gamma_n, gamma_p, temp_p, temp_n