import jax
import jax.numpy as jnp
from scipy.linalg import solve_banded
from jax.numpy.linalg import norm

from settings import tolerance, maxit

@jax.jit
def reorder_vec(y, idx):
    return y[idx]

def newton(fn_fast, jac_fn_fast, U, cs_pe1, cs_ne1, gamma_p, gamma_n, idx, re_idx, delta_t, tol=tolerance, verbose=False):
    count = 0
    res = 100
    fail = ''
    Uold = U

    while (count < maxit and res > tol):
        
        J = jac_fn_fast(U, Uold, cs_pe1, cs_ne1, delta_t).block_until_ready()
        y = fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n, delta_t).block_until_ready()
        
        y = reorder_vec(y, idx).block_until_ready();


        res = norm(y / norm(U, jnp.inf), jnp.inf)
        
        delta = solve_banded((11, 11), J, y)
        
        delta_reordered = reorder_vec(delta, re_idx).block_until_ready()
        U = U - delta_reordered
        
        count = count + 1

    if fail == None and jnp.any(jnp.isnan(delta)):
        fail = 'Nan'
        if verbose:
            print("nan solution")

    if fail == None and max(abs(jnp.imag(delta))) > 0:
        fail += 'Img'
        if verbose:
            print("solution complex")

    if fail == None and res > tol:
        fail += 'Nc'
        if verbose:
            print('Newton fail: no convergence')

    return U, fail