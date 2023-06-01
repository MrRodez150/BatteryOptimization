from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)

import timeit

from init import p2d_init_fast
from batteryBuilder import build_battery
from derivatives import partials, compute_jac
from p2dBuilder import get_battery_sections
from precompute import precompute
from p2dSolver import p2d_reorder_fn
from fghFunctions import objectiveFunctions, ineqConstraintFunctions
from auxiliaryExp import area
from settings import dxP, dxN

def p2d_simulate(x, Vpack, Ipack, verbose=False):

    start = timeit.default_timer()

    #Decision variables

    C = x["C"]
    la = x["la"]
    lp = x["lp"]
    lo = x["lo"]
    ln = x["ln"]
    lz = x["lz"]
    Lh = x["Lh"]
    Rp = x["Rp"]
    Rn = x["Rn"]
    Rcell = x["Rcell"]
    efp = x["efp"]
    efo = x["efo"]
    efn = x["efn"]
    mat = x["mat"]
    Np = x["Np"]
    Ns = x["Ns"]

    Icell = Ipack * C / Np

    #Construction of the battery

    p_data, n_data, o_data, a_data, z_data, e_data = build_battery(mat,efp,efo,efn,Rp,Rn,la,lp,lo,ln,lz)


    p_eq, n_eq, o_eq, a_eq, z_eq = get_battery_sections(p_data, n_data, o_data, a_data, z_data, Icell)

    #Preparing functions for simulations

    fn, _ = p2d_init_fast(p_eq, n_eq, o_eq, a_eq, z_eq, Icell)

    Ap, An, gamma_n, gamma_p, temp_p, temp_n = precompute(p_eq, n_eq)
    gamma_p_vec = gamma_p * jnp.ones(dxP)
    gamma_n_vec = gamma_n * jnp.ones(dxN)
    lu_p = splu(csc_matrix(Ap))
    lu_n = splu(csc_matrix(An))

    partial_fns = partials(a_eq, p_eq, o_eq, n_eq, z_eq)
    jac_fn = compute_jac(gamma_p_vec, gamma_n_vec, partial_fns, p_eq, n_eq, Icell)

    mid = timeit.default_timer()

    #Simulate

    U_fast, cmat_pe, cmat_ne, \
        voltage, temp, flux, ovpot, tempN, times, fail = p2d_reorder_fn(p_eq, o_eq, n_eq,
                                                                        lu_p, lu_n, temp_p, temp_n,
                                                                        gamma_p_vec, gamma_n_vec,
                                                                        fn, jac_fn, tol=1e-6, verbose=verbose)


    #Obtain objective funcions and constraint violations

    L=lp+lo+ln
    Lt = L+la+lz
    A = area(Lh,Lt,Rcell)

    objFun = objectiveFunctions(a_data, p_data, o_data, n_data, z_data, e_data, 
                                Icell, Np, Ns, Lt, A, 
                                voltage, temp, flux, ovpot, tempN, times)

    conFun = ineqConstraintFunctions(Vpack,Ns,voltage)

    end = timeit.default_timer()
    time = [end-start, mid-start, end-mid]

    return objFun, conFun, time, fail