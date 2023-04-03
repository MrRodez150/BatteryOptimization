from jax.config import config
config.update('jax_enable_x64', True)
from mainAux import get_sections_data, get_battery_equations
from solver import p2d_fn_solver


def p2d_simulate(x):

    Icell = x["Icell"]
    la = x["la"]
    lp = x["lp"]
    lo = x["lo"]
    ln = x["ln"]
    lz = x["lz"]
    Lh = x["Lh"]
    p_mat = x["mat"]
    Np = x["Np"]
    Ns = x["Ns"]
    RPp = x["Rp"]
    RPn = x["Rn"]
    Rcell = x["Rc"]
    C_rate = x["C"]
    ve_p = x["ve_p"]
    ve_o = x["ve_o"]
    ve_n = x["ve_n"]

    L = lp+lo+ln

    a_data, p_data, o_data, n_data, z_data, e_data = get_sections_data(la,ve_p,lp,RPp,ve_o,lo,ve_n,ln,RPn,lz,L,p_mat)

    p_eqn, n_eqn, o_eqn, a_eqn, z_eqn = get_battery_equations(a_data, p_data, o_data, n_data, z_data, Icell)

    _, voltage, temp, flux, time, fail = p2d_fn_solver(p_eqn, n_eqn, o_eqn, a_eqn, z_eqn, Icell)

    return voltage, temp, flux, time, fail



vars = {
    "Icell": 1.0,
    "la": 12e-6,
    "lp": 40e-6,
    "lo": 10e-6,
    "ln": 40e-6,
    "lz": 12e-6,
    "Lh": 40e-3,
    "mat": 'LCO',
    "Ns": 1,
    "Np": 1,
    "Rp": 0.5e-6,
    "Rn": 1e-6,
    "Rc": 4e-3,
    "C": 1.0,
    "ve_p": 0.1,
    "ve_o": 0.1,
    "ve_n": 0.1,
}

voltage, temp, flux, time, fail = p2d_simulate(vars)

if fail:
    print("No se logró realizar la simulación")
else:
    print("Total time for simulation: ", time)
