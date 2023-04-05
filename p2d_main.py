from jax.config import config
config.update('jax_enable_x64', True)
from mainAux import get_sections_data, get_battery_equations
from solver import p2d_fn_solver
from objFunctions import objectiveFunctions
from plotter import plotTimeChange


def p2d_simulate(x):

    Icell = -x["Icell"]
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

    voltage, temp, flux, ovpot, temp_n, times, time, fail = p2d_fn_solver(p_eqn, n_eqn, o_eqn, a_eqn, z_eqn, Icell)

    print(voltage)
    print(temp)
    print(flux)
    print(ovpot)
    print(temp_n)
    print(times)

    objFun = objectiveFunctions(a_data, p_data, o_data, n_data, z_data, e_data, Icell, Lh, Np, Ns, Rcell, L, voltage, temp, flux, ovpot, temp_n, times)
    
    return objFun, time, fail



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

oFn, sim_time, fail = p2d_simulate(vars)

#v_fig = plotTimeChange(times, voltage, 'voltage [V]')
#v_fig.show()

print(oFn)

print("Total time for simulation: ", sim_time)
