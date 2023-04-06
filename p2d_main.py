from jax.config import config
config.update('jax_enable_x64', True)

from mainAux import get_sections_data, get_battery_equations
from solver import p2d_fn_solver
from fghFunctions import area, objectiveFunctions, ineqConstraintFunctions
from plotter import plotTimeChange


def p2d_simulate(x, Vpack, Ipack):

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
    Lt = L + la + lz
    A = area(Lh,Lt,Rcell)

    Icell = -(Ipack*C_rate)/(Np*A)

    a_data, p_data, o_data, n_data, z_data, e_data = get_sections_data(la,ve_p,lp,RPp,ve_o,lo,ve_n,ln,RPn,lz,L,p_mat)

    p_eqn, n_eqn, o_eqn, a_eqn, z_eqn = get_battery_equations(a_data, p_data, o_data, n_data, z_data, Icell)

    voltage, temp, flux, ovpot, temp_n, times, time, fail = p2d_fn_solver(p_eqn, n_eqn, o_eqn, a_eqn, z_eqn, Icell)

    objFun = objectiveFunctions(a_data, p_data, o_data, n_data, z_data, e_data, 
                                Icell, Np, Ns, L, A, 
                                voltage, temp, flux, ovpot, temp_n, times)
    
    conFun = ineqConstraintFunctions(Vpack,Ns,voltage)

    """
    print(voltage)
    print(temp)
    print(flux)
    print(ovpot)
    print(temp_n)
    print(times)

    v_fig = plotTimeChange(times, voltage, 'voltage [V]')
    v_fig.show()
    """
    
    return objFun, conFun, time, fail



vars = {
    "C": 1.0,
    "la": 12e-6,
    "lp": 40e-6,
    "lo": 10e-6,
    "ln": 40e-6,
    "lz": 12e-6,
    "Lh": 40e-3,
    "Rp": 0.5e-6,
    "Rn": 1e-6,
    "Rc": 4e-3,
    "ve_p": 0.1,
    "ve_o": 0.1,
    "ve_n": 0.1,
    "mat": 'LCO',
    "Ns": 1,
    "Np": 1,
}

oFn, cFn, sim_time, fail = p2d_simulate(vars, 5.0, 1.0)

print(oFn)
print(cFn)

print("Total time for simulation: ", sim_time)
