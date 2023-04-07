from main_p2d import p2d_simulate

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
    "mat": 'LFP',
    "Ns": 1,
    "Np": 1,
}

oFn, cFn, sim_time, fail = p2d_simulate(vars, 5.0, 1.0)

print(oFn)
print(cFn)

print("Total time for simulation: ", sim_time)
