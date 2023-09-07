#%%
from main_p2d import p2d_simulate

#Aplication requirements
Vpack = 3.7
Iapp = -3

#Decision variables
vars = {
    "C": 0.5,
    "la": 12e-6,
    "lp": 40e-6,
    "lo": 10e-6,
    "ln": 40e-6,
    "lz": 12e-6,
    "Lh": 40e-3,
    "Rp": 20e-6,
    "Rn": 2e-6,
    "Rcell": 4e-3,
    "efp": 0.25,
    "efo": 0.02,
    "efn": 0.0326,
    "mat": 'LFP',
    "Ns": 1,
    "Np": 1,
}

oFn, cFn, sim_time, fail, fig = p2d_simulate(vars, Vpack, Iapp, verbose=True, plot=True)

print('objFn: ', oFn)
print('consFn: ', cFn)
print("Times for simulation: ", sim_time)
print('Failure: ', fail)

fig.show()
# %%
