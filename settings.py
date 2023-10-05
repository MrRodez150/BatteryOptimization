import numpy as np

global dxP
global dxN
global dxO
global dxA
global dxZ
global drP
global drN

dxP = 30;
dxN = 30;
dxO = 10;
dxA = 10;
dxZ = 10;
drP = 20;
drN = 20;

global sim_time
global div_t
global delta_t

sim_time = 3600;
delta_t = 1;
div_t = sim_time/delta_t;

global tolerance
global maxIter
global cutOff

tolerance=1e-6;
maxit=10;
cutOff = 2.7;

global nadir
global max_presure
global var_keys, oFn_keys, cFn_keys, limits, e, refdirs

nadir = [0.0, 5e-15, 20.0, 4e6];
max_presure = 5e9;

var_keys = ['C', 'la', 'lp', 'lo', 'ln', 'lz', 'Lh', 'Rp', 'Rn', 'Rcell', 'efp', 'efo', 'efn', 'mat', 'Ns', 'Np']
oFn_keys = ['SpecificEnergy', 'SEIGrouth', 'TempIncrease', 'Price']
cFn_keys = ['UpperViolation', 'LowerViolation', 'VolFracViolation']

limits = np.array([[0.2, 4.0],
                [12e-6, 30e-6],
                [40e-6, 250e-6],
                [10e-6, 100e-6],
                [40e-6, 150e-6],
                [12e-6, 30e-6],
                [40e-3, 100e-3],
                [0.2e-6, 20e-6],
                [0.5e-6, 50e-6],
                [4e-3, 25e-3],
                [0.01, 0.6],
                [0.01, 0.6],
                [0.01, 0.6]])

e = [[1,      1e-12,  1e-12,  1e-12],
    [1e-12,  1,      1e-12,  1e-12],
    [1e-12,  1e-12,  1,      1e-12],
    [1e-12,  1e-12,  1e-12,  1    ],
    [0.25,   0.25,   0.25,   0.25 ]]

