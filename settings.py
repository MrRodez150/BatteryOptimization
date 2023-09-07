global dxP
global dxN
global dxO
global dxA
global dxZ
global drP
global drN

dxP = 20;
dxN = 20;
dxO = 7;
dxA = 7;
dxZ = 7;
drP = 15;
drN = 15;

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

nadir = [0.0, 5e-15, 20.0, 4e6];