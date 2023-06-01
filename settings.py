global div_x_elec
global div_x_sep
global div_x_cc
global div_r

div_x_elec = 10;
div_x_sep = 5;
div_x_cc = 5;
div_r = 10;

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
delta_t = 10;
div_t = sim_time/delta_t;

global tolerance
global maxIter
global cutOff

tolerance=1e-7;
maxit=10;
cutOff = 2.5;