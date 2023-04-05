global div_x_elec
global div_x_sep
global div_x_cc
global div_r

div_x_elec = 10;
div_x_sep = 5;
div_x_cc = 5;
div_r = 10;

global sim_time
global div_t
global delta_t

sim_time = 3600;
delta_t = 5;
div_t = sim_time/delta_t;

global tol
global maxit

tol=1e-7;
maxit=10;