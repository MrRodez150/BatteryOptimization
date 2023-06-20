import numpy as np

def volumeFraction(eps:float,l:float,L:float):
    return (1-eps)*l/L

def interfacialArea(epsf:float,R:float):
    return 3*(1-epsf)/R

def eVolumeFraction(epsf_p,epsf_o,epsf_n):
    return (1-epsf_p)+(1-epsf_o)+(1-epsf_n)

def internalResistance(dat_a, dat_p, dat_n, dat_z):
    la = dat_a.l
    sig_a = dat_a.sigma
    lp = dat_p.l
    sig_p = dat_p.sigma
    ln = dat_n.l
    sig_n = dat_n.sigma
    lz = dat_z.l
    sig_z = dat_z.sigma
    return 1/(la*sig_a + lp*sig_p + ln*sig_n + lz*sig_z)

def turns(Rcell,Lt):
    return 2*np.pi*Rcell/Lt

def area(Lh,Lt,Rcell):
    tur = turns(Rcell,Lt)
    return (Lh*Lt) * (tur*np.sqrt(tur**2+1) + np.log(tur+np.sqrt(tur**2+1))) / (4*np.pi)

def lerho(data):
    return data.l*data.rho*data.epsf

def lrho(data):
    return data.l*data.rho

def mass(dat_a, dat_p, dat_o, dat_n, dat_z, dat_e):
     
    return lerho(dat_p) + lerho(dat_o) + lerho(dat_n) + lerho(dat_e) + lrho(dat_a) + lrho(dat_z)

