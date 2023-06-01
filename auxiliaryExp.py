import numpy as np

def volumeFraction(eps:float,l:float,L:float):
    return (1-eps)*l/L

def interfacialArea(epsf:float,R:float):
    return 3*(1-epsf)/R

def eVolumeFraction(epsf_p,epsf_o,epsf_n):
    return 1-epsf_p-epsf_o-epsf_n

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

def mass(dat_a, dat_p, dat_o, dat_n, dat_z, dat_e, L):
    
    rho_a = dat_a.rho
    rho_p = dat_p.rho
    rho_o = dat_o.rho
    rho_n = dat_n.rho
    rho_z = dat_z.rho
    rho_e = dat_e.rho
    eps_p = dat_p.epsf
    eps_o = dat_o.epsf
    eps_n = dat_n.epsf
    eps_e = dat_e.epsf
    la = dat_a.l
    lz = dat_z.l
    
    M = (L * (rho_p*eps_p + rho_o*eps_o + rho_n*eps_n + rho_e*eps_e) + rho_a*la + rho_z*lz)
    return M

