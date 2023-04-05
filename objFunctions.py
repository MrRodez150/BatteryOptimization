from scipy import integrate
import numpy as np

from auxiliaryExp import area, mass, internalResistance
from globalValues import F,R

def specificEnergy(v,i,t,M,A,Rx):
    Es = A*integrate.trapezoid(i*(v-(Rx*i)),t)/M
    return Es

def batteryPrice(data_a,data_p,data_o,data_n,data_z,data_e,Ns,Np,A,L):
    p_p = data_p.p
    rho_p = data_p.rho
    eps_p = data_p.eps

    p_o   = data_o.p
    rho_o = data_o.rho
    eps_o = data_o.eps

    p_n   = data_n.p
    rho_n = data_n.rho
    eps_n = data_n.eps

    p_e   = data_e.p
    rho_e = data_e.rho
    eps_e = data_e.eps

    la    = data_a.l
    p_a   = data_a.p
    rho_a = data_a.rho

    lz    = data_z.l
    p_z   = data_z.p
    rho_z = data_z.rho
    
    return Ns*Np*A*(L*(p_p*rho_p*eps_p + p_o*rho_o*eps_o + p_n*rho_n*eps_n + p_e*rho_e*eps_e) 
                    + la*p_a*rho_a + lz*p_z*rho_z)

def maxTempAvg(T):
    return np.mean(T)

def capFade(j,eta,T,mu,rho):
    var = ((0.5*F)/(R*T))*eta
    term2 = 2/(np.exp(2*var)-1)
    i0 = j*term2
    SEI_growth = i0*mu/(rho*F)
    return np.mean(SEI_growth)

def objectiveFunctions(data_a,data_p,data_o,data_n,data_z,data_e,
                       Icell,Lh,Np,Ns,Rcell,L,
                       volt,Temps,flux,etas,Tn,times):

    Lt = L + data_a.l + data_z.l
    A = area(Lh,Lt,Rcell)
    M = mass(data_a,data_p,data_o,data_n,data_z,data_e,L)
    Rx = internalResistance(data_a,data_p,data_n,data_z)

    Es = specificEnergy(volt,-Icell,times,M,A,Rx)
    SEIg = capFade(flux,etas,Tn,data_n.mu,data_n.rho)
    Tavg = maxTempAvg(Temps)
    P = batteryPrice(data_a,data_p,data_o,data_n,data_z,data_e,Ns,Np,A,L)

    return [Es, SEIg, Tavg, P]