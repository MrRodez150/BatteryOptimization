from scipy import integrate
import numpy as np

from auxiliaryExp import mass, internalResistance
from globalValues import F,R,T_ref

def specificEnergy(v,i,t,M,A,Rx):
    i = abs(i)
    Es = A*integrate.trapezoid(i*(v-(Rx*i)),t)/M
    return Es

def perho(data):
    return data.p*data.rho*data.epsf

def lprho(data):
    return data.l*data.p*data.rho

def batteryPrice(data_a,data_p,data_o,data_n,data_z,data_e,Ns,Np,A,L):
    return Ns*Np*A*(L*(perho(data_p) + perho(data_o) + perho(data_n) + perho(data_e)) 
                    + lprho(data_a) + lprho(data_z))

def maxTempAvg(T):
    return np.mean(T-T_ref)

def capFade(j,eta,T,mu,rho):
    var = ((0.5*F)/(R*T))*eta
    term2 = 2/(np.exp(2*var)-1)
    i0 = j*term2
    SEI_growth = i0*mu/(rho*F)
    return np.mean(SEI_growth)

def objectiveFunctions(data_a,data_p,data_o,data_n,data_z,data_e,
                       Icell,Np,Ns,L,A,
                       volt,Temps,flux,etas,Tn,times):

    M = mass(data_a,data_p,data_o,data_n,data_z,data_e,L)
    Rx = internalResistance(data_a,data_p,data_n,data_z)

    Es = specificEnergy(volt,-Icell,times,M,A,Rx)
    SEIg = capFade(flux,etas,Tn,data_n.mu,data_n.rho)
    Tavg = maxTempAvg(Temps)
    P = batteryPrice(data_a,data_p,data_o,data_n,data_z,data_e,Ns,Np,A,L)

    return [Es, SEIg, Tavg, P]

def ineqConstraintFunctions(Vpack,Ns,voltages):
    Vcell = np.mean(voltages)

    V_upper = Vcell*Ns - 1.05*Vpack
    V_lower = 0.95*Vpack - Vcell*Ns

    return [V_upper, V_lower]

def eqConsctraintFunctions(Vpack,Ns,voltages):
    Vcell = np.mean(voltages)

    V_eq = Vcell*Ns - Vpack

    return [V_eq]