from dataclasses import dataclass
from auxiliaryExp import volumeFraction, interfacialArea, eVolumeFraction


@dataclass
class electrode:
    #Material properties
    a: float;               #Specific interfacial area
    brugg: float;           #Bruggerman coefficient
    C: float;               #Specific heat
    c_s_max: float;         #Maximum solid phase concentration
    c_s_avg: float;         #Average solid phase concentration
    Ds: float;              #Solid phase diffusivity
    ED: float;              #Activation energy for the solid diffusion
    Ek: float;              #Activation energy for the reaction constant
    k: float;               #Reaction rate
    p: float;               #Price
    eps: float;             #Volume fraction
    lam: float;             #Thermal conductivity
    mu: float;              #Specific mass
    rho: float;             #Density
    sigma: float;           #Solid phase conductivity
    
    #Decision variables
    vareps: float;          #Porosity
    l: float;               #Thiknes
    Rp: float;              #Particle radius

def negative_electrode_data(ve,l,R,L):

    eps=volumeFraction(ve,l,L)
    a=interfacialArea(ve,eps,R)

    return electrode(
        #Mat prop
        a=a,
        brugg=4.0,
        C=706.9,
        c_s_max=30555.0,
        c_s_avg=26128,
        Ds=3.9e-14,
        ED=5000,
        Ek=5000,
        k=5.031e-11,
        p=60.0,
        eps=eps,
        lam=1.7,
        mu=1.201e-2,
        rho=2160.0,
        sigma=1000.0,
        #Dec vars
        vareps=ve,
        l=l,
        Rp=R
    )

def LFP_positive_electrode_data(ve,l,R,L):

    eps=volumeFraction(ve,l,L)
    a=interfacialArea(ve,eps,R)

    return electrode(
        #Mat prop
        a=a,
        brugg=4.0,
        C=1260.0,
        c_s_max=51554.0,
        c_s_avg=25751,
        Ds=4.295e-18,
        ED=5000,
        Ek=5000,
        k=7.882e-12,
        p=70.0,
        eps=eps,
        lam=0.15,
        mu=1.577e-1,
        rho=1132.0,
        sigma=0.4977,
        #Dec vars
        vareps=ve,
        l=l,
        Rp=R
    )

def LCO_positive_electrode_data(ve,l,R,L):

    eps=volumeFraction(ve,l,L)
    a=interfacialArea(ve,eps,R)

    return electrode(
        #Mat prop
        a=a,
        brugg=4.0,
        C=1269.0,
        c_s_max=51554.0,
        c_s_avg=25751,
        Ds=1.806e-14,
        ED=5000,
        Ek=5000,
        k=7.898e-12,
        p=140.0,
        eps=eps,
        lam=3.4,
        mu=9.787e-2,
        rho=3282,
        sigma=1.1901,
        #Dec vars
        vareps=ve,
        l=l,
        Rp=R
    )

@dataclass
class separator:
    #Material properties
    brugg: float;           #Bruggerman coefficient
    C:float;                #Specific heat
    p: float;               #Price
    eps: float;             #Volume fraction
    lam: float;             #Thermal conductivity
    rho: float;             #Density

    #Decision variables
    vareps: float;          #Pororsity
    l:float;                #Thikness

def separator_data(ve,l,L):

    eps=volumeFraction(ve,l,L)

    return separator(
        #Mat prop
        brugg=4.0,
        C=1.7,
        p=223.0,
        eps = eps,
        lam=0.2,
        rho=900.0,
        #Dec var
        vareps=ve,
        l=l,
    )

@dataclass
class electrolyte:
    #Material properties
    De: float;              #Liquid phase diffusivity
    p: float;               #Price
    eps: float;             #Volume fraction
    kappa: float;           #Liquid phase conductivity
    rho: float;             #Density

def electrolyte_data(eps_p,eps_o,eps_n):

    eps=eVolumeFraction(eps_p,eps_o,eps_n)

    return electrolyte(
        #Mat prop
        De=7.5e-10,
        p=820.0,
        eps=eps,
        kappa=0.62,
        rho=1220.0
    )

@dataclass
class c_collector:
    #Material properties
    lam: float;             #Thermal conductivity
    rho:float;              #Density
    C: float;               #Specific heat
    sigma: float;           #Conductivity
    p: float;               #Price
    
    #Decision variables
    l:float                 #Thikness

def Al_collector_data(l):

    return c_collector(
        #Mat prop
        C=910.0,
        p=100.0,
        lam=247.0,
        rho=2700.0,
        sigma=3.55e7,
        #Dec var
        l=l,
    )

def Cu_collector_data(l):

    return c_collector(
        #Mat prop
        C=390.0,
        p=100.0,
        lam=371.0,
        rho=8960.0,
        sigma=5.96e7,
        #Dec var
        l=l,
    )