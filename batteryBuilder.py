from dataclasses import dataclass
from auxiliaryExp import interfacialArea, volumeFraction, eVolumeFraction

@dataclass
class separator_constants:    
    brugg: float        #Bruggerman coefficient
    Cp:float            #Specific heat
    ce_0: float         #Initial electrolyte concentration
    l:float             #Thikness
    p:float             #Price
    eps: float          #Porocity
    epsf: float         #Volume fraction
    lam: float          #Thermal conductivity
    rho: float          #Density

def PP_separator_data(epsf,l):

    #epsf = volumeFraction(eps,l,L)

    return separator_constants(
        brugg=4, 
        Cp=1.7,
        ce_0 = 1000,
        l=l,
        p=223,
        eps=0.3,
        epsf=epsf,
        lam=0.2,
        rho=900)

@dataclass
class current_collector_constants:    
    Cp: float           #Specific heat
    l:float             #Thikness
    p: float            #Price
    tipo:str            #Material type
    lam: float          #Thermal conductivity
    rho:float           #Density
    sigma: float        #Electrical conductivity

def Al_cc_data(l):
    return current_collector_constants(
        Cp=910,
        l=l,
        p=100,
        lam=247,
        rho=2700,
        sigma=3.55e7,
        tipo='a')

def Cu_cc_data(l):
    return current_collector_constants(
        Cp=390,
        l=l,
        p=100,
        lam=371,
        rho=8960,
        sigma=5.96e7,
        tipo='z')


@dataclass
class electrode_constants:
    a: float            #Specific interfacial area
    brugg:float         #Bruggerman coefficient
    Cp: float           #Specific heat
    ce_0: float         #Initial electrolyte concentration
    cavg: float         #Average solid phase concentration
    cmax: float         #Maximum solid phase concentration
    Ds: float           #Solid phase diffusivity
    ED: float           #Activation energy for the solid diffusion
    Ek: float           #Activation energy for the reaction constant
    k:float             #Reaction rate constant
    l:float             #Thikness
    p: float            #Price
    Rp: float           #Particle radius
    tipo: str           #Material type
    eps: float          #Porosity
    epsf: float         #Volume fraction
    lam: float          #Thermal conductivity
    mu: float           #Specific mass
    rho: float          #Density
    sigma: float        #Solid phase conductivity
   

def LFP_electrode_data(epsf,R,l):

    a = interfacialArea(epsf,R)
    
    return electrode_constants(
        a = a, #885000,
        brugg = 4,
        Cp = 1260,
        ce_0 = 1000,
        cavg = 25751,
        cmax = 51554,
        Ds = 4.295e-14,
        ED = 5000,
        Ek = 5000,
        k = 7.882e-12,
        l = l,
        p=70,
        Rp = R,
        tipo ='p',
        eps = 0.3,
        epsf = epsf,
        lam = 0.15,
        mu = 1.577e-1,
        rho = 1132, 
        sigma = 0.4977)

def LCO_electrode_data(epsf,R,l):

    a = interfacialArea(epsf,R)
    
    return electrode_constants(
        a = a, #885000,
        brugg = 4,
        Cp = 1269,
        ce_0 = 1000,
        cavg = 25751,
        cmax = 51554,
        Ds = 1.806e-14,
        ED = 5000,
        Ek = 5000,
        k = 7.898e-12,
        l = l,
        p = 140,
        Rp = R,
        tipo ='p',
        eps = 0.3,
        epsf = epsf,
        lam = 3.4,
        mu = 9.787e-2,
        rho = 3282, 
        sigma = 1.1901)

def C6_electrode_data(epsf,R,l):

    return electrode_constants(
        a = 723600,
        brugg = 4,
        Cp = 706.9,
        ce_0 = 1000,
        cavg = 26128,
        cmax = 30555,
        Ds = 3.9e-14, 
        ED = 5000,
        Ek = 5000,
        k = 5.031e-11,
        l = l,
        p = 60,
        Rp= R,
        tipo = 'n',
        eps = 0.6,
        epsf = epsf,
        lam = 1.7,
        mu = 1.201e-2,
        rho = 2160,
        sigma = 1000)

@dataclass
class electrolyte_constants:

    De: float;              #Liquid phase diffusivity
    p: float;               #Price
    epsf: float              #Volume fraction
    kappa: float;           #Liquid phase conductivity
    rho: float;             #Density

def LPF_electrolyte_data(e_p,e_o,e_n):

    epsf = eVolumeFraction(e_p,e_o,e_n)

    return electrolyte_constants(
        De=7.5e-10,
        p=820.0,
        epsf=epsf,
        kappa=0.62,
        rho=1220.0
    )

def build_battery(mat, efp, efo, efn, Rp, Rn, la, lp, lo, ln, lz):
    if mat == 'LCO':
        p_data = LCO_electrode_data(efp,Rp,lp)
    elif mat == 'LFP':
        p_data = LFP_electrode_data(efp,Rp,lp)
    else:
        raise ValueError('Invalid positive electrode material')
    n_data = C6_electrode_data(efn,Rn,ln)
    o_data = PP_separator_data(efo,lo)
    a_data = Al_cc_data(la)
    z_data = Cu_cc_data(lz)
    e_data = LPF_electrolyte_data(efp,efo,efn)
    return p_data, n_data, o_data, a_data, z_data, e_data


"""Previous values"""
"""
def sep_constants():
    return separator_constants(
        brugg=4, 
        Cp=700, 
        l=25e-6, 
        eps=0.724, 
        lam=0.16, 
        rho=1100)

def a_cc_constants():
    return current_collector_constants(
        Cp=897,
        l=1e-5,
        lam=237,
        rho=2700,
        sigma=3.55e7,
        tipo='a')

def z_cc_constants():
    return current_collector_constants(
        Cp=385,
        l=1e-5,
        lam=401,
        rho=8940,
        sigma=5.96e7,
        tipo='z')

def p_electrode_constants():
    
    return electrode_constants(
        a = 885000,
        brugg = 4,
        Cp = 700,
        cavg = 25751,
        cmax = 51554,
        Ds = 1e-14,
        ED = 5000,
        Ek = 5000,
        k = 2.334e-11,
        l = 8e-5,
        Rp = 2e-6,
        tipo ='p',
        epsf = 0.025,
        eps = 0.385,
        lam = 2.1,
        rho = 2500, 
        sigma = 100)

def n_electrode_constants():
    
    return electrode_constants(
        a= 723600,
        brugg = 4,
        Cp = 700,
        cavg = 26128,
        cmax = 30555,
        Ds = 3.9e-14, 
        ED = 5000,
        Ek = 5000,
        k = 5.031e-11,
        l = 8.8e-5,
        Rp= 2e-6,
        tipo = 'n',
        eps = 0.485,
        epsf = 0.0326,
        lam = 1.7,
        rho = 2500,
        sigma = 100)

"""