def volumeFraction(vareps:float,l:float,L:float):
    return (1-vareps)*l/L

def interfacialArea(vareps:float,eps:float,R:float):
    return 3*(1-vareps-eps)/R

def eVolumeFraction(eps_p,eps_o,eps_n):
    return 1-eps_p-eps_o-eps_n