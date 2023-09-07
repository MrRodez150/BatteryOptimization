import numpy as np

def SBX(p1,p2,rnd=False):
    u = np.random.rand()
    if u <= 0.5:
        beta = 2*u**(1/31)
    else:
        u -= 0.5
        beta = (1/(1-2*u))**(1/31)
    
    c1 = ((p1+p2) - beta*abs(p2-p1))/2
    c2 = ((p1+p2) + beta*abs(p2-p1))/2

    if rnd:
        return round(c1), round(c2)
    else:
        return c1, c2

def PM(p, delta, rnd):
    u = np.random.rand()
    if u <= 0.5:
        beta = 2*u**(1/31) - 1
    else:
        u -= 0.5
        beta = 1 - (1-2*u)**(1/31)
    
    c = p + beta*delta

    if rnd:
        return round(c)
    else:
        return c