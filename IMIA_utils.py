import random
import numpy as np
import pandas as pd
from indicators import individualContribution

from settings import var_keys, oFn_keys, cFn_keys, nadir, max_presure, limits, e


limits = {var_keys[i]: limits[i] for i in range(len(limits))}

"""
==================================================================================================================================================================
Evaluate
==================================================================================================================================================================
"""

def evaluate(x, problem):
    oFn, cFn = problem.evaluate(x)

    res = {
        "SpecificEnergy": oFn[0],
        "SEIGrouth": oFn[1],
        "TempIncrease": oFn[2],
        "Price": oFn[3],
        "UpperViolation": cFn[0],
        "LowerViolation": cFn[1],
        "VolFracViolation": cFn[2],
    }
    
    res.update(x)

    return res


"""
==================================================================================================================================================================
Sorting
==================================================================================================================================================================
"""

def nonDomSort_recursive(F, index):
  fltr = np.logical_not((F[:, None] <= F).all(axis=2).sum(axis=1) == 1)
  F_dom = F[fltr]
  if len(F)==len(F_dom) or len(F_dom)==0:
    return F, index
  else:
    return nonDomSort_recursive(F_dom, index[fltr])

def nonDomSort(Q:pd.DataFrame):
    F = Q[oFn_keys].to_numpy()
    return nonDomSort_recursive(F, np.arange(len(F)))

"""
==================================================================================================================================================================
Selection
==================================================================================================================================================================
"""

def rouletteSelection(pop: pd.DataFrame):
    nadir_volume = np.prod(np.array(nadir))
    probabilities = []
    for _, ind in pop.iterrows():
        volume = 0
        for oFn in oFn_keys:
            volume *= ind[oFn]
        for cFn in cFn_keys:
            volume += ind[cFn]
        probabilities = np.append(probabilities, nadir_volume - volume)

    probabilities[probabilities<=0] = 1e-6
    probabilities = probabilities/np.sum(probabilities)

    parents_index = np.random.choice(range(len(pop)), 2, False, p=probabilities)

    return parents_index

"""
==================================================================================================================================================================
Mating
==================================================================================================================================================================
"""

def SBX(p1,p2,rnd=False):
    u = np.random.rand()
    if u <= 0.5:
        beta = 2*u**(1/31)
    else:
        u -= 0.5
        beta = (1/(1-2*u))**(1/31)

    if random.choice(['+','-']) == '+':
        c = ((p1+p2) + beta*abs(p2-p1))/2
    else:
        c = ((p1+p2) - beta*abs(p2-p1))/2

    if rnd:
        return round(c)
    else:
        return c
    
def PM(p,delta,rnd=False):
    u = np.random.rand()
    if u <= 0.5:
        beta = 2*u**(1/21) - 1.935
    else:
        beta = 1 - (1-2*(u-0.5))**(1/21)

    c = p + beta * 0.3*delta

    if rnd:
        return round(c)
    else:
        return c
    
def repair(var,lmts):
    #Lower bound
    if var < lmts[0]:
        var = 2*lmts[0] - var
        var = repair(var,lmts)
    #Upper bound
    if var > lmts[1]:
        var = 2*lmts[1] - var
        var = repair(var,lmts)

    return var

def generate_offspring(Pop: pd.DataFrame, problem):

    parents = rouletteSelection(Pop)
    parent1 = Pop.iloc[parents[0]]
    parent2 = Pop.iloc[parents[1]]

    off = {}
    for var in var_keys:

        p1 = parent1[var]
        p2 = parent2[var]

        if var == 'mat':
            c = random.choice([p1,p2])
            if np.random.rand() < 1/16:
                c = random.choice(['LCO','LFP'])

        elif var=='Ns' or var=='Np':
            c = SBX(p1,p2,True)
            if np.random.rand() < 1/16:
                c = PM(c,100,True)
            c = repair(c,[1,100])

        else:
            lim = limits[var]
            c = SBX(p1,p2,False)
            if np.random.rand() < 1/16:
                c = PM(c,lim[1]-lim[0],False)
            c = repair(c,lim)
        
        off.update({var: c})
    
    off = evaluate(off, problem)
    return pd.DataFrame(off, index=[0])

"""
==================================================================================================================================================================
Contribution
==================================================================================================================================================================
"""
def lessContribution(indicator, Rt, ref, indexes):
    if indicator.name=='HV':
        ref = np.min(Rt, axis=0)
    g_contr = indicator(Rt, ref)
    index = np.argmin(individualContribution(indicator,g_contr,Rt,ref))
    return indexes[index]

"""
==================================================================================================================================================================
Reference
==================================================================================================================================================================
"""
# def obtainReference_weightPointSelection(P:pd.DataFrame, presure=5e2):

#     ref_dir = get_reference_directions("energy", 4, 40, seed=150)
#     ref_dir = np.where(ref_dir==0, 1e-12, ref_dir)

#     f = P[oFn_keys].values
#     c = P[cFn_keys].values
#     c = np.where(c < 0, 0, c)

#     ref_pnts = np.empty(len(ref_dir))

#     for i in range(len(ref_dir)):
#         ak = f-np.array(np.min(f,axis=0))
#         ak = np.max(ak/ref_dir[i], axis=1)
#         ref_pnts[i] = np.argmin(ak + presure*np.sum(np.square(c), axis=1))
    
#     ref_pnts = np.unique(ref_pnts).astype(int)
#     ref = f[ref_pnts]
#     ref = ref[(ref[:, None] >= ref).all(axis=2).sum(axis=1) == 1]

#     return ref

def achivement(f, f_star, c, e, presure):

    ref = np.empty((len(e),len(f[0])))
    ref_index = np.empty(len(e))

    for i in range(len(e)):

        ref[i] = f[0]
        best_max = np.inf

        for h in range(len(f)):
            maximum = -np.inf
            for j in range(2):
                maximum = max(maximum, (f[h][j]-f_star[j])/e[i][j])
            if (maximum + presure*np.sum(c[h])**2) < best_max:
                ref[i] = f[h]
                ref_index[i] = h
                best_max = maximum

    ref_index = np.unique(ref_index).astype(int)

    return ref, ref_index

def obtain_aprox(ref, ref_dirs, alpha = 1):
    ak = (((np.sum(ref_dirs**alpha, axis=1))**(1/alpha)).reshape(len(ref_dirs),1))
    ak = np.where(ak==0, 1e-12, ak)
    y = ref_dirs/ak
    return y * (np.max(ref, axis=0) - np.min(ref, axis=0)) + np.min(ref, axis=0)

def n2one_dominates(y, ref):
    truth = y <= ref
    return any(np.logical_and(np.logical_and(truth[:,0],truth[:,1]),np.logical_and(truth[:,2],truth[:,3])))

def obtainReference_aproxContruction(P:pd.DataFrame, ref_dirs):

    f = P[oFn_keys].values
    c = P[cFn_keys].values
    c = np.where(c < 0, 0, c)

    f_star = np.min(f, axis=0)

    p = max_presure
    index = []
    while len(index)<len(oFn_keys)+1:
        ref_pnts, index = achivement(f, f_star, c, e, p)
        p *= 0.1
        if p<1:
            break

    alpha = 1
    y = obtain_aprox(ref_pnts, ref_dirs, alpha)

    while n2one_dominates(y, ref_pnts[-1]):
        alpha += 0.05
        if alpha >= 1e3:
            break
        y = obtain_aprox(ref_pnts, ref_dirs, alpha)

    while not(n2one_dominates(y, ref_pnts[-1])):
        alpha -= 0.05
        if alpha <= 0:
            break
        y = obtain_aprox(ref_pnts, ref_dirs, alpha)

    y = y[np.logical_not(np.isnan(y).any(axis=1))]
    y = y[np.logical_not(np.isinf(y).any(axis=1))]

    return y
