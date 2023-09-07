import os
import csv
import timeit
import random
import numpy as np
import pandas as pd
from threading import Thread
from pymoo.indicators.hv import HV
from smt.sampling_methods import LHS
from pymoo.util.ref_dirs import get_reference_directions

from main_p2d import p2d_simulate
#from dumeeProblem import p2d_simulate
from indicators import individualContribution
from indicators import SMS, R2, IGDplus, EpsPlus, DeltaP
from settings import nadir

"""
==================================================================================================================================================================
Global values section
==================================================================================================================================================================
"""

var_keys = ['C', 'la', 'lp', 'lo', 'ln', 'lz', 'Lh', 'Rp', 'Rn', 'Rcell', 'efp', 'efo', 'efn', 'mat', 'Ns', 'Np']
oFn_keys = ['SpecificEnergy', 'SEIGrouth', 'TempIncrease', 'Price']
cFn_keys = ['UpperViolation', 'LowerViolation', 'VolFracViolation']

limits = np.array([[0.2, 4.0],
                   [12e-6, 30e-6],
                   [40e-6, 250e-6],
                   [10e-6, 100e-6],
                   [40e-6, 150e-6],
                   [12e-6, 30e-6],
                   [40e-3, 100e-3],
                   [0.2e-6, 20e-6],
                   [0.5e-6, 50e-6],
                   [4e-3, 25e-3],
                   [0.01, 0.6],
                   [0.01, 0.6],
                   [0.01, 0.6]])

sampling = LHS(xlimits=limits)

limits = {var_keys[i]: limits[i] for i in range(len(limits))}

e = [[1,      1e-12,  1e-12,  1e-12],
     [1e-12,  1,      1e-12,  1e-12],
     [1e-12,  1e-12,  1,      1e-12],
     [1e-12,  1e-12,  1e-12,  1    ],
     [0.25,   0.25,   0.25,   0.25 ]]

ref_dirs = get_reference_directions("energy", 4, 40, seed=1)
ref_dirs = np.where(ref_dirs==0, 1e-12, ref_dirs)

"""
==================================================================================================================================================================
Evaluate
==================================================================================================================================================================
"""

def evaluate(x):
    oFn, cFn, _, fail = p2d_simulate(x, Vpack, Iapp)

    if (fail[0] != ''):
        oFn = nadir

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
Initialize
==================================================================================================================================================================
"""

def initialize_pop(pop_size):
    smpls = sampling(pop_size)
    pop = []
    for x in smpls:
        var = {
            "C": x[0],
            "la": x[1],
            "lp": x[2],
            "lo": x[3],
            "ln": x[4],
            "lz": x[5],
            "Lh": x[6],
            "Rp": x[7],
            "Rn": x[8],
            "Rcell": x[9],
            "efp": x[10],
            "efo": x[11],
            "efn": x[12],
            "mat": random.choice(['LCO','LFP']),
            "Ns": random.choice(range(1,101)),
            "Np": random.choice(range(1,101)),
        }
        ind = evaluate(var)
        pop.append(ind)

    return pd.DataFrame(pop)


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

def nonDomSort(Q:pd.DataFrame, ):
    F = Q[oFn_keys].to_numpy()
    #F = (F - F.min(axis=0))/(F.max(axis=0)-F.min(axis=0))
    return nonDomSort_recursive(F, np.arange(len(F)))

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

def generate_offspring(Pop: pd.DataFrame):

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
    
    off = evaluate(off)
    return pd.DataFrame(off, index=[0])

"""
==================================================================================================================================================================
Contribution
==================================================================================================================================================================
"""
def lessContribution(indicator, Rt, ref, indexes):
    g_contr = indicator(Rt, ref)
    index = np.argmin(individualContribution(indicator,g_contr,Rt,ref))
    return indexes[index]

"""
==================================================================================================================================================================
Reference
==================================================================================================================================================================
"""
def obtainReference_weightPointSelection(P:pd.DataFrame, presure=5e2):

    ref_dir = get_reference_directions("energy", 4, 40, seed=150)
    ref_dir = np.where(ref_dir==0, 1e-12, ref_dir)

    f = P[oFn_keys].values
    c = P[cFn_keys].values
    c = np.where(c < 0, 0, c)

    ref_pnts = np.empty(len(ref_dir))

    for i in range(len(ref_dir)):
        ak = f-np.array(np.min(f,axis=0))
        ak = np.max(ak/ref_dir[i], axis=1)
        ref_pnts[i] = np.argmin(ak + presure*np.sum(np.square(c), axis=1))
    
    ref_pnts = np.unique(ref_pnts).astype(int)
    ref = f[ref_pnts]
    ref = ref[(ref[:, None] >= ref).all(axis=2).sum(axis=1) == 1]

    return ref

def achivement(f, f_star, c, e, presure=1):

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

    return ref, ref_index

def obtain_aprox(ref, ref_dirs, alpha = 1):
    ak = (((np.sum(ref_dirs**alpha, axis=1))**(1/alpha)).reshape(len(ref_dirs),1))
    ak = np.where(ak==0, 1e-12, ak)
    y = ref_dirs/ak
    return y * (np.max(ref, axis=0) - np.min(ref, axis=0)) + np.min(ref, axis=0)

def n2one_dominates(y, ref):
    truth = y <= ref
    return any(np.logical_and(np.logical_and(truth[:,0],truth[:,1]),np.logical_and(truth[:,2],truth[:,3])))

def obtainReference_aproxContruction(P:pd.DataFrame):

    f = P[oFn_keys].values
    c = P[cFn_keys].values
    c = np.where(c < 0, 0, c)

    f_star = np.min(f, axis=0)

    ref_pnts, _ = achivement(f, f_star, c, e)

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

"""
==================================================================================================================================================================
Migration
==================================================================================================================================================================
"""

def migration(P:pd.DataFrame,n_islands=5,n_mig=1):
    ul = int(len(P)/n_islands)
    p = np.ones((n_islands,n_islands,n_mig),dtype=int)

    for i in range(n_islands):
        p[i] = np.random.choice(np.arange(i*ul,i*ul+ul), (n_islands,n_mig), False)
    
    mig_map = {p[j][i][k]:p[i][j][k] for i in range(n_islands) for j in range(n_islands) for k in range(n_mig)}

    P.rename(index=mig_map, inplace=True)
    P.sort_index(inplace=True)

    return P

"""
==================================================================================================================================================================
IBMOEA section
==================================================================================================================================================================
"""

def IBMOEA(P:pd.DataFrame,I,ref,f_mig,verbose=False):

    for g in range(f_mig):
        q = generate_offspring(P) 
        P = pd.concat([P,q], ignore_index=True)
        Rt, Rt_indexes = nonDomSort(P)
        if len(Rt_indexes) > 1:
            r = lessContribution(I,Rt,ref,Rt_indexes)
        else:
            r = Rt_indexes[0]
        P = P.drop(r).reset_index(drop=True)

        if verbose:
            if g%int(f_mig/5)==0 or g+1==f_mig:
                print(f'\t {I.name}: \t {g+1} / {f_mig}')

    return P


class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,**self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return

"""
==================================================================================================================================================================
IMIA section
==================================================================================================================================================================
"""

def IMIA(indicators, P=None, start_gen=0, i_pop=40, f_mig=40, n_mig=1, f_eval=60_000, verbose=False):
    
    n_islands = len(indicators)
    gens = int(np.ceil((f_eval/n_islands)/f_mig))
    samples = np.linspace(1,gens,15,dtype=int)
    islands = [None]*n_islands
    sub_P = [None]*n_islands

    if not(isinstance(P, pd.DataFrame)):
        if verbose:
            print('Initializing population...')
        P = initialize_pop(i_pop*n_islands)
        
    for g in range(start_gen, gens):
        
        if verbose:
            print(f'Generation: \t {g+1} / {gens}')

        start = timeit.default_timer()

        P = migration(P,n_islands,n_mig)

        ref = obtainReference_aproxContruction(P)

        for i in range(1,n_islands):
            islands[i] = ThreadWithReturnValue(target=IBMOEA, args=(P[i*i_pop:i*i_pop+i_pop],indicators[i],ref,f_mig,verbose))
            islands[i].start()

        sub_P[0] = IBMOEA(P[0:i_pop],indicators[0],ref,f_mig,verbose)

        for i in range(1,n_islands):
            sub_P[i] = islands[i].join()

        P = pd.concat(sub_P,ignore_index=True)
        
        end = timeit.default_timer()

        update_result(P, g+1, (g+2)*n_islands*f_mig, end-start, samples)
    
    print('All done!')

    return P

"""
==================================================================================================================================================================
File managment
==================================================================================================================================================================
"""
def update_result(P:pd.DataFrame, gen, n_eval, time, samples):
    
    P.to_csv(path + expName + f'_pop_{gen}.csv', index=False)

    if np.isin(gen-1, samples, invert=True):
        try:
            os.remove(path + expName + f'_pop_{gen-1}.csv')
        except FileNotFoundError:
            pass

    f = P[oFn_keys].values
    c = P[cFn_keys].values
    c = np.where(c < 0, 0, c)
    c = np.sum(c, axis=1)
    valid = f[np.where(c==0)]

    res = [gen, 
           n_eval, 
           ind(f), 
           len(valid), 
           ind(valid), 
           np.min(c), 
           np.mean(c), 
           np.max(c), 
           time]
    
    with open(path + expName + '_res.csv', 'a') as resf:
        writer = csv.writer(resf)
        writer.writerow(res)

    return True

"""
==================================================================================================================================================================
Run section
==================================================================================================================================================================
"""
ind = HV(ref_point=np.array(nadir))

path = "Experiments/IMIA/"

while True:   
    app = input("Application required?  EV/DR/CP: ")

    if app == 'EV':
        Vpack = 48
        Iapp = -80
        break
    elif app == 'DR':
        Vpack = 15
        Iapp = -22
        break
    elif app == 'CP':
        Vpack = 3.7
        Iapp = -3
        break
    else:
        print('Invalid Application, try again!')

exp = int(input("Experiment number: "))

expName = f'IMIA_V{Vpack}_I{abs(Iapp)}_E{exp}'
print(expName)

file_found=False



try:
    res = pd.read_csv(path + expName + '_res.csv')
    print('Checkpoint found, resuming')
    file_found = True 

except FileNotFoundError:
    print('No checkpoint found, starting over')
    with open(path + expName + '_res.csv', 'a') as resf:
            writer = csv.writer(resf)
            writer.writerow(["n_Gen", "n_Eval", "general_HV", "n_valid", "valid_HV", "min_CV", "mean_CV", "max_CV", "time"])

if file_found==True:
    s_gen = res['n_Gen'].max()
    try:
        P = pd.read_csv(path + expName + f'_pop_{s_gen}.csv')
    except FileNotFoundError:
        P = None
        s_gen = 0
    P = IMIA([SMS(),R2(),IGDplus(),EpsPlus(),DeltaP()], P=P, start_gen=s_gen, verbose=True)

else:
    P = IMIA([SMS(),R2(),IGDplus(),EpsPlus(),DeltaP()], verbose=True)

P.to_csv(path + expName + f'_pop_final', index=False)
print(P)