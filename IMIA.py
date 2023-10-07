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


from surr_P2D import BatteryP2D
from indicators import SMS, R2, IGDplus, EpsPlus, DeltaP
from settings import nadir, var_keys, oFn_keys, cFn_keys, limits
from IMIA_utils import generate_offspring, nonDomSort, lessContribution, obtainReference_NonDomValid, evaluate, until_valid

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
        ind = evaluate(var, problem)
        pop.append(ind)

    return pd.DataFrame(pop)


"""
==================================================================================================================================================================
Migration
==================================================================================================================================================================
"""

def migration(P:pd.DataFrame,n_islands,n_mig):
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
        q = generate_offspring(P, problem) 
        P = pd.concat([P,q], ignore_index=True)
        non_valid = P[(P[cFn_keys]>0).any(axis=1)]
        if len(non_valid) < 1:
            Rt, Rt_indexes = nonDomSort(P)
        else:
            Rt, Rt_indexes = nonDomSort(non_valid)
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

def IMIA(indicators, P=None, start_gen=0, i_pop=40, f_mig=40, n_mig=1, f_eval=60_000, verbose=False, history_points=2):
    
    n_islands = len(indicators)
    gens = int(np.ceil((f_eval/n_islands)/f_mig))
    samples = np.linspace(0,gens,history_points,dtype=int)
    islands = [None]*n_islands
    sub_P = [None]*n_islands

    if not(isinstance(P, pd.DataFrame)):
        if verbose:
            print('Initializing population...')
        start = timeit.default_timer()
        P = initialize_pop(i_pop*n_islands)
        P = until_valid(P, problem)
        #ref = obtainReference_aproxContruction(P,ref_dirs)
        ref = obtainReference_NonDomValid(P)
        end = timeit.default_timer()
        update_result(P, ref, 0, n_islands*i_pop, end-start, samples)
        
    for g in range(start_gen, gens):
        
        if verbose:
            print(f'Generation: \t {g+1} / {gens}')

        start = timeit.default_timer()

        #ref = obtainReference_aproxContruction(P, ref_dirs)
        if g == start_gen:
            P = until_valid(P, problem)
            ref = obtainReference_NonDomValid(P)
        else:
            ref = obtainReference_NonDomValid(P, ref)

        for i in range(1,n_islands):
            islands[i] = ThreadWithReturnValue(target=IBMOEA, args=(P[i*i_pop:i*i_pop+i_pop],indicators[i],ref,f_mig,verbose))
            islands[i].start()

        sub_P[0] = IBMOEA(P[0:i_pop],indicators[0],ref,f_mig,verbose)

        for i in range(1,n_islands):
            sub_P[i] = islands[i].join()

        P = pd.concat(sub_P,ignore_index=True)
        
        P = migration(P,n_islands,n_mig)

        end = timeit.default_timer()

        update_result(P, ref, g+1, (g+2)*n_islands*f_mig, end-start, samples)
    
    print('All done!')

    return P

"""
==================================================================================================================================================================
File managment
==================================================================================================================================================================
"""
def update_result(P:pd.DataFrame, ref, gen, n_eval, time, samples):
    
    P.to_csv(path + expName + f'_pop_{gen}.csv', index=False)

    if gen > 0 and np.isin(gen-1, samples, invert=True):
        try:
            os.remove(path + expName + f'_pop_{gen-1}.csv')
        except FileNotFoundError:
            pass

    f = P[oFn_keys].values
    c = P[cFn_keys].values
    c = np.where(c < 0, 0, c)
    c = np.sum(c, axis=1)
    valid = f[np.where(c==0)]

    if len(valid) > 0:
        v_HV = ind_HV(valid)
        v_R2 = ind_R2(valid, ref)
        v_IGDp = ind_IGDp(valid, ref)
        v_Ep = ind_Ep(valid, ref)
        v_Dp = ind_Dp(valid, ref)
    else:
        v_HV = '-'
        v_R2 = '-'
        v_IGDp = '-'
        v_Ep = '-'
        v_Dp = '-'

    res = [gen, 
           n_eval, 
           ind_HV(f),
           ind_R2(f, ref),
           ind_IGDp(f, ref),
           ind_Ep(f, ref),
           ind_Dp(f, ref),
           len(valid), 
           v_HV,
           v_R2,
           v_IGDp,
           v_Ep,
           v_Dp,
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

def IMIA_run(exp, Vpack, Iapp, i_pop=40, f_mig=40, n_mig=1, f_eval=60_000, verbose=False, h_p=2, pth="Experiments/IMIA/SurrProblem/"):
    
    global path, expName, ref_dirs, ind_HV, ind_R2, ind_IGDp, ind_Ep, ind_Dp, sampling, problem

    problem = BatteryP2D(Vpack,Iapp)

    ref_dirs = get_reference_directions("energy", 4, 40, seed=1)
    ref_dirs = np.where(ref_dirs==0, 1e-12, ref_dirs)

    sampling = LHS(xlimits=limits)
    
    path =  pth

    ind_HV = HV(ref_point=np.array(nadir))
    ind_R2 = R2()
    ind_IGDp = IGDplus()
    ind_Ep = EpsPlus()
    ind_Dp = DeltaP()

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
                writer.writerow(["n_Gen", "n_Eval", "g_HV", "g_R2", "g_IGD+", "g_e+", "g_Dp", "n_valid", "v_HV", "v_R2", "v_IGD+", "v_e+", "v_Dp", "min_CV", "mean_CV", "max_CV", "time"])

    P = None
    s_gen = 0
    
    if file_found:
        s_gen = res['n_Gen'].max()
        if str(s_gen) != 'nan':
            P = pd.read_csv(path + expName + f'_pop_{s_gen}.csv')
        else:
            s_gen = 0

    P = IMIA([SMS(),R2(),IGDplus(),EpsPlus(),DeltaP()], P=P, start_gen=s_gen, i_pop=i_pop, f_mig=f_mig, n_mig=n_mig, f_eval=f_eval, verbose=verbose, history_points=h_p)
    P.to_csv(path + expName + f'_pop_final', index=False)
    print(P)