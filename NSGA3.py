import csv
import timeit
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination

from pymooProblem import BatteryP2D_surr, BatteryP2D
from settings import nadir


def stepSolver(algorithm, path):

    header = ['C','la','lp','lo','ln','lz','Lh','Rcell','Rp','Rn','efp','efo','efn','mat','Ns','Np','SpecificEnergy','SEIGrouth','TempIncrease','Price','UpperViolation','LowerViolation','VolFracViolation']
    objFun = ['SpecificEnergy','SEIGrouth','TempIncrease','Price']
    constFun = ['UpperViolation','LowerViolation','VolFracViolation']

    while algorithm.has_next():
    
        start = timeit.default_timer()
        algorithm.next()
        end = timeit.default_timer()

        if algorithm.result().F is not None:
            valid = len(algorithm.result().F)
            hv = ind(algorithm.result().F)
            
        else:
            valid = 0
            hv = 0
            
        F = algorithm.result().pop.get("F")
        
        res = [algorithm.n_gen, 
                algorithm.evaluator.n_eval, 
                ind(F), 
                valid, 
                hv, 
                algorithm.pop.get("cv").min(), 
                algorithm.pop.get("cv").mean(), 
                algorithm.pop.get("cv").max(), 
                end - start]

        with open(path + expName + f'_res.csv', 'a') as resf:
            writer = csv.writer(resf)
            writer.writerow(res)
        
        if algorithm.n_gen in pop_pnts:

            x = algorithm.pop.get("X")
            f = algorithm.pop.get("F")
            g = algorithm.pop.get("G")

            with open(path + expName + f'_pop_{algorithm.n_gen}.csv', 'w') as popf:
                
                writer = csv.DictWriter(popf, fieldnames = header)
                writer.writeheader()

                for i in range(len(x)):
                    xfg_dict = {}
                    xfg_dict.update(x[i])
                    xfg_dict.update(dict(zip(objFun, f[i])))
                    xfg_dict.update(dict(zip(constFun, g[i])))
                    writer.writerow(xfg_dict)

    res = ["n_Gen", "n_Eval", "general_HV", "n_validSolutions", "valid_HV", "min_CV", "mean_CV", "max_CV", "time"]

    with open(path + expName + f'_res.csv', 'a') as resf:
        writer = csv.writer(resf)
        writer.writerow(res)



def NSGA3_run(exp, Vpack, Ipack, popul=200, func_eval=60_000, verbose=True, h_p=2, pth="Experiments/"):

    global expName, pop_pnts, ind
    ref_point = np.array(nadir)
    ind = HV(ref_point=ref_point)

    gens = int(func_eval/popul)
    pop_pnts = np.linspace(2, gens, h_p, dtype = "int")

    expName = f'NSGA3/NSGA3_V{Vpack}_I{Ipack}_E{exp}'
    print(expName)

    problem = BatteryP2D_surr(Vpack,Ipack,pth)

    ref_dirs = get_reference_directions("energy", 4, popul)

    termination = get_termination("n_gen", gens)

    algorithm = NSGA3(pop_size=popul,
                sampling=MixedVariableSampling(),
                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
                ref_dirs=ref_dirs)

    algorithm.setup(problem, termination=termination, verbose=verbose)

    res = stepSolver(algorithm, pth)

    return res
