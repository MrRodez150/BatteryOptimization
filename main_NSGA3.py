import csv
import dill
import timeit
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination

from pymooProblem import BatteryP2D
from settings import nadir


def stepSolver(algorithm, path):

    header = ['C','la','lp','lo','ln','lz','Lh','Rcell','Rp','Rn','efp','efo','efn','mat','Ns','Np','SpecificEnergy','SEIGrouth','TempIncrease','Price','UpperViolation','LowerViolation','VolFracViolation']
    objFun = ['SpecificEnergy','SEIGrouth','TempIncrease','Price']
    constFun = ['UpperViolation',	'LowerViolation','VolFracViolation']

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

        with open (path + expName + "_checkpoint", "wb") as cpf:
            dill.dump(algorithm, cpf)

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

#Configuration
func_eval = 60000
ref_point = np.array(nadir)
ind = HV(ref_point=ref_point)
path = "Experiments/NSGA3/"

#Aplication requirements
while True:
    app = input("Application required?  EV/DR/CP: ")

    if app == 'EV':
        Vpack = 48
        Ipack = -80
        break
    elif app == 'DR':
        Vpack = 15
        Ipack = -22
        break
    elif app == 'CP':
        Vpack = 3.7
        Ipack = -3
        break
    else:
        print('Invalid Application, try again!')

popul = int(input("Population: "))
gens = int(func_eval/popul)
pop_pnts = np.linspace(2, gens, 40, dtype = "int")
exp = int(input("Experiment number: "))

expName = f'NSGA3_P{popul}_G{gens}_V{Vpack}_I{Ipack}_E{exp}'
print(expName)

try:
    with open (path + expName + "_checkpoint", "rb") as cpf:
        algorithm = dill.load(cpf)
    print("Chekpoint loaded")
except:

    print("No checkpoint found, starting over")

    problem = BatteryP2D(Vpack,Ipack)

    ref_dirs = get_reference_directions("energy", 4, popul)

    termination = get_termination("n_gen", gens)

    algorithm = NSGA3(pop_size=popul,
                sampling=MixedVariableSampling(),
                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
                ref_dirs=ref_dirs)

    algorithm.setup(problem, termination=termination, verbose=True)

res = stepSolver(algorithm, path)
