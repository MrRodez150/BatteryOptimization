import csv
import numpy as np
import timeit
from pymoo.indicators.hv import HV
from pymoo.decomposition.pbi import PBI

from IMIA_pymoo import IMIA
from pymooProblem import BatteryP2D
from indicators import SMS, R2, IGDplus, EpsPlus, DeltaP
from settings import nadir

def stepSolver(algorithm):

    header = ['C','la','lp','lo','ln','lz','Lh','Rcell','Rp','Rn','efp','efo','efn','mat','Ns','Np','SpecificEnergy','SEIGrouth','TempIncrease','Price','UpperViolation','LowerViolation','VolFracViolation']
    objFun = ['SpecificEnergy','SEIGrouth','TempIncrease','Price']
    constFun = ['UpperViolation','LowerViolation','VolFracViolation']

    while algorithm.has_next():
    
        start = timeit.default_timer()
        algorithm.next()
        end = timeit.default_timer()

        for name in algorithm.indicators:
            island = algorithm.islands[name]
            if island.result().F is not None:
                valid = len(island.result().F)
                hv = ind(island.result().F)
                
            else:
                valid = 0
                hv = 0
            
            res = [island.n_gen, 
                    island.evaluator.n_eval, 
                    ind(island.pop.get("F")), 
                    valid, 
                    hv, 
                    island.pop.get("cv").min(), 
                    island.pop.get("cv").mean(), 
                    island.pop.get("cv").max(), 
                    end - start]

            with open(path + expName + f'_{name}_res.csv', 'a') as resf:
                writer = csv.writer(resf)
                writer.writerow(res)
            
            x = island.pop.get("X")
            f = island.pop.get("F")
            g = island.pop.get("G")

            with open(path + expName + f'_{name}_pop_{island.n_gen}.csv', 'w') as popf:
                
                writer = csv.DictWriter(popf, fieldnames = header)
                writer.writeheader()

                for i in range(len(x)):
                    xfg_dict = {}
                    xfg_dict.update(x[i])
                    xfg_dict.update(dict(zip(objFun, f[i])))
                    xfg_dict.update(dict(zip(constFun, g[i])))
                    writer.writerow(xfg_dict)

    res = ["n_Gen", "n_Eval", "general_HV", "n_validSolutions", "valid_HV", "min_CV", "mean_CV", "max_CV", "time"]

    with open(path + expName + f'_{name}_res.csv', 'a') as resf:
        writer = csv.writer(resf)
        writer.writerow(res)


#Configuration
ind = HV(ref_point=np.array(nadir))
path = "Experiments/IMIA/"

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

exp = int(input("Experiment number: "))

expName = f'IMIA_V{Vpack}_I{Ipack}_E{exp}'
print(expName)

problem = BatteryP2D(Vpack,Ipack)

indicators = [SMS(),
              IGDplus(),
              EpsPlus(),
              DeltaP(),
              R2([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0.25,0.25,0.25,0.25]],
                 PBI(nadir_point=np.array(nadir)))]

algorithm = IMIA(problem, 
                 indicators_set=indicators, 
                 pop_size = 200,
                 n_eval = 60_000,
                 f_mig = 40,
                 n_mig = 1,
                 verbose = True)

algorithm.initialize_islands()

res = stepSolver(algorithm)
