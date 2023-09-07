import time
import numpy as np
from pymoo.core.mixed import MixedVariableSampling
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from IBEMOA import IBEMOA, insert

path = "Experiments/IMIA/AproxPareto/"

class IMIA():
    def __init__(self,
                 problem,
                 indicators_set,
                 initialization=MixedVariableSampling(),
                 pop_size = 200,
                 n_eval = 60_000,
                 f_mig = 40,
                 n_mig = 1,
                 verbose = False):
        
        self.indicators_set = indicators_set
        self.indicators = [ind.name for ind in indicators_set]

        self.pop_size = pop_size
        self.f_mig = f_mig
        self.i_pop = pop_size/len(self.indicators)
        self.n_mig = n_mig
        self.migrations = 0

        self.problem =problem
        self.initialization = initialization
        self.termination = get_termination("n_eval", n_eval/len(self.indicators))
        self.evaluator = Evaluator()
        self.verbose = verbose
        

    def run(self):
        self.initialize_islands()
        while self.has_next():
            self.next()
        return self.result()
    
    def has_next(self):
        return not all([self.islands[ind].termination.has_terminated() for ind in self.indicators])
    
    def next(self):
        for ind in self.indicators:
            if self.verbose:
                print(ind)
            algorithm = self.islands[ind]
            while algorithm.n_iter < self.migrations*self.f_mig:
                algorithm.next()
        self.migrate()

    def migrate(self):
        for _ in range(len(self.n_mig)):
            for i1 in self.indicators:
                for i2 in self.indicators:
                    if i1 != i2:
                        p1 = self.islands[i1].pop
                        p2 = self.islands[i2].pop
                        i = np.random.randint(self.i_pop)
                        r = p1[i]
                        p2 = Population.merge(p2,r)
                        self.islands[i1].pop = np.delete(p1,i)

        for name in self.indicators:
            insert(self.islands[name].pop, name, self.pop_size)    

        self.migrations += 1

    def initialize_islands(self):
        islands = {ind.name: IBEMOA(ind, pop_size=self.i_pop, max_file_size=self.pop_size) for ind in self.indicators_set}
        pop = self.initialize_pop()
        for i in range(len(self.indicators)):
            islands[self.indicators[i]].setup(self.problem, 
                                              termination=self.termination,
                                              start_time = time.time(),
                                              n_iter = 1,
                                              opt = None,
                                              pop=pop[i*self.i_pop : i*self.i_pop+self.i_pop],
                                              is_initialized = True,
                                              verbose = self.verbose)
        self.initialize_reference(pop)
        self.islands = islands
        if self.verbose:
            print('Islands initialized')


    def initialize_reference(self, pop):
        
        front = NonDominatedSorting().do(pop.get("F"), only_non_dominated_front=True)
        for name in self.indicators:
            np.savetxt(path + f'{name}_pf.txt', pop[front].get("F"))
        if self.verbose:
            print('References loaded to files')


    def initialize_pop(self):
        
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)

        if pop is not None:
            pop.set("n_gen", 1)
            pop.set("n_iter", 1)
            self.evaluator.eval(self.problem, pop)

        return pop
    

    def result(self):

        return {ind: self.islands[ind].result() for ind in self.indicators}

