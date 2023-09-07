import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.core.survival import Survival
from pymoo.core.selection import Selection
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize

from indicators import individualContribution, RieszEnergy

path = "Experiemnts/IMIA/AproxPareto/"

def insert(pop_F, name, max_size):
    A = np.loadtxt(path + f'A_{name}.txt')

    for i in range(len(pop_F)):
        b = pop_F[i]
        for j in range(len(A)):
            a = A[j]
            if all(b<=a):
                A = np.delete(A,j,0)
    A = np.append(A,pop_F)

    Es = RieszEnergy()
    t_cont = Es(A)
    while len(a)>max_size:
        i_cont = (individualContribution(Es,t_cont,A))/2
        i = np.argmin(i_cont)
        A = np.delete(A, i, 0)
    
    np.savetxt(path + f'A_{name}.txt', A)


class LeastIndicatorContributionSurvival(Survival):

    def __init__(self, indicator, eps=10.0) -> None:
        super().__init__(filter_infeasible=False)
        self.eps = eps

        if indicator is None:
            raise Exception("Please provide the comparing function for the tournament selection!")
        else:
            self.indicator = indicator

    def _do(self, _, pop, *args, n_survive=None, ideal=None, nadir=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=True)

        # if the boundary points are not provided -> estimate them from pop
        if ideal is None:
            ideal = F.min(axis=0)
        if nadir is None:
            nadir = F.max(axis=0)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # get the actual front as individuals
            front = pop[front]
            front.set("rank", k)

            if len(survivors) + len(front) > n_survive:

                # normalize all the function values for the front
                F = front.get("F")
                F = normalize(F, ideal, nadir)

                # Obtain the reference 
                if self.indicator.pf_ref:
                    ref = np.loadtxt(path + f'A_{self.indicator.name}.txt')
                else:
                    ref = None
                
                # finally do the computation
                g_contr = self.indicator(F, ref)
                i_contr = individualContribution(self.indicator, g_contr, F)

                # current front sorted by crontribution
                while len(survivors) + len(front) > n_survive:
                    ind = np.argmin(i_contr)
                    front = np.delete(front, ind)
                    i_contr = np.delete(i_contr,ind,0)

            # extend the survivors by all or selected individuals
            survivors.extend(front)

        return Population.create(*survivors)
    

class CVWeightedRandomSelection(Selection):

    def _do(self, _, pop, n_select, n_parents, **kwargs):
        # number of random individuals needed
        n_random = n_select * n_parents

        cv = pop.get("CV")
        cv[cv<0] = 0
        dist_cv = np.sum(cv, axis=1)
        max_cv = np.max(dist_cv)+1e-6
        dist = max_cv-dist_cv
        probability = dist/np.sum(dist)

        P = np.random.choice(range(len(pop)), n_random, p=probability)

        return np.reshape(P, (n_select, n_parents))



class IBEMOA(GeneticAlgorithm):

    def __init__(self,
                 indicator,
                 pop_size=40,
                 max_file_size=200,
                 sampling=MixedVariableSampling(),
                 crossover=SBX(eta=30, prob=1.0),
                 mutation=PM(eta=20, prob=1/16),
                 mating=MixedVariableMating(selection=CVWeightedRandomSelection(),
                                            eliminate_duplicates=MixedVariableDuplicateElimination()),
                 eliminate_duplicates=MixedVariableDuplicateElimination(),
                 n_offsprings=2,
                 normalize=True,
                 output=MultiObjectiveOutput(),
                 **kwargs):

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         crossover=crossover,
                         mutation=mutation,
                         mating=mating,
                         survival=LeastIndicatorContributionSurvival(indicator),
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         advance_after_initial_infill=False,
                         **kwargs)

        self.max_file_size = max_file_size
        self.normalize = normalize
        self.indicator_name = indicator.name

    def _advance(self, infills=None, **kwargs):

        ideal, nadir = None, None

        F = self.pop.get("F")

        # estimate ideal and nadir from the current population (more robust then from doing it from merged)
        if self.normalize:
            ideal, nadir = F.min(axis=0), F.max(axis=0) + 1e-32

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self,
                                    ideal=ideal, nadir=nadir, **kwargs)
        
        insert(self.pop, self.indicator_name, self.max_file_size)
