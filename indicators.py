from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV
from pymoo.decomposition.pbi import PBI
import numpy as np

from settings import nadir


class SMS():
    def __init__(self):
        self.pf_ref = False
        self.name = 'HV'
    
    def __call__(self, F, ref):
        return self._do(F, ref)
    
    def _do(self, F, ref):
        ind = HV(ref_point=ref,nds=False)
        return ind(F)

class R2():
    def __init__(self, weights=[[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,0],
                                [0,0,0,1],
                                [0.25,0.25,0.25,0.25]],
                         unary_func=PBI()):
        self.pf_ref = False
        self.weights = np.array(weights)
        self.unary_func = unary_func
        self.name = 'R2'

    def __call__(self, F, ref):
        return self._do(F)

    def _do(self, F):

        res = 0
        for w in self.weights:
            val0 = -np.inf
            for a in F:
                val = self.unary_func(a,w)[0][0]
                if val > val0:
                    val0 = val
            res += val0

        return res/len(self.weights)
    
class IGDplus():
    def __init__(self):
        self.pf_ref = True
        self.name = 'IGD+'
    
    def __call__(self, F, pf):
        return self._do(F, pf)
    
    def _do(self, F, pf):
        ind = IGDPlus(pf)
        return ind(F)
    

class EpsPlus():
    def __init__(self):
        self.pf_ref = True
        self.name = 'e+'

    def __call__(self, F, pf):
        return self._do(F, pf)

    def _do(self, F, pf):

        val2 = -np.inf
        for z in pf:
            val1 = np.inf
            for a in F:
                val0 = a[0] - z[0]
                for i in range(1,len(a)):
                    dif = a[i] - z[i]
                    if dif > val0:
                        val0 = dif
                if val0 < val1:
                    val1 = val0
            if val1 > val2:
                val2 = val1

        return val2


class DeltaP():
    def __init__(self):
        self.pf_ref = True
        self.name = 'D_p'

    def __call__(self, F, pf):
        return self._do(F, pf)

    def _do(self, F, pf):

        return max(GD(pf)(F),IGD(pf)(F))
    

def individualContribution(indicator, total_contr, F, pf):

    res = np.zeros(len(F))
    for i in range(len(F)):
        F1 = np.delete(F,i,0)
        res[i] = total_contr - indicator(F1, pf)

    return res
        
class RieszEnergy():

    def __call__(self, F):
        return self._do(F)

    def _do(self, F):

        res = 0
        for i in range(len(F)):
            a = F[i]
            for b in np.delete(F,i,0):
                  euc_dist = 1 / (np.sqrt(np.sum((a - b) ** 2)) ** (len(a)**2))
                  res += euc_dist

        return res