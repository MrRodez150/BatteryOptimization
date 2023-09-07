import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from main_p2d import p2d_simulate

from settings import nadir

class BatteryP2D(ElementwiseProblem):
    def __init__(self, V, I, **kwargs):

        self.Vpack = V
        self.Iapp = I

        vars = {
            "C": Real(bounds=(0.2, 4.0)),
            "la": Real(bounds=(12e-6, 30e-6)),
            "lp": Real(bounds=(40e-6, 250e-6)),
            "lo": Real(bounds=(10e-6, 100e-6)),
            "ln": Real(bounds=(40e-6, 250e-6)),
            "lz": Real(bounds=(12e-6, 30e-6)),
            "Lh": Real(bounds=(40e-3, 100e-3)),
            "Rp": Real(bounds=(0.2e-6, 20e-6)),
            "Rn": Real(bounds=(0.5e-6, 50e-6)),
            "Rcell": Real(bounds=(4e-3, 25e-3)),
            "efp": Real(bounds=(0.01, 0.6)),
            "efo": Real(bounds=(0.01, 0.6)),
            "efn": Real(bounds=(0.01, 0.6)),
            "mat": Choice(options=['LCO','LFP']),
            "Ns": Integer(bounds=(1, 100)),
            "Np": Integer(bounds=(1, 100)),
        }
        
        super().__init__(vars=vars, n_obj=4, n_ieq_constr=3, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        oFn, cFn, _, fail = p2d_simulate(x, self.Vpack, self.Iapp)

        if (fail[0] == ''):
            out["F"] = oFn
            out["G"] = cFn
        else:
            out["F"] = nadir
            out["G"] = cFn

