import joblib
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from main_p2d import p2d_simulate

from fghFunctions import batteryPrice, ineqConstraintFunctions
from batteryBuilder import build_battery
from auxiliaryExp import area
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



class BatteryP2D_surr(ElementwiseProblem):
    def __init__(self, V, I, path, **kwargs):

        path = path+'surr/'
        app = f'{V}_{abs(I)}'

        if app == '48_80':
            self.ES_m = joblib.load(filename = path + f'surr_RFR_{app}_SpecificEnergy.joblib', mmap_mode='r')
            self.SEI_m = joblib.load(filename = path + f'surr_RFR_{app}_SEIGrouth.joblib', mmap_mode='r')
            self.T_m = joblib.load(filename = path + f'surr_RFR_{app}_TempIncrease.joblib', mmap_mode='r')
            self.V_m = joblib.load(filename = path + f'surr_SVR_{app}_Vcell.joblib', mmap_mode='r')

        elif app == '15_22':
            self.ES_m = joblib.load(filename = path + f'surr_RFR_{app}_SpecificEnergy.joblib', mmap_mode='r')
            self.SEI_m = joblib.load(filename = path + f'surr_RFR_{app}_SEIGrouth.joblib', mmap_mode='r')
            self.T_m = joblib.load(filename = path + f'surr_RFR_{app}_TempIncrease.joblib', mmap_mode='r')
            self.V_m = joblib.load(filename = path + f'surr_RFR_{app}_Vcell.joblib', mmap_mode='r')

        elif app == '3.7_3':
            self.ES_m = joblib.load(filename = path + f'surr_RFR_{app}_SpecificEnergy.joblib', mmap_mode='r')
            self.SEI_m = joblib.load(filename = path + f'surr_RFR_{app}_SEIGrouth.joblib', mmap_mode='r')
            self.T_m = joblib.load(filename = path + f'surr_RFR_{app}_TempIncrease.joblib', mmap_mode='r')
            self.V_m = joblib.load(filename = path + f'surr_RFR_{app}_Vcell.joblib', mmap_mode='r')
        
        else:
            raise ValueError('Desired configuration currently not supported')
        

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

        C = x["C"]
        la = x["la"]
        lp = x["lp"]
        lo = x["lo"]
        ln = x["ln"]
        lz = x["lz"]
        Lh = x["Lh"]
        Rp = x["Rp"]
        Rn = x["Rn"]
        Rcell = x["Rcell"]
        efp = x["efp"]
        efo = x["efo"]
        efn = x["efn"]
        mat = x["mat"]
        Np = x["Np"]
        Ns = x["Ns"]
        
        p_data, n_data, o_data, a_data, z_data, e_data = build_battery(mat,efp,efo,efn,Rp,Rn,la,lp,lo,ln,lz)
        A = area(Lh,la+lp+lo+ln+lz,Rcell)
        
        X = np.reshape([x[l] for l in self.vars], (1,-1))

        if X[0,13]=='LCO':
            X[0,13] = 0
        elif X[0,13]=='LFP':
            X[0,13] = 1
        else:
            raise ValueError('Material not defined')

        ES = self.ES_m.predict(X)[0]
        SEI = self.SEI_m.predict(X)[0]
        T = self.T_m.predict(X)[0]
        P = batteryPrice(a_data,p_data,o_data,n_data,z_data,e_data,Ns,Np,A)
        oFn = [-ES,SEI,T,P]

        V = self.V_m.predict(X)[0]
        cFn = ineqConstraintFunctions(self.Vpack,Ns,V,efp,efo,efn)

        out["F"] = oFn
        out["G"] = cFn