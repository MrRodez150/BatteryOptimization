import numpy as np
import joblib

from fghFunctions import batteryPrice, ineqConstraintFunctions
from batteryBuilder import build_battery
from auxiliaryExp import area

class BatteryP2D():
    def __init__(self, V, I, **kwargs):
    
        self.vars = ['C', 'la', 'lp', 'lo', 'ln', 'lz', 'Lh', 'Rcell', 'Rp', 'Rn', 'efp', 'efo', 'efn', 'mat', 'Ns', 'Np']
        
        path = 'Experiments/surr/'
        app = f'{V}_{abs(I)}'

        self.Vpack = V
        self.Iapp = I

        if app == '48_80':
            self.ES_m = joblib.load(filename = path + f'surr_RFR_{app}_SpecificEnergy.joblib', mmap_mode='r')
            self.SEI_m = joblib.load(filename = path + f'surr_RFR_{app}_SEIGrouth.joblib', mmap_mode='r')
            self.T_m = joblib.load(filename = path + f'surr_RFR_{app}_TempIncrease.joblib', mmap_mode='r')
            self.V_m = joblib.load(filename = path + f'surr_SVR_{app}_Vcell.joblib', mmap_mode='r')

        elif app == '15_22':
            self.ES_m = joblib.load(filename = path + f'surr_DTR_{app}_SpecificEnergy.joblib', mmap_mode='r')
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
        
    def evaluate(self, x):

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

        return [oFn,cFn]