import numpy as np

from fghFunctions import ineqConstraintFunctions

def p2d_simulate(x, Vpack, Ipack):

    #Decision variables

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

    Icell = Ipack * C / Np

    #Construction of the battery
    objFun = list(np.array([-(la+lz*efp),lp/Lh+efo,lo*Lh*efn,ln/efn])*abs(Icell)*Rcell*Rp*Rn)

    conFun = ineqConstraintFunctions(Vpack,Ns,np.random.rand(10),efp,efo,efn)

    fail = [np.random.choice(['','Nan'],p=[0.9,0.1]),0,0]

    return objFun, conFun, C, fail