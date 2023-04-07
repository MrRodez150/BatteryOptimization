import numpy as np
import csv

from main_p2d import p2d_simulate

Vpack = 5
Ipack = 1

docInfo = []
saveRate = 10000
path = "Experiments/MPoints/"
name = 'MPoints_'

def saveDoc(info, count):
    keys = info[0].keys()
    with open(path + name + f'{count}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = keys)
        writer.writeheader()
        writer.writerows(info)
    return None


l_sp = 2
v_sp = 2
r_sp = 2
cell_sp = 3
N_step = 19

mat_sp = ['LCO','LFP']
C_sp = [0.2,0.5,1.0,2.0]
N_sp = list(range(1,100,N_step))
l_elec_sp = np.linspace(40e-6,250e-6,l_sp)
l_sep_sp = np.linspace(10e-6,100e-6,l_sp)
l_cc_sp = np.linspace(12e-6,30e-6,l_sp)
ve_sp = np.linspace(0.1,0.6,v_sp)
Rp_sp = np.linspace(0.5e-6,20e-6,r_sp)
Rn_sp = np.linspace(1e-6,30e-6,r_sp)
Rc_sp = np.linspace(4e-3,25e-3,cell_sp)
Lh_sp = np.linspace(40e-3,100e-3,cell_sp)

total_points = len(mat_sp) * len(N_sp)**2 * len(C_sp) * l_sp**5 * cell_sp**2 * r_sp**2 * v_sp**3
#print('Total points: ',total_points)

count = 0

#search_sp = [[C, la, lp, lo, ln, lz, Lh, Rcell, Rp, Rn, vep, veo, ven, mat, Ns, Np] ]

for C in C_sp:
    for la in l_cc_sp:
        for lp in l_elec_sp:
            for lo in l_sep_sp:
                for ln in l_elec_sp:
                    for lz in l_cc_sp:
                        for Lh in Lh_sp:
                            for Rcell in Rc_sp:
                                for Rp in Rp_sp:
                                    for Rn in Rn_sp:
                                        for vep in ve_sp:
                                            for veo in ve_sp:
                                                for ven in ve_sp:
                                                    for mat in mat_sp:
                                                        for Ns in N_sp:
                                                            for Np in N_sp:

                                                                vars = {
                                                                    "C": C,
                                                                    "la": la,
                                                                    "lp": lp,
                                                                    "lo": lo,
                                                                    "ln": ln,
                                                                    "lz": lz,
                                                                    "Lh": Lh,
                                                                    "Rc": Rcell,
                                                                    "Rp": Rp,
                                                                    "Rn": Rn,
                                                                    "ve_p": vep,
                                                                    "ve_o": veo,
                                                                    "ve_n": ven,
                                                                    "mat": mat,
                                                                    "Ns": Ns,
                                                                    "Np": Np,
                                                                }


                                                                oFn, cFn, sim_time, fail = ([1,2,3,4], [5,6], 7, 8)
                                                                #oFn, cFn, sim_time, fail = p2d_simulate(vars, Vpack, Ipack)

                                                                res = {
                                                                    "SpecificEnergy": oFn[0],
                                                                    "SEIGrouth": oFn[1],
                                                                    "TempIncrease": oFn[2],
                                                                    "Price": oFn[3],
                                                                    "UpperViolation": cFn[0],
                                                                    "LowerViolation": cFn[1],
                                                                    "SimTime": sim_time,
                                                                    "Failure": fail,
                                                                }

                                                                res.update(vars)

                                                                docInfo.append(res)

                                                                count += 1

                                                                if count%saveRate == 0 or count == total_points:
                                                                    
                                                                    saveDoc(docInfo,count)
                                                                    docInfo = []

print(count)
