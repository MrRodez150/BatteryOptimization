import random
import numpy as np
import csv
from smt.sampling_methods import LHS
from main_p2d import p2d_simulate

#Aplication requirements
while True:   
    app = input("Application required?  EV/DR/CP: ")

    if app == 'EV':
        Vpack = 48
        Iapp = -80
        Ns_R = range(11,16)
        break
    elif app == 'DR':
        Vpack = 15
        Iapp = -22
        Ns_R = range(4,5)
        break
    elif app == 'CP':
        Vpack = 3.7
        Iapp = -3
        Ns_R = range(1,101)
        break
    else:
        print('Invalid Application, try again!')

#Experiment configuration
start_file = int(input("Starting file number: "))
end_file = int(input("End file number: "))
saveRate = int(input("Samples per file: "))

num_samples = saveRate * (end_file - start_file + 1)

file = start_file

path = "Experiments/LandscapeMac/"
name = f'HLS_{Vpack}_{abs(Iapp)}_'

def saveDoc(info, count):
    keys = info[0].keys()
    with open(path + name + f'{count}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = keys)
        writer.writeheader()
        writer.writerows(info)
    return None

docInfo = []

limits = np.array([[0.2, 4.0],
                   [12e-6, 30e-6],
                   [40e-6, 250e-6],
                   [10e-6, 100e-6],
                   [40e-6, 150e-6],
                   [12e-6, 30e-6],
                   [40e-3, 100e-3],
                   [0.2e-6, 20e-6],
                   [0.5e-6, 50e-6],
                   [4e-3, 25e-3],
                   [0.01, 0.6],
                   [0.01, 0.6],
                   [0.01, 0.6]])
sampling = LHS(xlimits=limits)

samples = sampling(num_samples)

count = 0
failed = 0

for x in samples:
    vars = {
        "C": x[0],
        "la": x[1],
        "lp": x[2],
        "lo": x[3],
        "ln": x[4],
        "lz": x[5],
        "Lh": x[6],
        "Rp": x[7],
        "Rn": x[8],
        "Rcell": x[9],
        "efp": x[10],
        "efo": x[11],
        "efn": x[12],
        "mat": random.choice(['LCO','LFP']),
        "Ns": random.choice(Ns_R),
        "Np": random.choice(range(1,101)),
    }
    
    oFn, cFn, sim_time, fail = p2d_simulate(vars, Vpack, Iapp)

    if fail[0] != '':
        failed += 1

    res = {
        "SpecificEnergy": oFn[0],
        "SEIGrouth": oFn[1],
        "TempIncrease": oFn[2],
        "Price": oFn[3],
        "UpperViolation": cFn[0],
        "LowerViolation": cFn[1],
        "VolFracViolation": cFn[2],
        "TotalTime": sim_time[0],
        "CompilationTime": sim_time[1],
        "SolvingTime": sim_time[2],
        "Failure": fail[0],
        "TimeSteps": fail[1],
        "LastVoltage": fail[2],
    }

    res.update(vars)

    docInfo.append(res)

    count += 1

    if count%saveRate == 0 or count == num_samples:
        
        saveDoc(docInfo,file)

        file += 1
        
        docInfo = []

    print(f'{app}\t{count}/{num_samples-count}|{failed}\t{file}')

