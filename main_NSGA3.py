from NSGA3 import NSGA3_run

# while True:   
#     app = input("Application required?  EV/DR/CP: ")

#     if app == 'EV':
#         Vpack = 48
#         Iapp = -80
#         break
#     elif app == 'DR':
#         Vpack = 15
#         Iapp = -22
#         break
#     elif app == 'CP':
#         Vpack = 3.7
#         Iapp = -3
#         break
#     else:
#         print('Invalid Application, try again!')

# exp = int(input("Experiment number: "))

NSGA3_run(exp=150, Vpack=48, Ipack=-80, popul=200, func_eval=30_000, verbose=True, h_p=2, pth="Experiments/NSGA3/")