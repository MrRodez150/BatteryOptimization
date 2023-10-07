from IMIA import IMIA_run

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

IMIA_run(exp=150, Vpack=3.7, Iapp=-3, i_pop=40, f_mig=40, n_mig=1, f_eval=6_000, verbose=True, h_p=2)