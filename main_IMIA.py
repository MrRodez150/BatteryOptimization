from IMIA import IMIA_run

app = input('App (EV/DR/CP): ') 
mode = input('Mode (RIB/NDV): ') 
for e in range(30):

    if app == 'EV':
        V = 48
        I = -80
    elif app == 'DR':
        V = 15
        I = -22
    elif app == 'CP':
        V = 3.7
        I = -3
    else:
        raise ValueError('Application not defined')

    IMIA_run(exp=e, Vpack=V, Iapp=I, 
            i_pop=40, f_mig=40, n_mig=1, f_eval=4_000, ref_mode=mode, 
            verbose=True, h_p=0, pth='Experiments/')
    
    #Inicialización de cambios
    