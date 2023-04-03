import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit
from jax.lax import reshape
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import spsolve

from config import div_x_elec, div_x_sep, div_x_cc, div_r
from globalValues import T_ref, F, R, t_plus, gamma
from coeffs import sCondCoeff, sDiffCoeff, eCondCoeff, kRateCoeff, eDiffCoeff

class electrodeEquations:

    def __init__(self, constants, o_constants, acc_constants, zcc_constants, tipo, delta_t):
        self.tipo = tipo

        self.a = constants.a
        self.brugg = constants.brugg
        self.C = constants.C
        self.cmax = constants.c_s_max
        self.cavg = constants.c_s_avg
        self.Ds = constants.Ds
        self.ED = constants.ED
        self.Ek = constants.Ek
        self.k = constants.k
        self.l = constants.l
        self.Rp = constants.Rp
        self.vareps = constants.vareps
        self.lam = constants.lam
        self.rho = constants.rho
        self.sigma_eff = sCondCoeff(constants.sigma,self.vareps,constants.eps)

        self.div_x = div_x_elec
        self.delta_x = self.l/self.div_x
        self.div_r = div_r
        self.delta_r = self.Rp/self.div_r

        self.sep_lam = o_constants.lam
        self.sep_vareps = o_constants.vareps
        self.sep_brugg = o_constants.brugg
        self.sep_delta_x = o_constants.l/div_x_sep

        self.acc_lam = acc_constants.lam
        self.acc_delta_x = acc_constants.l/div_x_cc

        self.zcc_lam = zcc_constants.lam
        self.zcc_delta_x = zcc_constants.l/div_x_cc

        self.delta_t = delta_t

        rpts = self.Rp*jnp.linspace(0,self.div_r,self.div_r+1)/self.div_r
        rpts_mid = self.Rp*(jnp.linspace(1,self.div_r,self.div_r)-0.5)/self.div_r

        rt1 = delta_t*rpts[0:self.div_r]**2/(rpts_mid**2*self.delta_r**2)
        rt2 = delta_t*rpts[1:self.div_r+1]**2/(rpts_mid**2*self.delta_r**2)

        self.row = jnp.hstack([0,0,jnp.arange(1,self.div_r+1,1),jnp.arange(1,self.div_r+1,1),  jnp.arange(1,self.div_r+1,1),  self.div_r+1,self.div_r+1])
        self.col = jnp.hstack([0,1,jnp.arange(1,self.div_r+1,1),jnp.arange(1,self.div_r+1,1)-1,jnp.arange(1,self.div_r+1,1)+1,self.div_r,  self.div_r+1])
        self.dat = jnp.hstack([-1,1,1+self.Ds*(rt1+rt2),-self.Ds*rt1,-self.Ds*rt2,-1/self.delta_r,1/self.delta_r])

        preA = csr_matrix((self.dat, (self.row, self.col))) 
        vec = jnp.hstack([jnp.zeros(self.div_r+1), 1])

        self.A = kron(identity(self.div_x), preA)
        self.temp_sol = spsolve(preA,vec)
        self.gamma = (self.temp_sol[self.div_r] + self.temp_sol[self.div_r+1])/2

    # General boundary conditions

    def interSecc_bc(self, f0, f1, g0, g1):
        ans = (f0+f1)/2 - (g0+g1)/2
        return ans.reshape()

    def cNewmann_bc(self,f0,f1,const=0):
        ans = (f1 - f0)/self.delta_x - const
        return ans.reshape()

    def cDirichlet_bc(self,f0,f1,const=0):
        ans = (f0 + f1)/2 - const
        return ans.reshape()

    # Solid phase concentration:

    def sPhaseConc(self,cs_0,cs,cs_1, cs_past, factor0, factor1):
        ans = (cs-cs_past)  + self.Ds*( cs*(factor0 + factor1) - factor0*cs_0 - factor1*cs_1)
        return ans.reshape()

    def sPhaseConc_bc(self,cs_0,cs_1,j,T):
        Ds_eff = sDiffCoeff(self.Ds,self.ED,T)
        ans = (cs_1-cs_0)/self.delta_r + j/Ds_eff
        return ans.reshape()

    # Electrolyte concentration:

    @partial(jit, static_argnums=(0,))
    def electConc(self, ce_0, ce, ce_1, T_0, T, T_1, j, ce_past):
        
        ce_mid0 = (ce_0+ce)/2 
        ce_mid1 = (ce+ce_1)/2
        T_mid0 = (T_0+T)/2
        T_mid1 = (T+T_1)/2

        Deff_0 = eDiffCoeff(self.vareps,self.brugg,ce_mid0,T_mid0)
        Deff_1 = eDiffCoeff(self.vareps,self.brugg,ce_mid1,T_mid1)
        
        ans = (ce-ce_past) - (self.delta_t/self.vareps)*( ( Deff_1*(ce_1 - ce)/self.delta_x - Deff_0*(ce - ce_0)/self.delta_x )/self.delta_x + self.a*(1-t_plus)*j ) 

        ans = ans.reshape()

        return ans

    def eConc_po_bc(self,ce_p_0,ce_p_1,T_p_0,T_p_1,ce_o_0,ce_o_1,T_o_0,T_o_1):
        
        Deff_p = eDiffCoeff(self.vareps,self.brugg,(ce_p_0+ce_p_1)/2,(T_p_0+T_p_1)/2)
        Deff_o = eDiffCoeff(self.sep_vareps,self.sep_brugg,(ce_o_0+ce_o_1)/2,(T_o_0+T_o_1)/2)    

        ans = -Deff_p*(ce_p_1 - ce_p_0)/self.delta_x + Deff_o*(ce_o_1 - ce_o_0)/self.sep_delta_x

        return ans.reshape()
    
    def eConc_on_bc(self,ce_n_0,ce_n_1,T_n_0,T_n_1,ce_o_0,ce_o_1,T_o_0,T_o_1):
        
        Deff_n = eDiffCoeff(self.vareps,self.brugg,(ce_n_0+ce_n_1)/2,(T_n_0+T_n_1)/2)
        Deff_o = eDiffCoeff(self.sep_vareps,self.sep_brugg,(ce_o_0+ce_o_1)/2,(T_o_0+T_o_1)/2)      
        
        ans = - Deff_o*(ce_o_1 - ce_o_0)/self.sep_delta_x + Deff_n*(ce_n_1 - ce_n_0)/self.delta_x

        return ans.reshape()

    # Electrolyte potential:

    def electPoten(self, ce_0, ce, ce_1, phie_0, phie, phie_1, T_0, T, T_1, j):
            
        ce_mid0 = (ce_0+ce)/2 
        ce_mid1 = (ce+ce_1)/2
        T_mid0 = (T_0+T)/2
        T_mid1 = (T+T_1)/2
        
        kapeff_0 = eCondCoeff(self.vareps,self.brugg,ce_mid0,T_mid0)
        kapeff_1 = eCondCoeff(self.vareps,self.brugg,ce_mid1,T_mid1)
        
        ans = (self.a*F*j + (kapeff_1*(phie_1 - phie)/self.delta_x - kapeff_0*(phie - phie_0)/self.delta_x)/self.delta_x
        - gamma*(kapeff_1*T_mid1*(jnp.log(ce_1) - jnp.log(ce))/self.delta_x - kapeff_0*T_mid0*(jnp.log(ce) - jnp.log(ce_0))/self.delta_x )/self.delta_x)
        
        return ans.reshape()
    
    def ePoten_po_bc(self,phie_p_0, phie_p_1, phie_o_0, phie_o_1, ce_p_0, ce_p_1, ce_o_0, ce_o_1, T_p_0, T_p_1, T_o_0, T_o_1):
        
        kapeff_p = eCondCoeff(self.vareps,self.brugg,(ce_p_0 + ce_p_1)/2,(T_p_0 + T_p_1)/2);
        kapeff_o = eCondCoeff(self.sep_vareps,self.sep_brugg,(ce_o_0 + ce_o_1)/2,(T_o_0 + T_o_1)/2);
        
        bc = -kapeff_p*(phie_p_1 - phie_p_0)/self.delta_x + kapeff_o*(phie_o_1 - phie_o_0)/self.sep_delta_x
        
        return bc.reshape()

    def ePoten_on_bc(self,phie_n_0, phie_n_1, phie_o_0, phie_o_1, ce_n_0, ce_n_1, ce_o_0, ce_o_1, T_n_0, T_n_1, T_o_0, T_o_1):
        
        kapeff_n = eCondCoeff(self.vareps,self.brugg,(ce_n_0 + ce_n_1)/2,(T_n_0 + T_n_1)/2);
        kapeff_o = eCondCoeff(self.sep_vareps,self.sep_brugg,(ce_o_0 + ce_o_1)/2,(T_o_0 + T_o_1)/2);
        
        bc = -kapeff_o*(phie_o_1 - phie_o_0)/self.sep_delta_x + kapeff_n*(phie_n_1 - phie_n_0)/self.delta_x
        
        return bc.reshape()


    # Solid phase potential:

    def sPhasePoten(self,phis_0, phis, phis_1, j):

        ans = (phis_0 - 2*phis + phis_1) - (self.a*F*j*self.delta_x**2)/self.sigma_eff
        
        return ans.reshape()
    
    def sPhasePoten_bc(self,phis_0,phis_1,i):

        ans = (phis_1-phis_0) + self.delta_x*i/self.sigma_eff

        return ans.reshape()

    # Heat generation:

    def ohmHeat(self,phis_0, phis_1, phie_0, phie_1, ce_0, ce, ce_1, T):
        
        kapeff = eCondCoeff(self.vareps,self.brugg,ce,T)
        
        ans = (self.sigma_eff*((phis_1 - phis_0)/(2*self.delta_x))**2 + kapeff*( (phie_1 - phie_0)/(2*self.delta_x) )**2 +
        (2*kapeff*R*T/F)*(1-t_plus)*( (jnp.log(ce_1) - jnp.log(ce_0))/(2*self.delta_x) )*( (phie_1 - phie_0)/(2*self.delta_x) ))
        
        return ans

    def rxnHeat(self,j,eta):
        ans = F*self.a*j*eta
        return ans
    
    def revHeat(self,j,T,cs):
        ans = F*self.a*j*T*self.entropyChange(cs)
        return ans

    # Temperature:

    def temperature(self, ce_0, ce, ce_1, phie_0, phie_1, phis_0, phis_1, T_0, T, T_1, j, eta, cs_1, gamma_c, T_past):
        
        cs = cs_1 - gamma_c*j/sDiffCoeff(self.Ds, self.ED, T)
        ans = ((T - T_past) - (self.delta_t/(self.rho*self.C))*(self.lam*(T_0 - 2*T + T_1)/self.delta_x**2
        + self.ohmHeat(phis_0, phis_1, phie_0, phie_1, ce_0, ce, ce_1, T) + self.rxnHeat(j,eta) + self.revHeat(j,T,cs) ))

        return ans.reshape()
    
    def temp_ap_bc(self,T_a_0, T_a_1, T_p_0, T_p_1): 
        bc = -self.acc_lam*(T_a_1 - T_a_0)/self.acc_delta_x + self.lam*(T_p_1 - T_p_0)/self.delta_x
        return bc.reshape()
    
    def temp_po_bc(self,T_p_0, T_p_1, T_o_0, T_o_1):
        bc = -self.lam*(T_p_1 - T_p_0)/self.delta_x+ self.sep_lam*(T_o_1 - T_o_0)/self.sep_delta_x
        return bc.reshape()
    
    def temp_on_bc(self,T_o_0, T_o_1, T_n_0, T_n_1):
        bc = -self.sep_lam*(T_o_1 - T_o_0)/self.sep_delta_x + self.lam*(T_n_1 - T_n_0)/self.delta_x
        return bc.reshape()
    
    def temp_nz_bc(self,T_n_0, T_n_1, T_z_0, T_z_1):
        bc = -self.lam*(T_n_1 - T_n_0)/self.delta_x+ self.zcc_lam*(T_z_1 - T_z_0)/self.zcc_delta_x
        return bc.reshape()

    # Ionic flux:

    def ionicFlux(self,j,ce,T,eta,cs1,gamma_c):

        cs = cs1 - gamma_c*j/sDiffCoeff(self.Ds, self.ED, T)
        keff = kRateCoeff(self.k,self.Ek,T)
        var = ((0.5*F)/(R*T))*eta
        term2 = (jnp.exp(var)-jnp.exp(-var))/2
        
        ans = j - 2*keff*jnp.sqrt(ce*(self.cmax - cs)*cs)*term2

        ans.reshape()

        return ans

    # Overpotential:

    def openCircPot_ref(self,cs):

        theta = cs/self.cmax

        if (self.tipo == "p"):
            ans = (-4.656 + 88.669*(theta**2) - 401.119*(theta**4) + 342.909*(theta**6) -  462.471*(theta**8) + 433.434*(theta**10))/\
            (-1 + 18.933*(theta**2) - 79.532*(theta**4) + 37.311*(theta**6) - 73.083*(theta**8) + 95.96*(theta**10))    
        elif(self.tipo == "n"):
            ans = 0.7222 + 0.1387*theta + 0.029*theta**(0.5) - 0.0172/theta + 0.0019/(theta**1.5) + 0.2808*jnp.exp(0.9 - 15*theta) - 0.7984*jnp.exp(0.4465*theta - 0.4108)
        else:
            raise ValueError("Undefined type value for the electrode")
        return ans
    
    def entropyChange(self,cs):
        
        theta = cs/self.cmax
        
        if (self.tipo == "p"):
            ans = -0.001*( (0.199521039 - 0.92837822*theta + 1.364550689000003*theta**2 - 0.6115448939999998*theta**3)/\
            (1 - 5.661479886999997*theta + 11.47636191*theta**2 - 9.82431213599998*theta**3 + \
             3.046755063*theta**4))
        elif (self.tipo == "n"):
        # typo for + 38379.18127*theta**7 
            ans = 0.001*(0.005269056 + 3.299265709*theta - 91.79325798*theta**2 + \
             1004.911008*theta**3 - 5812.278127*theta**4 + \
             19329.7549*theta**5 - 37147.8947*theta**6 + 38379.18127*theta**7 - \
             16515.05308*theta**8)/(1 - 48.09287227*theta + 1017.234804*theta**2 - 10481.80419*theta**3 +\
             59431.3*theta**4 - 195881.6488*theta**5 + 374577.3152*theta**6 -\
             385821.1607*theta**7 + 165705.8597*theta**8)
        else:
            raise ValueError("Undefined type value for the electrode")
        
        return ans

    def openCircuitPoten(self,cs,T):
        ans = self.openCircPot_ref(cs) + (T - T_ref)*self.entropyChange(cs)
        return ans

    def overPotential(self, eta, phis, phie, T, j, cs1, gamma_c):
        cs = cs1 - gamma_c*j/sDiffCoeff(self.Ds, self.ED, T)
        ans = eta - phis + phie + self.openCircuitPoten(cs,T)
        
        return ans.reshape()
    
    def openCircPot_start(self):

        theta = self.cavg/self.cmax

        if (self.tipo == "p"):
            ans = (-4.656 + 88.669*(theta**2) - 401.119*(theta**4) + 342.909*(theta**6) -  462.471*(theta**8) + 433.434*(theta**10))/\
            (-1 + 18.933*(theta**2) - 79.532*(theta**4) + 37.311*(theta**6) - 73.083*(theta**8) + 95.96*(theta**10))    
        elif(self.tipo == "n"):
            ans = 0.7222 + 0.1387*theta + 0.029*theta**(0.5) - 0.0172/theta + 0.0019/(theta**1.5) + 0.2808*jnp.exp(0.9 - 15*theta) - 0.7984*jnp.exp(0.4465*theta - 0.4108)
        else:
            raise ValueError("Undefined type value for the electrode")
        return ans