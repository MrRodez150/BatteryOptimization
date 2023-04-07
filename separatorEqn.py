import jax.numpy as jnp
from config import div_x_elec, div_x_sep
from globalValues import F, R, t_plus, gamma
from coeffs import eCondCoeff, eDiffCoeff

class separatorEquations:

    def __init__(self, constants, p_constants, n_constants, delta_t):
        self.brugg = constants.brugg
        self.C = constants.C
        self.ce_0 = constants.c_e_init
        self.l = constants.l
        self.vareps = constants.vareps
        self.lam = constants.lam
        self.rho = constants.rho

        self.div_x = div_x_sep
        self.delta_x = self.l/self.div_x

        self.p_lam = p_constants.lam
        self.p_vareps = p_constants.vareps
        self.p_brugg = p_constants.brugg
        self.p_delta_x = p_constants.l/div_x_elec

        self.n_lam = n_constants.lam
        self.n_vareps = n_constants.vareps
        self.n_brugg = n_constants.brugg
        self.n_delta_x = n_constants.l/div_x_elec

        self.delta_t = delta_t

    #Electrolyte concentration

    def electConc(self, ce_0, ce, ce_1, T_0, T, T_1, ce_past):
        
        ce_mid0 = (ce_0+ce)/2 
        ce_mid1 = (ce+ce_1)/2
        T_mid0 = (T_0+T)/2
        T_mid1 = (T+T_1)/2

        Deff_0 = eDiffCoeff(self.vareps,self.brugg,ce_mid0,T_mid0)
        Deff_1 = eDiffCoeff(self.vareps,self.brugg,ce_mid1,T_mid1)
        
        ans = (ce-ce_past) - (self.delta_t/self.vareps)*( Deff_1*(ce_1 - ce)/self.delta_x - Deff_0*(ce - ce_0)/self.delta_x )/self.delta_x

        return ans.reshape()
    
    def eConc_po_bc(self,ce_p_0,ce_p_1,T_p_0,T_p_1,ce_o_0,ce_o_1,T_o_0,T_o_1):
        
        Deff_p = eDiffCoeff(self.p_vareps,self.p_brugg,(ce_p_0+ce_p_1)/2,(T_p_0+T_p_1)/2)
        Deff_o = eDiffCoeff(self.vareps,self.brugg,(ce_o_0+ce_o_1)/2,(T_o_0+T_o_1)/2)        
        
        ans = -Deff_p*(ce_p_1 - ce_p_0)/self.p_delta_x + Deff_o*(ce_o_1 - ce_o_0)/self.delta_x

        return ans.reshape()
    
    def eConc_on_bc(self,ce_n_0,ce_n_1,T_n_0,T_n_1,ce_o_0,ce_o_1,T_o_0,T_o_1):
        
        Deff_n = eDiffCoeff(self.n_vareps,self.n_brugg,(ce_n_0+ce_n_1)/2,(T_n_0+T_n_1)/2)
        Deff_o = eDiffCoeff(self.vareps,self.brugg,(ce_o_0+ce_o_1)/2,(T_o_0+T_o_1)/2)      
        
        ans = - Deff_o*(ce_o_1 - ce_o_0)/self.delta_x + Deff_n*(ce_n_1 - ce_n_0)/self.n_delta_x

        return ans.reshape()
    
    # Electrolyte potential:

    def electPoten(self, ce_0, ce, ce_1, phie_0, phie, phie_1, T_0, T, T_1):
            
        ce_mid0 = (ce_0+ce)/2 
        ce_mid1 = (ce+ce_1)/2
        T_mid0 = (T_0+T)/2
        T_mid1 = (T+T_1)/2
        
        kapeff_0 = eCondCoeff(self.vareps,self.brugg,ce_mid0,T_mid0)
        kapeff_1 = eCondCoeff(self.vareps,self.brugg,ce_mid1,T_mid1)
        
        ans = ( (kapeff_1*(phie_1 - phie)/self.delta_x - kapeff_0*(phie - phie_0)/self.delta_x)/self.delta_x
        - gamma*(kapeff_1*T_mid1*(jnp.log(ce_1) - jnp.log(ce))/self.delta_x - kapeff_0*T_mid0*(jnp.log(ce) - jnp.log(ce_0))/self.delta_x )/self.delta_x)
        
        return ans.reshape()
    
    def ePoten_po_bc(self,phie_p_0, phie_p_1, phie_o_0, phie_o_1, ce_p_0, ce_p_1, ce_o_0, ce_o_1, T_p_0, T_p_1, T_o_0, T_o_1):
        
        kapeff_p = eCondCoeff(self.p_vareps,self.p_brugg,(ce_p_0 + ce_p_1)/2,(T_p_0 + T_p_1)/2);
        kapeff_o = eCondCoeff(self.vareps,self.brugg,(ce_o_0 + ce_o_1)/2,(T_o_0 + T_o_1)/2);
        
        bc = -kapeff_p*(phie_p_1 - phie_p_0)/self.p_delta_x + kapeff_o*(phie_o_1 - phie_o_0)/self.delta_x
        
        return bc.reshape()

    def ePoten_on_bc(self,phie_n_0, phie_n_1, phie_o_0, phie_o_1, ce_n_0, ce_n_1, ce_o_0, ce_o_1, T_n_0, T_n_1, T_o_0, T_o_1):
        
        kapeff_n = eCondCoeff(self.n_vareps,self.n_brugg,(ce_n_0 + ce_n_1)/2,(T_n_0 + T_n_1)/2);
        kapeff_o = eCondCoeff(self.vareps,self.brugg,(ce_o_0 + ce_o_1)/2,(T_o_0 + T_o_1)/2);
        
        bc = -kapeff_o*(phie_o_1 - phie_o_0)/self.delta_x + kapeff_n*(phie_n_1 - phie_n_0)/self.n_delta_x
        
        return bc.reshape()
    
    # Temperature
    
    def temperature(self, ce_0, ce, ce_1, phie_0, phie_1, T_0, T, T_1, T_past):
        
        ans = ((T - T_past) - (self.delta_t/(self.rho*self.C))*(self.lam*(T_0 - 2*T + T_1)/self.delta_x**2
        + self.ohmHeat(phie_0, phie_1, ce_0, ce, ce_1, T)))
        return ans.reshape()
    
    def temp_po_bc(self, T_p_0, T_p_1, T_o_0, T_o_1):
        bc = -self.p_lam*(T_p_1 - T_p_0)/self.p_delta_x+ self.lam*(T_o_1 - T_o_0)/self.delta_x
        return bc.reshape()
    
    def temp_on_bc(self, T_o_0, T_o_1, T_n_0, T_n_1):
        bc = -self.lam*(T_o_1 - T_o_0)/self.delta_x + self.n_lam*(T_n_1 - T_n_0)/self.n_delta_x
        return bc.reshape()
    
    # Heat generation:

    def ohmHeat(self, phie_0, phie_1, ce_0, ce, ce_1, T):
        
        kapeff = eCondCoeff(self.vareps,self.brugg,ce,T)
        
        ans = (kapeff*( (phie_1 - phie_0)/(2*self.delta_x) )**2 +
        (2*kapeff*R*T/F)*(1-t_plus)*( (jnp.log(ce_1) - jnp.log(ce_0))/(2*self.delta_x) )*( (phie_1 - phie_0)/(2*self.delta_x) ))
        
        return ans