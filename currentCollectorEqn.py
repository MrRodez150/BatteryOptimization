from config import div_x_cc
from globalValues import T_ref, h

class currentCollectorEquations:
    
    def __init__(self, constants, delta_t, Iapp):
        self.lam = constants.lam
        self.rho = constants.rho
        self.C = constants.C
        self.sigeff = constants.sigma
        self.l = constants.l

        self.delta_x = self.l/div_x_cc
        self.delta_t = delta_t

        self.I_cell = Iapp

    # Temperature
    
    def temperature(self,T_0,Tc,T_1, T_past):

        ans = (Tc - T_past) - (self.lam*(T_0 - 2*Tc + T_1)/self.delta_x**2 + self.I_cell**2/self.sigeff)*(self.delta_t/(self.rho*self.C))

        return ans.reshape()
    
    def temp_a_bc(self,T_0,T_1):
        bc = -self.lam*(T_1-T_0)/self.delta_x - h*(T_ref - (T_1+T_0)/2)
        return bc.reshape()
    
    def temp_z_bc(self,T_0,T_1):
        bc = -self.lam*(T_1-T_0)/self.delta_x - h*((T_0+T_1)/2 - T_ref)
        return bc.reshape()