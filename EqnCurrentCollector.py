from jax.config import config
config.update("jax_enable_x64", True)

from globalValues import T_ref, h
from settings import delta_t, dxA, dxZ

class CurrentCollectorEquation:
    
    def __init__(self, constants, Iapp):
        self.lam = constants.lam
        self.rho = constants.rho
        self.Cp = constants.Cp
        self.sigeff = constants.sigma
        self.l = constants.l
        if constants.tipo == 'a':
            self.M = dxA
        elif constants.tipo == 'z':
            self.M = dxZ
        self.hx = self.l/self.M;
        self.delta_t=delta_t
        self.Iapp = Iapp
    
    def temperature(self,Tn,Tc,Tp, Told):
        hx = self.hx
#        ans = (Tc - Told) -  ( self.lam*( Tn - 2*Tc + Tp)/hx**2 + \
#        + Iapp**2/self.sigeff )*(delta_t/(self.rho*self.Cp))
        ans = (Tc - Told) - (self.lam*(Tn - 2*Tc + Tp)/hx**2 + self.Iapp**2/self.sigeff)*(self.delta_t/(self.rho*self.Cp))
        return ans.reshape()
    
    """ boundary condition """
    def bc_temp_a(self,T0,T1):
        bc = -self.lam*(T1-T0)/self.hx - h*(T_ref - (T1+T0)/2)
        return bc.reshape()
    
    def bc_temp_z(self,T0,T1):
        bc = -self.lam*(T1-T0)/self.hx - h*((T0+T1)/2 - T_ref)
        return bc.reshape()
