import jax.numpy as jnp
from jax import grad
from jax.config import config
config.update("jax_enable_x64", True)

from globalValues import F, R, gamma, t_plus
from settings import dxO, dxP, dxN #, delta_t
import coeffs as coeffs

class SeparatorEquation:
    def __init__(self, constants, p_constants, n_constants):
        self.rho = constants.rho
        self.Cp = constants.Cp
        self.ce_0 = constants.ce_0
        self.eps = constants.eps
        self.lam = constants.lam
        self.brugg = constants.brugg
        self.l = constants.l
        #self.delta_t = delta_t
        self.dx = self.l/dxO
        self.pe = p_constants
        self.ne = n_constants
        self.pe_hx = self.pe.l/dxP
        self.ne_hx = self.ne.l/dxN
        
    def Qohm(self, phien, phiep, ce_1, ce_3, ce_2, T):
        eps = self.eps
        brugg = self.brugg
        dx = self.dx
        kapeff = coeffs.electrolyteConductCoeff(eps,brugg,ce_2,T)
        ans = kapeff*( (phiep - phien)/(2*dx) )**2
        + (2*kapeff*R*T/F)*(1-t_plus)*( (jnp.log(ce_3) - jnp.log(ce_1))/(2*dx) )*( (phiep - phien)/(2*dx) )
        return ans
        
    def electrolyte_conc(self,ce_1, ce_2, ce_3, Tn, Tc, Tp, uold, delta_t):
        eps = self.eps; brugg = self.brugg; dx = self.dx

        umid_r = (ce_3+ce_2)/2
        umid_l = (ce_1+ce_2)/2
        Tmid_r = (Tp+Tc)/2
        Tmid_l = (Tn+Tc)/2
        Deff_r = coeffs.electrolyteDiffCoeff(eps,brugg,umid_r,Tmid_r)
        Deff_l = coeffs.electrolyteDiffCoeff(eps,brugg,umid_l,Tmid_l)
 
        ans = (ce_2-uold) -  (delta_t/eps)*( Deff_r*(ce_3 - ce_2)/dx - Deff_l*(ce_2 - ce_1)/dx )/dx 
        return ans.reshape()
    
    def Du_Dun(self,ce_1,ce_2,Tn,Tc,delta_t):
        eps = self.eps; brugg = self.brugg; dx = self.dx
        umid_l = (ce_1+ce_2)/2
        Tmid_l = (Tn+Tc)/2
    
        Deff_l = coeffs.electrolyteDiffCoeff(eps,brugg,umid_l,Tmid_l)
        Deff_l_Du = grad(coeffs.electrolyteDiffCoeff,(2))(eps,brugg,umid_l,Tmid_l)
        ans = (delta_t/(eps*dx**2))*(Deff_l_Du*(ce_2-ce_1) + Deff_l)
        return ans

    
    # boundary condition for positive electrode
    def bc_u_sep_p(self,u0_pe,u1_pe,T0_pe,T1_pe,\
             u0_sep,u1_sep,T0_sep,T1_sep):
        eps_p = self.pe.eps; eps_s = self.eps
        brugg_p = self.pe.brugg; brugg_s = self.brugg
        Deff_pe = coeffs.electrolyteDiffCoeff(eps_p,brugg_p,(u0_pe + u1_pe)/2,(T0_pe + T1_pe)/2)
        Deff_sep = coeffs.electrolyteDiffCoeff(eps_s,brugg_s,(u0_sep + u1_sep)/2,(T0_sep + T1_sep)/2)
        bc = -Deff_pe*(u1_pe - u0_pe)/self.pe_hx + Deff_sep*(u1_sep - u0_sep)/self.dx
        return bc.reshape()

    # boundary condition for negative electrode
    def bc_u_sep_n(self,u0_ne,u1_ne,T0_ne,T1_ne,\
                 u0_sep,u1_sep,T0_sep,T1_sep):
        eps_n = self.ne.eps; eps_s = self.eps
        brugg_n = self.ne.brugg; brugg_s = self.ne.brugg
        
        Deff_ne = coeffs.electrolyteDiffCoeff(eps_n,brugg_n,(u0_ne + u1_ne)/2,(T0_ne + T1_ne)/2)
        Deff_sep = coeffs.electrolyteDiffCoeff(eps_s,brugg_s,(u0_sep + u1_sep)/2,(T0_sep + T1_sep)/2)
        
        bc = -Deff_sep*(u1_sep - u0_sep)/self.dx + Deff_ne*(u1_ne - u0_ne)/self.ne_hx
        return bc.reshape()
    
    def electrolyte_poten(self,ce_1, ce_2, ce_3, phien, phiec, phiep, Tn, Tc, Tp):
    
        eps = self.eps; brugg = self.brugg
        dx = self.dx; 
        
        umid_r = (ce_3+ce_2)/2; umid_l = (ce_1+ce_2)/2
        Tmid_r = (Tp+Tc)/2; Tmid_l = (Tn+Tc)/2
        
        kapeff_r = coeffs.electrolyteConductCoeff(eps,brugg,umid_r,Tmid_r)
        kapeff_l = coeffs.electrolyteConductCoeff(eps,brugg,umid_l,Tmid_l)
        
        ans = - ( kapeff_r*(phiep - phiec)/dx - kapeff_l*(phiec - phien)/dx )/dx + gamma*( kapeff_r*Tmid_r*(jnp.log(ce_3) - jnp.log(ce_2))/dx  \
            - kapeff_l*Tmid_l*(jnp.log(ce_2) - jnp.log(ce_1))/dx )/dx
        return ans.reshape()
    
    def bc_phie_ps(self,phie0_p, phie1_p, phie0_s, phie1_s, u0_p, u1_p, u0_s, u1_s, T0_p, T1_p, T0_s, T1_s):
        kapeff_p = coeffs.electrolyteConductCoeff(self.pe.eps,self.pe.brugg,(u0_p + u1_p)/2,(T0_p + T1_p)/2)
        kapeff_s = coeffs.electrolyteConductCoeff(self.eps,self.brugg,(u0_s + u1_s)/2,(T0_s + T1_s)/2)
        bc = -kapeff_p*(phie1_p - phie0_p)/self.pe_hx + kapeff_s*(phie1_s - phie0_s)/self.dx
        return bc.reshape()
    
    def bc_phie_sn(self,phie0_n, phie1_n, phie0_s, phie1_s, u0_n, u1_n, u0_s, u1_s, T0_n, T1_n, T0_s, T1_s):
        
        kapeff_n = coeffs.electrolyteConductCoeff(self.ne.eps,self.ne.brugg,(u0_n + u1_n)/2,(T0_n + T1_n)/2)
        
        kapeff_s = coeffs.electrolyteConductCoeff(self.eps,self.brugg,(u0_s + u1_s)/2,(T0_s + T1_s)/2)
        bc = -kapeff_s*(phie1_s - phie0_s)/self.dx + kapeff_n*(phie1_n - phie0_n)/self.ne_hx
        return bc.reshape()

    def temperature(self, ce_1, ce_2, ce_3, phien, phiep, Tn, Tc, Tp, Told, delta_t):
        dx = self.dx
#        ans = (Tc - Told) -  (delta_t/(self.rho*self.Cp))*( self.lam*( Tn - 2*Tc + Tp)/dx**2 + \
#        self.Qohm( phien, phiep, ce_1, ce_3, ce_2, Tc) )
        ans = (Tc - Told) -  (delta_t/(self.rho*self.Cp))*( self.lam*( Tn - 2*Tc + Tp)/dx**2 + \
        self.Qohm( phien, phiep, ce_1, ce_3, ce_2, Tc) )
        return ans.reshape()
    
    def bc_temp_ps(self,T0_p, T1_p, T0_s, T1_s):
        bc = -self.pe.lam*(T1_p - T0_p)/self.pe_hx + self.lam*(T1_s - T0_s)/self.dx
        return bc.reshape()
    
    def bc_temp_sn(self,T0_s, T1_s, T0_n, T1_n):
        bc = -self.lam*(T1_s - T0_s)/self.dx + self.ne.lam*(T1_n - T0_n)/self.ne_hx
        return bc.reshape()

