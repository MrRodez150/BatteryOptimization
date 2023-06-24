import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from functools import partial

import coeffs as coeffs
from globalValues import F, R, gamma, t_plus, T_ref
from settings import dxO, dxA, dxZ, drP, drN, dxP, dxN #, delta_t

class ElectrodeEquation:   
    
    def __init__(self, constants, s_constants, a_constants, z_constants):
        self.cavg = constants.cavg
        self.cmax = constants.cmax
        self.electrode_type = constants.tipo
        self.rho = constants.rho
        self.Cp = constants.Cp
        self.ce_0 = constants.ce_0
        self.sigma = constants.sigma
        self.epsf = constants.epsf
        self.eps = constants.eps
        self.lam = constants.lam
        self.brugg = constants.brugg
        self.a = constants.a
        self.Rp = constants.Rp
        self.k = constants.k
        self.l = constants.l
        self.Ds = constants.Ds
        self.sigma = constants.sigma
        self.Ek = constants.Ek
        self.ED = constants.ED
        self.Ds = constants.Ds

        #self.delta_t=delta_t

        self.sep = s_constants
        self.sep_dx = self.sep.l/dxO
        
        self.acc = a_constants
        self.acc_dx = self.acc.l/dxA
        
        self.zcc = z_constants
        self.zcc_dx = self.zcc.l/dxZ
        
        if self.electrode_type == 'p': 
            self.N = drP
            self.M = dxP
        elif self.electrode_type == 'n':
            self.N = drN
            self.M = dxN
        else:
            raise ValueError('Material type for electrode not defined')
            
        self.dx = self.l/self.M
        self.dr = self.Rp/self.N

        self.sigeff  = self.sigma*(1-self.eps - self.epsf)
        

   
    """ Equations for electrolyte concentration """

    @partial(jax.jit, static_argnums=(0,))
    def electrolyte_conc(self, ce0, ce1, ce2, T0, T1, T2, j, ce_old, delta_t):
        eps = self.eps
        brugg = self.brugg
        dx = self.dx
        a = self.a
        ce_mr = (ce2+ce1)/2
        ce_ml = (ce0+ce1)/2
        T_mr = (T2+T1)/2
        T_ml = (T0+T1)/2
        Deff_r = coeffs.electrolyteDiffCoeff(eps,brugg,ce_mr,T_mr)
        Deff_l = coeffs.electrolyteDiffCoeff(eps,brugg,ce_ml,T_ml)
        
        ans = (ce1-ce_old) - (delta_t/eps)*( ( Deff_r*(ce2 - ce1)/dx - Deff_l*(ce1 - ce0)/dx )/dx + a*(1-t_plus)*j ) 
    
        return ans.reshape()
    
    def bc_zero_neumann(self, c0, c1):
        bc =  c1 - c0
        return bc.reshape()

    def bc_const_dirichlet(self,c0, c1, constant):
        bc =  (c1 + c0)/2 - constant
        return bc.reshape()
    
    # boundary condition for positive electrode
    def bc_ce_po(self,ce0_p,ce1_p,T0_pe,T1_pe,ce0_o,ce1_o,T0_o,T1_o):
        
        eps_p = self.eps
        eps_o = self.sep.eps
        brugg_p = self.brugg
        brugg_o = self.sep.brugg
        
        
        Deff_pe = (coeffs.electrolyteDiffCoeff(eps_p,brugg_p,ce0_p,T0_pe) + coeffs.electrolyteDiffCoeff(eps_p,brugg_p,ce1_p,T1_pe))/2
        Deff_o = (coeffs.electrolyteDiffCoeff(eps_o,brugg_o,ce0_o,T0_o) + coeffs.electrolyteDiffCoeff(eps_o,brugg_o,ce1_o,T1_o))/2          
        
        bc = -Deff_pe*(ce1_p - ce0_p)/self.dx + Deff_o*(ce1_o - ce0_o)/self.sep_dx

        return bc.reshape()
    
    def bc_inter_cont(self, ce0, ce1, ce2, ce3):
        ans = (ce0+ce1)/2 - (ce2+ce3)/2
        return ans.reshape()

    # boundary condition for negative electrode
    def bc_ce_on(self,u0_ne,u1_ne,T0_ne,T1_ne,ce0_o,ce1_o,T0_o,T1_o):
        
        eps_n = self.eps
        eps_o = self.sep.eps
        brugg_n = self.brugg
        brugg_o = self.sep.brugg
        Deff_n = coeffs.electrolyteDiffCoeff(eps_n,brugg_n,(u0_ne + u1_ne)/2,(T0_ne + T1_ne)/2)
        Deff_o = coeffs.electrolyteDiffCoeff(eps_o,brugg_o,(ce0_o + ce1_o)/2,(T0_o + T1_o)/2)

        bc = -Deff_o*(ce1_o - ce0_o)/self.sep_dx + Deff_n*(u1_ne - u0_ne)/self.dx

        return bc.reshape()
    
    """ Equations for electrolyte potential """
    
    def electrolyte_poten(self,ce0, ce1, ce2, phie0, phie1, phie2, T0, T1, T2, j):
    
        eps = self.eps
        brugg = self.brugg
        dx = self.dx
        a = self.a
        
        ce_mr = (ce2+ce1)/2
        ce_ml = (ce0+ce1)/2
        T_mr = (T2+T1)/2
        T_ml = (T0+T1)/2
        
        kapeff_r = coeffs.electrolyteConductCoeff(eps,brugg,ce_mr,T_mr)
        kapeff_l = coeffs.electrolyteConductCoeff(eps,brugg,ce_ml,T_ml)
        
        ans = (a*F*j + (kapeff_r*(phie2 - phie1)/dx - kapeff_l*(phie1 - phie0)/dx)/dx 
               - gamma*(kapeff_r*T_mr*(jnp.log(ce2) - jnp.log(ce1))/dx 
                        - kapeff_l*T_ml*(jnp.log(ce1) - jnp.log(ce0))/dx )/dx)
        
        return ans.reshape()
    
    def bc_zero_dirichlet(self,phie0, phie1):
        ans= (phie0 + phie1)/2
        return ans.reshape()

    def bc_phie_p(self,phie0_p, phie1_p, phie0_s, phie1_s, u0_p, u1_p, u0_s, u1_s, T0_p, T1_p, T0_s, T1_s):
        
        kapeff_p = coeffs.electrolyteConductCoeff(self.eps,self.brugg,(u0_p + u1_p)/2,(T0_p + T1_p)/2)
        kapeff_s = coeffs.electrolyteConductCoeff(self.sep.eps,self.sep.brugg,(u0_s + u1_s)/2,(T0_s + T1_s)/2)
        
        bc = -kapeff_p*(phie1_p - phie0_p)/self.dx + kapeff_s*(phie1_s - phie0_s)/self.sep_dx
        return bc.reshape()
    
    def bc_phie_n(self,phie0_n, phie1_n, phie0_s, phie1_s, u0_n, u1_n, u0_s, u1_s, T0_n, T1_n, T0_s, T1_s):
        kapeff_n = coeffs.electrolyteConductCoeff(self.eps, self.brugg, (u0_n + u1_n)/2,(T0_n + T1_n)/2)
        kapeff_s = coeffs.electrolyteConductCoeff(self.sep.eps, self.sep.brugg, (u0_s + u1_s)/2,(T0_s + T1_s)/2)
        
        bc = -kapeff_s*(phie1_s - phie0_s)/self.sep_dx + kapeff_n*(phie1_n - phie0_n)/self.dx
        return bc.reshape()
    
    """ Equations for solid potential"""

    def solid_poten(self, phis0, phis1, phis2, j):
        dx = self.dx
        a = self.a
        sigeff = self.sigma*(1-self.eps-self.epsf)
        ans = ( phis0 - 2*phis1 + phis2) - (a*F*j*dx**2)/sigeff
        return ans.reshape()
    
    def bc_phis(self,phis0, phis1, source):
        sigeff = self.sigma*(1-self.eps-self.epsf)
        bc = ( phis1 - phis0 ) + self.dx*(source)/sigeff
        return bc.reshape()
    

    """ Equations for temperature """
    
    def temperature(self,ce0, ce1, ce2, phie0, phie2, phis0, phis2, T0, T1, T2,j,eta, cs_1, gamma_c, Told, delta_t):
        dx = self.dx
        cs = cs_1 - gamma_c*j/coeffs.solidDiffCoeff(self.Ds, self.ED, T1)
        ans = (T1 - Told) -  (delta_t/(self.rho*self.Cp))*( self.lam*( T0 - 2*T1 + T2)/dx**2 + \
        self.Qohm(phis0, phis2, phie0, phie2, ce0, ce2, ce1, T1) + self.Qrxn(j,eta) + self.Qrev(j,T1,cs) )
        return ans.reshape()
    
    # boundary conditions
    def bc_temp_ap(self,T0_acc, T1_acc, T0_pe, T1_pe): 
        bc = -self.acc.lam*(T1_acc - T0_acc)/self.acc_dx + self.lam*(T1_pe - T0_pe)/self.dx
        return bc.reshape()
    
    def bc_temp_po(self,T0_p, T1_p, T0_s, T1_s):
        bc = -self.lam*(T1_p - T0_p)/self.dx+ self.sep.lam*(T1_s - T0_s)/self.sep_dx
        return bc.reshape()
    
    def bc_temp_on(self,T0_s, T1_s, T0_n, T1_n):
        bc = -self.sep.lam*(T1_s - T0_s)/self.sep_dx + self.lam*(T1_n - T0_n)/self.dx
        return bc.reshape()
    
    def bc_temp_nz(self,T0_ne, T1_ne, T0_zcc, T1_zcc):
        bc = -self.lam*(T1_ne - T0_ne)/self.dx+ self.zcc.lam*(T1_zcc - T0_zcc)/self.zcc_dx
        return bc.reshape()
    
    """ Equations for ionic flux """
    
    def ionic_flux(self,j,u,T,eta,cs1, gamma_c):
        cs = cs1 - gamma_c*j/coeffs.solidDiffCoeff(self.Ds, self.ED,T )
        keff = self.k*jnp.exp( (-self.Ek/R)*((1/T) - (1/T_ref)))
        var = ((0.5*F)/(R*T))*eta
        term2 = (jnp.exp(var)-jnp.exp(-var))/2
        ans = j - 2*keff*jnp.sqrt(u*(self.cmax - cs)*cs)*term2
        return ans.reshape()
    
    
    """ Equations for over potential """

    def over_poten(self,eta, phis,phie, T, j,cs1, gamma_c):
        cs = cs1 - gamma_c*j/coeffs.solidDiffCoeff(self.Ds, self.ED, T)
        ans = eta - phis + phie + self.open_circuit_poten(cs,T)
        return ans.reshape()
    
    def open_circuit_poten(self,cs,T):
        Uref = self.open_circ_poten_ref(cs)
        ans = Uref + (T - T_ref)*self.entropy_change(cs)
        return ans
    
    def open_circ_poten_ref(self,cs):
        theta = cs/self.cmax
        if (self.electrode_type == 'p'):
            ans = (-4.656 + 88.669*(theta**2) - 401.119*(theta**4) + 342.909*(theta**6) -  462.471*(theta**8) + 433.434*(theta**10))/\
            (-1 + 18.933*(theta**2) - 79.532*(theta**4) + 37.311*(theta**6) - 73.083*(theta**8) + 95.96*(theta**10))    
        elif (self.electrode_type == 'n'):
            ans = 0.7222 + 0.1387*theta + 0.029*theta**(0.5) - 0.0172/theta + 0.0019/(theta**1.5) + 0.2808*jnp.exp(0.9 - 15*theta) - 0.7984*jnp.exp(0.4465*theta - 0.4108)
        else:
            raise ValueError('Type for electrode material not defined')
        return ans
    
    def entropy_change(self,cs):
        theta = cs/self.cmax
        if (self.electrode_type == 'p'):
            ans = -0.001*( (0.199521039 - 0.92837822*theta + 1.364550689000003*theta**2 - 0.6115448939999998*theta**3)/\
            (1 - 5.661479886999997*theta + 11.47636191*theta**2 - 9.82431213599998*theta**3 + \
             3.046755063*theta**4))
        elif (self.electrode_type == 'n'):
            ans = 0.001*(0.005269056 + 3.299265709*theta - 91.79325798*theta**2 + \
             1004.911008*theta**3 - 5812.278127*theta**4 + \
             19329.7549*theta**5 - 37147.8947*theta**6 + 38379.18127*theta**7 - \
             16515.05308*theta**8)/(1 - 48.09287227*theta + 1017.234804*theta**2 - 10481.80419*theta**3 +\
             59431.3*theta**4 - 195881.6488*theta**5 + 374577.3152*theta**6 -\
             385821.1607*theta**7 + 165705.8597*theta**8)
        else:
            raise ValueError('Type for electrode material not defined')
        
        return ans
    

    """ Heat source terms """
    
    def Qohm(self,phis0, phis2, phie0, phie2, ce0, ce2, ce1, T):
        eps = self.eps
        brugg = self.brugg
        dx = self.dx
        sigeff = self.sigma*(1-self.eps-self.epsf)
        kapeff = coeffs.electrolyteConductCoeff(eps,brugg,ce1,T)
        
        ans = sigeff*( (phis2 - phis0)/(2*dx) )**2 + kapeff*( (phie2 - phie0)/(2*dx) )**2 + \
        (2*kapeff*R*T/F)*(1-t_plus)*( (jnp.log(ce2) - jnp.log(ce0))/(2*dx) )*( (phie2 - phie0)/(2*dx) )
        
        return ans

    def Qrxn(self,j,eta):
        ans = F*self.a*j*eta
        return ans
    
    def Qrev(self,j,T,cs):
        ans = F*self.a*j*T*self.entropy_change(cs)
        return ans
