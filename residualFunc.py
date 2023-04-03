import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


from config import div_x_elec,div_x_sep,div_x_cc,delta_t

class residualFunction():
    def __init__(self,peq,neq,oeq,aeq,zeq,Iapp):
        
        #Spatial divitions per section
        
        self.pn_div = div_x_elec
        self.o_div = div_x_sep
        self.az_div = div_x_cc

        self.delta_t = delta_t

        #Needed values

        self.Icell = Iapp

        #Section equations

        self.a_eq = aeq
        self.p_eq = peq
        self.n_eq = neq
        self.o_eq = oeq
        self.z_eq = zeq

        #Starting points and spaces needed per variable and section

        self.ce_p_0 = 0
        self.ce_o_0 = self.ce_p_0 + self.pn_div + 2
        self.ce_n_0 = self.ce_o_0 + self.o_div + 2

        self.j_p_0 = self.ce_n_0 + self.pn_div + 2
        self.j_n_0 = self.j_p_0 + self.pn_div

        self.eta_p_0 = self.j_n_0 + self.pn_div
        self.eta_n_0 = self.eta_p_0 + self.pn_div

        self.phis_p_0 = self.eta_n_0 + self.pn_div
        self.phis_n_0 = self.phis_p_0 + self.pn_div + 2

        self.phie_p_0 = self.phis_n_0 + self.pn_div + 2
        self.phie_o_0 = self.phie_p_0 + self.pn_div + 2
        self.phie_n_0 = self.phie_o_0 + self.o_div + 2

        self.t_a_0 = self.phie_n_0 + self.pn_div + 2
        self.t_p_0 = self.t_a_0 + self.az_div + 2
        self.t_o_0 = self.t_p_0 + self.pn_div + 2
        self.t_n_0 = self.t_o_0 + self.o_div + 2
        self.t_z_0 = self.t_n_0 + self.az_div + 2

        spc_p = 4*(self.pn_div + 2) + 2*(self.pn_div)
        spc_n = 4*(self.pn_div + 2) + 2*(self.pn_div)
        spc_o = 3*(self.o_div + 2)
        spc_a = self.az_div + 2
        spc_z = self.az_div + 2

        self.spaces = spc_a + spc_p + spc_o + spc_n + spc_z


    @partial(jit, static_argnums=(0,))
    def res_ce_p(self, val, ce_vec, T_vec, j_vec, ce_vec_old, ce_vec_o, T_vec_o):
        
        ind = self.ce_p_0
        pd = self.pn_div
        peq = self.p_eq

        val = val.at[ind].set(peq.cNewmann_bc(ce_vec[0],
                                              ce_vec[1],
                                              0))
           
        val = val.at[ind+1 : ind+pd+1].set(vmap(peq.electConc)(ce_vec[0:pd],
                                                               ce_vec[1:pd+1],
                                                               ce_vec[2:pd+2],
                                                               T_vec[0:pd],
                                                               T_vec[1:pd+1],
                                                               T_vec[2:pd+2],
                                                               j_vec[0:pd],
                                                               ce_vec_old[1:pd+1]))
           
        val = val.at[ind+pd+1].set(
                                   peq.eConc_po_bc(ce_vec[pd],
                                                   ce_vec[pd+1],
                                                   T_vec[pd],
                                                   T_vec[pd+1],
                                                   ce_vec_o[0],
                                                   ce_vec_o[1],
                                                   T_vec_o[0],
                                                   T_vec_o[1]))
        
        return val
        

    @partial(jit, static_argnums=(0,))
    def res_ce_o(self, val, ce_vec, T_vec, ce_vec_old, ce_vec_p, T_vec_p, ce_vec_n, T_vec_n):
        
        ind = self.ce_o_0
        od = self.o_div
        peq = self.p_eq
        pd = self.pn_div
        sepq = self.o_eq

        val = val.at[ind].set(
                                   peq.interSecc_bc(ce_vec[0],
                                                    ce_vec[1],
                                                    ce_vec_p[pd],
                                                    ce_vec_p[pd + 1]))
           
        val = val.at[ind+1 : ind+od+1].set(
                                   vmap(sepq.electConc)(ce_vec[0:od],
                                                        ce_vec[1:od+1],
                                                        ce_vec[2:od+2],
                                                        T_vec[0:od],
                                                        T_vec[1:od+1],
                                                        T_vec[2:od+2],
                                                        ce_vec_old[1:od+1]))
           
        val = val.at[ind+od+1].set(
                                   peq.interSecc_bc(ce_vec_n[0],
                                                    ce_vec_n[1],
                                                    ce_vec[od],
                                                    ce_vec[od+1]))
        
        return val
    
    
    @partial(jit, static_argnums=(0,))
    def res_ce_n(self, val, ce_vec, T_vec, j_vec, ce_vec_old, ce_vec_sep, T_vec_sep):
        
        ind = self.ce_n_0
        nd = self.pn_div
        neq = self.n_eq
        od = self.o_div

        val = val.at[ind].set(
                                   neq.eConc_on_bc(ce_vec[0],
                                                  ce_vec[1],
                                                  T_vec[0],
                                                  T_vec[1],
                                                  ce_vec_sep[od],
                                                  ce_vec_sep[od+1],
                                                  T_vec_sep[od],
                                                  T_vec_sep[od+1]))
        
        val = val.at[ind+1 : ind+1+nd].set(
                           vmap(neq.electConc)(ce_vec[0:nd],
                                               ce_vec[1:nd+1],
                                               ce_vec[2:nd+2],
                                               T_vec[0:nd],
                                               T_vec[1:nd+1],
                                               T_vec[2:nd+2],
                                               j_vec[0:nd],
                                               ce_vec_old[1:nd+1]))
        
        val = val.at[ind+1+nd].set(
                                   neq.cNewmann_bc(ce_vec[nd],
                                                   ce_vec[nd+1],
                                                   0))
        
        return val
        
    
    @partial(jit, static_argnums=(0,))        
    def res_T_a(self, val, T_vec, T_vec_old, T_vec_pe):
        
        ind = self.t_a_0
        ad = self.az_div
        zeq = self.a_eq
        peq = self.p_eq

        val = val.at[ind].set(
                                   zeq.temp_a_bc(T_vec[0],
                                                 T_vec[1]))
        
        val = val.at[ind+1 : ind+ad+1].set(
                                   vmap(zeq.temperature)(T_vec[0:ad],
                                                         T_vec[1:ad+1],
                                                         T_vec[2:ad+2],
                                                         T_vec_old[1:ad+1]))
        
        val = val.at[ind+ad+1].set(
                                   peq.interSecc_bc(T_vec[ad],
                                                    T_vec[ad+1],
                                                    T_vec_pe[0],
                                                    T_vec_pe[1]))
        
        return val
        
    
    @partial(jit, static_argnums=(0,))
    def res_T_o(self, val, T_vec, ce_vec, phie_vec, T_vec_old, T_vec_p, T_vec_n):
        
        ind = self.t_o_0
        peq = self.p_eq
        oeq = self.o_eq
        pd = self.pn_div
        od = self.o_div

        val = val.at[ind].set(
                                   peq.interSecc_bc(T_vec_p[pd],
                                                    T_vec_p[pd+1],
                                                    T_vec[0],
                                                    T_vec[1]))
        
        val = val.at[ind+1 : ind+1+od].set(
                                   vmap(oeq.temperature)(ce_vec[0:od],
                                                         ce_vec[1:od+1],
                                                         ce_vec[2:od+2],
                                                         phie_vec[0:od],
                                                         phie_vec[2:od+2],
                                                         T_vec[0:od],
                                                         T_vec[1:od+1],
                                                         T_vec[2:od+2],
                                                         T_vec_old[1:od+1]))
        
        val = val.at[ind+1+od].set(
                                   peq.interSecc_bc(T_vec[od],
                                                    T_vec[od+1],
                                                    T_vec_n[0],
                                                    T_vec_n[1]))
        
        return val

    
    @partial(jit, static_argnums=(0,))
    def res_T_z(self, val, T_vec, T_vec_old, T_vec_n):
        
        ind = self.t_z_0
        nd = self.pn_div
        zd = self.az_div
        zeq = self.z_eq
        neq = self.n_eq
        
        val = val.at[ind].set(
                                   neq.interSecc_bc(T_vec_n[nd],
                                                    T_vec_n[nd+1],
                                                    T_vec[0],
                                                    T_vec[1]))
        
        val = val.at[ind+1 : ind+1+zd].set(
                                   vmap(zeq.temperature)(T_vec[0:zd],
                                                         T_vec[1:zd+1], 
                                                         T_vec[2:zd+2],
                                                         T_vec_old[1:zd+1]))
        
        val = val.at[ind+zd+1].set(
                                   zeq.temp_z_bc(T_vec[zd],
                                                 T_vec[zd+1]))
        
        return val
        
    
    @partial(jit, static_argnums=(0,))
    def res_phie_p(self, val, ce_vec, phie_vec, T_vec, j_vec, ce_vec_o, phie_vec_o, T_vec_o):
        
        ind = self.phie_p_0
        peq = self.p_eq
        pd = self.pn_div
        
        val = val.at[ind].set(
                                   peq.cNewmann_bc(phie_vec[0],
                                                   phie_vec[1]),
                                                   0)
        
        val = val.at[ind+1 : ind+pd+1].set(
                                   vmap(peq.electPoten)(ce_vec[0:pd],
                                                        ce_vec[1:pd+1],
                                                        ce_vec[2:pd+2],
                                                        phie_vec[0:pd],
                                                        phie_vec[1:pd+1],
                                                        phie_vec[2:pd+2],
                                                        T_vec[0:pd],
                                                        T_vec[1:pd+1],
                                                        T_vec[2:pd+2],
                                                        j_vec[0:pd]))
        
           
        val = val.at[ind+pd+1].set(
                                   peq.ePoten_po_bc(phie_vec[pd],
                                                    phie_vec[pd+1],
                                                    phie_vec_o[0],
                                                    phie_vec_o[1],
                                                    ce_vec[pd],
                                                    ce_vec[pd+1],
                                                    ce_vec_o[0],
                                                    ce_vec_o[1],
                                                    T_vec[pd],
                                                    T_vec[pd+1],
                                                    T_vec_o[0],
                                                    T_vec_o[1]))
        return val
        
    
    @partial(jit, static_argnums=(0,))
    def res_phie_o(self, val, ce_vec, phie_vec, T_vec, phie_vec_p, phie_vec_n):
        
        ind = self.phie_o_0
        peq = self.p_eq
        sepq= self.o_eq
        od = self.o_div
        neq = self.n_eq
        pd = self.pn_div

        val = val.at[ind].set(
                                   peq.interSecc_bc(phie_vec_p[pd],
                                                    phie_vec_p[pd+1],
                                                    phie_vec[0],
                                                    phie_vec[1]))
        
        val = val.at[ind+1 : ind+od+1].set(
                                   vmap(sepq.electPoten)(ce_vec[0:od],
                                                         ce_vec[1:od+1],
                                                         ce_vec[2:od+2],
                                                         phie_vec[0:od],
                                                         phie_vec[1:od+1],
                                                         phie_vec[2:od+2],
                                                         T_vec[0:od],
                                                         T_vec[1:od+1],
                                                         T_vec[2:od+2]))
        
        val = val.at[ind+od+1].set(
                                   neq.interSecc_bc(phie_vec_n[0],
                                                    phie_vec_n[1],
                                                    phie_vec[od],
                                                    phie_vec[od+1]))
        
        return val
        
    
    @partial(jit, static_argnums=(0,))
    def res_phie_n(self, val, ce_vec, phie_vec, T_vec, j_vec, ce_vec_o, phie_vec_o, T_vec_o):
        
        ind = self.phie_n_0
        neq = self.n_eq
        nd = self.pn_div
        od = self.o_div
        
        val = val.at[ind].set(
                                   neq.ePoten_on_bc(phie_vec[0],
                                                    phie_vec[1],
                                                    phie_vec_o[od],
                                                    phie_vec_o[od+1],
                                                    ce_vec[0],
                                                    ce_vec[1],
                                                    ce_vec_o[od],
                                                    ce_vec_o[od+1],
                                                    T_vec[0],
                                                    T_vec[1],
                                                    T_vec_o[od],
                                                    T_vec_o[od+1]))
        
        val = val.at[ind+1 : ind+nd+1].set(
                                   vmap(neq.electPoten)(ce_vec[0:nd],
                                                        ce_vec[1:nd+1],
                                                        ce_vec[2:nd+2],
                                                        phie_vec[0:nd],
                                                        phie_vec[1:nd+1],
                                                        phie_vec[2:nd+2],
                                                        T_vec[0:nd],
                                                        T_vec[1:nd+1],
                                                        T_vec[2:nd+2],
                                                        j_vec[0:nd]))
        
        val = val.at[ind+nd+1].set(
                                   neq.cDirichlet_bc(phie_vec[nd],
                                                     phie_vec[nd+1],
                                                     0))
        return val    
    
    
    @partial(jit, static_argnums=(0,))
    def res_phis(self, val, phis_p, j_vec_p, phis_n, j_vec_n):
        
        indp = self.phis_p_0
        peq= self.p_eq
        pd = self.pn_div
        
        val = val.at[indp].set(
                                   peq.sPhasePoten_bc(phis_p[0],
                                                     phis_p[1],
                                                     self.Icell))

        val = val.at[indp+1 : indp+pd+1].set(
                                   vmap(peq.sPhasePoten)(phis_p[0:pd],
                                                         phis_p[1:pd+1],
                                                         phis_p[2:pd+2],
                                                         j_vec_p[0:pd]))
        
        val = val.at[indp+pd+1].set(
                                   peq.sPhasePoten_bc(phis_p[pd],
                                                      phis_p[pd+1],
                                                      0))
        
        indn  = self.phis_n_0
        neq = self.n_eq
        nd = self.pn_div

        val = val.at[indn].set(
                                   neq.sPhasePoten_bc(phis_n[0],
                                                      phis_n[1],
                                                      0))
        
        val = val.at[indn+1 : indn+nd+1].set(
                                   vmap(neq.sPhasePoten)(phis_n[0:nd],
                                                         phis_n[1:nd+1],
                                                         phis_n[2:nd+2],
                                                         j_vec_n[0:nd]))
        
        val = val.at[indn+nd+1].set(
                                   neq.sPhasePoten_bc(phis_n[nd],
                                                      phis_n[nd+1],
                                                      self.Icell))
        
        return val
    
    
    @partial(jit, static_argnums=(0,))
    def res_T_p(self, val, T_vec, ce_vec, phie_vec, phis_vec, j_vec, eta_vec, cs_p1, gamma_p, T_vec_old, T_vec_a, T_vec_o):
        
        ind = self.t_p_0
        ad = self.az_div
        peq = self.p_eq
        pd = self.pn_div

        val = val.at[ind].set(
                                   peq.temp_ap_bc(T_vec_a[ad],
                                                  T_vec_a[ad+1],
                                                  T_vec[0],
                                                  T_vec[1]))
        
        val = val.at[ind+1 : ind+pd+1].set(
                                   vmap(peq.temperature)(ce_vec[0:pd],
                                                         ce_vec[1:pd+1],
                                                         ce_vec[2:pd+2],
                                                         phie_vec[0:pd],
                                                         phie_vec[2:pd+2],
                                                         phis_vec[0:pd],
                                                         phis_vec[2:pd+2],
                                                         T_vec[0:pd],
                                                         T_vec[1:pd+1],
                                                         T_vec[2:pd+2],
                                                         j_vec[0:pd],
                                                         eta_vec[0:pd],
                                                         cs_p1,
                                                         gamma_p,
                                                         T_vec_old[1:pd+1]))
        
        val = val.at[ind+pd+1].set(
                                   peq.temp_po_bc(T_vec[pd],
                                                  T_vec[pd+1],
                                                  T_vec_o[0],
                                                  T_vec_o[1]))
        return val
    
    
    @partial(jit, static_argnums=(0,))    
    def res_T_n(self, val, T_vec, ce_vec, phie_vec, phis_vec, j_vec, eta_vec, cs_n1, gamma_n, T_vec_old, T_vec_z, T_vec_o):
        
        ind = self.t_n_0
        od = self.o_div
        neq = self.n_eq
        nd = self.pn_div
        
        val = val.at[ind].set(
                                   neq.temp_on_bc(T_vec_o[od],
                                                  T_vec_o[od+1],
                                                  T_vec[0],
                                                  T_vec[1]))
        
        val = val.at[ind+1 : ind+1+nd].set(
                                   vmap(neq.temperature)(ce_vec[0:nd],
                                                         ce_vec[1:nd+1],
                                                         ce_vec[2:nd+2],
                                                         phie_vec[0:nd],
                                                         phie_vec[2:nd+2],
                                                         phis_vec[0:nd],
                                                         phis_vec[2:nd+2],
                                                         T_vec[0:nd],
                                                         T_vec[1:nd+1],
                                                         T_vec[2:nd+2],
                                                         j_vec[0:nd],
                                                         eta_vec[0:nd],
                                                         cs_n1,
                                                         gamma_n,
                                                         T_vec_old[1:nd+1]))
        
        val = val.at[ind+1+nd].set(
                                   neq.temp_nz_bc(T_vec[nd],
                                                  T_vec[nd+1],
                                                  T_vec_z[0],
                                                  T_vec_z[1]))
        return val
        
    
    @partial(jit, static_argnums=(0,))
    def res_j(self, val, j_vec_p, ce_vec_p, T_vec_p, eta_p, cs_p1, gamma_p, j_vec_n, ce_vec_n, T_vec_n, eta_n, cs_n1, gamma_n):

        indp = self.j_p_0
        peq= self.p_eq
        pd = self.pn_div

        val = val.at[indp : indp+pd].set(
                                   vmap(peq.ionicFlux)(j_vec_p,
                                                       ce_vec_p[1:pd+1],
                                                       T_vec_p[1:pd+1],
                                                       eta_p,
                                                       cs_p1,
                                                       gamma_p))
        
        indn  = self.j_n_0
        neq = self.n_eq
        nd = self.pn_div

        val = val.at[indn : indn+nd].set(
                                   vmap(neq.ionicFlux)(j_vec_n,
                                                       ce_vec_n[1:nd+1],
                                                       T_vec_n[1:nd+1],
                                                       eta_n,
                                                       cs_n1,
                                                       gamma_n))

        return val
    
    
    @partial(jax.jit, static_argnums=(0,))
    def res_eta(self, val, eta_p, phis_p, phie_p, T_vec_p, j_vec_p, cs_p1, gamma_p, eta_n, phis_n, phie_n, T_vec_n, j_vec_n, cs_n1, gamma_n):
        
        indp = self.eta_p_0
        pd = self.pn_div
        peq=self.p_eq
        
        val = val.at[indp : indp+pd].set(
                                   vmap(peq.overPotential)(eta_p,
                                                           phis_p[1:pd+1],
                                                           phie_p[1:pd+1],
                                                           T_vec_p[1:pd+1],
                                                           j_vec_p,
                                                           cs_p1,
                                                           gamma_p))
        
        indn = self.eta_n_0
        nd = self.pn_div
        neq = self.n_eq

        val = val.at[indn : indn+nd].set(
                                   vmap(neq.overPotential)(eta_n,
                                                           phis_n[1:nd+1],
                                                           phie_n[1:nd+1],
                                                           T_vec_n[1:nd+1],
                                                           j_vec_n,
                                                           cs_n1,
                                                           gamma_n))
        
        return val
