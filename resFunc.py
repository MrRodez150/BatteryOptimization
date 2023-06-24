from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap
from jax.config import config
config.update("jax_enable_x64", True)

from settings import dxA, dxP, dxO, dxN, dxZ
from unpack import ce_p0, ce_o0, ce_n0, j_p0, j_n0, eta_p0, eta_n0, phis_p0, phis_n0, phie_p0, phie_o0, phie_n0, t_a0, t_p0, t_o0, t_n0, t_z0


class ResidualFunction():
    def __init__(self, eqn_p, eqn_n, eqn_o, eqn_a, eqn_z, Iapp):

        self.eqn_p = eqn_p
        self.eqn_n = eqn_n
        self.eqn_o = eqn_o
        self.eqn_a = eqn_a
        self.eqn_z = eqn_z

        self.Iapp = Iapp

        Ntot_pe =  4*(dxP + 2) + 2*(dxP)
        Ntot_ne =  4*(dxN + 2) + 2*(dxN)
        Ntot_sep =  3*(dxO + 2)
        Ntot_acc = dxA + 2
        Ntot_zcc = dxZ + 2
        Ntot = Ntot_pe + Ntot_ne + Ntot_sep + Ntot_acc + Ntot_zcc
        self.Ntot = Ntot
       
       
        
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_ce_p(self, val, ce, T, j, ce_old, ce_o, T_o, delta_t):
        
        eqn_p = self.eqn_p
        
        val = val.at[ce_p0].set(
            eqn_p.bc_zero_neumann(ce[0], ce[1]))
           
        val = val.at[ce_p0+1 : ce_p0+dxP+1].set(
            vmap(eqn_p.electrolyte_conc)(ce[0:dxP], ce[1:dxP+1], ce[2:dxP+2],
                                         T[0:dxP], T[1:dxP+1], T[2:dxP+2],
                                         j[0:dxP],
                                         ce_old[1:dxP+1],
                                         delta_t*jnp.ones(dxP)))
           
        val = val.at[ce_p0+dxP+1].set(
            eqn_p.bc_ce_po(ce[dxP], ce[dxP+1],
                             T[dxP],T[dxP+1],
                             ce_o[0],ce_o[1],
                             T_o[0],T_o[1]))
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_ce_o(self,val, ce, T, ce_old, ce_p, T_p, ce_n, T_n, delta_t):
        
        eqn_p = self.eqn_p
        eqn_o = self.eqn_o

        val = val.at[ce_o0].set(
            eqn_p.bc_inter_cont(ce[0], ce[1],
                                ce_p[dxP], ce_p[dxP+1]))
           
        val = val.at[ce_o0+1 : ce_o0+dxO+1].set(vmap(
            eqn_o.electrolyte_conc)(ce[0:dxO], ce[1:dxO+1], ce[2:dxO+2],
                                    T[0:dxO], T[1:dxO+1], T[2:dxO+2],
                                    ce_old[1:dxO+1],
                                    delta_t*jnp.ones(dxO)))
           
        val = val.at[ce_o0+dxO+1].set(
            eqn_p.bc_inter_cont(ce_n[0], ce_n[1],
                                ce[dxO], ce[dxO+1]))
        
        return val
    
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_ce_n(self,val, ce, T, j, ce_old, ce_o, T_o, delta_t):

        eqn_n = self.eqn_n

        val = val.at[ce_n0].set(
            eqn_n.bc_ce_on(ce[0], ce[1],
                             T[0], T[1],
                             ce_o[dxO], ce_o[dxO+1],
                             T_o[dxO], T_o[dxO+1]))
        
        val = val.at[ce_n0+1 : ce_n0+dxN+1].set(vmap(
            eqn_n.electrolyte_conc)(ce[0:dxN], ce[1:dxN+1], ce[2:dxN+2],
                                    T[0:dxN], T[1:dxN+1], T[2:dxN+2],
                                    j[0:dxN],
                                    ce_old[1:dxN+1],
                                    delta_t*jnp.ones(dxN)))
        
        val = val.at[ce_n0+dxN+1].set(
            eqn_n.bc_zero_neumann(ce[dxN],ce[dxN+1]))
           
        return val
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))        
    def res_T_a(self,val, T, T_old, T_p, delta_t):

        eqn_a = self.eqn_a
        eqn_p = self.eqn_p

        val = val.at[t_a0].set(
            eqn_a.bc_temp_a(T[0], T[1]))
        
        val = val.at[t_a0+1 : t_a0+dxA+1].set(vmap(
            eqn_a.temperature)(T[0:dxA], T[1:dxA+1], T[2:dxA+2],
                               T_old[1:dxA+1],
                               delta_t*jnp.ones(dxA)))
        
        val = val.at[t_a0+dxA+1].set(
            eqn_p.bc_inter_cont(T[dxA], T[dxA+1],
                                T_p[0], T_p[1]))
        
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_o(self,val, T, ce, phie, T_old, T_p, T_n, delta_t):

        eqn_p = self.eqn_p
        eqn_o = self.eqn_o

        val = val.at[t_o0].set(
            eqn_p.bc_inter_cont(T_p[dxP], T_p[dxP+1],
                                T[0], T[1]))
        
        val = val.at[t_o0+1: t_o0+dxO+1].set(vmap(
            eqn_o.temperature)(ce[0:dxO], ce[1:dxO+1], ce[2:dxO+2],
                               phie[0:dxO], phie[2:dxO+2],
                               T[0:dxO], T[1:dxO+1], T[2:dxO+2],
                               T_old[1:dxO+1],
                               delta_t*jnp.ones(dxO)))
        
        val = val.at[t_o0+dxO+1].set(
            eqn_p.bc_inter_cont(T[dxO], T[dxO+1],
                                T_n[0], T_n[1]))
        
        return val

    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_z(self,val, T, T_old, T_n, delta_t):

        eqn_z = self.eqn_z
        eqn_n = self.eqn_n

        val = val.at[t_z0].set(
            eqn_n.bc_inter_cont(T_n[dxN], T_n[dxN+1],
                                T[0], T[1]))
        
        val = val.at[t_z0+1 : t_z0+dxZ+1].set(vmap(
            eqn_z.temperature)(T[0:dxZ], T[1:dxZ+1], T[2:dxZ+2],
                               T_old[1:dxZ+1],
                               delta_t*jnp.ones(dxZ)))
        
        val = val.at[t_z0+dxZ+1].set(
            eqn_z.bc_temp_z(T[dxZ], T[dxZ+1]))
        
        return val
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_p(self,val, ce, phie, T, j, ce_o, phie_o, T_o):
 
        eqn_p = self.eqn_p

        val = val.at[phie_p0].set(
            eqn_p.bc_zero_neumann(phie[0], phie[1]))
        
        val = val.at[phie_p0+1 : phie_p0+dxP+1].set(vmap(
            eqn_p.electrolyte_poten)(ce[0:dxP], ce[1:dxP+1], ce[2:dxP+2],
                                     phie[0:dxP], phie[1:dxP+1], phie[2:dxP+2],
                                     T[0:dxP], T[1:dxP+1], T[2:dxP+2],
                                     j[0:dxP]))
        
           
        val = val.at[phie_p0+dxP+1].set(
            eqn_p.bc_phie_p(phie[dxP], phie[dxP+1], 
                            phie_o[0], phie_o[1],
                            ce[dxP], ce[dxP+1],
                            ce_o[0], ce_o[1],
                            T[dxP], T[dxP+1],
                            T_o[0], T_o[1]))
        
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_o(self,val, ce, phie, T, phie_p, phie_n):

        eqn_p = self.eqn_p
        eqn_o= self.eqn_o
        eqn_n = self.eqn_n

        val = val.at[phie_o0].set(
            eqn_p.bc_inter_cont(phie_p[dxP], phie_p[dxP+1],
                                phie[0], phie[1]))
        
        val = val.at[phie_o0+1: phie_o0+dxO+1].set(vmap(
            eqn_o.electrolyte_poten)(ce[0:dxO], ce[1:dxO+1], ce[2:dxO+2],
                                     phie[0:dxO], phie[1:dxO+1], phie[2:dxO+2],
                                     T[0:dxO], T[1:dxO+1], T[2:dxO+2]))
        
        val = val.at[phie_o0 + dxO+1].set(
            eqn_n.bc_inter_cont(phie_n[0], phie_n[1],
                                phie[dxO], phie[dxO+1]))
        
        return val
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_n(self,val, ce, phie, T, j, ce_o, phie_o, T_o):
        
        eqn_n = self.eqn_n
        
        val = val.at[phie_n0].set(
            eqn_n.bc_phie_n(phie[0], phie[1],
                            phie_o[dxO], phie_o[dxO+1],
                            ce[0], ce[1],
                            ce_o[dxO], ce_o[dxO+1],
                            T[0], T[1],
                            T_o[dxO], T_o[dxO+1]))
        
        val = val.at[phie_n0+1: phie_n0+dxN+1].set(vmap(
            eqn_n.electrolyte_poten)(ce[0:dxN], ce[1:dxN+1], ce[2:dxN+2],
                                     phie[0:dxN], phie[1:dxN+1], phie[2:dxN+2],
                                     T[0:dxN], T[1:dxN+1], T[2:dxN+2],
                                     j[0:dxN]))
        
        val = val.at[phie_n0+dxN+1].set(
            eqn_n.bc_zero_dirichlet(phie[dxN], phie[dxN+1]))
        
        return val
        

    
    #    @jax.jit  
    @partial(jax.jit, static_argnums=(0,))
    def res_phis(self,val, phis_p, j_p, phis_n, j_n):

        eqn_p= self.eqn_p
        eqn_n = self.eqn_n

        val = val.at[phis_p0].set(
            eqn_p.bc_phis(phis_p[0], phis_p[1],
                          self.Iapp))
        
        val = val.at[phis_p0+1 : phis_p0+dxP+1].set(vmap(
            eqn_p.solid_poten)(phis_p[0:dxP], phis_p[1:dxP+1], phis_p[2:dxP+2],
                               j_p[0:dxP]))
        
        val = val.at[phis_p0+dxP+1].set(
            eqn_p.bc_phis(phis_p[dxP], phis_p[dxP+1],
                          0))
        
        
        val = val.at[phis_n0].set(
            eqn_n.bc_phis(phis_n[0], phis_n[1],
                          0))
        
        val = val.at[phis_n0+1 : phis_n0+dxN+1].set(vmap(
            eqn_n.solid_poten)(phis_n[0:dxN], phis_n[1:dxN+1], phis_n[2:dxN+2],
                               j_n[0:dxN]))
        
        val = val.at[phis_n0 + dxN+1].set(
            eqn_n.bc_phis(phis_n[dxN], phis_n[dxN+1],
                          self.Iapp))
        
        return val
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_p(self,val, T, ce, phie, phis, j, eta, cs_pe1, gamma_p, T_old, T_a, T_o, delta_t):

        eqn_p = self.eqn_p

        val = val.at[t_p0].set(
            eqn_p.bc_temp_ap(T_a[dxA], T_a[dxA+1],
                             T[0], T[1]))
        
        val = val.at[t_p0+1 : t_p0+dxP+1].set(vmap(
            eqn_p.temperature)(ce[0:dxP], ce[1:dxP+1], ce[2:dxP+2],
                               phie[0:dxP], phie[2:dxP+2],
                               phis[0:dxP], phis[2:dxP+2],
                               T[0:dxP], T[1:dxP+1], T[2:dxP+2],
                               j[0:dxP],
                               eta[0:dxP],
                               cs_pe1,
                               gamma_p,
                               T_old[1:dxP+1],
                               delta_t*jnp.ones(dxP)))
    
        val = val.at[t_p0+dxP+1].set(
            eqn_p.bc_temp_po(T[dxP], T[dxP+1],
                             T_o[0], T_o[1]))
        
        return val
    
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))    
    def res_T_n(self,val, T, ce, phie, phis, j, eta, cs_ne1, gamma_n, T_old, T_z, T_o, delta_t):

        eqn_n = self.eqn_n

        val = val.at[t_n0].set(
            eqn_n.bc_temp_on(T_o[dxO], T_o[dxO+1],
                             T[0], T[1]))
        
        val = val.at[t_n0+1 : t_n0+dxN+1].set(vmap(
            eqn_n.temperature)(ce[0:dxN], ce[1:dxN+1], ce[2:dxN+2],
                               phie[0:dxN], phie[2:dxN+2],
                               phis[0:dxN], phis[2:dxN+2],
                               T[0:dxN], T[1:dxN+1], T[2:dxN+2],
                               j[0:dxN],
                               eta[0:dxN],
                               cs_ne1,
                               gamma_n,
                               T_old[1:dxN+1],
                               delta_t*jnp.ones(dxN)))
        
        val = val.at[t_n0+dxN+1].set(
            eqn_n.bc_temp_nz(T[dxN], T[dxN+1],
                             T_z[0], T_z[1]))
        
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_j(self,val, j_p, ce_p, T_p, eta_pe, cs_pe1, gamma_p, j_n, ce_n, T_n, eta_ne, cs_ne1, gamma_n):

        eqn_p= self.eqn_p
        eqn_n = self.eqn_n

        val = val.at[j_p0 : j_p0+dxP].set(vmap(
            eqn_p.ionic_flux)(j_p,
                              ce_p[1:dxP+1],
                              T_p[1:dxP+1],
                              eta_pe,
                              cs_pe1,
                              gamma_p))
        
        val = val.at[j_n0 : j_n0+dxN].set(vmap(
            eqn_n.ionic_flux)(j_n,
                              ce_n[1:dxN+1],
                              T_n[1:dxN+1],
                              eta_ne,
                              cs_ne1,
                              gamma_n))
    
        return val
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_eta(self,val, eta_pe, phis_p, phie_p, T_p, j_p, cs_pe1, gamma_p, eta_ne, phis_n, phie_n, T_n, j_n, cs_ne1, gamma_n):

        eqn_p=self.eqn_p
        eqn_n=self.eqn_n

        val = val.at[eta_p0 : eta_p0+dxP].set(vmap(
            eqn_p.over_poten)(eta_pe,
                              phis_p[1:dxP+1],
                              phie_p[1:dxP+1],
                              T_p[1:dxP+1],
                              j_p,
                              cs_pe1,
                              gamma_p))
        
        val = val.at[eta_n0 : eta_n0+dxN].set(vmap(
            eqn_n.over_poten)(eta_ne,
                              phis_n[1:dxN+1],
                              phie_n[1:dxN+1],
                              T_n[1:dxN+1],
                              j_n,
                              cs_ne1,
                              gamma_n))
        return val


