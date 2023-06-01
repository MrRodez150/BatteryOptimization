from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap
from jax.config import config
config.update("jax_enable_x64", True)

from settings import dxA, dxP, dxO, dxN, dxZ, drP, drN


class ResidualFunctionFast():
    def __init__(self, peq, neq, sepq, accq, zccq, Iapp):
        self.dxP = dxP
        self.drP = drP
        self.dxN = dxN 
        self.drN = drN 
        self.dxO = dxO 
        self.dxA = dxA 
        self.dxZ = dxZ

        self.peq = peq
        self.neq = neq
        self.sepq = sepq
        self.accq = accq
        self.zccq = zccq

        self.Iapp = Iapp
        self.up0 =  0
        self.usep0 = self.up0 + dxP + 2
        self.un0 = self.usep0  + dxO+2
        
        self.jp0 = self.un0 + dxN + 2
        self.jn0 = self.jp0 + dxP
        
        self.etap0 = self.jn0 + dxN 
        self.etan0 = self.etap0 + dxP
        
        self.phisp0 = self.etan0 + dxN
        self.phisn0 = self.phisp0 + dxP + 2
        
        
        self.phiep0 = self.phisn0 + dxN +2
        self.phiesep0 = self.phiep0 + dxP + 2
        self.phien0 = self.phiesep0 + dxO + 2
        
        
        self.ta0 = self.phien0 + dxN + 2
        self.tp0 = self.ta0 + dxA+2
        self.tsep0 = self.tp0 + dxP+2
        self.tn0 = self.tsep0+ dxO+2
        self.tz0 = self.tn0 + dxN+2
        Ntot_pe =  4*(dxP + 2) + 2*(dxP)
        Ntot_ne =  4*(dxN + 2) + 2*(dxN)
        Ntot_sep =  3*(dxO + 2)
        Ntot_acc = dxA + 2
        Ntot_zcc = dxZ + 2
        Ntot = Ntot_pe + Ntot_ne + Ntot_sep + Ntot_acc + Ntot_zcc
        self.Ntot = Ntot
       
       
        
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_u_pe(self,val, uvec, Tvec, jvec, uvec_old, uvec_sep,Tvec_sep):
        up0 = self.up0
        dxP = self.dxP
        peq = self.peq
        val = val.at[up0].set(peq.bc_zero_neumann(uvec[0], uvec[1]))
           
        val = val.at[up0 + 1: up0 + dxP + 1].set(
                           vmap(peq.electrolyte_conc)(uvec[0:dxP], uvec[1:dxP+1], uvec[2:dxP+2],
                               Tvec[0:dxP], Tvec[1:dxP+1], Tvec[2:dxP+2], jvec[0:dxP],uvec_old[1:dxP+1]))
           
        val = val.at[up0 + dxP + 1].set(peq.bc_u_sep_p(uvec[dxP],uvec[dxP+1],Tvec[dxP],Tvec[dxP+1],\
         uvec_sep[0],uvec_sep[1],Tvec_sep[0],Tvec_sep[1]))
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_u_sep(self,val, uvec, Tvec, uvec_old, uvec_pe, Tvec_pe, uvec_ne, Tvec_ne):
        usep0 = self.usep0
        dxO = self.dxO
        peq = self.peq
        dxP = self.dxP
        sepq = self.sepq
        val = val.at[usep0].set(peq.bc_inter_cont(uvec[0], uvec[1], uvec_pe[dxP], uvec_pe[dxP + 1]) )
           
        val = val.at[usep0+ 1: usep0 + 1 + dxO].set(vmap(sepq.electrolyte_conc)(uvec[0:dxO], uvec[1:dxO+1], uvec[2:dxO+2], Tvec[0:dxO], Tvec[1:dxO+1], Tvec[2:dxO+2], uvec_old[1:dxO+1]))
           
        val = val.at[usep0+ dxO + 1].set(
        peq.bc_inter_cont(uvec_ne[0], uvec_ne[1], uvec[dxO], uvec[dxO+1]))
        return val
    
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_u_ne(self,val, uvec, Tvec, jvec, uvec_old, uvec_sep, Tvec_sep):
        un0= self.un0
        dxN = self.dxN
        neq = self.neq
        dxO = self.dxO
        val = val.at[un0].set(neq.bc_u_sep_n(uvec[0], uvec[1], Tvec[0], Tvec[1], uvec_sep[dxO], uvec_sep[dxO+1], Tvec_sep[dxO], Tvec_sep[dxO+1]))
        
        val = val.at[un0+ 1: un0+ 1 + dxN].set(
                           vmap(neq.electrolyte_conc)(uvec[0:dxN], uvec[1:dxN+1], uvec[2:dxN+2],
                               Tvec[0:dxN], Tvec[1:dxN+1], Tvec[2:dxN+2], jvec[0:dxN],uvec_old[1:dxN+1]))
        
        val = val.at[un0 + 1 + dxN].set(neq.bc_zero_neumann(uvec[dxN],uvec[dxN+1]))    
        return val
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))        
    def res_T_acc(self,val, Tvec, Tvec_old, Tvec_pe):
        ta0 = self.ta0
        dxA = self.dxA
        accq = self.accq
        peq = self.peq
        val = val.at[ta0].set(accq.bc_temp_a(Tvec[0], Tvec[1]))
        val = val.at[ta0 + 1: ta0 + dxA + 1].set(vmap(accq.temperature)(Tvec[0:dxA], Tvec[1:dxA+1], Tvec[2:dxA+2], Tvec_old[1:dxA+1]))
        val = val.at[ta0 + dxA + 1].set(peq.bc_inter_cont(Tvec[dxA], Tvec[dxA+1], Tvec_pe[0], Tvec_pe[1]))
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_sep(self,val, Tvec, uvec, phievec, Tvec_old, Tvec_pe, Tvec_ne):
        tsep0 = self.tsep0
        peq = self.peq
        sepq = self.sepq
        dxP = self.dxP
        dxO = self.dxO
        val = val.at[tsep0].set(peq.bc_inter_cont(Tvec_pe[dxP], Tvec_pe[dxP+1], Tvec[0], Tvec[1]))
        
        val = val.at[tsep0 + 1: tsep0 + 1 + dxO].set(
        vmap(sepq.temperature)(uvec[0:dxO], uvec[1:dxO+1], uvec[2:dxO+2], phievec[0:dxO], phievec[2:dxO+2],
        Tvec[0:dxO], Tvec[1:dxO+1], Tvec[2:dxO+2], Tvec_old[1:dxO+1]))
        
        val = val.at[ tsep0 + 1 + dxO].set(peq.bc_inter_cont(Tvec[dxO], Tvec[dxO+1], Tvec_ne[0], Tvec_ne[1]))
        return val

    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_zcc(self,val, Tvec, Tvec_old, Tvec_ne):
        tz0 = self.tz0
        dxN = self.dxN
        dxZ = self.dxZ
        zccq = self.zccq
        neq = self.neq
        val = val.at[tz0].set(neq.bc_inter_cont(Tvec_ne[dxN], Tvec_ne[dxN+1], Tvec[0], Tvec[1]))
        val = val.at[tz0+1:tz0 + 1 + dxZ].set(vmap(zccq.temperature)(Tvec[0:dxZ], Tvec[1:dxZ+1], Tvec[2:dxZ+2], Tvec_old[1:dxZ+1]))
        val = val.at[tz0+dxZ+1].set(zccq.bc_temp_z(Tvec[dxZ], Tvec[dxZ+1]))
        return val
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_pe(self,val, uvec, phievec, Tvec, jvec, uvec_sep, phievec_sep, Tvec_sep):
        phiep0 = self.phiep0
        peq = self.peq
        dxP = self.dxP
        val = val.at[phiep0].set(peq.bc_zero_neumann(phievec[0], phievec[1]))
        
        val = val.at[phiep0 + 1: phiep0 + dxP + 1].set(vmap(peq.electrolyte_poten)(uvec[0:dxP], uvec[1:dxP+1], uvec[2:dxP+2],
        phievec[0:dxP],phievec[1:dxP+1], phievec[2:dxP+2], Tvec[0:dxP], Tvec[1:dxP+1], Tvec[2:dxP+2], jvec[0:dxP]))
        
           
        val = val.at[phiep0 + dxP+1].set(peq.bc_phie_p(phievec[dxP], phievec[dxP+1],  phievec_sep[0], phievec_sep[1], \
                           uvec[dxP], uvec[dxP+1], uvec_sep[0], uvec_sep[1],\
                           Tvec[dxP], Tvec[dxP+1], Tvec_sep[0], Tvec_sep[1]))
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_sep(self,val, uvec, phievec, Tvec, phievec_pe, phievec_ne):
        phiesep0 = self.phiesep0
        peq = self.peq
        sepq= self.sepq
        dxO = self.dxO
        neq = self.neq
        dxP = self.dxP
        val = val.at[phiesep0].set(peq.bc_inter_cont(phievec_pe[dxP], phievec_pe[dxP+1], phievec[0], phievec[1]))
        
        val = val.at[phiesep0 + 1: phiesep0 + dxO + 1].set(vmap(sepq.electrolyte_poten)(uvec[0:dxO], uvec[1:dxO+1], uvec[2:dxO+2], phievec[0:dxO], phievec[1:dxO+1], phievec[2:dxO+2], Tvec[0:dxO], Tvec[1:dxO+1], Tvec[2:dxO+2]))
        
        val = val.at[phiesep0 + dxO+1].set(neq.bc_inter_cont(phievec_ne[0], phievec_ne[1], phievec[dxO], phievec[dxO+1]))
        return val
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_ne(self,val, uvec, phievec, Tvec, jvec, uvec_sep, phievec_sep, Tvec_sep):
        phien0 = self.phien0
        neq = self.neq
        dxN = self.dxN
        dxO = self.dxO
        
        val = val.at[phien0].set(neq.bc_phie_n(phievec[0], phievec[1], phievec_sep[dxO], phievec_sep[dxO+1],\
                           uvec[0], uvec[1], uvec_sep[dxO], uvec_sep[dxO+1], \
                           Tvec[0], Tvec[1], Tvec_sep[dxO], Tvec_sep[dxO+1]))
        
        val = val.at[phien0 + 1: phien0 + dxN + 1].set(vmap(neq.electrolyte_poten)(uvec[0:dxN], uvec[1:dxN+1], uvec[2:dxN+2],
        phievec[0:dxN],phievec[1:dxN+1], phievec[2:dxN+2], Tvec[0:dxN], Tvec[1:dxN+1], Tvec[2:dxN+2], jvec[0:dxN]))
        
        val = val.at[phien0 + dxN+1].set(neq.bc_zero_dirichlet(phievec[dxN], phievec[dxN+1]))
        return val
        
    
    
    
    #    @jax.jit  
    @partial(jax.jit, static_argnums=(0,))
    def res_phis(self,val, phis_pe, jvec_pe, phis_ne, jvec_ne):
        phisp0 = self.phisp0
        peq= self.peq
        dxP = self.dxP
        phisn0  = self.phisn0
        neq = self.neq
        dxN = self.dxN
        val = val.at[phisp0].set(peq.bc_phis(phis_pe[0], phis_pe[1], self.Iapp))
        #    val = val.at[phisp0], peq.bc_zero_dirichlet(phis_pe[0], phis_pe[1]))
        val = val.at[phisp0 + 1: phisp0 + dxP+1].set(vmap(peq.solid_poten)(phis_pe[0:dxP], phis_pe[1:dxP+1], phis_pe[2:dxP+2], jvec_pe[0:dxP]))
        val = val.at[phisp0 + dxP+1].set(peq.bc_phis(phis_pe[dxP], phis_pe[dxP+1], 0) )
        #    val = val.at[phisp0 + dxP+1], peq.bc_zero_dirichlet(phis_pe[dxP], phis_pe[dxP+1]) )
        
        val = val.at[phisn0].set(neq.bc_phis(phis_ne[0], phis_ne[1], 0))
        val = val.at[phisn0 + 1: phisn0 + dxN +1].set(vmap(neq.solid_poten)(phis_ne[0:dxN], phis_ne[1:dxN+1], phis_ne[2:dxN+2], jvec_ne[0:dxN]))
        val = val.at[phisn0 + dxN+1].set(neq.bc_phis(phis_ne[dxN], phis_ne[dxN+1], self.Iapp))
        #    val = val.at[phisn0 + dxN+1], neq.bc_zero_dirichlet(phis_ne[dxN], phis_ne[dxN+1]))
        return val
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_pe_fast(self,val, Tvec, uvec, phievec, phisvec, jvec, etavec, cs_pe1, gamma_p, Tvec_old, Tvec_acc, Tvec_sep):
        tp0 = self.tp0
        dxA = self.dxA
        peq = self.peq
        dxP = self.dxP
        val = val.at[tp0].set(peq.bc_temp_ap(Tvec_acc[dxA], Tvec_acc[dxA+1], Tvec[0], Tvec[1]))
        
        val = val.at[tp0 + 1: tp0 + dxP + 1].set(vmap(peq.temperature_fast)(uvec[0:dxP], uvec[1:dxP+1], uvec[2:dxP+2],\
                           phievec[0:dxP], phievec[2:dxP+2], phisvec[0:dxP], phisvec[2:dxP+2], \
                           Tvec[0:dxP], Tvec[1:dxP+1], Tvec[2:dxP+2], \
                           jvec[0:dxP], etavec[0:dxP], cs_pe1, gamma_p, peq.cmax*jnp.ones(dxP), Tvec_old[1:dxP+1]))
        
        val = val.at[tp0 + dxP + 1].set(peq.bc_temp_ps(Tvec[dxP], Tvec[dxP+1], Tvec_sep[0], Tvec_sep[1]))
        return val
    
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))    
    def res_T_ne_fast(self,val, Tvec, uvec, phievec, phisvec, jvec, etavec, cs_ne1, gamma_n, Tvec_old, Tvec_zcc, Tvec_sep):
        tn0= self.tn0
        dxO = self.dxO
        neq = self.neq
        dxN = self.dxN
        val = val.at[ tn0].set(neq.bc_temp_sn(Tvec_sep[dxO], Tvec_sep[dxO+1], Tvec[0], Tvec[1]))
        
        val = val.at[tn0 + 1: tn0+ 1 + dxN].set(vmap(neq.temperature_fast)(uvec[0:dxN], uvec[1:dxN+1], uvec[2:dxN+2],\
                           phievec[0:dxN], phievec[2:dxN+2], phisvec[0:dxN], phisvec[2:dxN+2], \
                           Tvec[0:dxN], Tvec[1:dxN+1], Tvec[2:dxN+2], \
                           jvec[0:dxN], etavec[0:dxN], cs_ne1, gamma_n, neq.cmax*jnp.ones(dxN), Tvec_old[1:dxN+1]))
        
        val = val.at[tn0+ 1 + dxN].set(neq.bc_temp_n(Tvec[dxN], Tvec[dxN+1], Tvec_zcc[0], Tvec_zcc[1]))
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_j_fast(self,val, jvec_pe, uvec_pe, Tvec_pe, eta_pe, cs_pe1, gamma_p, jvec_ne, uvec_ne, Tvec_ne, eta_ne, cs_ne1, gamma_n):
        jp0 = self.jp0
        peq= self.peq
        dxP = self.dxP
        jn0  = self.jn0
        neq = self.neq
        dxN = self.dxN
        val = val.at[jp0:jp0 + dxP].set(vmap(peq.ionic_flux_fast)(jvec_pe, uvec_pe[1:dxP+1], Tvec_pe[1:dxP+1], eta_pe, cs_pe1, gamma_p, peq.cmax*jnp.ones(dxP)))
        val = val.at[jn0: jn0 + dxN].set(vmap(neq.ionic_flux_fast)(jvec_ne, uvec_ne[1:dxN+1], Tvec_ne[1:dxN+1], eta_ne,cs_ne1, gamma_n, neq.cmax*jnp.ones(dxN)))
        return val
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_eta_fast(self,val, eta_pe, phis_pe, phie_pe, Tvec_pe, jvec_pe, cs_pe1, gamma_p, eta_ne, phis_ne, phie_ne, Tvec_ne, jvec_ne, cs_ne1, gamma_n):
        etap0 = self.etap0
        etan0 = self.etan0
        dxP = self.dxP; dxN = self.dxN; peq=self.peq; neq=self.neq; 
        val = val.at[etap0:etap0 + dxP].set(vmap(peq.over_poten_fast)(eta_pe, phis_pe[1:dxP+1], phie_pe[1:dxP+1], Tvec_pe[1:dxP+1], jvec_pe, cs_pe1, gamma_p, peq.cmax*jnp.ones(dxP)))
        val = val.at[etan0: etan0 + dxN].set(vmap(neq.over_poten_fast)(eta_ne, phis_ne[1:dxN+1], phie_ne[1:dxN+1], Tvec_ne[1:dxN+1],jvec_ne, cs_ne1, gamma_n, neq.cmax*jnp.ones(dxN)))
        return val



#    
