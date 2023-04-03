from globalValues import R, T_ref
import jax.numpy as jnp

def eDiffCoeff(vareps,brugg,ce,T):
    De_eff = (vareps**brugg)*1e-4*(10**(-4.43 - 54/(T - 229 - 5e-3*ce) - 0.22e-3*ce))
    return De_eff

def sCondCoeff(sigma,vareps,eps):
    sig_eff = sigma*(1-vareps-eps)
    return sig_eff

def eCondCoeff(vareps,brugg,ce,T):
    kap_eff = (vareps**brugg)*1e-4*ce*(-10.5 + 0.668e-3*ce + 0.494e-6*ce**2 + \
                                (0.074 - 1.78e-5*ce - 8.86e-10*ce**2)*T + \
                                (-6.96e-5 + 2.8e-8*ce)*T**2)**2
    return kap_eff

def eCondCoeff_delT(eps,brugg,u,T):
    kap_eff = 2*(eps**brugg)*1e-4*u*(-10.5 + 0.668*1e-3*u + 0.494*1e-6*u**2 + \
                                (0.074 - 1.78*1e-5*u - 8.86*1e-10*u**2)*T + \
                                (-6.96*1e-5 + 2.8*1e-8*u)*T**2)* \
                                ((0.074 - 1.78*1e-5*u - 8.86*1e-10*u**2) + 2*(-6.96*1e-5 + 2.8*1e-8*u)*T) 
    return kap_eff

def sDiffCoeff(Ds,EaD,T):
    Ds_eff = Ds*jnp.exp( -(EaD/R)*( (1/T) - (1/T_ref) ) )
    return Ds_eff

def kRateCoeff(k,Eak,T):
    Ds_eff = k*jnp.exp( -(Eak/R)*( (1/T) - (1/T_ref) ) )
    return Ds_eff