from jax import vmap, grad
import jax.numpy as jnp

from derivativesBuilder import *
from unpack import unpack

from settings import dxA, dxP, dxO, dxN, dxZ

class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))

def partials(eqn_a, eqn_p, eqn_o, eqn_n, eqn_z):

    #Positive electrode
    ce_p_d = (vmap(grad(eqn_p.electrolyte_conc, argnums=range(0, 8))))
    j_p_d = (vmap(grad(eqn_p.ionic_flux, argnums=range(0, 4))))
    eta_p_d = (vmap(grad(eqn_p.over_poten, argnums=range(0, 5))))
    phis_p_d = (vmap(grad(eqn_p.solid_poten, argnums=range(0, 4))))
    phie_p_d = (vmap(grad(eqn_p.electrolyte_poten, argnums=range(0, 10))))
    T_p_d = (vmap(grad(eqn_p.temperature, argnums=range(0, 12))))

    ce_p_bc0_d = grad(eqn_p.bc_zero_neumann, argnums=(0, 1))
    ce_p_bc1_d = grad(eqn_p.bc_ce_po, argnums=range(0, 8))

    phis_p_bc0_d = grad(eqn_p.bc_phis, argnums=(0, 1))
    phis_p_bc1_d = grad(eqn_p.bc_phis, argnums=(0, 1))

    phie_p_bc0_d = grad(eqn_p.bc_zero_neumann, argnums=(0, 1))
    phie_p_bc1_d = grad(eqn_p.bc_phie_p, argnums=range(0, 12))

    T_p_bc0_d = grad(eqn_p.bc_temp_ap, argnums=range(0, 4))
    T_p_bc1_d = grad(eqn_p.bc_temp_po, argnums=range(0, 4))

    #Separator
    ce_o_d = vmap(grad(eqn_o.electrolyte_conc, argnums=range(0, 6)))
    phie_o_d = vmap(grad(eqn_o.electrolyte_poten, argnums=range(0, 9)))
    T_o_d = vmap(grad(eqn_o.temperature, argnums=range(0, 8)))

    o_bc = grad(eqn_p.bc_inter_cont, argnums=range(0, 4))

    #Negative electrode
    ce_n_d = vmap(grad(eqn_n.electrolyte_conc, argnums=range(0, 8)))
    j_n_d = vmap(grad(eqn_n.ionic_flux, argnums=range(0, 4)))
    eta_n_d = vmap(grad(eqn_n.over_poten, argnums=range(0, 5)))
    phis_n_d = vmap(grad(eqn_n.solid_poten, argnums=range(0, 4)))
    phie_n_d = vmap(grad(eqn_n.electrolyte_poten, argnums=range(0, 10)))
    T_n_d = vmap(grad(eqn_n.temperature, argnums=range(0, 12)))

    ce_n_bc0_d = grad(eqn_n.bc_ce_on, argnums=range(0, 8))
    ce_n_bc1_d = grad(eqn_n.bc_zero_neumann, argnums=(0, 1))

    phis_n_bc0_d = grad(eqn_n.bc_phis, argnums=(0, 1))
    phis_n_bc1_d = grad(eqn_n.bc_phis, argnums=(0, 1))

    phie_n_bc0_d = grad(eqn_n.bc_phie_n, argnums=range(0, 12))
    phie_n_bc1_d = grad(eqn_n.bc_zero_dirichlet, argnums=(0, 1))

    T_n_bc0_d = grad(eqn_n.bc_temp_on, argnums=range(0, 4))
    T_n_bc1_d = grad(eqn_n.bc_temp_nz, argnums=range(0, 4))

    #Positive current collector
    T_a_bc0_d = grad(eqn_a.bc_temp_a, argnums=(0, 1))
    T_a_d = vmap(grad(eqn_a.temperature, argnums=range(0, 3)))
    T_a_bc1_d = grad(eqn_p.bc_inter_cont, argnums=range(0, 4))

    #Negative current collector
    T_z_bc0_d = grad(eqn_n.bc_inter_cont, argnums=range(0, 4))
    T_z_d = vmap(grad(eqn_z.temperature, argnums=range(0, 3)))
    T_z_bc1_d = grad(eqn_z.bc_temp_z, argnums=(0, 1))

    part = dict([
        ('ce_p_d', ce_p_d),
        ('j_p_d', j_p_d),
        ('eta_p_d', eta_p_d),
        ('phis_p_d', phis_p_d),
        ('phie_p_d', phie_p_d),
        ('T_p_d', T_p_d),
        ('ce_p_bc0_d',ce_p_bc0_d),
        ('ce_p_bc1_d', ce_p_bc1_d),
        ('phis_p_bc0_d', phis_p_bc0_d),
        ('phis_p_bc1_d', phis_p_bc1_d),
        ('phie_p_bc0_d', phie_p_bc0_d),
        ('phie_p_bc1_d', phie_p_bc1_d),
        ('T_p_bc0_d', T_p_bc0_d),
        ('T_p_bc1_d', T_p_bc1_d),

        ('ce_o_d', ce_o_d),
        ('phie_o_d', phie_o_d),
        ('T_o_d', T_o_d),
        ('o_bc', o_bc),

        ('ce_n_d', ce_n_d),
        ('j_n_d', j_n_d),
        ('eta_n_d', eta_n_d),
        ('phis_n_d', phis_n_d),
        ('phie_n_d', phie_n_d),
        ('T_n_d', T_n_d),
        ('ce_n_bc0_d', ce_n_bc0_d),
        ('ce_n_bc1_d', ce_n_bc1_d),
        ('phie_n_bc0_d', phie_n_bc0_d),
        ('phie_n_bc1_d', phie_n_bc1_d),
        ('phis_n_bc0_d', phis_n_bc0_d),
        ('phis_n_bc1_d', phis_n_bc1_d),
        ('T_n_bc0_d', T_n_bc0_d),
        ('T_n_bc1_d', T_n_bc1_d),

        ('T_a_bc0_d', T_a_bc0_d),
        ('T_a_d', T_a_d),
        ('T_a_bc1_d', T_a_bc1_d),

        ('T_z_bc0_d', T_z_bc0_d),
        ('T_z_d', T_z_d),
        ('T_z_bc1_d', T_z_bc1_d)
    ])

    return HashableDict(part)


# @partial(jax.jit, static_argnums=(4,5,6))
def compute_jac(gamma_p_vec, gamma_n_vec, part, Iapp):

    @jax.jit
    def jacfn(U, Uold, cs_pe1, cs_ne1, delta_t):
        
        Jnew = jnp.zeros([23, len(U)])

        ce_p, ce_o, ce_n, \
        T_a, T_p, T_o, T_n, T_z, \
        phie_p, phie_o, phie_n, \
        phis_p, phis_n, \
        j_p, j_n, \
        eta_p, eta_n = unpack(U)

        ce_p_old, ce_o_old, ce_n_old, \
        T_a_old, T_p_old, T_o_old, T_n_old, T_z_old, \
        _, _, _, _, _, _, _, _, _ = unpack(Uold)


        arg_up = [ce_p[0:dxP], ce_p[1:dxP + 1], ce_p[2:dxP + 2],
                  T_p[0:dxP], T_p[1:dxP + 1], T_p[2:dxP + 2],
                  j_p[0:dxP],
                  ce_p_old[1:dxP + 1],
                  delta_t*jnp.ones(dxP)]
        Jnew = build_dup(Jnew, part['ce_p_d'](*arg_up), dxA, dxP)

        arg_jp = [j_p[0:dxP],
                  ce_p[1:dxP + 1],
                  T_p[1:dxP + 1],
                  eta_p[0:dxP],
                  cs_pe1,
                  gamma_p_vec]
        Jnew = build_djp(Jnew, part['j_p_d'](*arg_jp), dxA, dxP)

        arg_etap = [eta_p[0:dxP],
                    phis_p[1:dxP + 1],
                    phie_p[1:dxP + 1],
                    T_p[1:dxP + 1],
                    j_p[0:dxP],
                    cs_pe1,
                    gamma_p_vec]
        Jnew = build_detap(Jnew, part['eta_p_d'](*arg_etap), dxA, dxP)

        arg_phisp = [phis_p[0:dxP], phis_p[1:dxP + 1], phis_p[2:dxP + 2],
                     j_p[0:dxP]]
        Jnew = build_dphisp(Jnew, part['phis_p_d'](*arg_phisp), dxA, dxP)

        arg_phiep = [ce_p[0:dxP], ce_p[1:dxP + 1], ce_p[2:dxP + 2],
                     phie_p[0:dxP], phie_p[1:dxP + 1], phie_p[2:dxP + 2],
                     T_p[0:dxP], T_p[1:dxP + 1], T_p[2:dxP + 2],
                     j_p[0:dxP]]
        Jnew = build_dphiep(Jnew, part['phie_p_d'](*arg_phiep), dxA, dxP)

        arg_Tp = [ce_p[0:dxP], ce_p[1:dxP + 1], ce_p[2:dxP + 2],
                  phie_p[0:dxP], phie_p[2:dxP + 2],
                  phis_p[0:dxP], phis_p[2:dxP + 2],
                  T_p[0:dxP], T_p[1:dxP + 1], T_p[2:dxP + 2],
                  j_p[0:dxP],
                  eta_p[0:dxP],
                  cs_pe1,
                  gamma_p_vec,
                  T_p_old[1:dxP + 1],
                  delta_t*jnp.ones(dxP)]
        Jnew = build_dTp(Jnew, part['T_p_d'](*arg_Tp), dxA, dxP)


        ce_p_bc0_d = part['ce_p_bc0_d'](ce_p[0], ce_p[1])
        ce_p_bc1_d = part['ce_p_bc1_d'](ce_p[dxP], ce_p[dxP + 1],
                                  T_p[dxP], T_p[dxP + 1],
                                  ce_o[0], ce_o[1],
                                  T_o[0], T_o[1])

        phis_p_bc0_d = part['phis_p_bc0_d'](phis_p[0], phis_p[1], Iapp)
        phis_p_bc1_d = part['phis_p_bc1_d'](phis_p[dxP], phis_p[dxP + 1], 0)

        phie_p_bc0_d = part['phie_p_bc0_d'](phie_p[0], phie_p[1])
        phie_p_bc1_d = part['phie_p_bc1_d'](phie_p[dxP], phie_p[dxP + 1],
                                        phie_o[0], phie_o[1],
                                        ce_p[dxP], ce_p[dxP + 1],
                                        ce_o[0], ce_o[1],
                                        T_p[dxP], T_p[dxP + 1],
                                        T_o[0], T_o[1])

        T_p_bc0_d = part['T_p_bc0_d'](T_a[dxA], T_a[dxA + 1],
                                  T_p[0], T_p[1])
        T_p_bc1_d = part['T_p_bc1_d'](T_p[dxP], T_p[dxP + 1],
                                  T_o[0], T_o[1])


        bc_p = dict([('u', jnp.concatenate((jnp.array(ce_p_bc0_d), jnp.array(ce_p_bc1_d)))),
                     ('phis', jnp.concatenate((jnp.array(phis_p_bc0_d), jnp.array(phis_p_bc1_d)))),
                     ('phie', jnp.concatenate((jnp.array(phie_p_bc0_d), jnp.array(phie_p_bc1_d)))),
                     ('T', jnp.concatenate((jnp.array(T_p_bc0_d), jnp.array(T_p_bc1_d))))])
        Jnew = build_bc_p(Jnew, bc_p, dxA, dxP)

        o_bc = part['o_bc'](ce_o[0], ce_o[1],
                            ce_p[dxP], ce_p[dxP + 1])
        Jnew = build_bc_s(Jnew, jnp.concatenate((jnp.array(o_bc), jnp.array(o_bc))), dxA, dxP, dxO)

        ce_o_d = part['ce_o_d'](ce_o[0:dxO], ce_o[1:dxO + 1], ce_o[2:dxO + 2],
                          T_o[0:dxO], T_o[1:dxO + 1], T_o[2:dxO + 2],
                          ce_o_old[1:dxO + 1],
                          delta_t*jnp.ones(dxO))
        Jnew = build_dus(Jnew, ce_o_d, dxA, dxP, dxO)

        phie_o_d = part['phie_o_d'](ce_o[0:dxO], ce_o[1:dxO + 1], ce_o[2:dxO + 2],
                                phie_o[0:dxO], phie_o[1:dxO + 1], phie_o[2:dxO + 2],
                                T_o[0:dxO], T_o[1:dxO + 1], T_o[2:dxO + 2])
        Jnew = build_dphies(Jnew, phie_o_d, dxA, dxP, dxO)

        T_o_d = part['T_o_d'](ce_o[0:dxO], ce_o[1:dxO + 1], ce_o[2:dxO + 2],
                          phie_o[0:dxO], phie_o[2:dxO + 2],
                          T_o[0:dxO], T_o[1:dxO + 1], T_o[2:dxO + 2],
                          T_o_old[1:dxO + 1],
                          delta_t*jnp.ones(dxO))
        Jnew = build_dTs(Jnew, T_o_d, dxA, dxP, dxO)

        ce_n_bc0_d = part['ce_n_bc0_d'](ce_n[0], ce_n[1],
                                  T_n[0], T_n[1],
                                  ce_o[dxO], ce_o[dxO + 1],
                                  T_o[dxO], T_o[dxO + 1])
        ce_n_d = part['ce_n_d'](ce_n[0:dxN], ce_n[1:dxN + 1], ce_n[2:dxN + 2],
                          T_n[0:dxN], T_n[1:dxN + 1], T_n[2:dxN + 2],
                          j_n[0:dxN],
                          ce_n_old[1:dxN + 1],
                          delta_t*jnp.ones(dxP))
        ce_n_bc1_d = part['ce_n_bc1_d'](ce_n[dxN], ce_n[dxN + 1])
        Jnew = build_dun(Jnew, ce_n_d, dxA, dxP, dxO, dxN)

        arg_jn = [j_n[0:dxN],
                  ce_n[1:dxN + 1],
                  T_n[1:dxN + 1],
                  eta_n[0:dxP],
                  cs_ne1,
                  gamma_n_vec]
        Jnew = build_djn(Jnew, part['j_n_d'](*arg_jn), dxA, dxP, dxO, dxN)

        arg_etan = [eta_n[0:dxN],
                    phis_p[1:dxN + 1],
                    phie_n[1:dxN + 1],
                    T_n[1:dxN + 1],
                    j_n[0:dxN],
                    cs_ne1,
                    gamma_n_vec]
        Jnew = build_detan(Jnew, part['eta_n_d'](*arg_etan), dxA, dxP, dxO, dxN)

        arg_phisn = [phis_n[0:dxN], phis_n[1:dxN + 1], phis_n[2:dxN + 2], 
                     j_n[0:dxN]]
        phis_n_bc0_d = part['phis_n_bc0_d'](phis_n[0], phis_n[1], 0)
        phis_n_bc1_d = part['phis_n_bc1_d'](phis_n[dxN], phis_n[dxN + 1], Iapp)
        Jnew = build_dphisn(Jnew, part['phis_n_d'](*arg_phisn), dxA, dxP, dxO, dxN)

        arg_phien = [ce_n[0:dxN], ce_n[1:dxN + 1], ce_n[2:dxN + 2],
                     phie_n[0:dxN], phie_n[1:dxN + 1], phie_n[2:dxN + 2],
                     T_n[0:dxN], T_n[1:dxN + 1], T_n[2:dxN + 2],
                     j_n[0:dxN]]
        Jnew = build_dphien(Jnew, part['phie_n_d'](*arg_phien), dxA, dxP, dxO, dxN)

        phie_n_bc0_d = part['phie_n_bc0_d'](phie_n[0], phie_n[1],
                                        phie_o[dxO], phie_o[dxO + 1],
                                        ce_n[0], ce_n[1],
                                        ce_o[dxO], ce_o[dxO + 1],
                                        T_n[0], T_n[1],
                                        T_o[dxO], T_o[dxO + 1])
        phie_n_bc1_d = part['phie_n_bc1_d'](phie_n[dxN], phie_n[dxN + 1])
        arg_Tn = [ce_n[0:dxN], ce_n[1:dxN + 1], ce_n[2:dxN + 2],
                  phie_n[0:dxN], phie_n[2:dxN + 2],
                  phis_n[0:dxN], phis_n[2:dxN + 2],
                  T_n[0:dxN], T_n[1:dxN + 1], T_n[2:dxN + 2],
                  j_n[0:dxN],
                  eta_n[0:dxN],
                  cs_ne1,
                  gamma_n_vec,
                  T_n_old[1:dxN + 1],
                  delta_t*jnp.ones(dxN)]
        Jnew = build_dTn(Jnew, part['T_n_d'](*arg_Tn), dxA, dxP, dxO, dxN)

        T_n_bc0_d = part['T_n_bc0_d'](T_o[dxO], T_o[dxO + 1],
                                  T_n[0], T_n[1])
        T_n_bc1_d = part['T_n_bc1_d'](T_n[dxN], T_n[dxN + 1],
                                  T_z[0], T_z[1])
        bc_n = dict([('u', jnp.concatenate((jnp.array(ce_n_bc0_d), jnp.array(ce_n_bc1_d)))),
                     ('phis', jnp.concatenate((jnp.array(phis_n_bc0_d), jnp.array(phis_n_bc1_d)))),
                     ('phie', jnp.concatenate((jnp.array(phie_n_bc0_d), jnp.array(phie_n_bc1_d)))),
                     ('T', jnp.concatenate((jnp.array(T_n_bc0_d), jnp.array(T_n_bc1_d))))
                     ])
        Jnew = build_bc_n(Jnew, bc_n, dxA, dxP, dxO, dxN)

        T_a_bc0_d = part['T_a_bc0_d'](T_a[0], T_a[1])
        T_a_d = part['T_a_d'](T_a[0:dxA], T_a[1:dxA + 1], T_a[2:dxA + 2], 
                          T_a_old[1:dxA + 1],
                          delta_t*jnp.ones(dxA))
        T_a_bc1_d = part['T_a_bc1_d'](T_a[dxA], T_a[dxA + 1],
                                  T_p[0], T_p[1])
        T_z_bc0_d = part['T_z_bc0_d'](T_n[dxN], T_n[dxN + 1],
                                  T_z[0], T_z[1])
        T_z_d = part['T_z_d'](T_z[0:dxZ], T_z[1:dxZ + 1], T_z[2:dxZ + 2],
                          T_z_old[1:dxZ + 1],
                          delta_t*jnp.ones(dxZ))
        T_z_bc1_d = part['T_z_bc1_d'](T_z[dxZ], T_z[dxZ + 1])
        Jnew = build_dTa(Jnew, T_a_d, dxA)
        Jnew = build_dTz(Jnew, T_z_d, dxA, dxP, dxO, dxN, dxZ)

        bc_acc = dict([
            ('acc', jnp.concatenate((jnp.array(T_a_bc0_d), jnp.array(T_a_bc1_d)))),
            ('zcc', jnp.concatenate((jnp.array(T_z_bc0_d), jnp.array(T_z_bc1_d))))
        ])
        Jnew = build_bc_cc(Jnew, bc_acc, dxA, dxP, dxO, dxN, dxZ)

        return Jnew
    
    return jacfn






