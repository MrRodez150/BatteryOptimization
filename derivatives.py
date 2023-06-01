from jax import vmap, grad
import jax.numpy as jnp

from derivativesBuilder import *
from unpack import unpack

from settings import dxA, dxP, dxO, dxN, dxZ

class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))

def partials(accq, peq, sepq, neq, zccq):

    dup = (vmap(grad(peq.electrolyte_conc, argnums=range(0, 8))))
    djp = (vmap(grad(peq.ionic_flux_fast, argnums=range(0, 4))))

    detap = (vmap(grad(peq.over_poten_fast, argnums=range(0, 5))))

    dphisp = (vmap(grad(peq.solid_poten, argnums=range(0, 4))))

    dphiep = (vmap(grad(peq.electrolyte_poten, argnums=range(0, 10))))

    dTp = (vmap(grad(peq.temperature_fast, argnums=range(0, 12))))


    dup_bc0 = grad(peq.bc_zero_neumann, argnums=(0, 1))
    dup_bcM = grad(peq.bc_u_sep_p, argnums=range(0, 8))

    dphisp_bc0 = grad(peq.bc_phis, argnums=(0, 1))
    dphisp_bcM = grad(peq.bc_phis, argnums=(0, 1))

    dphiep_bc0 = grad(peq.bc_zero_neumann, argnums=(0, 1))
    dphiep_bcM = grad(peq.bc_phie_p, argnums=range(0, 12))

    dTp_bc0 = grad(peq.bc_temp_ap, argnums=range(0, 4))
    dTp_bcM = grad(peq.bc_temp_ps, argnums=range(0, 4))



    bc_s = grad(peq.bc_inter_cont, argnums=range(0, 4))

    dus = vmap(grad(sepq.electrolyte_conc, argnums=range(0, 6)))

    dphies = vmap(grad(sepq.electrolyte_poten, argnums=range(0, 9)))


    dTs = vmap(grad(sepq.temperature, argnums=range(0, 8)))



    dun_bc0 = grad(neq.bc_u_sep_n, argnums=range(0, 8))
    dun = vmap(grad(neq.electrolyte_conc, argnums=range(0, 8)))
    dun_bcM = grad(neq.bc_zero_neumann, argnums=(0, 1))

    djn = vmap(grad(neq.ionic_flux_fast, argnums=range(0, 4)))


    detan = vmap(grad(neq.over_poten_fast, argnums=range(0, 5)))


    dphisn = vmap(grad(neq.solid_poten, argnums=range(0, 4)))
    dphisn_bc0 = grad(neq.bc_phis, argnums=(0, 1))
    dphisn_bcM = grad(neq.bc_phis, argnums=(0, 1))

    dphien = vmap(grad(neq.electrolyte_poten, argnums=range(0, 10)))

    dphien_bc0 = grad(neq.bc_phie_n, argnums=range(0, 12))
    dphien_bcM = grad(neq.bc_zero_dirichlet, argnums=(0, 1))


    dTn = vmap(grad(neq.temperature_fast, argnums=range(0, 12)))


    dTn_bc0 = grad(neq.bc_temp_sn, argnums=range(0, 4))
    dTn_bcM = grad(neq.bc_temp_n, argnums=range(0, 4))



    dTa_bc0 = grad(accq.bc_temp_a, argnums=(0, 1))
    dTa = vmap(grad(accq.temperature, argnums=range(0, 3)))
    dTa_bcM = grad(peq.bc_inter_cont, argnums=range(0, 4))

    dTz_bc0 = grad(neq.bc_inter_cont, argnums=range(0, 4))
    dTz = vmap(grad(zccq.temperature, argnums=range(0, 3)))
    dTz_bcM = grad(zccq.bc_temp_z, argnums=(0, 1))

    part = dict([
        ('dup', dup),
        ('djp', djp),
        ('detap', detap),
        ('dphisp', dphisp),
        ('dphiep', dphiep),
        ('dTp', dTp),
        ('dup_bc0',dup_bc0),
        ('dup_bcM', dup_bcM),
        ('dphisp_bc0', dphisp_bc0),
        ('dphisp_bcM', dphisp_bcM),
        ('dphiep_bc0', dphiep_bc0),
        ('dphiep_bcM', dphiep_bcM),
        ('dTp_bc0', dTp_bc0),
        ('dTp_bcM', dTp_bcM),

        ('dus', dus),
        ('dphies', dphies),
        ('dTs', dTs),
        ('bc_s', bc_s),

        ('dun', dun),
        ('djn', djn),
        ('detan', detan),
        ('dphisn', dphisn),
        ('dphien', dphien),
        ('dTn', dTn),
        ('dun_bc0', dun_bc0),
        ('dun_bcM', dun_bcM),
        ('dphien_bc0', dphien_bc0),
        ('dphien_bcM', dphien_bcM),
        ('dphisn_bc0', dphisn_bc0),
        ('dphisn_bcM', dphisn_bcM),
        ('dTn_bc0', dTn_bc0),
        ('dTn_bcM', dTn_bcM),

        ('dTa_bc0', dTa_bc0),
        ('dTa', dTa),
        ('dTa_bcM', dTa_bcM),

        ('dTz_bc0', dTz_bc0),
        ('dTz', dTz),
        ('dTz_bcM', dTz_bcM)
    ])

    return HashableDict(part)


# @partial(jax.jit, static_argnums=(4,5,6))
def compute_jac(gamma_p_vec, gamma_n_vec, part, peq, neq, Iapp):
    @jax.jit
    def jacfn(U, Uold, cs_pe1, cs_ne1):
        
        Jnew = jnp.zeros([23, len(U)])

        uvec_pe, uvec_sep, uvec_ne, \
        Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
        phie_pe, phie_sep, phie_ne, \
        phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack(U)

        uvec_old_pe, uvec_old_sep, uvec_old_ne, \
        Tvec_old_acc, Tvec_old_pe, Tvec_old_sep, Tvec_old_ne, Tvec_old_zcc, \
        _, _, _, _, _, _, _, _, _ = unpack(Uold)

        arg_up = [uvec_pe[0:dxP], uvec_pe[1:dxP + 1], uvec_pe[2:dxP + 2], Tvec_pe[0:dxP], Tvec_pe[1:dxP + 1], Tvec_pe[2:dxP + 2],
                  jvec_pe[0:dxP], uvec_old_pe[1:dxP + 1]]

        Jnew = build_dup(Jnew, part['dup'](*arg_up), dxA, dxP)

        arg_jp = [jvec_pe[0:dxP], uvec_pe[1:dxP + 1], Tvec_pe[1:dxP + 1], eta_pe[0:dxP], cs_pe1, gamma_p_vec,
                  peq.cmax * jnp.ones([dxP, 1])]
        Jnew = build_djp(Jnew, part['djp'](*arg_jp), dxA, dxP)

        arg_etap = [eta_pe[0:dxP], phis_pe[1:dxP + 1], phie_pe[1:dxP + 1], Tvec_pe[1:dxP + 1], jvec_pe[0:dxP], cs_pe1,
                    gamma_p_vec, peq.cmax * jnp.ones([dxP, 1])]

        Jnew = build_detap(Jnew, part['detap'](*arg_etap), dxA, dxP)
        arg_phisp = [phis_pe[0:dxP], phis_pe[1:dxP + 1], phis_pe[2:dxP + 2], jvec_pe[0:dxP]]

        Jnew = build_dphisp(Jnew, part['dphisp'](*arg_phisp), dxA, dxP)

        arg_phiep = [uvec_pe[0:dxP], uvec_pe[1:dxP + 1], uvec_pe[2:dxP + 2], phie_pe[0:dxP], phie_pe[1:dxP + 1],
                     phie_pe[2:dxP + 2],
                     Tvec_pe[0:dxP], Tvec_pe[1:dxP + 1], Tvec_pe[2:dxP + 2], jvec_pe[0:dxP]]

        Jnew = build_dphiep(Jnew, part['dphiep'](*arg_phiep), dxA, dxP)

        # temperature_fast(self,un, uc, up, phien, phiep, phisn, phisp, Tn, Tc, Tp,j,eta, cs_1, gamma_c, cmax, Told):
        arg_Tp = [uvec_pe[0:dxP], uvec_pe[1:dxP + 1], uvec_pe[2:dxP + 2],
                  phie_pe[0:dxP], phie_pe[2:dxP + 2],
                  phis_pe[0:dxP], phis_pe[2:dxP + 2],
                  Tvec_pe[0:dxP], Tvec_pe[1:dxP + 1], Tvec_pe[2:dxP + 2],
                  jvec_pe[0:dxP],
                  eta_pe[0:dxP],
                  cs_pe1, gamma_p_vec, peq.cmax * jnp.ones([dxP, 1]),
                  Tvec_old_pe[1:dxP + 1]]

        Jnew = build_dTp(Jnew, part['dTp'](*arg_Tp), dxA, dxP)

        dup_bc0 = part['dup_bc0'](uvec_pe[0], uvec_pe[1])
        dup_bcM = part['dup_bcM'](uvec_pe[dxP], uvec_pe[dxP + 1], Tvec_pe[dxP], Tvec_pe[dxP + 1],
                               uvec_sep[0], uvec_sep[1], Tvec_sep[0], Tvec_sep[1])

        dphisp_bc0 = part['dphisp_bc0'](phis_pe[0], phis_pe[1], Iapp)
        dphisp_bcM = part['dphisp_bcM'](phis_pe[dxP], phis_pe[dxP + 1], 0)

        dphiep_bc0 = part['dphiep_bc0'](phie_pe[0], phie_pe[1])
        dphiep_bcM = part['dphiep_bcM'](phie_pe[dxP], phie_pe[dxP + 1], phie_sep[0], phie_sep[1],
                                     uvec_pe[dxP], uvec_pe[dxP + 1], uvec_sep[0], uvec_sep[1],
                                     Tvec_pe[dxP], Tvec_pe[dxP + 1], Tvec_sep[0], Tvec_sep[1])

        dTp_bc0 = part['dTp_bc0'](Tvec_acc[dxA], Tvec_acc[dxA + 1], Tvec_pe[0], Tvec_pe[1])
        dTp_bcM = part['dTp_bcM'](Tvec_pe[dxP], Tvec_pe[dxP + 1], Tvec_sep[0], Tvec_sep[1])

        bc_p = dict([('u', jnp.concatenate((jnp.array(dup_bc0), jnp.array(dup_bcM)))),
                     ('phis', jnp.concatenate((jnp.array(dphisp_bc0), jnp.array(dphisp_bcM)))),
                     ('phie', jnp.concatenate((jnp.array(dphiep_bc0), jnp.array(dphiep_bcM)))),
                     ('T', jnp.concatenate((jnp.array(dTp_bc0), jnp.array(dTp_bcM))))])

        Jnew = build_bc_p(Jnew, bc_p, dxA, dxP)

        bc_s = part['bc_s'](uvec_sep[0], uvec_sep[1], uvec_pe[dxP], uvec_pe[dxP + 1])
        Jnew = build_bc_s(Jnew, jnp.concatenate((jnp.array(bc_s), jnp.array(bc_s))), dxA, dxP, dxO)
        dus = part['dus'](uvec_sep[0:dxO], uvec_sep[1:dxO + 1],
                       uvec_sep[2:dxO + 2], Tvec_sep[0:dxO],
                       Tvec_sep[1:dxO + 1], Tvec_sep[2:dxO + 2],
                       uvec_old_sep[1:dxO + 1])

        Jnew = build_dus(Jnew, dus, dxA, dxP, dxO)
        dphies = part['dphies'](uvec_sep[0:dxO], uvec_sep[1:dxO + 1],
                             uvec_sep[2:dxO + 2],
                             phie_sep[0:dxO], phie_sep[1:dxO + 1],
                             phie_sep[2:dxO + 2],
                             Tvec_sep[0:dxO], Tvec_sep[1:dxO + 1],
                             Tvec_sep[2:dxO + 2])
        Jnew = build_dphies(Jnew, dphies, dxA, dxP, dxO)

        dTs = part['dTs'](uvec_sep[0:dxO], uvec_sep[1:dxO + 1],
                       uvec_sep[2:dxO + 2],
                       phie_sep[0:dxO], phie_sep[2:dxO + 2],
                       Tvec_sep[0:dxO], Tvec_sep[1:dxO + 1],
                       Tvec_sep[2:dxO + 2],
                       Tvec_old_sep[1:dxO + 1])

        Jnew = build_dTs(Jnew, dTs, dxA, dxP, dxO)

        dun_bc0 = part['dun_bc0'](uvec_ne[0], uvec_ne[1], Tvec_ne[0], Tvec_ne[1],
                               uvec_sep[dxO], uvec_sep[dxO + 1], Tvec_sep[dxO], Tvec_sep[dxO + 1])
        dun = part['dun'](uvec_ne[0:dxN], uvec_ne[1:dxN + 1],
                       uvec_ne[2:dxN + 2],
                       Tvec_ne[0:dxN], Tvec_ne[1:dxN + 1],
                       Tvec_ne[2:dxN + 2],
                       jvec_ne[0:dxN], uvec_old_ne[1:dxN + 1])

        dun_bcM = part['dun_bcM'](uvec_ne[dxN], uvec_ne[dxN + 1])
        Jnew = build_dun(Jnew, dun, dxA, dxP, dxO, dxN)

        arg_jn = [jvec_ne[0:dxN], uvec_ne[1:dxN + 1], Tvec_ne[1:dxN + 1], eta_ne[0:dxP], cs_ne1, gamma_n_vec,
                  neq.cmax * jnp.ones([dxN, 1])]

        Jnew = build_djn(Jnew, part['djn'](*arg_jn), dxA, dxP, dxO, dxN)

        arg_etan = [eta_ne[0:dxN], phis_pe[1:dxN + 1], phie_ne[1:dxN + 1], Tvec_ne[1:dxN + 1], jvec_ne[0:dxN], cs_ne1,
                    gamma_n_vec,
                    neq.cmax * jnp.ones([dxN, 1])]

        Jnew = build_detan(Jnew, part['detan'](*arg_etan), dxA, dxP, dxO, dxN)

        arg_phisn = [phis_ne[0:dxN], phis_ne[1:dxN + 1], phis_ne[2:dxN + 2], jvec_ne[0:dxN]]

        dphisn_bc0 = part['dphisn_bc0'](phis_ne[0], phis_ne[1], 0)
        dphisn_bcM = part['dphisn_bcM'](phis_ne[dxN], phis_ne[dxN + 1], Iapp)

        Jnew = build_dphisn(Jnew, part['dphisn'](*arg_phisn), dxA, dxP, dxO, dxN)
        arg_phien = [uvec_ne[0:dxN], uvec_ne[1:dxN + 1], uvec_ne[2:dxN + 2], phie_ne[0:dxN], phie_ne[1:dxN + 1],
                     phie_ne[2:dxN + 2],
                     Tvec_ne[0:dxN], Tvec_ne[1:dxN + 1], Tvec_ne[2:dxN + 2], jvec_ne[0:dxN]]

        Jnew = build_dphien(Jnew, part['dphien'](*arg_phien), dxA, dxP, dxO, dxN)
        dphien_bc0 = part['dphien_bc0'](phie_ne[0], phie_ne[1], phie_sep[dxO], phie_sep[dxO + 1], \
                                                                   uvec_ne[0], uvec_ne[1], uvec_sep[dxO], uvec_sep[dxO + 1], \
                                                                   Tvec_ne[0], Tvec_ne[1], Tvec_sep[dxO], Tvec_sep[dxO + 1])
        dphien_bcM = part['dphien_bcM'](phie_ne[dxN], phie_ne[dxN + 1])

        arg_Tn = [uvec_ne[0:dxN], uvec_ne[1:dxN + 1], uvec_ne[2:dxN + 2],
                  phie_ne[0:dxN], phie_ne[2:dxN + 2], \
                  phis_ne[0:dxN], phis_ne[2:dxN + 2],
                  Tvec_ne[0:dxN], Tvec_ne[1:dxN + 1], Tvec_ne[2:dxN + 2],
                  jvec_ne[0:dxN], \
                  eta_ne[0:dxN],
                  cs_ne1, gamma_n_vec, neq.cmax * jnp.ones([dxN, 1]),
                  Tvec_old_ne[1:dxN + 1]]

        Jnew = build_dTn(Jnew, part['dTn'](*arg_Tn), dxA, dxP, dxO, dxN)
        dTn_bc0 = part['dTn_bc0'](Tvec_sep[dxO], Tvec_sep[dxO + 1], Tvec_ne[0], Tvec_ne[1])
        dTn_bcM = part['dTn_bcM'](Tvec_ne[dxN], Tvec_ne[dxN + 1], Tvec_zcc[0], Tvec_zcc[1])
        bc_n = dict([('u', jnp.concatenate((jnp.array(dun_bc0), jnp.array(dun_bcM)))),
                     ('phis', jnp.concatenate((jnp.array(dphisn_bc0), jnp.array(dphisn_bcM)))),
                     ('phie', jnp.concatenate((jnp.array(dphien_bc0), jnp.array(dphien_bcM)))),
                     ('T', jnp.concatenate((jnp.array(dTn_bc0), jnp.array(dTn_bcM))))
                     ])
        Jnew = build_bc_n(Jnew, bc_n, dxA, dxP, dxO, dxN)

        dTa_bc0 = part['dTa_bc0'](Tvec_acc[0], Tvec_acc[1])
        dTa = part['dTa'](Tvec_acc[0:dxA], Tvec_acc[1:dxA + 1],
                                                                        Tvec_acc[2:dxA + 2],
                                                                        Tvec_old_acc[1:dxA + 1])
        dTa_bcM = part['dTa_bcM'](Tvec_acc[dxA], Tvec_acc[dxA + 1], Tvec_pe[0], Tvec_pe[1])

        dTz_bc0 = part['dTz_bc0'](Tvec_ne[dxN], Tvec_ne[dxN + 1], Tvec_zcc[0], Tvec_zcc[1])
        dTz = part['dTz'](Tvec_zcc[0:dxZ], Tvec_zcc[1:dxZ + 1],
                                                                        Tvec_zcc[2:dxZ + 2],
                                                                        Tvec_old_zcc[1:dxZ + 1])
        dTz_bcM = part['dTz_bcM'](Tvec_zcc[dxZ], Tvec_zcc[dxZ + 1])

        Jnew = build_dTa(Jnew, dTa, dxA)
        Jnew = build_dTz(Jnew, dTz, dxA, dxP, dxO, dxN, dxZ)
        bc_acc = dict([
            ('acc', jnp.concatenate((jnp.array(dTa_bc0), jnp.array(dTa_bcM)))),
            ('zcc', jnp.concatenate((jnp.array(dTz_bc0), jnp.array(dTz_bcM))))
        ])
        Jnew = build_bc_cc(Jnew, bc_acc, dxA, dxP, dxO, dxN, dxZ)
        return Jnew
    return jacfn






