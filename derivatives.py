from jax import vmap, grad


class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))

def partials(aeq, peq, oeq, neq, zeq):

    #Positive electrode
    
    dce_p_bc1 = grad(peq.cNewmann_bc, argnums=(0, 1))
    dce_p = (vmap(grad(peq.electConc, argnums=range(0, 8))))
    dce_p_bc2 = grad(peq.eConc_po_bc, argnums=range(0, 8))
    
    dj_p = (vmap(grad(peq.ionicFlux, argnums=range(0, 4))))

    deta_p = (vmap(grad(peq.overPotential, argnums=range(0, 5))))

    dphis_p_bc1 = grad(peq.sPhasePoten_bc, argnums=(0, 1))
    dphis_p = (vmap(grad(peq.sPhasePoten, argnums=range(0, 4))))
    dphis_p_bc2 = grad(peq.sPhasePoten_bc, argnums=(0, 1))

    dphie_p_bc1 = grad(peq.cNewmann_bc, argnums=(0, 1))
    dphie_p = (vmap(grad(peq.electPoten, argnums=range(0, 10))))
    dphie_p_bc2 = grad(peq.ePoten_po_bc, argnums=range(0, 12))

    dT_p_bc1 = grad(peq.temp_ap_bc, argnums=range(0, 4))
    dT_p = (vmap(grad(peq.temperature, argnums=range(0, 12))))
    dT_p_bc2 = grad(peq.temp_po_bc, argnums=range(0, 4))

    #Separator

    bc_o = grad(peq.interSecc_bc, argnums=range(0, 4))

    dce_o = vmap(grad(oeq.electConc, argnums=range(0, 6)))

    dphie_o = vmap(grad(oeq.electPoten, argnums=range(0, 9)))

    dT_o = vmap(grad(oeq.temperature, argnums=range(0, 8)))

    #Negative electrode

    dce_n_bc1 = grad(neq.eConc_on_bc, argnums=range(0, 8))
    dce_n = vmap(grad(neq.electConc, argnums=range(0, 8)))
    dce_n_bc2 = grad(neq.cNewmann_bc, argnums=(0, 1))

    dj_n = vmap(grad(neq.ionicFlux, argnums=range(0, 4)))

    deta_n = vmap(grad(neq.overPotential, argnums=range(0, 5)))

    dphis_n_bc1 = grad(neq.sPhasePoten_bc, argnums=(0, 1))
    dphis_n = vmap(grad(neq.sPhasePoten, argnums=range(0, 4)))
    dphis_n_bc2 = grad(neq.sPhasePoten_bc, argnums=(0, 1))

    dphie_n_bc1 = grad(neq.ePoten_on_bc, argnums=range(0, 12))
    dphie_n = vmap(grad(neq.electPoten, argnums=range(0, 10)))
    dphie_n_bc2 = grad(neq.cDirichlet_bc, argnums=(0, 1))

    dT_n_bc1 = grad(neq.temp_on_bc, argnums=range(0, 4))
    dT_n = vmap(grad(neq.temperature, argnums=range(0, 12)))
    dT_n_bc2 = grad(neq.temp_nz_bc, argnums=range(0, 4))

    #Positive current collector

    dT_a_bc1 = grad(aeq.temp_a_bc, argnums=(0, 1))
    dT_a = vmap(grad(aeq.temperature, argnums=range(0, 3)))
    dT_a_bc2 = grad(peq.interSecc_bc, argnums=range(0, 4))

    #Negative current collector

    dT_z_bc1 = grad(neq.interSecc_bc, argnums=range(0, 4))
    dT_z = vmap(grad(zeq.temperature, argnums=range(0, 3)))
    dT_z_bc2 = grad(zeq.temp_z_bc, argnums=(0, 1))

    d = dict([
        ('dce_p', dce_p),
        ('dj_p', dj_p),
        ('deta_p', deta_p),
        ('dphis_p', dphis_p),
        ('dphie_p', dphie_p),
        ('dT_p', dT_p),
        ('dce_p_bc1',dce_p_bc1),
        ('dce_p_bc2', dce_p_bc2),
        ('dphis_p_bc1', dphis_p_bc1),
        ('dphis_p_bc2', dphis_p_bc2),
        ('dphie_p_bc1', dphie_p_bc1),
        ('dphie_p_bc2', dphie_p_bc2),
        ('dT_p_bc1', dT_p_bc1),
        ('dT_p_bc2', dT_p_bc2),

        ('dce_o', dce_o),
        ('dphie_o', dphie_o),
        ('dT_o', dT_o),
        ('bc_o', bc_o),

        ('dce_n', dce_n),
        ('dj_n', dj_n),
        ('deta_n', deta_n),
        ('dphis_n', dphis_n),
        ('dphie_n', dphie_n),
        ('dT_n', dT_n),
        ('dce_n_bc1', dce_n_bc1),
        ('dce_n_bc2', dce_n_bc2),
        ('dphie_n_bc1', dphie_n_bc1),
        ('dphie_n_bc2', dphie_n_bc2),
        ('dphis_n_bc1', dphis_n_bc1),
        ('dphis_n_bc2', dphis_n_bc2),
        ('dT_n_bc1', dT_n_bc1),
        ('dT_n_bc2', dT_n_bc2),

        ('dT_a_bc1', dT_a_bc1),
        ('dT_a', dT_a),
        ('dT_a_bc2', dT_a_bc2),

        ('dT_z_bc1', dT_z_bc1),
        ('dT_z', dT_z),
        ('dT_z_bc2', dT_z_bc2)
    ])

    return HashableDict(d)






