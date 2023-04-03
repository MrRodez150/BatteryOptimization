import functools

from jax import lax
import jax
import jax.numpy as jnp


@jax.jit
def array_update(state, update_element):
    element, ind = update_element
    J, start_index, row = state
    return (J.at[row, start_index + 6 * ind].set(element), start_index, row), ind

@jax.jit
def array_update_sep(state, update_element):
    element, ind = update_element
    J, start_index, row = state
    return (J.at[row, start_index + 3 * ind].set(element), start_index, row), ind



@functools.partial(jax.jit, static_argnums=(2, 3,))
def build_dce_p(J, partial, ad, pd):
    ranger_p = jnp.arange(1, pd)
    ranger_m = jnp.arange(0, pd - 1)
    # dudTp
    J, _, _ = lax.scan(array_update, (J, 5  + 5+ ad + 1, 0), (partial[5][0:-1], ranger_p))[0]
    J = J.at[2, ad + 1 + 5 + 6 * pd + 3].set(partial[5][-1])
    # dudum

    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5, 17), (partial[0][1:pd], ranger_m))[0]
    J = J.at[16, ad + 1].set(partial[0][0])
    # duduc
    ranger_c = jnp.arange(0, pd)
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5, 11), (partial[1][0:pd], ranger_c))[0]
    # dudup
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 6, 5), (partial[2][0:pd], ranger_c))[0]
    # dudTm
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 5, 12), (partial[3][1:pd], ranger_m))[0]
    # TODO
    J = J.at[13, ad + 1 + 3].set(partial[3][0])
    # dup[3][0] not assigned

    # dudTc
    # TODO : error
    J, _, _ = lax.scan(array_update, (J, 5 + 5 + ad + 1, 6), (partial[4][0:pd], ranger_c))[0]

    # dudj
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 1, 10), (partial[6][0:pd], ranger_c))[0]
    return J


@functools.partial(jax.jit, static_argnums=(2, 3,))
def build_dj_p(J, partial, ad, pd):
    ranger_c = jnp.arange(0, pd)
    # djdj
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 1, 11), (partial[0][0:pd], ranger_c))[0]
    # djdu
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5, 12), (partial[1][0:pd], ranger_c))[0]
    # djdT
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 5, 7), (partial[2][0:pd], ranger_c))[0]
    # djdeta
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 2, 10), (partial[3][0:pd], ranger_c))[0]
    return J


@functools.partial(jax.jit, static_argnums=(2, 3,))
def build_deta_p(J, partial, ad, pd):
    # detadeta
    ranger_c = jnp.arange(0, pd)
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 2, 11), (partial[0][0:pd], ranger_c))[0]
    # deta dphis
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 3, 10), (partial[1][0:pd], ranger_c))[0]
    # deta dphie
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 4, 9), (partial[2][0:pd], ranger_c))[0]
    # detadT
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 5, 8), (partial[3][0:pd], ranger_c))[0]
    # deta dj
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 1, 12), (partial[4][0:pd], ranger_c))[0]
    return J


@functools.partial(jax.jit, static_argnums=(2, 3,))
def build_dphis_p(J, partial, ad, pd):
    ranger_m = jnp.arange(0, pd - 1)
    # dphis dphism
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 3, 17), (partial[0][1:pd], ranger_m))[0]
    J = J.at[18, ad + 1 + 1].set(partial[0][0])
    # dphis dphisc
    ranger_c = jnp.arange(0, pd)
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 3, 11), (partial[1][0:pd], ranger_c))[0]
    # dphis dphisp
    ranger_p = jnp.arange(1, pd)
    # J = J.at[4, ad + 1 + 5 + 3], partial[2][0])
    # J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 3, 5), (partial[2][1:pd], ranger_p))[0]

    # TODO: check
    J = J.at[7, 5 + 6*pd + 1 + ad + 1].set(partial[2][pd-1])
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 3, 5), (partial[2][0:pd-1], ranger_p))[0]


    # dphis dj
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 1, 13), (partial[3][0:pd], ranger_c))[0]
    return J


@functools.partial(jax.jit, static_argnums=(2, 3,))
def build_dphie_p(J, partial, ad, pd):
    # dphie/dum
    ranger_m = jnp.arange(0, pd - 1)
    J = J.at[20, ad + 1].set(partial[0][0])
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5, 21), (partial[0][1:pd], ranger_m))[0]
    # dphie/duc
    ranger_c = jnp.arange(0, pd)
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5, 15), (partial[1][0:pd], ranger_c))[0]
    # dphie/dup
    J, _, _ = lax.scan(array_update, (J, 5 + 6 * 1 + ad + 1, 9), (partial[2][0:pd], ranger_c))[0]
    # dphie/dphiem
    J = J.at[18, ad + 1 + 2].set(partial[3][0])  # check
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 4, 17), (partial[3][1:pd], ranger_m))[0]
    # dphie/dphiec
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 4, 11), (partial[4][0:pd], ranger_c))[0]
    # dphie/dphiep
    ranger_p = jnp.arange(1, pd)
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 4, 5), (partial[5][0:pd - 1], ranger_p))[0]
    J = J.at[7, 5 + 6 * pd + 2 + ad + 1].set(partial[5][pd - 1])
    # dphiep/dTm
    # TODO
    # dphpep/dTm[0] check

    J, _, _ = lax.scan(array_update, (J, 5 + 5 + ad + 1, 16), (partial[6][1:pd], ranger_m))[0]

    # dphiep/dTc
    J, _, _ = lax.scan(array_update, (J, 5 + 5 + ad + 1, 10), (partial[7][0:pd], ranger_c))[0]

    # dphiep/dTp
    J, _, _ = lax.scan(array_update, (J, 5 + 5 + ad + 1, 4), (partial[8][0:pd - 1], ranger_p))[0]
    J = J.at[6, 5 + 6 * pd + 3 + ad + 1].set(partial[8][pd - 1])

    # dphie/dj
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 1, 14), (partial[9][0:pd], ranger_c))[0]
    return J


@functools.partial(jax.jit, static_argnums=(2, 3,))
def build_dT_p(J, partial, ad, pd):
    # dT/dum
    ranger_m = jnp.arange(0, pd - 1)
    J = J.at[21, ad + 1].set(partial[0][0])
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5, 22), (partial[0][1:pd], ranger_m))[0]

    # dT/duc
    ranger_c = jnp.arange(0, pd)
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5, 16), (partial[1][0:pd], ranger_c))[0]

    # dT/dup
    J, _, _ = lax.scan(array_update, (J, 5 + 6 + ad + 1, 10), (partial[2][0:pd], ranger_c))[0]

    # dT/phiem
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 4, 18), (partial[3][1:pd], ranger_m))[0]
    J = J.at[19, ad + 1 + 2].set(partial[3][0])

    # dT/dphiep
    ranger_p = jnp.arange(1, pd)
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 4, 6), (partial[4][0:pd - 1], ranger_p))[0]
    J = J.at[8, 5 + 6 * pd + 2 + ad + 1].set(partial[4][pd - 1])

    # dT/dphism
    J, _, _ = lax.scan(array_update, (J, 5 + 3 + ad + 1, 19), (partial[5][1:pd], ranger_m))[0]
    J = J.at[20, ad + 1 + 1].set(partial[5][0])

    # dT/dphisp
    J, _, _ = lax.scan(array_update, (J, 5 + 3 + ad + 1, 7), (partial[6][0:pd - 1], ranger_p))[0]
    J = J.at[9, 5 + 6 * pd + 2 + ad].set(partial[6][pd - 1])

    # dT/dTm
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 5, 17), (partial[7][1:pd], ranger_m))[0]
    J = J.at[18, ad + 1 + 3].set(partial[7][0])

    # dT/dTc
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 5, 11), (partial[8][0:pd], ranger_c))[0]

    # dT/dTp
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 5, 5), (partial[9][0:pd - 1], ranger_p))[0]
    J = J.at[7, 5 + 6 * pd + 2 + ad + 1 + 1].set(partial[9][pd])

    # dT/dj
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 1, 15), (partial[10][0:pd], ranger_c))[0]
    # dT/deta
    J, _, _ = lax.scan(array_update, (J, ad + 1 + 5 + 2, 14), (partial[11][0:pd], ranger_c))[0]
    return J


@jax.jit
def build_bc_p(J, bc, ad, pd):
    # u0 and uM
    p0 = ad + 1
    sep_o = 4 * (pd + 2) + 2 * pd + 1
    row_u = jnp.array([11, 6, 17, 11, 12, 8, 7, 4, 5, 2])
    col_u = p0 + jnp.array(
        [0, 5, 6 * (pd - 1) + 5, 6 * pd + 5, 6 * (pd - 1) + 10, 6 * pd + 5 + 3, sep_o, sep_o + 3, sep_o + 2,
         sep_o + 2 + 3])
    J = J.at[row_u, col_u].set(bc['ce'])

    row_phis = jnp.array([11, 4, 15, 11])
    col_phis = p0 + jnp.array([1, 5 + 3, 5 + 6*(pd-1) + 3, 5 + 6*pd + 1])
    J = J.at[row_phis, col_phis].set(bc['phis'])
    # TODO: check if this is correct
    J = J.at[7, p0 + 5 + 6*pd + 1 ].set(1)

    row_phie = jnp.array([11, 4, 15, 11, 8, 5, 19, 13, 9, 6, 14, 10, 7, 4])
    col_phie = p0 + jnp.array([2, 5 + 4, 5 + 6*(pd-1) + 4, 5 + 6*pd +2,
                              sep_o + 1, sep_o + 3 + 1, 5+6*(pd-1),
                              5 + 6*pd, sep_o, sep_o + 3, 5 + 6*(pd-1) + 5, 5 + 6*pd + 3,
                              sep_o + 2, sep_o + 3 + 2])
    J = J.at[row_phie, col_phie].set(bc['phie'])

    row_T = jnp.array([15, 10, 11, 4, 15, 11, 8, 5])
    col_T = jnp.array([ad, ad + 1 + 4, ad + 1 + 3, ad + 1 + 10, 5 + 6*(pd-1)+5+ad+1, 5+6*pd+3+ad+1, ad + 1 + sep_o + 2, ad + 1 + sep_o +2 + 3])
    J = J.at[row_T, col_T].set(bc['T'])
    return J

# @jax.jit
def build_bc_s(J, bc, ad, pd, od):
    p0 = ad + 1
    sep0 = 4 * (pd + 2) + 2 * pd + 1 + p0

    n0 = p0 + 4*(pd+2) + 2*pd  + 3*(od+2) + 1
    row_u = jnp.array([11, 8, 21, 15, 8, 4, 14, 11])
    col_u = jnp.array([sep0, sep0 + 3, 5 + 6*(pd-1) + p0, 5+6*pd + p0, n0, n0 + 4, sep0 + 3*od, sep0 + 3*(od+1) ])
    J = J.at[row_u, col_u].set(bc)

    row_phie = jnp.array([18, 14, 11, 8, 7, 1, 14, 11])
    col_phie = jnp.array([5+6*(pd-1) + 4 + p0, 5 + 6*pd +2 + p0, sep0+1, sep0 + 1 + 3, n0+2, n0+4 + 4, sep0 + 3*od + 1, sep0 + 3*od + 4])

    J = J.at[row_phie, col_phie].set(bc)

    row_T = jnp.array([18, 14, 11,8, 14, 11, 7, 1])
    col_T = jnp.array([5 + 6*(pd-1) + 5 + p0, 5 + 6*pd + 3 + p0, sep0 + 2, sep0 + 5, sep0 + 3*od + 2, sep0 + 3*od + 5, n0 + 3, n0 + 4 + 5])
    J = J.at[row_T, col_T].set(bc)

    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4))
def build_dce_o(J, partial, ad, pd, od):
    ranger_c = jnp.arange(0,od)
    p0 = ad + 1
    sep0 = 4*(pd+2) + 2*pd + 1 + p0
    # dus/dum
    J, _, _ = lax.scan(array_update_sep, (J, sep0, 14), (partial[0][0:od], ranger_c))[0]
    # dus/duc
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 3, 11), (partial[1][0:od], ranger_c))[0]
    #dus/dup
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 6, 8), (partial[2][0:od], ranger_c))[0]
    #dus/dTm
    J,_,_ = lax.scan(array_update_sep,(J, sep0 + 2, 12), (partial[3][0:od], ranger_c))[0]
    #dus/dTc
    J,_,_ = lax.scan(array_update_sep, (J, sep0 + 5, 9), (partial[4][0:od], ranger_c))[0]
    #dus/dTp
    J,_,_ = lax.scan(array_update_sep, (J, sep0 + 8, 6), (partial[5][0:od], ranger_c))[0]
    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4))
def build_dphie_o(J, partial, ad, pd, od):
    ranger_c = jnp.arange(0,od)
    p0 = ad + 1
    sep0 = 4*(pd+2) + 2*pd + 1 + p0
    # dphie/dum
    J, _, _ = lax.scan(array_update_sep, (J, sep0, 15), (partial[0][0:od], ranger_c))[0]
    # dphie/duc
    J, _, _ = lax.scan(array_update_sep, (J, sep0+3, 12), (partial[1][0:od], ranger_c))[0]
    #dphie/dup
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 6, 9), (partial[2][0:od], ranger_c))[0]
    #dphie/dphiem
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 1, 14), (partial[3][0:od], ranger_c))[0]
    #dphie/dphiec
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 4, 11), (partial[4][0:od], ranger_c))[0]
    #dphie/dphiep
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 7, 8), (partial[5][0:od], ranger_c))[0]
    #dphie/dTm
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 2, 13), (partial[6][0:od], ranger_c))[0]
    #dphie/dTc
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 5, 10), (partial[7][0:od], ranger_c))[0]
    #dphie/dTp
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 8, 7), (partial[8][0:od], ranger_c))[0]
    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4))
def build_dT_o(J, partial, ad, pd, od):
    ranger_c = jnp.arange(0,od)
    p0 = ad + 1
    sep0 = 4*(pd+2) + 2*pd + 1 + p0
    # dT/dum
    J,_,_ = lax.scan(array_update_sep, (J, sep0, 16), (partial[0][0:od], ranger_c))[0]
    # dT/duc
    J, _, _ = lax.scan(array_update_sep, (J, sep0+3, 13), (partial[1][0:od], ranger_c))[0]
    # dT/dup
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 6, 10), (partial[2][0:od], ranger_c))[0]
    # dT/dphiem
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 1, 15), (partial[3][0:od], ranger_c))[0]
    # dT/dphiep
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 7, 9), (partial[4][0:od], ranger_c))[0]
    # dT/dTm
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 2, 14), (partial[5][0:od], ranger_c))[0]
    # dT/dTc
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 5, 11), (partial[6][0:od], ranger_c))[0]
    # dT/dTp
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 8, 8), (partial[7][0:od], ranger_c))[0]
    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4))
def build_bc_n(J, bc, ad, pd, od, nd):
    p0 = ad + 1
    sep0 = 4*(pd+2) + 2*pd + 1 + p0
    n0 = p0 + 4 * (pd + 2) + 2 * pd + 3 * (od + 2) + 1
    row_u = jnp.array([11, 7, 8, 2, 17, 14, 15, 12, 17, 11])
    col_u = jnp.array([n0, n0 + 4, n0+3, n0 + 9,
                      sep0 + 3*od, sep0 + 3*od + 3,
                      sep0 + 3*od + 2, sep0 + 3*od + 5,
                      n0 + 4 + 6*(nd-1), n0 + 4 + 6*nd])
    J = J.at[row_u, col_u].set(bc['ce'])

    row_phie = jnp.array([11, 5, 18, 15, 13, 9, 19, 16, 10, 4, 17, 14, 15, 11])
    col_phie = jnp.array([n0 + 2, n0 + 8, sep0 + 3*od + 1, sep0 + 3*od + 4,
                         n0, n0 + 4, sep0 + 3*od, sep0 + 3*(od + 1),
                         n0 + 3, n0 + 9, sep0 + 3*od + 2, sep0 + 3*od + 5, n0 + 4 + 6*(nd-1) + 4, n0 + 4 + 6*nd +2])
    J = J.at[row_phie, col_phie].set(bc['phie'])

    row_phis = jnp.array([11, 5, 15, 11])
    col_phis = jnp.array([n0 + 1, n0 + 7, n0 + 4 + 6*(nd-1) + 3, n0 + 4 + 6*nd + 1])
    J = J.at[row_phis, col_phis].set(bc['phis'])

    row_T = jnp.array([18,15,11, 5, 15, 11, 10, 9])
    col_T = jnp.array([sep0 + 3*od + 2, sep0 + 3*od + 5, n0 + 3, n0 + 9,
                      n0 + 4 + 6*(nd-1) + 5, n0 + 4 + 6*nd + 3, n0 + 4 + 6*nd + 4, n0 + 4 + 6*nd + 4 + 1 ])
    J = J.at[row_T, col_T].set(bc['T'])
    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_dce_n(J, partial, ad, pd, od, nd):
    p0 = ad + 1
    n0 = p0 + 4 * (pd + 2) + 2 * pd + 3 * (od + 2) + 1
    ranger_p = jnp.arange(1, nd)
    ranger_m = jnp.arange(0, nd - 1)
    # dudTp
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 0), (partial[5][0:-1], ranger_p))[0]
    J = J.at[2, n0 + 4 + 6*nd].set(partial[5][-1])
    # dudum

    J, _, _ = lax.scan(array_update, (J, n0 + 4, 17), (partial[0][1:nd], ranger_m))[0]
    J = J.at[15, n0].set(partial[0][0])
    # duduc
    ranger_c = jnp.arange(0, nd)
    J, _, _ = lax.scan(array_update, (J, n0 + 4, 11), (partial[1][0:pd], ranger_c))[0]
    # dudup
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 6, 5), (partial[2][0:pd], ranger_c))[0]
    # dudTm
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 12), (partial[3][1:pd], ranger_m))[0]
    J = J.at[12, n0 + 3].set(partial[3][0])
    # dudTc
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 6), (partial[4][0:pd], ranger_c))[0]
    # dudj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 10), (partial[6][0:pd], ranger_c))[0]
    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_dj_n(J, partial, ad, pd, od, nd):
    ranger_c = jnp.arange(0, nd)
    p0 = ad + 1
    n0 = p0 + 4 * (pd + 2) + 2 * pd + 3 * (od + 2) + 1
    # djdj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 11), (partial[0][0:pd], ranger_c))[0]
    # djdu
    J, _, _ = lax.scan(array_update, (J, n0 + 4, 12), (partial[1][0:pd], ranger_c))[0]
    # djdT
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 7), (partial[2][0:pd], ranger_c))[0]
    # djdeta
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 2, 10), (partial[3][0:pd], ranger_c))[0]
    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_deta_n(J, partial, ad, pd, od, nd):
    p0 = ad + 1
    n0 = p0 + 4 * (pd + 2) + 2 * pd + 3 * (od + 2) + 1
    # detadeta
    ranger_c = jnp.arange(0, nd)
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 2, 11), (partial[0][0:pd], ranger_c))[0]
    # deta dphis
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 3, 10), (partial[1][0:pd], ranger_c))[0]
    # deta dphie
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 9), (partial[2][0:pd], ranger_c))[0]
    # detadT
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 8), (partial[3][0:pd], ranger_c))[0]
    # deta dj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 12), (partial[4][0:pd], ranger_c))[0]
    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_dphis_n(J, partial, ad, pd, od, nd):
    p0 = ad + 1
    n0 = p0 + 4 * (pd + 2) + 2 * pd + 3 * (od + 2) + 1
    ranger_m = jnp.arange(0, nd - 1)
    # dphis dphism
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 3, 17), (partial[0][1:nd], ranger_m))[0]
    J = J.at[17, n0 + 1].set(partial[0][0])
    # dphis dphisc
    ranger_c = jnp.arange(0, nd)
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 3, 11), (partial[1][0:nd], ranger_c))[0]
    # dphis dphisp
    ranger_p = jnp.arange(1, nd)
    J = J.at[7, n0 + 4 + 6*nd + 1 ].set(partial[2][nd-1])
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 3, 5), (partial[2][0:nd-1], ranger_p))[0]
    # dphis dj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 13), (partial[3][0:pd], ranger_c))[0]
    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_dphie_n(J, partial, ad, pd, od, nd):
    p0 = ad + 1
    n0 = p0 + 4 * (pd + 2) + 2 * pd + 3 * (od + 2) + 1
    # dphie/dum
    ranger_m = jnp.arange(0, nd - 1)
    J = J.at[19, n0].set(partial[0][0])
    J, _, _ = lax.scan(array_update, (J, n0 + 4, 21), (partial[0][1:nd], ranger_m))[0]
    # dphie/duc
    ranger_c = jnp.arange(0, nd)
    J, _, _ = lax.scan(array_update, (J,n0 + 4, 15), (partial[1][0:nd], ranger_c))[0]
    # dphie/dup
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 6, 9), (partial[2][0:nd], ranger_c))[0]
    # dphie/dphiem
    J = J.at[17, n0 + 2].set(partial[3][0])  # check
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 17), (partial[3][1:nd], ranger_m))[0]
    # dphie/dphiec
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 11), (partial[4][0:nd], ranger_c))[0]
    # dphie/dphiep
    ranger_p = jnp.arange(1, nd)
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 5), (partial[5][0:nd - 1], ranger_p))[0]
    J = J.at[7, n0 + 4 + 6*nd + 2].set(partial[5][nd - 1])

    # dphiep/dTm
    J = J.at[16, n0 + 3].set(partial[6][0])
    J, _, _ = lax.scan(array_update, (J, 5 + 4 + n0, 16), (partial[6][1:nd], ranger_m))[0]

    # dphiep/dTc
    J, _, _ = lax.scan(array_update, (J, 5 + n0 + 4, 10), (partial[7][0:pd], ranger_c))[0]

    # dphiep/dTp
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 4), (partial[8][0:pd - 1], ranger_p))[0]
    J = J.at[6, n0 + 4 + 6*nd + 3].set(partial[8][pd - 1])

    # dphie/dj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 14), (partial[9][0:pd], ranger_c))[0]
    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_dT_n(J, partial, ad, pd, od, nd):
    p0 = ad + 1
    n0 = p0 + 4 * (pd + 2) + 2 * pd + 3 * (od + 2) + 1
    # dT/dum
    ranger_m = jnp.arange(0, nd - 1)
    J = J.at[20, n0].set(partial[0][0])
    J, _, _ = lax.scan(array_update, (J, n0 + 4, 22), (partial[0][1:nd], ranger_m))[0]

    # dT/duc
    ranger_c = jnp.arange(0, nd)
    J, _, _ = lax.scan(array_update, (J, n0 + 4, 16), (partial[1][0:nd], ranger_c))[0]

    # dT/dup
    J, _, _ = lax.scan(array_update, (J, 4 + 6 + n0, 10), (partial[2][0:nd], ranger_c))[0]

    # dT/phiem
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 18), (partial[3][1:nd], ranger_m))[0]
    J = J.at[18, n0 + 2].set(partial[3][0])

    # dT/dphiep
    ranger_p = jnp.arange(1, pd)
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 6), (partial[4][0:nd - 1], ranger_p))[0]
    J = J.at[8, n0 + 4 + 6*nd + 2 ].set(partial[4][nd - 1])

    # dT/dphism
    J, _, _ = lax.scan(array_update, (J,n0 + 4 + 3, 19), (partial[5][1:nd], ranger_m))[0]
    J = J.at[19, n0 + 1].set(partial[5][0])

    # dT/dphisp
    J, _, _ = lax.scan(array_update, (J, 4 + 3 + n0, 7), (partial[6][0:nd - 1], ranger_p))[0]
    J = J.at[9, n0 + 4 + 6*nd + 1 ].set(partial[6][nd - 1])

    # dT/dTm
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 17), (partial[7][1:nd], ranger_m))[0]
    J = J.at[17, n0 + 3].set(partial[7][0])

    # dT/dTc
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 11), (partial[8][0:nd], ranger_c))[0]

    # dT/dTp
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 5), (partial[9][0:nd - 1], ranger_p))[0]
    J = J.at[7, n0 + 4 + 6*nd + 3].set(partial[9][nd])

    # dT/dj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 15), (partial[10][0:nd], ranger_c))[0]
    # dT/deta
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 2, 14), (partial[11][0:nd], ranger_c))[0]
    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4,5,6,))
def build_bc_cc(J, bc, ad, pd, od, nd, zd):
    p0 = ad + 1
    sep0 = 4 * (pd + 2) + 2 * pd + 1 + p0
    n0 = p0 + 4*(pd+2) + 2*pd  + 3*(od+2) + 1
    row_dTa = jnp.array([11, 10, 16, 11, 12, 5])
    col_dTa = jnp.array([0, 1, ad, 4 + p0, 3 + p0, 10 + p0])

    J = J.at[row_dTa, col_dTa].set(bc['acc'])
    row_dTz = jnp.array([16, 12, 11, 10, 12, 11])
    col_dTz = jnp.array([n0 + 4 + 6*(nd-1)+5, n0 + 4 + 6*nd + 3,n0 + 4 + 6*nd + 4, n0 + 4 + 6*nd + 5,
                        n0 + 4 + 6*nd + 4 + zd,n0 + 4 + 6*nd + 4 + zd+1 ])
    J = J.at[row_dTz, col_dTz].set(bc['zcc'])
    return J

@jax.jit
def array_update_acc(state, update_element):
    element, ind = update_element
    J, start_index, row = state
    return (J.at[row, start_index + ind].set(element), start_index, row), ind

# @functools.partial(jax.jit, static_argnums=(2,))
def build_dT_a(J, partial, ad):
    ranger = jnp.arange(0, ad)
    # dT/dTm
    J,_,_ = lax.scan(array_update_acc, (J, 0, 12), (partial[0][0:ad], ranger) )[0]
    # dT/dTc
    J,_,_ = lax.scan(array_update_acc, (J, 1, 11), (partial[1][0:ad], ranger))[0]
    # dT/dTp
    ranger_p = jnp.arange(0, ad-1)
    J,_,_ = lax.scan(array_update_acc, (J, 2, 10), (partial[2][0:ad-1], ranger_p))[0]
    J = J.at[6,ad + 1 + 4].set(partial[2][ad-1])

    return J

# @functools.partial(jax.jit, static_argnums=(2,3,4,5,6,))
def build_dT_z(J, partial, ad, pd, od, nd, zd):
    p0 = ad + 1
    sep0 = 4 * (pd + 2) + 2 * pd + 1 + p0
    n0 = p0 + 4 * (pd + 2) + 2 * pd + 3 * (od + 2) + 1
    ranger = jnp.arange(0, zd)
    # dT/dTm
    J,_,_ = lax.scan(array_update_acc, (J, n0 + 4 + 6*nd + 4, 12), (partial[0][0:zd], ranger) )[0]
    # dT/dTc
    J,_,_ = lax.scan(array_update_acc, (J, n0 + 4 + 6*nd + 5, 11), (partial[1][0:zd], ranger))[0]
    # dT/dTp
    J,_,_ = lax.scan(array_update_acc, (J,  n0 + 4 + 6*nd + 6, 10), (partial[2][0:ad], ranger))[0]

    return J
