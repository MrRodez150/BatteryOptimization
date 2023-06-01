import matplotlib.pyplot as plt
import jax.numpy as jnp

from settings import div_x_elec, div_x_sep, div_x_cc

def plot_pon(p, o, n, lp, lo, ln, y_lbl:str):

    xp = jnp.linspace(0, lp, int(div_x_elec)+2)
    xo = jnp.linspace(lp, lp+lo, int(div_x_sep)+2)
    xn = jnp.linspace(lp+lo, lp+lo+ln, int(div_x_elec)+2)
    fig = plt.figure()
    plt.plot(jnp.hstack([xp,xo,xn]),jnp.hstack([p, o, n]))
    plt.xlabel('x axis [m]')
    plt.ylabel(y_lbl)

    return fig

def plot_aponz(a, p, o, n, z, la, lp, lo, ln, lz, y_lbl:str):

    xa = jnp.linspace(0, la, int(div_x_cc)+2)
    xp = jnp.linspace(la, la+lp, int(div_x_elec)+2)
    xo = jnp.linspace(la+lp, la+lp+lo, int(div_x_sep)+2)
    xn = jnp.linspace(la+lp+lo, la+lp+lo+ln, int(div_x_elec)+2)
    xz = jnp.linspace(la+lp+lo+ln, la+lp+lo+ln+lz, int(div_x_cc)+2)
    fig = plt.figure()
    plt.plot(jnp.hstack([xa,xp,xo,xn,xz]),jnp.hstack([a, p, o, n, z]))
    plt.xlabel('x axis [m]')
    plt.ylabel(y_lbl)

    return fig
    
def plot_elec(y, lx, y_lbl):

    x = jnp.linspace(0, lx, len(y))
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel('x axis [m]')
    plt.ylabel(y_lbl)

    return fig

def plotTimeChange(t, y, y_lbl:str):

    fig = plt.figure()
    plt.plot(t,y)
    plt.xlabel('Time [s]')
    plt.ylabel(y_lbl)

    return fig


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:24:49 2021

@author: hanrach
"""

def plot_solid_conc(cmat_vec, N, M, elec):
        
    xp = elec.l*( jnp.arange(1,M+1) - 0.5 )/M
    yp = elec.Rp*( jnp.arange(1,N+1) - 0.5 )/N
    xx, yy = jnp.meshgrid(xp, yp);

    cmat = jnp.reshape(cmat_vec, [N+2, M], order="F")
    plt.figure()
    plt.contourf(xx,yy,cmat[1:N+1,:]); plt.colorbar()
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.show()
    
def plot_u(uvec_pe, uvec_sep, uvec_ne):
    plt.figure()
    plt.plot(jnp.hstack([uvec_pe, uvec_sep, uvec_ne]))
    plt.show()