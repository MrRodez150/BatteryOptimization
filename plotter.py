import matplotlib.pyplot as plt
import jax.numpy as jnp

from config import div_x_elec, div_x_sep, div_x_cc

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
