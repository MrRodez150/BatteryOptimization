import matplotlib.pyplot as plt
import jax.numpy as jnp

from settings import dxA, dxP, dxO, dxN, dxZ

def plot_pon(p, o, n, lp, lo, ln, y_lbl:str):

    xp = jnp.linspace(0, lp, int(dxP)+2)
    xo = jnp.linspace(lp, lp+lo, int(dxO)+2)
    xn = jnp.linspace(lp+lo, lp+lo+ln, int(dxN)+2)
    fig = plt.figure()
    plt.plot(jnp.hstack([xp,xo,xn]),jnp.hstack([p, o, n]))
    plt.xlabel('x axis [m]')
    plt.ylabel(y_lbl)

    return fig

def plot_aponz(a, p, o, n, z, la, lp, lo, ln, lz, y_lbl:str):

    xa = jnp.linspace(0, la, int(dxA)+2)
    xp = jnp.linspace(la, la+lp, int(dxP)+2)
    xo = jnp.linspace(la+lp, la+lp+lo, int(dxO)+2)
    xn = jnp.linspace(la+lp+lo, la+lp+lo+ln, int(dxN)+2)
    xz = jnp.linspace(la+lp+lo+ln, la+lp+lo+ln+lz, int(dxZ)+2)
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


def PlotTimeChange5vrs(t, V, T, Fn, On, Tn):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.spines.left.set_position(("axes", -0.2))
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    ax4.spines.right.set_position(("axes", 1.2))
    ax5 = ax1.twinx()
    ax5.spines.right.set_position(("axes", 1.4))
    
    
    ax1.set_xlabel('Time (s)')

    l1, = ax1.plot(t, T, label='Battery Temperature', color='dodgerblue')
    l2, = ax2.plot(t, V, label='Battery Voltage', color='red')
    l3, = ax3.plot(t, Fn, label='Ionic Flux (o/n)', color='orange')
    l4, = ax4.plot(t, On, label='Overpotential (o/n)', color='green')
    l5, = ax5.plot(t, Tn, label='Temperature (o/n)', color='blue')
    
    ax1.set_ylabel(r'Battery Temperature (K)')
    ax1.yaxis.label.set_color(l1.get_color())
    ax2.set_ylabel(r'Battery voltage (V)')
    ax2.yaxis.label.set_color(l2.get_color())
    ax3.set_ylabel(r'Interfacial Ionic Flux (mol/m$^2$s)')
    ax3.yaxis.label.set_color(l3.get_color())
    ax4.set_ylabel(r'Interfacial Overpotential (V)')
    ax4.yaxis.label.set_color(l4.get_color())
    ax5.set_ylabel(r'Interfacial Temperature (K)')
    ax5.yaxis.label.set_color(l5.get_color())

    return fig