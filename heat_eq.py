#!/usr/bin/env python
# coding: utf8
"""Test of the heat equation."""


# This is dependent on @fst_deriv@ being set to 1 in FiniteDifferenceEngine.py
# Need to generalize that somehow

# import sys
# import os
# import itertools as it


import numpy as np
import FiniteDifference.FiniteDifferenceEngine as FD
from FiniteDifference.Grid import Grid
import FiniteDifference.visualize as vis
from FiniteDifference.visualize import fp
import scipy.stats

FD.DEBUG = False

xmin = 0.0
xmax = np.pi

nx = 10.0+1
nt = 1000.0


t = 2.0
start = t*0.0
dt = t/nt
dx = (xmax-xmin)/nx

D = 1

def just_bounds(t, *x):
    c = x[0]*0
    c[0] = 1
    c[-1] = 1
    # print c
    # return c+1
    return c


coeffs = {}
# coeffs[(0,)] = just_bounds
coeffs[(0,0)] = lambda t, *x: x[0]*0+D**2
# coeffs[(1,1)] = lambda t, *x: x[1]*0+D

bounds = {}
# bounds[(0,)] = ((0, lambda t, *x: 0), (0, lambda t, *x: 0))
bounds[(0,0)] = ((None, lambda t, *x: 0), (None, lambda t, *x: 0))
# bounds[(0,0)] = ((None, lambda t, *x: 5), (None, lambda t, *x: 1))
# bounds[(1,1)] = ((None, lambda t, *x: 0), (None, lambda t, *x: 0))


def np_grad(V, n, dt, bounds=True):
    V = V.copy()
    for i in range(n):
        dvdx, dvdy = np.gradient(V)
        d2vdx2 = np.gradient(dvdx)[0] * coeffs[(0,0)](i, i, i)
        # d2vdy2 = np.gradient(dvdy)[1] * coeffs[(1,1)](i, i, i)

        # V += dt * (d2vdx2 + d2vdy2)
        if bounds:
            V[:,0]  = 0
            V[:,-1] = 0
            # V[0,:]  = 0
            # V[-1,:] = 0
    return V


def ref(t, x):
    s = np.sum(-(-1.0)**n * 1.0/n**2 * np.cos(n*x) * np.exp(-n**2*t)
            for n in range(1, 100))
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
    return np.pi**2 / 3 - 4 * s

def build_FDE_1D(bounds=bounds):
    def init(x):
        return x**2
        # mesh = x
        # return heat_kernel_nD(2, 0, mesh, start)

    G = Grid(mesh=(np.linspace(xmin, xmax, nx),),
             initializer=init)

    # vis.lineplot(G.domain[-1], *G.mesh)

    F = FD.FiniteDifferenceEngineADI(
        G, coefficients=coeffs, boundaries=bounds)
    return F


def build_FDE_2D(bounds=bounds):
    def init(x, y):
        mesh = np.sqrt(x[:,np.newaxis]**2+y**2)
        return heat_kernel_nD(2, 0, mesh, dt)

    G = Grid(mesh=(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21)),
             initializer=init)

    # vis.surface(G.domain[-1], *G.mesh)

    F = FD.FiniteDifferenceEngineADI(
        G, coefficients=coeffs, boundaries=bounds)
    return F


def plt(n):
    global F1, F2, F1b, F2b, Fd, Fdb, Fh, Fhb, V, Vb, A
    a = {}
    a['i'] = (F1.grid.domain[-1], Fh.grid.mesh[0])
    a['d'] = (Fd.grid.domain[-1], Fh.grid.mesh[0])
    # a['db'] = (Fdb.grid.domain[-1], Fh.grid.mesh[0])
    a['h'] = (Fh.grid.domain[-1], Fh.grid.mesh[0])
    # a['hb'] = (Fhb.grid.domain[-1], Fh.grid.mesh[0])
    # a['idb'] = (F1.grid.domain[-1] - Fdb.grid.domain[-1], Fh.grid.mesh[0])
    # a['ihb'] = (F1.grid.domain[-1] - Fhb.grid.domain[-1], Fh.grid.mesh[0])
    # a['dbhb'] = (Fdb.grid.domain[-1] - Fhb.grid.domain[-1], Fh.grid.mesh[0])
    # a['v'] = (V, Fh.grid.mesh[0])
    # a['vb'] = (Vb, Fh.grid.mesh[0])
    a['a'] = (A, Fh.grid.mesh[0])
    a['a0'] = (Fh.grid.domain[0], Fh.grid.mesh[0])
    a['ha'] = (Fh.grid.domain[-1] - A, Fh.grid.mesh[0])
    vis.lineplot(*a[n])
    vis.pylab.show()


def srf(n):
    global F1, F2, F1b, F2b, Fd, Fdb, Fh, Fhb, V, Vb, A
    a = {}
    a['i'] = (F1.grid.domain[-1], Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['d'] = (Fd.grid.domain[-1], Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['db'] = (Fdb.grid.domain[-1], Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['h'] = (Fh.grid.domain[-1], Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['hb'] = (Fhb.grid.domain[-1], Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['idb'] = (F1.grid.domain[-1] - Fdb.grid.domain[-1], Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['ihb'] = (F1.grid.domain[-1] - Fhb.grid.domain[-1], Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['dbhb'] = (Fdb.grid.domain[-1] - Fhb.grid.domain[-1], Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['v'] = (V, Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['vb'] = (Vb, Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['ab'] = (A, Fh.grid.mesh[0], Fh.grid.mesh[1])
    a['hbab'] = (Fhb.grid.domain[-1] - A, Fh.grid.mesh[0], Fh.grid.mesh[1])
    vis.surface(*a[n])


def heat_kernel_nD(n, pulse, mesh, t):
    return 1 / (4 * np.pi * t)**(n/2.0) * np.exp(-abs(pulse-mesh)**2/(4*t))

def main():
    """Run main."""
    global F1, F2, F1b, F2b, Fd, Fdb, Fh, Fhb, V, Vb, A
    F1 = build_FDE_1D()
    F1.solve_implicit(nt-(start/dt), dt)
    # F2 = build_FDE_2D()
    # F2.solve_implicit2(200, 1.0/200)

    Fi = build_FDE_1D()
    Fi.solve_implicit(nt-(start/dt), dt)

    Fd = build_FDE_1D()
    Fd.solve_douglas(nt-(start/dt), dt, theta=1)

    Fh = build_FDE_1D()
    Fh.solve_hundsdorferverwer(nt-(start/dt), dt, theta=1)

    F = build_FDE_1D()
    # V = np_grad(F.grid.domain[0], 200, 1.0/200)
    # Vb = np_grad(F.grid.domain[0], 200, 1.0/200, bounds=False)

    # A = heat_kernel_nD(1, 0, np.linspace(xmin,xmax,nx), t)
    A = ref(2, F.grid.mesh[0])

    plt('a')
    plt('i')
    plt('h')

    return 0

if __name__ == '__main__':
    main()
