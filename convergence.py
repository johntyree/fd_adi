#!/usr/bin/env python
# coding: utf8
"""<+Module Description.+>"""

from __future__ import division

import sys
# import os
# import itertools as it

import pylab

from FiniteDifference import utils
from FiniteDifference.visualize import surface, wireframe
from FiniteDifference.heston import HestonOption, hs_call_vector, HestonFiniteDifferenceEngine
from FiniteDifference.blackscholes import BlackScholesOption, BlackScholesFiniteDifferenceEngine

from FiniteDifference.Grid import Grid
from FiniteDifference import FiniteDifferenceEngine as FD

# FD.DEBUG = True

from FiniteDifference.visualize import fp



DefaultBS = BlackScholesOption(spot=100
                 , strike=100
                 , interest_rate=0.03
                 , volatility = 0.2
                 , tenor=1.0
                 )

BS = DefaultBS

DefaultHeston = HestonOption(spot=100
                 , strike=100
                 , interest_rate=0.03
                 , volatility = 0.2
                 , tenor=1.0
                 , mean_reversion = 1
                 , mean_variance = 0.12
                 , vol_of_variance = 0.3
                 , correlation = 0.4
                 )
H = DefaultHeston


def error_surface2d(V, F, errors, label="", trim=True):
    a = F.grid_analytical
    res = V
    xs, ys = F.grid.mesh
    if trim:
        res, a, xs, ys = trim2d(V, F)
    p_absolute_error(res, a, xs, ys, label=label)

def p_absolute_error(V, analytical, spots, vars, marker_idx=0, label="", bad=False):
    surface(V - analytical, spots, vars)
    # wireframe(V - analytical, spots, vars)
    if bad:
        label += " - bad analytical!"
    else:
        label += " - $||V - V*||^\infty = %.2e$" % max(abs(V-analytical).flat)
    pylab.title("Error in Price (%s)" % label)
    pylab.xlabel("Var")
    pylab.ylabel("% of strike")
    pylab.show()

def p_price(V, analytical, spots, vars, marker_idx=0, label="", bad=False):
    surface(V, spots, vars)
    # wireframe(V - analytical, spots, vars)
    if bad:
        label += " - bad analytical!"
    pylab.title("Price (%s)" % label)
    pylab.xlabel("Var")
    pylab.ylabel("% of strike")
    pylab.show()


p = p_absolute_error
pp = p_price


def dsdv_convergence(dt=1.0/1000, nv=500, ns=500, func=None):
    global F
    errors = []
    for i in range(4,10):
    # for i in range(5,6):
        nvols = nv
        nspots = ns
        if nv is None:
            nvols = 2**i
        if ns is None:
            nspots = 2**i
        schemes = {}
        schemes[(1,)] = [{"scheme": "forward"}]

        F = HestonFiniteDifferenceEngine(H, schemes=schemes, nspots=nspots,
                                         nvols=nvols, spotdensity=10, varexp=4,
                                         var_max=12, flip_idx_spot=True,
                                         flip_idx_var=True, verbose=False,
                                         force_exact=False)
        if func is None:
            func = 'hv'
        funcs = {
            'hv': lambda dt: F.solve_hundsdorferverwer(H.tenor/dt, dt, theta=0.65),
            'i' : lambda dt: F.solve_implicit(H.tenor/dt, dt),
            'd' : lambda dt: F.solve_implicit(H.tenor/dt, dt, theta=0.65),
            'smooth': lambda dt: F.smooth(H.tenor/dt, dt, smoothing_steps=1, scheme=F.solve_hundsdorferverwer)
        }
        labels = {
            'hv': "Hundsdorfer-Verwer",
            'i' : "Fully Implicit",
            'd' : "Douglas",
            'smooth': "Smoothed HV"
        }
        Vs = funcs[func](dt)
        # Vs = F.solve_hundsdorferverwer(H.tenor/dt, dt, theta=0.8)

        xs = F.spots
        ys = F.vars
        trimx = (0.0 * H.spot <= xs) & (xs <= 2.0*H.spot)
        trimy = ys <= 1.0
        tr = lambda x: x[trimx, :][:, trimy]

        xst = xs[trimx]
        yst = ys[trimy]
        res = tr(Vs)
        a = tr(F.grid_analytical)
        bad = F.BADANALYTICAL
        price_err = F.price - H.analytical
        inf_norm = max(abs(res-a).flat)
        norm2 = pylab.sqrt(sum(((res-a)**2).flat))
        err = norm2
        if nv is None:
            dx = max(ys) / nvols
        else:
            dx = max(xs) / nspots
        label = "avg $dx = %s$" % dx
        # err = abs(price_err[0,0])
        # print dt, price_err, err
        # pp(Vs, F.grid_analytical, xs, ys, label=label, bad=bad)
        # p(Vs, F.grid_analytical, xs, ys, label=label, bad=bad)
        p(res, a, xst, yst, label=label, bad=bad)
        errors.append((dx, err))
    print errors
    vals, errs = zip(*errors)
    pylab.plot(vals, errs)
    pylab.plot(vals, errs, 'ro')
    pylab.xlabel("dx")
    pylab.ylabel("Error")
    pylab.show()
    pylab.loglog(vals, errs)
    pylab.loglog(vals, errs, 'ro')
    pylab.xlabel("dx")
    pylab.ylabel("Error")
    pylab.show()


    return errors

class ConvergenceTester(object):
    funcs = {
        'hv': lambda F, dt: F.solve_hundsdorferverwer(F.option.tenor/dt, dt, theta=0.65, numpy=False),
        'i' : lambda F, dt: F.solve_implicit(F.option.tenor/dt, dt, numpy=False),
        'd' : lambda F, dt: F.solve_douglas(F.option.tenor/dt, dt, theta=0.65, numpy=False),
        'smooth': lambda F, dt: F.smooth(F.option.tenor/dt, dt, smoothing_steps=1, scheme=F.solve_hundsdorferverwer)
    }
    labels = {
        'hv': "Hundsdorfer-Verwer",
        'i' : "Fully Implicit",
        'd' : "Douglas",
        'smooth': "Smoothed HV"
    }

    def __init__(self, option, engine, engine_kwargs={}, **kwargs):
        self.option = option
        self.engine = engine
        self.engine_kwargs = engine_kwargs
        def noop(*x): return None
        kwargs.setdefault('display', noop)
        kwargs.setdefault('error_func', noop)
        kwargs.setdefault('update_kwargs', noop)
        kwargs.setdefault('func', 'hv')
        kwargs.setdefault('label', '')
        kwargs.update(self.labels)
        self.kwargs = kwargs

    def new(self):
        return self.engine(self.option, force_exact=False, verbose=False, **self.engine_kwargs)

    def dt(self):
        self.F = self.new()
        d = dict(self.kwargs)
        d['func_label'] = self.labels[d['func']]
        d['dim'] = self.F.grid.shape
        msg = ("Running convergence test in dt using %(func_label)s scheme. "
               "dim = %(dim)s." % d)
        print msg
        self.mode = 'dt'
        self.errors = []
        min_i, max_i = d.get('min_i', 1), d.get('max_i', 11)
        for i in range(min_i,max_i):
            dt = self.kwargs['update_kwargs'](self, i)
            F = self.new()
            V = self.funcs[self.kwargs['func']](F, dt)
            self.errors.append((dt, self.kwargs["error_func"](V, F)))
            self.kwargs['display'](V, F, self.errors, label=self.kwargs["label"])
        return self.errors

    def dx(self):
        self.F = self.new()
        self.mode = 'dx'
        self.errors = []
        d = dict(self.kwargs)
        d['func_label'] = self.labels[d['func']]
        msg = ("Running convergence test in dx using %(func_label)s scheme. "
               "dt = %(dt)s." % d)
        print msg
        for i in range(3,10):
            dx = self.kwargs['update_kwargs'](self, i)
            F = self.new()
            dx = self.kwargs['update_kwargs'](self, i)
            V = self.funcs[self.kwargs['func']](F, self.kwargs["dt"])
            self.errors.append((dx, self.kwargs["error_func"](V, F)))
            self.kwargs['display'](V, F, self.errors, label=self.kwargs["label"])
        return self.errors

    def show_convergence(self):
        vals, errs = zip(*self.errors)
            # print dt, price_err, err
            # pp(Vs, F.grid_analytical, xs, ys, label="$dt = %s$" % dt, bad=bad)
            # p(Vs, F.grid_analytical, xs, ys, label="$dt = %s$" % dt, bad=bad)
            # p(res, a, xst, yst, label="$dt = %s$" % dt, bad=bad)
        pylab.plot(vals, errs)
        pylab.plot(vals, errs, 'ro')
        pylab.xlabel(self.mode)
        pylab.ylabel("Error")
        pylab.show()
        pylab.loglog(vals, errs)
        pylab.loglog(vals, errs, 'ro')
        pylab.xlabel(self.mode)
        pylab.ylabel("Error")
        pylab.show()
        return vals, errs


def trim1d(V, F):
    xs = F.grid.mesh[0]
    trimx = (0.0 * F.option.spot <= xs) & (xs <= 2.0*F.option.spot)
    tr = lambda x: x[trimx]
    res = tr(V)
    a = tr(F.grid_analytical)
    return res, a, xs

def trim2d(V, F):
    xs = F.grid.mesh[0]
    ys = F.grid.mesh[1]
    trimx = (0.0 * F.option.spot <= xs) & (xs <= 2.0*F.option.spot)
    trimy = ys <= 1.0
    tr = lambda x: x[trimx, :][:, trimy]
    res = tr(V)
    a = tr(F.grid_analytical)
    return res, a, xs[trimx], ys[trimy]

def error1d(V, F):
    res, a, xs = trim1d(V, F)
    inf_norm = max(abs(res-a).flat)
    norm2 = pylab.sqrt(sum(((res-a)**2).flat))
    return norm2

def error2d(V, F):
    res, a, xs, ys = trim2d(V, F)
    inf_norm = max(abs(res-a).flat)
    norm2 = pylab.sqrt(sum(((res-a)**2).flat))
    return norm2


def dt_convergence(nspots=100, nvols=100, func=None):
    # global F, vals, errs
    errors = []
    for i in range(1,11):
        F = HestonFiniteDifferenceEngine(H, schemes=schemes, nspots=nspots,
                                         nvols=nvols, spotdensity=10, varexp=4,
                                         var_max=12, flip_idx_spot=True,
                                         flip_idx_var=True, verbose=False)
        if func is None:
            func = 'hv'
        dt = 1.0 / 2.0**i
        Vs = funcs[func](dt)

        xs = F.spots
        ys = F.vars
        trimx = (0.0 * H.spot <= xs) & (xs <= 2.0*H.spot)
        trimy = ys <= 1.0
        tr = lambda x: x[trimx, :][:, trimy]

        xst = xs[trimx]
        yst = ys[trimy]
        res = tr(Vs)
        a = tr(F.grid_analytical)
        bad = F.BADANALYTICAL
        price_err = F.price - H.analytical
        inf_norm = max(abs(res-a).flat)
        norm2 = pylab.sqrt(sum(((res-a)**2).flat))
        err = norm2
        # err = abs(price_err[0,0])
        # print dt, price_err, err
        # pp(Vs, F.grid_analytical, xs, ys, label="$dt = %s$" % dt, bad=bad)
        # p(Vs, F.grid_analytical, xs, ys, label="$dt = %s$" % dt, bad=bad)
        p(res, a, xst, yst, label="$dt = %s$" % dt, bad=bad)
        errors.append((dt, err))
    print errors
    vals, errs = zip(*errors)
    pylab.plot(vals, errs)
    pylab.plot(vals, errs, 'ro')
    pylab.xlabel("dt")
    pylab.ylabel("Error")
    pylab.show()
    pylab.loglog(vals, errs)
    pylab.loglog(vals, errs, 'ro')
    pylab.xlabel("dt")
    pylab.ylabel("Error")
    pylab.show()

    return errors

def main():
    if len(sys.argv) > 2:
        func = sys.argv[2]
    else:
        func=None
    m = 150
    if len(sys.argv) > 1:
        if sys.argv[1] == 'ds':
            dsdv_convergence(2**-6, ns=None, func=func)
        elif sys.argv[1] == 'dv':
            dsdv_convergence(2**-6, nv=None, func=func)
        else:
            dt_convergence(200, 200, func=func)
    else:
        dt_convergence(150, 150, func=func)

def _main():
    if len(sys.argv) > 2:
        func = sys.argv[2]
    else:
        func=None
    m = 150
    if len(sys.argv) > 1:
        if sys.argv[1] == 'ds':
            dsdv_convergence(2**-6, ns=None, func=func)
        elif sys.argv[1] == 'dv':
            dsdv_convergence(2**-6, nv=None, func=func)
        else:
            dt_convergence(200, 200, func=func)
    else:
        dt_convergence(150, 150, func=func)

if __name__ == '__main__':
    main()
