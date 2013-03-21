#!/usr/bin/env python
# coding: utf8
"""<+Module Description.+>"""

from __future__ import division

import sys
import os
# import itertools as it

import pylab
import numpy as np
import pickle

from FiniteDifference import utils
from FiniteDifference.visualize import surface, wireframe
from FiniteDifference.heston import HestonOption, hs_call_vector, HestonFiniteDifferenceEngine
from FiniteDifference.blackscholes import BlackScholesOption, BlackScholesFiniteDifferenceEngine
from FiniteDifference.Option import MeanRevertingProcess

array = np.array

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


def import_ConvergenceTest(fn):
    with open(fn) as f:
        l = f.read()
    c = eval(l)
    for mode in c.result:
        for arr in c.result[mode]:
            c.result[mode][arr] = np.loads(c.result[mode][arr])
    return c

class ConvergenceTest(object):
    attrs = ('Type',
             'option',
             'mode',
             'backend',
             'reference_solution',
             'scheme',
             'mesh',
             'state',
             'result'
             )
    def __init__(self, **kwargs):
        kwargs.setdefault('mode', 'dt')
        kwargs.setdefault('result', None)
        kwargs['Type'] = 'ConvergenceTest'
        for a in self.attrs:
            assert a in kwargs, "Missing attr %s" % (a,)
        for attr, val in kwargs.items():
            setattr(self, attr, val)


    def write(self, fout=None):
        if fout is None:
            fout = self.make_file_name()
        # print pickle.dumps(self)
        with open(fout, 'w') as fout:
            fout.write(repr(self))


    def make_file_name(self):
        fn = os.path.join("data_convergence", "{type}_strike-{k}_{scheme}_{mode}_{backend}.py")
        return fn.format(type=self.option.Type,
                         k=self.option.strike,
                         scheme=self.scheme,
                         mode=self.mode,
                         backend=self.backend)


    def __str__(self):
        l = ["{attr}={val}".format(attr=attr, val=repr(getattr(self, attr))) for attr in self.attrs if attr != 'result']
        s = self.Type + "(" + ", ".join(l) + ')'
        return s


    def __repr__(self):
        for mode in self.result:
            for arr in self.result[mode]:
                self.result[mode][arr] = self.result[mode][arr].dumps()
        np.set_printoptions(threshold=np.inf)
        l = ["{attr}={val}".format(attr=attr, val=repr(getattr(self, attr))) for attr in self.attrs]
        s = self.Type + "(" + ", ".join(l) + ')'
        for mode in self.result:
            for arr in self.result[mode]:
                self.result[mode][arr] = np.loads(self.result[mode][arr])
        return s


    def error2d_direct(self, err_func=None):
        if err_func is None:
            err_func = lambda tst, ref: pylab.mean(abs(tst-ref).flat)
            # inf_norm = max(abs(res-a).flat)
            # norm2 = pylab.sqrt(sum(((res-a)**2).flat))
            # meanerr = pylab.mean(abs(res - a).flat)
        xs, ys = self.mesh
        trimx = (0.0 * self.option.spot <= xs) & (xs <= 2.0*self.option.spot)
        trimy = ys <= 1.0
        tr = lambda x: x[trimx, :][:, trimy]
        a = tr(self.reference_solution)
        errs = []
        for res in self.result[self.mode]['error']:
            res = tr(res)
            errs.append(err_func(res, a))
        return errs


class ConvergenceTester(object):
    schemes = {
        'hv': lambda F, dt: F.solve_hundsdorferverwer(H.tenor/dt, dt, F.grid.domain[-1], theta=0.65),
        'i' : lambda F, dt: F.solve_implicit(H.tenor/dt, dt, F.grid.domain[-1]),
        'd' : lambda F, dt: F.solve_implicit(H.tenor/dt, dt, F.grid.domain[-1], theta=0.65),
        'smooth': lambda F, dt: F.smooth(H.tenor/dt, dt, F.grid.domain[-1], smoothing_steps=1)
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
        def noop(*x, **y): return None
        kwargs.setdefault('display', noop)
        kwargs.setdefault('error_func', noop)
        kwargs.setdefault('scheme', 'hv')
        kwargs.setdefault('label', '')
        kwargs.update(self.labels)
        self.kwargs = kwargs

    def new(self):
        return self.engine(self.option, verbose=False, **self.engine_kwargs)

    def dt(self):
        if self.kwargs['scheme'] is None:
            print "No scheme. No-op"
            return
        self.kwargs.setdefault('update_kwargs', lambda self, i: 2**-i)
        F = self.new()
        self.kwargs['dx'] = tuple("%.1e" % ((max(mesh) - min(mesh)) / len(mesh)) for mesh in F.grid.mesh)
        d = dict(self.kwargs)
        d['scheme_label'] = self.labels[d['scheme']]
        self.title = ("Convergence test in dt using %(scheme_label)s scheme. "
               "dx = %(dx)s" % d)
        print self.title
        self.mode = 'dt'
        self.errors = []
        # self.domains = []
        self.meshes = []
        min_i, max_i = d.get('min_i', 1), d.get('max_i', 8)
        null = np.empty(max_i - min_i)
        ct = ConvergenceTest(
            option=self.option,
            backend='GPU',
            reference_solution=None,
            scheme=d['scheme'],
            mesh=np.array(F.grid.mesh).copy(),
            state='INIT',
            result={'dt':{'sequence':null.copy(), 'error':null.copy()}})
        for i in range(min_i, max_i):
            F.grid.reset()
            dt = self.kwargs['update_kwargs'](self, i)
            V = self.schemes[self.kwargs['scheme']](F, dt)
            ct.result['dt']['sequence'][i - min_i] = dt
            err = self.kwargs["error_func"](V, F)
            print "dt:", dt, "err:", err
            ct.result['dt']['error'][i - min_i] = err
        return ct

    def dx(self):
        if self.kwargs['scheme'] is None:
            print "No scheme. No-op"
            return
        if 'update_kwargs' not in self.kwargs:
            raise ValueError("No definition for update_kwargs given. We don't"
                             " know how to advance the simulation.")
        self.mode = 'dx'
        d = dict(self.kwargs)
        d['scheme_label'] = self.labels[d['scheme']]
        self.title = ("Convergence test in dx using %(scheme_label)s scheme. "
               "dt = %(dt).2e." % d)
        print self.title
        min_i, max_i = d.get('min_i', 5), d.get('max_i', 10)
        null = np.empty(max_i - min_i)
        dx = self.kwargs['update_kwargs'](self, min_i)
        F = self.new()
        ct = ConvergenceTest(
            option=self.option,
            backend='GPU',
            reference_solution=None,
            scheme=d['scheme'],
            mesh=F.grid.mesh,
            state='INIT',
            mode='dx',
            result={'dx':{'sequence':null.copy(), 'error':null.copy()}})
        for i in range(min_i, max_i):
            dx = self.kwargs['update_kwargs'](self, i)
            self.F = self.new()
            F = self.F
            dx = self.kwargs['update_kwargs'](self, i)
            V = self.schemes[self.kwargs['scheme']](F, self.kwargs["dt"])
            ct.result['dx']['sequence'][i - min_i] = dx
            err = self.kwargs["error_func"](V, F)
            print "dx:", dx, "err:", err
            ct.result['dx']['error'][i - min_i] = err
            # self.kwargs['display'](V, F, self.errors, label=self.kwargs["label"])
            # self.domains.append(V)
            # self.dxs.append(dx)
            del F, self.F
        return ct

    def show_convergence(self):
        vals, errs = zip(*self.errors)
            # print dt, price_err, err
            # pp(Vs, F.grid_analytical, xs, ys, label="$dt = %s$" % dt, bad=bad)
            # p(Vs, F.grid_analytical, xs, ys, label="$dt = %s$" % dt, bad=bad)
            # p(res, a, xst, yst, label="$dt = %s$" % dt, bad=bad)
        pylab.plot(vals, errs)
        pylab.plot(vals, errs, 'ro')
        pylab.title(self.title)
        pylab.xlabel(self.mode)
        pylab.ylabel("Error")
        pylab.show()
        pylab.loglog(vals, errs)
        pylab.loglog(vals, errs, 'ro')
        pylab.title(self.title)
        pylab.xlabel(self.mode)
        pylab.ylabel("Error")
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
    meanerr = pylab.mean(abs(res - a).flat)
    return meanerr

def error2d(V, F):
    res, a, xs, ys = trim2d(V, F)
    inf_norm = max(abs(res-a).flat)
    norm2 = pylab.sqrt(sum(((res-a)**2).flat))
    meanerr = pylab.mean(abs(res - a).flat)
    return meanerr


def error_price(V, F):
    return F.price - F.option.analytical

def dt_convergence(nspots=100, nvols=100, scheme=None):
    # global F, vals, errs
    errors = []
    schemes = {
        'hv': lambda dt: F.solve_hundsdorferverwer(H.tenor/dt, dt, F.grid.domain[-1], theta=0.65),
        'i' : lambda dt: F.solve_implicit(H.tenor/dt, dt, F.grid.domain[-1]),
        'd' : lambda dt: F.solve_implicit(H.tenor/dt, dt, F.grid.domain[-1], theta=0.65),
        'smooth': lambda dt: F.smooth(H.tenor/dt, dt, F.grid.domain[-1], smoothing_steps=1)
    }
    for i in range(1,11):
        schemes = {}
        schemes[(1,)] = [{"scheme": "forward"}]
        F = HestonFiniteDifferenceEngine(H, schemes=schemes, nspots=nspots,
                                         nvols=nvols, spotdensity=10, varexp=4,
                                         var_max=12, flip_idx_spot=True,
                                         flip_idx_var=True, verbose=False)
        if scheme is None:
            scheme = 'hv'
        dt = 1.0 / 2.0**i
        print dt, tuple("%.1e" % ((max(mesh) - min(mesh)) / len(mesh)) for mesh in F.grid.mesh)
        print map(len, F.grid.mesh), map(min, F.grid.mesh), map(max, F.grid.mesh)

        Vs = schemes[scheme](dt)

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
        scheme = sys.argv[2]
    else:
        scheme=None
    m = 150
    if len(sys.argv) > 1:
        if sys.argv[1] == 'ds':
            dsdv_convergence(2**-6, ns=None, scheme=scheme)
        elif sys.argv[1] == 'dv':
            dsdv_convergence(2**-6, nv=None, scheme=scheme)
        else:
            dt_convergence(200, 200, scheme=scheme)
    else:
        dt_convergence(150, 150, scheme=scheme)

if __name__ == '__main__':
    main()
