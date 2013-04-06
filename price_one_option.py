# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division

import os
import sys
import time
import numpy as np
import argparse

import cPickle

import convergence as cv
import FiniteDifference as FD
# from FiniteDifference.FiniteDifferenceEngineGPU import FiniteDifferenceEngineADI as FDEGPU
from FiniteDifference.FiniteDifferenceEngineGPU import HestonFiniteDifferenceEngine as FDEGPU
import FiniteDifference.visualize as vis
# figure_dir = "/home/john/Filing_Cabinet/thesis/thesis_cudafd/tex/figures/archive"
data_dir = "/scratch/tyree/cudafd/src/fd_pricer/py_adi/data_convergence/"
figure_dir = os.path.join(data_dir, "figures")


fname = "temp"

def engineCPU(*args, **kwargs):
    F = FD.heston.HestonFiniteDifferenceEngine(*args, **kwargs)
    return F
engineCPU.__repr__ = lambda: "engineCPU"

def engineGPU(*args, **kwargs):
    F = FDEGPU(*args, **kwargs)
    # g = engineCPU(*args, **kwargs)
    # F.from_host_FiniteDifferenceEngine(g)
    return F
engineGPU.__str__ = lambda: "engineGPU"

def save(*args, **kwargs):
    args = list(args)
    args[0] = os.path.join(figure_dir, str(int(time.time())) + '_' + args[0])
    savefig(*args, **kwargs)


def mc_error(price):
    def newf(V, F):
        return abs(F.price - price)
    return newf

    # err = ct.dt()
    # key = ('heston', mode, option.strike)
    # errors[key] = err

def rundx(option, engine, dt, min_i, max_i, scheme):
    def update_kwargs(self, i):
        self.engine_kwargs['nspots'] = 2**(i-1)
        self.engine_kwargs['nvols'] = 2**(i-1)
        # self.engine_kwargs['nvols'] = 256
        try:
            xs = self.F.grid.mesh[0]
        except AttributeError:
            return
        return (max(xs) - min(xs)) / 2**i

    ct = cv.ConvergenceTester(option, engine,
            {'force_exact': False, 'spotdensity': 10, 'varexp': 4},
            dt=dt, min_i=4, max_i=9, scheme=scheme,
            update_kwargs=update_kwargs)
    ct.kwargs['error_func'] = ct.selfreference
    return ct.dx()


def read_args():
    parser = argparse.ArgumentParser(description="Run a convergence test")
    parser.add_argument('--scheme', help='scheme', choices="i,d,hv,s".split(','))
    parser.add_argument('-s','--spot', metavar='FLOAT', type=float, default=100.0)
    parser.add_argument('-k', '--strike', metavar='FLOAT', help='strike', type=float, default=99.0)
    parser.add_argument('-t', '--tenor', metavar='FLOAT', help='tenor', type=float, default=1.0)
    parser.add_argument('-r', '--interest-rate', metavar='FLOAT', type=float, default=0.06)
    parser.add_argument('--mean-reversion', metavar='FLOAT', type=float, default=1.0)
    parser.add_argument('--variance', metavar='FLOAT', type=float, default=0.04)
    parser.add_argument('-o', '--vol-of-var', metavar='FLOAT', type=float, default=0.001)
    parser.add_argument('--mean-variance', metavar='FLOAT', type=float, default=-1.0)
    parser.add_argument('-p', '--correlation', metavar='FLOAT', help='Correlation', type=float, default=0.0)
    parser.add_argument('--min_i', default=2, metavar='INT', type=int, help="Min iteration value (2**i)")
    parser.add_argument('--max_i', default=10, metavar='INT', type=int, help="Max iteration value (2**i)")
    parser.add_argument('--top', type=float, default=None)
    parser.add_argument('--barrier', action='store_const', dest='option', const=FD.heston.HestonBarrierOption, default=FD.heston.HestonOption)
    parser.add_argument('-nx', required=True, metavar='int', help='nspots/vols', nargs=2, type=int)
    parser.add_argument('-nt', required=True, metavar='int', help='timesteps', type=int)
    parser.add_argument('-v', action='count', dest='verbose')
    parser.add_argument('--gpu', action='store_const', default=False, const=engineGPU)
    parser.add_argument('--cpu', action='store_const', default=False, const=engineCPU)
    opt = parser.parse_args()
    if opt.mean_variance == -1:
        opt.mean_variance = opt.variance
    if opt.verbose:
        print "Verbosity:", opt.verbose
    return opt


def new_engine(opt):
    option = opt.option(
        spot=opt.spot,
        strike=opt.strike,
        interest_rate=opt.interest_rate,
        variance=opt.variance,
        tenor=opt.tenor,
        mean_reversion=opt.mean_reversion,
        mean_variance=opt.mean_variance,
        vol_of_variance=opt.vol_of_var,
        correlation=opt.correlation)
    if opt.top:
        option.top = (False, opt.top)

    if opt.verbose:
        print repr(option)
        print

    engine = opt.engine(option,
        grid=None,
        spot_max=1500.0,
        spot_min=0.0,
        spots=None,
        vars=None,
        var_max=10.0,
        nspots=opt.nx[0],
        nvols=opt.nx[1],
        spotdensity=7.0,
        varexp=4.0,
        force_exact=True,
        flip_idx_var=False,
        flip_idx_spot=False,
        schemes=None,
        coefficients=None,
        boundaries=None,
        cache=True,
        verbose=opt.verbose,
        force_bandwidth=None
        )

    return engine

def run(opt):
    e = new_engine(opt)
    e.solve_smooth(opt.nt, opt.tenor / opt.nt)
    s = np.searchsorted(np.round(e.grid.mesh[0], decimals=6), e.option.spot)
    v = np.searchsorted(np.round(e.grid.mesh[1], decimals=6), e.option.variance.value)
    wanted, found = (opt.spot, opt.variance), (e.grid.mesh[0][s], e.grid.mesh[1][v])
    np.testing.assert_almost_equal(wanted, found,
                                   decimal=10,
                                   err_msg="We have the wrong indices! %s %s" % (wanted, found))
    try:
        e.grid.domain[-1] = e.gpugrid.to_numpy()
    except AttributeError:
        pass
    print e.grid.domain[-1][s,v],
    try:
        print e.option.analytical,
    except NotImplementedError:
        pass
    except AttributeError:
        pass
    print
    return e

def filestring(opt, e):
    moneyness = opt.spot / opt.strike
    ir = opt.interest_rate
    var = opt.variance
    volofvar = opt.vol_of_var
    corr = opt.correlation
    rev = opt.mean_reversion
    tenor = opt.tenor
    nspot, nvols = opt.nx
    idxs, idxv = e.idx
    ntime = opt.nt
    gpu = "gpu" if opt.gpu == engineGPU else "cpu"
    fn = os.path.join(data_dir, "_".join(map(str, [
        gpu,
        "moneyness", moneyness,
        "tenor", tenor,
        "ir", ir,
        "var", var,
        "volofvar", volofvar,
        "corr", corr,
        "rev", rev,
        "nspot", nspot,
        "nvols", nvols,
        "idxs", idxs,
        "idxv", idxv,
        "ntime", ntime
    ])))
    fn += ".txt"
    return fn

def main():
    opt = read_args()
    print repr(opt)
    if opt.cpu:
        opt.engine = opt.cpu
        res = run(opt)
        with open(filestring(opt, res), 'w') as fn:
            cPickle.dump([res.grid.mesh, res.grid.domain[-1]], fn, -1)
    if opt.gpu:
        opt.engine = opt.gpu
        res = run(opt)
        with open(filestring(opt, res), 'w') as fn:
            cPickle.dump([res.grid.mesh, res.grid.domain[-1]], fn, -1)

def oldmain():
    opt = read_args()

    ctest = None
    if opt.nx is not None:
        ctester = cv.ConvergenceTester(option, opt.engine, {'nspots': opt.nx[0], 'nvols': opt.nx[1]},
                                    scheme=opt.scheme, max_i=10)
        ctester.kwargs['error_func'] = ctester.selfreference
        ctest = ctester.dt()
    else:
        ctest = rundx(option, opt.engine, 1./opt.dt, opt.min_i, opt.max_i, opt.scheme)

    ctest.reference_solution = ctest.result[ctest.mode]['error'][-1]
    # print ctest.error2d_direct()
    ctest.write()
    return ctest


if __name__ == '__main__':
    main()
