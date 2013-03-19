# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division

import os
import sys
import time
import numpy as np
import argparse

import convergence as cv
import FiniteDifference as FD
from FiniteDifference.FiniteDifferenceEngineGPU import FiniteDifferenceEngineADI as FDEGPU
import FiniteDifference.visualize as vis
# figure_dir = "/home/john/Filing_Cabinet/thesis/thesis_cudafd/tex/figures/archive"
figure_dir = "/scratch/tyree/cudafd/src/fd_pricer/py_adi/data_convergence/figures"


fname = "temp"

def engineGPU(*args, **kwargs):
    F = FDEGPU(FD.heston.HestonFiniteDifferenceEngine(*args, **kwargs))
    return F

def engineCPU(*args, **kwargs):
    F = FD.heston.HestonFiniteDifferenceEngine(*args, **kwargs)
    return F

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
            dt=dt, min_i=4, max_i=9, scheme=scheme, error_func=cv.error2d,
            update_kwargs=update_kwargs)
    return ct.dx()


def read_args():
    parser = argparse.ArgumentParser(description="Run a convergence test")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    backend_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-s', '--scheme', metavar='scheme', choices="i,d,hv,s".split(','))
    parser.add_argument('-k', '--strike', metavar='strike', type=float)
    parser.add_argument('--min_i', default=2, metavar='int', type=int, help="Min iteration value (2**i)")
    parser.add_argument('--max_i', default=8, metavar='int', type=int, help="Max iteration value (2**i)")
    mode_group.add_argument('-dx', metavar='nspots/vols', nargs=2, type=int)
    mode_group.add_argument('-dt', metavar='timesteps', type=int)
    backend_group.add_argument('--gpu', action='store_const', dest='engine', const=engineGPU)
    backend_group.add_argument('--cpu', action='store_const', dest='engine', const=engineCPU)
    return parser.parse_args()


def main():
    opt = read_args()

    option = FD.heston.HestonOption(tenor=1, strike=opt.strike, volatility=0.2,
                                    mean_reversion=1, vol_of_variance=0.2,
                                    correlation=-0.7)
    # option = FD.heston.HestonBarrierOption(tenor=1, strike=strike, volatility=0.2,
                                        # mean_reversion=1, vol_of_variance=0.2,
                                        # correlation=-0.7, top=(False, 120.0))
    ctest = None
    if opt.dx is not None:
        ctester = cv.ConvergenceTester(option, opt.engine, {'nspots': opt.dx[0], 'nvols': opt.dx[1]},
                                    scheme=opt.scheme, max_i=10, error_func=cv.error2d)
        ctest = ctester.dt()
    else:
        ctest = rundx(option, opt.engine, 1./opt.dt, opt.min_i, opt.max_i, opt.scheme)

    ctest.reference_solution = ctest.result[ctest.mode]['error'][-1]
    # print ctest.error2d_direct()
    ctest.write()


if __name__ == '__main__':
    main()
