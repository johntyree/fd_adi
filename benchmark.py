#!/usr/bin/env python
# coding: utf8
"""Benchmarking the FiniteDifferenceEngine."""

import sys
import argparse

from FiniteDifference import utils

from FiniteDifference.Grid import Grid
from FiniteDifference.FiniteDifferenceEngineGPU import FiniteDifferenceEngineADI as FDE_ADI_GPU
from FiniteDifference.FiniteDifferenceEngineGPU import HestonFiniteDifferenceEngine as HestonFDEGPU
from FiniteDifference.heston import HestonOption, HestonBarrierOption, hs_call_vector, HestonFiniteDifferenceEngine
from FiniteDifference.blackscholes import BlackScholesOption

from FiniteDifference.visualize import fp

Opt = HestonOption
# Opt = HestonBarrierOption

DefaultHeston = Opt( spot=100
                 , strike=100
                 , interest_rate=0.03
                 , volatility = 0.2
                 , tenor=1.0
                 , mean_reversion = 1
                 , mean_variance = 0.12
                 , vol_of_variance = 0.3
                 , correlation = 0.4
                 )

DefaultHeston = Opt( spot=100
                 , strike=100
                 , interest_rate=0.03
                 , volatility = 0.3
                 , tenor=2.0
                 , mean_reversion = 2
                 , mean_variance = 0.1
                 , vol_of_variance = 0.6
                 , correlation = 0.2
                 )


DefaultHeston = Opt( spot=100
                 , strike=130
                 , interest_rate=-0.1
                 , volatility = 0.43
                 , tenor=2.75
                 , mean_reversion = 4.2
                 , mean_variance = 0.21
                 , vol_of_variance = 0.5
                 , correlation = -0.3
                 )

# DefaultHeston = HestonOption(spot=100 , strike=100 , interest_rate=0.03 , volatility = 0.2
                # , tenor=1.0
                # , mean_reversion = 1
                # , mean_variance = 0.12
                # , vol_of_variance = 0.3
                # , correlation = 0.4
                # )

H = DefaultHeston
H.top = (False, 170)
print H
# H.bottom = (False, 85)

# trims = (H.strike * .2 < spots) & (spots < H.strike * 2.0)
# trimv = (0.0 < vars) & (vars < 1)  # v0*2.0)
# trims = slice(None)
# trimv = slice(None)


def create_cpu(nspots=30, nvols=30):
    F = HestonFiniteDifferenceEngine(H, nspots=nspots,
                                        nvols=nvols, spotdensity=10, varexp=4,
                                        var_max=12, verbose=False)
    F.init()
    F.operators[1].diagonalize()
    return F

def create_gpu(nspots=30, nvols=30):
    F = HestonFDEGPU(H, nspots=nspots,
                     nvols=nvols, spotdensity=10, varexp=4,
                     var_max=12, verbose=False)
    F.make_operator_templates()
    F.scale_and_combine_operators()
    return F


def run(dt, F=None, func=None, initial=None):
    if F is None:
        F = create()

    if func is None:
        func = 'hv'

    if initial is None:
        initial = F.grid.domain[0].copy()

    funcs = {
        'hv': lambda dt: F.solve_hundsdorferverwer(H.tenor/dt, dt, initial, 0.65),
        'i' : lambda dt: F.solve_implicit(H.tenor/dt, dt, initial),
        'd' : lambda dt: F.solve_douglas(H.tenor/dt, dt, initial, 0.65),
        # 'smooth': lambda dt: F.smooth(H.tenor/dt, dt, smoothing_steps=1, scheme=F.solve_hundsdorferverwer)
        'smooth': lambda dt: F.solve_smooth(H.tenor/dt, dt, initial, smoothing_steps=1)
    }
    labels = {
        'hv': "Hundsdorfer-Verwer",
        'i' : "Fully Implicit",
        'd' : "Douglas",
        'smooth': "Smoothed HV"
    }

    Vs = funcs[func](dt)
    return Vs


def read_args():
    parser = argparse.ArgumentParser(description="Run a benchmark")
    parser.add_argument('-s', '--scheme', metavar='scheme', choices="i,d,hv,s".split(','), default='hv')
    parser.add_argument('-n', '--steps', default=252, metavar='int', dest='n', type=int, help="Number of time steps")
    parser.add_argument('-ns', '--nspots', default=200, metavar='int', type=int, help="Number of spots")
    parser.add_argument('-nv', '--nvols', default=200, metavar='int', type=int, help="Number of volatilities")
    parser.add_argument('-gpu', action='store_const', dest='gpu', default=False, const=True)
    parser.add_argument('-cpu', action='store_const', dest='cpu', default=False, const=True)
    parser.add_argument('--no-run', action='store_const', dest='run', default=True, const=False)
    parser.add_argument('-mc', '--monte-carlo', metavar='int', type=int, dest='npaths', default=0)
    return parser.parse_args()


def main():
    opt = read_args()

    print opt.scheme, opt.nspots, opt.nvols, opt.n

    print
    if opt.cpu:
        utils.tic("CPU Create:")
        F = create_cpu(nspots=opt.nspots, nvols=opt.nvols)
        utils.toc()
        idx = F.idx
        if opt.run and opt.npaths:
            mc = F.option.monte_carlo()
            print mc
        if opt.run:
            print run(1.0/opt.n, F, opt.scheme)[idx]
        F.grid.reset()
    if opt.gpu:
        utils.tic("GPU Create 1:")
        FG = create_gpu(nspots=opt.nspots, nvols=opt.nvols)
        utils.toc()
        idx = FG.idx
        if opt.run:
            print run(1.0/opt.n, FG, opt.scheme, F.grid.domain[0])[idx],
            print

if __name__ == '__main__':
    main()
