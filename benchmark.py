#!/usr/bin/env python
# coding: utf8
"""Benchmarking the FiniteDifferenceEngine."""

import sys
# import os
# import itertools as it

from FiniteDifference import utils

from FiniteDifference.heston import HestonOption, hs_call_vector, HestonFiniteDifferenceEngine
from FiniteDifference.blackscholes import BlackScholesOption

from FiniteDifference.Grid import Grid

from FiniteDifference.FiniteDifferenceEngineGPU import FiniteDifferenceEngineADI as FDE_ADI_GPU

# FD.DEBUG = True

from FiniteDifference.visualize import fp


DefaultHeston = HestonOption( spot=100
                 , strike=100
                 , interest_rate=0.03
                 , volatility = 0.2
                 , tenor=1.0
                 , mean_reversion = 1
                 , mean_variance = 0.12
                 , vol_of_variance = 0.041
                 , correlation = 0.4
                 )

H = DefaultHeston

# trims = (H.strike * .2 < spots) & (spots < H.strike * 2.0)
# trimv = (0.0 < vars) & (vars < 1)  # v0*2.0)
# trims = slice(None)
# trimv = slice(None)

# Does better without upwinding here

def create(nspots=30, nvols=30):
    # schemes[(1,)] = [{"scheme": "forward"}]

    F = HestonFiniteDifferenceEngine(H, nspots=nspots,
                                        nvols=nvols, spotdensity=10, varexp=4,
                                        var_max=12, verbose=False)
    F.init()
    F.operators[1].diagonalize()
    return F

def run(F=None, func=None, initial=None):
    if F is None:
        F = create()

    if func is None:
        func = 'hv'

    if initial is None:
        initial = F.grid.domain[0].copy()

    funcs = {
        'hv': lambda dt: F.solve_hundsdorferverwer(H.tenor/dt, dt, initial, 0.65),
        'i' : lambda dt: F.solve_implicit(H.tenor/dt, dt, initial),
        'd' : lambda dt: F.solve_implicit(H.tenor/dt, dt, initial, 0.65),
        # 'smooth': lambda dt: F.smooth(H.tenor/dt, dt, smoothing_steps=1, scheme=F.solve_hundsdorferverwer)
        'smooth': lambda dt: F.smooth(H.tenor/dt, dt, initial, smoothing_steps=1)
    }
    labels = {
        'hv': "Hundsdorfer-Verwer",
        'i' : "Fully Implicit",
        'd' : "Douglas",
        'smooth': "Smoothed HV"
    }

    dt = 1.0 / 2.0**10
    Vs = funcs[func](dt)

    return Vs

def main():
    if len(sys.argv) > 1:
        func = sys.argv[1]
    else:
        func=None

    if len(sys.argv) > 2:
        nspots = int(sys.argv[2])
    else:
        nspots = 300
    if len(sys.argv) > 3:
        nvols = int(sys.argv[3])
    else:
        nvols = 300

    F = create(nspots=nspots, nvols=nvols)
    idx = F.idx
    FG = FDE_ADI_GPU(F)
    print run(F, 'hv')[idx]
    F.grid.reset()
    print run(FG, 'hv', F.grid.domain[0])[idx]

if __name__ == '__main__':
    main()
