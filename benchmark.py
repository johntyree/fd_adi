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
                 , correlation = 0.0
                 )

H = DefaultHeston

# trims = (H.strike * .2 < spots) & (spots < H.strike * 2.0)
# trimv = (0.0 < vars) & (vars < 1)  # v0*2.0)
# trims = slice(None)
# trimv = slice(None)

# Does better without upwinding here

def create(nspots=100, nvols=100):
    schemes = {}
    schemes[(1,)] = [{"scheme": "forward"}]

    F = HestonFiniteDifferenceEngine(H, schemes=schemes, nspots=nspots,
                                        nvols=nvols, spotdensity=10, varexp=4,
                                        var_max=12, flip_idx_spot=True,
                                        flip_idx_var=True, verbose=False)
    return F

def run(F=None, func=None):
    if F is None:
        F = create()

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

    dt = 1.0 / 2.0**10
    Vs = funcs[func](dt)

    return

def main():
    if len(sys.argv) > 1:
        func = sys.argv[1]
    else:
        func=None

    if len(sys.argv) > 2:
        nspots = int(sys.argv[2])
    else:
        nspots = 100
    if len(sys.argv) > 3:
        nvols = int(sys.argv[3])
    else:
        nvols = 100

    F = create(nspots=nspots, nvols=nvols)
    run(F)

if __name__ == '__main__':
    main()
