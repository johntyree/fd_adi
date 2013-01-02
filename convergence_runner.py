# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division

import os
import sys
import time
import numpy as np

import convergence as cv
import FiniteDifference as FD
import FiniteDifference.visualize as vis
figure_dir = "/home/john/Filing_Cabinet/thesis/thesis_cudafd/tex/figures/archive"


fname = "temp"


try:
    func = sys.argv[1]
    mode = sys.argv[2]
    if mode == 'dt':
        nspots = int(sys.argv[3])
        nvols = int(sys.argv[4])
    if mode == 'dx':
        dt = float(sys.argv[3])
except IndexError:
    print sys.argv[0], "<func> <mode> (<nspots> <nvols> | <dt>)"
    sys.exit(1)



def save(*args, **kwargs):
    args = list(args)
    args[0] = os.path.join(figure_dir, str(int(time.time())) + '_' + args[0])
    savefig(*args, **kwargs)

def rundt():
    engine = FD.heston.HestonFiniteDifferenceEngine
    ct = cv.ConvergenceTester(option, engine, {'nspots': nspots, 'nvols': nvols},
                                func=func, max_i=7, error_func=cv.error2d)
    err = ct.dt()
    key = ('heston', mode, option.strike)
    errors[key] = err

def rundx():
    def update_kwargs(self, i):
        self.engine_kwargs['nspots'] = 2**(i-1)
        self.engine_kwargs['nvols'] = 2**(i-1)
        # self.engine_kwargs['nvols'] = 256
        try:
            xs = self.F.grid.mesh[0]
        except AttributeError:
            return
        return (max(xs) - min(xs)) / 2**i

    engine = FD.heston.HestonFiniteDifferenceEngine
    ct = cv.ConvergenceTester(option, engine,
            {'force_exact': False, 'spotdensity': 10, 'varexp': 4},
            dt=dt, min_i=4, max_i=9, func=func, error_func=cv.error2d,
            update_kwargs=update_kwargs)
    err = ct.dx()
    key = ('heston', mode, option.strike)
    errors[key] = err

errors = {}

strike = 80.0
option = FD.heston.HestonOption(tenor=1, strike=strike, volatility=0.2, mean_reversion=1, vol_of_variance=0.2, correlation=-0.7)
print option
print option.analytical

fname = "vanilla_strike-%s_%s_%s" % (strike, func, mode)

rundx()

with open('data_convergence/%s.py' % fname, 'w') as fout:
    fout.write(str(errors) + '\n')

print errors
