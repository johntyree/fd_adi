#!/usr/bin/env python
# coding: utf8
"""<+Module Description.+>"""

from __future__ import division

import sys
import os
import itertools as it
import subprocess

from price_one_option import data_dir

os.chdir(os.path.join(data_dir, '..'))

moneyness = [0.93, 1, 1.06]
ir = [0.01, 0.1]

vol = [0.015, 0.35]
volvol = [0.001, 0.4]

corr = [-0.5, 0, 0.3]
rev = [0.1, 1, 5]
tenor = [0.25, 1.5]
space = [int(2**i) for i in range(6, 11)]
time = [int(2**i) for i in range(6, 11)]

params = it.product(
    moneyness,
    ir,
    vol,
    volvol,
    corr,
    rev,
    tenor,
    space,
    [s // 2 for s in space],
    time
)


def main():
    """Run main."""
    for mon, ir, vol, volvol, corr, rev, tenor, ns, nv, nt in params:
        spot = 100
        strike = spot / mon
        cmd = 'python'
        args = [cmd, 'price_one_option.py', '--scheme', 'hv', '-s', spot, '-k',
                strike, '-t', tenor, '-r', ir, '--mean-reversion', rev,
                '--variance', vol**2, '-o', volvol, '--mean-variance', vol**2,
                '-p', corr, '-nx', ns, nv, '-nt', nt, '-v', '--gpu']
        args = map(str, args)
        print args
        try:
            ret = subprocess.call(args)
        except AssertionError:
            print "PROBLEM! ASSERTION ERROR"
        if ret != 0:
            print "PROBLEM! EXIT CODE", ret
    return 0

if __name__ == '__main__':
    main()

# usage: price_one_option.py [-h] [--scheme {i,d,hv,s}] [-s FLOAT] [-k FLOAT]
                            # [-t FLOAT] [-r FLOAT] [--mean-reversion FLOAT]
                            # [--variance FLOAT] [-o FLOAT]
                            # [--mean-variance FLOAT] [-p FLOAT] [--min_i INT]
                            # [--max_i INT] [--top TOP] [--barrier] -nx int int
                            # -nt int [-v] [--gpu] [--cpu]
