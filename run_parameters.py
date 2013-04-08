#!/usr/bin/env python
# coding: utf8

from __future__ import division

import sys
import os
import itertools as it
import subprocess
import multiprocessing as m

import numpy as np

from price_one_option import data_dir

def price_one(params):
    mon, ir, vol, volvol, corr, rev, tenor, (ns, nv), nt = params
    spot = 100
    strike = np.round(spot / mon, decimals=2)
    cmd = 'python'
    args = [cmd, 'price_one_option.py', '--scheme', 'hv', '-s', spot, '-k',
            strike, '-t', tenor, '-r', ir, '--mean-reversion', rev,
            '--variance', vol**2, '-o', volvol, '--mean-variance', vol**2,
            '-p', corr, '-nx', ns, nv, '-nt', nt, '-v', back]
    args = map(str, args)
    print args
    try:
        ret = subprocess.call(args)
    except AssertionError:
        print "PROBLEM! ASSERTION ERROR"
    except e:
        print "Something Crazy", e
        sys.exit(1)
    if ret != 0:
        print "PROBLEM! EXIT CODE", ret
    return 0


def main():
    """Run main."""
    print sys.argv
    global back
    try:
        back = sys.argv[1]
    except IndexError:
        back = '--cpu'

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

    params = list(it.product(
        moneyness,
        ir,
        vol,
        volvol,
        corr,
        rev,
        tenor,
        [(s, s // 2) for s in space],
        time
    ))

    if back == '--gpu':
        procs = 1
    else:
        procs = m.cpu_count()
    procs = 1

    pool = m.Pool(processes=procs)
    pool.map(price_one, params)

if __name__ == '__main__':
    main()

# usage: price_one_option.py [-h] [--scheme {i,d,hv,s}] [-s FLOAT] [-k FLOAT]
                            # [-t FLOAT] [-r FLOAT] [--mean-reversion FLOAT]
                            # [--variance FLOAT] [-o FLOAT]
                            # [--mean-variance FLOAT] [-p FLOAT] [--min_i INT]
                            # [--max_i INT] [--top TOP] [--barrier] -nx int int
                            # -nt int [-v] [--gpu] [--cpu]
