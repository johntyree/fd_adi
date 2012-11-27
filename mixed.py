#!/usr/bin/env python
# coding: utf8
""""""

# import sys
# import os
# import itertools as it

import FiniteDifferenceEngine as FD
from pylab import *
import numpy as np
import scipy.sparse as sps

import Grid
from visualize import fp


vec0 = np.arange(1,5)
vec1 = np.arange(1,5)
d0 = len(vec0)
d1 = len(vec1)
g = Grid.Grid(mesh=(vec0, vec1), initializer=lambda x,y: x*y)


coeffs = {
    # D: U = 0              VN: dU/dS = 1
    (0,)  : lambda t, *x: x[0]*0+1.0,
    (1,)  : lambda t, *x: x[1]*0+1.0,
    (0,1) : lambda t, *x: x[0]*0,
}

bounds = {
    # D: U = 0              VN: dU/dS = 1
    (0,)  : ((0, lambda *x: 0.0), (0, lambda *x: 0.0)),
    (1,)  : ((0, lambda *x: 0.0), (0, lambda *x: 0.0))
}

F = FD.FiniteDifferenceEngineADI(g, boundaries=bounds, coefficients=coeffs)


# B0.D = sps.dia_matrix((np.arange(3*d0).reshape((3,d0)), B0.offsets), shape=B0.shape)
Bs = FD.BandedOperator.for_vector(vec0, derivative=1)
Bm1 = FD.BandedOperator.for_vector(vec1, derivative=1)
Bb1 = Bm1.copy()
Bp1 = Bm1.copy()



def replicate(n, x):
    ret = []
    for _ in xrange(n):
        ret.append(x.copy())
    return ret

def main():
    """Run main."""
    Bp1.offsets += d1
    Bm1.offsets -= d1
    # d0block = zeros((len(bo), d0))
    # Bp1.D.data = hstack((d0block, d0block, Bp1.data))
    Bps = [Bp1 * 0, Bp1 * 0] + replicate(d0-2, Bp1)
    Bms = replicate(d0-2, Bm1) + [Bm1 * 0, Bm1 * 0]
    Bbs = [Bb1 * 0] + replicate(d0-2, Bb1) +  [Bb1 * 0]

    B0 = FD.flatten_tensor([Bs * 0] + replicate(d1-2, Bs) +  [Bs * 0])
    B1 = FD.flatten_tensor([x.copy() for x in Bbs])

    offsets = Bs.offsets
    data = [Bps, Bbs, Bms]

    for row, o in enumerate(offsets):
        if o >= 0:
            for i in xrange(Bs.shape[0]-o):
                # print "(%i, %i)" % (row, i+o), "Block", i, i+o, "*",
                # Bs.data[row, i+o]
                # print data[row][i+o].data
                data[row][i+o] *= Bs.data[row, i+o]
                # print data[row][i+o].data
        else:
            for i in xrange(abs(o), Bs.shape[0]):
                # print "(%i, %i)" % (row, i-abs(o)), "Block", i, i-abs(o), "*", Bs.data[row, i-abs(o)]
                data[row][i-abs(o)] *= Bs.data[row, i-abs(o)]
                # print data[row][i-abs(o)].data
        print

    BP = FD.flatten_tensor(Bps)
    BB = FD.flatten_tensor(Bbs)
    BM = FD.flatten_tensor(Bms)
    # fp(BP.data)
    # fp(BB.data)
    # fp(BM.data)
    print
    # fp(vec1)
    B12 = BP+BB+BM

    fp(g)

    print
    fp(B0)
    fp(F.operators[0])
    print
    fp(B0.apply(g.T).T)
    print
    fp(diff(g, axis=0))
    print

    print
    # fp(B1)
    print
    fp(B1.apply(g))
    print
    fp(diff(g, axis=1))
    print

    print
    # fp(B12)
    print
    fp(B12.apply(g))
    print
    fp(diff(diff(g, axis=0),axis=1))
    print

    print
    fp(B12)
    print
    fp(F.operators[(0,1)])
    print


    print '==='
    return 0

if __name__ == '__main__':
    main()
