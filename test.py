#!/usr/bin/env python
# coding: utf8

# import sys
# import os
# import itertools as it


import numpy as np
import utils
from bisect import bisect_left
import scipy.sparse
from visualize import fp
import FiniteDifferenceEngine as FD

def nonuniform_forward_coefficients(deltas):
    """
    The coefficients for tridiagonal matrices operating on a non-uniform grid.

    L = spl.dia_matrix((fst, (2,1,0)), shape=(fst.shape[1], fst.shape[1]))
    """
    d = deltas.copy()
    fst = np.zeros((3,len(d)))
    snd = fst.copy()
    for i in range(1,len(d)-2):
        fst[0,i]   = (-2*d[i+1]-d[i+2]) / (d[i+1]*(d[i+1]+d[i+2]))
        fst[1,i+1] = (d[i+1] + d[i+2])  /         (d[i+1]*d[i+2])
        fst[2,i+2] = -d[i+1]           / (d[i+2]*(d[i+1]+d[i+2]))

        denom = (0.5*(d[i+2]+d[i+1])*d[i+2]*d[i+1]);
        snd[0,i]   =   d[i+2]         / denom
        snd[1,i+1] = -(d[i+2]+d[i+1]) / denom
        snd[2,i+2] =   d[i+1]         / denom

    # Use first order approximation for the last (inner) row
    fst[0, -2] = -1 / d[-1]
    fst[1, -1] =  1 / d[-1]


    L1 = scipy.sparse.dia_matrix((fst.copy(), (0, 1, 2)), shape=(len(d),len(d)))
    L2 = scipy.sparse.dia_matrix((snd.copy(), (0, 1, 2)), shape=(len(d),len(d)))
    return L1,L2


def nonuniform_center_coefficients(deltas):
    """
    The coefficients for tridiagonal matrices operating on a non-uniform grid.

    L = spl.dia_matrix((fst, (1,0,-1)), shape=(fst.shape[1], fst.shape[1]))
    """
    d = deltas.copy()
    fst = np.zeros((3,len(d)))
    snd = fst.copy()
    for i in range(1,len(d)-1):
        fst[0,i-1] =         -d[i+1]  / (d[i  ]*(d[i]+d[i+1]))
        fst[1,i]   = (-d[i] + d[i+1]) /         (d[i]*d[i+1])
        fst[2,i+1] =            d[i]  / (d[i+1]*(d[i]+d[i+1]))

        snd[0,i-1] = 2  / (d[i  ]*(d[i]+d[i+1]))
        snd[1,i]   = -2 /       (d[i]*d[i+1])
        snd[2,i+1] = 2  / (d[i+1]*(d[i]+d[i+1]))
    L1 = scipy.sparse.dia_matrix((fst.copy(), (-1, 0, 1)), shape=(len(d),len(d)))
    L2 = scipy.sparse.dia_matrix((snd.copy(), (-1, 0, 1)), shape=(len(d),len(d)))
    return L1, L2


def nonuniform_backward_coefficients(deltas):
    """
    The coefficients for tridiagonal matrices operating on a non-uniform grid.

    L = spl.dia_matrix((fst, (0,-1,-2)), shape=(fst.shape[1], fst.shape[1]))
    """
    d = deltas.copy()
    fst = np.zeros((3,len(d)))
    snd = fst.copy()
    for i in range(2,len(d)-1):
        fst[0, i-2] = d[i]             / (d[i-1]*(d[i-1]+d[i]));
        fst[1, i-1] = (-d[i-1] - d[i]) / (d[i-1]*d[i]);
        fst[2, i]   = (d[i-1]+2*d[i])  / (d[i]*(d[i-1]+d[i]));

        denom = (0.5*(d[i]+d[i-1])*d[i]*d[i-1]);
        snd[0, i-2] = d[i] / denom;
        snd[1, i-1] = -(d[i]+d[i-1]) / denom;
        snd[2, i]   = d[i-1] / denom;


    L1 = scipy.sparse.dia_matrix((fst.copy(), (-2, -1, 0)), shape=(len(d),len(d)))
    L2 = scipy.sparse.dia_matrix((snd.copy(), (-2, -1, 0)), shape=(len(d),len(d)))
    return L1,L2
    return fst, snd


def splice_diamatrix(top, bottom, idx=0):
    newoffsets = sorted(set(top.offsets).union(set(bottom.offsets)))
    newdata = np.zeros((len(newoffsets), top.shape[1]))

    # Copy the top part
    for torow, o in enumerate(newoffsets):
        if idx + o < 0:
            raise ValueError,("You are using forward or backward derivatives "
                              "too close to the edge of the vector. "
                              "(idx = %i, row offset = %i)" % (idx, o))
        if o in top.offsets:
            fromrow = bisect_left(top.offsets, o)
            newdata[torow,:idx+o] = top.data[fromrow, :idx+o]
        if o in bottom.offsets:
            fromrow = bisect_left(bottom.offsets, o)
            newdata[torow,idx+o:] = bottom.data[fromrow, idx+o:]

    newShape = (newdata.shape[1], newdata.shape[1])
    newOp = scipy.sparse.dia_matrix((newdata, newoffsets), shape=newShape)
    return newOp


def main():
    """Run main."""

    d = np.ones(10)
    # F1,F2 = nonuniform_forward_coefficients(d)
    # B1,B2 = nonuniform_backward_coefficients(d)
    # C1,C2 = nonuniform_center_coefficients(d)

    # newOp = splice_diamatrix(F1, C1, idx=3)
    # newOp = splice_diamatrix(newOp, B1, idx=7)

    # newOp = splice_diamatrix(C1, F1, idx=3)
    oldOp = utils.nonuniform_complete_coefficients(d, up_or_down='up', flip_idx=3)[0]

    # print np.allclose(newOp.todense(), oldOp.todense())

    C1 = FD.BandedOperator.from_vector(np.arange(10), scheme='center', derivative=1, order=2)
    F1 = FD.BandedOperator.from_vector(np.arange(10), scheme='forward', derivative=1, order=2)

    print C1
    print C1.D
    print C1.data

    fp(C1)
    print
    fp(F1)
    print C1.offsets
    print F1.offsets

    newOp = C1.splice_with(F1, 3, overwrite=False)
    print np.allclose(newOp.todense(), oldOp.todense())

    return 0

if __name__ == '__main__':
    main()
