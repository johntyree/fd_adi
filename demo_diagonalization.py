#!/usr/bin/env python
# coding: utf8
""""""

# import sys
# import os
# import itertools as it

from __future__ import division

from FiniteDifference.visualize import fp


import scipy.sparse as sps
import numpy as np

def compose(*funcs):
    names = []
    for f in reversed(funcs):
        names.append(f.__name__)
    def newf(x):
        for f in reversed(funcs):
            x = f(x)
        return x
    newf.__name__ = ' ∘ '.join(reversed(names))
    return newf



def foldin(bo):
    bo = bo.copy()
    block_len = bo.shape[0] // bo.blocks
    bottoms = foldbottom(bo.D.data, block_len, bo.blocks)
    tops = foldtop(bo.D.data, block_len, bo.blocks)
    # bo.D.offsets = bo.D.offsets[1:-1]
    return bo, tops, bottoms


def todia(A):
    d = sps.dia_matrix(A)
    d.data = d.data[::-1]
    d.offsets = d.offsets[::-1]
    return d


def diaPlusI(d):
    return todia(d + np.eye(d.data.shape[1]))


def foldMatFor(A, blocks):
    l = A.shape[0] // blocks
    data = np.zeros((3, A.shape[0]))
    data[1, :] = 1
    offsets = (1, 0, -1)
    m = len(A.offsets) // 2
    for b in range(blocks):
        data[0, b*l+1] = -A.data[m-2,b*l+2] / A.data[m-1,b*l+2] if A.data[m-1,b*l+2] else 0
        data[2, (b+1)*l-2] = -A.data[m+2,(b+1)*l-3] / A.data[m+1,(b+1)*l-3] if A.data[m+2,(b+1)*l-3] else 0
        d = sps.dia_matrix((data, offsets), shape=A.shape)
    return d


def apWithRes(A, U, R, blocks=1):
    d = foldMatFor(A, blocks)
    diagonalize = d.dot
    undiagonalize = d.todense().I.dot
    ret = undiagonalize(diagonalize(A).dot(U)) + R
    return d, ret


def apWithRes_(A, U, R):
    return A.dot(U) + R


def solveWithRes(A, U, R):
    A = todia(A)
    d = foldMatFor(A)
    diagonalize = compose(todia, d.dot)
    undiagonalize = compose(lambda x: x.A[0], d.todense().I.dot)
    diaA = diagonalize(A)
    fp(diaA)
    ret = diaA.todense().I.dot(d.dot(U-R)).A[0]
    return ret


def solveWithRes_(A, U, R):
    return A.I.dot(U-R).A[0]


def foldbottom(d, block_len, blocks):
    # d d bo.D.data
    # fp(bo.D.todense())
    # fp(d)
    # block_len = bo.shape[0] // bo.blocks
    factor = np.empty(blocks)
    m = d.shape[0] // 2
    for b in range(blocks):
        cn1 = m-1, (b+1)*block_len - 1
        bn  = m,   (b+1)*block_len - 1
        bn1 = m,   (b+1)*block_len - 1 - 1
        an  = m+1, (b+1)*block_len - 1 - 1
        an1 = m+1, (b+1)*block_len - 1 - 2
        zn  = m+2, (b+1)*block_len - 1 - 2
        print "Block %i %s, %s: d[zn] = %f, d[an1] = %f" % (b, zn, an1, d[zn], d[an1])
        print "second row:", d[an1], d[bn1], d[cn1]
        print "Bottom row:", d[zn], d[an], d[bn]
        factor[b] = -d[zn] / d[an1] if d[an1] != 0 else 0
        d[zn] += d[an1] * factor[b]
        d[an] += d[bn1] * factor[b]
        d[bn] += d[cn1] * factor[b]
    # fp(bo.D.todense())
    # fp(d)
    return factor


def foldtop(d, block_len, blocks):
    # d = bo.D.data
    # fp(bo.D.todense())
    # fp(d)
    # block_len = bo.shape[0] // bo.blocks
    factor = np.empty(blocks)
    m = d.shape[0] // 2
    for b in range(blocks):
        d0 = m-2, b*block_len + 2
        c0 = m-1, b*block_len + 1
        c1 = m-1, b*block_len + 2
        b0 = m,   b*block_len
        b1 = m,   b*block_len + 1
        a1 = m+1, b*block_len
        print "Block %i %s, %s: d[d0] = %f, d[c1] = %f" % (b, d0, c1, d[d0], d[c1])
        print "Top row   :", d[b0], d[c0], d[d0]
        print "Second row:", d[a1], d[b1], d[c1]
        factor[b] = -d[d0] / d[c1] if d[c1] != 0 else 0
        d[b0] += d[a1] * factor[b]
        d[c0] += d[b1] * factor[b]
        d[d0] += d[c1] * factor[b]
    # fp(bo.D.todense())
    # fp(d)
    return factor


