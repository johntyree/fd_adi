#!/usr/bin/env python
# coding: utf8

# import sys
# import os
import itertools
# from bisect import bisect_left
import unittest

import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.linalg as spl

import FiniteDifference.utils as utils
from FiniteDifference.utils import todia
from FiniteDifference.visualize import fp
# def fp(*x, **y):
    # pass
import FiniteDifference.Grid as Grid

import FiniteDifference.FiniteDifferenceEngine as FD
import FiniteDifference.BandedOperatorGPU as BOG

from FiniteDifference.blackscholes import BlackScholesFiniteDifferenceEngine, BlackScholesOption
from FiniteDifference.heston import HestonBarrierOption


def test_numpy_transpose_vs_rollaxis():
    a = np.ones((3,1,4,2,0,5))
    for axis in range(len(a.shape)):
        t = range(len(a.shape))
        utils.rolllist(t, axis, 0)
        ta = np.transpose(a, axes=t)
        npt.assert_(np.rollaxis(a, axis).shape == ta.shape)
        utils.rolllist(t, 0, axis)
        ta = np.transpose(a, axes=t)
        npt.assert_(a.shape == np.transpose(ta, axes=t).shape)


def apWithRes(A, U, R, blocks=1):
    A = todia(A)
    d = foldMatFor(A, blocks)
    diagonalize = d.dot
    undiagonalize = d.todense().I.dot
    ret = undiagonalize(diagonalize(A).dot(U)) + R
    return d, ret


def solveWithRes(A, U, R, blocks=1):
    A = todia(A)
    d = foldMatFor(A, blocks)
    diagonalize = compose(todia, d.dot)
    # undiagonalize = compose(lambda x: x.A[0], d.todense().I.dot)
    diaA = diagonalize(A)
    ret = diaA.todense().I.dot(d.dot(U-R)).A[0]
    return ret


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

def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
