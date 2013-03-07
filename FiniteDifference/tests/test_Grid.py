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

class Grid_test(unittest.TestCase):

    def setUp(self):
        spot_max = 1500.0
        var_max = 13.0
        nspots = 50
        nvols = 30
        spotdensity = 7.0  # infinity is linear?
        varexp = 4
        self.strike = spot_max / 4.0
        self.spots = utils.sinh_space(self.strike, spot_max, spotdensity, nspots)
        self.vars = utils.exponential_space(0.00, 0.04, var_max, varexp, nvols)
        self.Grid = Grid.Grid((self.spots, self.vars),
                initializer=lambda x0,x1: np.maximum(x0-self.strike,0))

    def test_mesh(self):
        G = self.Grid
        npt.assert_array_equal(G.mesh[0], self.spots)
        npt.assert_array_equal(G.mesh[1], self.vars)
        npt.assert_(G.ndim == len(G.mesh))
        npt.assert_(G.shape == tuple(map(len, G.mesh)))

    def test_domain(self):
        U = np.tile(np.maximum(0, self.spots - self.strike), (len(self.vars), 1)).T
        G = self.Grid
        # print G
        # print U
        npt.assert_array_equal(G.domain[-1], U)


    def test_copy():
        g = self.Grid
        h = g.copy()
        g.domain[1,:] = 4
        print g.mesh
        print h.mesh
        print
        print g.domain
        print h.domain
        print g.shape, h.shape
        assert(g.domain != h.domain)
        g.reset()
        assert(g.domain != h.domain)
        g.reset()
        print g.mesh
        print g.domain
        print g.shape
        return 0



def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
