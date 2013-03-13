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
from FiniteDifference.utils import todia, block_repeat, foldMatFor
from FiniteDifference.visualize import fp
# def fp(*x, **y):
    # pass
import FiniteDifference.Grid as Grid

import FiniteDifference.FiniteDifferenceEngine as FD
import FiniteDifference.FiniteDifferenceEngineGPU as FDG

from FiniteDifference.blackscholes import BlackScholesFiniteDifferenceEngine, BlackScholesOption
from FiniteDifference.heston import HestonBarrierOption



class BlackScholesOption_test(unittest.TestCase):

    def setUp(self):
        v = 0.04
        r = 0.06
        k = 99.0
        spot = 100.0
        t = 1.0
        self.dt = 1.0/150.0
        BS = BlackScholesOption(spot=spot, strike=k, interest_rate=r, variance=v, tenor=t)

        self.F = BlackScholesFiniteDifferenceEngine( BS
                                                   , spot_max=5000.0
                                                   , nspots=150
                                                   , spotdensity=1.0
                                                   , force_exact=True
                                                   , flip_idx_spot=False
                                                     )
        self.F.init()
        self.FG = FDG.FiniteDifferenceEngineADI(self.F)

    def test_implicit(self):
        t, dt = self.F.option.tenor, self.dt
        for o in self.F.operators.values():
            assert o.is_tridiagonal()
        V = self.FG.solve_implicit(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        ans = self.F.option.analytical
        # print "Spot:", self.F.option.spot
        # print "Price:", V, ans, V - ans
        npt.assert_allclose(V, ans, rtol=0.001)

    def test_douglas(self):
        t, dt = self.F.option.tenor, self.dt
        V = self.FG.solve_douglas(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        ans = self.F.option.analytical
        # print "Spot:", self.F.option.spot
        # print "Price:", V, ans, V - ans
        npt.assert_allclose(V, ans, rtol=0.001)

    def test_smooth(self):
        raise unittest.SkipTest
        t, dt = self.F.option.tenor, self.dt
        for o in self.F.operators.values():
            assert o.is_tridiagonal()
        V = self.FG.solve_smooth(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        ans = self.F.option.analytical
        # print "Spot:", self.F.option.spot
        # print "Price:", V, ans, V - ans
        npt.assert_allclose(V, ans, rtol=0.001)


class FiniteDifferenceEngineADI_test(unittest.TestCase):

    def setUp(self):
        s1_enable = 1
        s2_enable = 1
        v1_enable = 1
        v2_enable = 1
        r_enable = 1
        dirichlet_s = 1
        dirichlet_v = 1
        kappa = 1
        r = 0.06
        theta = 0.04
        sigma = 0.4
        rho = 0.6
        spot_max = 1500.0
        var_max = 13.0
        nspots = 5
        nvols = 5
        spotdensity = 7.0  # infinity is linear?
        varexp = 5

        up_or_down_spot = 'up'
        up_or_down_var = 'down'
        flip_idx_spot = nspots//2
        flip_idx_var = nvols//2
        k = spot_max / 4.0
        # spots = np.linspace(0, spot_max, nspots)
        # vars = np.linspace(0, var_max, nvols)
        spots = utils.sinh_space(k, spot_max, spotdensity, nspots)
        vars = utils.exponential_space(0.00, 0.04, var_max, varexp, nvols)
        def mu_s(t, *dim): return r * dim[0] * s1_enable
        def gamma2_s(t, *dim): return 0.5 * dim[1] * dim[0]**2 * s2_enable

        def mu_v(t, *dim): return kappa * (theta - dim[1]) * v1_enable
        def gamma2_v(t, *dim): return 0.5 * sigma**2 * dim[1] * v2_enable

        coeffs = {()   : lambda t: -r * r_enable,
                  (0,) : mu_s,
                  (0,0): gamma2_s,
                  (1,) : mu_v,
                  (1,1): gamma2_v,
                  (0,1): lambda t, *dim: dim[0] * dim[1] * rho * sigma
                  }
        coeffs = {k: lambda *x: 1 for k in coeffs}
        bounds = {
                (0,)  : ((0,    lambda *args: 0),    (1,    lambda *args: 1)),
                (0,0) : ((0,    lambda *args: 0),    (None, lambda *args: 1)),
                (1,)  : ((None, lambda *args: None), (None, lambda *args: None)),
                (1,1) : ((1,    lambda *args: 0.0),  (None, lambda *args: None)),
                }

        schemes = {}
        # G = Grid.Grid((spots, vars), initializer=lambda x0,x1: np.maximum(x0-k,0))
        self.G = Grid.Grid((spots, vars), initializer=lambda x0,x1: x0*x1)
        # print G

        self.F = FD.FiniteDifferenceEngineADI(self.G, coefficients=coeffs,
                boundaries=bounds, schemes=schemes, force_bandwidth=None)
        self.F.init()
        self.F.operators[1]
        self.FG = FDG.FiniteDifferenceEngineADI(self.F)


    def test_verify_simple_operators_0(self):
        ref = self.F.simple_operators[(0,)]
        tst = self.FG.simple_operators[(0,)].immigrate()
        npt.assert_equal(tst, ref)

    def test_verify_simple_operators_1(self):
        ref = self.F.simple_operators[(1,)]
        tst = self.FG.simple_operators[(1,)].immigrate()
        npt.assert_equal(tst, ref)

    def test_verify_simple_operators_00(self):
        ref = self.F.simple_operators[(0,0)]
        tst = self.FG.simple_operators[(0,0)].immigrate()
        npt.assert_equal(tst, ref)

    def test_verify_simple_operators_11(self):
        ref = self.F.simple_operators[(1,1)]
        ref.diagonalize()
        tst = self.FG.simple_operators[(1,1)]
        tst.diagonalize()
        tst = tst.immigrate()
        npt.assert_equal(tst, ref)

    def test_verify_simple_operators_01(self):
        ref = self.F.simple_operators[(0,1)]
        tst = self.FG.simple_operators[(0,1)].immigrate()
        npt.assert_equal(tst, ref)

    def test_combine_operators_0(self):
        ref = self.F.operators[0]
        fst = self.FG.simple_operators[(0,)]
        snd = self.FG.simple_operators[(0,0)]
        tst = (fst + snd) + 0.5
        tst = tst.immigrate()
        tst.derivative = ref.derivative
        npt.assert_equal(tst, ref)

    def test_combine_operators_1(self):
        ref = self.F.operators[1]
        npt.assert_array_equal([1, 0, -1, -2], ref.D.offsets)
        npt.assert_equal(ref.bottom_fold_status, "CAN_FOLD")
        fst = self.FG.simple_operators[(1,)]
        snd = self.FG.simple_operators[(1,1)]
        tst = (fst + snd) + 0.5
        tst = tst.immigrate()
        npt.assert_array_equal([1, 0, -1, -2], tst.D.offsets)
        npt.assert_equal(tst.bottom_fold_status, "CAN_FOLD")
        tst.derivative = ref.derivative
        fp(tst.D.data - ref.D.data, 'e')

        tstD = tst.D.data
        refD = ref.D.data
        npt.assert_array_almost_equal(tstD, refD)

        tst.D.data *= 0
        ref.D.data *= 0
        npt.assert_equal(tst, ref)

    def test_cross_derivative(self):
        crossOp = self.F.operators[(0,1)]
        crossOpGPU = self.FG.simple_operators[(0,1)]
        g = self.F.grid.domain[-1]
        x = self.F.grid.mesh[0]
        y = self.F.grid.mesh[1]

        d2gdxdy = crossOp.apply(g)
        d2gdxdyGPU = crossOpGPU.apply(g)
        npt.assert_array_almost_equal(d2gdxdy, d2gdxdyGPU)

        scale = np.random.random()
        crossOp *= scale
        crossOpGPU *= scale

        d2gdxdy_scaled = crossOp.apply(g)
        d2gdxdyGPU_scaled = crossOp.apply(g)
        npt.assert_array_almost_equal(d2gdxdy_scaled, d2gdxdyGPU_scaled)


def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
