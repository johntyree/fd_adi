#!/usr/bin/env python
# coding: utf8

from __future__ import division

import itertools
import unittest

import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.linalg as spl

import FiniteDifference.utils as utils
from FiniteDifference.utils import todia, block_repeat, foldMatFor
from FiniteDifference.visualize import fp
import FiniteDifference.Grid as Grid

import FiniteDifference.FiniteDifferenceEngine as FD
import FiniteDifference.FiniteDifferenceEngineGPU as FDG

from FiniteDifference.blackscholes import BlackScholesFiniteDifferenceEngine, BlackScholesOption
from FiniteDifference.heston import HestonOption, HestonBarrierOption, HestonFiniteDifferenceEngine


class HestonBarrierOption_test(unittest.TestCase):

    def setUp(self):
        DefaultHeston = HestonBarrierOption(spot=100
                        , strike=100
                        , interest_rate=0.06
                        , volatility = 0.2
                        , tenor=1.0
                        , mean_reversion = 1
                        , mean_variance = 0.12
                        , vol_of_variance = 0.3
                        , correlation = 0.4
                        , top = (False, 170.0)
                        )
        option = DefaultHeston
        # option = HestonOption(tenor=1, strike=99.0, volatility=0.2,
                                        # mean_reversion=3, mean_variance=0.04,
                                        # vol_of_variance=0.6, correlation=-0.7)


        self.dt = 1.0/150.0
        self.F = HestonFiniteDifferenceEngine(option, nspots=150,
                                                   nvols=80,
                                                   force_bandwidth=None,
                                                   flip_idx_var=False)


        self.F.init()
        self.F.operators[1].diagonalize()
        self.FG = FDG.HestonFiniteDifferenceEngine(option,
                                                   nspots=self.F.grid.shape[0],
                                                   nvols=self.F.grid.shape[1])
        self.FG.make_operator_templates()
        self.FG.set_zero_derivative()
        self.FG.scale_and_combine_operators()


    def test_implicit(self):
        t, dt = self.F.option.tenor, self.dt
        dt = 1/600.0
        for d, o in self.F.operators.items():
            if type(d) != tuple:
                assert o.is_tridiagonal(), "%s, %s" % (d, o.D.offsets)
        V = self.FG.solve_implicit(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        V2 = self.F.solve_implicit(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        # ans = self.F.option.analytical
        # print "Spot:", self.F.option.spot
        # print "Price:", V2, V, ans, V - ans
        npt.assert_array_almost_equal(V, V2, decimal=6)
        # npt.assert_allclose(V, ans, rtol=0.001)


    def test_douglas(self):
        t, dt = self.F.option.tenor, self.dt
        for d, o in self.F.operators.items():
            if type(d) != tuple:
                assert o.is_tridiagonal(), "%s, %s" % (d, o.D.offsets)
        V = self.FG.solve_douglas(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        V2 = self.F.solve_douglas(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        # ans = self.F.option.analytical
        # print "Spot:", self.F.option.spot
        # print "Price:", V, ans, V - ans
        # npt.assert_allclose(V, ans, rtol=0.001)
        npt.assert_array_almost_equal(V, V2, decimal=6)


    def test_hundsdorferverwer(self):
        t, dt = self.F.option.tenor, self.dt
        for d, o in self.F.operators.items():
            if type(d) != tuple:
                assert o.is_tridiagonal(), "%s, %s" % (d, o.D.offsets)
        V = self.FG.solve_hundsdorferverwer(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        V2 = self.F.solve_hundsdorferverwer(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        # ans = self.F.option.analytical
        # print "Spot:", self.F.option.spot
        # print "Price:", V2, V, ans, V - ans
        # npt.assert_allclose(V, ans, rtol=0.001)
        npt.assert_array_almost_equal(V, V2, decimal=7)


    def test_smooth(self):
        t, dt = self.F.option.tenor, self.dt
        for d, o in self.F.operators.items():
            if type(d) != tuple:
                assert o.is_tridiagonal(), "%s, %s" % (d, o.D.offsets)
        V = self.FG.solve_smooth(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        V2 = self.F.solve_smooth(t/dt, dt, self.F.grid.domain[-1])[self.F.idx]
        # ans = self.F.option.analytical
        # print "Spot:", self.F.option.spot
        # print "Price:", V, ans, V - ans
        # npt.assert_allclose(V, ans, rtol=0.001)
        npt.assert_array_almost_equal(V, V2, decimal=7)


class HestonOptionConstruction_test(unittest.TestCase):

    def setUp(self):
        DefaultHeston = HestonBarrierOption(spot=100
                        , strike=100
                        , interest_rate=0.03125
                        , volatility = 0.2
                        , tenor=1.0
                        , mean_reversion = 1
                        , mean_variance = 0.12
                        , vol_of_variance = 0.3
                        , correlation = 10.4
                        , top = (False, 170.0)
                        )
        option = DefaultHeston

        # option = HestonOption(tenor=1, strike=99.0, volatility=0.2,
                                        # mean_reversion=3, mean_variance=0.04,
                                        # vol_of_variance=0.6, correlation=-0.7)

        self.dt = 1.0/2.0
        self.F = HestonFiniteDifferenceEngine(option, nspots=5,
                                                   nvols=5,
                                                   force_bandwidth=None,
                                                   force_exact=False,
                                                   flip_idx_var=False)

        # self.F = HestonFiniteDifferenceEngine(H, nspots=100,
                                         # nvols=100, spotdensity=10, varexp=4,
                                         # var_max=12, flip_idx_spot=False,
                                         # flip_idx_var=False, verbose=False,
                                         # force_bandwidth=None,
                                         # force_exact=False)

        self.F.init()

        self.FGG = FDG.HestonFiniteDifferenceEngine(option,
                                                    force_exact=False,
                                                    nspots=self.F.grid.shape[0],
                                                    nvols=self.F.grid.shape[1])
        self.FGG.set_zero_derivative()
        self.FGG.make_operator_templates()


    def test_scale_and_combine_FGG_0(self):
        self.FGG.scale_and_combine_operators(self.FGG.simple_operators)

        ref = self.F.operators[0]
        tst = self.FGG.operators[0].immigrate()

        npt.assert_array_almost_equal(tst.D.data, ref.D.data, decimal=13)
        tst.D *= 0
        ref.D *= 0
        npt.assert_equal(tst, ref)


    def test_scale_and_combine_FGG_1(self):
        self.FGG.scale_and_combine_operators()

        ref = self.F.operators[1]
        tst = self.FGG.operators[1]
        tst = tst.immigrate()

        npt.assert_array_almost_equal(tst.D.data, ref.D.data, decimal=14)
        tst.D *= 0
        ref.D *= 0
        npt.assert_equal(tst, ref)


    def test_scale_and_combine_FGG_01(self):
        self.FGG.scale_and_combine_operators()
        ref = self.F.operators[(0,1)].copy()
        tst = self.FGG.operators[(0,1)].immigrate()
        npt.assert_array_almost_equal(tst.D.data, ref.D.data, decimal=14)
        tst.D *= 0
        ref.D *= 0
        npt.assert_equal(tst, ref)

        dom = np.random.random(self.FGG.grid.shape)
        ref = self.FGG.operators[(0,1)].apply(dom.copy())
        tst = self.F.operators[(0,1)].apply(dom)
        fp(ref - tst, 'e')
        npt.assert_array_almost_equal(tst, ref, decimal=14)


    def test_verify_spots_FGG(self):
        ref = self.F.spots.copy()
        tst = self.FGG.spots.copy()
        npt.assert_equal(tst, ref)


    def test_verify_vars_FGG(self):
        ref = self.F.vars.copy()
        tst = self.FGG.vars.copy()
        npt.assert_equal(tst, ref)


    def test_verify_simple_operators_0_FGG(self):
        ref = self.F.simple_operators[(0,)].copy()
        tst = self.FGG.simple_operators[(0,)].copy().immigrate()
        npt.assert_equal(tst, ref)
        ref = self.F.simple_operators[(0,)].copy()
        tst = self.FGG.simple_operators[(0,)].copy()
        ref.diagonalize(), tst.diagonalize()
        tst = tst.immigrate()
        npt.assert_equal(tst, ref)


    def test_verify_simple_operators_1_FGG(self):
        ref = self.F.simple_operators[(1,)].copy()
        tst = self.FGG.simple_operators[(1,)].copy().immigrate()
        npt.assert_equal(tst, ref)
        ref = self.F.simple_operators[(1,)].copy()
        tst = self.FGG.simple_operators[(1,)].copy()
        ref.diagonalize(), tst.diagonalize()
        tst = tst.immigrate()
        npt.assert_equal(tst, ref)


    def test_verify_simple_operators_00_FGG(self):
        ref = self.F.simple_operators[(0,0)].copy()
        tst = self.FGG.simple_operators[(0,0)].copy().immigrate()
        npt.assert_equal(tst, ref)
        ref = self.F.simple_operators[(0,0)].copy()
        tst = self.FGG.simple_operators[(0,0)].copy()
        ref.diagonalize(), tst.diagonalize()
        tst = tst.immigrate()
        npt.assert_equal(tst, ref)


    def test_verify_simple_operators_11_FGG(self):
        ref = self.F.simple_operators[(1,1)].copy()
        tst = self.FGG.simple_operators[(1,1)].copy()
        ref.diagonalize(), tst.diagonalize()
        tst = tst.immigrate()
        npt.assert_array_almost_equal(ref.bottom_factors, tst.bottom_factors)
        tst.bottom_factors *= 0
        ref.bottom_factors *= 0
        npt.assert_array_almost_equal(tst.D.data, ref.D.data, decimal=14)
        tst.D *= 0
        ref.D *= 0
        npt.assert_equal(tst, ref)


    def test_verify_simple_operators_01_FGG(self):
        ref = self.F.simple_operators[(0,1)]
        tst = self.FGG.simple_operators[(0,1)].immigrate()
        npt.assert_equal(tst, ref)


    def test_add_operators_0_FGG(self):
        fst = self.F.simple_operators[(0,)]
        snd = self.F.simple_operators[(0,0)]
        ref = (snd + fst) + 0.5
        fst = self.FGG.simple_operators[(0,)]
        snd = self.FGG.simple_operators[(0,0)]
        tst = (snd + fst) + 0.5
        tst = tst.immigrate()
        tst.derivative = ref.derivative
        npt.assert_array_almost_equal(ref.D.data, tst.D.data, decimal=12)
        tst.D.data *= 0
        ref.D.data *= 0
        npt.assert_equal(tst, ref)


    def test_add_operators_1(self):
        fst = self.F.simple_operators[(1,)]
        snd = self.F.simple_operators[(1,1)]
        ref = (snd + fst) + 0.5
        fst = self.FGG.simple_operators[(1,)]
        snd = self.FGG.simple_operators[(1,1)]
        tst = (snd + fst) + 0.5
        tst = tst.immigrate()
        tst.derivative = ref.derivative
        npt.assert_array_almost_equal(ref.D.data, tst.D.data)
        tst.D.data *= 0
        ref.D.data *= 0
        npt.assert_equal(tst, ref)


    def test_scale_operators_0_FGG(self):
        d = (0,)
        ref = self.F.simple_operators[d]
        ref.vectorized_scale(self.F.coefficient_vector(self.F.coefficients[d],
            self.F.t, d[0]))
        tst = self.FGG.simple_operators[d]
        tst.vectorized_scale_from_host(
                self.FGG.coefficient_vector(self.F.coefficients[d],
                    self.FGG.t, d[0]))
        tst = tst.immigrate()

        npt.assert_array_almost_equal(ref.D.data, tst.D.data)
        tst.D.data *= 0
        ref.D.data *= 0
        npt.assert_equal(tst, ref)


    def test_scale_operators_00_FGG(self):
        d = (0,0)
        ref = self.F.simple_operators[d]
        ref.vectorized_scale(
                self.F.coefficient_vector(self.F.coefficients[d], self.F.t,
                    d[0]))
        tst = self.FGG.simple_operators[d]
        tst.vectorized_scale_from_host(self.FGG.coefficient_vector(self.F.coefficients[d],
            self.FGG.t, d[0]))
        tst = tst.immigrate()

        npt.assert_array_almost_equal(ref.D.data, tst.D.data)
        tst.D.data *= 0
        ref.D.data *= 0
        npt.assert_equal(tst, ref)


    def test_scale_operators_1_FGG(self):
        d = (1,)
        ref = self.F.simple_operators[d]
        ref.vectorized_scale(self.F.coefficient_vector(self.F.coefficients[d],
            self.F.t, d[0]))
        tst = self.FGG.simple_operators[d]
        tst.vectorized_scale_from_host(self.FGG.coefficient_vector(self.F.coefficients[d],
            self.FGG.t, d[0]))
        tst = tst.immigrate()

        npt.assert_array_almost_equal(ref.D.data, tst.D.data)
        tst.D.data *= 0
        ref.D.data *= 0
        npt.assert_equal(tst, ref)


    def test_scale_operators_11_FGG(self):
        d = (1,1)
        ref = self.F.simple_operators[d]
        ref.vectorized_scale(
                self.F.coefficient_vector(self.F.coefficients[d], self.F.t,
                    d[0]))
        tst = self.FGG.simple_operators[d]
        tst.vectorized_scale_from_host(self.FGG.coefficient_vector(self.F.coefficients[d],
            self.FGG.t, d[0]))
        tst = tst.immigrate()

        npt.assert_array_almost_equal(ref.D.data, tst.D.data)
        tst.D.data *= 0
        ref.D.data *= 0
        npt.assert_equal(tst, ref)


    def test_scale_operators_01_FGG(self):
        d = (0,1)
        ref = self.F.simple_operators[d]
        ref.vectorized_scale(
                self.F.coefficient_vector(self.F.coefficients[d], self.F.t,
                    d[0]))
        tst = self.FGG.simple_operators[d]
        tst.vectorized_scale_from_host(self.FGG.coefficient_vector(self.F.coefficients[d],
            self.FGG.t, d[0]))
        tst = tst.immigrate()

        npt.assert_array_almost_equal(ref.D.data, tst.D.data)
        tst.D.data *= 0
        ref.D.data *= 0
        npt.assert_equal(tst, ref)

