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

import utils
from utils import todia
from visualize import fp
# def fp(*x, **y):
    # pass
import Grid

import FiniteDifferenceEngine as FD
import BandedOperatorGPU as BOG

from blackscholes import BlackScholesFiniteDifferenceEngine, BlackScholesOption
from heston import HestonBarrierOption


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


class BarrierOption_test(unittest.TestCase):

    def setUp(self):
        self.option = HestonBarrierOption()
        self.s = np.array((4.5, 0.2, 5.5, 0, 3, 5.3, 0.001, 24, 1.3, 2.5))
        self.state = np.array((1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=bool)

    def test_knockout_impossible(self):
        s = self.s.copy()
        state = self.state.copy()
        self.option.top = (False, np.infty)
        self.option.monte_carlo_callback(s, state)
        npt.assert_array_equal(np.ones(state.shape), state)

    def test_knockout_inevitable(self):
        self.option.top = (False, 0)
        self.option.monte_carlo_callback(self.s, self.state)
        npt.assert_array_equal(self.state, np.zeros(self.state.shape))

    def test_knockout_partial(self):
        self.option.top = (False, 3.0)
        self.option.monte_carlo_callback(self.s, self.state)
        res = np.array((0,1,0,1,0,0,1,0,1,1), dtype=bool)
        npt.assert_array_equal(self.state, res)

    def test_knockout_permanent(self):
        self.option.top = (False, 3.0)
        self.option.monte_carlo_callback(self.s, self.state)
        res = np.array((0,1,0,1,0,0,1,0,1,1), dtype=bool)
        self.s *= 0
        self.option.monte_carlo_callback(self.s, self.state)
        npt.assert_array_equal(self.state, res)

    def test_knockout_double(self):
        self.option.top = (False, 3.0)
        self.option.bottom = (False, 1.0)
        res = np.array((0,0,0,0,0,0,0,0,1,1), dtype=bool)
        self.option.monte_carlo_callback(self.s, self.state)
        npt.assert_array_equal(self.state, res)


class Cpp_test(unittest.TestCase):

    def setUp(self):
        print "Setting up Params for CPP tests"
        shape = (4,4)
        self.v1 = np.arange(shape[0]*shape[1], dtype=float)**2
        self.v2 = self.v1.copy()
        self.v2.resize(shape)

        coeffs = {(0,) : lambda *x: 1,
                  (0,0): lambda *x: 1,
                  (1,) : lambda *x: 1,
                  (1,1): lambda *x: 1,
                  (0,1): lambda *x: 1,
                  }
        bounds = {
                (0,)  : ((0, lambda *args: 0), (1, lambda *args: 1)),
                (0,0)  : ((0, lambda *args: 0), (None, lambda *args: 1)),
                (1,)  : ((None, lambda *args: None), (None, lambda *args: None)),
                (1,1)  : ((1, lambda *args: 0.0), (None, lambda *args: None)),
                }

        schemes = {}

        self.G = Grid.Grid([np.arange(shape[0]), np.arange(shape[1])], lambda x, y: (x*shape[1]+y)**2)
        print self.G
        self.F = FD.FiniteDifferenceEngineADI(self.G, coefficients=coeffs,
                boundaries=bounds, schemes=schemes, force_bandwidth=None)
        print "Setting up FDE for CPP tests"
        self.F.init()
        self.F.operators[0].R = np.arange(self.G.size, dtype=float)
        self.F.operators[1].R = np.arange(self.G.size, dtype=float)
        self.F.operators[1].diagonalize()
        print "Setup complete for CPP test"

    def test_SizedArray_roundtrip(self):
        npt.assert_array_equal(self.v1, FD.BO.test_SizedArray1_roundtrip(self.v1.copy()))

    def test_SizedArray_roundtrip2D(self):
        npt.assert_array_equal(self.v2, FD.BO.test_SizedArray2_roundtrip(self.v2.copy()))

    def test_migrate_0(self):
        B = self.F.operators[0]
        ref = B.copy()
        B.emigrate("C1 0")
        B.D.data *= 0
        B.immigrate("C1 0")
        assert ref == B

    def test_migrate_1(self):
        B = self.F.operators[1]
        ref = B.copy()
        print "bottom_is_folded", B.bottom_is_folded
        print type(B.D)
        B.emigrate("B test 1")
        B.D.data *= 0
        B.immigrate("B test 1")
        print type(B.D)
        assert ref == B

    def test_migrate_01(self):
        B = self.F.operators[(0,1)]
        B.use_csr_format()
        ref = B.copy()
        B.emigrate("B test 01")
        B.D.data *= 0
        B.immigrate("B test 01")
        npt.assert_array_equal(ref.D.todense(), B.D.todense())
        assert ref == B


    def test_SizedArray_transpose(self):
        ntests = 100
        for i in range(ntests):
            shape = tuple(np.random.random_integers(1, 100, 2))
            v2 = np.arange(shape[0]*shape[1], dtype=float).reshape(shape)
            npt.assert_array_equal(v2.T, FD.BO.test_SizedArray_transpose(v2.copy()))


    def test_tri_apply_axis_0(self):
        B0  = self.F.operators[0]
        # print "B0 data"
        # fp(B0.D.data)
        R0 = B0.R.copy()
        ref = B0.apply(self.v2)
        tst = B0.apply2(self.v2.copy())
        npt.assert_array_equal(R0, B0.R)
        npt.assert_array_equal(ref, tst)


    def test_tri_apply_axis_1(self):
        B1  = self.F.operators[1]
        # print "B1 data"
        # fp(B1.D.data)
        R1 = B1.R.copy()
        ref = B1.apply(self.v2)
        tst = B1.apply2(self.v2.copy())
        npt.assert_array_equal(R1, B1.R)
        npt.assert_array_equal(ref, tst)


    def test_csr_apply_0(self):
        vec = np.arange(30, dtype=float)
        B = BOG.for_vector(vec)
        ref = B.apply(vec)
        print B.D.tocsr().data
        print B.D.tocsr().indptr
        print B.D.tocsr().indices
        B.use_csr_format()
        tst = B.apply2(vec)
        npt.assert_array_equal(ref, tst)


    def test_csr_apply_01(self):
        B01  = self.F.operators[(0,1)]
        ref = B01.apply(self.v2)
        print B01.D.tocsr().data
        print B01.D.tocsr().indptr
        print B01.D.tocsr().indices
        tst = B01.apply2(self.v2.copy())
        npt.assert_array_equal(ref, tst)


    def test_csr_apply_random(self):
        B = self.F.operators[0] # Because we aren't transposing.
        B.R = None
        B.axis = 1
        B.dirichlet = (None, None)
        B.csr = True
        for i in range(5):
            sz = np.random.randint(3, 20)
            B.D = scipy.sparse.csr_matrix(np.random.random((sz*sz,sz*sz)))
            v = np.random.random((sz, sz))
            ref = B.apply(v)
            tst = B.apply2(v)
            npt.assert_array_almost_equal(ref, tst, decimal=8)


    def test_csr_scale(self):
        B = self.F.operators[0]
        B.D = scipy.sparse.csr_matrix(np.ones((5,5)))
        B.R = None
        B.dirichlet = (None, None)
        B.use_csr_format()
        ref = np.arange(B.D.shape[0], dtype=float).repeat(B.D.shape[1])
        ref.resize(B.D.shape)
        B.vectorized_scale(np.arange(B.D.shape[0], dtype=float))
        fp(ref)
        print
        fp(B.D)
        print
        fp(B.D - ref)
        npt.assert_array_equal(ref, B.D.todense())


    def test_GPUSolve_0(self):
        B = self.F.operators[0]
        B.D.data = np.random.random((B.D.data.shape))
        B.R = np.random.random(B.D.data.shape[1])
        B.D.data[0,0] = 0
        B.D.data[-1,-1] = 0
        origdata = B.D.data.copy()
        ref = B.solve(self.v2)
        tst = B.solve2(self.v2.copy())
        fp(ref - tst, 3, 'e')
        npt.assert_array_almost_equal(ref, tst, decimal=8)
        npt.assert_array_equal(origdata, B.D.data)

    def test_GPUSolve_1(self):
        B = self.F.operators[1]
        B.D.data = np.random.random((B.D.data.shape))
        B.R = np.random.random(B.D.data.shape[1])
        B.D.data[0,0] = 0
        B.D.data[-1,-1] = 0
        origdata = B.D.data.copy()
        ref = B.solve(self.v2)
        tst = B.solve2(self.v2.copy())
        fp(ref - tst, 3, 'e')
        npt.assert_array_almost_equal(ref, tst, decimal=8)
        npt.assert_array_equal(origdata, B.D.data)


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

    def test_implicit(self):
        t, dt = self.F.option.tenor, self.dt
        for o in self.F.operators.values():
            assert o.is_tridiagonal()
        V = self.F.solve_implicit(t/dt, dt)[self.F.idx]
        ans = self.F.option.analytical
        # print "Spot:", self.F.option.spot
        # print "Price:", V, ans, V - ans
        npt.assert_allclose(V, ans, rtol=0.001)

    def test_douglas(self):
        t, dt = self.F.option.tenor, self.dt
        V = self.F.solve_douglas(t/dt, dt)[self.F.idx]
        ans = self.F.option.analytical
        # print "Spot:", self.F.option.spot
        # print "Price:", V, ans, V - ans
        npt.assert_allclose(V, ans, rtol=0.001)

    def test_smooth(self):
        t, dt = self.F.option.tenor, self.dt
        for o in self.F.operators.values():
            assert o.is_tridiagonal()
        V = self.F.solve_smooth(t/dt, dt)[self.F.idx]
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

        # TODO:!!!!XXX TODO XXX
        # var_max = nvols-1
        # spot_max = nspots-1

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
        bounds = {
                        # D: U = 0              VN: dU/dS = 1
                (0,)  : ((0 if dirichlet_s else 1, lambda *args: 0.0), (1, lambda *args: 1.0)),
                        # D: U = 0              Free boundary
                (0,0) : ((0 if dirichlet_s else 1, lambda *args: 0.0), (None, lambda *x: 1.0)),
                        # Free boundary at low variance
                (1,)  : ((None, lambda *x: None),
                        # (0.0, lambda t, *dim: 0),
                        # D intrinsic value at high variance
                        # (None, lambda *x: None)
                        (0 if dirichlet_v else 1, lambda t, *dim: np.maximum(0.0, dim[0]-k))
                        ),
                        # # Free boundary
                (1,1) : ((1, lambda *x: 0),
                        # D intrinsic value at high variance
                        # (None, lambda *x: None)
                        (0 if dirichlet_v else 1, lambda t, *dim: np.maximum(0.0, dim[0]-k))
                        )
                }

        schemes = {}
        schemes[(0,)] = ({"scheme": "center"}, {"scheme": up_or_down_spot, "from" : flip_idx_spot})
        schemes[(1,)] = ({"scheme": "center"}, {"scheme": up_or_down_var, "from" : flip_idx_var})

        dss = np.hstack((np.nan, np.diff(spots)))
        dvs = np.hstack((np.nan, np.diff(vars)))
        As_ = utils.nonuniform_complete_coefficients(dss, up_or_down=up_or_down_spot,
                                                    flip_idx=flip_idx_spot)[0]
        Ass_ = utils.nonuniform_complete_coefficients(dss)[1]
        L1_ = []
        R1_ = []
        # As_, Ass_ = utils.nonuniform_forward_coefficients(dss)
        assert(not np.isnan(As_.data).any())
        assert(not np.isnan(Ass_.data).any())
        for j, v in enumerate(vars):
            # Be careful not to overwrite operators
            As, Ass = As_.copy(), Ass_.copy()
            m = 2

            mu_s = r * spots * s1_enable
            gamma2_s = 0.5 * v * spots ** 2 * s2_enable
            for i, z in enumerate(mu_s):
                # print z, coeffs[0,](0, spots[i])
                assert z == coeffs[0,](0, spots[i])
            for i, z in enumerate(gamma2_s):
                # print z, coeffs[0,0](0, spots[i], v)
                assert z == coeffs[0,0](0, spots[i], v)

            Rs = np.zeros(nspots)
            Rs[-1] = 1

            As.data[m - 2, 2:] *= mu_s[:-2]
            As.data[m - 1, 1:] *= mu_s[:-1]
            As.data[m, :] *= mu_s
            As.data[m + 1, :-1] *= mu_s[1:]
            As.data[m + 2, :-2] *= mu_s[2:]

            Rs *= mu_s

            Rss = np.zeros(nspots)
            Rss[-1] = 2 * dss[-1] / dss[-1] ** 2

            Ass.data[m + 1, -2] = 2 / dss[-1] ** 2
            Ass.data[m    , -1] = -2 / dss[-1] ** 2

            Ass.data[m - 2, 2:] *= gamma2_s[:-2]
            Ass.data[m - 1, 1:] *= gamma2_s[:-1]
            Ass.data[m, :] *= gamma2_s
            Ass.data[m + 1, :-1] *= gamma2_s[1:]
            Ass.data[m + 2, :-2] *= gamma2_s[2:]

            Rss *= gamma2_s

            L1_.append(As.copy())
            L1_[j].data += Ass.data
            L1_[j].data[m, :] -=  0.5 * r * r_enable
            L1_[j].data[m, 0] = 1 * dirichlet_s

            R1_.append((Rs + Rss).copy())
            R1_[j][0] = 0

        L2_ = []
        R2_ = []
        Av_ = utils.nonuniform_complete_coefficients(dvs, up_or_down=up_or_down_var,
                                                    flip_idx=flip_idx_var)[0]
        Avv_ = utils.nonuniform_complete_coefficients(dvs)[1]
        assert(not np.isnan(Av_.data).any())
        assert(not np.isnan(Avv_.data).any())
        for i, s in enumerate(spots):
            mu_v = kappa * (theta - vars) * v1_enable
            gamma2_v = 0.5 * sigma ** 2 * vars * v2_enable
            # for j, z in enumerate(mu_v):
                # assert z == coeffs[1,](0, 0, vars[j])
            # for j, z in enumerate(gamma2_v):
                # assert z == coeffs[1,1](0, 0, vars[j])

            # Be careful not to overwrite our operators
            Av, Avv = Av_.copy(), Avv_.copy()

            m = 2

            Av.data[m - 2, 2] = -dvs[1] / (dvs[2] * (dvs[1] + dvs[2]))
            Av.data[m - 1, 1] = (dvs[1] + dvs[2]) / (dvs[1] * dvs[2])
            Av.data[m    , 0] = (-2 * dvs[1] - dvs[2]) / (dvs[1] * (dvs[1] + dvs[2]))

            Av.data[m - 2, 2:] *= mu_v[:-2]
            Av.data[m - 1, 1:] *= mu_v[:-1]
            Av.data[m, :] *= mu_v
            Av.data[m + 1, :-1] *= mu_v[1:]
            Av.data[m + 2, :-2] *= mu_v[2:]

            Rv = np.zeros(nvols)
            Rv *= mu_v

            Avv.data[m - 1, 1] = 2 / dvs[1] ** 2
            Avv.data[m, 0] = -2 / dvs[1] ** 2

            Avv.data[m - 2, 2:] *= gamma2_v[:-2]
            Avv.data[m - 1, 1:] *= gamma2_v[:-1]
            Avv.data[m, :] *= gamma2_v
            Avv.data[m + 1, :-1] *= gamma2_v[1:]
            Avv.data[m + 2, :-2] *= gamma2_v[2:]

            Rvv = np.zeros(nvols)
            Rvv[0] = 2 * dvs[1] / dvs[1] ** 2
            Rvv *= gamma2_v

            L2_.append(Av.copy())
            L2_[i].data += Avv.data
            L2_[i].data[m, :] -= 0.5 * r * r_enable
            L2_[i].data[m, -1] = 1 * dirichlet_v  # This is to cancel out the previous value so we can
                                # set the dirichlet boundary condition using R.
                                # Then we have U_i + -U_i + R


            R2_.append(Rv + Rvv)
            R2_[i][-1] = 0# np.maximum(0, s - k)


        def flatten_tensor(mats):
            diags = np.hstack([x.data for x in mats])
            flatmat = scipy.sparse.dia_matrix((diags, mats[0].offsets), shape=(diags.shape[1], diags.shape[1]))
            return flatmat

        L1 = flatten_tensor(L1_)
        L2 = flatten_tensor(L2_)
        R1 = np.array(R1_).T
        R2 = np.array(R2_)

        self.As_ = As_
        self.Ass_ = Ass_
        self.L1_ = L1
        self.R1_ = R1
        self.L2_ = L2
        self.R2_ = R2
        # G = Grid.Grid((spots, vars), initializer=lambda x0,x1: np.maximum(x0-k,0))
        self.G = Grid.Grid((spots, vars), initializer=lambda x0,x1: x0*x1)
        # print G

        self.F = FD.FiniteDifferenceEngineADI(self.G, coefficients=coeffs,
                boundaries=bounds, schemes=schemes, force_bandwidth=None)
        self.F.init()


    def test_coefficient_vector(self):
        mesh = (np.arange(3), np.arange(10,12))
        G = Grid.Grid(mesh, initializer=lambda *x: 0.0)
        # print G
        F = FD.FiniteDifferenceEngineADI(G, coefficients={(0,1): lambda t, *x: 1})

        op = self.F.operators[(0,1)]
        op.D.data[op.D.data != 0] = op.D.data[op.D.data != 0]**0
        op.D = todia(op.D)
        # diamat = op.D
        # densemat = diamat.todense()

        # f0 = lambda t, *x: x[0]
        # f1 = lambda t, *x: x[1]
        f = lambda t, *x: x

        vec = F.coefficient_vector(f, 0, 0)
        ref = np.tile(mesh[0], G.shape[1]), np.repeat(mesh[1], G.shape[0])
        npt.assert_array_equal(ref, vec, "Failed when focused on dim: %i", 0)

        vec = F.coefficient_vector(f, 0, 1)
        ref = np.repeat(mesh[0], G.shape[1]), np.tile(mesh[1], G.shape[0])
        npt.assert_array_equal(ref, vec, "Failed when focused on dim: %i", 1)


    def test_combine_dimensional_operators_0(self):
        if self.F.operators[0].solve_banded_offsets[1] != 2:
            unittest.skipTest("Using first order boundary approximation. Top is tridiag.")
        oldL1 = self.L1_.copy()
        oldL1 = scipy.sparse.dia_matrix(oldL1.todense())
        oldL1.data = oldL1.data[::-1]
        oldL1.offsets = oldL1.offsets[::-1]
        # high, low = 2, -2
        # m = tuple(oldL1.offsets).index(0)
        # oldL1.data = oldL1.data[m-high:m-low+1]
        # oldL1.offsets = oldL1.offsets[m-high:m-low+1]

        oldR1 = self.R1_.T.flatten()

        L1 = self.F.operators[0]

        # oldL1.data = oldL1.data[:-1]
        # oldL1.offsets = oldL1.offsets[:-1]

        # print "offsets"
        # print oldL1.offsets, L1.D.offsets
        # print "old"
        # fp(oldL1.data)
        # print
        # print "new"
        # fp(L1.D.data)
        # print
        # print "diff"
        # fp(L1.D.data - oldL1.data)
        # print
        # print "old"
        # fp(oldL1.todense())
        # print
        # print "new"
        # fp(L1.D.todense())
        # print
        # print "diff"
        # fp(oldL1.todense() - L1.D.todense())
        # npt.assert_allclose(L1.D.todense(), oldL1.todense())
        # npt.assert_allclose(L1.D.data, oldL1.data)
        # print "old"
        # print oldR1
        # print
        # print "new"
        # print L1.R
        npt.assert_allclose(L1.R, oldR1)

        # L2 = self.F.operators[1]


    def test_combine_dimensional_operators_1(self):
        if self.F.operators[1].solve_banded_offsets[1] != 2:
            print self.skipTest("Using first order boundary approximation. Top is tridiag.")
        oldL2 = self.L2_.copy()
        oldL2 = todia(oldL2.todense())
        # high, low = 2, -2
        # m = tuple(oldL2.offsets).index(0)
        # oldL2.data = oldL2.data[m-high:m-low+1]
        # oldL2.offsets = oldL2.offsets[m-high:m-low+1]

        oldR2 = self.R2_.flatten()

        L2 = self.F.operators[1]

        # print "old"
        # fp(oldL2.data)
        # print
        # print "new"
        # fp(L2.D.data)
        # print "old"
        # fp(oldL2.todense())
        # print
        # print "new"
        # fp(L2.D.todense())
        # print
        # print "diff"
        # fp(oldL2.todense() - L2.D.todense())
        npt.assert_allclose(L2.D.data, oldL2.data)
        # print "old"
        # print oldR2
        # print
        # print "new"
        # print L2.R
        # print "diff"
        # fp(L2.R - oldR2)
        npt.assert_allclose(L2.R, oldR2)


    def test_numpy_vs_operators(self):
        spots = np.linspace(0,1, 4)
        vars = np.linspace(0,10, 4)
        # spots = self.G.mesh[0]
        # vars = self.G.mesh[1]
        G = Grid.Grid((spots, vars), initializer=lambda x0,x1: x1*1)
        coeffs = {()   : lambda t: 0,
                  (0,) : lambda t, *dim: dim[0],
                  (0,0): lambda t, *dim: dim[0],
                  (1,) : lambda t, *dim: 0*dim[1],
                  (1,1): lambda t, *dim: dim[1],
                  (0,1): lambda t, *dim: dim[0] * dim[1]
                  }
        bounds = {(0,) : ((None, lambda *x: None), (None, lambda *x: 3)),
                  (0,0): ((None, lambda *x: None), (None, lambda *x: 3)),
                  (1,) : ((None, lambda *x: None), (None, lambda *x: 1)),
                  (1,1): ((None, lambda *x: None), (None, lambda *x: 1)),
                  }
        F = FD.FiniteDifferenceEngineADI(G, coefficients=coeffs, boundaries=bounds, force_bandwidth=None)
        F.init()

        cb = utils.clear_boundary
        g = F.grid.domain[-1]

        op_ = {'delta': {}, 'grid_delta': {}, 'derivative': {}}
        np_ = {'delta': {}, 'grid_delta': {}, 'derivative': {}}

        dims = coeffs.keys()
        dims.remove(())


        for dim in dims:
            op_['grid_delta'][dim] = F.simple_operators[dim].copy()
            op_['grid_delta'][dim].R = None


        op_['delta'][(0,)] = op_['grid_delta'][(0,)].deltas[:,np.newaxis]
        op_['delta'][(1,)] = op_['grid_delta'][(1,)].deltas


        x = F.grid.mesh[0]; y = F.grid.mesh[1]
        np_['delta'][(0,)] = np.gradient(x)[:,np.newaxis]
        np_['delta'][(1,)] = np.gradient(y)
        np_['delta'][(0,)][0,0] = np.nan
        np_['delta'][(1,)][0] = np.nan


        for f in (op_, np_):
            f['delta'][(0,0)] = f['delta'][(0,)]**2
            f['delta'][(1,1)] = f['delta'][(1,)]**2
            f['delta'][(0,1)] = f['delta'][(1,)] * f['delta'][(0,)]
            f['delta'][(1,0)] = f['delta'][(1,)] * f['delta'][(0,)]



        np_['grid_delta'][(0,)], np_['grid_delta'][(1,)] = np.gradient(g)
        for fst in [0,1]:
            np_['grid_delta'][(fst,0)], np_['grid_delta'][(fst,1)] = np.gradient(np_['grid_delta'][(fst,)])


        Y,X = np.meshgrid(y, x)
        for dim in dims:
            op_['derivative'][dim] = cb(op_['grid_delta'][dim].apply(g), inplace=True)
            npt.assert_(op_['derivative'][dim].shape == g.shape)
            np_['derivative'][dim] = cb(np_['grid_delta'][dim] / np_['delta'][dim], inplace=True)
            np_['derivative'][dim] *= F.coefficients[dim](0,X,Y)
            op_['derivative'][dim] *= F.coefficients[dim](0,X,Y)

        np_['derivative'][(1,0)] = cb(np_['grid_delta'][(1,0)] / np_['delta'][(1,0)], inplace=True)
        np_['derivative'][(1,0)] *= F.coefficients[(0,1)](0,X,Y)

        for dim in dims:
            for x in ('delta', 'derivative'):
                msg = "Comparing %s in dimension %s" % (x, dim)
                try:
                    npt.assert_array_almost_equal(op_[x][dim], np_[x][dim], decimal=7, err_msg=msg)
                except AssertionError:
                    print msg, op_['grid_delta'][dim].axis
                    fp((op_['grid_delta'][dim]).D.data)
                    fp((op_[x][dim]), fmt='f')
                    fp((np_[x][dim]), fmt='f')
                    fp((op_[x][dim] - np_[x][dim]), fmt='f')
                # npt.assert_allclose(op_[x][dim], np_[x][dim], err_msg=msg)
                npt.assert_array_almost_equal(op_[x][dim], np_[x][dim], decimal=7, err_msg=msg)

        npt.assert_array_almost_equal(np_['derivative'][(0,1)], np_['derivative'][(1,0)], err_msg=msg)


    def test_dirichlet_boundary(self):
        spots = np.arange(5.)
        vars = np.arange(4.)

        def rndgrid(x0, x1):
            return np.random.random((x0.shape[0], x1.shape[1]))

        G = Grid.Grid((spots, vars), initializer=rndgrid)
        npt.assert_array_equal(G.shape, tuple(map(len, G.mesh)))
        coeffs = bounds = {}
        coeffs = {
                  (0,) : lambda t, *dim: 1,
                  # (0,0): lambda t, *dim: 0,
                  (1,) : lambda t, *dim: 0,
                  # (1,1): lambda t, *dim: 0,
                  # (0,1): lambda t, *dim: 0
                  }
        bounds = {
                  (0,) : ((0, lambda t, *x: 1), (0, lambda t, *x: 1)),
                  # (0,0): ((None, lambda *x: 1), (0, lambda *x: 1)),
                  (1,) : ((0, lambda *x: 1), (0, lambda *x: 1)),
                  # (1,1): ((0, lambda *x: 1), (0, lambda *x: 1)),
                  }
        F = FD.FiniteDifferenceEngineADI(G, coefficients=coeffs, boundaries=bounds, force_bandwidth=None)
        F.init()

        for d, o in F.simple_operators.items():
            l = [[1] * F.grid.shape[(o.axis + 1) % 2]] * 2
            npt.assert_array_equal(o.dirichlet, l, err_msg="Dim: %s, dirichlet: %s, expected: %s" % (d, o.dirichlet, l))
        for d, o in F.simple_operators.items():
            l = [[1] * F.grid.shape[(o.axis + 1) % 2]] * 2
            npt.assert_array_equal(o.dirichlet, l, err_msg="Dim: %s, dirichlet: %s, expected: %s" % (d, o.dirichlet, l))

        for d in bounds.keys():
            B = F.simple_operators[d]
            # print "Derivative", d, "Dirichlets", B.dirichlet
            g = (B+1).solve(F.grid.domain[-1])
            if B.axis == 1:
                g = g.T
            if B.dirichlet[0] is not None:
                npt.assert_array_equal(g[0,:], 1)
            if B.dirichlet[1] is not None:
                npt.assert_array_equal(g[-1,:], 1)

        for d in bounds.keys():
            B = F.simple_operators[d]
            # print "Derivative", d, "Dirichlets", B.dirichlet
            # print B.dirichlet
            g = B.apply(F.grid.domain[-1])
            if B.axis == 1:
                g = g.T
            if B.dirichlet[0] is not None:
                npt.assert_array_equal(g[0,:], 1)
            if B.dirichlet[1] is not None:
                npt.assert_array_equal(g[-1,:], 1)

        for d in bounds.keys():
            B = F.simple_operators[d]
            # print "Derivative", d, "Dirichlets", B.dirichlet
            # fp(B.data)
            g = (B+1).solve(F.grid.domain[-1])
            # fp(g)
            if B.axis == 1:
                g = g.T
            if B.dirichlet[0] is not None:
                npt.assert_array_equal(g[0,:], 1)
            if B.dirichlet[1] is not None:
                npt.assert_array_equal(g[-1,:], 1)


    def test_cross_derivative(self):
        raise unittest.SkipTest
        crossOp = self.F.operators[(0,1)]
        g = self.F.grid.domain[-1]
        x = self.F.grid.mesh[0]
        y = self.F.grid.mesh[1]

        dx = np.gradient(x)[:,np.newaxis]
        dy = np.gradient(y)
        dgdx = np.gradient(g)[0]
        manuald2gdxdy = np.gradient(dgdx)[1] / (dx * dy)
        manuald2gdxdy[:,0] = 0; manuald2gdxdy[:,-1] = 0
        manuald2gdxdy[0,:] = 0; manuald2gdxdy[-1,:] = 0
        X,Y = np.meshgrid(y, x)
        manuald2gdxdy *= self.F.coefficients[(0,1)](0, X, Y)

        d2gdxdy = crossOp.apply(g)

        # print "Cross op"
        # fp(crossOp.D.todense())
        # print crossOp.dirichlet
        # print crossOp.axis
        # print

        # print "manual"
        # fp(manuald2gdxdy)
        # print
        # print "new"
        # fp(d2gdxdy)
        # print
        # print "diff"
        # fp(d2gdxdy - manuald2gdxdy, fmt='e')
        npt.assert_array_almost_equal(manuald2gdxdy, d2gdxdy)

        scale = np.random.random()

        crossOp

        # print "Scaling CrossOp (%s)" % scale
        crossOp *= scale

        d2gdxdy_scaled = crossOp.apply(g)
        manuald2gdxdy_scaled = manuald2gdxdy * scale

        # print "Cross op"
        # print crossOp.dirichlet
        # print crossOp.axis
        # fp(crossOp.D.todense())
        # print

        # print "manual"
        # fp(manuald2gdxdy_scaled)
        # print
        # print "new"
        # fp(d2gdxdy_scaled)
        # print
        # print "diff"
        # fp(d2gdxdy_scaled - manuald2gdxdy, fmt='e')

        npt.assert_array_almost_equal(manuald2gdxdy_scaled, d2gdxdy_scaled)


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


class BandedOperator_test(unittest.TestCase):

    def setUp(self):
        k = 3.0
        nspots = 8
        spot_max = 1500.0
        spotdensity = 7.0  # infinity is linear?
        spots = utils.sinh_space(k, spot_max, spotdensity, nspots)
        self.flip_idx = 4
        self.vec = spots
        self.C1 = FD.BO.for_vector(self.vec, scheme='center', derivative=1, order=2)


    def test_addself(self):
        vec = self.vec
        # idx = self.flip_idx
        # d = np.hstack((np.nan, np.diff(vec)))
        for scheme in ["center", "forward", "backward"]:
            C1 = FD.BO.for_vector(vec, scheme=scheme, derivative=1, order=2)
            C2 = C1.add(C1)
            assert C2 is not C1
            assert C2.D.data is not C1.D.data
            npt.assert_array_equal(C2.D.offsets, C1.D.offsets)
            npt.assert_array_equal(C2.D.data, C1.D.data+C1.D.data)
            npt.assert_array_equal(C2.D.data, C1.D.data*2)
            npt.assert_array_equal(C2.R, C1.R*2)
            npt.assert_array_equal(C2.R, C1.R*+C1.R)


    def test_addoperator(self):
        vec = self.vec
        # idx = self.flip_idx
        # d = np.hstack((np.nan, np.diff(vec)))
        C1 = FD.BO.for_vector(vec, scheme='center', derivative=1, order=2)
        F1 = FD.BO.for_vector(vec, scheme='forward', derivative=1, order=2)
        oldCF1 = np.zeros((len(set(F1.D.offsets) | set(C1.D.offsets)), C1.D.shape[1]))
        oldCF1R = np.zeros_like(F1.R)

        # print "F1"
        # fp(F1.D.data)
        # print "C1"
        # fp(C1.D.data)

        CF1 = C1.add(F1)
        oldCF1[:4,:] += F1.D.data[:4, :]
        oldCF1[1:4,:] += C1.D.data
        oldCF1R = F1.R + C1.R
        npt.assert_array_equal(CF1.D.data, oldCF1)
        npt.assert_array_equal(CF1.R, oldCF1R+oldCF1R)
        npt.assert_array_equal(CF1.R, oldCF1R*2)


    def test_addoperator_inplace(self):
        vec = self.vec
        # idx = self.flip_idx
        # d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BO.for_vector(vec, scheme='backward', derivative=2, order=2, force_bandwidth=(-2,2))
        C2 = FD.BO.for_vector(vec, scheme='center', derivative=2, order=2, force_bandwidth=(-2,2))
        oldCB2 = np.zeros((len(set(B2.D.offsets) | set(C2.D.offsets)), C2.D.shape[1]))
        oldCB2[1:,:] += B2.D.data[1:, :]
        oldCB2[1:4,:] += C2.D.data[1:4, :]
        oldCB2R = np.zeros_like(B2.R)
        oldCB2R = B2.R + C2.R
        B2.add(C2, inplace=True)
        npt.assert_array_equal(oldCB2, B2.D.data)
        npt.assert_array_equal(oldCB2R, B2.R)

        B2 = FD.BO.for_vector(vec, scheme='backward', derivative=2, order=2)
        C2 = FD.BO.for_vector(vec, scheme='center', derivative=2, order=2)
        npt.assert_raises(ValueError, lambda: B2.add(C2, inplace=True))


    def test_mul(self):
        vec = self.vec
        # idx = self.flip_idx
        # d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BO.for_vector(vec, scheme='backward', derivative=2, order=2, force_bandwidth=(-2,2))
        B2.R += 1
        BB2 = B2
        assert (B2 is not B2 * 1)
        assert (B2 is not B2.mul(1))

        assert (B2 == B2 * 1)
        assert (B2 == B2.mul(1))

        assert (B2 * 6 == (B2 * 2) * 3)
        assert (B2.mul(6) == B2.mul(2).mul(3))

        B2Copy = B2.copy()
        B2 *= 2
        assert (BB2 is B2)
        assert (B2Copy is not B2)
        assert (B2Copy * 2 == B2)
        assert (BB2 is B2.mul(2, inplace=True))
        assert (BB2 is B2.mul(2, inplace=True).mul(2, inplace=True))

        assert B2 * 4 == B2.mul(4)


        dt = 0.04
        bold = B2.copy()
        B2 *= -dt
        B2 += 1
        assert (bold * -dt) + 1 == B2


    def test_eq(self):
        vec = self.vec
        # idx = self.flip_idx
        # d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BO.for_vector(vec, scheme='backward', derivative=2, order=2, force_bandwidth=(-2,2))
        assert B2 == B2
        C2 = FD.BO.for_vector(vec, scheme='center', derivative=2, order=2, force_bandwidth=(-2,2))
        # print C2, B2
        # fp(C2.D)
        # fp(B2.D)
        assert C2 != B2


    def test_addscalar(self):
        vec = self.vec
        # idx = self.flip_idx
        # d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BO.for_vector(vec, scheme='backward', derivative=2, order=2, force_bandwidth=(-2,2))
        origB2 = B2.copy()
        oldB2 = B2.copy()

        newB2 = B2.add(1.0)

        # Manually add 1 to main diag
        oldB2.D.data[tuple(oldB2.D.offsets).index(0)] += 1.0

        assert newB2 is not B2 # Created a new operator
        assert newB2.D.data is not B2.D.data # With new underlying data

        # print "New:"
        # fp(newB2.D.data)
        # print "Old:"
        # fp(oldB2.D.data)
        # print "diff:"
        # fp(newB2.D.data - oldB2.D.data)

        npt.assert_array_equal(newB2.D.data, oldB2.D.data) # Operations were the same
        # NO numpy assert here. We need "not equal"
        assert (newB2.D.data != origB2.D.data).any() # Operations changed our operator


    def test_addscalar_inplace(self):
        vec = self.vec
        # idx = self.flip_idx
        # d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BO.for_vector(vec, scheme='backward', derivative=2, order=2, force_bandwidth=(-2,2))
        origB2 = B2.copy()
        oldB2 = B2.copy()

        B2.add(1.0, inplace=True) # Add to main diag in place
        oldB2.D.data[tuple(oldB2.D.offsets).index(0)] += 1.0  # Manually add 1 to main diag in place

        npt.assert_array_equal(B2.D.data, oldB2.D.data) # Operations were the same
        assert (B2.D.data is not origB2.D.data)
        assert (B2.D.data != origB2.D.data).any() # Operations changed our operator


    def test_copy(self):
        vec = self.vec
        # idx = self.flip_idx
        # d = np.hstack((np.nan, np.diff(vec)))
        C1 = FD.BO.for_vector(vec, scheme='center', derivative=1, order=2, force_bandwidth=(-2,2))
        CC1 = C1.copy()
        CCC1 = CC1.copy()

        assert C1 is not CC1
        npt.assert_array_equal(C1.D.data, CC1.D.data)
        npt.assert_array_equal(C1.D.offsets, CC1.D.offsets)

        assert C1 is not CCC1
        npt.assert_array_equal(C1.D.data, CCC1.D.data)
        npt.assert_array_equal(C1.D.offsets, CCC1.D.offsets)

        assert CC1 is not CCC1
        npt.assert_array_equal(CC1.D.data, CCC1.D.data)
        npt.assert_array_equal(CC1.D.offsets, CCC1.D.offsets)


    def test_create(self):
        vec = self.vec
        last = len(vec)-1
        idx = 1
        d = np.hstack((np.nan, np.diff(vec)))
        deltas = d
        sch0 = 'center'
        axis = 1
        for sch1 in ['center', 'up', 'down']:
            for dv in [1,2]:
                oldX1 = utils.nonuniform_complete_coefficients(deltas, up_or_down=sch1, flip_idx=idx)[dv-1]
                X1 = FD.BO.for_vector(vec, scheme=sch1, derivative=dv, order=2, axis=axis)

                high, low = 1,-1
                if (sch0 == 'up' and idx > 1) or (sch1 == 'up' and idx < last-1):
                    high = 2
                if (sch0 == 'down' and idx > 2) or (sch1 == 'down' and idx < last):
                    low = -2
                m = tuple(oldX1.offsets).index(0)
                oldX1.data = oldX1.data[m-high:m-low+1]
                oldX1.offsets = oldX1.offsets[m-high:m-low+1]

                # print "old D.todense()"
                # fp(oldX1.D.todense())
                # print "new D.todense()"
                # fp(X1.D.todense())
                # print
                # print X1.shape, oldX1.shape
                # print (X1.D.offsets, oldX1.offsets),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert X1.axis == axis
                assert np.array_equal(X1.D.todense() == oldX1.todense()),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert np.array_equal(X1.D.offsets == oldX1.offsets),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert (X1.D.data.shape == oldX1.data.shape),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert np.array_equal(X1.D.data == oldX1.data),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)


    def test_splice_same(self):
        vec = self.vec
        last = len(vec)-1
        # deltas = np.hstack((np.nan, np.diff(vec)))

        # When bandwidth is the same
        # print "Splicing operators of the same width."
        for sch0,sch1 in itertools.product(['center', 'up', 'down'], repeat=2):
            for dv in [1,2]:
                for idx in range(0, len(vec)-1):
                    X1 = FD.BO.for_vector(vec, scheme=sch0, derivative=dv, order=2, force_bandwidth=(-2,2))+1
                    X2 = FD.BO.for_vector(vec, scheme=sch1, derivative=dv, order=2, force_bandwidth=(-2,2))+1
                    X12 = X1.splice_with(X2, idx)
                    manualX12 = np.vstack((X1.D.todense()[:idx, :], X2.D.todense()[idx:,:]))
                    manualX12 = scipy.sparse.dia_matrix(manualX12)
                    X12i = X1.splice_with(X2, idx, inplace=True)
                    assert X12i is X1

                    high, low = 1,-1
                    if (sch0 == 'up' and idx > 1) or (sch1 == 'up' and idx < last-1):
                        high = 2
                    if (sch0 == 'down' and idx > 2) or (sch1 == 'down' and idx < last):
                        low = -2
                    m = tuple(X12.D.offsets).index(0)
                    X12.D.data = X12.D.data[m-high:m-low+1]
                    X12.D.offsets = X12.D.offsets[m-high:m-low+1]

                    # print
                    # print "manual"
                    # fp(manualX12.data[::-1], 3)
                    # print
                    # print "new"
                    # # fp(X12.D.todense(), 3)
                    # # print
                    # fp(X12.D.data, 3)

                    # print
                    # print X12.shape, manualX12.shape
                    # print (X12.D.offsets, manualX12.offsets[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert np.array_equal(X12.D.todense() == manualX12.todense()),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert np.array_equal(X12.D.offsets == manualX12.offsets[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.D.data.shape == manualX12.data.shape),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert np.array_equal(X12.D.data == manualX12.data[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)


    def test_splice_different(self):
        vec = self.vec
        last = len(vec)-1
        # deltas = np.hstack((np.nan, np.diff(vec)))

        # When bandwidth is possibly different
        # print "Splicing operators of the different width."
        for sch0,sch1 in itertools.product(['center', 'up', 'down'], repeat=2):
            for dv in [1,2]:
                for idx in range(0, len(vec)-1):
                    # add identity to avoid empty center
                    X1 = FD.BO.for_vector(vec, scheme=sch0, derivative=dv, order=2)+1
                    X2 = FD.BO.for_vector(vec, scheme=sch1, derivative=dv, order=2)+1
                    X12 = X1.splice_with(X2, idx)
                    manualX12 = np.vstack((X1.D.todense()[:idx, :], X2.D.todense()[idx:,:]))
                    manualX12 = scipy.sparse.dia_matrix(manualX12)

                    high, low = 1,-1
                    if (sch0 == 'up' and idx > 1) or (sch1 == 'up' and idx < last-1):
                        high = 2
                    if (sch0 == 'down' and idx > 2) or (sch1 == 'down' and idx < last):
                        low = -2
                    m = tuple(X12.D.offsets).index(0)
                    X12.D.data = X12.D.data[m-high:m-low+1]
                    X12.D.offsets = X12.D.offsets[m-high:m-low+1]

                    # print
                    # print "manual"
                    # fp(manualX12.data[::-1], 3)
                    # print
                    # print "new"
                    # # fp(X12.D.todense(), 3)
                    # # print
                    # fp(X12.D.data, 3)

                    # print
                    # print X12.shape, manualX12.shape
                    # print (X12.D.offsets, manualX12.offsets[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert np.array_equal(X12.D.todense() == manualX12.todense()),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert np.array_equal(X12.D.offsets == manualX12.offsets[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.D.data.shape == manualX12.data.shape),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert np.array_equal(X12.D.data == manualX12.data[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)


class Operator_Folding_test(unittest.TestCase):

    def setUp(self):
        loops = 200
        blocks = 2
        decimals = 12

        self.loops = loops
        self.blocks = blocks
        self.decimals = decimals

        mat = np.matrix("""
                        -1.5 0.5 2 0 0;
                        -1   2  -1 0 0;
                        0   -1   2 -1 0;
                        0   0    -1 2 -1;
                        0   0   -1.5 0.5 2""").A
        diamat = todia(scipy.sparse.dia_matrix(mat))

        x = foldMatFor(diamat, 1).todense().A
        topx = x.copy()
        bottomx = x.copy()
        topx[-1,-2] = 0
        bottomx[0,1] = 0
        x, topx, bottomx = map(todia, [x, topx, bottomx])


        topblockx = scipy.sparse.block_diag([topx]*blocks)
        bottomblockx = scipy.sparse.block_diag([bottomx]*blocks)
        blockx = todia(scipy.sparse.block_diag([x]*blocks))
        self.blockxI = blockx.todense().I.A
        self.topblockxI = topblockx.todense().I.A
        self.bottomblockxI = bottomblockx.todense().I.A

        npt.assert_array_equal(todia(topblockx.dot(bottomblockx)).data, blockx.data)

        blockdiamat = todia(scipy.sparse.block_diag([diamat]*blocks))

        blocktridiamat = todia(blockx.dot(blockdiamat))
        topblocktridiamat = todia(topblockx.dot(blockdiamat))
        bottomblocktridiamat = todia(bottomblockx.dot(blockdiamat))

        self.blockdiamat = blockdiamat

        self.blockx = blockx
        self.topblockx = topblockx
        self.bottomblockx = bottomblockx

        self.blocktridiamat = blocktridiamat
        self.topblocktridiamat = topblocktridiamat
        self.bottomblocktridiamat = bottomblocktridiamat

        self.vec = np.random.random(mat.shape[1]*blocks)

        self.B = FD.BandedOperator((blockdiamat.data, (2, 1, 0, -1, -2)), inplace=False)
        self.B.blocks = blocks


    def test_diag_creation(self):
        vec = self.vec

        npt.assert_array_almost_equal(vec, self.blockxI.dot(self.blockx.dot(vec)))
        npt.assert_array_almost_equal(self.blockdiamat.dot(vec), self.blockxI.dot(self.blocktridiamat.dot(vec)))
        npt.assert_array_almost_equal(vec, self.bottomblockxI.dot(self.bottomblockx.dot(vec)))
        npt.assert_array_almost_equal(self.blockdiamat.dot(vec), self.bottomblockxI.dot(self.bottomblocktridiamat.dot(vec)))
        npt.assert_array_almost_equal(vec, self.topblockxI.dot(self.topblockx.dot(vec)))
        npt.assert_array_almost_equal(self.blockdiamat.dot(vec), self.topblockxI.dot(self.topblocktridiamat.dot(vec)))


    def test_normal(self):
        # Normal (no folding)
        npt.assert_array_equal(self.B.D.data, self.blockdiamat.data)
        npt.assert_array_equal(self.blockdiamat.dot(self.vec), self.B.apply(self.vec))

    def test_diagonalize2(self):
        mat = self.B.D.data.view().reshape(-1)
        zeros = mat == 0
        mat[:] = np.arange(self.B.D.data.size)
        mat[zeros] = 0
        B = self.B.copy()
        print "ref pre"
        fp(self.B.D)
        fp(self.B.D.data)

        print "Collected off-tridiag points as bottom_factors"
        block_len = B.shape[0] / B.blocks
        bottom_factors = B.D.data[-1,block_len-3::block_len]
        print B.blocks, len(bottom_factors)
        print bottom_factors

        self.B.diagonalize()
        B.diagonalize2()
        print "ref mid"
        fp(self.B.D)
        fp(self.B.D.data)
        print "tst mid"
        fp(B.D)
        fp(B.D.data)

        npt.assert_array_equal(self.B.D.data, B.D.data, err_msg="Diagonalize alone doesn't preserve operator matrix.")
        npt.assert_(B == self.B, msg="Diagonalize alone doesn't preserve operator.")

        B.undiagonalize()
        self.B.undiagonalize()
        npt.assert_(not B.is_tridiagonal())
        print "ref after"
        fp(self.B.D)
        fp(self.B.D.data)
        print "tst after"
        fp(B.D)
        fp(B.D.data)
        print "ref top"
        fp(self.B.top_factors or np.array([np.nan]))
        print "tst top"
        fp(B.top_factors or np.array([np.nan]))
        print "ref bot"
        fp(self.B.bottom_factors or np.array([np.nan]))
        print "tst bot"
        fp(B.bottom_factors or np.array([np.nan]))
        npt.assert_array_equal(self.B.D.data, B.D.data, err_msg="Undiagonalize roundtrip doesn't preserve operator matrix.")
        npt.assert_(B == self.B, msg="Undiagonalize roundtrip doesn't preserve operator.")




    def test_diagonalize(self):
        B = self.B.copy()
        print "ref pre"
        fp(self.B.D)
        fp(self.B.D.data)
        print "tst"
        fp(B.D)
        fp(B.D.data)
        B.diagonalize()
        print "tst mid"
        fp(B.D)
        fp(B.D.data)
        B.undiagonalize()
        npt.assert_(not B.is_tridiagonal())
        print "ref after"
        fp(self.B.D)
        fp(self.B.D.data)
        print "tst"
        fp(B.D)
        fp(B.D.data)
        npt.assert_array_equal(self.B.D.data, B.D.data, err_msg="Diagonalize roundtrip doesn't preserve operator matrix.")
        npt.assert_(B == self.B, msg="Diagonalize roundtrip doesn't preserve operator.")


    def test_fold_vector(self):
        B = self.B
        blockx = self.blockx
        bottomblockx = self.bottomblockx
        topblockx = self.topblockx
        vec = self.vec

        # Folding and unfolding
        Btop = B.copy()
        Btop.fold_top()
        Bbottom = B.copy()
        Bbottom.fold_bottom()
        B.diagonalize()

        v = np.asarray(Bbottom.fold_vector(vec.copy()))
        npt.assert_array_equal(bottomblockx.dot(vec), v, err_msg="Bottom folded differs from matrix'd op.")
        Bbottom.fold_vector(v, unfold=True)
        npt.assert_array_almost_equal(vec, v, decimal=14, err_msg="Bottom unfolded differs from orig.")

        v = np.asarray(Btop.fold_vector(vec.copy()))
        npt.assert_array_equal(topblockx.dot(vec), v, err_msg="Top folded differs from matrix'd op.")
        Btop.fold_vector(v, unfold=True)
        npt.assert_array_almost_equal(vec, v, decimal=14, err_msg="Top unfolded differs from orig.")

        v = np.asarray(B.fold_vector(vec.copy()))
        npt.assert_array_equal(blockx.dot(vec), v, err_msg="Both folded differs from matrix'd op.")
        B.fold_vector(v, unfold=True)
        npt.assert_array_almost_equal(vec, v, decimal=14, err_msg="Both unfolded differs from orig.")

        vec = self.vec.copy()
        B = self.B
        B.diagonalize()


    def test_explicit(self):
        Borig = self.B
        vec = self.vec.copy()
        blocktridiamat = self.blocktridiamat
        topblocktridiamat = self.topblocktridiamat
        bottomblocktridiamat = self.bottomblocktridiamat
        blockxI = self.blockxI
        topblockxI = self.topblockxI
        bottomblockxI = self.bottomblockxI
        loops = self.loops

        # Derivative Op explicit
        bvec = vec.copy()

        # Explicit
        Btop = Borig.copy()
        Btop.fold_top()
        npt.assert_array_equal(Btop.D.data[1:], topblocktridiamat.data, err_msg="Bottom op and mat not equal.")
        assert Btop.is_folded()
        assert not Btop.is_tridiagonal()
        for i in range(loops):
            bvec = topblockxI.dot(topblocktridiamat.dot(bvec))
            vec = Btop.apply(vec)
            npt.assert_array_almost_equal(bvec, vec, err_msg="Bottom Loop: %s" % i)
            np.assert_array_equal(vec, vec) # for NaNs
            np.assert_array_equal(bvec, bvec) # for NaNs
        vec = self.vec.copy()
        bvec = vec.copy()

        Bbottom = Borig.copy()
        Bbottom.fold_bottom()
        npt.assert_array_equal(Bbottom.D.data[:-1], bottomblocktridiamat.data, err_msg="Bottom op and mat not equal.")
        assert Bbottom.is_folded()
        assert not Bbottom.is_tridiagonal()
        for i in range(loops):
            bvec = bottomblockxI.dot(bottomblocktridiamat.dot(bvec))
            vec = Bbottom.apply(vec)
            npt.assert_array_almost_equal(bvec, vec, err_msg="Bottom Loop: %s" % i)
            np.assert_array_equal(vec, vec) # for NaNs
            np.assert_array_equal(bvec, bvec) # for NaNs
        vec = self.vec.copy()
        bvec = vec.copy()

        B = Borig.copy()
        B.diagonalize()
        npt.assert_array_equal(B.D.data, blocktridiamat.data)
        for i in range(loops):
            bvec = blockxI.dot(blocktridiamat.dot(bvec))
            vec = B.apply(vec)
            npt.assert_array_equal(bvec, vec)
            np.assert_array_equal(vec, vec)
            np.assert_array_equal(bvec, bvec)
        vec = self.vec.copy()
        bvec = vec.copy()


    def test_top_implicit(self):
        inv = np.linalg.inv
        norm = np.linalg.norm
        Borig = self.B
        vec = self.vec.copy()
        loops = self.loops

        blockdiamat = self.blockdiamat
        topblocktridiamat = self.topblocktridiamat
        topblockx = self.topblockx

        ref = vec.copy()
        ivec = vec.copy()
        B = Borig.copy()
        B.fold_top()
        for i in range(loops):
            ivec = inv(topblocktridiamat.todense()).dot(topblockx.dot(ivec)).A[0]
            ref = scipy.linalg.solve_banded((2,2), blockdiamat.data, ref)
            vec = B.solve(vec)
            ref /= norm(ref)
            vec /= norm(vec)
            ivec /= norm(ivec)
            # fp(ref - vec, 'e')
            npt.assert_array_almost_equal(ref, vec, decimal=self.decimals, err_msg="Loop: %s" % i)
            npt.assert_array_almost_equal(ref, ivec, decimal=self.decimals, err_msg="Loop: %s" % i)


    def test_bottom_implicit(self):
        inv = np.linalg.inv
        norm = np.linalg.norm
        Borig = self.B
        vec = self.vec.copy()
        loops = self.loops

        bottomblockx = self.bottomblockx
        bottomblocktridiamat = self.bottomblocktridiamat
        # bottomblockxI = self.bottomblockxI

        blockdiamat = self.blockdiamat

        ref = vec.copy()
        ivec = vec.copy()
        B = Borig.copy()
        B.fold_bottom()
        for i in range(loops):
            ivec = inv(bottomblocktridiamat.todense()).dot(bottomblockx.dot(ivec)).A[0]
            ref = scipy.linalg.solve_banded((2,2), blockdiamat.data, ref)
            vec = B.solve(vec)
            ref /= norm(ref)
            vec /= norm(vec)
            ivec /= norm(ivec)
            # fp(ivec - vec, 'e')
            npt.assert_array_almost_equal(ref, vec, decimal=self.decimals, err_msg="Loop: %s" % i)
            npt.assert_array_almost_equal(ref, ivec, decimal=self.decimals, err_msg="Loop: %s" % i)


    def test_both_implicit(self):
        inv = np.linalg.inv
        norm = np.linalg.norm
        Borig = self.B
        vec = self.vec.copy()
        loops = self.loops

        blockx = self.blockx
        blocktridiamat = self.blocktridiamat
        # blockxI = self.blockxI

        blockdiamat = self.blockdiamat

        ref = vec.copy()
        ivec = vec.copy()
        B = Borig.copy()
        B.diagonalize()
        for i in range(loops):
            ivec = inv(blocktridiamat.todense()).dot(blockx.dot(ivec)).A[0]
            ref = scipy.linalg.solve_banded((2,2), blockdiamat.data, ref)
            vec = B.solve(vec)
            ref /= norm(ref)
            vec /= norm(vec)
            ivec /= norm(ivec)
            # fp(ivec - vec, 'e')
            npt.assert_array_almost_equal(ref, vec, decimal=self.decimals, err_msg="Loop: %s" % i)
            npt.assert_array_almost_equal(ref, ivec, decimal=self.decimals, err_msg="Loop: %s" % i)


class ScalingFuncs(unittest.TestCase):

    def setUp(self):
        k = 3.0
        nspots = 7
        spot_max = 1500.0
        spotdensity = 7.0  # infinity is linear?
        vec = utils.sinh_space(k, spot_max, spotdensity, nspots)
        self.vec = vec

        def coeff(high,low=None):
            if low is not None:
                high, low = low, high
            return np.linspace(0, 1, len(vec))[low:high]
        def fcoeff(i):
            return np.linspace(0, 1, len(vec))[i]
        def f0(x): return x*0.0
        def fx(x): return x+2.0

        data = np.ones((5,len(vec)))
        data[0][:2] = 0
        data[1][0] = 0
        data[3][-1] = 0
        data[4][-2:] = 0
        offsets = [2,1,0,-1,-2]
        res = np.ones_like(vec)
        oldB = FD.BandedOperator((data, offsets), res)
        self.oldB = oldB

        manualB = oldB.copy()
        # newB = oldB.copy()
        # vecB = oldB.copy()
        manualB.D.data[0][2:] *= coeff(len(vec)-2)
        manualB.D.data[1][1:] *= coeff(len(vec)-1)
        manualB.D.data[2] *= coeff(len(vec))
        manualB.D.data[3][:-1] *= coeff(1, len(vec))
        manualB.D.data[4][:-2] *= coeff(2, len(vec))
        manualB.R *= coeff(len(vec))

        self.manualB = manualB

        self.coeff = coeff
        self.fcoeff = fcoeff
        self.f0 = f0
        self.fx = fx


    def test_vectorizedscale(self):
        no_nan = np.nan_to_num
        vec = self.vec
        oldB = self.oldB
        manualB = self.manualB
        flag = int(1)
        data = np.ones((5,len(vec)), dtype=float)*flag
        data[0][:2] = 0
        data[1][0] = 0
        data[3][-1] = 0
        data[4][-2:] = 0
        offsets = [2,1,0,-1,-2]
        res = np.ones_like(vec)

        newB = oldB.copy()
        newB.scale(self.f0)
        vecB = oldB.copy()
        vecB.vectorized_scale(self.f0(vec))


        npt.assert_array_equal(0, no_nan(newB.D.data))
        npt.assert_array_equal(0, no_nan(vecB.D.data))

        for dchlet in itertools.product([1.0, None], repeat=2):
            oldB = FD.BandedOperator((data.copy(), offsets), res.copy())
            oldB.dirichlet = dchlet
            veczeroB = oldB.copy()
            veczeroB.vectorized_scale(self.f0(vec))

            manualzeroB = np.zeros_like(veczeroB.D.data)
            if veczeroB.dirichlet[0] is not None:
                manualzeroB[0, 2] = flag
                manualzeroB[1, 1] = flag
                manualzeroB[2, 0] = flag
            if veczeroB.dirichlet[1] is not None:
                manualzeroB[2, -1] = flag
                manualzeroB[3, -2] = flag
                manualzeroB[4, -3] = flag

            # print dchlet
            # print
            # print "veczeroB"
            # fp(veczeroB.D.data)
            # print
            # print "manualzeroB"
            # fp(manualzeroB)
            # print

            manualB = oldB.copy()
            newB = oldB.copy()
            vecB = oldB.copy()
            bottom = 0
            top = last = manualB.shape[0]
            if dchlet[0]:
                bottom += 1
            if dchlet[1]:
                top -= 1
            manualB.D.data[0][bottom+2:]  *= vec[bottom : last-2]+2
            manualB.D.data[1][bottom+1:]  *= vec[bottom : last-1]+2
            manualB.D.data[2][bottom:top] *= vec[bottom : top]+2
            manualB.D.data[3][:top-1]     *= vec[1      : top]+2
            manualB.D.data[4][:top-2]     *= vec[2      : top]+2
            manualB.R[bottom:top]       *= vec[bottom:top]+2
            vecB.vectorized_scale(self.fx(vec))
            newB.scale(lambda i: vec[i]+2)
            # print "vec"
            # fp(vec)
            # print
            # print "manual"
            # fp(manualB.D.data)
            # print
            # print "newB"
            # fp(newB.D.data)
            # print
            # print "vecB"
            # fp(vecB.D.data)
            # print
            # print "manualR"
            # print manualB.R
            # print
            # print "vecR"
            # print vecB.R
            # print
            # print "newR"
            # print newB.R
            # print
            # print "manual"
            # fp(manualB.D)
            # print
            # print "newB"
            # fp(newB.D)
            npt.assert_array_equal(manualzeroB, veczeroB.D.data)
            npt.assert_(newB == vecB)
            npt.assert_array_equal(manualB, newB)
            npt.assert_array_equal(manualB, vecB)

        newB = oldB.copy()
        newB.scale(self.fcoeff)
        vecB = oldB.copy()
        vecB.vectorized_scale(self.coeff(len(vec)))
        # print "new"
        # fp(newB.data)
        # print
        # print "vec"
        # fp(vecB.data)
        # print
        # print "newR"
        # print newB.R
        assert newB == vecB


    def test_block_vectorizedscale(self):
        no_nan = np.nan_to_num
        vec = self.vec
        oldB = self.oldB
        manualB = self.manualB
        flag = int(1)
        data = np.ones((5,len(vec)), dtype=float)*flag
        data[0][:2] = 0
        data[1][0] = 0
        data[3][-1] = 0
        data[4][-2:] = 0
        offsets = [2,1,0,-1,-2]
        res = np.ones_like(vec)

        blocks = 3

        blockvec = np.tile(vec, blocks)

        newB = oldB.copy()
        newB.scale(self.f0)
        vecB = oldB.copy()
        vecB.vectorized_scale(self.f0(vec))


        npt.assert_array_equal(0, no_nan(newB.D.data))
        npt.assert_array_equal(0, no_nan(vecB.D.data))

        for dchlet in itertools.product([1., None], repeat=2):
            oldB = FD.BandedOperator((data.copy(), offsets), res.copy())
            oldB.dirichlet = dchlet

            veczeroB = block_repeat(oldB, blocks)
            veczeroB.vectorized_scale(self.f0(blockvec))
            manualzeroB = np.zeros_like(oldB.D.data)
            if veczeroB.dirichlet[0] is not None:
                manualzeroB[0, 2] = flag
                manualzeroB[1, 1] = flag
                manualzeroB[2, 0] = flag
            if veczeroB.dirichlet[1] is not None:
                manualzeroB[2, -1] = flag
                manualzeroB[3, -2] = flag
                manualzeroB[4, -3] = flag
            manualzeroB = np.tile(manualzeroB, blocks)

            # print dchlet
            # print
            # print "veczeroB"
            # fp(veczeroB.D.data)
            # print
            # print "manualzeroB"
            # fp(manualzeroB)
            # print

            manualB = oldB.copy()
            bottom = 0
            top = last = manualB.shape[0]
            if dchlet[0]:
                bottom += 1
            if dchlet[1]:
                top -= 1
            manualB.D.data[0][bottom+2:]  *= vec[bottom : last-2]+2
            manualB.D.data[1][bottom+1:]  *= vec[bottom : last-1]+2
            manualB.D.data[2][bottom:top] *= vec[bottom : top]+2
            manualB.D.data[3][:top-1]     *= vec[1      : top]+2
            manualB.D.data[4][:top-2]     *= vec[2      : top]+2
            manualB.R[bottom:top]       *= vec[bottom:top]+2
            manualB = block_repeat(manualB, blocks)

            vecB = block_repeat(oldB, blocks)
            vecB.vectorized_scale(self.fx(blockvec))

            newB = block_repeat(oldB, blocks)
            newB.scale(lambda i: blockvec[i]+2)
            # print "vec"
            # fp(vec)
            # print
            # print "manual"
            # fp(manualB.D.data)
            # print
            # print "newB"
            # fp(newB.D.data)
            # print
            # print "vecB"
            # fp(vecB.D.data)
            # print
            # print "manualR"
            # print manualB.R
            # print
            # print "vecR"
            # print vecB.R
            # print
            # print "newR"
            # print newB.R
            # print
            # print "manual"
            # fp(manualB.D)
            # print
            # print "newB"
            # fp(newB.D)
            npt.assert_array_equal(manualzeroB, veczeroB.D.data)
            npt.assert_(newB == vecB)
            npt.assert_array_equal(manualB, newB)
            npt.assert_array_equal(manualB, vecB)

        newB = oldB.copy()
        newB.scale(self.fcoeff)
        vecB = oldB.copy()
        vecB.vectorized_scale(self.coeff(len(vec)))
        # print "new"
        # fp(newB.data)
        # print
        # print "vec"
        # fp(vecB.data)
        # print
        # print "newR"
        # print newB.R
        assert newB == vecB


    def test_scale_dirichlet(self):
        # no_nan = np.nan_to_num
        vec = self.vec
        flag = int(1)
        data = np.ones((5,len(vec)), dtype=int)*flag
        data[0][:2] = 0
        data[1][0] = 0
        data[3][-1] = 0
        data[4][-2:] = 0
        offsets = [2,1,0,-1,-2]
        res = np.ones_like(vec)

        for dchlet in itertools.product([1., None], repeat=2):
            oldB = FD.BandedOperator((data.copy(), offsets), res.copy())
            oldB.dirichlet = dchlet
            zeroB = oldB.copy()
            zeroB.scale(self.f0)

            manualzeroB = np.zeros_like(zeroB.D.data)
            if zeroB.dirichlet[0] is not None:
                manualzeroB[0, 2] = flag
                manualzeroB[1, 1] = flag
                manualzeroB[2, 0] = flag
            if zeroB.dirichlet[1] is not None:
                manualzeroB[2, -1] = flag
                manualzeroB[3, -2] = flag
                manualzeroB[4, -3] = flag

            # print "Dirichlet", dchlet
            # print
            # print "zeroB"
            # fp(zeroB.D.data)
            # fp(zeroB.R)
            # print
            # print "manualzeroB"
            # fp(manualzeroB)
            # fp(manualzeroB.R)
            # print

            manualB = oldB.copy()
            newB = oldB.copy()
            bottom = 0
            top = manualB.shape[0]
            if dchlet[0]:
                bottom += 1
            if dchlet[1]:
                top -= 1
            manualB.D.data[0][bottom+2:] *= np.arange(bottom, manualB.shape[0]-2)+2
            manualB.D.data[1][bottom+1:] *= np.arange(bottom, manualB.shape[0]-1)+2
            manualB.D.data[2][bottom:top] *= np.arange(bottom, top)+2
            manualB.D.data[3][:top-1] *= np.arange(1, top)+2
            manualB.D.data[4][:top-2] *= np.arange(2, top)+2
            manualB.R[bottom:top] *= np.arange(bottom, top)+2
            newB.scale(self.fx)
            # print "manual"
            # fp(manualB.D.data)
            # print
            # print "new"
            # fp(newB.D.data)
            # print
            # print "manualR"
            # print manualB.R
            # print
            # print "newR"
            # print newB.R
            npt.assert_array_equal(manualzeroB, zeroB.D.data)
            assert manualB == newB


    def test_scale_no_dirichlet(self):
        no_nan = np.nan_to_num
        vec = self.vec
        def f0(x): return 0
        def fx(x): return x
        data = np.ones((5,len(vec)), dtype=float)
        data[0][:2] = 0
        data[1][0] = 0
        data[3][-1] = 0
        data[4][-2:] = 0
        offsets = [2,1,0,-1,-2]
        res = np.ones_like(vec)
        oldB = FD.BandedOperator((data, offsets), res)

        newB = oldB.copy()
        newB.scale(f0)
        # fp(newB.D.data)
        npt.assert_array_equal(0, no_nan(newB.D.data))

        manualB = oldB.copy()
        newB = oldB.copy()
        manualB.D.data[0][2:] *= np.arange(len(vec)-2)
        manualB.D.data[1][1:] *= np.arange(len(vec)-1)
        manualB.D.data[2] *= np.arange(len(vec))
        manualB.D.data[3][:-1] *= np.arange(1, len(vec))
        manualB.D.data[4][:-2] *= np.arange(2, len(vec))
        manualB.R *= np.arange(len(vec))
        newB.scale(fx)
        # print "manual"
        # fp(manualB.data)
        # print
        # print "new"
        # fp(newB.data)
        # print
        # print "manualR"
        # print manualB.R
        # print
        # print "newR"
        # print newB.R
        npt.assert_array_equal(manualB, newB)


def block_repeat(B, blocks):
    B = B.copy()
    B.D = scipy.sparse.dia_matrix((np.tile(B.D.data, blocks), B.D.offsets), [x*blocks for x in B.shape])
    B.R = np.tile(B.R, blocks)
    B.blocks = blocks
    B.shape = tuple(x*blocks for x in B.shape)
    return B


def implicit_manual(V, L1, R1x, L2, R2x, dt, n, spots, vars, coeffs, crumbs=[], callback=None):
    V = V.copy()

    # L1i = flatten_tensor(L1)
    L1i = L1.copy()
    R1 = np.array(R1x)

    # L2i = flatten_tensor(L2)
    L2i = L2.copy()
    R2 = np.array(R2x)

    m = 2

    # L  = (As + Ass - H.interest_rate*np.eye(nspots))*-dt + np.eye(nspots)
    L1i.data *= -dt
    L1i.data[m, :] += 1
    R1 *= dt

    L2i.data *= -dt
    L2i.data[m, :] += 1
    R2 *= dt

    offsets1 = (abs(min(L1i.offsets)), abs(max(L1i.offsets)))
    offsets2 = (abs(min(L2i.offsets)), abs(max(L2i.offsets)))

    dx = np.gradient(spots)[:,np.newaxis]
    dy = np.gradient(vars)
    X, Y = [dim.T for dim in np.meshgrid(spots, vars)]
    gradgrid = dt * coeffs[(0,1)](0, X, Y) / (dx * dy)
    gradgrid[:,0] = 0; gradgrid[:,-1] = 0
    gradgrid[0,:] = 0; gradgrid[-1,:] = 0

    print_step = max(1, int(n / 10))
    to_percent = 100.0 / n
    utils.tic("Impl:")
    for k in xrange(n):
        if not k % print_step:
            if np.isnan(V).any():
                print "Impl fail @ t = %f (%i steps)" % (dt * k, k)
                return crumbs
            print int(k * to_percent),
        if callback is not None:
            callback(V, ((n - k) * dt))
        Vsv = np.gradient(np.gradient(V)[0])[1] * gradgrid
        V = spl.solve_banded(offsets2, L2i.data,
                             (V + Vsv + R2).flat, overwrite_b=True).reshape(V.shape)
        V = spl.solve_banded(offsets1, L1i.data,
                             (V + R1).T.flat, overwrite_b=True).reshape(V.shape[::-1]).T
    crumbs.append(V.copy())
    utils.toc()
    return crumbs


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

def foldMatFor(A, blocks):
    l = A.shape[0] // blocks
    data = np.zeros((3, A.shape[0]))
    data[1, :] = 1
    offsets = (1, 0, -1)
    m = len(A.offsets) // 2
    for b in range(blocks):
        data[0, b*l+1] = -A.data[m-2,b*l+2] / A.data[m-1,b*l+2] if A.data[m-1,b*l+2] else 0
        data[2, (b+1)*l-2] = -A.data[m+2,(b+1)*l-3] / A.data[m+1,(b+1)*l-3] if A.data[m+2,(b+1)*l-3] else 0
        d = scipy.sparse.dia_matrix((data, offsets), shape=A.shape)
    return d


def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
