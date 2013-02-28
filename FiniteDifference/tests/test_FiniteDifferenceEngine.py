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

from FiniteDifference.blackscholes import BlackScholesFiniteDifferenceEngine, BlackScholesOption
from FiniteDifference.heston import HestonBarrierOption


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



def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
