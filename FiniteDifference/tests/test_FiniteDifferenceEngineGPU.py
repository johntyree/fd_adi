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
        self.F.operators[1].diagonalize()
        self.FG = FDG.FiniteDifferenceEngineADI(self.F)


    def test_verify_simple_operators_0(self):
        ref = self.F.simple_operators[(0,)]
        fp(ref.D)
        tst = self.FG.simple_operators[(0,)].immigrate()
        npt.assert_equal(tst, ref)

    def test_verify_simple_operators_1(self):
        ref = self.F.simple_operators[(1,)]
        tst = self.FG.simple_operators[(1,)].immigrate()
        npt.assert_equal(tst, ref)

    def test_verify_simple_operators_00(self):
        ref = self.F.simple_operators[(0,0)]
        fp(ref.D)
        tst = self.FG.simple_operators[(0,0)].immigrate()
        npt.assert_equal(tst, ref)

    def test_verify_simple_operators_11(self):
        raise unittest.SkipTest
        ref = self.F.simple_operators[(1,1)]
        ref.diagonalize()
        tst = self.FG.simple_operators[(1,1)]
        tst.diagonalize()
        tst = tst.immigrate()
        npt.assert_equal(tst, ref)

    def test_verify_simple_operators_01(self):
        raise unittest.SkipTest
        ref = self.F.simple_operators[(0,1)]
        tst = self.FG.simple_operators[(0,1)].immigrate()
        npt.assert_equal(tst, ref)

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
