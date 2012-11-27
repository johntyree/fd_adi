#!/usr/bin/env python
# coding: utf8

# import sys
# import os
import itertools
from bisect import bisect_left
import unittest

import numpy as np
import scipy.sparse

import utils
from visualize import fp
# def fp(*x, **y):
    # pass
import FiniteDifferenceEngine as FD
import Grid
from heston import HestonOption
from Option import BlackScholesOption




class BlackScholesOption_test(unittest.TestCase):

    def setUp(self):
        spot_max = 5000.0
        spotdensity = 1.0  # infinity is linear?
        v = 0.04
        r = 0.06
        k = 99.0
        spot = 100.0
        t = 1.0
        dt = 1.0/100.0
        nspots = 400
        # spots = np.linspace(0, np.log(spot)*2, nspots+1)
        spots = utils.sinh_space(spot, spot_max, spotdensity, nspots)

        # self.spot_idx = np.argmin(np.abs(spots - np.log(spot)))
        # self.spot = np.exp(spots[self.spot_idx])
        self.spot_idx = np.argmin(np.abs(spots - spot))
        spot = spots[self.spot_idx]

        # self.BS = BlackScholesOption(spot=np.exp(spot), strike=k, interest_rate=r,
                                     # variance=v, tenor=t)
        self.BS = BlackScholesOption(spot=spot, strike=k, interest_rate=r,
                                     variance=v, tenor=t)

        # G = Grid.Grid([spots], initializer=lambda *x: np.maximum(np.exp(x[0])-k,0))
        G = Grid.Grid([spots], initializer=lambda *x: np.maximum(x[0]-k,0))

        def mu_s(t, *dim):
            # return np.zeros_like(dim[0], dtype=float) + (r - 0.5 * v)
            return r * dim[0]

        def gamma2_s(t, *dim):
            # return 0.5 * v + np.zeros_like(dim[0], dtype=float)
            return 0.5 * v * dim[0]**2

        coeffs = {()   : lambda t: -r,
                  (0,) : mu_s,
                  (0,0): gamma2_s}

        bounds = {          # D: U = 0              VN: dU/dS = 1
                (0,)  : ((0, lambda *args: 0.0), (1, lambda t, x: 1.0)),
                # (0,)  : ((0, lambda *args: 0.0), (1, lambda t, *x: np.exp(x[0]))),
                        # D: U = 0              Free boundary
                (0,0) : ((0, lambda *args: 0.0), (None, lambda *x: None))}

        self.dt = dt
        self.F = FD.FiniteDifferenceEngineADI(G, coefficients=coeffs, boundaries=bounds)

    def test_implicit(self):
        t, dt = self.BS.tenor, self.dt
        V = self.F.solve_implicit(t/dt, dt)[-1][self.spot_idx]
        ans = self.BS.analytical
        print "Spot:", self.BS.spot
        print "Price:", V, ans, V - ans
        assert np.isclose(V, ans, rtol=0.001)

    def test_crank(self):
        t, dt = self.BS.tenor, self.dt
        V = self.F.solve_adi(t/dt, dt)[-1][self.spot_idx]
        ans = self.BS.analytical
        print "Spot:", self.BS.spot
        print "Price:", V, ans, V - ans
        assert np.isclose(V, ans, rtol=0.001)

    def test_smooth(self):
        t, dt = self.BS.tenor, self.dt
        V = self.F.smooth(t/dt, dt)[-1][self.spot_idx]
        ans = self.BS.analytical
        print "Spot:", self.BS.spot
        print "Price:", V, ans, V - ans
        assert np.isclose(V, ans, rtol=0.001)


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
        rho = 0.2
        spot_max = 1500.0
        var_max = 13.0
        nspots = 10
        nvols = 6
        spotdensity = 7.0  # infinity is linear?
        varexp = 4

        # TODO:!!!!XXX TODO XXX
        # var_max = nvols-1
        # spot_max = nspots-1

        up_or_down_spot = 'up'
        up_or_down_var = 'down'
        flip_idx_spot = 7
        flip_idx_var = 4
        k = spot_max / 4.0
        # spots = np.linspace(0, spot_max, nspots)
        vars = np.linspace(0, var_max, nvols)
        spots = utils.sinh_space(k, spot_max, spotdensity, nspots)
        # vars = utils.exponential_space(0.00, 0.04, var_max, varexp, nvols)
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
                (0,0) : ((0 if dirichlet_s else 1, lambda *args: 0.0), (None, lambda *x: None)),
                        # Free boundary at low variance
                (1,)  : ((None, lambda *x: None),
                        # (0.0, lambda t, *dim: 0),
                        # # D intrinsic value at high variance
                        (0 if dirichlet_v else 1, lambda t, *dim: np.maximum(0.0, dim[0]-k))),
                        # # Free boundary
                (1,1) : ((None, lambda *x: None),
                        # D intrinsic value at high variance
                        (0 if dirichlet_v else 1, lambda t, *dim: np.maximum(0.0, dim[0]-k)))
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
            L1_[j].data[m, 0] = -1 * dirichlet_s

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
            L2_[i].data[m, -1] = -1 * dirichlet_v  # This is to cancel out the previous value so we can
                                # set the dirichlet boundary condition using R.
                                # Then we have U_i + -U_i + R


            R2_.append(Rv + Rvv)
            R2_[i][-1] = np.maximum(0, s - k)

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
        G = Grid.Grid((spots, vars), initializer=lambda x0,x1: x0*x1)
        # print G

        self.F = FD.FiniteDifferenceEngineADI(G, coefficients=coeffs,
                boundaries=bounds, schemes=schemes, force_bandwidth=None)


    def test_combine_dimensional_operators(self):
        # assert False
        oldL1 = self.L1_.copy()
        oldL1 = scipy.sparse.dia_matrix(oldL1.todense())
        oldL1.data = oldL1.data[::-1]
        oldL1.offsets = oldL1.offsets[::-1]
        # high, low = 2, -2
        # m = tuple(oldL1.offsets).index(0)
        # oldL1.data = oldL1.data[m-high:m-low+1]
        # oldL1.offsets = oldL1.offsets[m-high:m-low+1]

        oldL2 = self.L2_.copy()
        oldL2 = scipy.sparse.dia_matrix(oldL2.todense())
        oldL2.data = oldL2.data[::-1]
        oldL2.offsets = oldL2.offsets[::-1]
        # high, low = 2, -2
        # m = tuple(oldL2.offsets).index(0)
        # oldL2.data = oldL2.data[m-high:m-low+1]
        # oldL2.offsets = oldL2.offsets[m-high:m-low+1]

        oldR1 = self.R1_.T.flatten()
        oldR2 = self.R2_.flatten()

        L1 = self.F.operators[0]

        # oldL1.data = oldL1.data[:-1]
        # oldL1.offsets = oldL1.offsets[:-1]
        # oldL2.data = oldL2.data[1:]
        # oldL2.offsets = oldL2.offsets[1:]

        # print "offsets"
        # print oldL1.offsets, L1.offsets
        # print "old"
        # fp(oldL1.data)
        # print
        # print "new"
        # fp(L1.data)
        # print
        # print "diff"
        # fp(L1.data - oldL1.data)
        # print
        # print "old"
        # fp(oldL1.todense())
        # print
        # print "new"
        # fp(L1.todense())
        # print
        # print "diff"
        # fp(oldL1.todense() - L1.todense())
        assert np.allclose(L1.todense(), oldL1.todense())
        assert np.allclose(L1.data, oldL1.data)
        # print "old"
        # print oldR1
        # print
        # print "new"
        # print L1.R
        assert np.allclose(L1.R, oldR1)

        L2 = self.F.operators[1]

        # print "old"
        # fp(oldL2.data)
        # print
        # print "new"
        # fp(L2.data)
        # print "old"
        # fp(oldL2.todense())
        # print
        # print "new"
        # fp(L2.todense())
        # print
        # print "diff"
        # fp(oldL2.todense() - L2.todense())
        assert np.allclose(L2.data, oldL2.data)
        # print "old"
        # print oldR2
        # print
        # print "new"
        # print L2.R
        # print "diff"
        # fp(L2.R - oldR2)
        assert np.allclose(L2.R, oldR2)


    def test_cross_derivative(self):
        crossOp = self.F.operators[(0,1)]
        g = self.F.grid.domain
        x = self.F.grid.mesh[0]
        y = self.F.grid.mesh[1]

        dx = np.gradient(x)[:,np.newaxis]
        dy = np.gradient(y)
        dgdx = np.gradient(g, 1)[0]
        manuald2gdxdy = np.gradient(dgdx)[1] / (dx * dy)
        manuald2gdxdy[:,0] = 0; manuald2gdxdy[:,-1] = 0
        manuald2gdxdy[0,:] = 0; manuald2gdxdy[-1,:] = 0
        X,Y = [a.T for a in np.meshgrid(x, y)]
        manuald2gdxdy *= self.F.coefficients[(0,1)](0, X, Y)

        d2gdxdy = crossOp.apply(g)

        # fp(crossOp.todense())

        # print "manual"
        # fp(manuald2gdxdy)
        # print "new"
        # fp(d2gdxdy)
        # print "diff"
        # fp(d2gdxdy - manuald2gdxdy, fmt='e')
        assert np.allclose(d2gdxdy, manuald2gdxdy)

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
        assert (G.mesh[0] == self.spots).all()
        assert (G.mesh[1] == self.vars).all()
        assert G.ndim == len(G.mesh)
        assert G.shape == tuple(map(len, G.mesh))

    def test_domain(self):
        U = np.tile(np.maximum(0, self.spots - self.strike), (len(self.vars), 1)).T
        G = self.Grid
        # print G
        # print U
        assert (G.domain == U).all()



class BandedOperator_test(unittest.TestCase):

    def setUp(self):
        k = 3.0
        nspots = 7
        spot_max = 1500.0
        spotdensity = 7.0  # infinity is linear?
        spots = utils.sinh_space(k, spot_max, spotdensity, nspots)
        self.flip_idx = 4
        self.vec = spots

    def test_addself(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        for scheme in ["center", "forward", "backward"]:
            C1 = FD.BandedOperator.for_vector(vec, scheme=scheme, derivative=1, order=2)
            C2 = C1.add(C1)
            assert C2 is not C1
            assert C2.data is not C1.data
            assert (C2.offsets == C1.offsets).all()
            assert (C2.data == C1.data+C1.data).all()
            assert (C2.data == C1.data*2).all()
            assert (C2.R == C1.R*2).all()
            assert (C2.R == C1.R*+C1.R).all()


    def test_addoperator(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        C1 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=1, order=2)
        F1 = FD.BandedOperator.for_vector(vec, scheme='forward', derivative=1, order=2)
        oldCF1 = np.zeros((len(set(F1.offsets) | set(C1.offsets)), C1.shape[1]))
        oldCF1R = np.zeros_like(F1.R)

        # print "F1"
        # fp(F1.data)
        # print "C1"
        # fp(C1.data)

        CF1 = C1.add(F1)
        oldCF1[:4,:] += F1.data[:4, :]
        oldCF1[1:4,:] += C1.data
        oldCF1R = F1.R + C1.R
        assert (CF1.data == oldCF1).all()
        assert (CF1.R == oldCF1R+oldCF1R).all()
        assert (CF1.R == oldCF1R*2).all()


    def test_addoperator_inplace(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2, force_bandwidth=(-2,2))
        C2 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=2, order=2, force_bandwidth=(-2,2))
        oldCB2 = np.zeros((len(set(B2.offsets) | set(C2.offsets)), C2.shape[1]))
        oldCB2[1:,:] += B2.data[1:, :]
        oldCB2[1:4,:] += C2.data[1:4, :]
        oldCB2R = np.zeros_like(B2.R)
        oldCB2R = B2.R + C2.R
        B2.add(C2, inplace=True)
        assert (oldCB2 == B2.data).all()
        assert (oldCB2R == B2.R).all()

        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2)
        C2 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=2, order=2)
        try:
            B2.add(C2, inplace=True)
        except ValueError:
            pass
        else:
            raise AssertionError("In place addition should fail for different sized operators.")

    def test_mul(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2, force_bandwidth=(-2,2))
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
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2, force_bandwidth=(-2,2))
        assert B2 == B2
        C2 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=2, order=2, force_bandwidth=(-2,2))
        assert C2 != B2


    def test_addscalar(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2, force_bandwidth=(-2,2))
        origB2 = B2.copy()
        oldB2 = B2.copy()

        newB2 = B2.add(1.0)
        oldB2.data[tuple(oldB2.offsets).index(0)] += 1.0  # Manually add 1 to main diag

        assert newB2 is not B2 # Created a new operator
        assert newB2.data is not B2.data # With new underlying data
        assert (newB2.data == oldB2.data).all() # Operations were the same
        assert (newB2.data != origB2.data).any() # Operations changed our operator


    def test_addscalar_inplace(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2, force_bandwidth=(-2,2))
        origB2 = B2.copy()
        oldB2 = B2.copy()

        B2.add(1.0, inplace=True) # Add to main diag in place
        oldB2.data[tuple(oldB2.offsets).index(0)] += 1.0  # Manually add 1 to main diag in place

        assert (B2.data == oldB2.data).all() # Operations were the same
        assert (B2.data is not origB2.data)
        assert (B2.data != origB2.data).any() # Operations changed our operator


    def test_copy(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        C1 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=1, order=2, force_bandwidth=(-2,2))
        CC1 = C1.copy()
        CCC1 = CC1.copy()

        assert C1 is not CC1
        assert np.all(C1.data == CC1.data)
        assert np.all(C1.offsets == CC1.offsets)

        assert C1 is not CCC1
        assert np.all(C1.data == CCC1.data)
        assert np.all(C1.offsets == CCC1.offsets)

        assert CC1 is not CCC1
        assert np.all(CC1.data == CCC1.data)
        assert np.all(CC1.offsets == CCC1.offsets)


    def test_create(self):
        vec = self.vec
        last = len(vec)-1
        idx = 1
        d = np.hstack((np.nan, np.diff(vec)))
        deltas = d
        sch0 = 'center'
        for sch1 in ['center', 'up', 'down']:
            for dv in [1,2]:
                oldX1 = utils.nonuniform_complete_coefficients(deltas, up_or_down=sch1, flip_idx=idx)[dv-1]
                X1 = FD.BandedOperator.for_vector(vec, scheme=sch1, derivative=dv, order=2)

                high, low = 1,-1
                if (sch0 == 'up' and idx > 1) or (sch1 == 'up' and idx < last-1):
                    high = 2
                if (sch0 == 'down' and idx > 2) or (sch1 == 'down' and idx < last):
                    low = -2
                m = tuple(oldX1.offsets).index(0)
                oldX1.data = oldX1.data[m-high:m-low+1]
                oldX1.offsets = oldX1.offsets[m-high:m-low+1]

                # print "old todense()"
                # fp(oldX1.todense())
                # print "new todense()"
                # fp(X1.todense())
                # print
                # print X1.shape, oldX1.shape
                # print (X1.offsets, oldX1.offsets),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert (X1.todense() == oldX1.todense()).all(),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert (X1.offsets == oldX1.offsets).all(),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert (X1.data.shape == oldX1.data.shape),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert (X1.data == oldX1.data).all(),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)


    def test_splice_same(self):
        vec = self.vec
        last = len(vec)-1
        deltas = np.hstack((np.nan, np.diff(vec)))

        # When bandwidth is the same
        # print "Splicing operators of the same width."
        for sch0,sch1 in itertools.product(['center', 'up', 'down'], repeat=2):
            for dv in [1,2]:
                for idx in range(0, len(vec)-1):
                    X1 = FD.BandedOperator.for_vector(vec, scheme=sch0, derivative=dv, order=2, force_bandwidth=(-2,2))+1
                    X2 = FD.BandedOperator.for_vector(vec, scheme=sch1, derivative=dv, order=2, force_bandwidth=(-2,2))+1
                    X12 = X1.splice_with(X2, idx)
                    manualX12 = np.vstack((X1.todense()[:idx, :], X2.todense()[idx:,:]))
                    manualX12 = scipy.sparse.dia_matrix(manualX12)
                    X12i = X1.splice_with(X2, idx, inplace=True)
                    assert X12i is X1

                    high, low = 1,-1
                    if (sch0 == 'up' and idx > 1) or (sch1 == 'up' and idx < last-1):
                        high = 2
                    if (sch0 == 'down' and idx > 2) or (sch1 == 'down' and idx < last):
                        low = -2
                    m = tuple(X12.offsets).index(0)
                    X12.data = X12.data[m-high:m-low+1]
                    X12.offsets = X12.offsets[m-high:m-low+1]

                    # print
                    # print "manual"
                    # fp(manualX12.data[::-1], 3)
                    # print
                    # print "new"
                    # # fp(X12.todense(), 3)
                    # # print
                    # fp(X12.data, 3)

                    # print
                    # print X12.shape, manualX12.shape
                    # print (X12.offsets, manualX12.offsets[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.todense() == manualX12.todense()).all(),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.offsets == manualX12.offsets[::-1]).all(),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.data.shape == manualX12.data.shape),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.data == manualX12.data[::-1]).all(),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)


    def test_splice_different(self):
        vec = self.vec
        last = len(vec)-1
        deltas = np.hstack((np.nan, np.diff(vec)))

        # When bandwidth is possibly different
        # print "Splicing operators of the different width."
        for sch0,sch1 in itertools.product(['center', 'up', 'down'], repeat=2):
            for dv in [1,2]:
                for idx in range(0, len(vec)-1):
                    # add identity to avoid empty center
                    X1 = FD.BandedOperator.for_vector(vec, scheme=sch0, derivative=dv, order=2)+1
                    X2 = FD.BandedOperator.for_vector(vec, scheme=sch1, derivative=dv, order=2)+1
                    X12 = X1.splice_with(X2, idx)
                    manualX12 = np.vstack((X1.todense()[:idx, :], X2.todense()[idx:,:]))
                    manualX12 = scipy.sparse.dia_matrix(manualX12)

                    high, low = 1,-1
                    if (sch0 == 'up' and idx > 1) or (sch1 == 'up' and idx < last-1):
                        high = 2
                    if (sch0 == 'down' and idx > 2) or (sch1 == 'down' and idx < last):
                        low = -2
                    m = tuple(X12.offsets).index(0)
                    X12.data = X12.data[m-high:m-low+1]
                    X12.offsets = X12.offsets[m-high:m-low+1]

                    # print
                    # print "manual"
                    # fp(manualX12.data[::-1], 3)
                    # print
                    # print "new"
                    # # fp(X12.todense(), 3)
                    # # print
                    # fp(X12.data, 3)

                    # print
                    # print X12.shape, manualX12.shape
                    # print (X12.offsets, manualX12.offsets[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.todense() == manualX12.todense()).all(),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.offsets == manualX12.offsets[::-1]).all(),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.data.shape == manualX12.data.shape),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.data == manualX12.data[::-1]).all(),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)


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
        def f0(x): return x*0
        def fx(x): return x+2

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
        newB = oldB.copy()
        vecB = oldB.copy()
        manualB.data[0][2:] *= coeff(len(vec)-2)
        manualB.data[1][1:] *= coeff(len(vec)-1)
        manualB.data[2] *= coeff(len(vec))
        manualB.data[3][:-1] *= coeff(1, len(vec))
        manualB.data[4][:-2] *= coeff(2, len(vec))
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
        data = np.ones((5,len(vec)), dtype=int)*flag
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

        assert (no_nan(newB.data) == 0).all()
        assert (no_nan(vecB.data) == 0).all()

        for dchlet in itertools.product([True, False], repeat=2):
            oldB = FD.BandedOperator((data.copy(), offsets), res.copy())
            oldB.is_dirichlet = dchlet
            veczeroB = oldB.copy()
            veczeroB.vectorized_scale(self.f0(vec))

            manualzeroB = np.zeros_like(veczeroB.data)
            if veczeroB.is_dirichlet[0]:
                manualzeroB[0, 2] = flag
                manualzeroB[1, 1] = flag
                manualzeroB[2, 0] = flag
            if veczeroB.is_dirichlet[1]:
                manualzeroB[2, -1] = flag
                manualzeroB[3, -2] = flag
                manualzeroB[4, -3] = flag

            # print dchlet
            # print
            # print "veczeroB"
            # fp(veczeroB.data)
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
            manualB.data[0][bottom+2:]  *= vec[bottom : last-2]+2
            manualB.data[1][bottom+1:]  *= vec[bottom : last-1]+2
            manualB.data[2][bottom:top] *= vec[bottom : top]+2
            manualB.data[3][:top-1]     *= vec[1      : top]+2
            manualB.data[4][:top-2]     *= vec[2      : top]+2
            manualB.R[bottom:top]       *= vec[bottom : top]+2
            vecB.vectorized_scale(self.fx(vec))
            newB.scale(lambda i: vec[i]+2)
            # print "vec"
            # fp(vec)
            # print
            # print "manual"
            # fp(manualB.data)
            # print
            # print "newB"
            # fp(newB.data)
            # print
            # print "vecB"
            # fp(vecB.data)
            # print
            # print "manualR"
            # print manualB.R
            # print
            # print "vecR"
            # print vecB.R
            assert (veczeroB.data == manualzeroB).all()
            assert newB == vecB
            assert manualB == newB
            assert manualB == vecB

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
        no_nan = np.nan_to_num
        vec = self.vec
        flag = int(1)
        data = np.ones((5,len(vec)), dtype=int)*flag
        data[0][:2] = 0
        data[1][0] = 0
        data[3][-1] = 0
        data[4][-2:] = 0
        offsets = [2,1,0,-1,-2]
        res = np.ones_like(vec)

        for dchlet in itertools.product([True, False], repeat=2):
            oldB = FD.BandedOperator((data.copy(), offsets), res.copy())
            oldB.is_dirichlet = dchlet
            zeroB = oldB.copy()
            zeroB.scale(self.f0)

            manualzeroB = np.zeros_like(zeroB.data)
            if zeroB.is_dirichlet[0]:
                manualzeroB[0, 2] = flag
                manualzeroB[1, 1] = flag
                manualzeroB[2, 0] = flag
            if zeroB.is_dirichlet[1]:
                manualzeroB[2, -1] = flag
                manualzeroB[3, -2] = flag
                manualzeroB[4, -3] = flag

            # print dchlet
            # print
            # print "zeroB"
            # fp(zeroB.data)
            # print
            # print "manualzeroB"
            # fp(manualzeroB)
            # print

            manualB = oldB.copy()
            newB = oldB.copy()
            bottom = 0
            top = manualB.shape[0]
            if dchlet[0]:
                bottom += 1
            if dchlet[1]:
                top -= 1
            manualB.data[0][bottom+2:] *= np.arange(bottom, manualB.shape[0]-2)+2
            manualB.data[1][bottom+1:] *= np.arange(bottom, manualB.shape[0]-1)+2
            manualB.data[2][bottom:top] *= np.arange(bottom, top)+2
            manualB.data[3][:top-1] *= np.arange(1, top)+2
            manualB.data[4][:top-2] *= np.arange(2, top)+2
            manualB.R[bottom:top] *= np.arange(bottom, top)+2
            newB.scale(self.fx)
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
            assert (zeroB.data == manualzeroB).all()
            assert manualB == newB


    def test_scale_no_dirichlet(self):
        no_nan = np.nan_to_num
        vec = self.vec
        def f0(x): return 0
        def fx(x): return x
        data = np.ones((5,len(vec)))
        data[0][:2] = 0
        data[1][0] = 0
        data[3][-1] = 0
        data[4][-2:] = 0
        offsets = [2,1,0,-1,-2]
        res = np.ones_like(vec)
        oldB = FD.BandedOperator((data, offsets), res)

        newB = oldB.copy()
        newB.scale(f0)
        assert (no_nan(newB.data) == 0).all()

        manualB = oldB.copy()
        newB = oldB.copy()
        manualB.data[0][2:] *= np.arange(len(vec)-2)
        manualB.data[1][1:] *= np.arange(len(vec)-1)
        manualB.data[2] *= np.arange(len(vec))
        manualB.data[3][:-1] *= np.arange(1, len(vec))
        manualB.data[4][:-2] *= np.arange(2, len(vec))
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
        assert manualB == newB


def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
