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
import Grid
import unittest
from heston import bs_call_delta


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



class BlackScholes(unittest.TestCase):

    def setUp(self):
        spot_max = 2500.0
        spotdensity = 10.0  # infinity is linear?
        v = 0.04
        r = 0.06
        k = 99.0
        spot = 100.0
        # spots = np.linspace(0, np.log(spot)*2, nspots+1)
        t = 1.0
        dt = 1.0/300.0
        nspots = 2000
        spots = utils.sinh_space(spot, spot_max, spotdensity, nspots)
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

        bounds = {
                        # D: U = 0              VN: dU/dS = 1
                (0,)  : ((0, lambda *args: 0.0), (1, lambda t, x: 1.0)),
                # (0,)  : ((0, lambda *args: 0.0), (1, lambda t, *x: np.exp(x[0]))),
                        # D: U = 0              Free boundary
                (0,0) : ((0, lambda *args: 0.0), (None, lambda *x: None))}

        # self.spot_idx = np.argmin(np.abs(spots - np.log(spot)))
        # self.spot = np.exp(spots[self.spot_idx])
        self.spot_idx = np.argmin(np.abs(spots - spot))
        self.spot = spots[self.spot_idx]
        self.t, self.dt = t, dt
        self.F = FD.FiniteDifferenceEngineADI(G, coefficients=coeffs, boundaries=bounds)
        self.ans = bs_call_delta(self.spot, k, r, np.sqrt(v), t)[0]

    def test_implicit(self):
        t, dt = self.t, self.dt
        V = self.F.impl(t/dt, dt)[-1][self.spot_idx]
        print "Spot:", self.spot
        print "Price:", V, self.ans
        assert np.isclose(V, self.ans)
        assert np.isclose(V, self.ans, rtol=0.005, atol=0.005)

    def test_crank(self):
        t, dt = self.t, self.dt
        V = self.F.crank(t/dt, dt)[-1][self.spot_idx]
        print "Spot:", self.spot
        print "Price:", V, self.ans
        assert np.isclose(V, self.ans)
        assert np.isclose(V, self.ans, rtol=0.005, atol=0.005)

    def test_smooth(self):
        t, dt = self.t, self.dt
        V = self.F.smooth(t/dt, dt)[-1][self.spot_idx]
        print "Spot:", self.spot
        print "Price:", V, self.ans
        assert np.isclose(V, self.ans)
        assert np.isclose(V, self.ans, rtol=0.005, atol=0.005)




class something(unittest.TestCase):

    def __init__(self, other):
        unittest.TestCase.__init__(self, other)
        r = 2.0
        k = 3.0
        kappa = 1
        theta = 0.04
        sigma = 0.4
        rho = 0
        up_or_down_spot = ''
        up_or_down_var = ''
        flip_idx_spot = 2
        flip_idx_var = 2
        spot_max = 1500.0
        var_max = 13.0
        nspots = 5
        nvols = 3
        spotdensity = 7.0  # infinity is linear?
        varexp = 4
        spots = np.arange(5.0)
        vars = np.linspace(0, 1, 3)
        spots = utils.sinh_space(k, spot_max, spotdensity, nspots)
        # vars = utils.exponential_space(0.00, 0.04, var_max, varexp, nvols)
        dss = np.hstack((np.nan, np.diff(spots)))
        nvols = len(vars)
        dvs = np.hstack((np.nan, np.diff(vars)))
        nspots = len(spots)
        As_ = utils.nonuniform_complete_coefficients(dss, up_or_down=up_or_down_spot,
                                                    flip_idx=flip_idx_spot)[0]
        Ass_ = utils.nonuniform_complete_coefficients(dss)[1]
        L1_ = []
        R1_ = []
        # As_, Ass_ = utils.nonuniform_forward_coefficients(dss)
        assert(not np.isnan(As_.data).any())
        assert(not np.isnan(Ass_.data).any())
        for j, v in enumerate(vars):
            # Be careful not to overwrite our operators
            As, Ass = As_.copy(), Ass_.copy()
            m = 2

            mu_s = r * spots
            gamma2_s = 0.5 * v * spots ** 2

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
            L1_[j].data[m, :] -=  0.5 * r
            L1_[j].data[m, 0] = -1

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
            mu_v = kappa * (theta - vars)
            gamma2_v = 0.5 * sigma ** 2 * vars

            # Be careful not to overwrite our operators
            Av, Avv = Av_.copy(), Avv_.copy()

            m = 2

            Av.data[m - 2, 2] = -dvs[1] / (dvs[2] * (dvs[1] + dvs[2]))
            Av.data[m - 1, 1] = (dvs[1] + dvs[2]) / (dvs[1] * dvs[2])
            Av.data[m, 0] = (-2 * dvs[1] - dvs[2]) / (dvs[1] * (dvs[1] + dvs[2]))

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
            L2_[i].data[m, :] -= 0.5 * r
            L2_[i].data[m, -1] = -1  # This is to cancel out the previous value so we can
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
        self.spots = spots
        self.vars = vars
        self.nvols = nvols
        self.strike = k
        self.r = r
        self.sigma = sigma
        self.kappa = kappa
        self.theta = theta
        self.vec = spots
        self.flip_idx_spot = flip_idx_spot
        self.flip_idx_var = flip_idx_var
        self.flip_idx = 4


    def test_Grid_create(self):
        G = Grid.Grid((self.spots, self.vars), initializer=lambda x0,x1: np.maximum(x0-self.strike,0))
        U = np.tile(np.maximum(0, self.spots - self.strike), (self.nvols, 1)).T
        print G
        print U
        assert G.ndim == 2
        assert (G.mesh[0] == self.spots).all()
        assert (G.mesh[1] == self.vars).all()
        assert G.shape == tuple(map(len, G.mesh))
        assert (G.domain == U).all()


    def test_FDADI_combine_dimensional_operators(self):
        G = Grid.Grid((self.spots, self.vars), initializer=lambda x0,x1: np.maximum(x0-self.strike,0))
        # print G

        def mu_s(t, *dim):
            return self.r * dim[0]

        def gamma2_s(t, *dim):
            return 0.5 * dim[1] * dim[0]**2

        def mu_v(t, *dim):
            return self.kappa * (self.theta - dim[1])

        def gamma2_v(t, *dim):
            return 0.5 * self.sigma**2 * dim[1]
        k = self.strike
        coeffs = {()   : lambda t: -self.r,
                  (0,) : mu_s,
                  (0,0): gamma2_s,
                  (1,) : mu_v,
                  (1,1): gamma2_v}
        bounds = {
                        # D: U = 0              VN: dU/dS = 1
                (0,)  : ((0, lambda *args: 0.0), (1, lambda *args: 1.0)),
                        # D: U = 0              Free boundary
                (0,0) : ((0, lambda *args: 0.0), (None, lambda *x: None)),
                        # Free boundary at low variance
                (1,)  : ((None, lambda *x: None),
                        # D intrinsic value at high variance
                        (0.0, lambda t, *dim: np.maximum(0.0, dim[0]-k))),
                        # Free boundary
                (1,1) : ((None, lambda *x: None),
                        # D intrinsic value at high variance
                        (0.0, lambda t, *dim: np.maximum(0.0, dim[0]-k)))}
        F = FD.FiniteDifferenceEngineADI(G, coefficients=coeffs, boundaries=bounds)

        oldL1 = self.L1_
        oldL2 = self.L2_
        oldR1 = self.R1_.T.flatten()
        oldR2 = self.R2_.flatten()
        # manualL1 = FD.flattan_tensor(F.operators[(0,)]) + F.operators[(0,0)]
        # manualL2 = F.operators[(1,)] + F.operators[(1,1)]
        L1 = F.operators[0]
        L2 = F.operators[1]

        # print "old"
        # fp(oldL1.data)
        # print
        # print "new"
        # fp(L1.data)
        assert (L1.data == oldL1.data).all()
        # print "old"
        # print oldR1
        # print
        # print "new"
        # print L1.R
        assert (L1.R == oldR1).all()
        # print "old"
        # fp(oldL2.data)
        # print
        # print "new"
        # fp(L2.data)
        #TODO: Somehow this boundary condition is failing. As if it's not using the
        # right args or right dim or something?
        assert (L2.data == oldL2.data).all()
        print "old"
        print oldR2
        print
        print "new"
        print L2.R
        print "diff"
        fp(L2.R - oldR2)
        assert (L2.R == oldR2).all()


    def test_operatoraddself(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        C1 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=1, order=2)
        C2 = C1.add(C1)
        assert C2 is not C1
        assert C2.data is not C1.data
        assert (C2.offsets == C1.offsets).all()
        assert (C2.data == C1.data+C1.data).all()
        assert (C2.data == C1.data*2).all()
        assert (C2.R == C1.R*2).all()
        assert (C2.R == C1.R*+C1.R).all()


    def test_operatoraddoperator(self):
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
        oldCF1[1:4,:] += C1.data[1:4, :]
        oldCF1R = F1.R + C1.R
        assert (CF1.data == oldCF1).all()
        assert (CF1.R == oldCF1R+oldCF1R).all()
        assert (CF1.R == oldCF1R*2).all()


    def test_operatoraddoperator_inplace(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2)
        C2 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=2, order=2)
        oldCB2 = np.zeros((len(set(B2.offsets) | set(C2.offsets)), C2.shape[1]))
        oldCB2[1:,:] += B2.data[1:, :]
        oldCB2[1:4,:] += C2.data[1:4, :]
        oldCB2R = np.zeros_like(B2.R)
        oldCB2R = B2.R + C2.R
        B2.add(C2, inplace=True)
        assert (oldCB2 == B2.data).all()
        assert (oldCB2R == B2.R).all()


    def test_operatormul(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2)
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


    def test_operatoreq(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2)
        assert B2 == B2
        C2 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=2, order=2)
        assert C2 != B2


    def test_operatoraddscalar(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2)
        origB2 = B2.copy()
        oldB2 = B2.copy()

        newB2 = B2.add(1.0)
        oldB2.data[tuple(oldB2.offsets).index(0)] += 1.0  # Manually add 1 to main diag

        assert newB2 is not B2 # Created a new operator
        assert newB2.data is not B2.data # With new underlying data
        assert (newB2.data == oldB2.data).all() # Operations were the same
        assert (newB2.data != origB2.data).any() # Operations changed our operator


    def test_operatoraddscalar_inplace(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2)
        origB2 = B2.copy()
        oldB2 = B2.copy()

        B2.add(1.0, inplace=True) # Add to main diag in place
        oldB2.data[tuple(oldB2.offsets).index(0)] += 1.0  # Manually add 1 to main diag in place

        assert (B2.data == oldB2.data).all() # Operations were the same
        assert (B2.data is not origB2.data)
        assert (B2.data != origB2.data).any() # Operations changed our operator

    def test_operatorvectorizedscale(self):
        no_nan = np.nan_to_num
        vec = self.vec
        def coeff(high,low=None):
            if low is not None:
                high, low = low, high
            return np.linspace(0, 1, len(vec))[low:high]
        def fcoeff(i):
            return np.linspace(0, 1, len(vec))[i]
        def f0(x): return x*0
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
        vecB = oldB.copy()
        newB.scale(f0)
        vecB.vectorized_scale(f0(vec))
        assert (no_nan(newB.data) == 0).all()
        assert (no_nan(vecB.data) == 0).all()
        assert newB == vecB

        manualB = oldB.copy()
        newB = oldB.copy()
        vecB = oldB.copy()
        manualB.data[0][2:] *= coeff(len(vec)-2)
        manualB.data[1][1:] *= coeff(len(vec)-1)
        manualB.data[2] *= coeff(len(vec))
        manualB.data[3][:-1] *= coeff(1, len(vec))
        manualB.data[4][:-2] *= coeff(2, len(vec))
        manualB.R *= coeff(len(vec))
        newB.scale(fcoeff)
        vecB.vectorized_scale(coeff(len(vec)))
        # print "manual"
        # fp(manualB.data)
        # print
        # print "new"
        # fp(newB.data)
        # print
        # print "vec"
        # print fp(vecB.data)
        # print
        # print "manualR"
        # print manualB.R
        # print
        # print "newR"
        # print newB.R
        assert manualB == newB
        assert manualB == vecB

    def test_operatorscale(self):
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
        print "manual"
        fp(manualB.data)
        print
        print "new"
        fp(newB.data)
        print
        print "manualR"
        print manualB.R
        print
        print "newR"
        print newB.R
        assert manualB == newB


    def test_operatorcopy(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        C1 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=1, order=2)
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


    def test_operatorcreate(self):
        vec = self.vec
        idx = 1
        d = np.hstack((np.nan, np.diff(vec)))

        oldC1 = utils.nonuniform_complete_coefficients(d, up_or_down='', flip_idx=idx)[0]
        C1 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=1, order=2)
        # print "old data"
        # fp(np.hstack((oldC1.offsets[:,np.newaxis], oldC1.data)))
        # print "new data"
        # fp(np.hstack((C1.offsets[:,np.newaxis], C1.data)))

        # print "old todense()"
        # fp(oldC1.todense())
        # print "new todense()"
        # fp(C1.todense())
        assert np.all(C1.data == oldC1.data)
        assert np.all(C1.todense() == oldC1.todense())

        oldF1 = utils.nonuniform_complete_coefficients(d, up_or_down='up', flip_idx=idx)[0]
        F1 = FD.BandedOperator.for_vector(vec, scheme='forward', derivative=1, order=2)
        # print "old data"
        # fp(np.hstack((oldF1.offsets[:,np.newaxis], oldF1.data)))
        # print "new data"
        # fp(np.hstack((F1.offsets[:,np.newaxis], F1.data)))

        # print "old todense()"
        # fp(oldF1.todense())
        # print "new todense()"
        # fp(F1.todense())
        assert np.all(F1.todense() == oldF1.todense())
        assert np.all(F1.data == oldF1.data)

        oldB1 = utils.nonuniform_complete_coefficients(d, up_or_down='down', flip_idx=idx)[0]
        B1 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=1, order=2)
        # print "old data"
        # fp(np.hstack((oldB1.offsets[:,np.newaxis], oldB1.data)))
        # print "new data"
        # fp(np.hstack((B1.offsets[:,np.newaxis], B1.data)))

        # print "old todense()"
        # fp(oldB1.todense())
        # print "new todense()"
        # fp(B1.todense())
        assert np.all(B1.todense() == oldB1.todense())
        assert np.all(B1.data == oldB1.data)

        oldC2 = utils.nonuniform_complete_coefficients(d, up_or_down='', flip_idx=idx)[1]
        C2 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=2, order=2)
        # print "old data"
        # fp(np.hstack((oldC2.offsets[:,np.newaxis], oldC2.data)))
        # print "new data"
        # fp(np.hstack((C2.offsets[:,np.newaxis], C2.data)))

        # print "old todense()"
        # fp(oldC2.todense(), 4)
        # print "new todense()"
        # fp(C2.todense(), 4)
        assert np.all(C2.todense() == oldC2.todense())
        assert np.all(C2.data == oldC2.data)

        oldF2 = utils.nonuniform_complete_coefficients(d, up_or_down='up', flip_idx=idx)[1]
        F2 = FD.BandedOperator.for_vector(vec, scheme='forward', derivative=2, order=2)
        # print "old data"
        # fp(np.hstack((oldF2.offsets[:,np.newaxis], oldF2.data)))
        # print "new data"
        # fp(np.hstack((F2.offsets[:,np.newaxis], F2.data)))

        # print "old todense()"
        # fp(oldF2.todense())
        # print "new todense()"
        # fp(F2.todense())
        assert np.all(F2.todense() == oldF2.todense())
        assert np.all(F2.data == oldF2.data)

        oldB2 = utils.nonuniform_complete_coefficients(d, up_or_down='down', flip_idx=idx)[1]
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2)
        print "old data"
        fp(np.hstack((oldB2.offsets[:,np.newaxis], oldB2.data)), 3)
        print "new data"
        fp(np.hstack((B2.offsets[:,np.newaxis], B2.data)), 3)

        print "old todense()"
        fp(oldB2.todense())
        print "new todense()"
        fp(B2.todense())
        assert np.all(B2.todense() == oldB2.todense())
        assert np.all(B2.data == oldB2.data)


    def test_operatorsplice(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))

        for idx in range(1, len(vec)-1):
            oldCC1 = utils.nonuniform_complete_coefficients(d, up_or_down='', flip_idx=idx)[0]
            C1 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=1, order=2)
            CC1 = C1.splice_with(C1, idx, overwrite=False)
            # print "old"
            # fp(oldCC1.todense())
            # print "new"
            # fp(CC1.todense())
            assert np.allclose(CC1.todense(), oldCC1.todense())

        for idx in range(1, len(vec)-1):
            oldCF1 = utils.nonuniform_complete_coefficients(d, up_or_down='up', flip_idx=idx)[0]
            C1 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=1, order=2)
            F1 = FD.BandedOperator.for_vector(vec, scheme='forward', derivative=1, order=2)
            CF1 = C1.splice_with(F1, idx, overwrite=False)
            # print "old"
            # fp(oldCF1.todense())
            # print "new"
            # fp(CF1.todense())
            assert np.allclose(CF1.todense(), oldCF1.todense())

        for idx in range(1, len(vec)-1):
            oldCB1 = utils.nonuniform_complete_coefficients(d, up_or_down='down', flip_idx=idx)[0]
            C1 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=1, order=2)
            B1 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=1, order=2)
            CB1 = C1.splice_with(B1, idx, overwrite=False)
            # print "old"
            # fp(oldCB1.todense())
            # print "new"
            # fp(CB1.todense())
            assert np.allclose(CB1.todense(), oldCB1.todense())

        for idx in range(1, len(vec)-1):
            oldCC2 = utils.nonuniform_complete_coefficients(d, up_or_down='', flip_idx=idx)[1]
            C2 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=2, order=2)
            CC2 = C2.splice_with(C2, idx, overwrite=False)
            # print "old"
            # fp(oldCC2.todense())
            # print "new"
            # fp(CC2.todense())
            assert np.allclose(CC2.todense(), oldCC2.todense())

        for idx in range(1, len(vec)-1):
            oldCF2 = utils.nonuniform_complete_coefficients(d, up_or_down='up', flip_idx=idx)[1]
            C2 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=2, order=2)
            F2 = FD.BandedOperator.for_vector(vec, scheme='forward', derivative=2, order=2)
            CF2 = C2.splice_with(F2, idx, overwrite=False)
            # print "old"
            # fp(oldCF2.todense())
            # print "new"
            # fp(CF2.todense())
            assert np.allclose(CF2.todense(), oldCF2.todense())

        for idx in range(1, len(vec)-1):
            oldCB2 = utils.nonuniform_complete_coefficients(d, up_or_down='down', flip_idx=idx)[1]
            C2 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=2, order=2)
            B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2)
            CB2 = C2.splice_with(B2, idx, overwrite=False)
            # print "old"
            # fp(oldCB2.todense())
            # print "new"
            # fp(CB2.todense())
            assert np.allclose(CB2.todense(), oldCB2.todense())


        # print "oldCF"
        # fp(oldCF.todense())
        # print
        # print "newCF"
        # fp(CF.todense())


def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
