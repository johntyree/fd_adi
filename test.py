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

class something(unittest.TestCase):

    def __init__(self, other):
        unittest.TestCase.__init__(self, other)
        spots = np.arange(5.0)
        dss = np.hstack((np.nan, np.diff(spots)))
        vars = [0.5, 1]
        r = 2.0
        up_or_down_spot = ''
        flip_idx_spot = 0
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
            gamma2_s = 0.5 * v * spots ** 2 * 0

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
            L1_[j].data[m, :] -=  0.5 * r * 0

            R1_.append((Rs + Rss).copy())

        def flatten_tensor(mats):
            diags = np.hstack([x.data for x in mats])
            flatmat = scipy.sparse.dia_matrix((diags, mats[0].offsets), shape=(diags.shape[1], diags.shape[1]))
            return flatmat

        L1 = flatten_tensor(L1_)
        R1 = np.array(R1_).T

        self.As_ = As_
        self.Ass_ = Ass_
        self.L1_ = L1
        self.R1_ = R1
        self.spots = spots
        self.vars = vars
        self.r = r
        self.vec = np.arange(10)
        self.flip_idx = 4


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


    def test_operatoraddoperator(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        C1 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=1, order=2)
        F1 = FD.BandedOperator.for_vector(vec, scheme='forward', derivative=1, order=2)
        oldCF1 = np.zeros((len(set(F1.offsets) | set(C1.offsets)), C1.shape[1]))

        CF1 = C1.add(F1)
        oldCF1[:3,:] += F1.data[:3, :]
        oldCF1[1:4,:] += C1.data[1:4, :]
        assert (CF1.data == oldCF1).all()


    def test_operatoraddoperator_inplace(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2)
        C2 = FD.BandedOperator.for_vector(vec, scheme='center', derivative=2, order=2)
        oldCB2 = np.zeros((len(set(B2.offsets) | set(C2.offsets)), C2.shape[1]))
        oldCB2[1:,:] += B2.data[1:, :]
        oldCB2[1:4,:] += C2.data[1:4, :]
        B2.add(C2, inplace=True)
        assert (oldCB2 == B2.data).all()


    def test_operatormul(self):
        vec = self.vec
        idx = self.flip_idx
        d = np.hstack((np.nan, np.diff(vec)))
        B2 = FD.BandedOperator.for_vector(vec, scheme='backward', derivative=2, order=2)
        BB2 = B2
        assert (B2 is not B2.mul(1))
        assert (B2 == B2.mul(1))
        assert (B2.mul(6) == B2.mul(2).mul(3))

        assert (BB2 is B2.mul(2, inplace=True))
        assert (BB2 is B2.mul(2, inplace=True).mul(2, inplace=True))


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


    def test_operatorscale(self):
        G = Grid.Grid((self.spots, self.vars), initializer=lambda x0,x1: np.maximum(self.spots,0))
        def mu_s(t, *dim): return self.r * dim[0]
        def gamma2_s(t, *dim): return 0.5 * dim[1] * dim[0]**2 * 0
        coeffs = {(0,) : mu_s,
                (0,0): gamma2_s}
        F = FD.FiniteDifferenceEngine(G, coefficients=coeffs)
        newL1 = F.operators[(0,)]


        print "old"
        fp(self.L1_.data)
        print
        print "new"
        fp(newL1.data)
        assert (self.L1_.data == newL1.data).all()



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
        # fp(oldC2.todense())
        # print "new todense()"
        # fp(C2.todense())
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
        # print "old data"
        # fp(np.hstack((oldB2.offsets[:,np.newaxis], oldB2.data)))
        # print "new data"
        # fp(np.hstack((B2.offsets[:,np.newaxis], B2.data)))

        # print "old todense()"
        # fp(oldB2.todense())
        # print "new todense()"
        # fp(B2.todense())
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

    return 0

if __name__ == '__main__':
    main()
