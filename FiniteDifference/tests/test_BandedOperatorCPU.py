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
import FiniteDifference.Grid

import FiniteDifference.FiniteDifferenceEngine as FD
import FiniteDifference.BandedOperator as BO

from FiniteDifference.blackscholes import BlackScholesFiniteDifferenceEngine, BlackScholesOption
from FiniteDifference.heston import HestonBarrierOption

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
                assert np.array_equal(X1.D.todense(), oldX1.todense()),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert np.array_equal(X1.D.offsets, oldX1.offsets),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert (X1.D.data.shape == oldX1.data.shape),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                assert np.array_equal(X1.D.data, oldX1.data),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)


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
                    assert np.array_equal(X12.D.todense(), manualX12.todense()),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert np.array_equal(X12.D.offsets, manualX12.offsets[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert X12.D.data.shape == manualX12.data.shape,  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert np.array_equal(X12.D.data, manualX12.data[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)


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
                    assert np.array_equal(X12.D.todense(), manualX12.todense()),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert np.array_equal(X12.D.offsets, manualX12.offsets[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert (X12.D.data.shape == manualX12.data.shape),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)
                    assert np.array_equal(X12.D.data, manualX12.data[::-1]),  "%s+%s (dv %i) idx %i" % (sch0, sch1, dv, idx)


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
            npt.assert_array_equal(vec, vec) # for NaNs
            npt.assert_array_equal(bvec, bvec) # for NaNs
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
            npt.assert_array_equal(vec, vec) # for NaNs
            npt.assert_array_equal(bvec, bvec) # for NaNs
        vec = self.vec.copy()
        bvec = vec.copy()

        B = Borig.copy()
        B.diagonalize()
        npt.assert_array_equal(B.D.data, blocktridiamat.data)
        for i in range(loops):
            bvec = blockxI.dot(blocktridiamat.dot(bvec))
            vec = B.apply(vec)
            npt.assert_array_equal(bvec, vec)
            npt.assert_array_equal(vec, vec)
            npt.assert_array_equal(bvec, bvec)
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


def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
