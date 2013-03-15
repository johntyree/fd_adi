#!/usr/bin/env python
# coding: utf8

import itertools
import unittest

import FiniteDifference.utils as utils

import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.linalg as spl

from FiniteDifference.utils import todia, block_repeat, foldMatFor
from FiniteDifference.utils import todia
from FiniteDifference.visualize import fp
import FiniteDifference.Grid as Grid

import FiniteDifference.FiniteDifferenceEngine as FD
import FiniteDifference.BandedOperatorGPU as BOG
import FiniteDifference.BandedOperator as BO

from FiniteDifference.blackscholes import BlackScholesFiniteDifferenceEngine, BlackScholesOption
from FiniteDifference.heston import HestonBarrierOption


class Cpp_test(unittest.TestCase):

    def setUp(self):
        # print "Setting up Params for CPP tests"
        shape = (25,25)
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
                (0,)  : ((0,    lambda *args: 0),    (1,    lambda *args: 1)),
                (0,0) : ((0,    lambda *args: 0),    (None, lambda *args: 1)),
                (1,)  : ((None, lambda *args: None), (None, lambda *args: None)),
                (1,1) : ((1,    lambda *args: 0.0),  (None, lambda *args: None)),
                }

        schemes = {}

        self.G = Grid.Grid([np.arange(shape[0]), np.arange(shape[1])], lambda x, y: (x*shape[1]+y)**2)
        self.F = FD.FiniteDifferenceEngineADI(self.G, coefficients=coeffs,
                boundaries=bounds, schemes=schemes, force_bandwidth=None)
        # print "Setting up FDE for CPP tests"
        self.F.init()
        self.F.operators[0].R = np.arange(self.G.size, dtype=float)
        self.F.operators[1].R = np.arange(self.G.size, dtype=float)
        # print "Setup complete for CPP test"


    def test_SizedArray_roundtrip(self):
        npt.assert_array_equal(self.v1, BOG.test_SizedArray1_roundtrip(self.v1.copy()))


    def test_SizedArray_roundtrip2D(self):
        npt.assert_array_equal(self.v2, BOG.test_SizedArray2_roundtrip(self.v2.copy()))


    def test_migrate_0(self):
        B = self.F.operators[0]
        ref = B.copy()
        B = BOG.BandedOperator(B, "test 0")
        B = B.immigrate("test 0")
        assert ref == B


    def test_migrate_1(self):
        B = self.F.operators[1]
        npt.assert_array_equal([1, 0, -1, -2], B.D.offsets)
        npt.assert_equal(B.bottom_fold_status, "CAN_FOLD")
        ref = B.copy()
        B = BOG.BandedOperator(B, "test 1")
        B = B.immigrate("test 1")
        npt.assert_array_equal([1, 0, -1, -2], B.D.offsets)
        npt.assert_equal(B.bottom_fold_status, "CAN_FOLD")
        assert ref == B


    def test_migrate_1_diag(self):
        B = self.F.operators[1]
        npt.assert_array_equal([1, 0, -1, -2], B.D.offsets)
        npt.assert_equal(B.bottom_fold_status, "CAN_FOLD")
        ref = B.copy()
        ref.diagonalize()
        npt.assert_array_equal([1, 0, -1], ref.D.offsets)
        npt.assert_equal(ref.bottom_fold_status, "FOLDED")
        B = BOG.BandedOperator(B, "test 1")
        B.diagonalize()
        B = B.immigrate("test 1")
        npt.assert_array_equal([1, 0, -1], B.D.offsets)
        npt.assert_equal(B.bottom_fold_status, "FOLDED")
        # fp(ref.D)
        # fp(B.D)
        assert ref == B


    def test_migrate_01(self):
        B = self.F.operators[(0,1)]
        B.D = utils.todia(B.D.tocoo().todia())
        ref = B.copy()
        B = BOG.BandedOperator(B, "test 01")
        B = B.immigrate("test 01")
        npt.assert_array_equal(ref.D.todense(), B.D.todense())
        assert ref == B


    def test_SizedArray_transpose(self):
        ntests = 100
        for i in range(ntests):
            shape = tuple(np.random.random_integers(1, 100, 2))
            v2 = np.arange(shape[0]*shape[1], dtype=float).reshape(shape)
            npt.assert_array_equal(v2.T, BOG.test_SizedArray_transpose(v2.copy()))


    def test_tri_apply_axis_0(self):
        B = self.F.operators[0]
        # print "B0 data"
        # fp(B0.D.data)
        R = B.R.copy()
        ref = B.apply(self.v2)
        B = BOG.BandedOperator(B, "C1 0")
        tst = B.apply(self.v2.copy())
        B = B.immigrate("C1 0")
        npt.assert_array_equal(R, B.R)
        npt.assert_array_equal(ref, tst)


    def test_tri_apply_axis_1(self):
        B = self.F.operators[0]
        # print "B0 data"
        # fp(B0.D.data)
        R = B.R.copy()
        ref = B.apply(self.v2)
        B = BOG.BandedOperator(B)
        tst = B.apply(self.v2.copy())
        B = B.immigrate()
        npt.assert_array_equal(R, B.R)
        npt.assert_array_equal(ref, tst)


    def test_csr_apply_0(self):
        vec = np.arange(30, dtype=float)
        B = BO.for_vector(vec)
        ref = B.apply(vec)
        B.is_mixed_derivative = True
        B = BOG.BandedOperator(B)
        tst = B.apply(vec)
        npt.assert_array_equal(ref, tst)


    def test_csr_apply_01(self):
        B01  = self.F.operators[(0,1)]
        B01G = BOG.BandedOperator(B01)
        B01 *= 0.023934
        B01G *= 0.023934
        ref = B01.apply(self.v2)
        tst = B01G.apply(self.v2.copy())
        # fp(ref - tst, 'e')
        npt.assert_array_almost_equal(ref, tst, decimal=15)


    def test_csr_apply_random(self):
        B = self.F.operators[0] # Because we aren't transposing.
        B.R = None
        B.axis = 1
        B.dirichlet = (None, None)
        B.is_mixed_derivative = True
        for i in range(5):
            sz = np.random.randint(3, 20)
            B.D = scipy.sparse.csr_matrix(np.random.random((sz*sz,sz*sz)))
            BG = BOG.BandedOperator(B)
            v = np.random.random((sz, sz))
            ref = B.apply(v)
            tst = BG.apply(v)
            npt.assert_array_almost_equal(ref, tst, decimal=8)


    def test_csr_scale(self):
        B = self.F.operators[0]
        B.D = scipy.sparse.csr_matrix(np.ones((5,5)))
        B.R = None
        B.dirichlet = (None, None)
        B.is_mixed_derivative = True
        ref = np.arange(B.D.shape[0], dtype=float).repeat(B.D.shape[1])
        ref.resize(B.D.shape)
        B = BOG.BandedOperator(B)
        B.vectorized_scale(np.arange(B.operator_rows, dtype=float))
        B = B.immigrate()
        # fp(ref)
        # print
        # fp(B.D)
        # print
        # fp(B.D - ref)
        npt.assert_array_equal(ref, B.D.todense())


    def test_csr_scalar(self):
        scalar = 0.235
        B = self.F.operators[0]
        B.D = scipy.sparse.csr_matrix(np.ones((5,5)))
        B.R = None
        B.dirichlet = (None, None)
        B.is_mixed_derivative = True
        ref = np.ones(B.D.shape[0], dtype=float).repeat(B.D.shape[1]) * scalar
        ref.resize(B.D.shape)
        B = BOG.BandedOperator(B)
        B *= scalar
        B = B.immigrate()
        # fp(ref)
        # print
        # fp(B.D)
        # print
        # fp(B.D - ref)
        npt.assert_array_equal(ref, B.D.todense())


    def test_copy_tri(self):
        ref = self.F.operators[0]
        ref2 = ref.copy()
        Gtst1 = BOG.BandedOperator(ref)
        Gtst2 = BOG.BandedOperator(ref2)
        del ref2
        tst1 = Gtst1.immigrate()
        tst2 = Gtst2.immigrate()
        del Gtst2
        npt.assert_array_equal(tst1.D.todense(), tst2.D.todense())
        del tst2
        # fp(ref.D)
        # print
        # fp(tst.D)
        # print
        # fp(tst.D - ref.D)
        npt.assert_array_equal(ref.D.todense(), tst1.D.todense())
        npt.assert_equal(ref, tst1)


    def test_copy_csr(self):
        ref = self.F.operators[(0,1)]
        tst = BOG.BandedOperator(ref).copy().immigrate()
        ref.D = utils.todia(ref.D.tocoo().todia())
        # fp(ref.D)
        # print
        # fp(tst.D)
        # print
        # fp(tst.D - ref.D)
        npt.assert_array_equal(ref.D.todense(), tst.D.todense())
        npt.assert_equal(tst, ref)


    def test_GPUSolve_0(self):
        v = np.arange(10, dtype=float)
        B = BO.for_vector(v)
        B.D.data[1,:] += 1
        ref = B.solve(v**2)
        BG = BOG.BandedOperator(B)
        tst = BG.solve(v**2)
        # fp(B.D.data)
        npt.assert_allclose(tst, ref)

        B = self.F.operators[0]
        blen = B.D.shape[0] / B.blocks
        B.D.data = np.arange(B.D.data.size, dtype=float).reshape(B.D.data.shape)
        B.R = np.zeros(B.D.data.shape[1])
        B.D.data[0,0::blen] = 0
        B.D.data[-1,blen-1::blen] = 0
        origdata = B.D.data.copy()
        ref = B.solve(self.v2)
        B = BOG.BandedOperator(B)
        tst = B.solve(self.v2.copy())
        B = B.immigrate()
        # fp(ref - tst, 3, 'e')
        npt.assert_array_almost_equal(origdata, B.D.data)
        npt.assert_allclose(ref, tst)


    def test_GPUSolve_1(self):
        B = self.F.operators[1]
        B.diagonalize()
        blen = B.D.shape[0] / B.blocks

        B.D.data = np.random.random((B.D.data.shape))
        B.R = np.random.random(B.D.data.shape[1])
        B.D.data[0,0::blen] = 0
        B.D.data[-1,blen-1::blen] = 0
        ref = B.solve(self.v2)
        B.undiagonalize()
        origdata = B.D.data.copy()

        BG = BOG.BandedOperator(B)
        BG.diagonalize()
        tst = BG.solve(self.v2.copy())
        BG.undiagonalize()
        BG = BG.immigrate()
        # fp(ref - tst, 2, 'e')
        npt.assert_array_equal(origdata, B.D.data)
        npt.assert_allclose(ref, tst)



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


    def test_diagonalize(self):
        mat = self.B.D.data.view().reshape(-1)
        zeros = mat == 0
        mat[:] = np.arange(self.B.D.data.size)
        mat[zeros] = 0
        B = self.B.copy()
        # print "ref pre"
        # fp(self.B.D)
        # fp(self.B.D.data)

        # print "Collected off-tridiag points as bottom_factors"
        block_len = B.shape[0] / B.blocks
        bottom_factors = B.D.data[-1,block_len-3::block_len]
        # print B.blocks, len(bottom_factors)
        # print bottom_factors

        self.B.diagonalize()
        B.diagonalize()
        # print "ref mid"
        # fp(self.B.D)
        # fp(self.B.D.data)
        # print "tst mid"
        # fp(B.D)
        # fp(B.D.data)

        npt.assert_array_equal(self.B.D.data, B.D.data, err_msg="Diagonalize alone doesn't preserve operator matrix.")
        npt.assert_(B == self.B, msg="Diagonalize alone doesn't preserve operator.")

        B.undiagonalize()
        self.B.undiagonalize()
        npt.assert_(not B.is_tridiagonal())
        # print "ref after"
        # fp(self.B.D)
        # fp(self.B.D.data)
        # print "tst after"
        # fp(B.D)
        # fp(B.D.data)
        # print "ref top"
        # fp(self.B.top_factors or np.array([np.nan]))
        # print "tst top"
        # fp(B.top_factors or np.array([np.nan]))
        # print "ref bot"
        # fp(self.B.bottom_factors or np.array([np.nan]))
        # print "tst bot"
        # fp(B.bottom_factors or np.array([np.nan]))
        npt.assert_array_equal(self.B.D.data, B.D.data, err_msg="Undiagonalize roundtrip doesn't preserve operator matrix.")
        npt.assert_(B == self.B, msg="Undiagonalize roundtrip doesn't preserve operator.")


def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
