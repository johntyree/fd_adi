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


class Operations_test(unittest.TestCase):

    def setUp(self):
        # print "Setting up Params for CPP tests"
        shape = (5,4)
        self.v1 = np.arange(shape[0]*shape[1], dtype=float)**2
        self.v2 = self.v1.copy()
        self.v2.resize(shape)

        coeffs = {(0,) : lambda t, *x: x[0]*x[1],
                  (0,0): lambda t, *x: x[0]*x[1],
                  (1,) : lambda t, *x: x[0]*x[1],
                  (1,1): lambda t, *x: x[0]*x[1],
                  (0,1): lambda t, *x: x[0]*x[1],
                  }
        bounds = {
                (0,)  : ((0,    lambda *args: 0),    (1,    lambda *args: 1)),
                (0,0) : ((0,    lambda *args: 0),    (None, lambda *args: 1)),
                (1,)  : ((None, lambda *args: None), (None, lambda *args: None)),
                (1,1) : ((1,    lambda *args: 0.0),  (None, lambda *args: None)),
                }

        schemes = {}

        self.G = Grid.Grid([np.arange(shape[0], dtype=float)**2, np.arange(shape[1], dtype=float)**2], lambda x, y: (x*shape[1]+y)**2)
        self.F = FD.FiniteDifferenceEngineADI(self.G, coefficients=coeffs,
                boundaries=bounds, schemes=schemes, force_bandwidth=None)
        # print "Setting up FDE for CPP tests"
        self.F.init()
        self.F.operators[0].R = np.arange(self.G.size, dtype=float)
        self.F.operators[1].R = np.arange(self.G.size, dtype=float)
        # print "Setup complete for CPP test"


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
        npt.assert_array_almost_equal(B.D.data, ref.D.data, decimal=14)
        ref.D *= 0
        B.D *= 0
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
        npt.assert_array_almost_equal(B.D.data, ref.D.data, decimal=14)
        ref.D *= 0
        B.D *= 0
        assert ref == B


    def test_migrate_01(self):
        B = self.F.operators[(0,1)]
        B.D = utils.todia(B.D.tocoo().todia())
        ref = B.copy()
        B = BOG.BandedOperator(B, "test 01")
        B = B.immigrate("test 01")
        npt.assert_array_equal(ref.D.todense(), B.D.todense())
        assert ref == B


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


    def test_derivative_solve_0(self):
        x = np.linspace(0,1,500)
        B = BO.for_vector(x)
        B.D.data[1,:] += 1
        B.D.data[1,0] = B.D.data[1,-1] = 2
        BG = BOG.BandedOperator(B)
        ref = np.e**x
        tst = BG.apply(ref) / 2
        fp(ref - tst, 'e')
        npt.assert_allclose(tst, ref, rtol=1e-4, atol=1e-6,
                            err_msg="d/dx (apply) not accurate")
        # fp(B.D.data)
        tst = BG.solve(ref) * 2
        npt.assert_allclose(ref, tst, rtol=1e-4, atol=1e-6,
                            err_msg="integral (solve) not accurate")


    def test_axis_0_derivative_1(self):
        B = self.F.simple_operators[(0,)]
        BG = BOG.for_vector(self.F.grid.mesh[0],
                            self.F.grid.shape[1], 1, 0).immigrate()

        # We don't care about the deltas, just duplicate here to pass the
        # equality check
        BG.deltas = B.deltas

        npt.assert_array_equal(B.D.data, BG.D.data)
        npt.assert_equal(B, BG)


    def test_axis_0_derivative_2(self):
        B = self.F.simple_operators[(0,0)]
        BG = BOG.for_vector(self.F.grid.mesh[0],
                            self.F.grid.shape[1], 2, 0).immigrate()

        # We don't care about the deltas, just duplicate here to pass the
        # equality check
        BG.deltas = B.deltas

        npt.assert_array_equal(B.D.data, BG.D.data)
        npt.assert_equal(B, BG)


    def test_axis_1_derivative_1(self):
        B = self.F.simple_operators[(1,)]
        BG = BOG.for_vector(self.F.grid.mesh[1],
                            self.F.grid.shape[0], 1, 1).immigrate()

        # We don't care about the deltas, just duplicate here to pass the
        # equality check
        BG.deltas = B.deltas

        npt.assert_array_equal(B.D.data, BG.D.data)
        npt.assert_equal(B, BG)


    def test_axis_1_derivative_2(self):
        B = self.F.simple_operators[(1,1)]
        BG = BOG.for_vector(self.F.grid.mesh[1],
                            self.F.grid.shape[0], 2, 1).immigrate()

        # We don't care about the deltas, just duplicate here to pass the
        # equality check
        BG.deltas = B.deltas

        fp(B.D.data)
        print
        fp(BG.D.data)
        print
        fp(B.D.data - BG.D.data)

        print B.D.offsets
        fp(B.D)
        print BG.D.offsets
        fp(BG.D)

        npt.assert_array_equal(B.D.data, BG.D.data)
        npt.assert_equal(B, BG)


    def test_mixed_derivative(self):
        B = self.F.simple_operators[(0,1)]
        # BG = BOG.mixed_for_vector(self.F.grid.mesh[0],
                                  # self.F.grid.mesh[1]).immigrate()
        B.D = scipy.sparse.coo_matrix(B.D)

        n0 = self.F.grid.mesh[0].size
        n1 = self.F.grid.mesh[1].size

        compute_deltas = lambda v: np.hstack((np.nan, np.diff(v)))

        d0 = compute_deltas(self.F.grid.mesh[0])
        d1 = compute_deltas(self.F.grid.mesh[1])

        dlen = n1 * (n0-2)
        nnz = 9 * (dlen - 2 * (n0-2))

        data = np.zeros(n1*(n0-2)*9)

        sup = np.zeros(n1)
        mid = sup.copy()
        sub = sup.copy()

        for i in xrange(1, len(d1)-1):
            sup[i] =            d1[i]  / (d1[i+1]*(d1[i]+d1[i+1]))
            mid[i] = (-d1[i] + d1[i+1]) /         (d1[i]*d1[i+1])
            sub[i] =         -d1[i+1]  / (d1[i  ]*(d1[i]+d1[i+1]))

        supsup = np.tile(sup, n0-2)
        supmid = np.tile(mid, n0-2)
        supsub = np.tile(sub, n0-2)

        midsup = supsup.copy()
        midmid = supmid.copy()
        midsub = supsub.copy()

        subsup = supsup.copy()
        submid = supmid.copy()
        subsub = supsub.copy()

        for i in xrange(1, len(d0)-1):
            supsup[(i-1)*(n0-1):(i)*(n0-1)] *= d0[i]  / (d0[i+1]*(d0[i]+d0[i+1]))
            supmid[(i-1)*(n0-1):(i)*(n0-1)] *= d0[i]  / (d0[i+1]*(d0[i]+d0[i+1]))
            supsub[(i-1)*(n0-1):(i)*(n0-1)] *= d0[i]  / (d0[i+1]*(d0[i]+d0[i+1]))

            midsup[(i-1)*(n0-1):(i)*(n0-1)] *= (-d0[i] + d0[i+1]) / (d0[i]*d0[i+1])
            midmid[(i-1)*(n0-1):(i)*(n0-1)] *= (-d0[i] + d0[i+1]) / (d0[i]*d0[i+1])
            midsub[(i-1)*(n0-1):(i)*(n0-1)] *= (-d0[i] + d0[i+1]) / (d0[i]*d0[i+1])

            subsup[(i-1)*(n0-1):(i)*(n0-1)] *= -d0[i+1]  / (d0[i]*(d0[i]+d0[i+1]))
            submid[(i-1)*(n0-1):(i)*(n0-1)] *= -d0[i+1]  / (d0[i]*(d0[i]+d0[i+1]))
            subsub[(i-1)*(n0-1):(i)*(n0-1)] *= -d0[i+1]  / (d0[i]*(d0[i]+d0[i+1]))

        tst = np.vstack([supsup,supmid,supsub,midsup,midmid,midsub,subsup,submid,subsub])
        fp(tst, 2)
        print
        B.D = utils.todia(scipy.sparse.dia_matrix(B.D))
        BOG.scipy_to_cublas(B)
        ref = B.D.data[:,n1:-n1]
        fp(ref, 2)
        print
        fp(tst-ref, 'e')

        print d0
        print d1
        print B.deltas


        # fp(B.D)
        # print B.D.nnz
        # print B.D.indices
        # print B.D.indptr
        # print B.D.data
        # fp(BG.D)

        # npt.assert_array_equal(B.D.data, BG.D.data)
        # npt.assert_equal(B, BG)
        assert False


def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
