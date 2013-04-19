#!/usr/bin/env python
# coding: utf8

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

from FiniteDifference import BandedOperatorGPU as BOG

from FiniteDifference.SizedArrayPtr import SizedArrayPtr


class FiniteDifferenceEngineADI_from_Host_test(unittest.TestCase):

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
        coeffs = {k: lambda *x: 1 for k in coeffs}
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
        # self.F.operators[1].diagonalize()
        self.FG = FDG.FiniteDifferenceEngineADI()
        self.FG.from_host_FiniteDifferenceEngine(self.F)


    def test_csr_solve(self):
        raise unittest.SkipTest
        A = self.F.operators[1].copy() + 1
        Tri = BOG.BandedOperator(A)
        A.diagonalize()
        Tri.diagonalize()

        # ref = A.apply(self.F.grid.domain[-1])
        # tst = Tri.apply(self.FG.grid.domain[-1])

        # fp(tst - ref, 'e')
        # npt.assert_array_almost_equal(ref, tst, decimal=11)

        A.is_mixed_derivative = True

        Csr = BOG.BandedOperator(A)

        # ref = Tri.immigrate()
        # tst = Csr.immigrate()
        # fp(tst.D.data)
        # fp(tst.D.indices)
        # fp(tst.D.indptr)
        # tst.is_mixed_derivative = False
        # npt.assert_equal(tst, ref)

        # domT = self.FG.gpugrid.copy(True)
        # domC = self.FG.gpugrid.copy(True)
        # domT = SizedArrayPtr(self.F.grid.domain[-1])
        # domC = SizedArrayPtr(self.F.grid.domain[-1])

        # Csr.solve_(domC, True)
        # tst = Csr.immigrate().solve(domC.to_numpy())
        # Tri.solve_(domT, True)

        tst = domC.to_numpy()
        ref = domT.to_numpy()

        print "Ref"
        fp(ref)
        print "Test"
        fp(tst)
        print "Diff"
        fp(tst - ref)

        npt.assert_array_almost_equal(ref, tst, decimal=11)
        assert False


    def test_verify_simple_operators_0(self):
        ref = self.F.simple_operators[(0,)].copy()
        tst = self.FG.simple_operators[(0,)].copy().immigrate()
        npt.assert_equal(tst, ref)
        ref = self.F.simple_operators[(0,)].copy()
        tst = self.FG.simple_operators[(0,)].copy()
        ref.diagonalize(), tst.diagonalize()
        tst = tst.immigrate()
        npt.assert_equal(tst, ref)


    def test_verify_simple_operators_1(self):
        ref = self.F.simple_operators[(1,)].copy()
        tst = self.FG.simple_operators[(1,)].copy().immigrate()
        npt.assert_equal(tst, ref)
        ref = self.F.simple_operators[(1,)].copy()
        tst = self.FG.simple_operators[(1,)].copy()
        ref.diagonalize(), tst.diagonalize()
        tst = tst.immigrate()
        npt.assert_equal(tst, ref)


    def test_verify_simple_operators_00(self):
        ref = self.F.simple_operators[(0,0)].copy()
        tst = self.FG.simple_operators[(0,0)].copy().immigrate()
        npt.assert_equal(tst, ref)
        ref = self.F.simple_operators[(0,0)].copy()
        tst = self.FG.simple_operators[(0,0)].copy()
        ref.diagonalize(), tst.diagonalize()
        tst = tst.immigrate()
        npt.assert_equal(tst, ref)


    def test_verify_simple_operators_11(self):
        ref = self.F.simple_operators[(1,1)].copy()
        tst = self.FG.simple_operators[(1,1)].copy()
        tst = tst.immigrate()
        npt.assert_equal(tst, ref)
        ref = self.F.simple_operators[(1,1)].copy()
        tst = self.FG.simple_operators[(1,1)].copy()
        ref.diagonalize(), tst.diagonalize()
        tst = tst.immigrate()
        npt.assert_equal(tst, ref)


    def test_verify_simple_operators_01(self):
        ref = self.F.simple_operators[(0,1)]
        tst = self.FG.simple_operators[(0,1)].immigrate()
        npt.assert_equal(tst, ref)


    def test_combine_operators_0(self):
        ref = self.F.operators[0]
        fst = self.FG.simple_operators[(0,)]
        snd = self.FG.simple_operators[(0,0)]
        tst = (snd + fst) + 0.5
        tst = tst.immigrate()
        tst.derivative = ref.derivative
        tstD = tst.D.data
        refD = ref.D.data
        npt.assert_array_almost_equal(tstD, refD)

        tst.D.data *= 0
        ref.D.data *= 0
        npt.assert_equal(tst, ref)


    def test_combine_operators_1(self):
        ref = self.F.operators[1]
        npt.assert_array_equal([1, 0, -1, -2], ref.D.offsets)
        npt.assert_equal(ref.bottom_fold_status, "CAN_FOLD")
        fst = self.FG.simple_operators[(1,)]
        snd = self.FG.simple_operators[(1,1)]
        tst = (fst + snd) + 0.5
        tst = tst.immigrate()
        npt.assert_array_equal([1, 0, -1, -2], tst.D.offsets)
        npt.assert_equal(tst.bottom_fold_status, "CAN_FOLD")
        tst.derivative = ref.derivative
        fp(tst.D.data - ref.D.data, 'e')

        tstD = tst.D.data
        refD = ref.D.data
        npt.assert_array_almost_equal(tstD, refD)

        tst.D.data *= 0
        ref.D.data *= 0
        npt.assert_equal(tst, ref)


    def test_cross_derivative(self):
        crossOp = self.F.operators[(0,1)]
        crossOpGPU = self.FG.simple_operators[(0,1)]
        g = self.F.grid.domain[-1]

        d2gdxdy = crossOp.apply(g)
        d2gdxdyGPU = crossOpGPU.apply(g)
        npt.assert_array_almost_equal(d2gdxdy, d2gdxdyGPU)

        scale = np.random.random()
        crossOp *= scale
        crossOpGPU.mul_scalar_from_host(scale, inplace=True)

        d2gdxdy_scaled = crossOp.apply(g)
        d2gdxdyGPU_scaled = crossOp.apply(g)
        npt.assert_array_almost_equal(d2gdxdy_scaled, d2gdxdyGPU_scaled)


    def test_GPUSolve_0(self):
       B = self.F.operators[0]
       blen = B.D.shape[0] / B.blocks
       B.D.data = np.arange(B.D.data.size, dtype=float).reshape(B.D.data.shape)
       B.R = np.zeros(B.D.data.shape[1])
       B.D.data[0,0::blen] = 0
       B.D.data[-1,blen-1::blen] = 0
       origdata = B.D.data.copy()
       ref = B.solve(self.F.grid.domain[-1])
       B = FDG.BOG.BandedOperator(B)
       tst = B.solve(self.F.grid.domain[-1])
       B = B.immigrate()
       # fp(ref - tst, 3, 'e')
       npt.assert_array_almost_equal(origdata, B.D.data)
       npt.assert_allclose(ref, tst, atol=1e-6)


    def test_GPUSolve_1(self):
        B = self.F.operators[1]
        B.diagonalize()
        blen = B.D.shape[0] / B.blocks

        B.D.data = np.random.random((B.D.data.shape))
        B.R = np.random.random(B.D.data.shape[1])
        B.D.data[0,0::blen] = 0
        B.D.data[-1,blen-1::blen] = 0
        ref = B.solve(self.F.grid.domain[-1])
        B.undiagonalize()
        origdata = B.D.data.copy()

        BG = FDG.BOG.BandedOperator(B)
        BG.diagonalize()
        tst = BG.solve(self.F.grid.domain[-1])
        BG.undiagonalize()
        BG = BG.immigrate()
        # fp(ref - tst, 2, 'e')
        npt.assert_array_equal(origdata, B.D.data)
        npt.assert_allclose(ref, tst)


    def test_firsts(self):
        r = self.F._dummy()[0]
        t = self.FG._dummy()[0]
        for ref, tstG in zip(r, t):
            tst = tstG.immigrate()
            if not tst.is_mixed_derivative:
                blen = ref.D.shape[0] // ref.blocks
                if tst.dirichlet[0] is not None:
                    tst.D.data[1, ::blen] = 1
                if tst.dirichlet[1] is not None:
                    tst.D.data[1, blen-1::blen] = 1
            npt.assert_array_equal(ref.D.data, tst.D.data)
            npt.assert_equal(ref, tst)


    def test_Les(self):
        r = self.F._dummy()[1]
        t = self.FG._dummy()[1]
        for ref, tstG in zip(r, t):
            tst = tstG.immigrate()
            if not tst.is_mixed_derivative:
                blen = ref.D.shape[0] // ref.blocks
                if tst.dirichlet[0] is not None:
                    tst.D.data[1, ::blen] = 1
                if tst.dirichlet[1] is not None:
                    tst.D.data[1, blen-1::blen] = 1
            npt.assert_array_equal(ref.D.data, tst.D.data)
            npt.assert_equal(ref, tst)


    def test_Lis(self):
        r = self.F._dummy()[2]
        t = self.FG._dummy()[2]
        for ref, tstG in zip(r, t):
            tst = tstG.immigrate()
            if not tst.is_mixed_derivative:
                blen = ref.D.shape[0] // ref.blocks
                if tst.dirichlet[0] is not None:
                    tst.D.data[1, ::blen] = 1
                if tst.dirichlet[1] is not None:
                    tst.D.data[1, blen-1::blen] = 1
            npt.assert_array_almost_equal(ref.D.data, tst.D.data, decimal=15)
            ref.D *= 0
            tst.D *= 0
            npt.assert_equal(ref, tst)


    def test_Orig(self):
        ref = self.F._dummy()[3]
        tst = self.FG._dummy()[3]
        # print
        # fp(ref - tst, 'e')
        npt.assert_array_equal(ref, tst)


    def test_Y(self):
        ref = self.F._dummy()[4]
        tst = self.FG._dummy()[4]
        # print
        # fp(ref - tst, 'e')
        npt.assert_array_almost_equal(ref, tst)


    def test_V(self):
        ref = self.F._dummy()[5]
        tst = self.FG._dummy()[5]
        # print
        # fp(ref - tst, 'e')
        npt.assert_array_almost_equal(ref, tst)



def main():
    """Run main."""
    import nose
    nose.main()
    return 0

if __name__ == '__main__':
    main()
