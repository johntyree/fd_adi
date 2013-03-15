#!/usr/bin/env python
# coding: utf8
"""<+Module Description.+>"""

from __future__ import division

import sys
import os
import itertools as it

import unittest
import numpy as np
import numpy.testing as npt

from FiniteDifference.SizedArrayPtr import SizedArrayPtr, SizedArrayPtr_i

class SizedArray_test(unittest.TestCase):

    def setUp(self):
        self.scalar = 23.214
        self.v = np.ones(8)
        self.v2 = self.v + 2
        self.S = SizedArrayPtr(self.v.copy())
        self.S2 = SizedArrayPtr(self.v2.copy())

    def test_deep_copy(self):
        v = self.v.copy()
        S = self.S.copy(True)
        npt.assert_array_equal(v, S.to_numpy())

    def test_shallow_copy(self):
        v = self.v.copy()
        S = self.S.copy(False)
        npt.assert_array_equal(v, S.to_numpy())
        del self.S
        npt.assert_raises(RuntimeError, S.to_numpy)

    def test_pluseq_scalar(self):
        self.S.pluseq_scalar(self.scalar)
        ref = self.v + self.scalar
        tst = self.S.to_numpy()
        npt.assert_array_equal(ref, tst)

    def test_minuseq_scalar(self):
        self.S.minuseq_scalar(self.scalar)
        ref = self.v - self.scalar
        tst = self.S.to_numpy()
        npt.assert_array_equal(ref, tst)

    def test_timeseq_scalar(self):
        self.S.timeseq_scalar(self.scalar)
        ref = self.v * self.scalar
        tst = self.S.to_numpy()
        npt.assert_array_equal(ref, tst)

    def test_shallow_modify_scalar(self):
        S = self.S.copy(False)
        self.S.minuseq_scalar(self.scalar)
        ref = self.v - self.scalar
        tst = S.to_numpy()
        npt.assert_array_equal(ref, tst)

    def test_pluseq(self):
        ref = self.v + self.v2
        self.S.pluseq(self.S2)
        tst = self.S.to_numpy()
        # Preservation
        npt.assert_array_equal(self.v2, self.S2.to_numpy())
        npt.assert_array_equal(ref, tst)

    def test_minuseq(self):
        ref = self.v - self.v2
        self.S.minuseq(self.S2)
        tst = self.S.to_numpy()
        # Preservation
        npt.assert_array_equal(self.v2, self.S2.to_numpy())
        npt.assert_array_equal(ref, tst)

    def test_timeseq(self):
        ref = self.v * self.v2
        self.S.timeseq(self.S2)
        tst = self.S.to_numpy()
        # Preservation
        npt.assert_array_equal(self.v2, self.S2.to_numpy())
        npt.assert_array_equal(ref, tst)


class SizedArray_i_test(unittest.TestCase):

    def setUp(self):
        self.scalar = 23
        self.v = np.ones(8, dtype=np.int32)
        self.v2 = self.v + 2
        self.S = SizedArrayPtr_i(self.v.copy())
        self.S2 = SizedArrayPtr_i(self.v2.copy())

    def test_deep_copy(self):
        v = self.v.copy()
        S = self.S.copy(True)
        npt.assert_array_equal(v, S.to_numpy())

    def test_shallow_copy(self):
        v = self.v.copy()
        S = self.S.copy(False)
        npt.assert_array_equal(v, S.to_numpy())
        del self.S
        npt.assert_raises(RuntimeError, S.to_numpy)

    def test_pluseq_scalar(self):
        self.S.pluseq_scalar(self.scalar)
        ref = self.v + self.scalar
        tst = self.S.to_numpy()
        npt.assert_array_equal(ref, tst)

    def test_minuseq_scalar(self):
        self.S.minuseq_scalar(self.scalar)
        ref = self.v - self.scalar
        tst = self.S.to_numpy()
        npt.assert_array_equal(ref, tst)

    def test_timeseq_scalar(self):
        self.S.timeseq_scalar(self.scalar)
        ref = self.v * self.scalar
        tst = self.S.to_numpy()
        npt.assert_array_equal(ref, tst)

    def test_shallow_modify_scalar(self):
        S = self.S.copy(False)
        self.S.minuseq_scalar(self.scalar)
        ref = self.v - self.scalar
        tst = S.to_numpy()
        npt.assert_array_equal(ref, tst)

    def test_pluseq(self):
        ref = self.v + self.v2
        self.S.pluseq(self.S2)
        tst = self.S.to_numpy()
        # Preservation
        npt.assert_array_equal(self.v2, self.S2.to_numpy())
        npt.assert_array_equal(ref, tst)

    def test_minuseq(self):
        ref = self.v - self.v2
        self.S.minuseq(self.S2)
        tst = self.S.to_numpy()
        # Preservation
        npt.assert_array_equal(self.v2, self.S2.to_numpy())
        npt.assert_array_equal(ref, tst)

    def test_timeseq(self):
        ref = self.v * self.v2
        self.S.timeseq(self.S2)
        tst = self.S.to_numpy()
        # Preservation
        npt.assert_array_equal(self.v2, self.S2.to_numpy())
        npt.assert_array_equal(ref, tst)
