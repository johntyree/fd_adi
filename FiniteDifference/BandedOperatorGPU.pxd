# coding: utf8

import numpy as np
cimport numpy as np

from cpython cimport bool
from libcpp cimport bool as cbool
from libcpp.string cimport string as cpp_string

from _CSRBandedOperatorGPU cimport _CSRBandedOperator
from _TriBandedOperatorGPU cimport _TriBandedOperator, to_string
from FiniteDifference.VecArray cimport SizedArray
from FiniteDifference.SizedArrayPtr cimport SizedArrayPtr

cdef extern from "backtrace.h":
    pass


cdef extern from "filter.h":
    pass


cdef class BandedOperator(object):
    cdef public:
        attrs
        axis
        derivative
        order
        blocks
        cbool is_mixed_derivative
        top_fold_status
        bottom_fold_status
        deltas

    cdef _TriBandedOperator *thisptr_tri
    cdef _CSRBandedOperator *thisptr_csr

    cdef  emigrate_csr(self, other, tag=*)
    cdef  emigrate_tri(self, other, tag=*)
    cdef  immigrate_csr(self, tag=*)
    cdef  immigrate_tri(self, tag=*)
    cdef inline no_mixed(self)
    cpdef add(self, val, inplace=*)
    cpdef add_operator(BandedOperator self, BandedOperator other)
    cpdef add_scalar(self, float other)
    cpdef apply(self, np.ndarray V, overwrite=*)
    cpdef cbool is_folded(self)
    cpdef copy(self)
    cpdef diagonalize(self)
    cpdef emigrate(self, other, tag=*)
    cpdef fold_bottom(self, unfold=*)
    cpdef fold_top(self, unfold=*)
    cpdef immigrate(self, tag=*)
    cpdef mul(self, val, inplace=*)
    cpdef solve(self, np.ndarray V, overwrite=*)
    cpdef undiagonalize(self)
    cpdef vectorized_scale(self, np.ndarray arr)

cdef inline int sign(int i)

cdef inline unsigned int get_real_index(double[:] haystack, double needle)
cdef inline unsigned int get_int_index(int[:] haystack, int needle)

cdef  cublas_to_scipy(B)
cdef  scipy_to_cublas(B)
