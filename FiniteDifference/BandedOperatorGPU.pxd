# coding: utf8

import numpy as np
cimport numpy as np

from cpython cimport bool
from libcpp cimport bool as cbool
from libcpp.string cimport string as cpp_string

from _CSRBandedOperatorGPU cimport _CSRBandedOperator
from _TriBandedOperatorGPU cimport _TriBandedOperator, to_string
from FiniteDifference.VecArray cimport SizedArray

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
        cbool top_is_folded
        cbool bottom_is_folded
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


cdef class SizedArrayPtr(object):
    cdef SizedArray[double] *p
    cdef cpp_string tag
    cdef store(self, SizedArray[double] *p, cpp_string tag=*)
    cpdef from_numpy(self, np.ndarray a, cpp_string tag=*)
    cpdef to_numpy(self)


cdef inline int sign(int i)

cdef inline unsigned int get_real_index(double[:] haystack, double needle) except +
cdef inline unsigned int get_int_index(int[:] haystack, int needle) except +

cdef  cublas_to_scipy(B)
cdef  scipy_to_cublas(B)
