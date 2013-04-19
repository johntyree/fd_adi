# coding: utf8


import numpy as np
cimport numpy as np

from cpython cimport bool
from libcpp cimport bool as cbool
from libcpp.string cimport string as cpp_string

from _CSRBandedOperatorGPU cimport _CSRBandedOperator
from _CSRBandedOperatorGPU cimport mixed_for_vector as mixed_for_vector_
from _TriBandedOperatorGPU cimport _TriBandedOperator, to_string
from _TriBandedOperatorGPU cimport for_vector as for_vector_
from FiniteDifference.VecArray cimport SizedArray
from FiniteDifference.SizedArrayPtr cimport SizedArrayPtr, SizedArrayPtr_i
from FiniteDifference.SizedArrayPtr cimport from_SizedArray, from_SizedArray_i


# Include these headers for debugging.
cdef extern from "backtrace.h":
    pass
cdef extern from "filter.h":
    pass


cdef class BandedOperator(object):

    cdef public:
        attrs
        axis
        blocks
        bottom_fold_status
        cbool is_mixed_derivative
        deltas
        derivative
        order
        top_fold_status

    cdef _TriBandedOperator *thisptr_tri
    cdef _CSRBandedOperator *thisptr_csr

    # cdef void (*coefficient_scale)(double *, Py_ssize_t, double *, Py_ssize_t)

    cdef  emigrate_csr(self, other, tag=*)
    cdef  emigrate_tri(self, other, tag=*)
    cdef  immigrate_csr(self, tag=*)
    cdef  immigrate_tri(self, tag=*)
    cdef inline no_mixed(self)
    cpdef add(self, val, inplace=*)
    cpdef add_operator(BandedOperator self, BandedOperator other)
    cpdef add_scalar_from_host(self, float other)
    cpdef add_scalar(self, SizedArrayPtr other, Py_ssize_t index)
    cpdef apply(self, np.ndarray V, overwrite=*)
    cpdef apply_(self, SizedArrayPtr V, overwrite)
    cpdef enable_residual(self, cbool)
    cpdef cbool is_folded(self)
    cpdef cbool is_foldable(self)
    cpdef copy(self)
    cpdef diagonalize(self)
    cpdef emigrate(self, other, tag=*)
    cpdef fold_bottom(self, unfold=*)
    cpdef fold_top(self, unfold=*)
    cpdef immigrate(self, tag=*)
    cpdef mul_scalar_from_host(self, double v, bool inplace=*)
    cpdef solve(self, np.ndarray V, overwrite=*)
    cpdef solve_(self, SizedArrayPtr V, overwrite)
    cpdef fake_solve_(self, SizedArrayPtr V, overwrite)
    cpdef undiagonalize(self)
    cpdef vectorized_scale_from_host(self, np.ndarray vector)
    cpdef vectorized_scale(self, SizedArrayPtr vector)


cpdef for_vector(np.ndarray v, int blocks, int derivative, int axis, barrier)
cpdef mixed_for_vector(np.ndarray v0, np.ndarray v1)

cdef inline int sign(int i)

cpdef  cublas_to_scipy(B)
cpdef  scipy_to_cublas(B)
