# coding: utf8

import sys
import os
import itertools as it

import numpy as np
cimport numpy as np

from cpython cimport bool
from libcpp cimport bool as cbool
from libcpp.pair cimport pair
from libcpp.string cimport string as cpp_string

from FiniteDifference.thrust.host_vector cimport host_vector
from FiniteDifference.thrust.device_vector cimport device_vector

REAL = np.float64

cdef extern from "backtrace.h":
    pass

cdef extern from "filter.h":
    pass

cdef extern from "VecArray.h":

    cdef cppclass GPUVec[T](device_vector):
        pass

    cdef cppclass SizedArray[T]:
        # T *data
        Py_ssize_t size
        Py_ssize_t ndim
        Py_ssize_t[8] shape
        cpp_string name
        SizedArray(T*, int, np.npy_intp*, cpp_string name) except +
        T get(int i) except +
        T get(int i, int j) except +
        void reshape(Py_ssize_t h, Py_ssize_t w) except +
        void flatten() except +
        void transpose(int) except +
        cpp_string show()

cdef extern from "_CSRBandedOperatorGPU.cuh":

    cdef cppclass _CSRBandedOperator:
        Py_ssize_t operator_rows
        Py_ssize_t blocks
        GPUVec[double] data
        GPUVec[int] row_ptr
        GPUVec[int] row_ind
        GPUVec[int] col_ind
        SizedArray[double] *apply(SizedArray[double] &) except +
        void vectorized_scale(SizedArray[double] &vector) except +

        _CSRBandedOperator(
            SizedArray[double] &data,
            SizedArray[int] &row_ptr,
            SizedArray[int] &row_ind,
            SizedArray[int] &col_ind,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            cpp_string name
        ) except +

cdef extern from "_TriBandedOperatorGPU.cuh":

    cdef cppclass _TriBandedOperator:
        cbool is_folded() except +
        cbool has_residual
        cbool bottom_is_folded
        cbool top_is_folded
        SizedArray[double] *apply(SizedArray[double] &) except +
        void add_scalar(double val) except +
        void vectorized_scale(SizedArray[double] &vector) except +
        void add_operator(_TriBandedOperator &other) except +
        Py_ssize_t operator_rows
        Py_ssize_t block_len
        Py_ssize_t blocks
        void undiagonalize() except +
        void diagonalize() except +
        void fold_bottom(cbool unfold) except +
        void fold_top(cbool unfold) except +
        int solve(SizedArray[double] &) except +
        SizedArray[double] bottom_factors
        SizedArray[double] top_factors
        SizedArray[double] high_dirichlet,
        SizedArray[double] low_dirichlet,
        SizedArray[double] diags
        SizedArray[double] R
        cbool has_high_dirichlet,
        cbool has_low_dirichlet,
        _TriBandedOperator(
            SizedArray[double] data,
            SizedArray[double] R,
            SizedArray[double] high_dirichlet,
            SizedArray[double] low_dirichlet,
            SizedArray[double] top_factors,
            SizedArray[double] bottom_factors,
            unsigned int axis,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            cbool has_high_dirichlet,
            cbool has_low_dirichlet,
            cbool top_is_folded,
            cbool bottom_is_folded,
            cbool has_residual
        ) except +


    void cout(SizedArray[int] *a)
    void cout(SizedArray[double] *a)
    void cout(SizedArray[double] a)
    void cout(_TriBandedOperator *a)
    cpp_string to_string(double *)
    cpp_string to_string(void  *)
    cpp_string to_string(cbool)
    cpp_string to_string(SizedArray[int] *a)
    cpp_string to_string(SizedArray[double] *a)
    cpp_string to_string(SizedArray[double] a)
    cpp_string to_string(_TriBandedOperator *a)


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

    cpdef cbool is_folded(self)

    cdef inline no_mixed(self)

    cpdef copy(self)
    cpdef emigrate(self, other, tag=*)
    cdef  emigrate_tri(self, other, tag=*)
    cdef  emigrate_csr(self, other, tag=*)
    cpdef immigrate(self, tag=*)
    cdef  immigrate_tri(self, tag=*)
    cdef  immigrate_csr(self, tag=*)
    cpdef diagonalize(self)
    cpdef undiagonalize(self)
    cpdef fold_bottom(self, unfold=*)
    cpdef fold_top(self, unfold=*)
    cpdef apply(self, np.ndarray V, overwrite=*)
    cpdef solve(self, np.ndarray V, overwrite=*)
    cpdef add_operator(BandedOperator self, BandedOperator other)
    cpdef add_scalar(self, float other)
    cpdef vectorized_scale(self, np.ndarray arr)

    cpdef mul(self, val, inplace=*)
    cpdef add(self, val, inplace=*)

cdef inline int sign(int i)

cdef inline unsigned int get_real_index(double[:] haystack, double needle) except +
cdef inline unsigned int get_int_index(int[:] haystack, int needle) except +

cdef  scipy_to_cublas(B)
cdef  cublas_to_scipy(B)
