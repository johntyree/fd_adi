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
        void reshape(Py_ssize_t h, Py_ssize_t w)
        void flatten()
        void transpose(int)
        cpp_string show()

cdef extern from "_CSRBandedOperatorGPU.cuh":

    cdef cppclass _CSRBandedOperator:
        GPUVec[double] data
        GPUVec[int] row_ind
        GPUVec[int] col_ind
        SizedArray[double] *apply(SizedArray[double] &) except +
        void vectorized_scale(SizedArray[double] &vector) except +

        _CSRBandedOperator(
            SizedArray[double] &data,
            SizedArray[int] &row_ind,
            SizedArray[int] &col_ind,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks
            )

        _CSRBandedOperator(
            SizedArray[double] &data,
            SizedArray[int] &row_ind,
            SizedArray[int] &col_ind,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            cpp_string name
            )

cdef extern from "_TriBandedOperatorGPU.cuh":

    cdef cppclass _TriBandedOperator:
        void view()
        cbool is_folded()
        cbool has_residual
        SizedArray[double] *apply(SizedArray[double] &)
        void add_scalar(double val)
        void vectorized_scale(SizedArray[double] &vector)
        void add_operator(_TriBandedOperator &other)
        void status()
        int solve(SizedArray[double] &)
        SizedArray[int] offsets
        SizedArray[double] diags
        SizedArray[double] R
        _TriBandedOperator(
            SizedArray[double] data,
            SizedArray[double] R,
            SizedArray[int] offsets,
            SizedArray[double] high_dirichlet,
            SizedArray[double] low_dirichlet,
            SizedArray[double] top_factors,
            SizedArray[double] bottom_factors,
            unsigned int axis,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            cbool has_high_dirichlet,
            cbool has_low_dirichlet,
            cbool has_residual
            )


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
        location
        axis
        derivative
        order
        D
        R
        deltas
        dirichlet
        solve_banded_offsets
        unsigned int blocks
        shape
        cbool csr
        top_factors, bottom_factors

    cdef _TriBandedOperator *thisptr_tri
    cdef _CSRBandedOperator *thisptr_csr

    cdef inline cbool _is_folded(self)

    cpdef copy(self)
    cpdef emigrate(self, tag=*)
    cdef  emigrate_tri(self, tag=*)
    cdef  emigrate_csr(self, tag=*)
    cpdef immigrate(self, tag=*)
    cdef  immigrate_tri(self, tag=*)
    cdef  immigrate_csr(self, tag=*)
    cpdef diagonalize(self)
    cpdef undiagonalize(self)
    cpdef foldbottom(self, unfold=*)
    cpdef foldtop(self, unfold=*)
    cpdef fold_vector(self, double[:] v, unfold=*)
    cpdef cbool is_tridiagonal(self)
    cpdef use_csr_format(self, cbool b=*)
    cpdef cbool is_cross_derivative(self)
    cpdef apply(self, V, overwrite=*)
    cpdef apply2(self, np.ndarray V, overwrite=*)
    cpdef solve(self, V, overwrite=*)
    cpdef solve2(self, np.ndarray V, overwrite=*)
    cpdef applyboundary(self, boundary, mesh)
    cpdef splice_with(self, begin, at, inplace=*)
    cpdef add_operator(BandedOperator self, BandedOperator other, cbool inplace=*)
    cpdef add_scalar(self, float other, cbool inplace=*)
    cpdef vectorized_scale(self, np.ndarray arr) except +
    cpdef scale(self, func)

    cpdef mul(self, val, inplace=*)
    cpdef add(self, val, inplace=*)

cpdef for_vector(vector, scheme=*, derivative=*, order=*, residual=*, force_bandwidth=*, axis=*)
cpdef check_derivative(d)
cpdef check_order(order)
cpdef forwardcoeffs(deltas, derivative=*, order=*, force_bandwidth=*)
cpdef centercoeffs(deltas, derivative=*, order=*, force_bandwidth=*)
cpdef backwardcoeffs(deltas, derivative=*, order=*, force_bandwidth=*)

cdef inline int sign(int i)

cdef inline unsigned int get_real_index(double[:] haystack, double needle)
cdef inline unsigned int get_int_index(int[:] haystack, int needle)

