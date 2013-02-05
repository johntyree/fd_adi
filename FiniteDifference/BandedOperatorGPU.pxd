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

REAL = np.float64

cdef extern from "_BandedOperatorGPU.cuh":

    cdef cppclass SizedArray [T]:
        # T *data
        Py_ssize_t size
        Py_ssize_t ndim
        Py_ssize_t[8] shape
        cpp_string name
        SizedArray(T*, int, np.npy_intp*, cpp_string name)
        T &operator()(int i)
        T &operator()(int i, int j)
        T &idx(int i)
        T &idx(int i, int j)
        void reshape(Py_ssize_t h, Py_ssize_t w)
        void flatten()
        void transpose(int)
        cpp_string show()


    cdef cppclass _BandedOperator:
        void view()
        cbool is_folded()
        SizedArray[double] *apply(SizedArray[double] &)
        void add_scalar(double val)
        void vectorized_scale(SizedArray[double] vector)
        void add_operator(_BandedOperator &other)
        void status()
        int solve(SizedArray[double] &)
        SizedArray[int] offsets
        SizedArray[double] diags
        SizedArray[double] R
        _BandedOperator(
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
    void cout(_BandedOperator *a)
    cpp_string to_string(SizedArray[int] *a)
    cpp_string to_string(SizedArray[double] *a)
    cpp_string to_string(SizedArray[double] a)
    cpp_string to_string(_BandedOperator *a)


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
        top_factors, bottom_factors

    cdef _BandedOperator *thisptr

    cdef inline cbool _is_folded(self)

    cpdef copy(self)
    cpdef emigrate(self, tag=*)
    cpdef immigrate(self, tag=*)
    cpdef diagonalize(self)
    cpdef undiagonalize(self)
    cpdef foldbottom(self, unfold=*)
    cpdef foldtop(self, unfold=*)
    cpdef fold_vector(self, double[:] v, unfold=*)
    cpdef cbool is_tridiagonal(self)
    cpdef apply(self, V, overwrite=*)
    cpdef apply2(self, np.ndarray V, overwrite=*)
    cpdef solve(self, V, overwrite=*)
    cpdef solve2(self, np.ndarray V, overwrite=*)
    cpdef applyboundary(self, boundary, mesh)
    cpdef splice_with(self, begin, at, inplace=*)
    cpdef add_operator(BandedOperator self, BandedOperator other, cbool inplace=*)
    cpdef add_scalar(self, float other, cbool inplace=*)
    cpdef vectorized_scale(self, np.ndarray arr)
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

