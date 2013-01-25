# coding: utf8

import sys
import os
import itertools as it

import numpy as np
cimport numpy as np

from cpython cimport bool
from libcpp cimport bool as cbool
from libcpp.pair cimport pair

REAL = np.float64
ctypedef double double

cdef extern from "_BandedOperatorGPU.cuh" namespace "CPU":


    cdef cppclass SizedArray [T]:
        T *data
        Py_ssize_t size
        Py_ssize_t ndim
        Py_ssize_t[8] shape
        SizedArray(T*, Py_ssize_t, Py_ssize_t*)
        T &operator()(int i)
        T &operator()(int i, int j)

    void cout(SizedArray[int] *a)

    cdef cppclass _BandedOperator:
        void view()
        cbool is_folded()
        void apply(SizedArray[double], bool)
        void add_scalar(double val)
        void vectorized_scale(SizedArray[double] vector)
        void status()
        SizedArray[int] offsets
        SizedArray[double] data
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
    cpdef emigrate(self)
    cpdef immigrate(self)
    cpdef diagonalize(self)
    cpdef undiagonalize(self)
    cpdef foldbottom(self, unfold=*)
    cpdef foldtop(self, unfold=*)
    cpdef fold_vector(self, double[:] v, unfold=*)
    cpdef cbool is_tridiagonal(self)
    cpdef apply(self, V, overwrite=*)
    cpdef solve(self, V, overwrite=*)
    cpdef applyboundary(self, boundary, mesh)
    cpdef splice_with(self, begin, at, inplace=*)
    cpdef add_operator(BandedOperator self, BandedOperator other, cbool inplace=*)
    cpdef add_scalar(self, float other, cbool inplace=*)
    cpdef vectorized_scale(self, double[:] arr)
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

