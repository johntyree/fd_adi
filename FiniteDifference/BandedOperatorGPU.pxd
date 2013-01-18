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
ctypedef np.float64_t REAL_t

cdef extern from "_BandedOperatorGPU.cuh" namespace "CPU":

    void print_array(double *, Py_ssize_t)

    void vectorized_scale(
          pair[Py_ssize_t, double*] vector
        , pair[Py_ssize_t, double*] data
        , pair[Py_ssize_t, double*] R
        , pair[Py_ssize_t, int*] offsets
        , Py_ssize_t operator_rows
        , Py_ssize_t blocks
        , cbool low_dirichlet
        , cbool high_dirichlet
    )


cdef class BandedOperator(object):
    cdef public attrs
    cdef public unsigned int blocks, order, axis
    cdef public D, R, deltas, dirichlet, solve_banded_offsets, derivative
    cdef public shape
    cdef public top_factors, bottom_factors

    cdef inline cbool _is_folded(self)

    cpdef copy(self)
    cpdef diagonalize(self)
    cpdef undiagonalize(self)
    cpdef foldbottom(self, unfold=*)
    cpdef foldtop(self, unfold=*)
    cpdef fold_vector(self, REAL_t[:] v, unfold=*)
    cpdef cbool is_tridiagonal(self)
    cpdef apply(self, V, overwrite=*)
    cpdef solve(self, V, overwrite=*)
    cpdef applyboundary(self, boundary, mesh)
    cpdef splice_with(self, begin, at, inplace=*)
    cpdef add_operator(BandedOperator self, BandedOperator other, cbool inplace=*)
    cpdef add_scalar(self, float other, cbool inplace=*)
    cpdef vectorized_scale(self, REAL_t[:] arr)
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

cdef inline unsigned int get_real_index(REAL_t[:] haystack, REAL_t needle)

cdef inline unsigned int get_int_index(int[:] haystack, int needle)

