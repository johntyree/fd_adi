# coding: utf8

cimport numpy as np

# cimport cython
from cpython cimport bool
from libcpp cimport bool as cbool


REAL = np.float64


cdef class BandedOperator(object):
    cdef public:
        attrs
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
        cbool is_mixed_derivative
        top_factors, bottom_factors
        cbool top_is_folded
        cbool bottom_is_folded


    cpdef cbool is_folded(self)

    cpdef copy(self)
    cpdef diagonalize(self)
    cpdef undiagonalize(self)
    cpdef fold_bottom(self, unfold=*)
    cpdef fold_top(self, unfold=*)
    cpdef fold_vector(self, double[:] v, unfold=*)
    cpdef cbool is_tridiagonal(self)
    cpdef apply(self, V, overwrite=*)
    cpdef solve(self, V, overwrite=*)
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

cdef inline unsigned int get_real_index(double[:] haystack, double needle) except +
cdef inline unsigned int get_int_index(int[:] haystack, int needle) except +

