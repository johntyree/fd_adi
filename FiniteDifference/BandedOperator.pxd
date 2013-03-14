# coding: utf8

cimport numpy as np

from cpython cimport bool
from libcpp cimport bool as cbool

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
        top_fold_status
        bottom_fold_status

    cpdef add(self, val, inplace=*)
    cpdef add_operator(BandedOperator self, BandedOperator other, cbool inplace=*)
    cpdef add_scalar(self, float other, cbool inplace=*)
    cpdef apply(self, V, overwrite=*)
    cpdef applyboundary(self, boundary, mesh)
    cpdef cbool is_folded(self)
    cpdef cbool is_foldable(self)
    cpdef cbool is_tridiagonal(self)
    cpdef clear_residual(self)
    cpdef copy(self)
    cpdef diagonalize(self)
    cpdef fold_bottom(self, unfold=*)
    cpdef fold_top(self, unfold=*)
    cpdef fold_vector(self, double[:] v, unfold=*)
    cpdef mul(self, val, inplace=*)
    cpdef scale(self, func)
    cpdef solve(self, V, overwrite=*)
    cpdef splice_with(self, begin, at, inplace=*)
    cpdef undiagonalize(self)
    cpdef vectorized_scale(self, np.ndarray arr)


cpdef for_vector(vector, scheme=*, derivative=*, order=*, residual=*, force_bandwidth=*, axis=*)

cpdef backwardcoeffs(deltas, derivative=*, order=*, force_bandwidth=*)
cpdef centercoeffs(deltas, derivative=*, order=*, force_bandwidth=*)
cpdef forwardcoeffs(deltas, derivative=*, order=*, force_bandwidth=*)

cpdef check_derivative(d)
cpdef check_order(order)

cdef inline int sign(int i)

cdef inline unsigned int get_real_index(double[:] haystack, double needle)
cdef inline unsigned int get_int_index(int[:] haystack, int needle)
