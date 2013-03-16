# coding: utf8


cimport numpy as np

cimport cython

from FiniteDifference.thrust.device_ptr cimport device_ptr

cdef extern from "<thrust/copy.h>" namespace "thrust":

    double* copy_n(device_ptr[double], Py_ssize_t, double *) except +
    int* copy_n(device_ptr[int], Py_ssize_t, int *) except +
