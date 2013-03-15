# coding: utf8


import numpy as np
cimport numpy as np

from libcpp.string cimport string as cpp_string
from libcpp cimport bool as cbool

from FiniteDifference.VecArray cimport SizedArray


cdef class SizedArrayPtr(object):

    cdef SizedArray[double] *p
    cdef public cpp_string tag
    cdef public Py_ssize_t size

    cdef store(self, SizedArray[double] *p, cpp_string tag=*)
    cpdef alloc(self, int sz, cpp_string tag=*)
    cpdef from_numpy(self, np.ndarray a, cpp_string tag=*)
    cpdef to_numpy(self)
    cpdef SizedArrayPtr copy(self, cbool deep)
    cpdef copy_from(self, SizedArrayPtr other)
    cpdef pluseq(self, SizedArrayPtr other)
    cpdef minuseq(self, SizedArrayPtr other)
    cpdef minuseq_over2(self, SizedArrayPtr other)
    cpdef timeseq(self, SizedArrayPtr other)
    cpdef pluseq_scalar(self, double other)
    cpdef minuseq_scalar(self, double other)
    cpdef timeseq_scalar(self, double other)


cdef class SizedArrayPtr_i(object):

    cdef SizedArray[int] *p
    cdef public cpp_string tag
    cdef public Py_ssize_t size

    cdef store(self, SizedArray[int] *p, cpp_string tag=*)
    cpdef alloc(self, int sz, cpp_string tag=*)
    cpdef from_numpy(self, np.ndarray a, cpp_string tag=*)
    cpdef to_numpy(self)
    cpdef SizedArrayPtr_i copy(self, cbool deep)
    cpdef copy_from(self, SizedArrayPtr_i other)
    cpdef pluseq(self, SizedArrayPtr_i other)
    cpdef minuseq(self, SizedArrayPtr_i other)
    cpdef timeseq(self, SizedArrayPtr_i other)
    cpdef pluseq_scalar(self, int other)
    cpdef minuseq_scalar(self, int other)
    cpdef timeseq_scalar(self, int other)


cdef from_SizedArray(SizedArray[double] &v)

cdef from_SizedArray_i(SizedArray[int] &v)


cdef SizedArray[double]* to_SizedArray(np.ndarray v, cpp_string name) except NULL

cdef SizedArray[int]* to_SizedArray_i(np.ndarray v, cpp_string name) except NULL
