# coding: utf8

import numpy as np
cimport numpy as np

from FiniteDifference.VecArray cimport SizedArray

from libcpp.string cimport string as cpp_string
from libcpp cimport bool as cbool

cdef class SizedArrayPtr(object):
    cdef SizedArray[double] *p
    cdef cpp_string tag
    cdef store(self, SizedArray[double] *p, cpp_string tag=*)
    cpdef from_numpy(self, np.ndarray a, cpp_string tag=*)
    cpdef to_numpy(self)
    cpdef SizedArrayPtr copy(self, cbool deep)


cdef class SizedArrayPtr_i(object):
    cdef SizedArray[int] *p
    cdef cpp_string tag
    cdef store(self, SizedArray[int] *p, cpp_string tag=*)
    cpdef from_numpy(self, np.ndarray a, cpp_string tag=*)
    cpdef to_numpy(self)
    cpdef SizedArrayPtr_i copy(self, cbool deep)

cdef from_SizedArray(SizedArray[double] &v)
cdef from_SizedArray_i(SizedArray[int] &v)
