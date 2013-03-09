# coding: utf8

import numpy as np
cimport numpy as np

from FiniteDifference.VecArray cimport SizedArray

from libcpp.string cimport string as cpp_string

cdef class SizedArrayPtr(object):
    cdef SizedArray[double] *p
    cdef cpp_string tag
    cdef store(self, SizedArray[double] *p, cpp_string tag=*)
    cpdef from_numpy(self, np.ndarray a, cpp_string tag=*)
    cpdef to_numpy(self)
