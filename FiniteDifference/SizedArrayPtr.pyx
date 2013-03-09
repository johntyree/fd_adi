# coding: utf8
# cython: annotate = True
# cython: infer_types = True
# cython: embedsignature = True
# distutils: language = c++
# distutils: sources = FiniteDifference/backtrace.c FiniteDifference/filter.c

from __future__ import division

import sys
import os
import itertools as it

import numpy as np
cimport numpy as np
from FiniteDifference.VecArray cimport to_string

from libcpp.string cimport string as cpp_string

cimport cython
from cython.operator cimport dereference as deref

cdef class SizedArrayPtr(object):

    def __init__(self, a=None, tag="Unknown"):
        if a is not None:
            self.from_numpy(a, tag)
            # print "imported from numpy in constructor"

    cdef store(self, SizedArray[double] *p, cpp_string tag="Unknown"):
        if self.p:
            raise RuntimeError("SizedArrayPtr is single assignment")
        self.p = new SizedArray[double](deref(p))
        # print "SAPtr -> Storing %s:" % tag, to_string(p)

    cpdef from_numpy(self, np.ndarray a, cpp_string tag="Unknown"):
        if self.p:
            print "SizedArrayPtr is single assignment"
            raise RuntimeError("SizedArrayPtr is single assignment")
        self.p = to_SizedArray(a, tag)
        # print "Numpy -> Storing %s: %s" % (tag, to_string(self.p))

    cpdef to_numpy(self):
        # print "Converting", to_string(deref(self.p))
        # print "ndim", self.p.ndim, self.p.shape[0], self.p.shape[1]
        if self.p.ndim == 2:
            a = from_SizedArray_2(deref(self.p))
        else:
            a = from_SizedArray(deref(self.p))
        assert a.ndim == self.p.ndim
        return a

    def __dealloc__(self):
        if self.p:
            print "Freeing %s:" % (self.tag,), to_string(self.p)
            del self.p

    def __str__(self):
        return "SizedArrayPtr (%s)@%s" % (self.tag, to_string(self.p))



cdef inline from_SizedArray(SizedArray[double] &v):
    sz = v.size
    cdef np.ndarray[double, ndim=1] s = np.empty(sz, dtype=float)
    cdef int i
    for i in range(sz):
        s[i] = v.get(i)
    return s


cdef inline from_SizedArray_2(SizedArray[double] &v):
    assert v.ndim == 2, ("Using from_SizedArray_2 on an array of dim %s" % v.ndim)
    cdef np.ndarray[double, ndim=2] s = np.empty((v.shape[0], v.shape[1]), dtype=float)
    cdef int i, j
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            s[i, j] = v.get(i, j)
    return s

cdef inline SizedArray[double]* to_SizedArray(np.ndarray v, name):
    assert v.dtype.type == np.float64, ("Types don't match! Got (%s) expected (%s)."
                                      % (v.dtype.type, np.float64))
    cdef double *ptr
    if not v.flags.c_contiguous:
        v = v.copy("C")
    return new SizedArray[double](<double *>np.PyArray_DATA(v), v.ndim, v.shape, name)
