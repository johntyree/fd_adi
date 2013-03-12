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
from libcpp cimport bool as cbool

from cython.operator cimport dereference as deref

cdef class SizedArrayPtr(object):

    def __init__(self, a=None, tag="Unknown"):
        if a is not None:
            self.from_numpy(a, tag)
            # print "imported from numpy in constructor"

    # def __iadd__(self, other):
        # self.p

    cdef store(self, SizedArray[double] *p, cpp_string tag="Unknown"):
        if self.p:
            raise RuntimeError("SizedArrayPtr is single assignment")
        self.p = p
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
        a = from_SizedArray(deref(self.p))
        assert a.ndim == self.p.ndim
        return a

    cpdef SizedArrayPtr copy(self, cbool deep):
        cdef SizedArray[double] *x = new SizedArray[double](deref(self.p), deep)
        u = SizedArrayPtr()
        u.store(x)
        return u

    cpdef pluseq(self, SizedArrayPtr other):
        self.p.pluseq(deref(other.p))

    cpdef minuseq(self, SizedArrayPtr other):
        self.p.minuseq(deref(other.p))

    cpdef timeseq(self, SizedArrayPtr other):
        self.p.timeseq(deref(other.p))

    cpdef pluseq_scalar(self, double other):
        self.p.pluseq_scalar(other)

    cpdef minuseq_scalar(self, double other):
        self.p.minuseq_scalar(other)

    cpdef timeseq_scalar(self, double other):
        self.p.timeseq_scalar(other)


    def __dealloc__(self):
        if self.p:
            print "Freeing %s:" % (self.tag,), to_string(self.p)
            del self.p

    def __str__(self):
        return "SizedArrayPtr (%s)@%s" % (self.tag, to_string(self.p))


cdef class SizedArrayPtr_i(object):

    def __init__(self, a=None, tag="Unknown"):
        if a is not None:
            self.from_numpy(a, tag)
            # print "imported from numpy in constructor"

    cdef store(self, SizedArray[int] *p, cpp_string tag="Unknown"):
        if self.p:
            raise RuntimeError("SizedArrayPtr_i is single assignment")
        self.p = p
        # print "SAPtr -> Storing %s:" % tag, to_string(p)

    cpdef SizedArrayPtr_i copy(self, cbool deep):
        cdef SizedArray[int] *x = new SizedArray[int](deref(self.p), deep)
        u = SizedArrayPtr_i()
        u.store(x)
        return u

    cpdef from_numpy(self, np.ndarray a, cpp_string tag="Unknown"):
        if self.p:
            print "SizedArrayPtr_i is single assignment"
            raise RuntimeError("SizedArrayPtr_i is single assignment")
        self.p = to_SizedArray_i(a, tag)
        # print "Numpy -> Storing %s: %s" % (tag, to_string(self.p))

    cpdef to_numpy(self):
        # print "Converting", to_string(deref(self.p))
        # print "ndim", self.p.ndim, self.p.shape[0], self.p.shape[1]
        a = from_SizedArray_i(deref(self.p))
        assert a.ndim == self.p.ndim
        return a

    cpdef pluseq(self, SizedArrayPtr_i other):
        self.p.pluseq(deref(other.p))

    cpdef minuseq(self, SizedArrayPtr_i other):
        self.p.minuseq(deref(other.p))

    cpdef timeseq(self, SizedArrayPtr_i other):
        self.p.timeseq(deref(other.p))

    cpdef pluseq_scalar(self, int other):
        self.p.pluseq_scalar(other)

    cpdef minuseq_scalar(self, int other):
        self.p.minuseq_scalar(other)

    cpdef timeseq_scalar(self, int other):
        self.p.timeseq_scalar(other)

    def __dealloc__(self):
        if self.p:
            print "Freeing %s:" % (self.tag,), to_string(self.p)
            del self.p

    def __str__(self):
        return "SizedArrayPtr (%s)@%s" % (self.tag, to_string(self.p))



cdef SizedArray[double]* to_SizedArray(np.ndarray v, name):
    assert v.dtype.type == np.float64, ("Types don't match! Got (%s) expected (%s)."
                                      % (v.dtype.type, np.float64))
    if not v.flags.c_contiguous:
        v = v.copy("C")
    return new SizedArray[double](<double *>np.PyArray_DATA(v), v.ndim, v.shape, name, True)


cdef SizedArray[int]* to_SizedArray_i(np.ndarray v, cpp_string name):
    assert v.dtype.type == np.int32, ("Types don't match! Got (%s) expected (%s)."
                                      % (v.dtype.type, np.int32))
    if not v.flags.c_contiguous:
        v = v.copy("C")
    return new SizedArray[int](<int *>np.PyArray_DATA(v), v.ndim, v.shape, name, True)


cdef from_SizedArray_i(SizedArray[int] &v):
    s = np.empty(v.size, dtype=np.int32)
    cdef int i, j
    for i in range(v.size):
        s[i] = v.get(i)
    shp = []
    for i in range(v.ndim):
        shp.append(v.shape[i])
    s = s.reshape(shp)
    return s


cdef from_SizedArray(SizedArray[double] &v):
    print "from_SizedArray:", v.name
    s = np.empty(v.size, dtype=float)
    cdef int i, j
    for i in range(v.size):
        s[i] = v.get(i)
    shp = []
    for i in range(v.ndim):
        shp.append(v.shape[i])
    s = s.reshape(shp)
    return s
