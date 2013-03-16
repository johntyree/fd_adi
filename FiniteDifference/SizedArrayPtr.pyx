# coding: utf8
# cython: annotate = True
# cython: infer_types = True
# cython: embedsignature = True
# distutils: language = c++
# distutils: sources = FiniteDifference/backtrace.c FiniteDifference/filter.c FiniteDifference/_SizedArrayPtr_GPU_Code.cu


from __future__ import division

import numpy as np
cimport numpy as np

from libcpp.string cimport string as cpp_string
from libcpp cimport bool as cbool
from cython.operator cimport dereference as deref

from FiniteDifference.VecArray cimport to_string

from FiniteDifference.thrust.copy cimport copy_n

cdef class SizedArrayPtr(object):

    def __init__(self, a=None, tag="Unknown"):
        if a is not None:
            self.from_numpy(a, tag)
            # print "imported from numpy in constructor"


    cpdef alloc(self, int sz, cpp_string tag="Unknown"):
        if self.p:
            del self.p
        self.p = new SizedArray[double](sz, tag)
        self.size = sz
        return self


    cdef store(self, SizedArray[double] *p, cpp_string tag="Unknown"):
        if self.p:
            raise RuntimeError("SizedArrayPtr is single assignment")
        self.p = p
        self.size = self.p.size
        # print "SAPtr -> Storing %s:" % tag, to_string(p)


    cpdef from_numpy(self, np.ndarray a, cpp_string tag="Unknown"):
        if self.p:
            print "SizedArrayPtr is single assignment"
            raise RuntimeError("SizedArrayPtr is single assignment")
        self.p = to_SizedArray(a, tag)
        self.size = self.p.size
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


    cpdef copy_from(self, SizedArrayPtr other):
        self.p.copy_from(deref(other.p))


    cpdef pluseq(self, SizedArrayPtr other):
        self.p.pluseq(deref(other.p))


    cpdef minuseq(self, SizedArrayPtr other):
        self.p.minuseq(deref(other.p))


    cpdef minuseq_over2(self, SizedArrayPtr other):
        self.p.minuseq_over2(deref(other.p))


    cpdef timeseq(self, SizedArrayPtr other):
        self.p.timeseq(deref(other.p))


    cpdef pluseq_scalar(self, double other):
        self.p.pluseq(other)


    cpdef minuseq_scalar(self, double other):
        self.p.minuseq(other)


    cpdef timeseq_scalar(self, double other):
        self.p.timeseq(other)


    def __dealloc__(self):
        if self.p:
            # print "Freeing %s:" % (self.tag,), to_string(self.p)
            del self.p


    def __str__(self):
        return "SizedArrayPtr (%s)@%s" % (self.tag, to_string(self.p))


cdef class SizedArrayPtr_i(object):

    def __init__(self, a=None, tag="Unknown"):
        if a is not None:
            self.from_numpy(a, tag)
            # print "imported from numpy in constructor"


    cpdef alloc(self, int sz, cpp_string tag="Unknown"):
        if self.p:
            del self.p
        self.p = new SizedArray[int](sz, tag)
        self.size = self.p.size
        return self


    cdef store(self, SizedArray[int] *p, cpp_string tag="Unknown"):
        if self.p:
            raise RuntimeError("SizedArrayPtr_i is single assignment")
        self.p = p
        self.size = self.p.size
        # print "SAPtr -> Storing %s:" % tag, to_string(p)


    cpdef SizedArrayPtr_i copy(self, cbool deep):
        cdef SizedArray[int] *x = new SizedArray[int](deref(self.p), deep)
        u = SizedArrayPtr_i()
        u.store(x)
        return u


    cpdef copy_from(self, SizedArrayPtr_i other):
        self.p.copy_from(deref(other.p))


    cpdef from_numpy(self, np.ndarray a, cpp_string tag="Unknown"):
        if self.p:
            print "SizedArrayPtr_i is single assignment"
            raise RuntimeError("SizedArrayPtr_i is single assignment")
        self.p = to_SizedArray_i(a, tag)
        self.size = self.p.size
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
        self.p.pluseq(other)


    cpdef minuseq_scalar(self, int other):
        self.p.minuseq(other)


    cpdef timeseq_scalar(self, int other):
        self.p.timeseq(other)


    def __dealloc__(self):
        if self.p:
            # print "Freeing %s:" % (self.tag,), to_string(self.p)
            del self.p


    def __str__(self):
        return "SizedArrayPtr (%s)@%s" % (self.tag, to_string(self.p))


cdef SizedArray[double]* to_SizedArray(np.ndarray v, cpp_string name) except NULL:
    assert v.dtype.type == np.float64, ("Types don't match! Got (%s) expected (%s)."
                                      % (v.dtype.type, np.float64))
    assert v.size, "Can't convert a size 0 vector. (%s)" % v.size
    if not v.flags.c_contiguous:
        v = v.copy("C")
    return new SizedArray[double](<double *>np.PyArray_DATA(v), v.ndim, v.shape, name, True)


cdef SizedArray[int]* to_SizedArray_i(np.ndarray v, cpp_string name) except NULL:
    assert v.dtype.type == np.int32, ("Types don't match! Got (%s) expected (%s)."
                                      % (v.dtype.type, np.int32))
    assert v.size, "Can't convert a size 0 vector. (%s)" % v.size
    if not v.flags.c_contiguous:
        v = v.copy("C")
    return new SizedArray[int](<int *>np.PyArray_DATA(v), v.ndim, v.shape, name, True)


cdef from_SizedArray_i(SizedArray[int] &v):
    s = np.empty(v.size, dtype=np.int32)
    cdef int i
    copy_n(v.data, v.size, <int *>np.PyArray_DATA(s))
    shp = []
    for i in range(v.ndim):
        shp.append(v.shape[i])
    s = s.reshape(shp)
    return s


cdef from_SizedArray(SizedArray[double] &v):
    s = np.empty(v.size, dtype=float)
    cdef int i
    copy_n(v.data, v.size, <double *>np.PyArray_DATA(s))
    shp = []
    for i in range(v.ndim):
        shp.append(v.shape[i])
    s = s.reshape(shp)
    return s
