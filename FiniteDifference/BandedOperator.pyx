# coding: utf8
# cython: profile=True
# cython: infer_types=True
"""Description."""

# import sys
# import os
# import itertools as it

from bisect import bisect_left

import numpy as np
cimport numpy as np
import scipy.sparse
import itertools
import utils
import scipy.linalg as spl

cimport cython

from cpython cimport bool
from libcpp cimport bool as cbool

REAL = np.float64
ctypedef np.float64_t REAL_t

cdef class BandedOperator(object):

    attrs = ('derivative', 'order', 'axis', 'deltas', 'dirichlet', 'blocks')

    cdef public unsigned int blocks, order, axis
    cdef public D, R, deltas, dirichlet, solve_banded_offsets, derivative
    cdef public shape

    def __init__(self, data_offsets, residual=None, int inplace=True,
            int derivative=1, int order=2, deltas=None, int axis=0):
        """
        A linear operator for discrete derivatives.
        Consist of a banded matrix (B.D) and a residual vector (B.R) for things
        like

            U2 = L*U1 + Rj  -->   U2 = B.apply(U1)
            U2 = L.I * (U1 - R) --> U2 = B.solve(U1)
        """

        data, offsets = data_offsets
        assert data.shape[1] > 3, "Vector too short to use finite differencing."
        if not inplace:
            data = data.copy()
            offsets = tuple(offsets)
            if residual is not None:
                residual = residual.copy()
        size = data.shape[1]
        self.shape = (size, size)
        self.D = scipy.sparse.dia_matrix((data, offsets), shape=self.shape)
        if residual is None:
            self.R = np.zeros(self.shape[0])
        elif residual.shape[0] == self.shape[0] and residual.ndim == 1:
            self.R = residual
        else:
            raise ValueError("Residual vector has wrong shape: got %i,"
                             "expected %i." % (residual.shape[0], size))

        # NB: When adding something here, also add to BandedOperator.attrs
        self.blocks = 1
        self.derivative = derivative
        self.order = order
        self.deltas = deltas if deltas is not None else np.array([np.nan])
        self.solve_banded_offsets = (abs(min(offsets)), abs(max(offsets)))
        self.dirichlet = [None, None]
        self.axis = axis

    def copy_meta_data(self, other, **kwargs):
        for attr in self.attrs:
            if attr not in kwargs:
                setattr(self, attr, getattr(other, attr))
            else:
                setattr(self, attr, kwargs[attr])
        self.dirichlet = list(other.dirichlet)

    # def __eq__(self, other):
        # return self.__richcmp__(other, 2)

    def __richcmp__(self, other, op):
        true = op == 2
        false = op == 3

        no_nan = np.nan_to_num
        for attr in self.attrs:
            if attr == 'deltas':
                continue
            if getattr(self, attr) != getattr(other, attr):
                return false

        if ((self.D.data == other.D.data).all()
                and (self.D.offsets == other.D.offsets).all()
                and (self.R == other.R).all()
                and (no_nan(self.deltas) == no_nan(other.deltas)).all()
                and (self.shape == other.shape)):
            return true
        else:
            return false


    @classmethod
    def for_vector(cls, vector, scheme="center", derivative=1, order=2,
            residual=None, force_bandwidth=None, axis=0):
        """
        A linear operator for discrete derivative of @vector@.

        @derivative@ is a tuple specifying the sequence of derivatives. For
        example, `(0,0)` is the second derivative in the first dimension.
        """

        cls.check_derivative(derivative)

        deltas = np.hstack((np.nan, np.diff(vector)))
        scheme = scheme.lower()

        bw = force_bandwidth
        if scheme.startswith("forward") or scheme.startswith('up'):
            data, offsets = cls.forwardcoeffs(deltas, derivative=derivative, order=order, force_bandwidth=bw)
        elif scheme.startswith("backward") or scheme.startswith('down'):
            data, offsets = cls.backwardcoeffs(deltas, derivative=derivative, order=order, force_bandwidth=bw)
        elif scheme.startswith("center") or scheme == "":
            data, offsets = cls.centercoeffs(deltas, derivative=derivative, order=order, force_bandwidth=bw)
        else:
            raise ValueError("Unknown scheme: %s" % scheme)

        self = BandedOperator((data, offsets), residual=residual, axis=axis)
        self.derivative = derivative
        self.order = order
        self.deltas = deltas
        self.axis = axis
        return self

    def copy(self):
        B = BandedOperator((self.D.data, self.D.offsets), residual=self.R, inplace=False)
        B.copy_meta_data(self)
        return B


    def apply(self, V, overwrite=False):
        if not overwrite:
            V = V.copy()
        t = range(V.ndim)
        utils.rolllist(t, self.axis, V.ndim-1)
        V = np.transpose(V, axes=t)

        if self.dirichlet[0] is not None:
            # print "Setting V[0,:] to", self.dirichlet[0]
            V[...,0] = self.dirichlet[0]
        if self.dirichlet[1] is not None:
            # print "Setting V[-1,:] to", self.dirichlet[-1]
            V[...,-1] = self.dirichlet[1]


        if self.R is not None:
            ret = self.D.dot(V.flat) + self.R
        else:
            ret = self.D.dot(V.flat)

        ret = ret.reshape(V.shape)

        t = range(V.ndim)
        utils.rolllist(t, V.ndim-1, self.axis)
        ret = np.transpose(ret, axes=t)

        return ret

    def solve(self, V, overwrite=False):
        if not overwrite:
            V = V.copy()
        t = range(V.ndim)
        utils.rolllist(t, self.axis, V.ndim-1)
        V = np.transpose(V, axes=t)

        if self.dirichlet[0] is not None:
            V[...,0] = self.dirichlet[0]
        if self.dirichlet[1] is not None:
            V[...,-1] = self.dirichlet[1]

        if self.R is not None:
            V0 = V.flat - self.R
        else:
            V0 = V

        ret = spl.solve_banded(self.solve_banded_offsets,
                self.D.data, V0.flat,
                overwrite_ab=overwrite, overwrite_b=True).reshape(V.shape)

        t = range(V.ndim)
        utils.rolllist(t, V.ndim-1, self.axis)
        ret = np.transpose(ret, axes=t)

        return ret


    @cython.boundscheck(False)
    def applyboundary(self, boundary, mesh):
        """
        @boundary@ is a tuple from FiniteDifferenceEngine.boundaries.

        data are the packed diagonals and residual is the residual vector.
        """
        B = self
        cdef REAL_t[:,:] Bdata = B.D.data
        cdef REAL_t[:] R
        if B.R is None:
            R = np.zeros(B.shape[0])
        else:
            R = B.R
        # cdef int lower_type, upper_type
        # cdef REAL_t lower_val, upper_val
        cdef int m = get_int_index(B.D.offsets, 0)
        cdef REAL_t [:] d = B.deltas
        cdef double recip_denom
        cdef double fst_deriv
        derivative = B.derivative

        if boundary is None:
            lower_type = upper_type = None
        else:
            try:
                (lower_type, lower_val), (upper_type, upper_val) = boundary
            except TypeError:
                raise TypeError("boundary must be a 2-tuple of 2-tuples or"
                                " None. See FiniteDifferenceEngine.")
            except ValueError:
                raise ValueError("boundary must be a 2-tuple of 2-tuples or"
                                " None. See FiniteDifferenceEngine.")

        # Doing lower boundary
        if lower_type == 0:
            # Dirichlet boundary. No derivatives, but we need to preserve the
            # value we get, because we will have already forced it.
            Bdata[m, 0] = 1
            B.dirichlet[0] = lower_val
            pass
        elif lower_type == 1:
            # Von Neumann boundary, we specify it directly.
            R[0] = lower_val
        elif lower_type is None and derivative == 1:
            # Free boundary
            # Second order forward approximation
            # XXX: This is dangerous! We can't do it if data is not wide enough
            assert m-2 >= 0, ("Not wide enough."
                    "\nB.D.data.shape = %s"
                    "\nB.derivative = %s"
                    "\nB.D.offsets = %s"
                    "\nm = %s"
                    "\nboundary = %s"
                    ) % (B.D.data.shape, B.derivative, B.D.offsets, m, boundary)
            Bdata[m - 2, 2] = -d[1] / (d[2] * (d[1] + d[2]))
            Bdata[m - 1, 1] = (d[1] + d[2]) / (d[1] * d[2])
            Bdata[m,     0] = (-2 * d[1] - d[2]) / (d[1] * (d[1] + d[2]))
            # Bdata[m, 0] = -1.0 / d[1]
            # Bdata[m - 1, 1] = 1.0 / d[1]
            # print Bdata
        elif lower_type is None and derivative == 2:
            # If we know the first derivative, Extrapolate second derivative by
            # assuming the first stays constant.
            if lower_val is not None:
                # print "%s %s Assuming first derivative is %s for second." % (B.axis, B.derivative, lower_val,)
                fst_deriv = lower_val
                assert m-1 >= 0
                Bdata[m-1, 1] =  2 / d[1]**2
                Bdata[m,   0] = -2 / d[1]**2
                R[0]         =  -fst_deriv * 2 / d[1]
            # Otherwise just compute it with forward differencing
            else:
                # print "%s %s Computing second derivative directly." % (B.axis, B.derivative,)
                assert m-2 >= 0
                recip_denom = 1.0 / (0.5*(d[2]+d[1])*d[2]*d[1]);
                Bdata[m-2,2] = d[1]         * recip_denom
                Bdata[m-1,1] = -(d[2]+d[1]) * recip_denom
                Bdata[m,0]   = d[2]         * recip_denom
        else:
            raise NotImplementedError("Can't handle derivatives higher than"
                                      " order 2 at boundaries. (%s)" % derivative)

        # Doing upper boundary
        if upper_type == 0:
            # Dirichlet boundary. No derivatives, but we need to preserve the
            # value we get, because we will have already forced it.
            Bdata[m, -1] = 1
            B.dirichlet[1] = upper_val
            pass
        elif upper_type == 1:
            # Von Neumann boundary, we specify it directly.
            R[-1] = upper_val
        elif upper_type is None and derivative == 1:
            # Second order backward approximation
            assert m+2 < B.D.data.shape[0]
            # XXX: This is dangerous! We can't do it if data is not wide enough
            Bdata[m  , -1] = (d[-2]+2*d[-1])  / (d[-1]*(d[-2]+d[-1]))
            Bdata[m+1, -2] = (-d[-2] - d[-1]) / (d[-2]*d[-1])
            Bdata[m+2, -3] = d[-1]             / (d[-2]*(d[-2]+d[-1]))
            # First order backward
            # Bdata[m, -1] = 1.0 / d[-1]
            # Bdata[m + 1, -2] = -1.0 / d[-1]
        elif upper_type is None and derivative == 2:
            # if B.R is None:
                # R = np.zeros(B.D.data.shape[1])
            # If we know the first derivative, Extrapolate second derivative by
            # assuming the first stays constant.
            if upper_val is not None:
                fst_deriv = upper_val
                assert m+1 < B.D.data.shape[0]
                Bdata[m+1, -2] =  2 / d[-1]**2
                Bdata[m,   -1] = -2 / d[-1]**2
                R[-1]          =  fst_deriv * 2 / d[-1]
            # Otherwise just compute it with backward differencing
            else:
                assert m-2 >= 0
                recip_denom = 1.0 / (0.5*(d[-2]+d[-1])*d[-2]*d[-1]);
                Bdata[m+2,-3] = d[-1]         * recip_denom
                Bdata[m+1,-2] = -(d[-2]+d[-1]) * recip_denom
                Bdata[m,-1]   = d[-2]         * recip_denom
        else:
            raise NotImplementedError("Can't handle derivatives higher than"
                                      " order 2 at boundaries. (%s)" % derivative)
        B.R = np.asarray(R)

        # if upper_type == 1 or upper_type is None:
            # print "Derivative:", derivative
            # print "boundary:", boundary
            # print "R:", B.R



    @staticmethod
    def check_derivative(d):
        mixed = False
        try:
            d = tuple(d)
            if len(d) > 2:
                raise NotImplementedError, "Can't do more than 2nd order derivatives."
            if len(set(d)) != 1:
                mixed = True
            map(int, d)
            d = len(d)
        except TypeError:
            try:
                d = int(d)
            except TypeError:
                raise TypeError("derivative must be a number or an iterable of numbers")
        if d > 2 or d < 1:
            raise NotImplementedError, "Can't do 0th order or more than 2nd order derivatives."
        return mixed


    @staticmethod
    def check_order(order):
        if order != 2:
            raise NotImplementedError, ("Order must be 2")


    # def __getattr__(self, name):
        # return self.D.__getattribute__(name)


    # @property
    # def deltas(self):
        # return self._deltas


    @classmethod
    def forwardcoeffs(cls, deltas, derivative=1, order=2, force_bandwidth=None):
        d = deltas

        cls.check_derivative(derivative)
        cls.check_order(order)

        if derivative == 1:
            if force_bandwidth is not None:
                l, u = [int(o) for o in force_bandwidth]
                # print "High and low", u, l
                offsets = range(u, l-1, -1)
            else:
                if order == 2:
                    offsets = [2, 1,0,-1]
                else:
                    raise NotImplementedError
            data = np.zeros((len(offsets),len(d)))
            m = offsets.index(0)
            # print "OFFSETS from forward 1:", m, offsets
            assert m-2 >= 0
            assert m < data.shape[0]
            for i in range(1,len(d)-2):
                data[m-1,i+1] = (d[i+1] + d[i+2])  /         (d[i+1]*d[i+2])
                data[m-2,i+2] = -d[i+1]            / (d[i+2]*(d[i+1]+d[i+2]))
                data[m,i]     = (-2*d[i+1]-d[i+2]) / (d[i+1]*(d[i+1]+d[i+2]))
                # data[m-1,i+1] = i
                # data[m-2,i+2] = i
                # data[m,i]     = i
            # Use centered approximation for the last (inner) row.
            data[m-1,-1] =           d[-2]  / (d[-1]*(d[-2]+d[-1]))
            data[m,  -2] = (-d[-2] + d[-1]) /        (d[-2]*d[-1])
            data[m+1,-3] =          -d[-1]  / (d[-2]*(d[-2]+d[-1]))

            # print "DATA from forward"
            # print data
            # print

        elif derivative == 2:
            if force_bandwidth is not None:
                l, u = [int(o) for o in force_bandwidth]
                offsets = range(u, l-1, -1)
                # print "High and low", u, l
            else:
                if order == 2:
                    offsets = [2, 1, 0,-1]
                else:
                    raise NotImplementedError
            data = np.zeros((len(offsets),len(d)))
            m = offsets.index(0)
            # print "OFFSETS from forward 2:", m, offsets
            for i in range(1,len(d)-2):
                denom = (0.5*(d[i+2]+d[i+1])*d[i+2]*d[i+1]);
                data[m-2,i+2] =   d[i+1]         / denom
                data[m-1,i+1] = -(d[i+2]+d[i+1]) / denom
                data[m,i]     =   d[i+2]         / denom
            # Use centered approximation for the last (inner) row
            data[m-1,-1] = 2  / (d[-1]*(d[-2]+d[-1]))
            data[m  ,-2] = -2 /       (d[-2]*d[-1])
            data[m+1,-3] = 2  / (d[-2  ]*(d[-2]+d[-1]))
        else:
            raise NotImplementedError, ("Order must be 1 or 2")
        return (data, offsets)



    @classmethod
    def centercoeffs(cls, deltas, derivative=1, order=2, force_bandwidth=None):
        """Centered differencing coefficients."""
        d = deltas

        cls.check_derivative(derivative)
        cls.check_order(order)

        if derivative == 1:
            if force_bandwidth is not None:
                l, u = [int(o) for o in force_bandwidth]
                # print "High and low", u, l
                offsets = range(u, l-1, -1)
            else:
                if order == 2:
                    # TODO: Be careful here, why is this 10-1?
                    offsets = [1,0,-1]
                else:
                    raise NotImplementedError
            data = np.zeros((len(offsets),len(d)))
            m = offsets.index(0)
            # print "OFFSETS from center 1:", m, offsets
            assert m-1 >= 0
            assert m+1 < data.shape[0]
            for i in range(1,len(d)-1):
                data[m-1,i+1] =            d[i]  / (d[i+1]*(d[i]+d[i+1]))
                data[m  ,i  ] = (-d[i] + d[i+1]) /         (d[i]*d[i+1])
                data[m+1,i-1] =         -d[i+1]  / (d[i  ]*(d[i]+d[i+1]))
        elif derivative == 2:
            if force_bandwidth is not None:
                l, u = [int(o) for o in force_bandwidth]
                offsets = range(u, l-1, -1)
            else:
                if order == 2:
                    # TODO: Be careful here, why is this 10-1?
                    offsets = [1,0,-1]
                else:
                    raise NotImplementedError
            data = np.zeros((len(offsets),len(d)))
            m = offsets.index(0)
            # print "OFFSETS from center 2:", m, offsets
            # Inner rows
            for i in range(1,len(d)-1):
                data[m-1,i+1] =  2 / (d[i+1]*(d[i]+d[i+1]))
                data[m  ,i  ] = -2 /       (d[i]*d[i+1])
                data[m+1,i-1] =  2 / (d[i  ]*(d[i]+d[i+1]))
        else:
            raise NotImplementedError("Derivative must be 1 or 2")

        return (data, offsets)


    @classmethod
    def backwardcoeffs(cls, deltas, derivative=1, order=2, force_bandwidth=None):
        d = deltas

        cls.check_derivative(derivative)
        cls.check_order(order)

        if derivative == 1:
            if force_bandwidth is not None:
                l, u = [int(o) for o in force_bandwidth]
                offsets = range(u, l-1, -1)
            else:
                if order == 2:
                    offsets = [1,0,-1,-2]
                else:
                    raise NotImplementedError
            data = np.zeros((len(offsets),len(d)))
            m = offsets.index(0)
            for i in range(2,len(d)-1):
                data[m, i]     = (d[i-1]+2*d[i])  / (d[i]*(d[i-1]+d[i]));
                data[m+1, i-1] = (-d[i-1] - d[i]) / (d[i-1]*d[i]);
                data[m+2, i-2] = d[i]             / (d[i-1]*(d[i-1]+d[i]));
            # Use centered approximation for the first (inner) row.
            data[m-1,2] =          d[1]  / (d[2]*(d[1]+d[2]))
            data[m,  1] = (-d[1] + d[2]) /       (d[1]*d[2])
            data[m+1,0] =         -d[2]  / (d[1]*(d[1]+d[2]))
        elif derivative == 2:
            if force_bandwidth is not None:
                l, u = [int(o) for o in force_bandwidth]
                offsets = range(u, l-1, -1)
            else:
                if order == 2:
                    offsets = [1,0,-1,-2]
                else:
                    raise NotImplementedError
            data = np.zeros((len(offsets),len(d)))
            m = offsets.index(0)
            for i in range(2,len(d)-1):
                denom = (0.5*(d[i]+d[i-1])*d[i]*d[i-1]);
                data[m,     i] =   d[i-1]       / denom;
                data[m+1, i-1] = -(d[i]+d[i-1]) / denom;
                data[m+2, i-2] =   d[i]         / denom;
            # Use centered approximation for the first (inner) row
            data[m+1,0] =  2 / (d[1  ]*(d[1]+d[2]))
            data[m,1]   = -2 /       (d[1]*d[2])
            data[m-1,2] =  2 / (d[2]*(d[1]+d[2]))
        else:
            raise NotImplementedError, ("Derivative must be 1 or 2")

        return (data, offsets)


    def splice_with(self, begin, at, inplace=False):
        """
        Splice a second operator into this one by replacing rows after @at@.
        If inplace is True, splice it directly into this object.
        """
        newoffsets = sorted(set(self.D.offsets).union(set(begin.D.offsets)), reverse=True)

        if inplace:
            if tuple(newoffsets) != tuple(self.D.offsets):
                raise ValueError("Operators have different offsets, cannot"
                        " splice inplace.")

        if at < 0:
            at = self.shape[0] + at

        # Handle the two extremes
        if at == self.shape[0]-1:
            if inplace:
                B = self
            else:
                B = self.copy()
        elif at == 0:
            if inplace:
                B = self
                B.D = begin.D.copy()
                B.R = begin.R.copy()
                B.copy_meta_data(begin)
            else:
                B = begin.copy()

        # If it's not extreme, it must be a normal splice
        else:
            if inplace:
                B = self
            else:
                newdata = np.zeros((len(newoffsets), self.D.data.shape[1]))
                B = BandedOperator((newdata, newoffsets), residual=self.R)

            last = B.shape[1]
            for torow, o in enumerate(B.D.offsets):
                splitidx = max(min(at+o, last), 0)
                if o in self.D.offsets:
                    fromrow = list(self.D.offsets).index(o)
                    dat = self.D.data[fromrow, :splitidx]
                else:
                    dat = 0
                B.D.data[torow, :splitidx] = dat
                if o in begin.D.offsets:
                    fromrow = list(begin.D.offsets).index(o)
                    dat = begin.D.data[fromrow, splitidx:last]
                else:
                    dat = 0
                B.D.data[torow, splitidx:last] = dat

            # handle the residual vector
            if B.R is not None:
                B.R[splitidx:last] = begin.R[splitidx:last]
            else:
                B.R = begin.R.copy()
                B.R[:splitidx] = 0

            B.copy_meta_data(self)
            B.dirichlet[0] = self.dirichlet[0]
            B.dirichlet[1] = begin.dirichlet[1]
        return B


    def __mul__(self, val):
        return self.mul(val, inplace=False)
    def __imul__(self, val):
        return self.mul(val, inplace=True)

    def mul(self, val, inplace=False):
        if inplace:
            B = self
        else:
            B = self.copy()

        B.vectorized_scale(np.ones(B.shape[0]) * val)
        # block_len = B.shape[0] / float(B.blocks)
        # assert block_len == int(block_len)
        # for i in range(B.blocks):
            # end = i*block_len
            # if B.dirichlet[0] is not None:
                # end += 1
            # begin = i*block_len + block_len
            # if B.dirichlet[1] is not None:
                # begin -= 1
            # B.D.data[m,end:begin] *= val
            # B.R[end:begin] *= val

        # if B.dirichlet[0] is None:
            # B.D.data[0] *= val
        # if B.dirichlet[1] is None:
            # B.D.data[-1] *= val
            # B.R[-1] *= val
        # B.D.data[1:-1] *= val
        # B.R[1:-1] *= val
        return B


    def __add__(self, other):
        return self.add(other, inplace=False)
    def __iadd__(self, other):
        return self.add(other, inplace=True)

    def add(self, other, cbool inplace=False):
        if isinstance(other, BandedOperator):
            return self.add_operator(other, inplace)
        else:
            return self.add_scalar(other, inplace)


    # TODO: This needs to be faster
    def add_operator(BandedOperator self, BandedOperator other, cbool inplace=False):
        """
        Add a second BandedOperator to this one.
        Does not alter self.R, the residual vector.
        """
        cdef REAL_t[:,:] data = self.D.data
        cdef int[:] selfoffsets = np.array(self.D.offsets)
        cdef int[:] otheroffsets = np.array(other.D.offsets)
        cdef unsigned int num_otheroffsets = otheroffsets.shape[0]
        cdef np.ndarray[REAL_t, ndim=2] newdata
        cdef int[:] Boffsets
        cdef int o
        cdef unsigned int i
        cdef BandedOperator B
        cdef cbool fail

        if self.axis != other.axis:
            raise ValueError("Both operators must operate on the same axis."
                    " (%s != %s)" % (self.axis, other.axis))
        # Verify that they are compatible
        if self.shape[1] != other.shape[1]:
            raise ValueError("Both operators must have the same length")
        # If we're adding it directly to this one
        if inplace:
            # The diagonals have to line up.
            fail = False
            if selfoffsets.shape[0] != otheroffsets.shape[0]:
                fail = True
            for i in range(num_otheroffsets):
                if otheroffsets[i] != selfoffsets[i]:
                    fail = True
            if fail:
                print "Self offsets:", self.D.offsets
                print "Them offsets:", other.D.offsets
                raise ValueError("Both operators must have (exactly) the"
                                    " same offsets to add in-place.")
            B = self
            Boffsets = selfoffsets
        # Otherwise we are adding directly to this one.
        else:
            # Calculate the offsets that the new one will have.
            Boffsets = np.array(sorted(set(selfoffsets).union(set(otheroffsets)),
                                reverse=True), dtype=np.int32)
            newdata = np.zeros((Boffsets.shape[0], self.shape[1]))
            # And make a new operator with the appropriate shape
            # Remember to carry the residual with us.
            B = BandedOperator((newdata, Boffsets), self.R)
            # Copy our data into the new operator since carefully, since we
            # may have resized.
            for i in range(selfoffsets.shape[0]):
                o = selfoffsets[i]
                fro = get_int_index(selfoffsets, o)
                to = get_int_index(Boffsets, o)
                # print "fro(%i) -> to(%i)" % (fro, to)
                B.D.data[to] += data[fro]
            B.copy_meta_data(self)
        # Copy the data from the other operator over
        # Don't double the dirichlet boundaries!
        for i in range(num_otheroffsets):
            fro = i
            o = otheroffsets[i]
            to = get_int_index(Boffsets, o)
            if o == 0:
                # We have to do the main diagonal block_wise becaues of the
                # dirichlet boundary
                block_len = B.shape[0] / float(B.blocks)
                assert block_len == int(block_len)
                for i in range(B.blocks):
                    end = i*block_len
                    if B.dirichlet[0] is not None:
                        end += 1
                    begin = i*block_len + block_len
                    if B.dirichlet[1] is not None:
                        begin -= 1
                    B.D.data[to,end:begin] += other.D.data[fro,end:begin]
            else:
                end = 0
                begin = B.D.data.shape[1]
                B.D.data[to,end:begin] += other.D.data[fro,end:begin]
        # Now the residual vector from the other one
        if other.R is not None:
            if B.R is None:
                B.R = other.R.copy()
            else:
                B.R += other.R

        return B


    def add_scalar(self, float other, cbool inplace=False):
        """
        Add a scalar to the main diagonal or
        Does not alter self.R, the residual vector.
        """
        if inplace:
            B = self
        else:
            B = self.copy()
        # We add it to the main diagonal.
        cdef np.ndarray[int, ndim=1] selfoffsets = np.array(self.D.offsets)
        cdef unsigned int m = get_int_index(selfoffsets, 0)
        cdef unsigned int blocks = B.blocks
        cdef unsigned int block_len

        if m > len(selfoffsets)-1:
            raise NotImplementedError("Cannot (yet) add scalar to operator"
                                        " without main diagonal.")
        block_len = B.shape[0] / blocks
        # assert block_len == int(block_len)
        data = B.D.data
        for i in range(blocks):
            end = i*block_len
            if B.dirichlet[0] is not None:
                end += 1
            begin = i*block_len + block_len
            if B.dirichlet[1] is not None:
                begin -= 1
            data[m,end:begin] += other

        # Don't touch the residual.
        return B


    @cython.boundscheck(False)
    def vectorized_scale(self, REAL_t[:] vector):
        """
        @vector@ is the correpsonding mesh vector of the current dimension.

        Also applies to the residual vector self.R.

        See FiniteDifferenceEngine.coefficients.
        """
        cdef unsigned int operator_rows = self.shape[0]
        cdef unsigned int blocks = self.blocks
        cdef unsigned int block_len = operator_rows / blocks
        # cdef np.ndarray[REAL_t, ndim=1] vec
        cdef int [:] offsets = np.array(self.D.offsets)
        cdef REAL_t [:] R = self.R
        cdef REAL_t[:,:] data = self.D.data
        cdef unsigned int noffsets = len(self.D.offsets)
        cdef signed int o
        cdef unsigned int i, j, begin, end, vbegin
        if blocks > 1 and vector.shape[0] == block_len:
            vector = np.tile(vector, self.blocks)
        assert vector.shape[0] == operator_rows
        cdef cbool low_dirichlet = self.dirichlet[0] is not None
        cdef cbool high_dirichlet = self.dirichlet[1] is not None
        for row in range(noffsets):
            o = offsets[row]
            while block_len / 2 < o:
                o -= block_len
            while o < block_len / -2:
                o += block_len
            vbegin = begin = 0
            vend = end = block_len
            if o >= 0:
                begin = o + low_dirichlet
                vbegin = low_dirichlet
                vend -= o
            if o <= 0:
                end += o - high_dirichlet
                vbegin -= o
                vend -= high_dirichlet
            # data[row].reshape(blocks, block_len).T[begin:end, :] *= vector.reshape(blocks,block_len).T[vbegin:vend, :]
            vbegin = vbegin - begin
            for i in range(blocks):
                for j in range(begin, end):
                    data[row, j] *= vector[j+vbegin]
                # data[row, begin: end] *= vector[vbegin:vend]
                begin += block_len
                end += block_len
                # vbegin += block_len
                # vend += block_len

        if low_dirichlet:
            begin = 1
        else:
            begin = 0
        if high_dirichlet:
            end = block_len - 1
        else:
            end = block_len
        for i in range(blocks):
            for j in range(begin, end):
                R[j] *= vector[j]
            begin += block_len
            end += block_len
        # self.R.reshape(blocks, block_len)[:,begin:end] *= vector.reshape(blocks, block_len)[:, begin:end]
        return


    def scale(self, func):
        """
        func must be compatible with the following:
            func(x)
        Where x is the correpsonding index of the current dimension.

        Also applies to the residual vector self.R.

        See FiniteDifferenceEngine.coefficients.
        """
        block_len = self.shape[0] / float(self.blocks)
        assert block_len == int(block_len)
        block_len = int(block_len)
        for i in range(self.blocks):
            for row, o in enumerate(self.D.offsets):
                begin = i*block_len
                end = i*block_len + block_len
                # if o == 0:
                if o >= 0 and self.dirichlet[0] is not None:
                    begin += 1
                if o <= 0 and self.dirichlet[1] is not None:
                    end -= 1
                if o > 0:
                    begin += o
                elif o < 0:
                    end += o
                # print "offset %s, %s to %s" % (o, begin, end)
                for k in range(begin, end):
                    # print "i =", k-begin
                    self.D.data[row, k] *= func(k-o)
            begin = i*block_len
            end = i*block_len + block_len
            if self.dirichlet[0] is not None:
                begin += 1
            if self.dirichlet[1] is not None:
                end -= 1
            for k in range(begin,end):
                self.R[k] *= func(k)
            # if o > 0:
                # for i in range(begin, self.shape[0]-o):
                    # self.D.data[row,i+o] *= func(i)
            # elif o == 0:
                # for i in range(begin, end):
                    # self.D.data[row,i] *= func(i)
            # elif o < 0:
                # for i in range(end):
                    # self.D.data[row, i-abs(o)] *= func(i)


cdef inline int sign(int i):
    if i < 0:
        return -1
    else:
        return 1

@cython.boundscheck(False)
cdef inline unsigned int get_real_index(REAL_t[:] haystack, REAL_t needle):
    cdef unsigned int length = haystack.shape[0]
    for i in range(length):
        if needle == haystack[i]:
            return i
    raise ValueError("Value not in array: %s" % needle)


@cython.boundscheck(False)
cdef inline unsigned int get_int_index(int[:] haystack, int needle):
    cdef unsigned int length = haystack.shape[0]
    for i in range(length):
        if needle == haystack[i]:
            return i
    raise ValueError("Value not in array: %s" % needle)
