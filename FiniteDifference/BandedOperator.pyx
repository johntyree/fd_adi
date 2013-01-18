# coding: utf8
# cython: annotate = True
# cython: infer_types = True
# cython: profile = True
# distutils: language = c++
# distutils: sources = FiniteDifference/BandedOperatorCPU.cpp
"""Description."""

from bisect import bisect_left

import numpy as np
cimport numpy as np
import scipy.sparse
import itertools
import utils
import scipy.linalg as spl

cimport cython


cdef class BandedOperator(object):

    def __init__(self, data_offsets, residual=None, int inplace=True,
            int derivative=1, int order=2, deltas=None, int axis=0):
        """
        A linear operator for discrete derivatives.
        Consist of a banded matrix (B.D) and a residual vector (B.R) for things
        like

            U2 = L*U1 + Rj  -->   U2 = B.apply(U1)
            U2 = L.I * (U1 - R) --> U2 = B.solve(U1)
        """

        self.attrs = ('derivative', 'order', 'axis', 'deltas', 'dirichlet', 'blocks', 'top_factors', 'bottom_factors')
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
        self.top_factors = None
        self.bottom_factors = None
        self.axis = axis

    def copy_meta_data(self, other, **kwargs):
        for attr in self.attrs:
            if attr not in kwargs:
                setattr(self, attr, getattr(other, attr))
            else:
                setattr(self, attr, kwargs[attr])
        self.dirichlet = list(other.dirichlet)


    def __richcmp__(self, other, op):
        true = op == 2
        false = op == 3

        def no_nan(x):
            return np.array(0) if x is None else np.nan_to_num(x)

        for attr in self.attrs:
            if (   attr == 'deltas'
                or attr == 'top_factors'
                or attr == 'bottom_factors'):
                continue
            if getattr(self, attr) != getattr(other, attr):
                print "%s:" % attr,  getattr(self, attr), getattr(other, attr)
                return false

        if ((self.shape == other.shape)
                and (self.D.offsets == other.D.offsets).all()
                and (no_nan(self.top_factors) == no_nan(other.top_factors)).all()
                and (no_nan(self.bottom_factors) == no_nan(other.bottom_factors)).all()
                and (no_nan(self.deltas) == no_nan(other.deltas)).all()
                and (self.R == other.R).all()
                and (self.D.data == other.D.data).all()):
            return true
        else:
            print "shape", (self.shape == other.shape)
            print "offsets", (self.D.offsets == other.D.offsets).all()
            print "top_fact", (no_nan(self.top_factors) == no_nan(other.top_factors)).all()
            print "bot_fact", (no_nan(self.bottom_factors) == no_nan(other.bottom_factors)).all()
            print "deltas", (no_nan(self.deltas) == no_nan(other.deltas)).all()
            print "R", (self.R == other.R).all()
            print "Data", (self.D.data == other.D.data).all()
            return false


    cpdef copy(self):
        B = BandedOperator((self.D.data, self.D.offsets), residual=self.R, inplace=False)
        B.copy_meta_data(self)
        return B


    cpdef diagonalize(self):
        # This is an ugly heuristic
        top = 0
        bot = len(self.D.offsets)
        if 2 in self.D.offsets:
            self.foldtop()
            top += 1
        if -2 in self.D.offsets:
            self.foldbottom()
            bot -= 1
        self.solve_banded_offsets = (1,1)
        self.D = scipy.sparse.dia_matrix((self.D.data[top:bot],
                                          self.D.offsets[top:bot]),
                                         shape=self.shape)


    cpdef undiagonalize(self):
        data = np.zeros((5, self.shape[0]))
        offsets = np.array((2, 1, 0, -1, -2), dtype=np.int32)
        selfoffsets = self.D.offsets
        for i in range(selfoffsets.shape[0]):
            o = selfoffsets[i]
            fro = get_int_index(selfoffsets, o)
            to = get_int_index(offsets, o)
            data[to] += self.D.data[fro]
        self.D = scipy.sparse.dia_matrix((data, offsets), shape=self.shape)
        if self.top_factors is not None:
            self.foldtop(unfold=True)
            self.top_factors = None
        if self.bottom_factors is not None:
            self.foldbottom(unfold=True)
            self.bottom_factors = None
        self.solve_banded_offsets = (abs(min(offsets)), abs(max(offsets)))


    cpdef foldbottom(self, unfold=False):
        d = self.D.data
        m = get_int_index(self.D.offsets, 0)
        for i in [1, 0,-1, -2]:
            if self.D.offsets[m-i] != i:
                raise ValueError("Operator data is the wrong shape. Requires "
                        "contiguous offsets (1, 0, -1, -2).")
        blocks = self.blocks
        block_len = self.shape[0] // blocks
        if not unfold:
            self.bottom_factors = np.empty(blocks)
        for b in range(blocks):
            cn1 = m-1, (b+1)*block_len - 1
            bn  = m,   (b+1)*block_len - 1
            bn1 = m,   (b+1)*block_len - 1 - 1
            an  = m+1, (b+1)*block_len - 1 - 1
            an1 = m+1, (b+1)*block_len - 1 - 2
            zn  = m+2, (b+1)*block_len - 1 - 2
            # print "Block %i %s, %s: d[zn] = %f, d[an1] = %f" % (b, zn, an1, d[zn], d[an1])
            # print "Second row:", d[an1], d[bn1], d[cn1]
            # print "Bottom row:", d[zn], d[an], d[bn]
            if unfold:
                d[zn] -= d[an1] * self.bottom_factors[b]
                d[an] -= d[bn1] * self.bottom_factors[b]
                d[bn] -= d[cn1] * self.bottom_factors[b]
            else:
                self.bottom_factors[b] = -d[zn] / d[an1] if d[an1] != 0 else 0
                d[zn] += d[an1] * self.bottom_factors[b]
                d[an] += d[bn1] * self.bottom_factors[b]
                d[bn] += d[cn1] * self.bottom_factors[b]
        if unfold:
            self.bottom_factors = None


    cpdef foldtop(self, unfold=False):
        d = self.D.data
        m = get_int_index(self.D.offsets, 0)
        for i in [2, 1, 0,-1]:
            if self.D.offsets[m-i] != i:
                raise ValueError("Operator data is the wrong shape. Requires "
                        "contiguous offsets (2, 1, 0, -1).")
        blocks = self.blocks
        block_len = self.shape[0] // blocks
        if not unfold:
            self.top_factors = np.empty(blocks)
        for b in range(blocks):
            a1 = m+1, b*block_len
            b0 = m,   b*block_len
            b1 = m,   b*block_len + 1
            c0 = m-1, b*block_len + 1
            c1 = m-1, b*block_len + 2
            d0 = m-2, b*block_len + 2
            # print "Block %i %s, %s: d[d0] = %f, d[c1] = %f" % (b, d0, c1, d[d0], d[c1])
            # print "Top row   :", d[b0], d[c0], d[d0]
            # print "Second row:", d[a1], d[b1], d[c1]
            if unfold:
                d[d0] -= d[c1] * self.top_factors[b]
                d[c0] -= d[b1] * self.top_factors[b]
                d[b0] -= d[a1] * self.top_factors[b]
            else:
                self.top_factors[b] = -d[d0] / d[c1] if d[c1] != 0 else 0
                d[d0] += d[c1] * self.top_factors[b]
                d[c0] += d[b1] * self.top_factors[b]
                d[b0] += d[a1] * self.top_factors[b]
        if unfold:
            self.top_factors = None


    cpdef fold_vector(self, REAL_t[:] v, unfold=False):
        cdef int direction, u0, u1, un ,un1
        blocks = self.blocks
        block_len = self.shape[0] // blocks

        for b in range(blocks):
            u0 = b*block_len
            u1 = u0 + 1
            un = (b+1)*block_len - 1
            un1 = un - 1
            # print u0, u1, un1, un
            # print "[%f, %f .. %f, %f]" % (v[u0], v[u1], v[un1], v[un])
            direction = -1 if unfold else 1
            if self.top_factors is not None:
                v[u0] += direction * v[u1]  * self.top_factors[b]
            if self.bottom_factors is not None:
                v[un] += direction * v[un1] * self.bottom_factors[b]
        return np.asarray(v)


    cdef inline cbool _is_folded(self):
        return self.top_factors is not None or self.bottom_factors is not None
    def is_folded(self):
        return self._is_folded()


    cpdef print_array(self, REAL_t[:] arr):
        c_print_array(&arr[0], arr.shape[0])


    cpdef cbool is_tridiagonal(self):
        return (    self.D.offsets[0] == 1
                and self.D.offsets[1] == 0
                and self.D.offsets[2] == -1)


    cpdef apply(self, V, overwrite=False):
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

        if self._is_folded():
            # print "apply folded!"
            ret = self.fold_vector(self.D.dot(V.flat), unfold=True)
        else:
            ret = self.D.dot(V.flat)

        if self.R is not None:
            ret += self.R

        ret = ret.reshape(V.shape)

        t = range(V.ndim)
        utils.rolllist(t, V.ndim-1, self.axis)
        ret = np.transpose(ret, axes=t)

        return ret


    cpdef solve(self, V, overwrite=False):
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
            V0 = V.ravel()

        if self._is_folded():
            # print "solve Folded"
            V0 = self.fold_vector(V0)

        ret = spl.solve_banded(self.solve_banded_offsets,
                self.D.data, V0.flat,
                overwrite_ab=overwrite, overwrite_b=True).reshape(V.shape)

        t = range(V.ndim)
        utils.rolllist(t, V.ndim-1, self.axis)
        ret = np.transpose(ret, axes=t)

        return ret


    # @cython.boundscheck(False)
    cpdef applyboundary(self, boundary, mesh):
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
            # assert m-2 >= 0, ("Not wide enough."
                    # "\nB.D.data.shape = %s"
                    # "\nB.derivative = %s"
                    # "\nB.D.offsets = %s"
                    # "\nm = %s"
                    # "\nboundary = %s"
                    # ) % (B.D.data.shape, B.derivative, B.D.offsets, m, boundary)
            # Bdata[m - 2, 2] = -d[1] / (d[2] * (d[1] + d[2]))
            # Bdata[m - 1, 1] = (d[1] + d[2]) / (d[1] * d[2])
            # Bdata[m,     0] = (-2 * d[1] - d[2]) / (d[1] * (d[1] + d[2]))

            # Try first order to preserve tri-diag
            Bdata[m - 1, 1] =  1 / d[1]
            Bdata[m,     0] = -1 / d[1]
        elif lower_type is None and derivative == 2:
            # If we know the first derivative, Extrapolate second derivative by
            # assuming the first stays constant.
            if lower_val is not None:
                # print "%s %s Assuming first derivative is %s for second." % (B.axis, B.derivative, lower_val,)
                fst_deriv = lower_val
                assert m-1 >= 0, "First derivative tried to reach diag %s" % (m-1)
                Bdata[m-1, 1] =  2 / d[1]**2
                Bdata[m,   0] = -2 / d[1]**2
                R[0]         =  -fst_deriv * 2 / d[1]
            # Otherwise just compute it with forward differencing
            else:
                # print "%s %s Computing second derivative directly." % (B.axis, B.derivative,)
                assert m-2 >= 0, "Second derivative tried to reach diag %s" % (m-2)
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
            # # Second order backward approximation
            # assert m+2 < B.D.data.shape[0]
            # # XXX: This is dangerous! We can't do it if data is not wide enough
            # Bdata[m  , -1] = (d[-2]+2*d[-1])  / (d[-1]*(d[-2]+d[-1]))
            # Bdata[m+1, -2] = (-d[-2] - d[-1]) / (d[-2]*d[-1])
            # Bdata[m+2, -3] = d[-1]             / (d[-2]*(d[-2]+d[-1]))
            # First order backward
            Bdata[m, -1]     =  1.0 / d[-1]
            Bdata[m + 1, -2] = -1.0 / d[-1]
        elif upper_type is None and derivative == 2:
            # if B.R is None:
                # R = np.zeros(B.D.data.shape[1])
            # If we know the first derivative, Extrapolate second derivative by
            # assuming the first stays constant.
            if upper_val is not None:
                fst_deriv = upper_val
                assert m+1 < B.D.data.shape[0], "Second Derivative tried to reach diag %s" % (m+1)
                Bdata[m,   -1] = -2 / d[-1]**2
                Bdata[m+1, -2] =  2 / d[-1]**2
                R[-1]          =  fst_deriv * 2 / d[-1]
            # Otherwise just compute it with backward differencing
            else:
                assert m+2 < B.D.data.shape[0], "Second Derivative tried to reach diag %s" % (m-2)
                recip_denom = 1.0 / (0.5*(d[-2]+d[-1])*d[-2]*d[-1]);
                Bdata[m+2,-3] = d[-1]          * recip_denom
                Bdata[m+1,-2] = -(d[-2]+d[-1]) * recip_denom
                Bdata[m,-1]   = d[-2]          * recip_denom
        else:
            raise NotImplementedError("Can't handle derivatives higher than"
                                      " order 2 at boundaries. (%s)" % derivative)
        B.R = np.asarray(R)

    cpdef splice_with(self, begin, at, inplace=False):
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


    cpdef mul(self, val, inplace=False):
        return self.__imul__(val) if inplace else self.__mul__(val)
    def __mul__(self, val):
        B = self.copy()
        return B.__imul__(val)
    def __imul__(self, val):
        self.vectorized_scale(np.ones(self.shape[0]) * val)
        return self

    cpdef add(self, val, inplace=False):
        return self.__iadd__(val) if inplace else self.__add__(val)
    def __add__(self, val):
        if isinstance(val, BandedOperator):
            return self.add_operator(val, False)
        else:
            return self.add_scalar(val, False)
    def __iadd__(self, val):
        if isinstance(val, BandedOperator):
            return self.add_operator(val, True)
        else:
            return self.add_scalar(val, True)


    # TODO: This needs to be faster
    cpdef add_operator(BandedOperator self, BandedOperator other, cbool inplace=False):
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
        if self._is_folded() or other._is_folded():
            raise ValueError("Cannot add diagonalized operators.")
        # If we're adding it directly to this one
        if inplace:
            # The diagonals have to line up.
            fail = False
            if selfoffsets.shape[0] != otheroffsets.shape[0]:
                fail = True
            else:
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


    cpdef add_scalar(self, float other, cbool inplace=False):
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


    cpdef vectorized_scale(self, REAL_t[:] vector):
        # return self.py_vectorized_scale(vector)
        cdef int[:] npoffsets = self.D.offsets
        cdef pair[Py_ssize_t, int*] offsets
        offsets.first = npoffsets.shape[0]
        offsets.second = &npoffsets[0]
        cdef cbool low_dirichlet = self.dirichlet[0] is not None
        cdef cbool high_dirichlet = self.dirichlet[1] is not None

        cpu_vectorized_scale(
              to_pair(vector)
            , to_pair2D(self.D.data)
            , to_pair(self.R)
            , offsets
            , self.shape[0]
            , self.blocks
            , low_dirichlet
            , high_dirichlet)
        return


    #TODO
    # @cython.boundscheck(False)
    cdef py_vectorized_scale(self, REAL_t[:] vector):
        """
        @vector@ is the correpsonding mesh vector of the current dimension.

        Also applies to the residual vector self.R.

        See FiniteDifferenceEngine.coefficients.
        """
        cdef unsigned int operator_rows = self.shape[0]
        cdef unsigned int blocks = self.blocks
        cdef unsigned int block_len = operator_rows / blocks
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

        if low_dirichlet:
            for b in range(blocks):
                vector[b*block_len] = 1
        if high_dirichlet:
            for b in range(blocks):
                vector[(b+1)*block_len - 1] = 1

        for row in range(noffsets):
            o = offsets[row]
            if o >= 0: # upper diags
                for i in range(operator_rows - o):
                    data[row, i+o] *= vector[i]
            else: # lower diags
                for i in range(-o, operator_rows):
                    data[row, i+o] *= vector[i]

        for i in range(operator_rows):
            R[i] *= vector[i]
        return


    cpdef scale(self, func):
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
                if o >= 0 and self.dirichlet[0] is not None:
                    begin += 1
                if o <= 0 and self.dirichlet[1] is not None:
                    end -= 1
                if o > 0:
                    begin += o
                elif o < 0:
                    end += o
                for k in range(begin, end):
                    self.D.data[row, k] *= func(k-o)
            begin = i*block_len
            end = i*block_len + block_len
            if self.dirichlet[0] is not None:
                begin += 1
            if self.dirichlet[1] is not None:
                end -= 1
            for k in range(begin,end):
                self.R[k] *= func(k)


cdef inline int sign(int i):
    if i < 0:
        return -1
    else:
        return 1

# @cython.boundscheck(False)
cdef inline unsigned int get_real_index(REAL_t[:] haystack, REAL_t needle):
    cdef unsigned int length = haystack.shape[0]
    for i in range(length):
        if needle == haystack[i]:
            return i
    raise ValueError("Value not in array: %s" % needle)


# @cython.boundscheck(False)
cdef inline unsigned int get_int_index(int[:] haystack, int needle):
    cdef unsigned int length = haystack.shape[0]
    for i in range(length):
        if needle == haystack[i]:
            return i
    raise ValueError("Value not in array: %s" % needle)


cpdef check_derivative(d):
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
        raise NotImplementedError, "Can't do 0th order or more than 2nd order derivatives. (%s)" % d
    return mixed


cpdef check_order(order):
    if order != 2:
        raise NotImplementedError, ("Order must be 2")


cpdef for_vector(vector, scheme="center", derivative=1, order=2,
        residual=None, force_bandwidth=None, axis=0):
    """
    A linear operator for discrete derivative of @vector@.

    @derivative@ is a tuple specifying the sequence of derivatives. For
    example, `(0,0)` is the second derivative in the first dimension.
    """

    check_derivative(derivative)

    deltas = np.hstack((np.nan, np.diff(vector)))
    scheme = scheme.lower()

    bw = force_bandwidth
    if scheme.startswith("forward") or scheme.startswith('up'):
        data, offsets = forwardcoeffs(deltas, derivative=derivative, order=order, force_bandwidth=bw)
    elif scheme.startswith("backward") or scheme.startswith('down'):
        data, offsets = backwardcoeffs(deltas, derivative=derivative, order=order, force_bandwidth=bw)
    elif scheme.startswith("center") or scheme == "":
        data, offsets = centercoeffs(deltas, derivative=derivative, order=order, force_bandwidth=bw)
    else:
        raise ValueError("Unknown scheme: %s" % scheme)

    B = BandedOperator((data, offsets), residual=residual, axis=axis)
    B.derivative = derivative
    B.order = order
    B.deltas = deltas
    B.axis = axis
    return B


cpdef forwardcoeffs(deltas, derivative=1, order=2, force_bandwidth=None):
    d = deltas

    check_derivative(derivative)
    check_order(order)

    if derivative == 1:
        if force_bandwidth is not None:
            l, u = [int(o) for o in force_bandwidth]
            offsets = range(u, l-1, -1)
        else:
            if order == 2:
                offsets = [2, 1,0,-1]
            else:
                raise NotImplementedError
        data = np.zeros((len(offsets),len(d)))
        m = offsets.index(0)
        assert m-2 >= 0
        assert m < data.shape[0]
        for i in range(1,len(d)-2):
            data[m-1,i+1] = (d[i+1] + d[i+2])  /         (d[i+1]*d[i+2])
            data[m-2,i+2] = -d[i+1]            / (d[i+2]*(d[i+1]+d[i+2]))
            data[m,i]     = (-2*d[i+1]-d[i+2]) / (d[i+1]*(d[i+1]+d[i+2]))
        # Use centered approximation for the last (inner) row.
        data[m-1,-1] =           d[-2]  / (d[-1]*(d[-2]+d[-1]))
        data[m,  -2] = (-d[-2] + d[-1]) /        (d[-2]*d[-1])
        data[m+1,-3] =          -d[-1]  / (d[-2]*(d[-2]+d[-1]))

    elif derivative == 2:
        if force_bandwidth is not None:
            l, u = [int(o) for o in force_bandwidth]
            offsets = range(u, l-1, -1)
        else:
            if order == 2:
                offsets = [2, 1, 0,-1]
            else:
                raise NotImplementedError
        data = np.zeros((len(offsets),len(d)))
        m = offsets.index(0)
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


cpdef centercoeffs(deltas, derivative=1, order=2, force_bandwidth=None):
    """Centered differencing coefficients."""
    d = deltas

    check_derivative(derivative)
    check_order(order)

    if derivative == 1:
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
        # Inner rows
        for i in range(1,len(d)-1):
            data[m-1,i+1] =  2 / (d[i+1]*(d[i]+d[i+1]))
            data[m  ,i  ] = -2 /       (d[i]*d[i+1])
            data[m+1,i-1] =  2 / (d[i  ]*(d[i]+d[i+1]))
    else:
        raise NotImplementedError("Derivative must be 1 or 2")

    return (data, offsets)


cpdef backwardcoeffs(deltas, derivative=1, order=2, force_bandwidth=None):
    d = deltas

    check_derivative(derivative)
    check_order(order)

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


cdef inline pair[Py_ssize_t, double *] to_pair2D(REAL_t [:,:] v):
    cdef pair[Py_ssize_t, double *] p
    p.first = v.shape[0]
    p.second = &v[0,0]
    return p

cdef inline pair[Py_ssize_t, double *] to_pair(REAL_t [:] v):
    cdef pair[Py_ssize_t, double *] p
    p.first = v.shape[0]
    p.second = &v[0]
    return p


