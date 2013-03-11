# coding: utf8
# cython: annotate = True
# cython: infer_types = True
# cython: embedsignature = True
# distutils: language = c++
# distutils: sources = FiniteDifference/_GPU_Code.cu FiniteDifference/backtrace.c FiniteDifference/filter.c


from bisect import bisect_left

import numpy as np
cimport numpy as np
import scipy.sparse

cimport cython
from cython.operator import dereference as deref

import FiniteDifference.utils as utils

from FiniteDifference.BandedOperator import BandedOperator as BO
from FiniteDifference.BandedOperator cimport BandedOperator as BO
from FiniteDifference.VecArray cimport GPUVec

FOLDED = "FOLDED"
CAN_FOLD = "CAN_FOLD"
CANNOT_FOLD = "CANNOT_FOLD"


cdef class BandedOperator(object):

    def __init__(self, other=None, tag="Constructor"):
        """
        A linear operator for discrete derivatives.
        Consist of a banded matrix (B.D) and a residual vector (B.R) for things
        like

            U2 = L*U1 + Rj  -->   U2 = B.apply(U1)
            U2 = L.I * (U1 - R) --> U2 = B.solve(U1)
        """
        self.attrs = ('deltas', 'derivative', 'is_mixed_derivative', 'order', 'axis',
                      'blocks', 'top_fold_status', 'bottom_fold_status')

        if other:
            self.emigrate(other, tag)


    property operator_rows:
        def __get__(self):
            if self.is_mixed_derivative:
                return self.thisptr_csr.operator_rows
            else:
                return self.thisptr_tri.operator_rows


    property shape:
        def __get__(self):
            return (self.operator_rows, self.operator_rows)


    def __dealloc__(self):
        if self.thisptr_csr:
            del self.thisptr_csr
        elif self.thisptr_tri:
            del self.thisptr_tri


    def copy_meta_data(self, other, **kwargs):
        for attr in self.attrs:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            elif attr in ('deltas', 'top_factors', 'bottom_factors'):
                setattr(self, attr, getattr(other, attr).copy())
            else:
                setattr(self, attr, getattr(other, attr))


    def __richcmp__(self, other, op):
        raise NotImplementedError


    cpdef copy(self):
        cdef BandedOperator B = BandedOperator()
        if self.is_mixed_derivative:
            B.thisptr_csr = new _CSRBandedOperator(
                  self.thisptr_csr.data
                , self.thisptr_csr.row_ptr
                , self.thisptr_csr.row_ind
                , self.thisptr_csr.col_ind
                , self.thisptr_csr.operator_rows
                , self.blocks
                , self.thisptr_csr.name
            )
        else:
            B.thisptr_tri = new _TriBandedOperator(
                      self.thisptr_tri.diags
                    , self.thisptr_tri.R
                    , self.thisptr_tri.high_dirichlet
                    , self.thisptr_tri.low_dirichlet
                    , self.thisptr_tri.top_factors
                    , self.thisptr_tri.bottom_factors
                    , self.axis
                    , self.thisptr_tri.operator_rows
                    , self.thisptr_tri.blocks
                    , self.thisptr_tri.has_high_dirichlet
                    , self.thisptr_tri.has_low_dirichlet
                    , self.thisptr_tri.top_fold_status
                    , self.thisptr_tri.bottom_fold_status
                    , self.thisptr_tri.has_residual
                    )
        B.copy_meta_data(self)
        return B


    cpdef emigrate(self, other, tag=""):
        self.copy_meta_data(other)
        if self.is_mixed_derivative:
            return self.emigrate_csr(other, tag)
        else:
            return self.emigrate_tri(other, tag)


    cpdef immigrate(self, tag=""):
        if self.is_mixed_derivative:
            return self.immigrate_csr(tag)
        else:
            return self.immigrate_tri(tag)


    cdef emigrate_csr(self, other, tag=""):
        if tag:
            print "Emigrate CSR:", tag, to_string(self.thisptr_csr)
        assert not (self.thisptr_csr)
        csr = other.D.tocsr()
        coo = csr.tocoo()
        cdef:
            SizedArrayPtr data = SizedArrayPtr(csr.data, "data")
            SizedArrayPtr_i row_ptr = SizedArrayPtr_i(csr.indptr, "row_ptr")
            SizedArrayPtr_i row_ind = SizedArrayPtr_i(coo.row, "row_ind")
            SizedArrayPtr_i col_ind = SizedArrayPtr_i(csr.indices, "col_ind")

        self.thisptr_csr = new _CSRBandedOperator(
                  deref(data.p)
                , deref(row_ptr.p)
                , deref(row_ind.p)
                , deref(col_ind.p)
                , other.D.shape[0]
                , other.blocks
                , tag
                )
        del data, row_ptr, row_ind, col_ind


    cdef immigrate_csr(self, tag=""):
        if tag:
            print "Immigrate CSR:", tag, to_string(self.thisptr_csr)
        assert (self.thisptr_csr)

        data = from_SizedArray(self.thisptr_csr.data)
        indices = from_SizedArray_i(self.thisptr_csr.col_ind)
        indptr = from_SizedArray_i(self.thisptr_csr.row_ptr)

        shp = (self.thisptr_csr.operator_rows, self.thisptr_csr.operator_rows)
        mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=shp).todia()
        mat = utils.todia(mat)

        cdef BO B = BO((mat.data,mat.offsets), residual=None, inplace=False,
            deltas=self.deltas,
            derivative=self.derivative,
            order=self.order,
            axis=self.axis)
        B.is_mixed_derivative = True
        B.blocks = self.blocks
        return B


    cdef emigrate_tri(self, other, tag=""):
        if tag:
            print "Emigrate Tri:", tag, "<- offsets", other.D.offsets
        assert not self.thisptr_tri

        other = other.copy()

        diad = False
        if other.top_fold_status == CAN_FOLD or other.bottom_fold_status == CAN_FOLD:
            diad = True
            other.diagonalize()
            self.top_fold_status = other.top_fold_status
            self.bottom_fold_status = other.bottom_fold_status

        scipy_to_cublas(other)

        tops = np.atleast_1d(other.top_factors if other.top_factors is not None else [0.0]*other.blocks)
        bots = np.atleast_1d(other.bottom_factors if other.bottom_factors is not None else [0.0]*other.blocks)
        cdef:
            SizedArrayPtr diags = SizedArrayPtr(other.D.data, "data")
            SizedArrayPtr R = SizedArrayPtr(other.R, "R")
            SizedArrayPtr low_dirichlet = SizedArrayPtr(np.atleast_1d(other.dirichlet[0] or [0.0]), "low_dirichlet")
            SizedArrayPtr high_dirichlet = SizedArrayPtr(np.atleast_1d(other.dirichlet[1] or [0.0]), "high_dirichlet")
            SizedArrayPtr top_factors = SizedArrayPtr(tops, "top_factors")
            SizedArrayPtr bottom_factors = SizedArrayPtr(bots, "bottom_factors")

        self.thisptr_tri = new _TriBandedOperator(
                  deref(diags.p)
                , deref(R.p)
                , deref(high_dirichlet.p)
                , deref(low_dirichlet.p)
                , deref(top_factors.p)
                , deref(bottom_factors.p)
                , other.axis
                , other.shape[0]
                , other.blocks
                , other.dirichlet[1] is not None
                , other.dirichlet[0] is not None
                , other.top_fold_status
                , other.bottom_fold_status
                , other.R is not None
                )

        del diags, R, high_dirichlet, low_dirichlet, top_factors, bottom_factors

        if diad:
            self.undiagonalize()


    cdef immigrate_tri(self, tag=""):
        cdef BO B

        if tag:
            print "Immigrate Tri:", tag, to_string(self.thisptr_tri)
        assert self.thisptr_tri != <void *>0


        bots = from_SizedArray(self.thisptr_tri.bottom_factors)
        tops = from_SizedArray(self.thisptr_tri.top_factors)

        center = 1
        bottom = 2
        if self.top_fold_status == CAN_FOLD:
            center += 1
            bottom += 1
        if self.bottom_fold_status == CAN_FOLD:
            bottom += 1

        block_len = self.thisptr_tri.block_len

        data = np.zeros((bottom+1, self.operator_rows), dtype=float)
        offsets = -np.arange(bottom+1) + center
        data[center-1:center+2, :] = from_SizedArray(self.thisptr_tri.diags)

        if self.top_fold_status == CAN_FOLD:
            data[0,::block_len] = tops
        if self.bottom_fold_status == CAN_FOLD:
            data[-1,block_len-1::block_len] = bots

        R = from_SizedArray(self.thisptr_tri.R) if self.thisptr_tri.has_residual else None

        B = BO((data, offsets), residual=R, inplace=False,
            deltas=self.deltas,
            derivative=self.derivative,
            order=self.order,
            axis=self.axis)

        h = tuple(from_SizedArray(self.thisptr_tri.high_dirichlet)) if self.thisptr_tri.has_high_dirichlet else None
        l = tuple(from_SizedArray(self.thisptr_tri.low_dirichlet)) if self.thisptr_tri.has_low_dirichlet else None
        B.dirichlet = [l,h]
        B.blocks = self.thisptr_tri.blocks
        B.top_fold_status = self.top_fold_status
        B.bottom_fold_status = self.bottom_fold_status

        cublas_to_scipy(B)

        B.bottom_factors = bots if B.bottom_fold_status == FOLDED else None
        B.top_factors = tops if B.top_fold_status == FOLDED else None

        B.solve_banded_offsets = (abs(min(B.D.offsets)), abs(max(B.D.offsets)))
        return B


    cpdef diagonalize(self):
        self.top_fold_status = FOLDED if self.top_fold_status == CAN_FOLD else CANNOT_FOLD
        self.bottom_fold_status = FOLDED if self.bottom_fold_status == CAN_FOLD else CANNOT_FOLD
        self.thisptr_tri.diagonalize()


    cpdef undiagonalize(self):
        self.top_fold_status = CAN_FOLD if self.top_fold_status == FOLDED else CANNOT_FOLD
        self.bottom_fold_status = CAN_FOLD if self.bottom_fold_status == FOLDED else CANNOT_FOLD
        self.thisptr_tri.undiagonalize()


    cpdef cbool is_folded(self):
        return (self.top_fold_status == FOLDED
                or self.bottom_fold_status == FOLDED)


    # cpdef apply_mul(self, SizedArrayPtr sa_V, double x):
        # self.apply_(sa_V, overwrite=True)


    cpdef apply_(self, SizedArrayPtr sa_V, overwrite=False):
        cdef SizedArrayPtr sa_U

        if overwrite:
            sa_U = sa_V
        else:
            sa_U = sa_V.copy(True)

        # if self.thisptr_tri:
            # self.thisptr_tri.apply(deref(sa_U.p))
        # else:
            # self.thisptr_csr.apply(deref(sa_U.p))
        return sa_U


    cpdef apply(self, np.ndarray V, overwrite=False):
        cdef SizedArrayPtr sa_V = SizedArrayPtr(V, "sa_V apply")
        cdef SizedArrayPtr sa_U
        sa_U = self.apply_(sa_V, overwrite)
        # if not overwrite:
            # sa_U = SizedArrayPtr()
            # sa_U.store(x)
        # else:
            # sa_U = sa_V
        V = sa_U.to_numpy()
        del sa_V, sa_U
        return V


    cpdef solve_(self, SizedArrayPtr sa_V, overwrite=False):
        assert not self.is_mixed_derivative
        cdef SizedArrayPtr sa_U
        if overwrite:
            sa_U = sa_V
        else:
            sa_U = sa_V.copy(True)
        # self.thisptr_tri.solve(deref(sa_U.p))
        return sa_U


    cpdef solve(self, np.ndarray V, overwrite=False):
        cdef SizedArrayPtr sa_V = SizedArrayPtr(V, "sa_V solve")
        cdef SizedArrayPtr sa_U = self.solve_(sa_V, overwrite)
        V = sa_U.to_numpy()
        del sa_V, sa_U
        return V


    cdef inline no_mixed(self):
        if self.is_mixed_derivative:
            raise ValueError("Operation not supported with mixed operator.")


    cpdef fold_top(self, unfold=False):
        self.no_mixed()
        self.thisptr_tri.fold_top(unfold)
        self.top_fold_status == CAN_FOLD if unfold else FOLDED


    cpdef fold_bottom(self, unfold=False):
        self.no_mixed()
        self.thisptr_tri.fold_bottom(unfold)
        self.bottom_fold_status == CAN_FOLD if unfold else FOLDED


    cpdef mul(self, val, inplace=False):
        if inplace:
            self *= val
            return self
        else:
            return self * val
    def __mul__(self, val):
        B = self.copy()
        B.__imul__(val)
        return B
    def __imul__(self, val):
        self.vectorized_scale(np.ones(self.shape[0]) * val)
        return self

    cpdef add(self, val, inplace=False):
        if inplace:
            self += val
            return self
        else:
            return self + val
    def __add__(self, val):
        B = self.copy()
        if isinstance(val, BandedOperator):
            B.add_operator(val)
        else:
            B.add_scalar(val)
        return B
    def __iadd__(self, val):
        if isinstance(val, BandedOperator):
            self.add_operator(val)
        else:
            self.add_scalar(val)
        return self


    cpdef add_operator(BandedOperator self, BandedOperator other):
        """
        Add a second BandedOperator to this one.
        Does not alter self.R, the residual vector.
        """
        # TODO: Move these errors into C++

        # Verify that they are compatible
        if self.is_mixed_derivative:
            raise NotImplementedError("No add to mixed operator")
        elif other.is_mixed_derivative:
            raise NotImplementedError("No add mixed operator to this one")
        elif self.axis != other.axis:
            raise ValueError("Both operators must operate on the same axis."
                    " (%s != %s)" % (self.axis, other.axis))
        elif self.thisptr_tri.operator_rows != other.thisptr_tri.operator_rows:
            raise ValueError("Both operators must have the same length")
        elif self.is_folded():
            raise NotImplementedError("No add to diagonalized operator.")
        elif other.is_folded():
            raise NotImplementedError("No add diagonalized operator to this one.")

        # Copy the data from the other operator over
        self.thisptr_tri.add_operator(deref(other.thisptr_tri))

        if other.top_fold_status == CAN_FOLD:
            self.top_fold_status = CAN_FOLD
        if other.bottom_fold_status == CAN_FOLD:
            self.bottom_fold_status = CAN_FOLD
        return


    cpdef add_scalar(self, float other):
        """
        Add a scalar to the main diagonal
        Does not alter self.R, the residual vector.
        """
        self.thisptr_tri.add_scalar(other)
        return

    cpdef vectorized_scale_(self, SizedArrayPtr vector):
        """
        @vector@ is the correpsonding mesh vector of the current dimension.

        Also applies to the residual vector self.R.

        See FiniteDifferenceEngine.coefficients.
        """
        if self.thisptr_tri:
            self.thisptr_tri.vectorized_scale(deref(vector.p))
        elif self.thisptr_csr:
            self.thisptr_csr.vectorized_scale(deref(vector.p))


    cpdef vectorized_scale(self, np.ndarray vector):
        cdef SizedArrayPtr v = SizedArrayPtr(vector, "Vectorized scale v")
        self.vectorized_scale_(v)
        del v
        return


cdef inline int sign(int i):
    if i < 0:
        return -1
    else:
        return 1


# @cython.boundscheck(False)
cdef inline unsigned int get_real_index(double[:] haystack, double needle):
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


def test_SizedArray_transpose(np.ndarray[ndim=2, dtype=double] v):
    cdef SizedArrayPtr s = SizedArrayPtr(v, "transpose s")
    v[:] = 0
    s.p.transpose(1)
    v = s.to_numpy()
    del s
    return v


def test_SizedArray1_roundtrip(np.ndarray[ndim=1, dtype=double] v):
    cdef SizedArrayPtr s = SizedArrayPtr(v, "Round Trip")
    v[:] = 0
    return s.to_numpy()


def test_SizedArray2_roundtrip(np.ndarray[ndim=2, dtype=double] v):
    cdef SizedArrayPtr s = SizedArrayPtr(v, "Round Trip")
    v[:,:] = 0
    return s.to_numpy()


cdef cublas_to_scipy(B):
    # Shift because of scipy/cublas row configuration
    for row, o in enumerate(B.D.offsets):
        if o > 0:
            B.D.data[row,o:] = B.D.data[row,:-o]
            B.D.data[row,:o] = 0
        if o < 0:
            B.D.data[row,:o] = B.D.data[row,-o:]
            B.D.data[row,o:] = 0


cdef scipy_to_cublas(B):
    # We have to shift the offsets between scipy and cublas
    for row, o in enumerate(B.D.offsets):
        if o > 0:
            B.D.data[row,:-o] = B.D.data[row,o:]
            B.D.data[row,-o:] = 0
        if o < 0:
            B.D.data[row,-o:] = B.D.data[row,:o]
            B.D.data[row,:-o] = 0
