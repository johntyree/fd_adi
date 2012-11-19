#!/usr/bin/env python
# coding: utf8
"""<+Module Description.+>"""

# import sys
# import os
# import itertools as it

from bisect import bisect_left

import numpy as np
import scipy.sparse



class FiniteDifferenceEngine(object):
    def __init__(self, grid, coefficients=[]):
        """
        Coefficients is a list of tuples of functions with c[i][j] referring to the
        coefficient of the j'th derivative in the i'th dimension. It
        counts missing higher order derivatives as 0.

        The functions MUST be able to handle dims+1 arguments, with the first
        being time and the rest corresponding to the dimensions given by @grid.shape@.

        Ex. (2D grid)
            [ (lambda t, x, x2: 0.5, lambda t, x, x2: x),
              # python magic lets be more general than (2*x2*t)
              (lambda t, *dims: 2*dims[1]*t) ]

        is interpreted as:
            0.5*(dU/dx1) + x1*(d²U/dx1²) + 2*x2*t*(dU/dx2) + 0*(d²U/dx2²)
        """
        self.grid = grid
        self.coefficients = coefficients

    def solve(self):
        """Run all the way to the terminal condition."""
        raise NotImplementedError


class BandedOperator(object):
    def __init__(self, (data, offsets)):
        """
        A linear operator for discrete derivatives. Based on the Scipy
        DIAmatrix object.
        """
        size = data.shape[1]
        shape = (size, size)
        self.D = scipy.sparse.dia_matrix((data, offsets), shape=shape, dtype=float)


    @classmethod
    def from_vector(cls, vector, scheme="center", derivative=1, order=1):

        deltas = np.hstack((np.nan, np.diff(vector)))

        if scheme.lower().startswith("forward"):
            data, offsets = cls.forwardcoeffs(deltas, derivative=1, order=order)
        elif scheme.lower().startswith("center"):
            data, offsets = cls.centercoeffs(deltas, derivative=1, order=order)
        elif scheme.lower().startswith("backward"):
            data, offsets = cls.backwardcoeffs(deltas, derivative=1, order=order)

        self = BandedOperator((data, offsets))
        # self._deltas = deltas
        return self


    def __getattr__(self, name):
        return self.D.__getattribute__(name)


    # @property
    # def deltas(self):
        # return self._deltas

    @staticmethod
    def forwardcoeffs(deltas, derivative=1, order=2):
        d = deltas
        data = np.zeros((3,len(d)))

        if order != 2:
            raise NotImplementedError, ("Order must be 2")

        if derivative == 1:
            offsets = [0,1,2]
            for i in range(1,len(d)-2):
                data[0,i]   = (-2*d[i+1]-d[i+2]) / (d[i+1]*(d[i+1]+d[i+2]))
                data[1,i+1] = (d[i+1] + d[i+2])  /         (d[i+1]*d[i+2])
                data[2,i+2] = -d[i+1]           / (d[i+2]*(d[i+1]+d[i+2]))
            # Use first order approximation for the last (inner) row
            data[0, -2] = -1 / d[-1]
            data[1, -1] =  1 / d[-1]
        elif derivative == 2:
            offsets = [0,1,2]
            for i in range(1,len(d)-2):
                denom = (0.5*(d[i+2]+d[i+1])*d[i+2]*d[i+1]);
                data[0,i]   =   d[i+2]         / denom
                data[1,i+1] = -(d[i+2]+d[i+1]) / denom
                data[2,i+2] =   d[i+1]         / denom
        else:
            raise NotImplementedError, ("Order must be 1 or 2")

        return data, offsets

    @staticmethod
    def centercoeffs(deltas, derivative=1, order=2):
        """Centered differencing coefficients."""
        d = deltas
        data = np.zeros((3,len(d)))

        if order != 2:
            raise NotImplementedError, ("Order must be 2")

        if derivative == 1:
            offsets = [-1,0,1]
            for i in range(1,len(d)-1):
                data[0,i-1] =         -d[i+1]  / (d[i  ]*(d[i]+d[i+1]))
                data[1,i]   = (-d[i] + d[i+1]) /         (d[i]*d[i+1])
                data[2,i+1] =            d[i]  / (d[i+1]*(d[i]+d[i+1]))
        elif derivative == 2:
            offsets = [-1,0,1]
            for i in range(1,len(d)-1):
                data[0,i-1] = 2  / (d[i  ]*(d[i]+d[i+1]))
                data[1,i]   = -2 /       (d[i]*d[i+1])
                data[2,i+1] = 2  / (d[i+1]*(d[i]+d[i+1]))
        else:
            raise NotImplementedError, ("Derivative must be 1 or 2")

        return data, offsets


    @staticmethod
    def backwardcoeffs(deltas, derivative=1, order=2):
        d = deltas
        data = np.zeros((3,len(d)))

        if order != 2:
            raise NotImplementedError, ("Order must be 2")

        if derivative == 1:
            offsets = [-2, 1, 0]
            for i in range(1,len(d)-2):
                data[0, i-2] = d[i]             / (d[i-1]*(d[i-1]+d[i]));
                data[1, i-1] = (-d[i-1] - d[i]) / (d[i-1]*d[i]);
                data[2, i]   = (d[i-1]+2*d[i])  / (d[i]*(d[i-1]+d[i]));
            # Use first order approximation for the first (inner) row
                data[1, 1] =  1 / d[-1]
                data[2, 0] = -1 / d[-1]
        elif derivative == 2:
            offsets = [-2, 1, 0]
            for i in range(1,len(d)-2):
                denom = (0.5*(d[i]+d[i-1])*d[i]*d[i-1]);
                data[0, i-2] = d[i] / denom;
                data[1, i-1] = -(d[i]+d[i-1]) / denom;
                data[2, i]   = d[i-1] / denom;
        else:
            raise NotImplementedError, ("Derivative must be 1 or 2")

        return data, offsets

    def splice_with(self, bottom, at, overwrite=False):
        """
        Splice a second operator into this one by replacing rows after @at@.
        If overwrite is True, split it in place.
        """
        newoffsets = sorted(set(self.offsets).union(set(bottom.offsets)))
        newdata = np.zeros((len(newoffsets), self.shape[1]))

        # Copy the self part
        # print offsets
        # print newoffsets
        for torow, o in enumerate(newoffsets):
            if at + o < 0:
                raise ValueError,("You are using forward or backward derivatives "
                                "too close to the edge of the vector. "
                                "(at = %i, row offset = %i)" % (at, o))
            if o in self.offsets:
                fromrow = bisect_left(self.offsets, o)
                newdata[torow,:at+o] = self.data[fromrow, :at+o]
            if o in bottom.offsets:
                fromrow = bisect_left(bottom.offsets, o)
                newdata[torow,at+o:] = bottom.data[fromrow, at+o:]

        newShape = (newdata.shape[1], newdata.shape[1])
        if overwrite:
            newOp = self
        else:
            newOp = self.copy()
        newOp.D = scipy.sparse.dia_matrix((newdata, newoffsets), shape=newShape)
        # Update any attributes here!  (none right now)
        return newOp

    def copy(self):
        B = BandedOperator((self.D.data, self.D.offsets))
        # copy attributes here!
        return B



def splice_operator(top, bottom, idx=0):

    newoffsets = sorted(set(top.offsets).union(set(bottom.offsets)))
    newdata = np.zeros((len(newoffsets), top.shape[1]))

    # Copy the top part
    # print top.offsets
    # print newoffsets
    for torow, o in enumerate(newoffsets):
        if idx + o < 0:
            raise ValueError,("You are using forward or backward derivatives "
                              "too close to the edge of the vector. "
                              "(idx = %i, row offset = %i)" % (idx, o))
        if o in top.offsets:
            fromrow = bisect_left(top.offsets, o)
            newdata[torow,:idx+o] = top.data[fromrow, :idx+o]
        if o in bottom.offsets:
            fromrow = bisect_left(bottom.offsets, o)
            newdata[torow,idx+o:] = bottom.data[fromrow, idx+o:]

    newShape = (newdata.shape[1], newdata.shape[1])
    newOp = scipy.sparse.dia_matrix((newdata, newoffsets), shape=newShape)
    return newOp



class FiniteDifferenceEngineADI(FiniteDifferenceEngine):
    def __init__(self):
        FiniteDifferenceEngine.__init__(self)


def main():
    """Run main."""

    return 0

if __name__ == '__main__':
    main()
