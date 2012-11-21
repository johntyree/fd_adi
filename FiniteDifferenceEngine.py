#!/usr/bin/env python
# coding: utf8
"""<+Module Description.+>"""

# import sys
# import os
# import itertools as it

from bisect import bisect_left

import numpy as np
import scipy.sparse
import itertools



class FiniteDifferenceEngine(object):
    def __init__(self, grid, coefficients={}):
        """
        Coefficients is a dict of tuple, function pairs with c[i,j] referring to the
        coefficient of the i j derivative, dU/didj. Absent pairs are counted as zeros.

        The functions MUST be able to handle dims+1 arguments, with the first
        being time and the rest corresponding to the dimensions given by @grid.shape@.

        Still need a good way to handle cross terms.

        N.B. You don't actually want to do this with lambdas. They aren't real
        closures. Weird things will happen.

        Ex. (2D grid)
            { (0,)  : lambda t, x0, x1: 0.5,
              (0,0) : lambda t, x0, x1: x,
              # python magic lets be more general than (2*x1*t)
              (1,)  : lambda t, *dims: 2*dims[1]*t
              (0,1) : lambda t, *dims: dims[0]*dims[1]
            }

        is interpreted as:
            0.5*(dU/dx1) + x1*(d²U/dx1²) + 2*x2*t*(dU/dx2) + 0*(d²U/dx2²) + x1*x2*(d²U/dx1dx2)

        Can't do this with C/Cuda of course... maybe cython?
        """
        self.grid = grid
        self.operators = {}
        self.coefficients = coefficients
        self.t = 0
        self.make_discrete_operators()


    # This is really tied to 2D operations right now
    # We will assume we are working in either dimension 0 or dimension 1
    # In either case, the dimension we care about is x0 and the other one is x1
    # everything else will raise NotImplementedError

    def make_discrete_operators(self):
        ndim = self.grid.ndim
        coeffs = self.coefficients
        for d in coeffs.keys():
            BandedOperator.check_derivative(d)
            dim = d[0]
            otherdims = range(ndim+1)
            otherdims.remove(dim)
            # Make an operator for this dimension
            Binit = BandedOperator.for_vector(self.grid.mesh[dim], scheme='center', derivative=len(d), order=2)

            # take cartesian product of other dimension values
            argset = itertools.product(*(self.grid.mesh[i] for i in otherdims))
            # pair our current dimension with all combinations of the other dimensions
            Bs = []
            for a in argset:
                # Make a new operator
                B = Binit.copy()
                # Give it the specified coefficient
                def func(i):
                    args = list(a)
                    args.insert(dim, self.grid.mesh[dim][i])
                    return coeffs[d](self.t, *args)
                B.scale(func)
                Bs.append(B)
            self.operators[d] = flatten_tensor(Bs)
        return

    def solve(self):
        """Run all the way to the terminal condition."""
        raise NotImplementedError


class BandedOperator(object):
    def __init__(self, (data, offsets)):
        """
        A linear operator for discrete derivatives.
        Consist of a banded matrix (B.D) and a residual vector (B.R) for things
        like

            U2 = L*U1 + Rj  -->   U2 = B.apply(U1)
            U2 = L.I * (U1 - R) --> U2 = B.solve(U1)
        """
        size = data.shape[1]
        shape = (size, size)
        self.D = scipy.sparse.dia_matrix((data, offsets), shape=shape, dtype=float)
        self.R = np.zeros(shape[0], dtype=float)

    @classmethod
    def for_vector(cls, vector, scheme="center", derivative=1, order=1):
        """
        A linear operator for discrete derivative of @vector@.

        @derivative@ is a tuple specify the sequence of derivatives. For
        example, `(0,0)` is the second derivative in the first dimension.
        """

        cls.check_derivative(derivative)


        deltas = np.hstack((np.nan, np.diff(vector)))

        if scheme.lower().startswith("forward"):
            data, offsets = cls.forwardcoeffs(deltas, derivative=derivative, order=order)
        elif scheme.lower().startswith("center"):
            data, offsets = cls.centercoeffs(deltas, derivative=derivative, order=order)
        elif scheme.lower().startswith("backward"):
            data, offsets = cls.backwardcoeffs(deltas, derivative=derivative, order=order)

        self = BandedOperator((data, offsets))
        # self._deltas = deltas
        return self


    def apply(self, vector, overwrite=False):
        return self.D.dot(vector) + self.R


    def solve(self, vector, overwrite=False):
        return self.D.solve_banded(self.D.offsets, self.D.data,
                vector + self.R, overwrite_b=True)

    @staticmethod
    def check_derivative(d):
        try:
            d = tuple(d)
            if len(d) > 2:
                raise NotImplementedError, "Can't do more than 2nd order derivatives."
            if len(set(d)) != 1:
                #TODO
                raise NotImplementedError, "Restricted to 2D problems without cross derivatives."
            map(int, d)
            d = len(d)
        except TypeError:
            try:
                d = int(d)
            except TypeError:
                raise TypeError("derivative must be a number or an iterable of numbers")
        if d > 2 or d < 1:
            raise NotImplementedError, "Can't do 0th order or more than 2nd order derivatives."


    @staticmethod
    def check_order(order):
        if order != 2:
            raise NotImplementedError, ("Order must be 2")


    def __getattr__(self, name):
        return self.D.__getattribute__(name)


    # @property
    # def deltas(self):
        # return self._deltas


    @classmethod
    def forwardcoeffs(cls, deltas, derivative=1, order=2):
        d = deltas
        data = np.zeros((5,len(d)))

        cls.check_derivative(derivative)
        cls.check_order(order)

        if derivative == 1:
            offsets = [2,1,0,-1,-2]
            m = offsets.index(0)
            for i in range(1,len(d)-2):
                data[m-1,i+1] = (d[i+1] + d[i+2])  /         (d[i+1]*d[i+2])
                data[m-2,i+2] = -d[i+1]            / (d[i+2]*(d[i+1]+d[i+2]))
                data[m,i]     = (-2*d[i+1]-d[i+2]) / (d[i+1]*(d[i+1]+d[i+2]))
            # Use first order approximation for the last (inner) row
            data[m-1, -1] =  1 / d[-1]
            data[m, -2] = -1 / d[-1]
        elif derivative == 2:
            offsets = [2,1,0,-1,-2]
            m = offsets.index(0)
            for i in range(1,len(d)-2):
                denom = (0.5*(d[i+2]+d[i+1])*d[i+2]*d[i+1]);
                data[m-2,i+2] =   d[i+1]         / denom
                data[m-1,i+1] = -(d[i+2]+d[i+1]) / denom
                data[m,i]     =   d[i+2]         / denom
            # Use centered approximation for the last (inner) row
            data[m-1, -1] = 2  / (d[i+1]*(d[i]+d[i+1]))
            data[m  ,-2] = -2 /       (d[i]*d[i+1])
            data[m+1,-3] = 2  / (d[i  ]*(d[i]+d[i+1]))
        else:
            raise NotImplementedError, ("Order must be 1 or 2")

        return data, offsets


    @classmethod
    def centercoeffs(cls, deltas, derivative=1, order=2):
        """Centered differencing coefficients."""
        d = deltas
        data = np.zeros((5,len(d)))

        cls.check_derivative(derivative)
        cls.check_order(order)

        if derivative == 1:
            offsets = [2,1,0,-1,-2]
            m = offsets.index(0)
            for i in range(1,len(d)-1):
                data[m-1,i+1] =            d[i]  / (d[i+1]*(d[i]+d[i+1]))
                data[m  ,i  ] = (-d[i] + d[i+1]) /         (d[i]*d[i+1])
                data[m+1,i-1] =         -d[i+1]  / (d[i  ]*(d[i]+d[i+1]))
        elif derivative == 2:
            offsets = [2,1,0,-1,-2]
            m = offsets.index(0)
            for i in range(1,len(d)-1):
                data[m-1,i+1] =  2 / (d[i  ]*(d[i]+d[i+1]))
                data[m  ,i  ] = -2 /       (d[i]*d[i+1])
                data[m+1,i-1] =  2 / (d[i+1]*(d[i]+d[i+1]))
        else:
            raise NotImplementedError, ("Derivative must be 1 or 2")

        return data, offsets


    @classmethod
    def backwardcoeffs(cls, deltas, derivative=1, order=2):
        d = deltas
        data = np.zeros((5,len(d)))

        cls.check_derivative(derivative)
        cls.check_order(order)


        if derivative == 1:
            offsets = [2,1,0,-1,-2]
            m = offsets.index(0)
            for i in range(2,len(d)-1):
                data[m, i]     = (d[i-1]+2*d[i])  / (d[i]*(d[i-1]+d[i]));
                data[m+1, i-1] = (-d[i-1] - d[i]) / (d[i-1]*d[i]);
                data[m+2, i-2] = d[i]             / (d[i-1]*(d[i-1]+d[i]));
            # Use first order approximation for the first (inner) row
            data[m,   1] =  1 / d[-1]
            data[m+1, 0] = -1 / d[-1]
        elif derivative == 2:
            offsets = [2,1,0,-1,-2]
            m = offsets.index(0)
            for i in range(2,len(d)-1):
                denom = (0.5*(d[i]+d[i-1])*d[i]*d[i-1]);
                data[m,     i] =   d[i-1]       / denom;
                data[m+1, i-1] = -(d[i]+d[i-1]) / denom;
                data[m+2, i-2] =   d[i]         / denom;
            # Use centered approximation for the first (inner) row
            data[m+1,0] = 2  / (d[i+1]*(d[i]+d[i+1]))
            data[m,1]   = -2 /       (d[i]*d[i+1])
            data[m-1,2] = 2  / (d[i  ]*(d[i]+d[i+1]))
        else:
            raise NotImplementedError, ("Derivative must be 1 or 2")

        return data, offsets


    def splice_with(self, bottom, at, overwrite=False):
        """
        Splice a second operator into this one by replacing rows after @at@.
        If overwrite is True, split it in place.
        """
        newoffsets = sorted(set(self.offsets).union(set(bottom.offsets)), reverse=True)
        newdata = np.zeros((len(newoffsets), self.shape[1]))

        if any(at - o < 0 for o in newoffsets):
            print "Returning bottom cause we splicin' it all..."
            return bottom.copy()
        if any(at + o > self.shape[1] for o in [x for x in  newoffsets if x < 2]):
            print "Returning self cause we ain't splicin' shit..."
            return self.copy()

        from visualize import fp
        # print "self"
        # fp(self.todense())
        # print "bottom"
        # fp(bottom.todense())
        for torow, o in enumerate(newoffsets):
            if at - o < 0 or at + o > self.shape[1]:
                raise ValueError("You are reaching beyond the edge of the "
                                 "vector. (at = %i, row offset = %i)" % (at, o))
            if o in self.offsets:
                fromrow = list(self.offsets).index(o)
                newdata[torow,:at+o] = self.data[fromrow, :at+o]
                # print "new[%i, :%i+%i] = self[%i, :%i+%i]" % (torow, at, o, fromrow, at, o)
            if o in bottom.offsets:
                fromrow = list(bottom.offsets).index(o)
                newdata[torow,at+o:] = bottom.data[fromrow, at+o:]
                # print "new[%i, :%i+%i] = bottom[%i, :%i+%i]" % (torow, at, o, fromrow, at, o)

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


    def mul(self, val, inplace=False):
        if inplace:
            B = self
        else:
            B = self.copy()
        B.data *= val
        return B


    def add(self, addendfunc):
        raise NotImplementedError

    def __eq__(self, other):
        return ((self.data == other.data).all()
            and (self.offsets == other.offsets).all()
            and (self.shape == other.shape)
            and (self.R == other.R).all()
            )


    def add(self, other, inplace=False):
        selfoffsets = tuple(self.offsets)
        if isinstance(other, BandedOperator):
            otheroffsets = tuple(other.offsets)
            if self.shape[1] != other.shape[1]:
                raise ValueError("Both operators must have the same length")
            if inplace:
                if otheroffsets != selfoffsets and inplace:
                    raise ValueError("Both operators must have (exactly) the"
                                     " same offsets to add in-place.")
                B = self
                Boffsets = selfoffsets
            else:
                newoffsets = sorted(set(selfoffsets).union(set(otheroffsets)),
                                    reverse=True)
                newdata = np.zeros((len(newoffsets), self.shape[1]))
                B = BandedOperator((newdata, newoffsets))
                Boffsets = tuple(B.offsets)
                for o in selfoffsets:
                    fro = selfoffsets.index(o)
                    to = Boffsets.index(o)
                    # print "fro(%i) -> to(%i)" % (fro, to)
                    B.data[to] += self.data[fro]
            for o in otheroffsets:
                fro = otheroffsets.index(o)
                to = Boffsets.index(o)
                B.data[to] += other.data[fro]
        else:
            m = selfoffsets.index(0)
            if m > len(selfoffsets)-1:
                raise NotImplementedError("Cannot (yet) add scalar to operator"
                                          " without main diagonal.")
            if inplace:
                B = self
            else:
                B = self.copy()
            B.data[m] += other

        return B





    def scale(self, func):
        """
        func must be compatible with the following:
            func(x)
        Where x is the correpsonding value of the current dimension.

        See FiniteDifferenceEngine.coefficients.
        """
        for row, o in enumerate(self.offsets):
            if o >= 0:
                for i in xrange(self.shape[0]-abs(o)):
                    self.data[row,o+i] *= func(i)
            else:
                for i in xrange(self.shape[0]-abs(o)):
                    self.data[row, i] *= func(i-o)

                  # (2 to end)     (begin to end-1)
        # As.data[m - 2, 2:] *= mu_s[:-2]
        # As.data[m - 1, 1:] *= mu_s[:-1]
        # As.data[m, :] *= mu_s
        # As.data[m + 1, :-1] *= mu_s[1:]
        # As.data[m + 2, :-2] *= mu_s[2:]



def flatten_tensor(mats):
    diags = np.hstack([x.data for x in mats])
    flatmat = BandedOperator((diags, mats[0].offsets))
    return flatmat


class FiniteDifferenceEngineADI(FiniteDifferenceEngine):
    def __init__(self):
        FiniteDifferenceEngine.__init__(self)


def main():
    """Run main."""

    return 0

if __name__ == '__main__':
    main()
