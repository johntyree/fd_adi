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

# In order to do the boundary conditions correctly, we MUST handle them with
# the FiniteDifferenceEngine so that we'll have full access to grid.mesh and
# time, required for lambda t, *dims:...



class FiniteDifferenceEngine(object):
    def __init__(self, grid, coefficients={}, boundaries={}):
        """
        @coefficients@ is a dict of tuple, function pairs with c[i,j] referring to the
        coefficient of the i j derivative, dU/didj. Absent pairs are counted as zeros.

        The functions MUST be able to handle dims+1 arguments, with the first
        being time and the rest corresponding to the dimensions given by @grid.shape@.

        Still need a good way to handle cross terms.

        N.B. You don't actually want to do this with lambdas. They aren't real
        closures. Weird things will happen.

        Ex. (2D grid)
            { (None,): lambda t, x0, x1: 0.06 # 0th derivative
              (0,)  : lambda t, x0, x1: 0.5,
              (0,0) : lambda t, x0, x1: x,
              # python magic lets be more general than (2*x1*t)
              (1,)  : lambda t, *dims: 2*dims[1]*t
              (0,1) : lambda t, *dims: dims[0]*dims[1]
            }

        is interpreted as:
            0.5*(dU/dx1) + x1*(d²U/dx1²) + 2*x2*t*(dU/dx2)
                + 0*(d²U/dx2²) + x1*x2*(d²U/dx1dx2)

        A similar scheme applies for the boundaries conditions.  @boundaries@
        is a dict of 2-tuples corresponding to the lower and upper boundary
        (where lower means low index in @grid.mesh@). Each can have one of
        two possible values.

        The first element is a number corresponding to the type of boundary
        condition.
            None signifies free boundaries, simply satisfying the PDE.
            0 is a dirchlet boundary condition
            1 is a Von Neuman boundary

        It's up to you to make sure this makes sense. If you specify a
        dirichlet boundary somewhere, then it doesn't make sense to also
        specify a Von Neumann boundary there.

        The second element is a function in the same form as for the coefficients,
        representing the value of the boundary.

        Ex. (Heston PDE, x0 = spot, x1 = variance)
                    # 0'th derivative term (ex. -rU)
                    # This can only depend on time!
            {()    : lambda t: -self.r
                    # dirichlet: U = 0         # VN: dUdS = 1
             (0,)  : ((1, lambda *args: 0), (1, lambda *args:1))
                    # dirichlet: U = 0         # Free boundary
             (0,0) : ((1, lambda *args: 0), (None, lambda *x: None)),
                    # Free boundary at low variance
             (1,)  : ((None, lambda *x: None),
                    # dirichlet: intrinsic value at high variance
                       (1, lambda t, *dims: np.maximum(0, dim[0]-k)))
                    # Free boundary at low variance
                    # VN: second derivative is 0 at high variance
            (1,1)  : ((None, lambda *x: None), (1, lambda *args:0))
             }

        Can't do this with C/Cuda of course... maybe cython?
        """
        self.grid = grid
        self.operators = {}
        self.coefficients = coefficients
        self.boundaries = boundaries
        self.t = 0
        self.make_discrete_operators()


    def make_discrete_operators(self):
        # TODO: Hardcoding in the boundary conditions... :(
        ndim = self.grid.ndim
        coeffs = self.coefficients
        bounds = self.boundaries
        order = 2
        # key = dim, val = [Bool, Bool] # upper lower
        dirichlets = {}
        def wrapfunc(f, args, dim):
            x = list(args)
            x.insert(dim, None)
            def newf(i):
                x[dim] = self.grid.mesh[dim][i]
                # print "Args for dim %i: %s" % (dim, x)
                return f(self.t, *x)
            return newf

        for d in coeffs.keys():
            # Don't need an operator for the 0th derivative
            if d == ():
                continue
            BandedOperator.check_derivative(d)
            dim = d[0]
            otherdims = range(ndim)
            otherdims.remove(dim)
            Bs = []
            # Make an operator for this dimension
            #TODO
            # schemes[d] = [('center', 0), ('backward', 22)]
            Binit = BandedOperator.for_vector(self.grid.mesh[dim],
                        scheme='center', derivative=len(d), order=order)
            m = tuple(Binit.offsets).index(0)
            if m > len(Binit.offsets)-1:
                raise ValueError("No main diagonal!")
            #TODO: Splice here
            # take cartesian product of other dimension values
            argset = list(itertools.product(*(self.grid.mesh[i] for i in otherdims)))
            if dim not in dirichlets:
                dirichlets[dim] = [None] * len(argset)
            # pair our current dimension with all combinations of the other dimensions
            for col, a in enumerate(argset):
                # Make a new operator
                B = Binit.copy()

                # Give it the specified coefficient
                # Here we wrap the functions with the appropriate values for
                # this particular dimension.
                b = bounds[d]
                lowfunc = wrapfunc(b[0][1], a, dim)
                highfunc = wrapfunc(b[1][1], a, dim)
                # print "old b(%s) -> (%s, %s)" % (d, b[0][1](0), b[1][1](-1))
                # print b
                b = ((b[0][0], lowfunc(0)), (b[1][0], highfunc(-1)))
                # print "new b(%s) -> (%s (%s), %s (%s))" % (d, b[0][1], lowval, b[1][1], highval)
                # print b
                # print "Boundary:", d
                B.applyboundary(b)
                if dim == 1:
                    g = bounds[d][1][1]
                    f = wrapfunc(g, a, dim)
                    # import pdb; pdb.set_trace() ### XXX BREAKPOINT

                B.scale(wrapfunc(coeffs[d], a, dim))
                # If it's a dirichlet boundary we mark it because we have to
                # handle it absolutely last
                lf = hf = None
                if b[0][0] == 0:
                    lf = lowfunc
                    # bounds[dim].append(b)
                if b[1][0] == 0:
                    hf = highfunc
                if lowfunc or highfunc:
                    dirichlets[dim][col] = (lf, hf)
                    # bounds[dim].append(b)
                # print
                Bs.append(B)

            # Combine operators for this dimension
            if dim not in self.operators:
                self.operators[dim] = Bs
            else:
                for col, b in enumerate(Bs):
                    self.operators[dim][col] += b

        # Now the 0th derivative
        #TODO: This function is ONLY dependent on time. NOT MESH
        if () in coeffs:
            for Bs in self.operators.values():
                for B in Bs:
                    B += coeffs[()](self.t) / float(ndim)

        # Now eveerything else is done. Enforce dirichlet boundaries
        for (dim, funcs) in dirichlets.items():
            # print "Mesh(%i) has %i ops with %i funcs:" % (dim,
                    # len(self.operators[dim]), len(funcs)), self.grid.mesh[dim]
            # otherdims = range(ndim)
            # otherdims.remove(dim)
            # argset = list(itertools.product(*(self.grid.mesh[i] for i in otherdims)))
            for i, B in enumerate(self.operators[dim]):
                lowfunc, highfunc = funcs[i]
                if lowfunc:
                    B.R[0] = lowfunc(0)
                    B.data[m, 0] = -1
                if highfunc:
                    B.R[-1] = highfunc(-1)
                    B.data[m, -1] = -1

        for d in range(ndim):
            # ops = [o for o in coeffs.keys() if all(i == d for i in o)]
            # self.operators[d] = self.operators[ops[0]]
            # for o in ops[1:]:
                # for i, b in enumerate(self.operators[d]):
                    # b += self.operators[o][i]
            self.operators[d] = flatten_tensor(self.operators[d])
        return


    def solve(self):
        """Run all the way to the terminal condition."""
        raise NotImplementedError


class BandedOperator(object):
    def __init__(self, (data, offsets), residual=None, inplace=True):
        """
        A linear operator for discrete derivatives.
        Consist of a banded matrix (B.D) and a residual vector (B.R) for things
        like

            U2 = L*U1 + Rj  -->   U2 = B.apply(U1)
            U2 = L.I * (U1 - R) --> U2 = B.solve(U1)
        """
        if not inplace:
            data = data.copy()
            offsets = tuple(offsets)
            if residual is not None:
                residual = residual.copy()
        size = data.shape[1]
        shape = (size, size)
        self.D = scipy.sparse.dia_matrix((data, offsets), shape=shape, dtype=float)
        if residual is None:
            self.R = np.zeros(shape[0], dtype=float)
        elif residual.shape[0] == self.shape[0] and residual.ndim == 1:
            self.R = residual
        else:
            raise ValueError("Residual vector has wrong shape: got %i,"
                             "expected %i." % (residual.shape[0], size))
        self.derivative = None
        self.order = None
        self.deltas = np.array([np.nan])

    @classmethod
    def for_vector(cls, vector, scheme="center", derivative=1, order=1, residual=None):
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

        self = BandedOperator((data, offsets), residual=residual)
        self.derivative = derivative
        self.order = order
        self.deltas = deltas
        return self

    def copy(self):
        B = BandedOperator((self.data, self.offsets), residual=self.R, inplace=False)
        B.derivative = self.derivative
        B.order = self.order
        B.deltas = self.deltas
        return B



    def apply(self, vector, overwrite=False):
        return self.D.dot(vector) + self.R


    def applyboundary(self, boundary):
        """
        @boundary@ is a tuple from FiniteDifferenceEngine.boundaries.

        data are the packed diagonals and residual is the residual vector.
        """
        B = self
        m = tuple(B.offsets).index(0)
        d = B.deltas
        B.R = np.zeros(B.data.shape[1])
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
            # Dirichlet boundary. No derivatives.
            pass
        elif lower_type == 1:
            # Von Neumann boundary, we specify it directly.
            B.R[0] = lower_val
        elif lower_type is None and derivative == 1:
            # Free boundary
            # Second order forward approximation
            B.data[m - 2, 2] = -d[1] / (d[2] * (d[1] + d[2]))
            B.data[m - 1, 1] = (d[1] + d[2]) / (d[1] * d[2])
            B.data[m,     0] = (-2 * d[1] - d[2]) / (d[1] * (d[1] + d[2]))
        elif lower_type is None and derivative == 2:
            # Extrapolate second derivative by assuming the first stays
            # constant.
            B.data[m-1, 1] =  2 / d[1]**2
            B.data[m,   0] = -2 / d[1]**2
            B.R[0]         =  2 / d[1]
        else:
            raise NotImplementedError("Can't handle derivatives higher than"
                                      " order 2 at boundaries. (%s)" % derivative)

        # Doing upper boundary
        if upper_type == 0:
            # Dirichlet boundary. No derivatives.
            pass
        elif upper_type == 1:
            # Von Neumann boundary, we specify it directly.
            B.R[-1] = upper_val
        elif upper_type is None and derivative == 1:
            # Second order backward approximation
            B.data[m  , -1] = (d[-2]+2*d[-1])  / (d[-1]*(d[-2]+d[-1]))
            B.data[m+1, -2] = (-d[-2] - d[-1]) / (d[-2]*d[-1])
            B.data[m+2, -3] = d[-1]             / (d[-2]*(d[-2]+d[-1]))
        elif upper_type is None and derivative == 2:
            if B.R is None:
                B.R = np.zeros(B.data.shape[1])
            # Extrapolate second derivative by assuming the first stays
            # constant.
            B.data[m+1, -2] =  2 / d[-1]**2
            B.data[m,   -1] = -2 / d[-1]**2
            B.R[-1]         =  2 / d[-1]
        else:
            raise NotImplementedError("Can't handle derivatives higher than"
                                      " order 2 at boundaries. (%s)" % derivative)

        # if upper_type == 1 or upper_type is None:
            # print "Derivative:", derivative
            # print "boundary:", boundary
            # print "R:", B.R


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
            # Use centered approximation for the last (inner) row.
            data[m-1,-1] =           d[-2]  / (d[-1]*(d[-2]+d[-1]))
            data[m,  -2] = (-d[-2] + d[-1]) /        (d[-2]*d[-1])
            data[m+1,-3] =          -d[-1]  / (d[-2]*(d[-2]+d[-1]))

        elif derivative == 2:
            offsets = [2,1,0,-1,-2]
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
            # Inner rows
            for i in range(1,len(d)-1):
                data[m-1,i+1] =  2 / (d[i+1]*(d[i]+d[i+1]))
                data[m  ,i  ] = -2 /       (d[i]*d[i+1])
                data[m+1,i-1] =  2 / (d[i  ]*(d[i]+d[i+1]))
        else:
            raise NotImplementedError("Derivative must be 1 or 2")

        return (data, offsets)


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
            # Use centered approximation for the first (inner) row.
            data[m-1,2] =          d[1]  / (d[2]*(d[1]+d[2]))
            data[m,  1] = (-d[1] + d[2]) /       (d[1]*d[2])
            data[m+1,0] =         -d[2]  / (d[1]*(d[1]+d[2]))
        elif derivative == 2:
            offsets = [2,1,0,-1,-2]
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


    def splice_with(self, bottom, at, overwrite=False):
        """
        Splice a second operator into this one by replacing rows after @at@.
        If overwrite is True, split it in place.
        """
        newoffsets = sorted(set(self.offsets).union(set(bottom.offsets)), reverse=True)
        newdata = np.zeros((len(newoffsets), self.shape[1]))

        if any(at - o < 0 for o in newoffsets):
            # print "Returning bottom cause we splicin' it all..."
            return bottom.copy()
        if any(at + o > self.shape[1] for o in [x for x in  newoffsets if x < 2]):
            # print "Returning self cause we ain't splicin' shit..."
            return self.copy()

        # from visualize import fp
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


    def __mul__(self, val, inplace=False):
        return self.mul(val, inplace=inplace)
    def __imul__(self, val, inplace=True):
        return self.mul(val, inplace=inplace)

    def mul(self, val, inplace=False):
        if inplace:
            B = self
        else:
            B = self.copy()
        B.data *= val
        B.R *= val
        return B


    def __eq__(self, other):
        no_nan = np.nan_to_num
        #TODO: These asserts are just to make testing easier, remove them.
        assert (self.data == other.data).all()
        assert (self.offsets == other.offsets).all()
        assert (self.shape == other.shape)
        assert (no_nan(self.deltas) == no_nan(other.deltas)).all()
        assert (self.order == other.order)
        assert (self.derivative == other.derivative)
        assert (self.R == other.R).all()
        return ((self.data == other.data).all()
            and (self.offsets == other.offsets).all()
            and (self.shape == other.shape)
            and (no_nan(self.deltas) == no_nan(other.deltas)).all()
            and (self.order == other.order)
            and (self.derivative == other.derivative)
            and (self.R == other.R).all()
            )


    def __add__(self, other):
        return self.add(other, inplace=False)
    def __iadd__(self, other):
        return self.add(other, inplace=True)


    def add(self, other, inplace=False):
        """
        Add a scalar to the main diagonal or a second BandedOperator to this one.
        Does not alter self.R, the residual vector.
        """
        selfoffsets = tuple(self.offsets)

        # If we're adding two operators together
        if isinstance(other, BandedOperator):
            otheroffsets = tuple(other.offsets)
            # Verify that they are compatible
            if self.shape[1] != other.shape[1]:
                raise ValueError("Both operators must have the same length")
            # If we're adding it directly to this one
            if inplace:
                # The diagonals have to line up.
                if otheroffsets != selfoffsets:
                    raise ValueError("Both operators must have (exactly) the"
                                     " same offsets to add in-place.")
                B = self
                Boffsets = selfoffsets
            # Otherwise we are adding directly to this one.
            else:
                # Calculate the offsets that the new one will have.
                Boffsets = sorted(set(selfoffsets).union(set(otheroffsets)),
                                    reverse=True)
                newdata = np.zeros((len(Boffsets), self.shape[1]))
                # And make a new operator with the appropriate shape
                # Remember to carry the residual with us.
                B = BandedOperator((newdata, Boffsets), self.R)
                # Copy our data into the new operator since carefully, since we
                # may have resized.
                for o in selfoffsets:
                    fro = selfoffsets.index(o)
                    to = Boffsets.index(o)
                    # print "fro(%i) -> to(%i)" % (fro, to)
                    B.data[to] += self.data[fro]
            # Copy the data from the other operator over
            for o in otheroffsets:
                fro = otheroffsets.index(o)
                to = Boffsets.index(o)
                B.data[to] += other.data[fro]
            # Now the residual vector from the other one
            if other.R is not None:
                if B.R is None:
                    B.R = other.R.copy()
                else:
                    B.R += other.R
        # If we aren't adding an operator, we are adding a scalar
        else:
            # We add it to the main diagonal.
            m = selfoffsets.index(0)
            if m > len(selfoffsets)-1:
                raise NotImplementedError("Cannot (yet) add scalar to operator"
                                          " without main diagonal.")
            if inplace:
                B = self
            else:
                B = self.copy()
            B.data[m] += other
            # Don't touch the residual.
        return B


    def scale(self, func):
        """
        func must be compatible with the following:
            func(x)
        Where x is the correpsonding value of the current dimension.

        Also applies to the residual vector self.R.

        See FiniteDifferenceEngine.coefficients.
        """
        for row, o in enumerate(self.offsets):
            if o >= 0:
                for i in xrange(self.shape[0]-o):
                    self.data[row,i+o] *= func(i)
            else:
                for i in xrange(abs(o),self.shape[0]):
                    self.data[row, i-abs(o)] *= func(i)
        for i in xrange(self.shape[0]):
            self.R[i] *= func(i)

                  # (2 to end)     (begin to end-1)
        # As.data[m - 2, 2:] *= mu_s[:-2]
        # As.data[m - 1, 1:] *= mu_s[:-1]
        # As.data[m, :] *= mu_s
        # As.data[m + 1, :-1] *= mu_s[1:]
        # As.data[m + 2, :-2] *= mu_s[2:]


def flatten_tensor(mats):
    diags = np.hstack([x.data for x in mats])
    residual = np.hstack([x.R for x in mats])
    flatmat = BandedOperator((diags, mats[0].offsets), residual=residual)
    return flatmat


class FiniteDifferenceEngineADI(FiniteDifferenceEngine):
    def __init__(self):
        FiniteDifferenceEngine.__init__(self)


def main():
    """Run main."""

    return 0

if __name__ == '__main__':
    main()
