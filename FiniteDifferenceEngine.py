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
import utils
import scipy.linalg as spl

from visualize import fp



class FiniteDifferenceEngine(object):

    def __init__(self, grid, coefficients={}, boundaries={}, schemes={},
            force_bandwidth=None):
        """
        @coefficients@ is a dict of tuple, function pairs with c[i,j] referring to the
        coefficient of the i j derivative, dU/didj. Absent pairs are counted as zeros.

        The functions MUST be able to handle dims+1 arguments, with the first
        being time and the rest corresponding to the dimensions given by @grid.shape@.

        Still need a good way to handle cross terms.

        N.B. You don't actually want to do this with lambdas. They aren't real
        closures. Weird things will happen.

        Ex. (2D grid)

            {(None,): lambda t, x0, x1: 0.06 # 0th derivative
              (0,)  : lambda t, x0, x1: 0.5,
              (0,0) : lambda t, x0, x1: x,
              # python magic lets be more general than (2*x1*t)
              (1,)  : lambda t, *dims: 2*dims[1]*t
              (0,1) : lambda t, *dims: dims[0]*dims[1]}

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
            (1,1)  : ((None, lambda *x: None), (1, lambda *args:0))}


        Again we do a similar encoding for the FD schemes used.

        @schemes@ is a dict of tuples of dicts as follows.

        Ex. (Centered in all cases. Switch to upwinding at index 10 in
                convection term in x1 dimension.)

            {(0,) : ({"scheme": "center"},),
            (0,0): ({"scheme": "center"},),
            (1,) : ({"scheme": "center"},
                    {"scheme": "backward", "from" : flip_idx_var}),
            (1,1): ({"scheme": "center"},)}

        Any missing values are determined by @self.default_scheme@ and
        @self.default_order@ (making this particular example largely
        redundant).

        Can't do this with C/Cuda of course... maybe cython?
        """
        self.grid = grid
        self.ndim = self.grid.ndim
        self.coefficients = coefficients
        self.boundaries = boundaries
        self.schemes = schemes
        self.force_bandwidth = force_bandwidth
        self.t = 0
        self.default_scheme = 'center'
        self.default_order = 2

        # Setup
        self.operators = {}


    def solve(self):
        """Run all the way to the terminal condition."""
        raise NotImplementedError


class FiniteDifferenceEngineADI(FiniteDifferenceEngine):

    def __init__(self, grid, coefficients={}, boundaries={}, schemes={},
            force_bandwidth=None):

        FiniteDifferenceEngine.__init__(self, grid, coefficients=coefficients,
                boundaries=boundaries, schemes=schemes, force_bandwidth=force_bandwidth)
        self.make_discrete_operators()

    def make_operator_template(self, d, dim=None, force_bandwidth=None):
        # Make an operator template for this dimension
        Binit = None
        for sd in self.schemes.get(d, ({},)): # a tuple of dicts
            s = sd.get('scheme', self.default_scheme)
            if not s:
                s = self.default_scheme
            idx = sd.get('from', 0)
            o = sd.get('order', self.default_order)
            B = BandedOperator.for_vector(self.grid.mesh[dim],
                    scheme=s, derivative=len(d), order=o,
                    force_bandwidth=force_bandwidth)
            if Binit is not None:
                # print "splicing with %s at %i," % (s, idx),
                # They asked for more than one scheme,
                # Splice the operators together at row idx
                # TODO: Not inplace until we figure out how to pre-allocate the
                # right size
                Binit = Binit.splice_with(B, idx, inplace=False)
            else:
                # print "%s: Starting with %s," % (d, s),
                Binit = B
        # print "done."
        return Binit

    def min_possible_bandwidth(self, derivative_tuple):
        high = low = 0
        d = len(derivative_tuple)
        if derivative_tuple not in self.boundaries:
            self.boundaries[derivative_tuple] = ((None, None), (None, None))
        b = self.boundaries[derivative_tuple]
        for sd in self.schemes.get(derivative_tuple, ({},)): # a tuple of dicts
            # print "checking scheme from %s: %s" % (derivative_tuple, sd)
            s = sd.get('scheme', self.default_scheme)
            if sd.get('from', 0) == 0: # We're overwriting previous schemes
                # print "Reset!"
                high = low = 0
            if s == 'center' or '':
                # print "Center requires -1, 1"
                high = max(1, high)
                low = min(-1, low)
            elif s == 'forward' or s == 'up':
                # print "Forward requires 0, 2"
                high = max(2, high)
                low = min(0, low)
            elif s == 'backward' or s == 'down':
                # print "Backward requires -2, 0"
                high = max(0, high)
                low = min(-2, low)
        if b[0][0] is None:
            if d == 1:
                h = 2
            elif d == 2:
                h = 1
            # print ("Low free boundary requires forward"
                    # " difference (%i): %i (have %i)" % (d, h, high))
            high = max(h, high)
        if b[1][0] is None:
            if d == 1:
                l = -2
            elif d == 2:
                l = -1
            # print ("High free boundary requires backward"
                   # " difference (%i): %i (have %i)" % (d, l, low))
            low = min(l, low)
        # print "%s requires (%s, %s)" % (derivative_tuple, high, low)
        return low, high


    def make_discrete_operators(self):
        ndim = self.grid.ndim
        coeffs = self.coefficients
        bounds = self.boundaries
        force_bandwidth = self.force_bandwidth
        dirichlets = {}
        mixed = {}

        # We partially apply the function to all of the values except this
        # dimension, and change the semantics to accept an index into mesh
        # rather than the mesh value itself. This way the caller doesn't need
        # to know anything other than row it's working on.
        # f : (t x R^n -> R) -> Z -> R
        def wrapscalarfunc(f, args, dim):
            x = list(args)
            x.insert(dim, None)
            def newf(i):
                x[dim] = self.grid.mesh[dim][i]
                return f(self.t, *x)
            return newf

        # Here we do the same except we go ahead and evalutate the function for
        # the entire vector. This is just for numpy's speed and is otherwise
        # redundant.

        def evalvectorfunc(f, args, dim):
            x = list(args)
            x.insert(dim, self.grid.mesh[dim])
            return f(self.t, *x)

        # d = (0,0) or so...
        for d in coeffs.keys():
            # Don't need an operator for the 0th derivative
            # Mixed derivatives are handled specially
            if d == ():
                continue

            mix = BandedOperator.check_derivative(d)
            if mix:
                mixed[d] = True
                continue

            dim = d[0]
            otherdims = range(ndim)
            otherdims.remove(dim)
            Bs = []

            # # Make an operator template for this dimension
            low, high = self.min_possible_bandwidth(d)
            bw = force_bandwidth
            # print "Minimum bandwidth for %s: %s" % (d, (low, high))
            if bw is not None:
                if (bw[0] > low or bw[1] < high):
                    raise ValueError("Your chosen scheme is too wide for the"
                            " specified bandwidth. (%s needs %s)" %
                            (bw, (low, high)))
                low, high = bw
            Binit = self.make_operator_template(d, dim=dim,
                    force_bandwidth=(low, high))
            # if dim == 1:
                # print Binit.data
                # print
            offs = Binit.offsets
            assert max(offs) >= high, "(%s < %s)" % (max(offs), high)
            assert min(offs) <= low,  "(%s > %s)" % (min(offs), low)
            # m is the main diagonal's index in B.data
            m = tuple(Binit.offsets).index(0)
            if m > len(Binit.offsets)-1:
                #TODO: Why no main diag? Should be possible, if not probable.
                raise ValueError("No main diagonal!")
            # Take cartesian product of other dimension values
            argset = itertools.product(*(self.grid.mesh[i] for i in otherdims))
            # Pair our current dimension with all combinations of the other
            # dimension values
            for a in argset:
                # Make a new operator from our template
                B = Binit.copy()

                # Adjust the boundary conditions as necessary
                if d in bounds:
                    b = bounds[d]
                    lowfunc = wrapscalarfunc(b[0][1], a, dim)
                    highfunc = wrapscalarfunc(b[1][1], a, dim)
                    b = ((b[0][0], lowfunc(0)), (b[1][0], highfunc(-1)))
                    B.applyboundary(b)
                    # If it's a dirichlet boundary we mark it because we have to
                    # handle it absolutely last, it doesn't get scaled.
                    lf = hf = None
                    if B.is_dirichlet[0]:
                        lf = lowfunc
                    if B.is_dirichlet[1]:
                        hf = highfunc
                    if lf or hf:
                        if dim not in dirichlets:
                            dirichlets[dim] = []
                        dirichlets[dim].append((lf, hf))

                # Give the operator the right coefficient
                # Here we wrap the functions with the appropriate values for
                # this particular dimension.
                if d in coeffs:
                    B.vectorized_scale(evalvectorfunc(coeffs[d], a, dim))
                Bs.append(B)

            # Combine scaled derivatives for this dimension
            if dim not in self.operators:
                self.operators[dim] = Bs
            else:
                ops = self.operators[dim] # list reference
                for col, b in enumerate(Bs):
                    if tuple(ops[col].offsets) == tuple(b.offsets):
                        ops[col] += b
                    else:
                        ops[col] = ops[col] + b

        # Now the 0th derivative (x * V)
        # We split this evenly among each dimension
        #TODO: This function is ONLY dependent on time. NOT MESH
        if () in coeffs:
            for Bs in self.operators.values():
                for B in Bs:
                    B += coeffs[()](self.t) / float(ndim)

        # Handle the mixed derivatives. Not very DRY, refactor someday?
        # iters over keys by default
        # First we build a normal central difference vector in the first
        # dimension.
        for d in mixed:
            if len(d) > 2:
                raise NotImplementedError("Derivatives must be 2nd order or"
                        " less.")
            # Just to make sure
            for sd in self.schemes.get(d, ({},)): # a tuple of dicts
                s = sd.get('scheme', self.default_scheme)
                if s != 'center':
                    raise NotImplementedError("Mixed derivatives can only be"
                        " done with central differencing.")

            d0_size = len(self.grid.mesh[d[0]])
            d1_size = len(self.grid.mesh[d[1]])

            # If we were doing mixed derivatives here, we'd have to do splicing
            # now at the meta-operator level.
            # Bs = self.make_operator_template(d, dim=d[0])
            # Bm1 = self.make_operator_template(d, dim=d[1])
            Bs = BandedOperator.for_vector(self.grid.mesh[0], derivative=1)
            Bm1 = BandedOperator.for_vector(self.grid.mesh[1], derivative=1)
            Bb1 = Bm1.copy()
            Bp1 = Bm1.copy()

            # TODO: Hardcoding in for centered differencing
            Bps = [Bp1 * 0, Bp1 * 0] + replicate(d0_size-2, Bp1)
            Bbs = [Bb1 * 0] + replicate(d0_size-2, Bb1) +  [Bb1 * 0]
            Bms = replicate(d0_size-2, Bm1) + [Bm1 * 0, Bm1 * 0]

            offsets = Bs.offsets
            data = [Bps, Bbs, Bms]
            for row, o in enumerate(offsets):
                if o >= 0:
                    for i in xrange(Bs.shape[0]-o):
                        # print "(%i, %i)" % (row, i+o), "Block", i, i+o, "*", Bs.data[row, i+o]
                        a = (np.array(self.grid.mesh[0][i]).repeat(d1_size),)
                        vec = evalvectorfunc(coeffs[d], a, 1)
                        print "= high"
                        fp(data[row][i+o])
                        data[row][i+o].vectorized_scale(vec)
                        fp(data[row][i+o])
                        print "="
                        data[row][i+o] *= Bs.data[row, i+o]
                else:
                    for i in xrange(abs(o), Bs.shape[0]):
                        # print "(%i, %i)" % (row, i-abs(o)), "Block", i, i-abs(o), "*", Bs.data[row, i-abs(o)]
                        a = (np.array(self.grid.mesh[0][i]).repeat(d1_size),)
                        vec = evalvectorfunc(coeffs[d], a, 1)
                        print "= low"
                        fp(data[row][i-abs(o)])
                        data[row][i-abs(o)].vectorized_scale(vec)
                        fp(data[row][i-abs(o)])
                        print "="
                        data[row][i-abs(o)] *= Bs.data[row, i-abs(o)]


            # We flatten here because it's faster
            Bps[0].offsets += d1_size
            Bms[0].offsets -= d1_size
            BP = flatten_tensor_aligned(Bps, check=False)
            BB = flatten_tensor_aligned(Bbs, check=False)
            BM = flatten_tensor_aligned(Bms, check=False)
            self.operators[d] = BP+BB+BM

        # Now everything else is done we can apply dirichlet boundaries
        for (dim, funcs) in dirichlets.items():
            # For each operator in this dimension
            for i, B in enumerate(self.operators[dim]):
                m = tuple(B.offsets).index(0)
                lowfunc, highfunc = funcs[i]
                if lowfunc:
                    # Cancel out the old value
                    B.data[m, 0] = -1
                    # Replace with the new one
                    B.R[0] = lowfunc(0)
                if highfunc:
                    B.data[m, -1] = -1
                    B.R[-1] = highfunc(-1)

        # Flatten into one large operator for speed
        # We'll apply this to a flattened domain and then reshape it.
        for d in self.operators.keys():
            if isinstance(self.operators[d], list):
                self.operators[d] = flatten_tensor_misaligned(self.operators[d])
        return


    def impl(self, n, dt, crumbs=[], callback=None):
        n = int(n)
        if crumbs:
            V = crumbs[-1]
        else:
            V = self.grid.domain.copy()
            crumbs.append(V)

        ops = []
        mixed = None
        for d, op in self.operators.items():
            if not isinstance(d, tuple):
                ops.append(op)
            else:
                if mixed:
                    mixed += op
                else:
                    mixed = op
        if mixed is not None:
            mixed *= dt

        ordered_ops = sorted(ops)
        Ls = np.roll([(op * -dt).add(1, inplace=True) for op in ordered_ops], -1)

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("Impl:")
        for k in xrange(n):
            if not k % print_step:
                if np.isnan(V).any():
                    print "Impl fail @ t = %f (%i steps)" % (dt * k, k)
                    return crumbs
                print int(k * to_percent),
            if callback is not None:
                callback(V, ((n - k) * dt))
            if mixed is not None:
                V += mixed.apply(V)
            for L in Ls:
                V = L.solve(V).T
        crumbs.append(V.copy())
        utils.toc()
        return crumbs


    def crank(self, n, dt, crumbs=[], callback=None):
        n = int(n)
        if crumbs:
            V = crumbs[-1]
        else:
            V = self.grid.domain.copy()
            crumbs.append(V)
        dt *= 0.5
        crumbs.append(V)

        ordered_ops = sorted(self.operators.items())
        # Have to roll the first one because of the scheme
        Les = np.roll([(op *  dt).add(1, inplace=True) for d, op in ordered_ops], -1)
        Lis = ((op * -dt).add(1, inplace=True) for d, op in ordered_ops)
        Ls = zip(Les, Lis)


        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("Crank:")
        for k in xrange(n):
            if not k % print_step:
                if np.isnan(V).any():
                    print "Crank fail @ t = %f (%i steps)" % (dt * k, k)
                    return crumbs
                print int(k * to_percent),
            if callback is not None:
                callback(V, ((n - k) * dt))
            for Le, Li in Ls:
                V = Le.apply(V).T
                V = Li.solve(V)
            crumbs.append(V.copy())
        utils.toc()
        return crumbs


    def smooth(self, n, dt, crumbs=[], callback=None, smoothing_steps=2):
        V = self.impl(smoothing_steps*2, dt*0.5, crumbs=crumbs)
        return self.crank(n-smoothing_steps, dt, crumbs=V)

def replicate(n, x):
    ret = []
    for _ in xrange(n):
        ret.append(x.copy())
    return ret


class BandedOperator(object):


    def __init__(self, data_offsets, residual=None, inplace=True):
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
        self.solve_banded_offsets = (abs(min(offsets)), abs(max(offsets)))
        self.is_dirichlet = [False, False]

    @classmethod
    def for_vector(cls, vector, scheme="center", derivative=1, order=2, residual=None, force_bandwidth=None):
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
        B.is_dirichlet = self.is_dirichlet
        return B


    def apply(self, V, overwrite=False):
        return (self.D.dot(V.flat) + self.R).reshape(V.shape)


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
            self.is_dirichlet[0] = True
            pass
        elif lower_type == 1:
            # Von Neumann boundary, we specify it directly.
            B.R[0] = lower_val
        elif lower_type is None and derivative == 1:
            # Free boundary
            # Second order forward approximation
            # XXX: This is dangerous! We can't do it if data is not wide enough
            assert m-2 >= 0, ("Not wide enough."
                    "\nB.data.shape = %s"
                    "\nB.derivative = %s"
                    "\nB.offsets = %s"
                    "\nm = %s"
                    "\nboundary = %s"
                    ) % (B.data.shape, B.derivative, B.offsets, m, boundary)
            B.data[m - 2, 2] = -d[1] / (d[2] * (d[1] + d[2]))
            B.data[m - 1, 1] = (d[1] + d[2]) / (d[1] * d[2])
            B.data[m,     0] = (-2 * d[1] - d[2]) / (d[1] * (d[1] + d[2]))
            # B.data[m, 0] = -1.0 / d[1]
            # B.data[m - 1, 1] = 1.0 / d[1]
            # print B.data
        elif lower_type is None and derivative == 2:
            # Extrapolate second derivative by assuming the first stays
            # constant.
            assert m-1 >= 0
            B.data[m-1, 1] =  2 / d[1]**2
            B.data[m,   0] = -2 / d[1]**2
            B.R[0]         =  2 / d[1]
        else:
            raise NotImplementedError("Can't handle derivatives higher than"
                                      " order 2 at boundaries. (%s)" % derivative)

        # Doing upper boundary
        if upper_type == 0:
            # Dirichlet boundary. No derivatives.
            self.is_dirichlet[1] = True
            pass
        elif upper_type == 1:
            # Von Neumann boundary, we specify it directly.
            B.R[-1] = upper_val
        elif upper_type is None and derivative == 1:
            # Second order backward approximation
            assert m+2 < B.data.shape[0]
            # XXX: This is dangerous! We can't do it if data is not wide enough
            B.data[m  , -1] = (d[-2]+2*d[-1])  / (d[-1]*(d[-2]+d[-1]))
            B.data[m+1, -2] = (-d[-2] - d[-1]) / (d[-2]*d[-1])
            B.data[m+2, -3] = d[-1]             / (d[-2]*(d[-2]+d[-1]))
            # First order backward
            # B.data[m, -1] = 1.0 / d[-1]
            # B.data[m + 1, -2] = -1.0 / d[-1]
        elif upper_type is None and derivative == 2:
            if B.R is None:
                B.R = np.zeros(B.data.shape[1])
            # Extrapolate second derivative by assuming the first stays
            # constant.
            assert m+1 < B.data.shape[0]
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


    def solve(self, V, overwrite=False):
        return spl.solve_banded(self.solve_banded_offsets,
                self.D.data, (V.flat - self.R),
                overwrite_ab=overwrite, overwrite_b=True).reshape(V.shape)

    @staticmethod
    def check_derivative(d):
        mixed = False
        try:
            d = tuple(d)
            if len(d) > 2:
                raise NotImplementedError, "Can't do more than 2nd order derivatives."
            if len(set(d)) != 1:
                mixed = True
                # #TODO
                # raise NotImplementedError, "Restricted to 2D problems without cross derivatives."
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


    def __getattr__(self, name):
        return self.D.__getattribute__(name)


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


    def splice_with(self, bottom, at, inplace=False):
        """
        Splice a second operator into this one by replacing rows after @at@.
        If inplace is True, splice it directly into this object.
        """
        newoffsets = sorted(set(self.offsets).union(set(bottom.offsets)), reverse=True)

        if inplace:
            if tuple(newoffsets) != tuple(self.offsets):
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
                B.D = bottom.D.copy()
                B.R = bottom.R.copy()
                B.order = bottom.order
                B.derivative = bottom.derivative
                B.deltas = bottom.deltas
                B.is_dirichlet = bottom.is_dirichlet
            else:
                B = bottom.copy()

        # If it's not extreme, it must be a normal splice
        else:
            if inplace:
                B = self
            else:
                newdata = np.zeros((len(newoffsets), self.data.shape[1]))
                B = BandedOperator((newdata, newoffsets), residual=self.R)

            last = B.shape[1]
            for torow, o in enumerate(B.offsets):
                splitidx = max(min(at+o, last), 0)
                if o in self.offsets:
                    fromrow = list(self.offsets).index(o)
                    dat = self.data[fromrow, :splitidx]
                else:
                    dat = 0
                B.data[torow, :splitidx] = dat
                if o in bottom.offsets:
                    fromrow = list(bottom.offsets).index(o)
                    dat = bottom.data[fromrow, splitidx:last]
                else:
                    dat = 0
                B.data[torow, splitidx:last] = dat

            # handle the residual vector
            if B.R is not None:
                B.R[splitidx:last] = bottom.R[splitidx:last]
            else:
                B.R = bottom.R.copy()
                B.R[:splitidx] = 0

            B.order = self.order
            B.derivative = self.derivative
            B.deltas = self.deltas
            B.is_dirichlet[0] = self.is_dirichlet[0]
            B.is_dirichlet[1] = bottom.is_dirichlet[1]
        return B


    def __mul__(self, val, inplace=False):
        return self.mul(val, inplace=inplace)
    def __imul__(self, val, inplace=True):
        return self.mul(val, inplace=inplace)

    def mul(self, val, inplace=False):
        if inplace:
            B = self
        else:
            B = self.copy()

        if not B.is_dirichlet[0]:
            B.data[0] *= val
            B.R[0] *= val
        if not B.is_dirichlet[1]:
            B.data[-1] *= val
            B.R[-1] *= val
        B.data[1:-1] *= val
        B.R[1:-1] *= val
        return B


    def __eq__(self, other):
        no_nan = np.nan_to_num
        # assert (self.data == other.data).all()
        # assert (self.offsets == other.offsets).all()
        # assert (self.shape == other.shape)
        # assert (no_nan(self.deltas) == no_nan(other.deltas)).all()
        # assert (self.order == other.order)
        # assert (self.derivative == other.derivative)
        # assert (self.R == other.R).all()
        try:
            return ((self.data == other.data).all()
                and (self.offsets == other.offsets).all()
                and (self.shape == other.shape)
                and (no_nan(self.deltas) == no_nan(other.deltas)).all()
                and (self.order == other.order)
                and (self.derivative == other.derivative)
                and (self.R == other.R).all())
        except:
            return False


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

    def vectorized_scale(self, vec):
        """
        @vec@ is the correpsonding mesh vector of the current dimension.

        Also applies to the residual vector self.R.

        See FiniteDifferenceEngine.coefficients.
        """
        bottom = 0
        top = last = self.shape[0]
        if self.is_dirichlet[0]:
            bottom += 1
        if self.is_dirichlet[1]:
            top -= 1
        for row, o in enumerate(self.offsets):
            if o > 0:
                self.data[row, bottom+o:] *= vec[bottom:last-o]
            elif o == 0:
                self.data[row, bottom:top] *= vec[bottom:top]
            elif o < 0:
                self.data[row, :top+o] *= vec[-o:top]
        self.R[bottom:top] *= vec[bottom:top]


    def scale(self, func):
        """
        func must be compatible with the following:
            func(x)
        Where x is the correpsonding index of the current dimension.

        Also applies to the residual vector self.R.

        See FiniteDifferenceEngine.coefficients.
        """
        bottom = 0
        top = self.shape[0]
        if self.is_dirichlet[0]:
            bottom += 1
        if self.is_dirichlet[1]:
            top -= 1
        for row, o in enumerate(self.offsets):
            if o > 0:
                for i in xrange(bottom, self.shape[0]-o):
                    self.data[row,i+o] *= func(i)
            elif o == 0:
                for i in xrange(bottom, top):
                    self.data[row,i] *= func(i)
            elif o < 0:
                for i in xrange(top):
                    self.data[row, i-abs(o)] *= func(i)

        for i in xrange(bottom, top):
            self.R[i] *= func(i)


    def compound_with(self, other):
        """
        It works the same was as scale and is used for making
        mixed derivatives.
        """
        offsets = self.offsets

        for row, o in enumerate(offsets):
            if o >= 0:
                for i in xrange(Bs.shape[0]-o):
                    print "(%i, %i)" % (row, i+o), "Block", i, i+o, "*",
                    Bs.data[row, i+o]
                    print data[row][i+o].data
                    data[row][i+o] *= Bs.data[row, i+o]
                    # print data[row][i+o].data
            else:
                for i in xrange(abs(o), Bs.shape[0]):
                    print "(%i, %i)" % (row, i-abs(o)), "Block", i, i-abs(o), "*", Bs.data[row, i-abs(o)]
                    data[row][i-abs(o)] *= Bs.data[row, i-abs(o)]
                    # print data[row][i-abs(o)].data
            # print


def flatten_tensor_aligned(mats, check=True):
    if check:
        assert len(set(tuple(m.offsets) for m in mats)) == 1
    residual = np.hstack([x.R for x in mats])
    diags = np.hstack([x.data for x in mats])
    return BandedOperator((diags, mats[0].offsets), residual=residual)



def flatten_tensor_misaligned(mats):
    offsets = set()
    offsets.update(*[set(tuple(m.offsets)) for m in mats])
    offsets = sorted(offsets, reverse=True)
    newlen = sum(len(m.data[0]) for m in mats)
    newdata = np.zeros((len(offsets), newlen))
    bottom = top = 0
    #TODO: Search sorted magic?
    for m in mats:
        top += len(m.data[0])
        for fro, o in enumerate(m.offsets):
            to = offsets.index(o)
            newdata[to, bottom:top] += m.data[fro]
        bottom = top
    residual = np.hstack([x.R for x in mats])
    flatmat = BandedOperator((newdata, offsets), residual=residual)
    return flatmat



def main():
    """Run main."""

    return 0

if __name__ == '__main__':
    main()
