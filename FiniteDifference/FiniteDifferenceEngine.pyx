# coding: utf8
# cython: profile=True
# cython: infer_types=True
# cython: annotate=True
"""<+Module Description.+>"""

# import sys
# import os
# import itertools as it

# TODO: This needs a partial redesign on how to handle boundary conditions.
# This just isn't flexible enough. We need to be able to do things like enforce
# 1st derivative boundaries when the is no first derivative operator, etc.

from bisect import bisect_left

import numpy as np
cimport numpy as np
import scipy.sparse
import itertools
import utils
import scipy.linalg as spl

from BandedOperator import BandedOperator

from visualize import fp

DEBUG = False


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
        closures. Weird things might happen.

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
        self.shape = grid.shape
        self.ndim = self.grid.ndim
        self.coefficients = coefficients
        self.boundaries = boundaries
        self.schemes = dict(schemes)
        self.force_bandwidth = force_bandwidth
        self.t = 0
        self.default_scheme = 'center'
        self.default_order = 2

        # Setup
        self.operators = {}
        self.simple_operators = {}


    def solve(self):
        """Run all the way to the terminal condition."""
        raise NotImplementedError


class FiniteDifferenceEngineADI(FiniteDifferenceEngine):

    def __init__(self, grid, coefficients={}, boundaries={}, schemes={},
            force_bandwidth=None):

        FiniteDifferenceEngine.__init__(self, grid, coefficients=coefficients,
                boundaries=boundaries, schemes=schemes, force_bandwidth=force_bandwidth)
        self.make_discrete_operators()

    def make_operator_template(self, d, dim, force_bandwidth=None):
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
                    force_bandwidth=force_bandwidth, axis=dim)
            if Binit is not None:
                if idx >= B.D.shape[0]-1:
                    raise ValueError("Cannot splice beyond the end of the "
                                     "vector. %i >= %i" % (idx, B.D.shape[0]-1))
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
        explain = False
        high = low = 0
        d = len(derivative_tuple)
        if derivative_tuple not in self.boundaries:
            raise ValueError("%s has no boundary specified." %
                    (derivative_tuple,))
        else:
            b = self.boundaries[derivative_tuple]
        for sd in self.schemes.get(derivative_tuple, ({},)): # a tuple of dicts
            # print "checking scheme from %s: %s" % (derivative_tuple, sd)
            s = sd.get('scheme', self.default_scheme)
            if sd.get('from', 0) == 0: # We're overwriting previous schemes
                # print "Reset!"
                high = low = 0
            if s == 'center' or '':
                if explain:
                    print "Center requires -1, 1"
                high = max(1, high)
                low = min(-1, low)
            elif s == 'forward' or s == 'up':
                if explain:
                    print "Forward requires 0, 2"
                high = max(2, high)
                low = min(0, low)
            elif s == 'backward' or s == 'down':
                if explain:
                    print "Backward requires -2, 0"
                high = max(0, high)
                low = min(-2, low)
        if b[0][0] is None:
            if d == 1:
                h = 2
            elif d == 2:
                if b[0][1](0,0) is not None:
                    h = 1
                else:
                    h = 2
            if explain:
                print ("Low free boundary requires forward"
                    " difference (%i): %i (have %i)" % (d, h, high))
            high = max(h, high)
        if b[1][0] is None:
            if d == 1:
                l = -2
            elif d == 2:
                if b[1][1](0,0) is not None:
                    l = -1
                else:
                    l = -2
            if explain:
                print ("High free boundary requires backward"
                       " difference (%i): %i (have %i)" % (d, l, low))
            low = min(l, low)
        if explain:
            print "FINAL: %s needs (%s, %s)" % (derivative_tuple, high, low)
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
        # to know anything other than the row it's working on.
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
            vec = f(self.t, *x)
            if np.isscalar(vec):
                vec += np.zeros_like(self.grid.mesh[dim])
            return vec

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
            Binit = self.make_operator_template(d, dim,
                    force_bandwidth=(low, high))
            assert Binit.axis == dim, "Binit.axis %s, dim %s" % (Binit.axis, dim)
            # if dim == 1:
                # print Binit.D.data
                # print
            offs = Binit.D.offsets
            assert max(offs) >= high, "(%s < %s)" % (max(offs), high)
            assert min(offs) <= low,  "(%s > %s)" % (min(offs), low)

            # Take cartesian product of other dimension values
            otherdims = range(ndim)
            otherdims.remove(dim)
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
                    # if DEBUG and B.axis == 1:
                        # from IPython.core.debugger import Tracer; Tracer()() ### XXX BREAKPOINT

                    B.applyboundary(b, self.grid.mesh)
                    # If we have a dirichlet boundary, save our function
                    if b[0][0] == 0:
                        B.dirichlet[0] =  b[0][1]
                    if b[1][0] == 0:
                        B.dirichlet[1] =  b[1][1]


                    # If it's a dirichlet boundary we mark it because we have to
                    # handle it absolutely last, it doesn't get scaled.
                    # lf = hf = None
                    # if B.dirichlet[0] is not None:
                        # lf = lowfunc
                    # if B.dirichlet[1] is not None:
                        # hf = highfunc
                    # if lf or hf:
                        # vals = dirichlets.setdefault(dim, [])
                        # vals.append((lval, hval))
                else:
                    raise ValueError("%s has no boundary condition specified." % d)

                # Give the operator the right coefficient
                # Here we wrap the functions with the appropriate values for
                # this particular dimension.
                if d in coeffs:
                    B.vectorized_scale(evalvectorfunc(coeffs[d], a, dim))
                Bs.append(B)
            self.simple_operators[d] = flatten_tensor_aligned(Bs, check=False)


            # Combine scaled derivatives for this dimension
            if dim not in self.operators:
                self.operators[dim] = Bs
            else:
                ops = self.operators[dim] # list reference
                for col, b in enumerate(Bs):
                    assert b.axis == dim
                    if tuple(ops[col].D.offsets) == tuple(b.D.offsets):
                        ops[col] += b
                    else:
                        # print col, dim, ops[col].axis, b.axis
                        ops[col] = ops[col] + b

        # Now the 0th derivative (x * V)
        # We split this evenly among each dimension
        #TODO: This function is ONLY dependent on time. NOT MESH
        # Also, this gets plowed during the solve steps if we have dirichlet
        # boundaries
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
            # TODO: We'll need to do complicated transposing for this in the
            # general case
            Bs = BandedOperator.for_vector(self.grid.mesh[0], derivative=1, axis=0)
            Bm1 = BandedOperator.for_vector(self.grid.mesh[1], derivative=1, axis=1)
            Bb1 = Bm1.copy()
            Bp1 = Bm1.copy()

            # TODO: Hardcoding in for centered differencing
            Bps = [Bp1 * 0, Bp1 * 0] + replicate(d0_size-2, Bp1)
            Bbs = [Bb1 * 0] + replicate(d0_size-2, Bb1) +  [Bb1 * 0]
            Bms = replicate(d0_size-2, Bm1) + [Bm1 * 0, Bm1 * 0]

            offsets = Bs.D.offsets
            data = [Bps, Bbs, Bms]
            for row, o in enumerate(offsets):
                if o >= 0:
                    for i in range(Bs.shape[0]-o):
                        # print "(%i, %i)" % (row, i+o), "Block", i, i+o, "*", Bs.data[row, i+o]
                        a = (np.array(self.grid.mesh[0][i]).repeat(d1_size),)
                        vec = evalvectorfunc(coeffs[d], a, 1)
                        # print "= high"
                        # fp(data[row][i+o])
                        data[row][i+o].vectorized_scale(vec)
                        # fp(data[row][i+o])
                        # print "="
                        data[row][i+o] *= Bs.D.data[row, i+o]
                else:
                    for i in range(abs(o), Bs.shape[0]):
                        # print "(%i, %i)" % (row, i-abs(o)), "Block", i, i-abs(o), "*", Bs.data[row, i-abs(o)]
                        a = (np.array(self.grid.mesh[0][i]).repeat(d1_size),)
                        vec = evalvectorfunc(coeffs[d], a, 1)
                        # print "= low"
                        # fp(data[row][i-abs(o)])
                        data[row][i-abs(o)].vectorized_scale(vec)
                        # fp(data[row][i-abs(o)])
                        # print "="
                        data[row][i-abs(o)] *= Bs.D.data[row, i-abs(o)]


            # We flatten here because it's faster
            Bps[0].D.offsets += d1_size
            Bms[0].D.offsets -= d1_size
            BP = flatten_tensor_aligned(Bps, check=False)
            BB = flatten_tensor_aligned(Bbs, check=False)
            BM = flatten_tensor_aligned(Bms, check=False)
            self.operators[d] = BP+BB+BM

        # Dirichlet boundaries are handled in the apply and solve methods.

        # # Now everything else is done we can apply dirichlet boundaries
        # for (dim, funcs) in dirichlets.items():
            # # For each operator in this dimension
            # for i, B in enumerate(self.operators[dim]):
                # m = tuple(B.D.offsets).index(0)
                # lowfunc, highfunc = funcs[i]
                # if lowfunc:
                    # # Cancel out the old value
                    # # B.D.data[m, 0] = -1
                    # # Replace with the new one
                    # B.R[0] = lowfunc(0)
                # if highfunc:
                    # # B.D.data[m, -1] = -1
                    # B.R[-1] = highfunc(-1)

        # Flatten into one large operator for speed
        # We'll apply this to a flattened domain and then reshape it.
        for d in self.operators.keys():
            if isinstance(self.operators[d], list):
                self.operators[d] = flatten_tensor_misaligned(self.operators[d])
        return

    def cross_term(self, V, numpy=True):
        if (0,1) in self.coefficients:
            if numpy:
                x = self.grid.mesh[0]
                y = self.grid.mesh[1]
                dx = np.gradient(x)[:,np.newaxis]
                dy = np.gradient(y)
                Y, X = np.meshgrid(y, x)
                gradgrid = self.coefficients[(0,1)](0, X, Y) / (dx * dy)
                gradgrid[:,0] = 0; gradgrid[:,-1] = 0
                gradgrid[0,:] = 0; gradgrid[-1,:] = 0
                ret = np.gradient(np.gradient(V)[0])[1] * gradgrid
            else:
                ret = self.operators[(0,1)].apply(V)
        else:
            ret = 0
        return ret

    def solve_implicit2(self, n, dt, initial=None, callback=None, numpy=False):
        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[0].copy()
            self.grid.domain.append(V)

        Lis = [(o * -dt).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("solve_implicit:\t")
        for k in range(n):
            if not k % print_step:
                if np.isnan(V).any():
                    print "Impl fail @ t = %f (%i steps)" % (dt * k, k)
                    return V
                print int(k * to_percent),
            if callback is not None:
                callback(V, ((n - k) * dt))
            V += self.cross_term(V, numpy=numpy) * dt
            for L in Lis:
                V = L.solve(V)
        self.grid.domain.append(V.copy())
        utils.toc(':  \t')
        return V

    def solve_implicit(self, n, dt, initial=None, callback=None, numpy=False):
        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[0].copy()
            self.grid.domain.append(V)

        # for d, o in self.operators.items():
            # print d
            # fp(o)
            # print

        Lis = [(o * -dt).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        Lis = np.roll(Lis, -1)

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("solve_implicit:\t")
        for k in range(n):
            if not k % print_step:
                if np.isnan(V).any():
                    print "Impl fail @ t = %f (%i steps)" % (dt * k, k)
                    return V
                print int(k * to_percent),
            if callback is not None:
                callback(V, ((n - k) * dt))
            V += self.cross_term(V, numpy=numpy) * dt
            for L in Lis:
                V = L.solve(V)
            self.grid.domain.append(V.copy())
        utils.toc(':  \t')
        return V


    def solve_hundsdorferverwer(self, n, dt, initial=None, theta=0.5, callback=None,
            numpy=False):

        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[0].copy()
            self.grid.domain.append(V)

        Firsts = [(o * dt) for o in self.operators.values()]

        Les = [(o * theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [(o * (theta * -dt)).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        for L in itertools.chain(Les, Lis):
            L.R = None

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("Hundsdorfer-Verwer:\t")
        for k in range(n):
            if not k % print_step:
                if np.isnan(V).any():
                    print "Hundsdorfer-Verwer fail @ t = %f (%i steps)" % (dt * k, k)
                    return V
                print int(k * to_percent),
            if callback is not None:
                callback(V, ((n - k) * dt))

            Y = V.copy()
            for L in Firsts:
                Y += L.apply(V)
            Y0 = Y.copy()

            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)
            Y2 = Y.copy()

            Y = Y0
            for L in Firsts:
                no_residual = L.R
                L.R = None
                Y += 0.5 * L.apply(Y2-V)
                L.R = no_residual

            V = Y2

            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)

            V = Y

            self.grid.domain.append(V.copy())
        utils.toc(':  \t')
        return V


    def solve_craigsneyd2(self, n, dt, initial=None, theta=0.5, callback=None,
            numpy=False):

        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[0].copy()
            self.grid.domain.append(V)

        Firsts = [(o * dt) for o in self.operators.values()]

        Les = [(o * theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [(o * (theta * -dt)).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        for L in itertools.chain(Les, Lis):
            L.R = None


        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("Craig-Sneyd 2:\t")
        for k in range(n):
            if not k % print_step:
                if np.isnan(V).any():
                    print "Craig-Sneyd 2 fail @ t = %f (%i steps)" % (dt * k, k)
                    return V
                print int(k * to_percent),
            if callback is not None:
                callback(V, ((n - k) * dt))

            Y = V.copy()
            for L in Firsts:
                Y += L.apply(V)
            Y0 = Y.copy()

            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)
            Y2 = Y.copy()

            Y = Y0 + theta * dt * self.cross_term(Y2 - V, numpy=False)
            for L in Firsts:
                Y += (0.5 - theta) * L.apply(Y2-V)

            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)
            V = Y

            self.grid.domain.append(V.copy())
        utils.toc(':  \t')
        return V


    def solve_craigsneyd(self, n, dt, initial=None, theta=0.5, callback=None,
            numpy=False):

        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[0].copy()
            self.grid.domain.append(V)

        Firsts = [(o * dt) for o in self.operators.values()]

        Les = [(o * theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [(o * (theta * -dt)).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        for L in itertools.chain(Les, Lis):
            L.R = None

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("Craig-Sneyd:\t")
        for k in range(n):
            if not k % print_step:
                if np.isnan(V).any():
                    print "Craig-Sneyd fail @ t = %f (%i steps)" % (dt * k, k)
                    return V
                print int(k * to_percent),
            if callback is not None:
                callback(V, ((n - k) * dt))

            Y = V.copy()
            for L in Firsts:
                Y += L.apply(V)
            Y0 = Y.copy()

            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)

            Y = Y0 + 0.5 *dt * self.cross_term(Y - V, numpy=False)
            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)
            V = Y

            self.grid.domain.append(V.copy())
        utils.toc(':  \t')
        return V


    def solve_douglas(self, n, dt, initial=None, theta=0.5, callback=None,
            numpy=False):

        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[0].copy()
            self.grid.domain.append(V)

        Firsts = [(o * dt) for d, o in self.operators.items()]

        Les = [(o * theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [(o * (theta * -dt)).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        for L in itertools.chain(Les, Lis):
            L.R = None

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("Douglas:\t")
        for k in range(n):
            if not k % print_step:
                if np.isnan(V).any():
                    print "Douglas fail @ t = %f (%i steps)" % (dt * k, k)
                    return V
                print int(k * to_percent),
            if callback is not None:
                callback(V, ((n - k) * dt))

            Y = V.copy()
            for L in Firsts:
                Y += L.apply(V)

            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)
            V = Y

            self.grid.domain.append(V.copy())
        utils.toc(':  \t')
        return V


    def solve_john_adi(self, n, dt, initial=None, callback=None, numpy=False):
        #TODO: I don't think this satisfies... well.. anything.
        theta = 0.5
        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[0].copy()
            self.grid.domain.append(V)

        Les = [(o * ((1-theta)*dt)).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [(o * (-theta*dt)).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        # Have to roll the first one because of the scheme
        # TODO: What the hell does that mean? Explain it.
        Les = np.roll(Les, -1)

        Leis = zip(Les, Lis)

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("solve_john_adi:\t")
        for k in range(n):
            if not k % print_step:
                if np.isnan(V).any():
                    print "solve_john_adi fail @ t = %f (%i steps)" % (dt * k, k)
                    return V
                print int(k * to_percent),
            if callback is not None:
                callback(V, ((n - k) * dt))
            for Le, Li in Leis:
                V = Le.apply(V)
                V += self.cross_term(V, numpy=numpy) * 0.5 * dt
                V = Li.solve(V)
            self.grid.domain.append(V.copy())
        utils.toc(':  \t')
        return V


    def smooth(self, n, dt, initial=None, callback=None, smoothing_steps=2,
            scheme=None):
        if scheme is None:
            scheme = self.solve_douglas
        V = self.solve_implicit(smoothing_steps*2, dt*0.5, initial=initial)
        return scheme(n-smoothing_steps, dt, initial=V)


def replicate(n, x):
    ret = []
    for _ in range(n):
        ret.append(x.copy())
    return ret



def flatten_tensor_aligned(mats, check=True):
    if check:
        assert len(set(tuple(m.D.offsets) for m in mats)) == 1
    residual = np.hstack([x.R for x in mats])
    diags = np.hstack([x.D.data for x in mats])
    B = BandedOperator((diags, mats[0].D.offsets), residual=residual)
    B.copy_meta_data(mats[0], dirichlet=mats[0].dirichlet, derivative=None)
    B.dirichlet = list(zip(*(m.dirichlet for m in mats)))
    for i, bound in enumerate(mats[0].dirichlet):
        if bound is None:
            B.dirichlet[i] = None
    B.blocks = len(mats)
    return B


def flatten_tensor_misaligned(mats):
    offsets = set()
    offsets.update(*[set(tuple(m.D.offsets)) for m in mats])
    offsets = sorted(offsets, reverse=True)
    newlen = sum(len(m.D.data[0]) for m in mats)
    newdata = np.zeros((len(offsets), newlen))
    begin = end = 0
    #TODO: Search sorted magic?
    for m in mats:
        end += len(m.D.data[0])
        for fro, o in enumerate(m.D.offsets):
            to = offsets.index(o)
            newdata[to, begin:end] += m.D.data[fro]
        begin = end
    residual = np.hstack([x.R for x in mats])
    B = BandedOperator((newdata, offsets), residual=residual)
    B.copy_meta_data(mats[0])
    B.dirichlet = list(zip(*(m.dirichlet for m in mats)))
    for i, bound in enumerate(mats[0].dirichlet):
        if bound is None:
            B.dirichlet[i] = None
        else:
            B.dirichlet[i] = np.array(B.dirichlet[i])
    B.blocks = len(mats)
    return B



def main():
    """Run main."""

    return 0

if __name__ == '__main__':
    main()
