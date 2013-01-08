# coding: utf8
# cython: profile=True
# cython: infer_types=True
# cython: annotate=True
"""<+Module Description.+>"""

# import sys
# import os
# import itertools as it

# TODO: This needs a partial redesign on how to handle boundary conditions.
# This just isn't flexible enough.

from types import MethodType

from bisect import bisect_left

import sys

import numpy as np
cimport numpy as np
import scipy.sparse
import itertools
import utils
import scipy.linalg as spl

import itertools as it

# import BandedOperator as BO
# BandedOperator = BO.BandedOperator
import BandedOperatorGPU as BO
BandedOperator = BO.BandedOperator

from visualize import fp

DEBUG = False

from Option import Option

cdef class FiniteDifferenceEngine(object):
    cdef public shape
    cdef public ndim
    cdef public coefficients
    cdef public boundaries
    cdef public schemes
    cdef public t
    cdef public default_scheme
    cdef public default_order
    cdef public grid

    # Setup
    cdef public operators
    cdef public simple_operators

    def __init__(self, grid, coefficients={}, boundaries={}, schemes={}):
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
        self.t = 0
        self.default_scheme = 'center'
        self.default_order = 2

        # Setup
        self.operators = {}
        self.simple_operators = {}



    def solve(self):
        """Run all the way to the terminal condition."""
        raise NotImplementedError


class FiniteDifferenceEngineADE(FiniteDifferenceEngine):

    def __init__(self, grid, coefficients={}, boundaries={}, schemes={}):
        self.grid = grid
        self.coefficients = coefficients
        self.boundaries = boundaries
        self.schemes = schemes
        self._initialized = False
        self.mu = 1
        self.gamma2 = 1
        self.r = 0
        FiniteDifferenceEngine.__init__(self, self.grid, coefficients=self.coefficients,
                boundaries=self.boundaries, schemes=self.schemes)


    def initialized(f):
        def newf(self, *args, **kwargs):
            if not self._initialized:
                self._initialized = True
            return f(self, *args, **kwargs)
        return newf


    def solve_exp(self, n, dt, bfunc):
        """What is this doing? Who knows!? Not you."""
        # n = int(n)
        # dx = self.grid.dx[0][1]
        # mu = self.mu / (2*dx)
        # gamma2 = self.gamma2 / (dx*dx)
        # r = self.r
        # v = self.grid.domain[-1].copy()
        # count = 0
        # for t in range(1, n+1, 1):
            # vm = v
            # v = np.zeros_like(v)
            # for i in range(1, v.shape[0]-1):
                # # for j in range(1, v.shape[1]-1):
                # v[i] = vm[i] + dt * (
                      # gamma2[i] * (vm[i-1] - 2*vm[i] + vm[i+1])
                    # + mu[i]     * (vm[i+1] - vm[i-1])
                    # - r * vm[i]
                # )
            # # duds = mu[-1] * self.grid.mesh[0][-1]
            # # d2uds = gamma2[-1]*(self.grid.mesh[0][-1]*2*dx + 2*vm[-2] - 2*vm[-1])
            # # v[-1] = vm[-1] + dt*(duds + d2uds + -r*vm[-1])
            # v[0] = bfunc(t, 0)
            # v[-1] = bfunc(t, -1)
            # self.grid.domain.append(v)
        # return v
        pass


    def solve(self, n, dt):
        n = int(n)
        cdef double dx = self.grid.dx[0][1]
        cdef double[:] mu = self.mu / (2*dx)
        cdef double[:] gamma2 = self.gamma2 / (dx*dx)
        cdef double r = self.r
        cdef double [:] v
        cdef int i, t
        for t in range(1, n+1):
            # Loop over interior
            vm = self.grid.domain[-1]
            v = np.zeros_like(vm, dtype=float)
            for i in range(v.shape[0]-2, 0, -1):
                v[i] =((
                    dt * (gamma2[i] + mu[i]) * v[i+1]
                    + dt * (gamma2[i] - mu[i]) * vm[i-1]
                    + (1 - dt * r - dt * gamma2[i] + dt * mu[i]) *vm[i])
                / (1 + dt * (gamma2[i] + mu[i]))
            )
            duds = mu[-1] * self.grid.mesh[0][-1]
            d2uds = gamma2[-1]*(self.grid.mesh[0][-1]*2*dx + 2*vm[-2] - 2*vm[-1])
            v[-1] = vm[-1] + dt*(duds + d2uds + -r*vm[-1])
            v1 = np.asarray(v)

            # Loop over interior
            v = np.zeros_like(vm, dtype=float)
            duds = mu[-1] * self.grid.mesh[0][-1]
            d2uds = gamma2[-1]*(self.grid.mesh[0][-1]*2*dx + 2*vm[-2] - 2*vm[-1])
            v[-1] = vm[-1] + dt*(duds + d2uds + -r*vm[-1])
            for i in range(1, v.shape[0]-1):
                v[i] = ((vm[i]
                        - dt * r*vm[i]
                        - dt * mu[i]     * (v[i-1] + vm[i] - vm[i+1])
                        + dt * gamma2[i] * (v[i-1] - vm[i] + vm[i+1]))
                    / (1 + dt*gamma2[i] - dt*mu[i]))
            v2 = np.asarray(v)
            v = (v1+v2)/2
            self.grid.domain.append(v)
        return self.grid.domain[-1]

def initialized(f):
    """Create compound operators and initialize the underlying FiniteDifferenceEngine.
    Under normal circumstances, this is called automatically. If you want to
    access the operators before they are used, you must call this yourself to
    create them."""
    def newf(self, *args, **kwargs):
        if not self._initialized:
            FiniteDifferenceEngine.__init__(self, self.grid, coefficients=self.coefficients,
                    boundaries=self.boundaries, schemes=self.schemes)
            self.make_discrete_operators()
            self._initialized = True
        return f(self, *args, **kwargs)
    newf.__name__ = f.__name__
    return newf

cdef class FiniteDifferenceEngineADI(FiniteDifferenceEngine):

    cdef public force_bandwidth
    cdef public _initialized

    def __init__(self, grid, coefficients={}, boundaries={}, schemes={},
            force_bandwidth=None):
        self.grid = grid
        self.coefficients = coefficients
        self.boundaries = boundaries
        self.schemes = schemes
        self.force_bandwidth = force_bandwidth
        self._initialized = False


    @initialized
    def init(self):
        return

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



    # We partially apply the function to all of the values except this
    # dimension, and change the input to accept an index into mesh
    # rather than the mesh value itself. This way the caller doesn't need
    # to know anything other than the row it's working on.
    # f : (t x R^n -> R) -> Z -> R
    def wrapscalarfunc(self, f, args, dim):
        x = list(args)
        x.insert(dim, None)
        def newf(i):
            x[dim] = self.grid.mesh[dim][i]
            return f(self.t, *x)
        return newf

    # Here we do the same except we go ahead and evalutate the function for
    # the entire vector. This is just for numpy's speed and is otherwise
    # redundant.
    def evalvectorfunc(self, f, args, dim):
        x = list(args)
        x.insert(dim, self.grid.mesh[dim])
        vec = f(self.t, *x)
        if np.isscalar(vec):
            vec += np.zeros_like(self.grid.mesh[dim])
        return vec


    def coefficient_vector(self, f, t, dims):
        """Evaluate f with the cartesian product of the elements of
        self.grid.mesh, ordered by dims. The first dim is the fastest varying
        (column major).
        Example (actual implementation is vectorized):
            mesh = [(1,2,3), (4,5,6), (7,8,9)]
            dims = (2,0,1)
            output = [f(a,b,c)
                        for b in mesh[1]
                        for a in mesh[0]
                        for c in mesh[2]
                        ]
        """
        gridsize = self.grid.size
        mesh = self.grid.mesh[dims]
        # This can be rewritten with repeat and tile, not sure if faster
        args = np.fromiter(it.chain(*it.izip(*it.product(*mesh))), float).reshape(mesh.shape[0], gridsize)
        ret = f(t, *iter(args))
        return ret


    def make_operator_templates(self, force_bandwidth):
        templates = {}
        mixed = {}
        # d = (0,0), for example ...
        for d in self.coefficients.keys():
            # Don't need an operator for the 0th derivative
            if d == ():
                continue

            # Mixed derivatives are handled specially
            mix = BandedOperator.check_derivative(d)
            if mix:
                mixed[d] = True
                continue

            dim = d[0]

            # Make an operator template for this dimension
            low, high = self.min_possible_bandwidth(d)
            bw = force_bandwidth
            # print "Minimum bandwidth for %s: %s" % (d, (low, high))
            if bw:
                if (bw[0] > low or bw[1] < high):
                    raise ValueError("Your chosen scheme is too wide for the"
                            " specified bandwidth. (%s needs %s)" %
                            (bw, (low, high)))
                low, high = bw
            Binit = self.make_operator_template(d, dim,
                                                force_bandwidth=(low, high))
            assert Binit.axis == dim, "Binit.axis %s, dim %s" % (Binit.axis, dim)
            offs = Binit.D.offsets
            assert max(offs) >= high, "(%s < %s)" % (max(offs), high)
            assert min(offs) <= low,  "(%s > %s)" % (min(offs), low)
            templates[d] = Binit
        return templates, mixed


    def make_discrete_operators(self):
        ndim = self.grid.ndim
        coeffs = self.coefficients
        bounds = self.boundaries
        force_bandwidth = self.force_bandwidth
        dirichlets = {}
        templates, mixed = self.make_operator_templates(force_bandwidth)

        for d, Binit in templates.items():
            dim = d[0]
            ColumnOperators = []
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
                    lowfunc  = self.wrapscalarfunc(b[0][1], a, dim)
                    highfunc = self.wrapscalarfunc(b[1][1], a, dim)
                    b = ((b[0][0], lowfunc(0)), (b[1][0], highfunc(-1)))

                    B.applyboundary(b, self.grid.mesh)
                    # If we have a dirichlet boundary, save our function
                    if b[0][0] == 0:
                        B.dirichlet[0] =  b[0][1]
                    if b[1][0] == 0:
                        B.dirichlet[1] =  b[1][1]



            # Give the operator the right coefficient
            # Here we wrap the functions with the appropriate values for
            # this particular dimension.
            # for a in argset:
                if d in coeffs:
                    B.vectorized_scale(self.evalvectorfunc(coeffs[d], a, dim))
                ColumnOperators.append(B)
            self.simple_operators[d] = flatten_tensor_aligned(ColumnOperators, check=False)


            # Combine scaled derivatives for this dimension
            if dim not in self.operators:
                self.operators[dim] = ColumnOperators
            else:
                ops = self.operators[dim] # list reference
                for col, b in enumerate(ColumnOperators):
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
            for ColumnOperators in self.operators.values():
                for B in ColumnOperators:
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
            # ColumnOperators = self.make_operator_template(d, dim=d[0])
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
                        vec = self.evalvectorfunc(coeffs[d], a, 1)
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
                        vec = self.evalvectorfunc(coeffs[d], a, 1)
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

        # Flatten into one large operator for speed
        # The apply and solve methods will flatten the domain (V.flat) and then
        # reshape it.
        for d in self.operators.keys():
            if isinstance(self.operators[d], list):
                self.operators[d] = flatten_tensor_misaligned(self.operators[d])
        return


    def cross_term(self, V, numpy=True):
        """Apply the cross derivative operator, either using Numpy's gradient()
        or our own matrix."""
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

    @initialized
    def solve_implicit(self, n, dt, initial=None, callback=None, numpy=False):
        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[-1].copy()
            self.grid.domain.append(V)

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
                sys.stdout.flush()
            if callback is not None:
                callback(V, ((n - k) * dt))
            V += self.cross_term(V, numpy=numpy) * dt
            for L in Lis:
                V = L.solve(V)
        utils.toc(':  \t')
        self.grid.domain.append(V.copy())
        return V

    @initialized
    def solve_explicit(self, n, dt, initial=None, callback=None, numpy=False):
        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[-1].copy()
            self.grid.domain.append(V)

        Ls = [(o * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]


        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("solve_explicit:\t")
        for k in range(n):
            if not k % print_step:
                if np.isnan(V).any():
                    print "Expl fail @ t = %f (%i steps)" % (dt * k, k)
                    return V
                print int(k * to_percent),
                sys.stdout.flush()
            if callback is not None:
                callback(V, ((n - k) * dt))
            V += self.cross_term(V, numpy=numpy) * dt
            for L in Ls:
                V += L.apply(V)
        utils.toc(':  \t')
        self.grid.domain.append(V.copy())
        return V


    @initialized
    def solve_hundsdorferverwer(self, n, dt, initial=None, theta=0.5, callback=None,
            numpy=False):

        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[-1].copy()
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
                sys.stdout.flush()
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

        utils.toc(':  \t')
        self.grid.domain.append(V.copy())
        return V


    @initialized
    def solve_craigsneyd2(self, n, dt, initial=None, theta=0.5, callback=None,
            numpy=False):

        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[-1].copy()
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
                sys.stdout.flush()
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

        utils.toc(':  \t')
        self.grid.domain.append(V.copy())
        return V


    @initialized
    def solve_craigsneyd(self, n, dt, initial=None, theta=0.5, callback=None,
            numpy=False):

        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[-1].copy()
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
                sys.stdout.flush()
            if callback is not None:
                callback(V, ((n - k) * dt))

            Y = V.copy()
            for L in Firsts:
                Y += L.apply(V)
            Y0 = Y.copy()

            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)

            Y = Y0 + (0.5*dt) * self.cross_term(Y - V, numpy=False)
            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)
            V = Y

        utils.toc(':  \t')
        self.grid.domain.append(V.copy())
        return V


    @initialized
    def solve_douglas(self, n, dt, initial=None, theta=0.5, callback=None,
            numpy=False):

        n = int(n)
        if initial is not None:
            V = initial.copy()
        else:
            V = self.grid.domain[-1].copy()
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
                sys.stdout.flush()
            if callback is not None:
                callback(V, ((n - k) * dt))

            Y = V.copy()
            for L in Firsts:
                Y += L.apply(V)

            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)
            V = Y

        utils.toc(':  \t')
        self.grid.domain.append(V.copy())
        return V


    @initialized
    def solve_smooth(self, n, dt, initial=None, callback=None, smoothing_steps=2,
            scheme=None):
        if scheme is None:
            scheme = self.solve_hundsdorferverwer
        V = self.solve_implicit(smoothing_steps*2, dt*0.5, initial=initial)
        # V = self.solve_douglas(smoothing_steps*2, dt*0.5, theta=1, initial=initial)
        return scheme(n-smoothing_steps, dt, initial=V, theta=0.60)


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


if __name__ == '__main__':
    print "This is just a library."
