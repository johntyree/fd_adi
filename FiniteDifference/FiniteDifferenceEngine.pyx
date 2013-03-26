# coding: utf8
# cython: annotate=True
# cython: infer_types=True
# distutils: language = c++


import sys
import itertools

import numpy as np
cimport numpy as np
import scipy.sparse
import scipy.linalg as spl


import FiniteDifference.utils as utils

import FiniteDifference.BandedOperator as BO
cimport FiniteDifference.BandedOperator as BO
BandedOperator = BO.BandedOperator

from FiniteDifference.visualize import fp

from FiniteDifference.Option import Option


DEBUG = False

REAL = np.float64
ctypedef np.float64_t REAL_t


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


cdef class FiniteDifferenceEngine(object):

    cdef public:
        boundaries
        coefficients
        default_order
        default_scheme
        grid
        ndim
        operators
        schemes
        shape
        simple_operators
        t


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


cdef class FiniteDifferenceEngineADI(FiniteDifferenceEngine):

    cdef public:
        force_bandwidth
        _initialized


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
            B = BO.for_vector(self.grid.mesh[dim], s, len(d), o, None, force_bandwidth, dim)
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
                h = 1
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
                l = -1
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
            vec = f(self.t, *x)
            if np.isscalar(vec):
                vec = float(vec)
            elif vec is not None and vec.dtype != 'float64':
                vec = vec.astype('float64')
            return vec
        newf.__name__ = f.__name__
        return newf


    # Here we do the same except we go ahead and evalutate the function for
    # the entire vector. This is just for numpy's speed and is otherwise
    # redundant.
    def evalvectorfunc(self, f, args, dim):
        x = list(args)
        x.insert(dim, self.grid.mesh[dim])
        vec = f(self.t, *x)
        if np.isscalar(vec):
            vec += np.zeros_like(self.grid.mesh[dim]).astype(float)
        if vec.dtype != 'float64':
            vec = vec.astype('float64')
        return vec


    def coefficient_vector(self, f, t, dim):
        """Evaluate f with the cartesian product of the elements of
        self.grid.mesh, ordered such that dim is the fastest varying. The
        relative order of the other dimensions remains the same.

        Example (actual implementation is vectorized):
            mesh = [(1,2,3), (4,5,6), (7,8,9)]
            dim = 1
            newmesh = [(4,5,6), (1,2,3), (7,8,9)]
            [f(4,1,7), f(5,1,7), ..., f(4,2,7), f(5,2,7), ..., f(5,3,9), f(6,3,9)]
            output = [f(a,b,c)
                        for b in mesh[1]
                        for a in mesh[0]
                        for c in mesh[2]
                        ]
        """
        gridsize = self.grid.size
        mesh = list(self.grid.mesh)
        m = mesh.pop(dim)
        mesh.append(m)
        # This can be rewritten with repeat and tile, not sure if faster
        args = np.fromiter(
            itertools.chain(*itertools.izip(*itertools.product(*mesh))), float)
        args = np.split(args, self.grid.ndim)
        m = args.pop()
        args.insert(dim, m)
        ret = f(t, *iter(args))
        if np.isscalar(ret):
            ret = np.repeat(<float>ret, gridsize)
        return ret


    # TODO: This guy is *WAY* too big
    def make_discrete_operators(self):
        templates = {}
        mixed_derivs = {}
        coeffs = self.coefficients
        bounds = self.boundaries
        force_bandwidth = self.force_bandwidth
        # d = (0,0), for example ...
        for d in coeffs.keys():
            # Don't need an operator for the 0th derivative
            if d == ():
                continue

            # Mixed derivatives are handled specially
            mix = BO.check_derivative(d)
            if mix:
                mixed_derivs[d] = True
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
            B = self.make_operator_template(d, dim,
                                            force_bandwidth=(low, high))
            assert B.axis == dim, "B.axis %s, dim %s" % (B.axis, dim)
            offs = B.D.offsets
            assert max(offs) >= high, "(%s < %s)" % (max(offs), high)
            assert min(offs) <= low,  "(%s > %s)" % (min(offs), low)

            # Adjust the boundary conditions as necessary
            lowbound = []
            highbound = []
            if d in bounds:
                # Take cartesian product of other dimension values
                otherdims = range(self.grid.ndim)
                otherdims.remove(B.axis)
                argset = itertools.product(*(self.grid.mesh[i] for i in otherdims))

                # Pair our current dimension with all combinations of the other
                # dimension values
                for a in argset:
                    b = bounds[d]
                    lowfunc  = self.wrapscalarfunc(b[0][1], a, dim)
                    highfunc = self.wrapscalarfunc(b[1][1], a, dim)
                    b = ((b[0][0], lowfunc(0)), (b[1][0], highfunc(-1)))

                    B.applyboundary(b, self.grid.mesh)
                    # If we have a dirichlet boundary, save our function
                    if b[0][0] == 0:
                        lowbound.append(b[0][1])
                    if b[1][0] == 0:
                        highbound.append(b[1][1])

            Bs = replicate(self.grid.size / self.grid.shape[B.axis], B)
            Bs = flatten_tensor_aligned(Bs)
            if lowbound:
                Bs.dirichlet[0] = tuple(lowbound) if len(lowbound) > 1 else lowbound[0]
            if highbound:
                Bs.dirichlet[1] = tuple(highbound) if len(highbound) > 1 else highbound[0]
            # if not Bs.is_tridiagonal():
                # Bs.diagonalize()
            templates[d] = Bs

        # TODO
        # This expects only 2 dimensions
        self.check_mixed_derivative_parameters(mixed_derivs.keys())
        for d in mixed_derivs.keys():
            d0_size = len(self.grid.mesh[d[0]])
            d1_size = len(self.grid.mesh[d[1]])

            # TODO: We'll need to do complicated transposing for this in the
            # general case
            Bs = BO.for_vector(self.grid.mesh[d[0]], "center", 1, 2, None, None, 0)
            Bm1 = BO.for_vector(self.grid.mesh[d[1]], "center", 1, 2, None, None, 1)
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
                        # a = (np.array(self.grid.mesh[d[0]][i]).repeat(d1_size),)
                        # vec = self.evalvectorfunc(coeffs[d], a, 1)
                        # data[row][i+o].vectorized_scale(vec)
                        data[row][i+o] *= Bs.D.data[row, i+o]
                else:
                    for i in range(abs(o), Bs.shape[0]):
                        # a = (np.array(self.grid.mesh[d[0]][i]).repeat(d1_size),)
                        # vec = self.evalvectorfunc(coeffs[d], a, 1)
                        # data[row][i-abs(o)].vectorized_scale(vec)
                        data[row][i-abs(o)] *= Bs.D.data[row, i-abs(o)]

            # We flatten here because it's faster
            # Check is set to False because we're only faking that the offsets.
            # The resulting operator will take the offsets from only the first
            # in the list.
            Bps[0].D.offsets += d1_size
            Bms[0].D.offsets -= d1_size
            BP = flatten_tensor_aligned(Bps, check=False)
            BB = flatten_tensor_aligned(Bbs, check=False)
            BM = flatten_tensor_aligned(Bms, check=False)
            templates[d] = BP + BM + BB
            templates[d].is_mixed_derivative = True
        self.simple_operators = templates
        self.scale_and_combine_operators()
        return mixed_derivs


    def check_mixed_derivative_parameters(self, mds):
        for d in mds:
            if len(d) > 2:
                raise NotImplementedError("Derivatives must be 2nd order or less.")
            # Just to make sure
            for sd in self.schemes.get(d, ({},)): # a tuple of dicts
                s = sd.get('scheme', self.default_scheme)
                if s != 'center':
                    raise NotImplementedError("Mixed derivatives can only be"
                        " done with central differencing.")


    def scale_and_combine_operators(self):
        coeffs = self.coefficients
        self.operators = {}

        for d, op in sorted(self.simple_operators.items()):
            op = op.copy()
            dim = op.axis
            if d in coeffs:
                op.vectorized_scale(self.coefficient_vector(coeffs[d], self.t, dim))

            if len(set(d)) > 1:
                self.operators[d] = op
            else:
                # Combine scaled derivatives for this dimension
                if dim not in self.operators:
                    self.operators[dim] = op
                    # 0th derivative (r * V) is split evenly among each dimension
                    #TODO: This function is ONLY dependent on time. NOT MESH
                    if () in coeffs:
                        self.operators[dim] += coeffs[()](self.t) / float(self.grid.ndim)
                else:
                    if tuple(self.operators[dim].D.offsets) == tuple(op.D.offsets):
                        self.operators[dim] += op
                    else:
                        # print col, dim, combined_ops[dim].axis, self.simple_operators[dim].axis
                        self.operators[dim] = self.operators[dim] + op


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

        if (0,1) in self.operators:
            self.operators[(0,1)] *= dt

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
            if (0,1) in self.operators:
                V += self.operators[(0,1)].apply(V)
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

        tags = dict()
        for L in itertools.chain(Les, Lis, Firsts):
            if L.is_foldable():
                L.diagonalize()
                tags[id(L)] = 1

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
                V += L.apply(Y)

            Z = V.copy()

            for Le, Li in zip(Les, Lis):
                Z -= Le.apply(Y)
                Z = Li.solve(Z)

            Y -= Z

            for L in Firsts:
                no_residual = L.R
                L.R = None
                V -= 0.5 * L.apply(Y)
                L.R = no_residual

            for Le, Li in zip(Les, Lis):
                V -= Le.apply(Z)
                V = Li.solve(V)

        for i in tags:
            for L in itertools.chain(Les, Lis, Firsts):
                if id(L) == i:
                    L.undiagonalize()
                    break

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
    def dummy(self):
        n = 1
        dt = 0.01
        theta = 0.5
        initial = np.arange(self.shape[0] * self.shape[1], dtype=float)
        initial = initial.reshape(self.shape[0], self.shape[1])

        Firsts = [(o * dt) for d, o in self.operators.items()]

        Les = [(o * theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [(o * (theta * -dt)).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        for L in itertools.chain(Les, Lis):
            L.clear_residual()

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n if n != 0 else 0
        utils.tic("Dummy CPU:\t")
        V = initial
        Orig = V.copy()
        Y = V.copy()
        for k in range(n):
            if not k % print_step:
                print int(k * to_percent),
                sys.stdout.flush()

            Y = V.copy()
            for L in Firsts:
                Y += L.apply(V)

            for Le, Li in zip(Les, Lis):
                Y -= Le.apply(V)
                Y = Li.solve(Y)

        return Firsts, Les, Lis, Orig, V, Y


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

        tags = dict()
        for L in itertools.chain(Les, Lis, Firsts):
            if L.is_foldable():
                L.diagonalize()
                tags[id(L)] = 1

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

        for i in tags:
            for L in itertools.chain(Les, Lis, Firsts):
                if id(L) == i:
                    L.undiagonalize()
                    break

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
    return [x.copy() for _ in range(n)]

def flatten_tensor_aligned(mats, check=True):
    if check:
        assert len(set(tuple(m.D.offsets) for m in mats)) == 1
    residual = np.hstack([x.R for x in mats])
    diags = np.hstack([x.D.data for x in mats])
    B = BandedOperator((diags, mats[0].D.offsets), residual=residual)
    # TODO: Not sure why I set derivative to be None. Passes tests without it.
    # Need derivative to be copied for making blockwise ops.
    # B.copy_meta_data(mats[0], dirichlet=mats[0].dirichlet, derivative=None)
    B.copy_meta_data(mats[0])
    if len(mats) > 1:
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
    if len(mats) > 1:
        B.dirichlet = list(zip(*(m.dirichlet for m in mats)))
        for i, bound in enumerate(mats[0].dirichlet):
            if bound is None:
                B.dirichlet[i] = None
    B.blocks = len(mats)
    return B


if __name__ == '__main__':
    print "This is just a library."
