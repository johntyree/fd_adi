# coding: utf8
# cython: annotate=True
# cython: infer_types=True
# distutils: language = c++
# distutils: sources = FiniteDifference/_coefficients.cu FiniteDifference/VecArray.cu FiniteDifference/backtrace.c FiniteDifference/filter.c


import sys
import itertools


import numpy as np
cimport numpy as np

from bisect import bisect_left

cimport cython
from cython.operator import dereference as deref

import FiniteDifference.utils as utils

import FiniteDifference.BandedOperatorGPU as BOG
cimport FiniteDifference.BandedOperatorGPU as BOG

import FiniteDifference.BandedOperator as BO
cimport FiniteDifference.BandedOperator as BO
BandedOperator = BO.BandedOperator

from FiniteDifference.VecArray cimport SizedArray, GPUVec
from FiniteDifference.SizedArrayPtr cimport SizedArrayPtr

from FiniteDifference.Option import Option, BarrierOption
from FiniteDifference.Grid import Grid

cimport coefficients

from VecArray cimport to_string


cdef class FiniteDifferenceEngine(object):

    cdef public:
        boundaries
        coefficients
        default_order
        default_scheme
        grid
        grid_analytical
        ndim
        shape
        simple_operators
        operators
        option
        t # Current time
        n # Current step t / dt
        cache
        schemes
        SizedArrayPtr zero_derivative_coefficient
        SizedArrayPtr gpugridmesh0
        SizedArrayPtr gpugridmesh1
        SizedArrayPtr gpugrid
        SizedArrayPtr scaling_vec



    def __init__(self):
        """
        @coefficients@ is a dict of tuple, function pairs with c[i,j] referring to the
        coefficient of the i j derivative, dU/didj. Absent pairs are counted as zeros.

        The functions MUST be able to handle dims+1 arguments, with the first
        being time and the rest corresponding to the dimensions given by @grid.shape@.

        Still need a good way to handle cross terms.

        N.B. You don't actually want to do this with lambdas. They aren't real
        closures. Weird things might happen.

        Ex. (2D grid)

            {(): lambda t, x0, x1: 0.06 # 0th derivative
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
        self.simple_operators = {}
        self.operators = {}
        self.t = 0
        self.n = 0
        self.default_scheme = 'center'
        self.default_order = 2
        self.scaling_vec = SizedArrayPtr(tag="scaling_vec")
        self.zero_derivative_coefficient = SizedArrayPtr(tag="zero_derivative_coefficient")

    def fill_gpugridmesh_from_grid(self):
        i = self.grid.ndim
        if i > 0:
            self.gpugridmesh0 = SizedArrayPtr(self.grid.mesh[0])
        if i > 1:
            self.gpugridmesh1 = SizedArrayPtr(self.grid.mesh[1])


    def from_host_FiniteDifferenceEngine(self, other):
        other.init()
        for attr in ['option', 'grid_analytical']:
            if hasattr(other, attr):
                setattr(self, attr, getattr(other, attr))
            else:
                setattr(self, attr, None)
        self.grid = other.grid.copy()
        self.gpugrid = SizedArrayPtr(self.grid.domain[-1], "FDEGPU.grid")
        self.fill_gpugridmesh_from_grid()
        self.scaling_vec.alloc(self.gpugrid.size, self.scaling_vec.tag)
        self.shape = self.grid.shape
        self.ndim = self.grid.ndim
        self.coefficients = other.coefficients

        # Setup
        try:
            if self.option.correlation == 0:
                other.operators.pop((0,1))
                other.simple_operators.pop((0,1))
        except KeyError:
            pass
        except AttributeError:
            pass
        self.operators = {k:BOG.BandedOperator(v) for k,v in other.operators.items()}
        self.simple_operators = {k:BOG.BandedOperator(v) for k,v in other.simple_operators.items()}


    def solve(self):
        """Run all the way to the terminal condition."""
        raise NotImplementedError


cdef class FiniteDifferenceEngineADI(FiniteDifferenceEngine):

    def __init__(self):
        FiniteDifferenceEngine.__init__(self)


    def coefficient_vector(self, f, t, dim):
        """
        Evaluate f with the cartesian product of the elements of
        self.grid.mesh, ordered such that dim is the fastest varying. The
        relative order of the other dimensions remains the same.
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


    def scale_and_combine_operators(self, operators=None):
        if operators is None:
            try:
                self.make_operator_templates()
            except AttributeError:
                pass
            operators = self.simple_operators
        self.operators = {}
        coeffs = self.coefficients
        on_gpu = type(coeffs) == list

        for d, op in sorted(operators.items()):
            op = op.copy()
            dim = op.axis
            if not on_gpu:
                op.vectorized_scale_from_host(self.coefficient_vector(coeffs[d], self.t, dim))
            else:
                if d in coeffs:
                    if d == (0,):
                        coefficients.scale_0(self.t,
                                self.option.interest_rate.value,
                                deref(self.gpugridmesh0.p),
                                deref(self.gpugridmesh1.p),
                                deref(self.scaling_vec.p)
                        )
                    elif d == (0,0):
                        coefficients.scale_00(self.t,
                                self.option.interest_rate.value,
                                deref(self.gpugridmesh0.p),
                                deref(self.gpugridmesh1.p),
                                deref(self.scaling_vec.p)
                        )
                    elif d == (1,):
                        coefficients.scale_1(self.t,
                                self.option.interest_rate.value,
                                deref(self.gpugridmesh0.p),
                                deref(self.gpugridmesh1.p),
                                self.option.variance.reversion,
                                self.option.variance.mean,
                                deref(self.scaling_vec.p)
                        )
                    elif d == (1,1):
                        coefficients.scale_11(self.t,
                                self.option.interest_rate.value,
                                deref(self.gpugridmesh0.p),
                                deref(self.gpugridmesh1.p),
                                self.option.variance.volatility,
                                deref(self.scaling_vec.p)
                        )
                    elif d == (0,1):
                        coefficients.scale_01(self.t,
                                self.option.interest_rate.value,
                                deref(self.gpugridmesh0.p),
                                deref(self.gpugridmesh1.p),
                                self.option.variance.volatility,
                                self.option.correlation,
                                deref(self.scaling_vec.p)
                        )
                    else:
                        assert False, "All ops should be GPU scaled now."
                    op.vectorized_scale(self.scaling_vec)

            if len(set(d)) > 1:
                self.operators[d] = op
            else:
                # Combine scaled derivatives for this dimension
                if dim not in self.operators:
                    self.operators[dim] = op
                    # 0th derivative (r * V) is split evenly among each dimension
                    if () in coeffs:
                        if not on_gpu:
                            self.operators[dim] += coeffs[()](self.t) / float(self.grid.ndim)
                        elif (self.zero_derivative_coefficient.p == NULL):
                            assert False, ("Zero derviative has not been set.")
                        else:
                            self.operators[dim].add_scalar(self.zero_derivative_coefficient, self.n)
                else:
                    self.operators[dim] += op
        return self.scaling_vec


    def cross_term(self, V, numpy=True):
        """Apply the cross derivative operator."""
        if (0,1) in self.coefficients:
            ret = self.operators[(0,1)].apply(V)
        else:
            ret = 0
        return ret


    def dummy(self):
        n = 1
        dt = 0.01
        theta = 0.5
        initial = np.arange(self.shape[0] * self.shape[1], dtype=float)
        initial = initial.reshape(self.shape[0], self.shape[1])

        Firsts = [o.mul_scalar_from_host(dt) for d, o in self.operators.items()]

        Les = [o.mul_scalar_from_host(theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [o.mul_scalar_from_host(-dt*theta).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        for L in itertools.chain(Les, Lis):
            L.enable_residual(False)

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n if n != 0 else 0
        utils.tic("Dummy GPU:\t")
        cdef SizedArrayPtr V = SizedArrayPtr(initial)
        Orig = V.copy(True)
        Y = V.copy(True)
        X = V.copy(True)

        tags = dict()
        for L in itertools.chain(Les, Lis, Firsts):
            if L.is_foldable():
                L.diagonalize()
                tags[id(L)] = 1

        for k in range(n):
            if not k % print_step:
                print int(k * to_percent),
                sys.stdout.flush()

            Y.copy_from(V)
            for L in Firsts:
                X.copy_from(Y)
                L.apply_(X, overwrite=True)
                V.pluseq(X)

            for Le, Li in zip(Les, Lis):
                X.copy_from(Y)
                Le.apply_(X, overwrite=True)
                V.minuseq(X)
                Li.solve_(V, overwrite=True)

        for i in tags:
            for L in itertools.chain(Les, Lis, Firsts):
                if id(L) == i:
                    L.undiagonalize()
                    break

        return Firsts, Les, Lis, Orig.to_numpy(), Y.to_numpy(), V.to_numpy()


    def solve_implicit(self, n, dt, np.ndarray initial):
        n = int(n)
        cdef SizedArrayPtr V = SizedArrayPtr(initial)
        cdef SizedArrayPtr dt_ = SizedArrayPtr(np.atleast_1d(dt))
        self.solve_implicit_(n, dt_, V)
        ret = V.to_numpy()
        del V
        return ret


    cpdef solve_implicit_(self, n, SizedArrayPtr dt, SizedArrayPtr V, callback=None, numpy=False):
        if callback or numpy:
            raise NotImplementedError("Callbacks and Numpy not available for GPU solver.")

        self.scale_and_combine_operators()

        dt.timeseq_scalar_from_host(-1)
        Lis = [(o * dt).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        dt.timeseq_scalar_from_host(-1)

        Lis = np.roll(Lis, -1)

        if (0,1) in self.operators:
            self.operators[(0,1)] *= dt

        for o in Lis:
            if o.top_fold_status == 'CAN_FOLD':
                o.diagonalize()
            if o.bottom_fold_status == 'CAN_FOLD':
                o.diagonalize()

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("solve_implicit:\t")

        for k in range(n):
            if not k % print_step:
                print int(k * to_percent),
                sys.stdout.flush()
            if (0,1) in self.operators:
                U = V.copy(True)
                self.operators[(0,1)].apply_(U, overwrite=True)
                V.pluseq(U)
                del U
            for L in Lis:
                L.solve_(V, overwrite=True)
        utils.toc(':  \t')


    def solve_douglas(self, n, dt, np.ndarray initial, theta=0.5, callback=None, numpy=False):
        if callback or numpy:
            raise NotImplementedError("Callbacks and Numpy not available for GPU solver.")
        n = int(n)
        cdef SizedArrayPtr V = SizedArrayPtr(initial)
        self.solve_douglas_(n, dt, V, theta)
        ret = V.to_numpy()
        del V
        return ret


    cpdef solve_douglas_(self, int n, double dt, SizedArrayPtr V, double theta=0.5):

        Firsts = [o.mul_scalar_from_host(dt) for d, o in self.operators.items()]

        Les = [o.mul_scalar_from_host(theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [o.mul_scalar_from_host(theta * -dt).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        for L in itertools.chain(Les, Lis):
            L.enable_residual(False)

        tags = dict()
        for L in itertools.chain(Les, Lis, Firsts):
            if L.is_foldable():
                L.diagonalize()
                tags[id(L)] = 1

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("Douglas:\t")
        Y = V.copy(True)
        X = SizedArrayPtr().alloc(V.size)
        for k in range(n):
            if not k % print_step:
                # if np.isnan(V).any():
                    # print "Douglas fail @ t = %f (%i steps)" % (dt * k, k)
                    # return V
                print int(k * to_percent),
                sys.stdout.flush()
            for L in Firsts:
                X.copy_from(Y)
                L.apply_(X, overwrite=True)
                V.pluseq(X)

            for Le, Li in zip(Les, Lis):
                X.copy_from(Y)
                Le.apply_(X, overwrite=True)
                V.minuseq(X)
                Li.solve_(V, overwrite=True)
            Y.copy_from(V)

        utils.toc(':  \t')


    def solve_hundsdorferverwer(self, n, dt, initial=None, theta=0.5, callback=None, numpy=False):
        if callback or numpy:
            raise NotImplementedError("Callbacks and Numpy not available for GPU solver.")
        n = int(n)
        cdef SizedArrayPtr V = SizedArrayPtr(initial)
        cdef SizedArrayPtr dt_ = SizedArrayPtr(np.atleast_1d(dt))
        cdef SizedArrayPtr theta_ = SizedArrayPtr(np.atleast_1d(theta))
        self.solve_hundsdorferverwer_(n, dt_, V, theta_)
        ret = V.to_numpy()
        del V
        return ret


    cpdef preprocess_operators(self, SizedArrayPtr gpu_dt, SizedArrayPtr gpu_theta):

        self.scale_and_combine_operators()

        withdt = {k: (o * gpu_dt) for k,o in self.operators.iteritems()}

        # Don't touch this if it doesn't exist
        if self.zero_derivative_coefficient.p != NULL:
            for d, o in withdt.iteritems():
                if np.isscalar(d):
                    o.add_scalar(self.zero_derivative_coefficient, self.n)

        Firsts = withdt.values()

        Les = [(o * gpu_theta)
            for d, o in sorted(withdt.iteritems())
            if type(d) != tuple]
        Lis = [(o * -gpu_theta).add(1, inplace=True)
            for d, o in sorted(withdt.iteritems())
            if type(d) != tuple]

        del withdt

        for L in itertools.chain(Les, Lis):
            L.enable_residual(False)

        return Firsts, Les, Lis


    cpdef solve_hundsdorferverwer_(self, n, SizedArrayPtr dt, SizedArrayPtr V, SizedArrayPtr theta):

        # Don't touch this if it doesn't exist
        if self.zero_derivative_coefficient.p != NULL:
            self.zero_derivative_coefficient.timeseq(dt)


        Firsts, Les, Lis = self.preprocess_operators(dt, theta)

        tags = dict()
        for L in itertools.chain(Les, Lis, Firsts):
            if L.is_foldable():
                L.diagonalize()
                tags[id(L)] = 1

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("Hundsdorfer-Verwer:\t")
        # Pre allocate
        X = SizedArrayPtr().alloc(V.p.size)
        Y = SizedArrayPtr().alloc(V.p.size)
        Z = SizedArrayPtr().alloc(V.p.size)
        for k in range(n):
            if not k % print_step:
                # if np.isnan(V).any():
                    # print "Hundsdorfer-Verwer fail @ t = %f (%i steps)" % (dt * k, k)
                    # return V
                print int(k * to_percent),
                sys.stdout.flush()

            Y.copy_from(V)
            for L in Firsts:
                X.copy_from(Y)
                L.apply_(X, overwrite=True)
                V.pluseq(X)

            Z.copy_from(V)

            for Le, Li in zip(Les, Lis):
                X.copy_from(Y)
                Le.apply_(X, overwrite=True)
                Z.minuseq(X)
                Li.solve_(Z, overwrite=True)

            Y.minuseq_over2(Z)

            for L in Firsts:
                L.enable_residual(False)
                X.copy_from(Y)
                L.apply_(X, overwrite=True)
                V.minuseq(X)
                L.enable_residual(True)

            # del Firsts

            for Le, Li in zip(Les, Lis):
                X.copy_from(Z)
                Le.apply_(X, overwrite=True)
                V.minuseq(X)
                Li.solve_(V, overwrite=True)

            # del Les, Lis

        utils.toc(':  \t')


    def solve_smooth(self, n, dt, initial=None, callback=None, smoothing_steps=2,
            scheme=None):
        if scheme:
            raise NotImplementedError("Changing smoothing schemes not supported on GPU.")
        n = int(n)
        cdef SizedArrayPtr V
        if initial is not None:
            V = SizedArrayPtr(initial)
        else:
            V = self.gpugrid
        cdef SizedArrayPtr dt_ = SizedArrayPtr(np.atleast_1d(dt))
        self.solve_smooth_(n, dt_, V, smoothing_steps)
        ret = V.to_numpy()
        del V
        return ret


    cpdef solve_smooth_(self, n, SizedArrayPtr dt, SizedArrayPtr V, smoothing_steps=2):
        halfdt = SizedArrayPtr()
        halfdt.alloc(1)
        halfdt.copy_from(dt)
        halfdt.timeseq_scalar_from_host(0.5)
        theta = SizedArrayPtr(np.atleast_1d(0.6))
        self.solve_implicit_(smoothing_steps*2, halfdt, V)
        self.solve_hundsdorferverwer_(n-smoothing_steps, dt, V, theta)


cdef class HestonFiniteDifferenceEngine(FiniteDifferenceEngineADI):

    cdef public:
        vars
        spots

    """FDEGPU specialized for Heston options."""
    def __init__(self, option,
            grid=None,
            spot_max=1500.0,
            spot_min=0.0,
            spots=None,
            vars=None,
            var_max=10.0,
            nspots=100,
            nvols=100,
            spotdensity=7.0,
            varexp=4.0,
            force_exact=True,
            flip_idx_var=False,
            flip_idx_spot=False,
            schemes=None,
            coefficients=None,
            boundaries=None,
            cache=True,
            verbose=True,
            force_bandwidth=None
            ):
        """@option@ is a HestonOption"""
        FiniteDifferenceEngineADI.__init__(self)

        if schemes is not None or flip_idx_var or flip_idx_spot:
            raise NotImplementedError, "Only central differencing supported on GPU"

        self.coefficients = [(0,), (0,0), (1,), (1,1), (0,1), ()]
        self.cache = cache
        assert isinstance(option, Option)
        self.option = option

        # if isinstance(option, BarrierOption):
            # if option.top:
                # if option.top[0]: # Knockin, not sure about implementing this
                    # raise NotImplementedError("Knockin barriers are not supported.")
                # else:
                    # spot_max = option.top[1]
                    # if grid:
                        # assert np.allclose(spot_max, max(grid.mesh[0]))
                    # boundaries[(0,)] = (boundaries[(0,)][0], (0, lambda *x: 0.0))
                    # boundaries[(0,0)] = boundaries[(0,)]
            # if option.bottom:
                # if option.bottom[0]: # Knockin, not sure about implementing this
                    # raise NotImplementedError("Knockin barriers are not supported.")
                # else:
                    # spot_min = option.bottom[1]
                    # boundaries[(0,)] = ((0, lambda *x: 0.0), boundaries[(0,)][1])
                    # boundaries[(0,0)] = boundaries[(0,)]


        if grid:
            self.spots = grid.mesh[0]
            self.vars = grid.mesh[1]
        else:
            if vars is None:
                # vars = np.linspace(0, var_max, nvols)
                vars = utils.exponential_space(0.00, option.variance.value, var_max,
                                            varexp, nvols,
                                            force_exact=force_exact)
            self.vars = vars
            if spots is None:
                # spots = np.linspace(0,spot_max,nspots)
                if isinstance(option, BarrierOption) and option.top and not option.top[0]:
                        p = 3
                        spots = np.linspace(0, spot_max**p, nspots)**(1.0/p)
                        print "Barrier spots"
                else:
                    spots = utils.sinh_space(option.strike-spot_min, spot_max-spot_min, spotdensity, nspots, force_exact=force_exact) + spot_min
            self.spots = spots
            grid = Grid([self.spots, self.vars], initializer=lambda *x: np.maximum(x[0]-option.strike,0))


        newstrike = self.spots[np.argmin(np.abs(self.spots - option.strike))]
        self.spots[np.argmin(np.abs(self.spots - option.spot))] = option.spot


        self.grid = grid
        self.gpugrid = SizedArrayPtr(self.grid.domain[-1], "FDEGPU.grid")
        self.fill_gpugridmesh_from_grid()
        self.scaling_vec.alloc(self.gpugrid.size, self.scaling_vec.tag)


    def make_operator_templates(self):
        m0 = self.grid.mesh[0]
        m1 = self.grid.mesh[1]
        self.zero_derivative_coefficient = SizedArrayPtr(
            np.array(-self.option.interest_rate.value / self.grid.ndim)
        )

        self.simple_operators[(0,)] = BOG.for_vector(m0, m1.size, 1, 0)
        self.simple_operators[(0,)].has_low_dirichlet = True

        self.simple_operators[(0,0)] = BOG.for_vector(m0, m1.size, 2, 0)
        self.simple_operators[(0,0)].has_low_dirichlet = True

        self.simple_operators[(1,)] = BOG.for_vector(m1, m0.size, 1, 1)

        self.simple_operators[(1,1)] = BOG.for_vector(m1, m0.size, 2, 1)

        try:
            if self.option.correlation != 0:
                self.simple_operators[(0,1)] = BOG.mixed_for_vector(m0, m1)
        except AttributeError:
            pass


    @property
    def idx(self):
        ids = bisect_left(np.round(self.spots, decimals=4), np.round(self.option.spot, decimals=4))
        idv = bisect_left(np.round(self.vars, decimals=4), np.round(self.option.variance.value, decimals=4))
        return (ids, idv)

    @property
    def price(self):
        return self.grid.domain[-1][self.idx]


    @property
    def grid_analytical(self):
        raise NotImplementedError
        # H = self.option
        # if isinstance(H, BarrierOption):
            # raise NotImplementedError("No analytical solution for Heston barrier options.")
        # hs = hs_call_vector(self.spots, H.strike,
            # H.interest_rate.value, np.sqrt(self.vars), H.tenor,
            # H.variance.reversion, H.variance.mean, H.variance.volatility,
            # H.correlation, HFUNC=HestonCos, cache=self.cache)

        # if max(hs.flat) > self.spots[-1] * 2:
            # self.BADANALYTICAL = True
            # print "Warning: Analytical solution looks like trash."
        # else:
            # self.BADANALYTICAL = False
        # return hs

