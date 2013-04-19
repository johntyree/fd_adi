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

from FiniteDifference.VecArray cimport SizedArray
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
        barrier
        SizedArrayPtr zero_derivative_coefficient
        np.ndarray zero_derivative_coefficient_host
        SizedArrayPtr gpugridmesh0
        SizedArrayPtr gpugridmesh1
        SizedArrayPtr gpugrid
        SizedArrayPtr scaling_vec



    def __init__(self):
        """
        Directly instantiating this class is only useful if you want to build
        it from a prototypical @FiniteDifferenceEngine@. In that case, call the
        `from_host_FiniteDifferenceEngine(proto)` method.

        There is no GPU implementation of the boundary conditions or scaling
        functions embedded into this class and it does not honor those
        parameters. If the engine is already initialized on the CPU, then
        building this from it will simply move the operators and domain to the
        GPU, skipping the creation step. (This implies that time dependent
        operators are not possible via this method)
        """
        self.simple_operators = {}
        self.operators = {}
        self.t = 0
        self.n = 0
        self.barrier = 0
        self.default_scheme = 'center'
        self.default_order = 2
        self.scaling_vec = SizedArrayPtr(tag="scaling_vec")
        self.zero_derivative_coefficient = SizedArrayPtr(tag="zero_derivative_coefficient")


    def _fill_gpugridmesh_from_grid(self):
        i = self.grid.ndim
        if i > 0:
            self.gpugridmesh0 = SizedArrayPtr(self.grid.mesh[0])
        if i > 1:
            self.gpugridmesh1 = SizedArrayPtr(self.grid.mesh[1])


    def from_host_FiniteDifferenceEngine(self, other):
        """
        Move a FiniteDifferenceEngine onto the GPU.
        """
        other.init()
        for attr in ['option', 'grid_analytical']:
            if hasattr(other, attr):
                setattr(self, attr, getattr(other, attr))
            else:
                setattr(self, attr, None)
        self.grid = other.grid.copy()
        self.gpugrid = SizedArrayPtr(self.grid.domain[-1], "FDEGPU.grid")
        self._fill_gpugridmesh_from_grid()
        self.scaling_vec.alloc(self.gpugrid.size, self.scaling_vec.tag)
        self.shape = self.grid.shape
        self.ndim = self.grid.ndim
        self.coefficients = other.coefficients

        # Remove cross deriv operator if correlation is 0
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


cdef class FiniteDifferenceEngineADI(FiniteDifferenceEngine):

    def __init__(self):
        """
        This class exists because an ADE method was also considered. It was
        scrapped so only one type of FDE remains.
        """
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


    cpdef preprocess_operators(self, SizedArrayPtr dt, SizedArrayPtr theta):
        """Combine operators together as needed by the schemes.
        This is factored out because it needs to be done at each time step when
        time dependent params are used.
        """

        if self.zero_derivative_coefficient.p == NULL:
            self.set_zero_derivative()

        self.scale_and_combine_operators()

        withdt = {k: (o * dt) for k,o in self.operators.iteritems()}

        Firsts = withdt.values()

        Les = [(o * theta)
            for d, o in sorted(withdt.iteritems())
            if type(d) != tuple]
        Lis = [(o * -theta).add(1, inplace=True)
            for d, o in sorted(withdt.iteritems())
            if type(d) != tuple]

        del withdt

        for L in itertools.chain(Les, Lis):
            L.enable_residual(False)

        return Firsts, Les, Lis


    cpdef scale_and_combine_operators(self, operators=None):
        """
        Scale each operator by the appropriate coefficient based on the PDE.
        If this is being done on the GPU we call out to cuda kernels. Otherwise
        it's done in the same way as the CPU implementation, then copied over.
        """
        if operators is None:
            try:
                self.make_operator_templates()
            except AttributeError:
                pass
            operators = self.simple_operators  # single derivative operators
        self.operators = {}
        coeffs = self.coefficients

        # Usually, coeffs would be a dict containing lambdas representing the
        # functions which are the coefficients. If it's just a list then we
        # must be on the GPU with hardcoded implementations.
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


    def _dummy(self):
        """ This method is only for testing. Ignore."""
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


    cpdef set_zero_derivative(self):
        """Compute the zero derivative coefficient (-r(t) in the Heston case)
        and move it over to the GPU.
        """
        if self.zero_derivative_coefficient_host is None:
            if () in self.coefficients:
                try:
                    self.zero_derivative_coefficient_host = np.atleast_1d(
                        self.coefficients[()](self.t) / float(self.grid.ndim))
                except TypeError:
                    raise RuntimeError("Zero derivative coefficient has not been set.")
            else:
                try:
                    self.zero_derivative_coefficient_host = np.atleast_1d(
                        -self.option.interest_rate.value / self.grid.ndim)
                    print "Zero Derivative not set. Taking (-r / ndim)."
                except AttributeError:
                    raise RuntimeError("Zero derivative coefficient has not been set.")
        self.zero_derivative_coefficient = SizedArrayPtr(self.zero_derivative_coefficient_host)


    def solve_implicit(self, n, dt, initial=None):
        """Wrapper for implicit scheme using CPU-based data."""
        n = int(n)
        cdef SizedArrayPtr V
        if initial is not None:
            V = SizedArrayPtr(initial)
        else:
            V = self.gpugrid
        cdef SizedArrayPtr dt_ = SizedArrayPtr(np.atleast_1d(dt))
        self.solve_implicit_(n, dt_, V)
        ret = V.to_numpy()
        del V
        return ret


    cpdef solve_implicit_(self, n, SizedArrayPtr dt, SizedArrayPtr V, callback=None, numpy=False):
        """The fully implicit scheme on the GPU."""

        if callback or numpy:
            raise NotImplementedError("Callbacks and Numpy not available for GPU solver.")

        theta = SizedArrayPtr(np.atleast_1d(1.0))

        f, l, Lis = self.preprocess_operators(dt, theta)
        del f, l

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


    def solve_douglas(self, n, dt, initial=None, theta=0.5, callback=None, numpy=False):
        """Wrapper for the douglas scheme using CPU based data."""
        if callback or numpy:
            raise NotImplementedError("Callbacks and Numpy not available for GPU solver.")
        n = int(n)
        cdef SizedArrayPtr V
        if initial is not None:
            V = SizedArrayPtr(initial)
        else:
            V = self.gpugrid
        self.solve_douglas_(n, dt, V, theta)
        ret = V.to_numpy()
        del V
        return ret


    cpdef solve_douglas_(self, int n, double dt, SizedArrayPtr V, double theta=0.5):
        """The Douglas ADI scheme on the GPU"""

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
            # Preprocess here to regenerate operators for new timestep
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
        """Wrapper for HV ADI scheme."""
        if callback or numpy:
            raise NotImplementedError("Callbacks and Numpy not available for GPU solver.")
        n = int(n)
        cdef SizedArrayPtr V
        if initial is not None:
            V = SizedArrayPtr(initial)
        else:
            V = self.gpugrid
        cdef SizedArrayPtr dt_ = SizedArrayPtr(np.atleast_1d(dt))
        cdef SizedArrayPtr theta_ = SizedArrayPtr(np.atleast_1d(theta))
        self.solve_hundsdorferverwer_(n, dt_, V, theta_)
        ret = V.to_numpy()
        del V
        return ret


    cpdef solve_hundsdorferverwer_(self, n, SizedArrayPtr dt, SizedArrayPtr V, SizedArrayPtr theta):
        """HV ADI scheme implemented on the GPU"""

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
            # Preprocess here to regenerate operators for new timestep

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
        """
        Perform @smoothing_steps@ fully implicit steps, then do the HV
        scheme to completion.
        """
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
        """
        This creates a GPU based FiniteDifferenceEngine that is hardcoded to
        solve Vanilla Heston Options. It doesn't honor @coefficients@ or
        @boundaries@. If you want them, then create the FDE first and import it
        as an FDEGPUADI with the from_host... method

        @option@ is a HestonOption
        """
        FiniteDifferenceEngineADI.__init__(self)

        if schemes is not None or flip_idx_var or flip_idx_spot:
            raise NotImplementedError, "Only central differencing supported on GPU"

        self.coefficients = [(0,), (0,0), (1,), (1,1), (0,1), ()]
        self.cache = cache
        assert isinstance(option, Option)
        self.option = option

        if isinstance(option, BarrierOption):
            if option.top:
                if option.top[0]: # Knockin, not sure about implementing this
                    raise NotImplementedError("Knockin barriers are not supported.")
                else:
                    spot_max = option.top[1]
                    if grid:
                        assert np.allclose(spot_max, max(grid.mesh[0]))
            if option.bottom:
                raise NotImplementedError("Bottom barriers are not supported for GPU creation.")
                # if option.bottom[0]: # Knockin, not sure about implementing this
                    # raise NotImplementedError("Knockin barriers are not supported.")
                # else:
                    # spot_min = option.bottom[1]


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
                        # spots = np.linspace(0, spot_max**p, nspots)**(1.0/p)
                        spots = utils.exponential_space(0.00, self.option.strike, spot_max,
                                                    1.0/p, nspots,
                                                    force_exact=force_exact)
                        print "Barrier spots"
                        self.barrier = self.option.top[1]
                        print "Barrier @", self.barrier
                else:
                    spots = utils.sinh_space(option.strike-spot_min, spot_max-spot_min, spotdensity, nspots, force_exact=force_exact) + spot_min
            self.spots = spots
            grid = Grid([self.spots, self.vars], initializer=lambda *x: np.maximum(x[0]-option.strike,0))


        newstrike = self.spots[np.argmin(np.abs(self.spots - option.strike))]
        self.spots[np.argmin(np.abs(self.spots - option.spot))] = option.spot


        self.grid = grid
        self.gpugrid = SizedArrayPtr(self.grid.domain[-1], "FDEGPU.grid")
        self._fill_gpugridmesh_from_grid()
        self.scaling_vec.alloc(self.gpugrid.size, self.scaling_vec.tag)
        self.zero_derivative_coefficient_host = np.atleast_1d(
                -self.option.interest_rate.value / self.grid.ndim)


    def make_operator_templates(self):
        """
        Create the single derivative operators for the Heston PDE
        """
        m0 = self.grid.mesh[0]
        m1 = self.grid.mesh[1]
        self.zero_derivative_coefficient_host = np.atleast_1d(
            -self.option.interest_rate.value / self.grid.ndim)

        self.simple_operators[(0,)] = BOG.for_vector(m0, m1.size, 1, 0, self.barrier)
        self.simple_operators[(0,)].has_low_dirichlet = True
        if self.barrier:
            self.simple_operators[(0,)].has_high_dirichlet = True

        self.simple_operators[(0,0)] = BOG.for_vector(m0, m1.size, 2, 0, self.barrier)
        self.simple_operators[(0,0)].has_low_dirichlet = True
        if self.barrier:
            self.simple_operators[(0,0)].has_high_dirichlet = True

        self.simple_operators[(1,)] = BOG.for_vector(m1, m0.size, 1, 1, self.barrier)

        self.simple_operators[(1,1)] = BOG.for_vector(m1, m0.size, 2, 1, self.barrier)

        try:
            if self.option.correlation != 0:
                self.simple_operators[(0,1)] = BOG.mixed_for_vector(m0, m1)
        except AttributeError:
            pass


    @property
    def idx(self):
        """
        The indices in the domain corresponding to the Option paramters.
        """
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

