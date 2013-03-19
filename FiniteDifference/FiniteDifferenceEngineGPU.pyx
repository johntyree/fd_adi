# coding: utf8
# cython: annotate=True
# cython: infer_types=True
# distutils: language = c++


import sys
import itertools


import numpy as np
cimport numpy as np


import FiniteDifference.utils as utils

import FiniteDifference.BandedOperatorGPU as BOG
cimport FiniteDifference.BandedOperatorGPU as BOG

import FiniteDifference.BandedOperator as BO
cimport FiniteDifference.BandedOperator as BO
BandedOperator = BO.BandedOperator

from FiniteDifference.VecArray cimport SizedArray, GPUVec
from FiniteDifference.SizedArrayPtr cimport SizedArrayPtr


cdef class FiniteDifferenceEngine(object):

    cdef public:
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
        t


    cdef SizedArrayPtr gpugrid


    def __init__(self, other):
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
        other.init()
        for attr in ['option', 'grid_analytical']:
            if hasattr(other, attr):
                setattr(self, attr, getattr(other, attr))
            else:
                setattr(self, attr, None)
        self.grid = other.grid.copy()
        self.gpugrid = SizedArrayPtr(self.grid.domain[-1], "FDEGPU.grid")
        self.shape = self.grid.shape
        self.ndim = self.grid.ndim
        self.coefficients = other.coefficients
        self.t = 0
        self.default_scheme = 'center'
        self.default_order = 2

        # Setup
        self.operators = {k:BOG.BandedOperator(v) for k,v in other.operators.items()}
        self.simple_operators = {k:BOG.BandedOperator(v) for k,v in other.simple_operators.items()}


    def solve(self):
        """Run all the way to the terminal condition."""
        raise NotImplementedError


cdef class FiniteDifferenceEngineADI(FiniteDifferenceEngine):

    def __init__(self, other):
        FiniteDifferenceEngine.__init__(self, other)


    def scale_and_combine_operators(self):
        raise NotImplementedError
        # coeffs = self.coefficients
        # self.operators = {}

        # for d, op in self.simple_operators.items():
            # op = op.copy()
            # dim = op.axis
            # if d in coeffs:
                # op.vectorized_scale(self.coefficient_vector(coeffs[d], self.t, dim))

            # if len(set(d)) > 1:
                # self.operators[d] = op
            # else:
                # # Combine scaled derivatives for this dimension
                # if dim not in self.operators:
                    # self.operators[dim] = op
                    # # 0th derivative (r * V) is split evenly among each dimension
                    # #TODO: This function is ONLY dependent on time. NOT MESH
                    # if () in coeffs:
                        # self.operators[dim] += coeffs[()](self.t) / float(self.grid.ndim)
                # else:
                    # if tuple(self.operators[dim].D.offsets) == tuple(op.D.offsets):
                        # self.operators[dim] += op
                    # else:
                        # # print col, dim, combined_ops[dim].axis, self.simple_operators[dim].axis
                        # self.operators[dim] = self.operators[dim] + op


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

        Firsts = [(o * dt) for d, o in self.operators.items()]

        Les = [(o * theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [(o * (theta * -dt)).add(1, inplace=True)
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
        self.solve_implicit_(n, dt, V)
        ret = V.to_numpy()
        del V
        return ret


    cpdef solve_implicit_(self, n, dt, SizedArrayPtr V, callback=None, numpy=False):
        if callback or numpy:
            raise NotImplementedError("Callbacks and Numpy not available for GPU solver.")
        Lis = [(o * -dt).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

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
        Firsts = [(o * dt) for d, o in self.operators.items()]

        Les = [(o * theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [(o * (theta * -dt)).add(1, inplace=True)
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
        self.solve_hundsdorferverwer_(n, dt, V, theta)
        ret = V.to_numpy()
        del V
        return ret


    cpdef solve_hundsdorferverwer_(self, n, dt, SizedArrayPtr V, theta=0.5):
        Firsts = [(o * dt) for o in self.operators.values()]

        Les = [(o * theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [(o * (theta * -dt)).add(1, inplace=True)
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

            for Le, Li in zip(Les, Lis):
                X.copy_from(Z)
                Le.apply_(X, overwrite=True)
                V.minuseq(X)
                Li.solve_(V, overwrite=True)

        utils.toc(':  \t')

        for i in tags:
            for L in itertools.chain(Les, Lis, Firsts):
                if id(L) == i:
                    L.undiagonalize()
                    break


    def solve_smooth(self, n, dt, initial=None, callback=None, smoothing_steps=2,
            scheme=None):
        if scheme:
            raise NotImplementedError("Changing smoothing schemes not supported on GPU.")
        n = int(n)
        cdef SizedArrayPtr V = SizedArrayPtr(initial)
        self.solve_smooth_(n, dt, V, smoothing_steps)
        ret = V.to_numpy()
        del V
        return ret


    cpdef solve_smooth_(self, n, dt, SizedArrayPtr V, smoothing_steps=2):
        self.solve_implicit_(smoothing_steps*2, dt*0.5, V)
        self.solve_hundsdorferverwer_(n-smoothing_steps, dt, V, theta=0.60)
