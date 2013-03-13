# coding: utf8
# cython: annotate=True
# cython: infer_types=True
# distutils: language = c++


from bisect import bisect_left

import sys

import numpy as np
cimport numpy as np

import itertools

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
        shape
        ndim
        coefficients
        t
        default_scheme
        default_order

    cdef SizedArrayPtr grid

    # Setup
    cdef public operators
    cdef public simple_operators

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
        self.grid = SizedArrayPtr(other.grid.domain[-1], "FDEGPU.grid")
        self.shape = other.grid.shape
        self.ndim = self.grid.p.ndim
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


    def solve_implicit_(self, n, dt, SizedArrayPtr V):
        Lis = [(o * -dt).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        Lis = np.roll(Lis, -1)

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("solve_implicit:\t")
        for k in range(n):
            if not k % print_step:
                print int(k * to_percent),
                sys.stdout.flush()
            if (0,1) in self.operators:
                U = V.copy(True)
                self.operators[(0,1)].apply_(U)
                U.timeseq_scalar(dt)
                V.pluseq(U)
                del U
            for L in Lis:
                L.solve_(V, overwrite=True)
        utils.toc(':  \t')


    def solve_implicit(self, n, dt, np.ndarray initial):
        n = int(n)
        cdef SizedArrayPtr V = SizedArrayPtr(initial)
        self.solve_implicit_(n, dt, V)
        ret = V.to_numpy()
        del V
        return ret


    def solve_douglas(self, n, dt, np.ndarray initial, theta=0.5):
        n = int(n)
        cdef SizedArrayPtr V = SizedArrayPtr(initial)
        self.solve_douglas_(n, dt, V)
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
            L.clear_residual()

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("Douglas:\t")
        Y = V.copy(True)
        for k in range(n):
            if not k % print_step:
                # if np.isnan(V).any():
                    # print "Douglas fail @ t = %f (%i steps)" % (dt * k, k)
                    # return V
                print int(k * to_percent),
                sys.stdout.flush()
            for L in Firsts:
                X = Y.copy(True)
                L.apply_(X, overwrite=True)
                V.pluseq(X)
                del X

            for Le, Li in zip(Les, Lis):
                X = Y.copy(True)
                Le.apply_(X, overwrite=True)
                V.minuseq(X)
                del X
                Li.solve_(V, overwrite=True)
            Y = V.copy(True)

        utils.toc(':  \t')
        return V


    # def solve_craigsneyd(self, n, dt, initial=None, theta=0.5, callback=None,
            # numpy=False):

        # n = int(n)
        # cdef SizedArray[double] V = initial.copy()

        # Firsts = [(o * dt) for o in self.operators.values()]

        # Les = [(o * theta * dt)
               # for d, o in sorted(self.operators.iteritems())
               # if type(d) != tuple]
        # Lis = [(o * (theta * -dt)).add(1, inplace=True)
               # for d, o in sorted(self.operators.iteritems())
               # if type(d) != tuple]

        # for L in itertools.chain(Les, Lis):
            # L.R = None

        # print_step = max(1, int(n / 10))
        # to_percent = 100.0 / n
        # utils.tic("Craig-Sneyd:\t")
        # for k in range(n):
            # if not k % print_step:
                # if np.isnan(V).any():
                    # print "Craig-Sneyd fail @ t = %f (%i steps)" % (dt * k, k)
                    # return V
                # print int(k * to_percent),
                # sys.stdout.flush()
            # if callback is not None:
                # callback(V, ((n - k) * dt))

            # Y = V.copy()
            # for L in Firsts:
                # Y += L.apply(V)
            # Y0 = Y.copy()

            # for Le, Li in zip(Les, Lis):
                # Y -= Le.apply(V)
                # Y = Li.solve(Y)

            # Y = Y0 + (0.5*dt) * self.cross_term(Y - V, numpy=False)
            # for Le, Li in zip(Les, Lis):
                # Y -= Le.apply(V)
                # Y = Li.solve(Y)
            # V = Y

        # utils.toc(':  \t')
        # return V


    # def solve_craigsneyd2(self, n, dt, initial=None, theta=0.5, callback=None,
            # numpy=False):

        # cdef SizedArray[double] V = initial.copy()
        # n = int(n)

        # Firsts = [(o * dt) for o in self.operators.values()]

        # Les = [(o * theta * dt)
               # for d, o in sorted(self.operators.iteritems())
               # if type(d) != tuple]
        # Lis = [(o * (theta * -dt)).add(1, inplace=True)
               # for d, o in sorted(self.operators.iteritems())
               # if type(d) != tuple]

        # for L in itertools.chain(Les, Lis):
            # L.R = None


        # print_step = max(1, int(n / 10))
        # to_percent = 100.0 / n
        # utils.tic("Craig-Sneyd 2:\t")
        # for k in range(n):
            # if not k % print_step:
                # if np.isnan(V).any():
                    # print "Craig-Sneyd 2 fail @ t = %f (%i steps)" % (dt * k, k)
                    # return V
                # print int(k * to_percent),
                # sys.stdout.flush()
            # if callback is not None:
                # callback(V, ((n - k) * dt))

            # Y = V.copy()
            # for L in Firsts:
                # Y += L.apply(V)
            # Y0 = Y.copy()

            # for Le, Li in zip(Les, Lis):
                # Y -= Le.apply(V)
                # Y = Li.solve(Y)
            # Y2 = Y.copy()

            # Y = Y0 + theta * dt * self.cross_term(Y2 - V, numpy=False)
            # for L in Firsts:
                # Y += (0.5 - theta) * L.apply(Y2-V)

            # for Le, Li in zip(Les, Lis):
                # Y -= Le.apply(V)
                # Y = Li.solve(Y)
            # V = Y

        # utils.toc(':  \t')
        # return V


    def solve_hundsdorferverwer(self, n, dt, initial=None, theta=0.5, callback=None,
            numpy=False):
        Firsts = [(o * dt) for o in self.operators.values()]

        Les = [(o * theta * dt)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]
        Lis = [(o * (theta * -dt)).add(1, inplace=True)
               for d, o in sorted(self.operators.iteritems())
               if type(d) != tuple]

        for L in itertools.chain(Les, Lis):
            L.clear_residual()

        print_step = max(1, int(n / 10))
        to_percent = 100.0 / n
        utils.tic("Hundsdorfer-Verwer:\t")
        for k in range(n):
            if not k % print_step:
                # if np.isnan(V).any():
                    # print "Hundsdorfer-Verwer fail @ t = %f (%i steps)" % (dt * k, k)
                    # return V
                print int(k * to_percent),
                sys.stdout.flush()

            Y = V.copy()
            for L in Firsts:
                X = Y.copy(True)
                L.apply_(X, overwrite=True)
                V.pluseq(X)
                del X
            Y0 = V.copy()

            for Le, Li in zip(Les, Lis):
                X = Y.copy(True)
                Le.apply_(X, overwrite=True)
                V.minuseq(X)
                del X
                Li.solve(V)
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
        return V


    # def solve_smooth(self, n, dt, initial=None, callback=None, smoothing_steps=2,
            # scheme=None):
        # if scheme is None:
            # scheme = self.solve_hundsdorferverwer
        # V = self.solve_implicit(smoothing_steps*2, dt*0.5, initial=initial)
        # # V = self.solve_douglas(smoothing_steps*2, dt*0.5, theta=1, initial=initial)
        # return scheme(n-smoothing_steps, dt, initial=V, theta=0.60)
