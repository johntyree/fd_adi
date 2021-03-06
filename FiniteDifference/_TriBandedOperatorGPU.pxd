

from libcpp cimport bool as cbool
from libcpp.string cimport string as cpp_string

from VecArray cimport SizedArray, to_string

from FiniteDifference.thrust.device_ptr cimport device_ptr


cdef extern from "_TriBandedOperatorGPU.cuh":

    cdef cppclass _TriBandedOperator:
        cbool is_folded() except +
        cbool has_residual
        cpp_string bottom_fold_status
        cpp_string top_fold_status
        Py_ssize_t block_len
        Py_ssize_t blocks
        Py_ssize_t operator_rows

        void add_operator(_TriBandedOperator &other) except +
        void add_scalar_from_host(double val) except +
        void add_scalar(device_ptr[double]) except +
        void apply(SizedArray[double] &) except +
        void diagonalize() except +
        void fold_bottom(cbool unfold) except +
        void fold_top(cbool unfold) except +
        void solve(SizedArray[double] &) except +
        void fake_solve(SizedArray[double] &) except +
        void undiagonalize() except +
        void vectorized_scale(SizedArray[double] &vector) except +

        SizedArray[double] bottom_factors
        SizedArray[double] top_factors
        SizedArray[double] high_dirichlet,
        SizedArray[double] low_dirichlet,
        SizedArray[double] diags
        SizedArray[double] R
        SizedArray[double] deltas
        cbool has_high_dirichlet,
        cbool has_low_dirichlet,


        _TriBandedOperator(
            SizedArray[double] data,
            SizedArray[double] R,
            SizedArray[double] deltas,
            SizedArray[double] high_dirichlet,
            SizedArray[double] low_dirichlet,
            SizedArray[double] top_factors,
            SizedArray[double] bottom_factors,
            unsigned int axis,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            cbool has_high_dirichlet,
            cbool has_low_dirichlet,
            cpp_string top_fold_status,
            cpp_string bottom_fold_status,
            cbool has_residual
        ) except +

    _TriBandedOperator * for_vector(SizedArray[double] &,
            Py_ssize_t blocks, Py_ssize_t derivative, Py_ssize_t axis,
            cbool barrier) except +
