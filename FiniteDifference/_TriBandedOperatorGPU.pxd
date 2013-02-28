
from libcpp cimport bool as cbool
from libcpp.string cimport string as cpp_string
from VecArray cimport SizedArray, GPUVec, to_string

cdef extern from "_TriBandedOperatorGPU.cuh":

    cdef cppclass _TriBandedOperator:
        cbool is_folded() except +
        cbool has_residual
        cbool bottom_is_folded
        cbool top_is_folded
        SizedArray[double] *apply(SizedArray[double] &) except +
        void add_scalar(double val) except +
        void vectorized_scale(SizedArray[double] &vector) except +
        void add_operator(_TriBandedOperator &other) except +
        Py_ssize_t operator_rows
        Py_ssize_t block_len
        Py_ssize_t blocks
        void undiagonalize() except +
        void diagonalize() except +
        void fold_bottom(cbool unfold) except +
        void fold_top(cbool unfold) except +
        int solve(SizedArray[double] &) except +
        SizedArray[double] bottom_factors
        SizedArray[double] top_factors
        SizedArray[double] high_dirichlet,
        SizedArray[double] low_dirichlet,
        SizedArray[double] diags
        SizedArray[double] R
        cbool has_high_dirichlet,
        cbool has_low_dirichlet,
        _TriBandedOperator(
            SizedArray[double] data,
            SizedArray[double] R,
            SizedArray[double] high_dirichlet,
            SizedArray[double] low_dirichlet,
            SizedArray[double] top_factors,
            SizedArray[double] bottom_factors,
            unsigned int axis,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            cbool has_high_dirichlet,
            cbool has_low_dirichlet,
            cbool top_is_folded,
            cbool bottom_is_folded,
            cbool has_residual
        ) except +
