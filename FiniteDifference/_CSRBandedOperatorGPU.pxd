

from libcpp.string cimport string as cpp_string
from libcpp cimport bool as cbool
from VecArray cimport SizedArray


cdef extern from "_CSRBandedOperatorGPU.cuh":

    cdef cppclass _CSRBandedOperator:

        unsigned int axis,
        Py_ssize_t operator_rows,
        Py_ssize_t blocks,
        cbool has_high_dirichlet,
        cbool has_low_dirichlet,
        cpp_string top_fold_status,
        cpp_string bottom_fold_status,
        cbool has_residual,
        cpp_string name
        SizedArray[double] data
        SizedArray[int] row_ptr
        SizedArray[int] row_ind
        SizedArray[int] col_ind
        SizedArray[double] &R,
        SizedArray[double] &high_dirichlet,
        SizedArray[double] &low_dirichlet,
        SizedArray[double] &top_factors,
        SizedArray[double] &bottom_factors,
        void apply(SizedArray[double] &) except +
        void solve(SizedArray[double] &) except +
        void fake_solve(SizedArray[double] &) except +
        void vectorized_scale(SizedArray[double] &vector) except +


        _CSRBandedOperator(
            SizedArray[double] &data,
            SizedArray[int] &row_ptr,
            SizedArray[int] &row_ind,
            SizedArray[int] &col_ind,
            SizedArray[double] &R,
            SizedArray[double] &high_dirichlet,
            SizedArray[double] &low_dirichlet,
            SizedArray[double] &top_factors,
            SizedArray[double] &bottom_factors,
            unsigned int axis,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            cbool has_high_dirichlet,
            cbool has_low_dirichlet,
            cpp_string top_fold_status,
            cpp_string bottom_fold_status,
            cbool has_residual,
            cpp_string name
        ) except +

    _CSRBandedOperator * mixed_for_vector(SizedArray[double] &, SizedArray[double] &) except +
