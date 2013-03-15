

from libcpp.string cimport string as cpp_string
from VecArray cimport SizedArray


cdef extern from "_CSRBandedOperatorGPU.cuh":

    cdef cppclass _CSRBandedOperator:

        Py_ssize_t operator_rows
        Py_ssize_t blocks
        cpp_string name
        SizedArray[double] data
        SizedArray[int] row_ptr
        SizedArray[int] row_ind
        SizedArray[int] col_ind
        void apply(SizedArray[double] &) except +
        void vectorized_scale(SizedArray[double] &vector) except +


        _CSRBandedOperator(
            SizedArray[double] &data,
            SizedArray[int] &row_ptr,
            SizedArray[int] &row_ind,
            SizedArray[int] &col_ind,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            cpp_string name
        ) except +


