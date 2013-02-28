
from libcpp.string cimport string as cpp_string
from VecArray cimport GPUVec, SizedArray

cdef extern from "_CSRBandedOperatorGPU.cuh":

    cdef cppclass _CSRBandedOperator:
        Py_ssize_t operator_rows
        Py_ssize_t blocks
        cpp_string name
        GPUVec[double] data
        GPUVec[int] row_ptr
        GPUVec[int] row_ind
        GPUVec[int] col_ind
        SizedArray[double] *apply(SizedArray[double] &) except +
        void vectorized_scale(SizedArray[double] &vector) except +


        _CSRBandedOperator(
            GPUVec[double] &data,
            GPUVec[int] &row_ptr,
            GPUVec[int] &row_ind,
            GPUVec[int] &col_ind,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            cpp_string name
        ) except +


