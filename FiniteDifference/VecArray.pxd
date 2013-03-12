
from libcpp.string cimport string as cpp_string
from libcpp cimport bool as cbool

cimport numpy as np

from FiniteDifference.thrust.device_vector cimport device_vector
from FiniteDifference.thrust.device_ptr cimport device_ptr

cdef extern from "VecArray.h":

    cdef cppclass GPUVec[T](device_vector):
        GPUVec(GPUVec) except +
        T* raw() except +


    cdef cppclass SizedArray[T]:
        cbool owner
        device_ptr[T] data
        Py_ssize_t size
        Py_ssize_t ndim
        Py_ssize_t[8] shape
        cpp_string name
        SizedArray() except +
        SizedArray(Py_ssize_t, cpp_string) except +
        SizedArray(SizedArray[T], cbool) except +
        SizedArray(T*, Py_ssize_t, cpp_string name, cbool) except +
        SizedArray(T*, int, np.npy_intp*, cpp_string name, cbool) except +
        void pluseq(SizedArray[T] &x) except +
        void minuseq(SizedArray[T] &x) except +
        void timeseq(SizedArray[T] &x) except +
        void pluseq_scalar(T x) except +
        void minuseq_scalar(T x) except +
        void timeseq_scalar(T x) except +
        T get(int i) except +
        T get(int i, int j) except +
        void reshape(Py_ssize_t h, Py_ssize_t w) except +
        void flatten() except +
        void transpose(int) except +
        cpp_string show() except +

    void cout(void *) except +
    cpp_string to_string(void  *) except +
    cpp_string to_string(cbool) except +
    cpp_string to_string(SizedArray[double]) except +
    cpp_string to_string(SizedArray[int]) except +

