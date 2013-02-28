
from libcpp.string cimport string as cpp_string

cimport numpy as np

from FiniteDifference.thrust.device_vector cimport device_vector

cdef extern from "VecArray.h":

    cdef cppclass GPUVec[T](device_vector):
        GPUVec(GPUVec)
        T* raw()


    cdef cppclass SizedArray[T]:
        GPUVec[T] data
        Py_ssize_t size
        Py_ssize_t ndim
        Py_ssize_t[8] shape
        cpp_string name
        SizedArray(SizedArray[T]) except +
        SizedArray(T*, int, np.npy_intp*, cpp_string name) except +
        T get(int i) except +
        T get(int i, int j) except +
        void reshape(Py_ssize_t h, Py_ssize_t w) except +
        void flatten() except +
        void transpose(int) except +
        cpp_string show()

    void cout(void *)
    cpp_string to_string(void  *)
    cpp_string to_string(cbool)
    cpp_string to_string(SizedArray[double])
    cpp_string to_string(SizedArray[int])

