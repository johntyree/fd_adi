
cdef extern from "<thrust/device_ptr.h>" namespace "thrust":
    cdef cppclass device_ptr[T]:
        device_ptr() nogil except +
        device_ptr(device_ptr&) nogil except +
        device_ptr(T *) nogil except +
        T* get() nogil except +
        T& operator[](size_t) nogil except +
        T& operator*(device_ptr&) nogil except +
        bint operator==(device_ptr&, device_ptr&) nogil
        bint operator!=(device_ptr&, device_ptr&) nogil
        bint operator<(device_ptr&, device_ptr&) nogil
        bint operator>(device_ptr&, device_ptr&) nogil
        bint operator<=(device_ptr&, device_ptr&) nogil
        bint operator>=(device_ptr&, device_ptr&) nogil
