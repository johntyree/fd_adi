cdef extern from "<thrust/device_vector.h>" namespace "thrust":
    cdef cppclass device_vector[T]:
        cppclass iterator:
            T& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
            bint operator<(iterator) nogil
            bint operator>(iterator) nogil
            bint operator<=(iterator) nogil
            bint operator>=(iterator) nogil
        cppclass reverse_iterator:
            T& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(reverse_iterator) nogil
            bint operator!=(reverse_iterator) nogil
            bint operator<(reverse_iterator) nogil
            bint operator>(reverse_iterator) nogil
            bint operator<=(reverse_iterator) nogil
            bint operator>=(reverse_iterator) nogil
        #cppclass const_iterator(iterator):
        #    pass
        #cppclass const_reverse_iterator(reverse_iterator):
        #    pass
        device_vector() nogil except +
        device_vector(device_vector&) nogil except +
        device_vector(size_t) nogil except +
        device_vector(size_t, T&) nogil except +
        device_vector(T*, T*) nogil except +
        T& operator[](size_t) nogil
        #device_vector& operator=(device_vector&)
        bint operator==(device_vector&, device_vector&) nogil
        bint operator!=(device_vector&, device_vector&) nogil
        bint operator<(device_vector&, device_vector&) nogil
        bint operator>(device_vector&, device_vector&) nogil
        bint operator<=(device_vector&, device_vector&) nogil
        bint operator>=(device_vector&, device_vector&) nogil
        void assign(size_t, T&) nogil
        #void assign[input_iterator](input_iterator, input_iterator)
        T& at(size_t) nogil
        T& back() nogil
        iterator begin() nogil
        #const_iterator begin()
        size_t capacity() nogil
        void clear() nogil
        bint empty() nogil
        iterator end() nogil
        #const_iterator end()
        iterator erase(iterator) nogil
        iterator erase(iterator, iterator) nogil
        T& front() nogil
        iterator insert(iterator, T&) nogil
        void insert(iterator, size_t, T&) nogil
        void insert(iterator, iterator, iterator) nogil
        size_t max_size() nogil
        void pop_back() nogil
        void push_back(T&) nogil
        reverse_iterator rbegin() nogil
        #const_reverse_iterator rbegin()
        reverse_iterator rend() nogil
        #const_reverse_iterator rend()
        void reserve(size_t) nogil
        void resize(size_t) nogil
        void resize(size_t, T&) nogil
        size_t size() nogil
        void swap(device_vector&) nogil
