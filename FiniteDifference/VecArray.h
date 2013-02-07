#ifndef VECARRAY_H
#define VECARRAY_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>

#include "_kernels.h"

template <typename T>
struct GPUVec : thrust::device_vector<T> {

    GPUVec() : thrust::device_vector<T>() {}

    template<typename X>
    GPUVec(const X &x)
        : thrust::device_vector<T>(x) {}

    template<typename X, typename Y>
    GPUVec(const X &x, const Y &y)
        : thrust::device_vector<T>(x, y) {}

    template<typename X, typename Y, typename Z>
    GPUVec(const X &x, const Y &y, const Z &z)
        : thrust::device_vector<T>(x, y, z) {}

    inline thrust::device_reference<T> ref() {
        return thrust::device_reference<T>(ptr());
    }

    inline thrust::device_ptr<T> ptr() {
        return thrust::device_ptr<T>(raw());
    }

    inline T *raw() {
        return thrust::raw_pointer_cast(this->data());
    }
};

template <typename T>
struct HostVec : thrust::host_vector<T> {

    HostVec() : thrust::host_vector<T>() {}

    template<typename X>
    HostVec(const X &x)
        : thrust::host_vector<T>(x) {}

    template<typename X, typename Y>
    HostVec(const X &x, const Y &y)
        : thrust::host_vector<T>(x, y) {}

    template<typename X, typename Y, typename Z>
    HostVec(const X &x, const Y &y, const Z &z)
        : thrust::host_vector<T>(x, y, z) {}

    T *raw() {
        return thrust::raw_pointer_cast(this->data());
    }
};


template <typename T>
void print_array(T *a, Py_ssize_t len) {
    std::ostream_iterator<T> out = std::ostream_iterator<T>(std::cout, " ");
    std::copy(a, a+len, out);
}

template <typename T>
std::ostream & operator<<(std::ostream &os, thrust::host_vector<T> const &v) {
    os << "HOST addr(" << &v << ") size(" << v.size() << ")  [ ";
    std::ostream_iterator<T> out = std::ostream_iterator<T>(os, " ");
    std::copy(v.begin(), v.end(), out);
    return os << "]";
}

template <typename T>
std::ostream & operator<<(std::ostream &os, thrust::device_vector<T> const &v) {
    os << "DEVICE addr(" << &v << ") size(" << v.size() << ")  [ ";
    std::ostream_iterator<T> out = std::ostream_iterator<T>(os, " ");
    std::copy(v.begin(), v.end(), out);
    return os << "]";
}

template<typename T>
struct SizedArray {
    GPUVec<T> data;
    Py_ssize_t ndim;
    Py_ssize_t size;
    Py_ssize_t shape[8];
    std::string name;

    SizedArray(Py_ssize_t size, std::string name)
        : data(size), size(size), name(name) {
            sanity_check();
    }

    SizedArray(SizedArray<T> const &S)
        : data(S.data), ndim(S.ndim), size(S.size), name(S.name) {
            for (Py_ssize_t i = 0; i < ndim; ++i) {
                shape[i] = S.shape[i];
            }
            sanity_check();
    }

    SizedArray(thrust::host_vector<T> d, int ndim, intptr_t *s, std::string name)
        : data(d), ndim(ndim), size(1), name(name) {
            for (Py_ssize_t i = 0; i < ndim; ++i) {
                shape[i] = s[i];
                size *= shape[i];
            }
            sanity_check();
    }

    SizedArray(T *rawptr, int ndim, intptr_t *s, std::string name)
        : ndim(ndim), size(1), name(name) {
            for (Py_ssize_t i = 0; i < ndim; ++i) {
                shape[i] = s[i];
                size *= shape[i];
            }
            data.assign(rawptr, rawptr+size);
            sanity_check();
    }

    void sanity_check() {
        if (static_cast<Py_ssize_t>(data.size()) != size) {
            LOG("\ndata.size()("<<data.size()<<") != size("<<size<<")");
            assert(false);
        }
        for (int i = 0; i < ndim; ++i) {
            if (shape[i] == 0) {
                LOG("shape["<<i<<"] is "<<i<<"... ndim("<<ndim<<")");
                assert(false);
            }
        }
    }

    void reshape(Py_ssize_t h, Py_ssize_t w) {
        if (h*w != size) {
            LOG("Height("<<h<<") x Width("<<w<<") != Size("<<size<<")");
            assert(false);
        }
        shape[0] = h;
        shape[1] = w;
        ndim = 2;
    }

    void flatten() {
        shape[0] = size;
        shape[1] = 0;
        ndim = 1;
    }

    void transpose(int strategy) {
        assert (ndim == 2);
        //XXX
        thrust::device_ptr<double> out = thrust::device_malloc<double>(data.size());
        switch (strategy) {
            case 1:
                transposeNoBankConflicts(out.get(), data.raw(), shape[0], shape[1]);
                break;
            default:
                std::cerr << "\nUnknown Transpose Strategy.\n";
                assert(false);
        }
        reshape(shape[1], shape[0]);
        data.assign(out, out+size);
        thrust::device_free(out);
    }

    std::string show() {
        std::string s0 = to_string(*this);
        std::string s1 = to_string(data);
        return s0 + " (" + s1 + ")";
    }


    inline int idx(int i) {
        assert (ndim >= 1);
        int idx = i;
        if (idx < 0 || size <= idx) {
            LOG(name  << " idx("<<idx<<") not in range [0, Size("<<size<<"))");
            assert(0);
        }
        return idx;
    }

    inline int idx(int i, int j) {
        assert (ndim == 2);
        int idx = i * shape[1] + j;
        if (i >= shape[0]) {
            LOG(name  << " i("<<i<<")"
                << "not in range [0, shape[0]("<<shape[0]<<")).");
            assert(0);
        } else if (j >= shape[1]) {
            LOG(name  << " j("<<j<<")"
                << "not in range [0, shape[1]("<<shape[1]<<")).");
            assert(0);
        } else if (idx < 0 || size <= idx) {
            LOG("\nNot only are we out of range, but you wrote the"
                << " single-dimension tests wrong, obviously.");
            LOG(name  << " i("<<i<<") j("<<j<<") Shape("
                <<shape[0]<<','<<shape[1]<<") idx("<<idx
                <<") not in range [0, Size("<<size<<"))\n");
            assert(0);
        }
        return idx;
    }

    inline void set(int i, T x) {
        data[idx(i)] = x;
    }
    inline T get(int i) {
        return data[idx(i)];
    }

    inline void set(int i, int j, T x) {
        data[idx(i, j)] = x;
    }
    inline T get(int i, int j) {
        return data[idx(i, j)];
    }

};

template <typename T>
std::ostream & operator<<(std::ostream & os, SizedArray<T> const &sa) {
    return os << sa.name << ": addr("<<&sa<<") size("
        <<sa.size<<") ndim("<<sa.ndim<< ") ["
        << sa.data << " ]";
}


#endif /* end of include guard */
