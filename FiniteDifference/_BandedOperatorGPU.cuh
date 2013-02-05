
#ifndef _BandedOperatorGPU_cuh
#define _BandedOperatorGPU_cuh

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <thrust/tuple.h>

#define ENDL std::cout << std::endl

typedef double REAL_t;

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

    T *raw() {
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

typedef thrust::tuple<REAL_t,REAL_t,REAL_t> Triple;
typedef long int Py_ssize_t;

template <typename T>
void print_array(T *a, Py_ssize_t len) {
    std::ostream_iterator<T> out = std::ostream_iterator<T>(std::cout, " ");
    std::copy(a, a+len, out);
}

void transposeDiagonal(REAL_t *odata, REAL_t *idata, int width, int height);
void transposeNoBankConflicts(REAL_t *odata, REAL_t *idata, int width, int height);
void transposeNaive(REAL_t *odata, REAL_t *idata, int width, int height);


template <typename T>
std::ostream & operator<<(std::ostream &os, thrust::host_vector<T> const &v) {
    os << "addr(" << &v << ") size(" << v.size() << ")  [ ";
    std::ostream_iterator<T> out = std::ostream_iterator<T>(os, " ");
    std::copy(v.begin(), v.end(), out);
    return os << "]";
}

template <typename T>
std::ostream & operator<<(std::ostream &os, thrust::device_vector<T> const &v) {
    os << "addr(" << &v << ") size(" << v.size() << ")  [ ";
    std::ostream_iterator<T> out = std::ostream_iterator<T>(os, " ");
    std::copy(v.begin(), v.end(), out);
    return os << "]";
}



template <typename T>
void cout(T const &a) {
    std::cout << a;
}

template <typename T>
std::string to_string(T const &a) {
    std::ostringstream s;
    s << a;
    return s.str();
}



template<typename T>
struct SizedArray {
    HostVec<T> data;
    Py_ssize_t ndim;
    Py_ssize_t size;
    Py_ssize_t shape[8];
    std::string name;

    SizedArray(SizedArray<T> const &S)
        : data(S.data), ndim(S.ndim), size(S.size), name(S.name) {
            for (Py_ssize_t i = 0; i < ndim; ++i) {
                shape[i] = S.shape[i];
            }
            sanity_check();
            if (name == "R") {
                std::cout << "In copy ctor: " << data << "\n";
            }
    }

    SizedArray(thrust::host_vector<T> d, int ndim, intptr_t *s, std::string name)
        : data(d), ndim(ndim), size(1), name(name) {
            for (Py_ssize_t i = 0; i < ndim; ++i) {
                shape[i] = s[i];
                size *= shape[i];
            }
            sanity_check();
            if (name == "R") {
                std::cout << "In vect ctor: " << data << "\n";
            }
    }

    SizedArray(T *d, int ndim, intptr_t *s, std::string name)
        : ndim(ndim), size(1), name(name) {
            for (Py_ssize_t i = 0; i < ndim; ++i) {
                shape[i] = s[i];
                size *= shape[i];
            }
            data = thrust::host_vector<T>(d, d+size);
            sanity_check();
            if (name == "R") {
                std::cout << "In raw ctor: " << data << "\n";
            }
    }

    void sanity_check() {
        if (static_cast<Py_ssize_t>(data.size()) != size) {
            std::cout << "\ndata.size()("<<data.size()<<") != size("<<size<<")\n";
            assert(0);
        }
        for (int i = 0; i < ndim; ++i) {
            assert(shape[i] != 0);
        }
    }

    void reshape(Py_ssize_t h, Py_ssize_t w) {
        if (h*w != size) {
            std::cout << "Height("<<h<<") x Width("<<w<<") != Size("<<size<<")\n";
            assert(0);
        }
        shape[0] = h;
        shape[1] = w;
        ndim = 2;
    }

    T *raw() {
        return thrust::raw_pointer_cast(data.data());
    }

    void flatten() {
        shape[0] = size;
        shape[1] = 0;
        ndim = 1;
    }

    void transpose(int strategy) {
        assert (ndim == 2);
        //XXX
        GPUVec<T> in(data);
        thrust::fill(data.begin(), data.end(), 0);
        GPUVec<T> out(data);
        assert(in.size() == static_cast<size_t>(shape[0]*shape[1]));
        assert(out.size() == static_cast<size_t>(shape[0]*shape[1]));
        // ENDL;
        // std::cout << in << " " << shape[0] << ' ' << shape[1]; ENDL;
        if (strategy != 1) {
            std::cout << "Only accepting strategy 1 (NoBankConflicts)!\n";
            assert(0);
        }
        switch (strategy) {
            case 0:
                transposeDiagonal(out.raw(), in.raw(), shape[0], shape[1]);
                break;
            case 1:
                transposeNoBankConflicts(out.raw(), in.raw(), shape[0], shape[1]);
                break;
            case 2:
                transposeNaive(out.raw(), in.raw(), shape[0], shape[1]);
                break;
            default:
                std::cerr << "\nUnknown Transpose Strategy.\n";
                assert(0);
        }
        std::swap(shape[0], shape[1]);
        // std::cout << out << " " << shape[0] << ' ' << shape[1]; ENDL;
        thrust::copy(out.begin(), out.end(), data.begin());
    }

    std::string show() {
        std::string s0 = to_string(*this);
        std::string s1 = to_string(data);
        return s0 + " (" + s1 + ")";
    }


    inline T &operator()(int i) {
        assert (ndim >= 1);
        int idx = i;
        if (idx < 0 || size <= idx) {
            std::cout << std::endl;
            std::cout << name  << " idx("<<idx<<") not in range [0, Size("<<size<<"))\n";
            assert(0);
        }
        return data[idx];
    }

    inline T &operator()(int i, int j) {
        assert (ndim == 2);
        int idx = i * shape[1] + j;
        if (i >= shape[0]) {
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << name  << " i("<<i<<")"
                << "not in range [0, shape[0]("<<shape[0]<<")).\n";
            assert(0);
        } else if (j >= shape[1]) {
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << name  << " j("<<j<<")"
                << "not in range [0, shape[1]("<<shape[1]<<")).\n";
            assert(0);
        } else if (idx < 0 || size <= idx) {
            std::cout << std::endl;
            std::cout << "\nNot only are we out of range, but you wrote the"
                << " single-dimension tests wrong, obviously.\n";
            std::cout << name  << " idx("<<idx<<") not in range [0, Size("<<size<<"))\n";
            std::cout << std::endl;
            assert(0);
        }
        return data[idx];
    }

    inline T &idx(int i) {
        return operator()(i);
    }
    inline T &idx(int i, int j) {
        return operator()(i, j);
    }

};

template <typename T>
std::ostream & operator<<(std::ostream & os, SizedArray<T> const &sa) {
    return os << sa.name << ": addr("<<&sa<<") size("<<sa.size<<") ndim("<<sa.ndim<< ")";
}


class _BandedOperator {
    public:
        SizedArray<double> diags;
        SizedArray<double> R;
        SizedArray<double> high_dirichlet;
        SizedArray<double> low_dirichlet;
        SizedArray<double> top_factors;
        SizedArray<double> bottom_factors;
        SizedArray<int> offsets;

        void status();
        void verify_diag_ptrs();
        bool is_folded();
        SizedArray<double> *apply(SizedArray<double> &);
        int solve(SizedArray<double> &);
        void add_scalar(double val);
        void vectorized_scale(SizedArray<double> &vector);
        void add_operator(_BandedOperator &other);

        _BandedOperator(
            SizedArray<double> &data,
            SizedArray<double> &R,
            SizedArray<int> &offsets,
            SizedArray<double> &high_dirichlet,
            SizedArray<double> &low_dirichlet,
            SizedArray<double> &top_factors,
            SizedArray<double> &bottom_factors,
            unsigned int axis,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            bool has_high_dirichlet,
            bool has_low_dirichlet,
            bool has_residual
            );

    private:
        unsigned int axis;
        Py_ssize_t main_diag;
        Py_ssize_t operator_rows;
        Py_ssize_t blocks;
        Py_ssize_t block_len;
        double *sup, *mid, *sub;
        bool has_high_dirichlet;
        bool has_low_dirichlet;
        bool has_residual;
};

#endif
