
#ifndef _BandedOperatorGPU_cuh
#define _BandedOperatorGPU_cuh

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <sstream>
#include <algorithm>

#define ENDL std::cout << std::endl

typedef double REAL_t;
typedef long int Py_ssize_t;

template <typename T>
void print_array(T *a, Py_ssize_t len) {
    std::ostream_iterator<T> out = std::ostream_iterator<T>(std::cout, " ");
    std::copy(a, a+len, out);
}

void transposeDiagonal(REAL_t *odata, REAL_t *idata, int width, int height);
void transposeNoBankConflicts(REAL_t *odata, REAL_t *idata, int width, int height);
void transposeNaive(REAL_t *odata, REAL_t *idata, int width, int height);

namespace CPU {

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
T *raw(thrust::host_vector<T> &v) {
   return thrust::raw_pointer_cast(v.data());
}

template<typename T>
T *raw(thrust::device_vector<T> &v) {
   return thrust::raw_pointer_cast(v.data());
}

template<typename T>
struct SizedArray {
    /* T *data; */
    thrust::host_vector<T> data;
    Py_ssize_t ndim;
    Py_ssize_t size;
    Py_ssize_t shape[8];
    SizedArray(T *d, int ndim, intptr_t *s)
        : ndim(ndim), size(1) {
            for (Py_ssize_t i = 0; i < ndim; ++i) {
                shape[i] = s[i];
                size *= shape[i];
            }
            data = thrust::host_vector<T>(d, d+size);
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

    void flatten() {
        shape[0] = size;
        shape[1] = 0;
        ndim = 1;
    }

    void transpose(int strategy) {
        assert (ndim == 2);
        //XXX
        thrust::device_vector<T> in(data);
        thrust::fill(data.begin(), data.end(), 0);
        thrust::device_vector<T> out(data);
        assert(in.size() == static_cast<size_t>(shape[0]*shape[1]));
        assert(out.size() == static_cast<size_t>(shape[0]*shape[1]));
        ENDL;
        std::cout << in << " " << shape[0] << ' ' << shape[1]; ENDL;
        switch (strategy) {
            case 0:
                transposeDiagonal(raw(out), raw(in), shape[0], shape[1]);
                break;
            case 1:
                transposeNoBankConflicts(raw(out), raw(in), shape[0], shape[1]);
                break;
            case 2:
                transposeNaive(raw(out), raw(in), shape[0], shape[1]);
                break;
            default:
                std::cerr << "\nUnknown Transpose Strategy.\n";
                assert(0);
        }
        std::swap(shape[0], shape[1]);
        std::cout << out << " " << shape[0] << ' ' << shape[1]; ENDL;
        thrust::copy(out.begin(), out.end(), data.begin());
    }

    std::string to_string() {
        std::string s0 = CPU::to_string(*this);
        std::string s1 = CPU::to_string(data);
        return s0 + " (" + s1 + ")";
    }


    inline T &operator()(int i) {
        assert (ndim >= 1);
        int idx = i;
        assert (0 <= idx && idx < size);
        return data[idx];
    }

    inline T &operator()(int i, int j) {
        assert (ndim == 2);
        int idx = i * shape[1] + j;
        if (idx < 0 || size <= idx) {
            std::cout << "Index: " << idx << " larger than size: " << size <<
                "." << std::endl;
            assert(0);
        }
        return data[idx];
    }

    inline T &idx(int i) {
        return operator()(i);
    }

};

template <typename T>
std::ostream & operator<<(std::ostream & os, SizedArray<T> const &sa) {
    return os << "addr(" << &sa << ") size(" << sa.size << ") ndim(" << sa.ndim << ")";
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
        int apply(SizedArray<double> &, bool);
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
        Py_ssize_t blocks;
        Py_ssize_t operator_rows;
        Py_ssize_t block_len;
        Py_ssize_t main_diag;
        bool has_high_dirichlet;
        bool has_low_dirichlet;
        bool has_residual;
        unsigned int axis;
        double *sub, *mid, *sup;
};


} // namespace CPU

#endif
