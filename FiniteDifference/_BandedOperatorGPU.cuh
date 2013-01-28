
#include <thrust/host_vector.h>

typedef long int Py_ssize_t;

namespace CPU {

template <typename T>
void cout(T a) {
    std::cout << a;
}


template<typename T>
struct SizedArray {
    /* T *data; */
    thrust::host_vector<T> data;
    Py_ssize_t size;
    const Py_ssize_t ndim;
    Py_ssize_t shape[8];
    SizedArray(T *d, Py_ssize_t ndim, Py_ssize_t *s)
        : ndim(ndim), size(1) {
            for (Py_ssize_t i = 0; i < ndim; ++i) {
                shape[i] = s[i];
                size *= shape[i];
            }
            data = thrust::host_vector<T>(size);
            thrust::copy(d, d+size, data.begin());
    }

    inline T &operator()(long i) {
        assert (ndim == 1);
        long idx = i;
        assert (0 <= idx && idx < size);
        return data[idx];
    }

    inline T &operator()(long i, long j) {
        assert (ndim == 2);
        long idx = i * shape[1] + j;
        assert (0 <= idx && idx < size);
        return data[idx];
    }

};

template <typename T>
std::ostream & operator<<(std::ostream & os, SizedArray<T> const &sa) {
    return os << "addr(" << &sa << ") size(" << sa.size << ") ndim(" << sa.ndim << ")";
}

class _BandedOperator {

    private:
        Py_ssize_t blocks;
        Py_ssize_t operator_rows;
        Py_ssize_t block_len;
        Py_ssize_t main_diag;
        bool has_high_dirichlet;
        bool has_low_dirichlet;
        bool has_residual;
        unsigned int axis;

    public:
        SizedArray<double> data;
        SizedArray<double> R;
        SizedArray<double> high_dirichlet;
        SizedArray<double> low_dirichlet;
        SizedArray<double> top_factors;
        SizedArray<double> bottom_factors;
        SizedArray<int> offsets;

        void view();
        void status();
        bool is_folded();
        void apply(SizedArray<double>, bool);
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
};


} // namespace CPU
