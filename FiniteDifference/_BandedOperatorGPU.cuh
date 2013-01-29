
#include <thrust/host_vector.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

typedef double REAL_t;
typedef long int Py_ssize_t;

namespace CPU {

template <typename T>
void cout(T a) {
    std::cout << a;
}


void transposeDiagonal(REAL_t *odata, REAL_t *idata, int width, int height);

template<typename T>
T *raw(thrust::host_vector<T> &v) {
   return thrust::raw_pointer_cast(v.data());
}

template<typename T>
struct SizedArray {
    /* T *data; */
    thrust::host_vector<T> data;
    Py_ssize_t size;
    const Py_ssize_t ndim;
    Py_ssize_t shape[8];
    SizedArray(T *d, int ndim, intptr_t *s)
        : ndim(ndim), size(1) {
            for (Py_ssize_t i = 0; i < ndim; ++i) {
                shape[i] = s[i];
                size *= shape[i];
            }
            data = thrust::host_vector<T>(size);
            thrust::copy(d, d+size, data.begin());
    }

    void transpose() {
        assert (ndim == 2);
        transposeDiagonal(raw(data), raw(data), shape[0], shape[1]);
    }


    inline T &operator()(int i) {
        assert (ndim == 1);
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
