
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#define ENDL std::cout << std::endl

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/version.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <algorithm>
#include <cstdlib>

#include <iostream>
#include <vector>
#include <iterator>
#include <cassert>
#include <ctime>
#include <stdexcept>

#include <sys/select.h>
#include <sys/time.h>

#include "_BandedOperatorGPU.cuh"

template <typename T>
void print_array(T *a, Py_ssize_t len) {
    std::ostream_iterator<T> out = std::ostream_iterator<T>(std::cout, " ");
    std::copy(a, a+len, out);
}

template <typename T, typename U>
int find_index(T haystack, U needle, int max) {
    int idx;
    for (idx = 0; idx < max; ++idx) {
        if (haystack[idx] == needle) break;
    }
    if (idx >= max) idx = -1;
    return idx;
}

namespace CPU {

_BandedOperator::_BandedOperator(
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
        ) :
    data(data),
    R(R),
    offsets(offsets),
    high_dirichlet(high_dirichlet),
    low_dirichlet(low_dirichlet),
    top_factors(top_factors),
    bottom_factors(bottom_factors),
    axis(axis),
    main_diag(find_index(offsets.data, 0, offsets.size)),
    operator_rows(operator_rows),
    blocks(blocks),
    block_len(operator_rows / blocks),
    has_high_dirichlet(has_high_dirichlet),
    has_low_dirichlet(has_low_dirichlet),
    has_residual(has_residual)
    { }

void _BandedOperator::apply(
        SizedArray<double> V,
        bool overwrite) {
    assert(overwrite);

    if (axis == 0) {
        // Transpose somehow
    }

    if (has_low_dirichlet) {
        /* print "Setting V[0,:] to", self.dirichlet[0] */
        // Some kind of thrust thing?
        /* V[...,0] = low_dirichlet[i] */
    }
    if (has_high_dirichlet) {
        /* print "Setting V[0,:] to", self.dirichlet[0] */
        // Some kind of thrust thing?
        /* V[...,-1] = high_dirichlet[i] */
    }

    if (is_folded()) {
        /* ret = fold_vector(self.D.dot(V.flat), unfold=True) */
    } else {
        /* ret = self.D.dot(V.flat) */
    }

    if (has_residual) {
        /* ret += self.R; */
    }

    /* ret = ret.reshape(V.shape) */

    /* t = range(V.ndim) */
    /* utils.rolllist(t, V.ndim-1, self.axis) */

    // Transpose back
    /* return ret; */
}


struct periodic_from_to_mask : thrust::unary_function<int, bool> {
    int begin;
    int end;
    int period;

    periodic_from_to_mask(int begin, int end, int period)
        : begin(begin-1), end(end+1), period(period) {
        }

    __host__ __device__
    bool operator()(int idx) {
        return (idx % period != begin && idx % period != end);
    }
};

void _BandedOperator::add_operator(_BandedOperator &other) {
        /*
         * Add a second BandedOperator to this one.
         * Does not alter self.R, the residual vector.
         */
        /* cdef double[:,:] data = self.D.data */
        /* cdef int[:] selfoffsets = np.array(self.D.offsets) */
        /* cdef int[:] otheroffsets = np.array(other.D.offsets) */
        /* cdef unsigned int num_otheroffsets = otheroffsets.shape[0] */
        /* cdef np.ndarray[double, ndim=2] newdata */
        /* cdef int[:] Boffsets */
        /* cdef int o */
        /* cdef unsigned int i */
        /* cdef BandedOperator B */
        /* cdef cbool fail */

        timeval sleeptime = {1, 0};

        int begin = has_low_dirichlet;
        int end = block_len-1 - has_high_dirichlet;
        int to, fro;
        for (int row = 0; row < other.offsets.size; row++) {
            fro = row;
            to = find_index(offsets.data, other.offsets(row), offsets.size);
            if (offsets(to) != other.offsets(fro)) {
                std::cout << std::endl;
                std::cout << "to: " << to << "(";
                print_array(&offsets(0), offsets.size);
                std::cout << ")";
                std::cout << "fro: " << fro << "(";
                print_array(&other.offsets(0), other.offsets.size);
                std::cout << ")" << std::endl;
                select(1, NULL, NULL, NULL, &sleeptime);
                assert (offsets(to) == other.offsets(fro));
            }

            if (offsets(fro) == 0) {
                thrust::transform_if(
                        &data(to, 0),
                        &data(to, 0) + operator_rows,
                        &other.data(fro, 0),
                        thrust::make_counting_iterator(0),
                        &data(to, 0),
                        thrust::plus<double>(),
                        periodic_from_to_mask(begin, end, block_len));
            } else {
                thrust::transform(
                        &other.data(fro, 0),
                        &other.data(fro, 0) + operator_rows,
                        &data(to, 0),
                        &data(to, 0),
                        thrust::plus<double>());
            }
        }
        thrust::transform(
                R.data.begin(),
                R.data.end(),
                other.R.data.begin(),
                R.data.begin(),
                thrust::plus<double>());





        /* for i in range(num_otheroffsets): */
            /* fro = i */
            /* o = otheroffsets[i] */
            /* to = get_int_index(Boffsets, o) */
            /* if o == 0: */
                /* # We have to do the main diagonal block_wise because of the */
                /* # dirichlet boundary */
                /* block_len = B.shape[0] / float(B.blocks) */
                /* assert block_len == int(block_len) */
                /* for i in range(B.blocks): */
                    /* begin = i*block_len */
                    /* if B.dirichlet[0] is not None: */
                        /* begin += 1 */
                    /* end = i*block_len + block_len */
                    /* if B.dirichlet[1] is not None: */
                        /* end -= 1 */
                    /* B.D.data[to,begin:end] += other.D.data[fro,begin:end] */
            /* else: */
                /* begin = 0 */
                /* end = B.D.data.shape[1] */
                /* B.D.data[to,begin:end] += other.D.data[fro,begin:end] */
        /* # Now the residual vector from the other one */
        /* if other.R is not None: */
            /* if B.R is None: */
                /* B.R = other.R.copy() */
            /* else: */
                /* B.R += other.R */

        /* return B */
}



void _BandedOperator::add_scalar(double val) {
    /* Add a scalar to the main diagonal.
     * Does not alter the residual vector.
     */
    // We add it to the main diagonal.

    int begin = has_low_dirichlet;
    int end = block_len-1 - has_high_dirichlet;

    assert(main_diag < offsets.size);

    thrust::transform_if(
            &data(main_diag, 0),
            &data(main_diag, operator_rows),
            thrust::make_constant_iterator(val),
            thrust::make_counting_iterator(0),
            &data(main_diag, 0),
            thrust::plus<double>(),
            periodic_from_to_mask(begin, end, block_len));

}



bool _BandedOperator::is_folded() {
    return false;
}

void _BandedOperator::status() {
    /* private: */
        /* Py_ssize_t blocks; */
        /* Py_ssize_t operator_rows; */
        /* Py_ssize_t block_len; */
        /* Py_ssize_t main_diag; */
        /* bool has_high_dirichlet; */
        /* bool has_low_dirichlet; */
        /* bool has_residual; */
        /* unsigned int axis; */
        /* SizedArray<double> data; */
        /* SizedArray<double> R; */
        /* SizedArray<double> high_dirichlet; */
        /* SizedArray<double> low_dirichlet; */
        /* SizedArray<double> top_factors; */
        /* SizedArray<double> bottom_factors; */

    std::cout << "Status of: " << this << std::endl;
    std::cout << "C_data:    " << data << std::endl;
    std::cout << "C_R:       " << R << std::endl;
    std::cout << "C_offests: " << offsets << std::endl;
    std::cout << "C_axis:    " << axis << std::endl;
    /* std::cout << "C_<++>: " << &this-><++> << std::endl; */
    /* std::cout << "C_<++>: " << &this-><++> << std::endl; */
    /* std::cout << "C_<++>: " << &this-><++> << std::endl; */
    /* std::cout << "C_<++>: " << &this-><++> << std::endl; */
    /* std::cout << "C_OFFSETS: " << &this->offsets << std::endl; */
}

void _BandedOperator::vectorized_scale(SizedArray<double> &vector) {
    Py_ssize_t vsize = vector.size;
    Py_ssize_t noffsets = offsets.size;
    Py_ssize_t block_len = operator_rows / blocks;

    assert(operator_rows % vsize == 0);

    if (has_low_dirichlet) {
        for (Py_ssize_t b = 0; b < blocks; ++b) {
            vector(b*block_len % vsize) = 1;
        }
    }

    if (has_high_dirichlet) {
        for (Py_ssize_t b = 0; b < blocks; ++b) {
            vector((b+1)*block_len - 1 % vsize) = 1;
        }
    }

    for (Py_ssize_t row = 0; row < noffsets; ++row) {
        int o = offsets(row);
        if (o >= 0) { // upper diags
            for (int i = 0; i < (int)operator_rows - o; ++i) {
                data(row, i+o) *= vector(i % vsize);
            }
        } else { // lower diags
            for (int i = -o; i < (int)operator_rows; ++i) {
                data(row, i+o) *= vector(i % vsize);
            }
        }
    }

    for (Py_ssize_t i = 0; i < operator_rows; ++i) {
        R(i) *= vector(i % vsize);
    }
    return;
}

} // namespace CPU

int main () {

    thrust::host_vector<double> a(10);
    int block_len = 5;
    int begin = 1;
    int end = block_len-1 - 1;

    thrust::transform_if(
            a.begin(),
            a.end(),
            thrust::make_constant_iterator(2),
            thrust::make_counting_iterator(0),
            a.begin(),
            thrust::plus<double>(),
            CPU::periodic_from_to_mask(begin, end, block_len));

    printf("\n");
    print_array(a.data(), a.size());
    time_t theTime;
    time(&theTime);   // get the calendar time
    tm *t = localtime( &theTime );  // convert to local
    std::cout << "The time is: " << asctime(t);
    return 0;
}
