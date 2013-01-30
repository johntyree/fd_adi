
#include "GNUC_47_compat.h"

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

#include <cusparse_v2.h>

#include "_BandedOperatorGPU.cuh"

__global__
void d_transposeDiagonal(
    REAL_t *odata, REAL_t *idata, int width, int height) {
    __shared__ REAL_t tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    // diagonal reordering
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }
    int xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;
    xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y+i][threadIdx.x] =
            idata[index_in+i*width];
    }
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out+i*height] =
            tile[threadIdx.x][threadIdx.y+i];
    }
}

void transposeDiagonal(REAL_t *odata, REAL_t *idata,
        int width, int height) {
    dim3 grid(width/TILE_DIM, height/TILE_DIM);
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    d_transposeDiagonal<<<grid, threads>>>(odata, idata, width, height);
}


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
    sup_p(raw(data.data)),
    mid_p(raw(data.data) + operator_rows),
    sub_p(raw(data.data) + 2*operator_rows),
    has_high_dirichlet(has_high_dirichlet),
    has_low_dirichlet(has_low_dirichlet),
    has_residual(has_residual)
    { }

void _BandedOperator::verify_diag_ptrs() {
    std::cout << "Operator Rows: " << operator_rows << "\tdata.size: " <<
        data.size;
    ENDL;
    for (int i = 0; i < operator_rows; i++) {
        if (sup_p[i] != data(main_diag-1, i)) {
            std::cout << "sup_p @ " << i << " = " << sup_p[i] << " !=  " << data(main_diag-1,i);
            ENDL;
            assert(0);
        }
        if (mid_p[i] != data(main_diag, i)) {
            std::cout << "mid_p @ " << i << " = " << mid_p[i] << " !=  " << data(main_diag,i);
            ENDL;
            assert(0);
        }
        if (sub_p[i] != data(main_diag+1, i)) {
            std::cout << "sub_p @ " << i << " = " << sub_p[i] << " !=  " << data(main_diag+1,i);
            ENDL;
            assert(0);
        }
    }
}


int _BandedOperator::apply(
        SizedArray<double> &V,
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
    return 0;
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

        timeval sleeptime = {1, 0};

        int begin = has_low_dirichlet;
        int end = block_len-1 - has_high_dirichlet;
        int o, to, fro;
        for (int i = 0; i < other.offsets.size; i++) {
            fro = i;
            o = other.offsets(i);
            to = find_index(offsets.data, o, offsets.size);
            if (offsets(to) != o) {
                std::cout << std::endl;
                std::cout << "to: " << to << "(";
                print_array(&offsets(0), offsets.size);
                std::cout << ")";
                std::cout << "fro: " << fro << "(";
                print_array(&other.offsets(0), other.offsets.size);
                std::cout << ")" << std::endl;
                select(1, NULL, NULL, NULL, &sleeptime);
                assert (offsets(to) == o);
            }
            if (o == 0) {
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
                        &other.data(fro, 0) + other.data.shape[1],
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

int _BandedOperator::solve(SizedArray<double> &V) {
    std::cout << "Begin C Solve\n";
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE Library initialization failed." << std::endl;
        return 1;
    }


    thrust::device_vector<double> d_V(V.size);
    std::cout << "Copy Host->Dev... ";
    thrust::copy(V.data.begin(), V.data.end(), d_V.begin());
    std::cout << "OK\n";
    /* verify_diag_ptrs(); */
    std::cout << "CUSPARSE... ";
    /* status = cusparseDgtsvStridedBatch(handle, V.size, */
            /* sub_p, mid_p, sup_p, */
            /* raw(d_V), */
            /* 1, V.size); */
    std::cout << "OK\n";
    /* cudaDeviceSynchronize(); */
    /* if (status != CUSPARSE_STATUS_SUCCESS) { */
        /* std::cerr << "CUSPARSE tridiag system solve failed." << std::endl; */
        /* return 1; */
    /* } */
    std::cout << "Copy Dev->Host... ";
    thrust::copy(d_V.begin(), d_V.end(), V.data.begin());
    std::cout << "OK\n";
    std::cout << "End C Solve\n";
    return 0;
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
