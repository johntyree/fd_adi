
#include "GNUC_47_compat.h"

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/version.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

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

#include "_TriBandedOperatorGPU.cuh"
#include <stdio.h>


void debug_printer(const char *type, const char *fn, const char *func, int line, std::string msg) {
    std::cout
        << type << ": "
        << fn << "(" << line << "): "
        << func;
        if (msg.size() != 0) {
            std::cout << "\n\t" << msg;
        }
        std::cout << std::endl;
}

#define TILE_DIM 1
#define BLOCK_ROWS 1

#define FULLTRACE 0;

__global__
void d_transposeNaive(REAL_t *odata, REAL_t* idata, int height, int width) {
    int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
    int index_in = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
        odata[index_out+i] = idata[index_in+i*width];
    }
}

void transposeNaive(REAL_t *odata, REAL_t *idata, int height, int width) {
    dim3 grid(max(1, width/TILE_DIM), max(1, height/TILE_DIM));
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    std::cout << "Grid: "<<grid.x<<", "<<grid.y
        << " Threads: "<<threads.x<<", "<<threads.y<<"\n";
    cudaDeviceSynchronize();
    d_transposeNaive<<<grid, threads>>>(odata, idata, height, width);
    cudaDeviceSynchronize();
}

__global__
void d_transposeNoBankConflicts(
        REAL_t *odata, REAL_t *idata, int height, int width) {
    __shared__ REAL_t tile[TILE_DIM][TILE_DIM+1];
    int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (index_in + i * width < width*height) {
            // printf("Thread %i ((%i, %i) -> %i)\n", index_in, threadIdx.y+i, threadIdx.x, index_in+i*width);
            tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
        }
    }
    __syncthreads();
    for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (index_out + i * height < width*height
                && threadIdx.x + TILE_DIM * blockIdx.y < height) {
            odata[index_out+i*height] =
                tile[threadIdx.x][threadIdx.y+i];
        }
    }
}

void transposeNoBankConflicts(REAL_t *odata, REAL_t *idata,
        int height, int width) {
    dim3 grid(ceil((float)width/TILE_DIM), ceil((float)height/TILE_DIM));
    dim3 threads(TILE_DIM,BLOCK_ROWS);
    // printf("Grid: %i %i  Threads: %i %i", grid.x, grid.y, threads.x, threads.y);
    /* std::cout << "Grid: "<<grid.y<<", "<<grid.x */
        /* << " Threads: "<<threads.x<<", "<<threads.y<<"\n"; */
    cudaDeviceSynchronize();
    d_transposeNoBankConflicts<<<grid, threads>>>(odata, idata, height, width);
    cudaDeviceSynchronize();
}

__global__
void d_transposeDiagonal(
    REAL_t *odata, REAL_t *idata, int height, int width) {
    __shared__ REAL_t tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    // diagonal reordering
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid/gridDim.y) + blockIdx_y) % gridDim.x;
    }
    int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + yIndex * width;
    xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + yIndex * height;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        /* if (index_in + i * width < width*height) { */
            tile[threadIdx.y + i][threadIdx.x] =
                idata[index_in + i * width];
        /* } */
    }
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        /* if (index_out + i * height < width*height) { */
            odata[index_out + i * height] =
                tile[threadIdx.x][threadIdx.y + i];
        /* } */
    }
}

void transposeDiagonal(REAL_t *odata, REAL_t *idata,
        int height, int width) {
    dim3 grid(max(1, width/TILE_DIM), max(1, height/TILE_DIM));
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    /* std::cout << "Grid: "<<grid.x<<", "<<grid.y */
        /* << " Threads: "<<threads.x<<", "<<threads.y<<"\n"; */
    cudaDeviceSynchronize();
    d_transposeDiagonal<<<grid, threads>>>(odata, idata, height, width);
    /* BALLS<<<grid, threads>>>(odata, idata, width, height); */
    cudaDeviceSynchronize();
}


template <typename T, typename U>
int find_index(T haystack, U needle, int max) {
    int idx;
    for (idx = 0; idx < max; ++idx) {
        if (haystack[idx] == needle) break;
    }
    if (idx >= max) {
        /* LOG("Did not find "<<needle<<" before reaching index "<<max<<"."); */
        /* std::cout << '\t'; */
        /* std::cout << "Haystack [ "; */
        /* for (idx = 0; idx < max; ++idx) { */
            /* std::cout << haystack[idx] << " "; */
        /* } */
        /* std::cout << "]"; ENDL; */
        idx = -1;
    }
    return idx;
}

_TriBandedOperator::_TriBandedOperator(
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
    diags(data),
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
    sup(diags.data.ptr()),
    mid(diags.data.ptr() + operator_rows),
    sub(diags.data.ptr() + 2*operator_rows),
    has_high_dirichlet(has_high_dirichlet),
    has_low_dirichlet(has_low_dirichlet),
    has_residual(has_residual),
    is_tridiagonal(offsets.size == 3 && main_diag != -1)
    {
        verify_diag_ptrs();
        status = cusparseCreate(&handle);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "CUSPARSE Library initialization failed." << std::endl;
            assert(false);
        }
    }

void _TriBandedOperator::verify_diag_ptrs() {
    assert(sup.get() != 0);
    assert(mid.get() != 0);
    assert(sub.get() != 0);
    if (main_diag == -1) {
        /* LOG("No main diag means not tridiagonal, hopefully."); */
        return;
    }
    if (offsets.size != 3) {
        /* LOG("Not tridiagonal. Skipping diag ptrs check."); */
        return;
    }
    int idx;
    /* LOG("main_diag("<<main_diag<<")"); */
    idx = diags.idx(main_diag-1, 0);
    if (*sup != diags.data[idx]
            || (sup.get() != (&diags.data[diags.idx(0,0)]).get())) {
        LOG("sup[0] = " << *sup << " <->  " << diags.get(0,0));
        LOG("sup = " << sup.get() << " <->  "
                << (&diags.data[diags.idx(0, 0)]).get());
        assert(0);
    }
    if (*mid != diags.data[diags.idx(main_diag, 0)]
            || (mid.get() != (&diags.data[diags.idx(main_diag, 0)]).get())) {
        LOG("mid[0] = " << *mid << " !=  " << diags.get(main_diag,0));
        LOG("mid = " << mid.get() << " <->  "
                << (&diags.data[diags.idx(main_diag, 0)]).get());
        assert(0);
    }
    if (*sub != diags.data[diags.idx(main_diag+1, 0)]
            || (sub.get() != (&diags.data[diags.idx(main_diag+1, 0)]).get())) {
        LOG("sub[0] = " << *sub << " !=  " << diags.get(main_diag+1,0));
        LOG("sub = " << sub.get() << " <->  "
                << (&diags.data[diags.idx(main_diag+1, 0)]).get());
        assert(0);
    }
}


struct zipdot3 : thrust::binary_function<const Triple &, const Triple &, REAL_t> {
    __host__ __device__
    REAL_t operator()(const Triple &diags, const Triple &x) {
        using thrust::get;
        const REAL_t a = get<0>(diags);
        const REAL_t b = get<1>(diags);
        const REAL_t c = get<2>(diags);
        const REAL_t x0 = get<0>(x);
        const REAL_t x1 = get<1>(x);
        const REAL_t x2 = get<2>(x);
        return a*x0 + b*x1 + c*x2;
    }
};

SizedArray<double> *_TriBandedOperator::apply(SizedArray<double> &V) {
    FULLTRACE;
    using std::cout;
    using std::endl;
    const unsigned N = V.size;
    thrust::device_vector<double> in(V.data);
    thrust::device_vector<double> out(N);

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

    thrust::device_vector<double> a(sub, sub+N);
    thrust::device_vector<double> b(mid, mid+N);
    thrust::device_vector<double> c(sup, sup+N);

    out[0] = b[0]*in[0] + c[0]*in[1];
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(a.begin()+1, b.begin()+1, c.begin()+1)),
        thrust::make_zip_iterator(thrust::make_tuple(a.end()-1, b.end()-1, c.end()-1)),
        thrust::make_zip_iterator(thrust::make_tuple(in.begin(), in.begin()+1, in.begin()+2)),
        out.begin()+1,
        zipdot3()
    );
    out[N-1] = a[N-1]*in[N-2] + b[N-1]*in[N-1];

    SizedArray<double> *U = new SizedArray<double>(out,
            V.ndim, V.shape, "CPP Solve U from V");

    if (has_residual) {
        /* ret += self.R; */
    }

    /* ret = ret.reshape(V.shape) */

    /* t = range(V.ndim) */
    /* utils.rolllist(t, V.ndim-1, self.axis) */

    // Transpose back
    /* return ret; */
    FULLTRACE;
    return U;
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

void _TriBandedOperator::add_operator(_TriBandedOperator &other) {
    FULLTRACE;
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
        o = other.offsets.get(i);
        to = find_index(offsets.data, o, offsets.size);
        if (offsets.get(to) != o) {
            std::cout << std::endl;
            std::cout << "to: " << to << "(";
            /* print_array(&offsets(0), offsets.size); */
            std::cout << offsets.data;
            std::cout << ")";
            std::cout << "fro: " << fro << "(";
            std::cout << other.offsets.data;
            /* print_array(&other.offsets(0), other.offsets.size); */
            std::cout << ")" << std::endl;
            select(1, NULL, NULL, NULL, &sleeptime);
            assert (offsets.get(to) == o);
        }
        /* LOG("Adding offset " << o << "."); */
        if (o == 0) {
            thrust::transform_if(
                    &diags.data[diags.idx(to, 0)],
                    &diags.data[diags.idx(to, 0)] + operator_rows,
                    &other.diags.data[diags.idx(fro, 0)],
                    thrust::make_counting_iterator(0),
                    &diags.data[diags.idx(to, 0)],
                    thrust::plus<double>(),
                    periodic_from_to_mask(begin, end, block_len));
        } else {
            thrust::transform(
                    &other.diags.data[diags.idx(fro, 0)],
                    &other.diags.data[diags.idx(fro, 0)] + other.diags.shape[1],
                    &diags.data[diags.idx(to, 0)],
                    &diags.data[diags.idx(to, 0)],
                    thrust::plus<double>());
        }
    }
    /* LOG("Adding R."); */
    thrust::transform(
            R.data.begin(),
            R.data.end(),
            other.R.data.begin(),
            R.data.begin(),
            thrust::plus<double>());
    FULLTRACE;
}



void _TriBandedOperator::add_scalar(double val) {
    FULLTRACE;
    /* Add a scalar to the main diagonal.
     * Does not alter the residual vector.
     */
    // We add it to the main diagonal.

    int begin = has_low_dirichlet;
    int end = block_len-1 - has_high_dirichlet;

    assert(main_diag < offsets.size);

    thrust::transform_if(
            &diags.data[diags.idx(main_diag, 0)],
            &diags.data[diags.idx(main_diag, 0)] + operator_rows,
            thrust::make_constant_iterator(val),
            thrust::make_counting_iterator(0),
            &diags.data[diags.idx(main_diag, 0)],
            thrust::plus<double>(),
            periodic_from_to_mask(begin, end, block_len));
    FULLTRACE;
}

bool _TriBandedOperator::is_folded() {
    return false;
}



int _TriBandedOperator::solve(SizedArray<double> &V) {
    verify_diag_ptrs();

    /* std::cout << "Begin C Solve\n"; */
    /* std::cout << "Copy Host->Dev... " << V.data << ' '; */
    GPUVec<double> d_V(V.data);
    GPUVec<double> d_sup(sup, sup+V.size);
    GPUVec<double> d_mid(mid, mid+V.size);
    GPUVec<double> d_sub(sub, sub+V.size);
    /* std::cout << "OK\n"; */

    /* std::cout << "CUSPARSE... "; */
    status = cusparseDgtsvStridedBatch(handle, V.size,
            d_sub.raw(), d_mid.raw(), d_sup.raw(),
            d_V.raw(),
            1, V.size);
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE tridiag system solve failed." << std::endl;
        return 1;
    }
    /* std::cout << "OK\n"; */

    /* std::cout << "Copy Dev->Host... " << d_V << ' '; */
    thrust::copy(d_V.begin(), d_V.end(), V.data.begin());
    /* std::cout << "OK\n"; */
    /* std::cout << "End C Solve\n"; */
    FULLTRACE;
    return 0;
}


void _TriBandedOperator::vectorized_scale(SizedArray<double> &vector) {
    FULLTRACE;
    Py_ssize_t vsize = vector.size;
    Py_ssize_t noffsets = offsets.size;
    Py_ssize_t block_len = operator_rows / blocks;

    assert(operator_rows % vsize == 0);

    if (has_low_dirichlet) {
        for (Py_ssize_t b = 0; b < blocks; ++b) {
            vector.data[vector.idx(b*block_len % vsize)] = 1;
        }
    }

    if (has_high_dirichlet) {
        for (Py_ssize_t b = 0; b < blocks; ++b) {
            vector.data[vector.idx((b+1)*block_len - 1 % vsize)] = 1;
        }
    }

    for (Py_ssize_t row = 0; row < noffsets; ++row) {
        int o = offsets.get(row);
        if (o >= 0) { // upper diags
            for (int i = 0; i < (int)operator_rows - o; ++i) {
                diags.data[diags.idx(row, i+o)] *= vector.data[vector.idx(i % vsize)];
            }
        } else { // lower diags
            for (int i = -o; i < (int)operator_rows; ++i) {
                diags.data[diags.idx(row, i+o)] *= vector.data[vector.idx(i % vsize)];
            }
        }
    }
    /* LOG("Scaled data."); */

    for (Py_ssize_t i = 0; i < operator_rows; ++i) {
        R.data[R.idx(i)] *= vector.data[vector.idx(i % vsize)];
    }
    /* LOG("Scaled R."); */
    return;
}

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
            periodic_from_to_mask(begin, end, block_len));

    printf("\n");
    print_array(a.data(), a.size());
    time_t theTime;
    time(&theTime);   // get the calendar time
    tm *t = localtime( &theTime );  // convert to local
    std::cout << "The time is: " << asctime(t);
    return 0;
}