#include "GNUC_47_compat.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/version.h>

#include "repeated_range.h"

#include "_CSRBandedOperatorGPU.cuh"

__device__ static const REAL_t one = 1;
__device__ static const REAL_t zero = 0;


std::ostream & operator<<(std::ostream & os, _CSRBandedOperator const &B) {
    return os << B.name << ": addr("<<&B<<")\n\t"
        << "operator_rows(" << B.operator_rows << ") "
        << "blocks(" << B.blocks <<") "
        << "nnz(" << B.nnz << ")\n\t"
        << "data(" << B.data << ")\n\t"
        << "row_ptr(" << B.row_ptr << ")\n\t"
        << "row_ind(" << B.row_ind << ")\n\t"
        << "col_ind(" << B.col_ind << ")\n"
        ;
}


_CSRBandedOperator::_CSRBandedOperator(
        SizedArray<double> &data,
        SizedArray<int> &row_ptr,
        SizedArray<int> &row_ind,
        SizedArray<int> &col_ind,
        Py_ssize_t operator_rows,
        Py_ssize_t blocks,
        std::string name
        ) :
    data(data, true),
    row_ptr(row_ptr, true),
    row_ind(row_ind, true),
    col_ind(col_ind, true),
    name(name),
    operator_rows(operator_rows),
    blocks(blocks),
    block_len(operator_rows / blocks),
    nnz(data.size)
    {
        status = cusparseCreate(&handle);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "CUSPARSE Library initialization failed." << std::endl;
            assert(false);
        }
    }


void _CSRBandedOperator::apply(SizedArray<double> &V) {
    FULLTRACE;

    if (V.size != operator_rows) {
        DIE(V.name << ": Dimension mismatch. V(" <<V.size<<") vs "<<operator_rows);
    }

    cusparseMatDescr_t mat_description;
    status = cusparseCreateMatDescr(&mat_description);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        DIE("CUSPARSE matrix description init failed.");
    }

    thrust::copy(V.data, V.data + V.size, V.tempspace);

    double zero = 0, one = 1;
    status = cusparseDcsrmv(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            operator_rows,
            operator_rows,
            nnz,
            &one,
            mat_description,
            data.data.get(),
            row_ptr.data.get(),
            col_ind.data.get(),
            V.tempspace.get(),
            &zero,
            V.data.get()
            );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        DIE("CUSPARSE CSR MV product failed.");
    }

    FULLTRACE;
    return;
}


template<typename Tuple>
struct compute_index : public thrust::unary_function<Tuple, int> {
    int h, w;

    compute_index(int h, int w) : h(h), w(w) {}

    __host__ __device__
    int operator()(Tuple i) {
        return thrust::get<0>(i);
    }
};


void _CSRBandedOperator::vectorized_scale(SizedArray<double> &vector) {
    FULLTRACE;

    Py_ssize_t vsize = vector.size;

    if (operator_rows % vsize != 0) {
        DIE("Vector length does not divide "
            "evenly into operator size. Cannot scale. "
            "vsize("<<vsize<<") "
            "operator_rows("<<operator_rows<<") "
            "blocks("<<blocks<<") "
            "Vector ndim("<<vector.ndim<<") "
            "Vector shape("<<vector.shape[0]<<", "<<vector.shape[1]<<")"
           );
    }


    typedef thrust::tuple<int, int> IntTuple;
    typedef thrust::device_vector<REAL_t>::iterator Iterator;

    repeated_range<Iterator> v(vector.data, vector.data + vector.size, operator_rows / vsize);

    /* TODO: This can be optimized by noting that we only use the row_ind and can make
     * do with project_1st(). */
    thrust::transform(data.data, data.data + data.size,
        thrust::make_permutation_iterator(v.begin(),
            thrust::make_transform_iterator(
                thrust::make_zip_iterator(
                    thrust::make_tuple(row_ind.data, col_ind.data)
                    ),
                compute_index<IntTuple>(block_len, block_len)
                )
            ),
        data.data,
        thrust::multiplies<double>());

    FULLTRACE;
    return;
}
