
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
#include "repeated_range.h"

#include <algorithm>
#include <cstdlib>

#include <iostream>
#include <cassert>

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
    data(data.data),
    row_ptr(row_ptr.data),
    row_ind(row_ind.data),
    col_ind(col_ind.data),
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
        /* LOG("CSRBandedOperator constructed: " << *this); */
    }

SizedArray<double> *_CSRBandedOperator::apply(SizedArray<double> &V) {
    if (V.size != operator_rows) {
        DIE(V.name << ": Dimension mismatch. V(" <<V.size<<") vs "<<operator_rows);
    }
    // XXX: This can possibly be optimized to run in place...
    SizedArray<double> *U = new SizedArray<double>(V.size, "CSR Solve U from V");
    U->shape[0] = V.shape[0];
    U->shape[1] = V.shape[1];
    U->ndim = V.ndim;
    cusparseMatDescr_t mat_description;
    status = cusparseCreateMatDescr(&mat_description);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        DIE("CUSPARSE matrix description init failed.");
    }
    double zero = 0, one = 1;
    status = cusparseDcsrmv(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            operator_rows,
            operator_rows,
            nnz,
            &one,
            mat_description,
            data.raw(),
            row_ptr.raw(),
            col_ind.raw(),
            V.data.raw(),
            &zero,
            U->data.raw()
            );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        DIE("CUSPARSE CSR MV product failed.");
    }

    return U;
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
    Py_ssize_t vsize = vector.size;

    if (operator_rows % vsize != 0) {
        DIE("Vector length does not divide "
            "evenly into operator size. Cannot scale. "
            "vsize("<<vsize<<") "
            "operator_rows("<<operator_rows<<") "
            "blocks("<<blocks<<")"
           );
    }


    typedef thrust::tuple<int, int> IntTuple;
    typedef thrust::device_vector<REAL_t>::iterator Iterator;

    repeated_range<Iterator> v(vector.data.begin(), vector.data.end(), operator_rows / vsize);

    thrust::transform(data.begin(), data.end(),
        thrust::make_permutation_iterator(v.begin(),
            thrust::make_transform_iterator(
                thrust::make_zip_iterator(
                    thrust::make_tuple(row_ind.begin(), col_ind.begin())
                    ),
                compute_index<IntTuple>(block_len, block_len)
                )
            ),
            data.begin(),
            thrust::multiplies<double>());
    FULLTRACE;
    return;
}
