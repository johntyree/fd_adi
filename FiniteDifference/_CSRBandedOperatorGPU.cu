
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
#include <cassert>

#include "_CSRBandedOperatorGPU.cuh"

__device__ const REAL_t one = 1;
__device__ const REAL_t zero = 0;

_CSRBandedOperator::_CSRBandedOperator(
        SizedArray<double> &data,
        SizedArray<int> &row_ind,
        SizedArray<int> &col_ind,
        Py_ssize_t operator_rows,
        Py_ssize_t blocks
        ) :
    data(data.data),
    row_ind(row_ind.data),
    col_ind(col_ind.data),
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


SizedArray<double> *_CSRBandedOperator::apply(SizedArray<double> &V) {
    const unsigned N = V.size;
    thrust::device_vector<double> in(V.data);
    thrust::device_vector<double> out(N);

    // XXX: This can possibly be optimized to run in place...
    SizedArray<double> *U = new SizedArray<double>(V.size, "CSR Solve U from V");
    cusparseMatDescr_t mat_description;
    status = cusparseCreateMatDescr(&mat_description);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE matrix description init failed." << std::endl;
        assert(false);
    }
    status = cusparseDcsrmv(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            operator_rows,
            operator_rows,
            nnz,
            &one,
            mat_description,
            data.raw(),
            row_ind.raw(),
            col_ind.raw(),
            V.data.raw(),
            &zero,
            U->data.raw()
            );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE CSR MV product failed." << std::endl;
        assert(false);
    }

    return U;
}


void _CSRBandedOperator::vectorized_scale(SizedArray<double> &vector) {
    FULLTRACE;
    Py_ssize_t vsize = vector.size;
    Py_ssize_t block_len = operator_rows / blocks;

    assert(operator_rows % vsize == 0);
    return;
}
