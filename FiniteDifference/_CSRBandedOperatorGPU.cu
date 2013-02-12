
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
#include <thrust/iterator/repeat_iterator.h>

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

void _CSRBandedOperator::vectorized_scale(SizedArray<double> &vector) {
    Py_ssize_t vsize = vector.size;
    Py_ssize_t block_len = operator_rows / blocks;

    /* repeat_iterator<REAL_t> spot (spots. */

    if (operator_rows % vsize == 0) {
        DIE("Vector length does not divide "
                "evenly into operator size. Cannot scale.");
    }
    return;
}
