
#ifndef _CSRBandedOperatorGPU_cuh
#define _CSRBandedOperatorGPU_cuh

#include <cusparse_v2.h>

#include "common.h"
#include "backtrace.h"
#include "VecArray.h"

class _CSRBandedOperator {
    public:
        GPUVec<double> data;
        GPUVec<int> row_ptr;
        GPUVec<int> row_ind;
        GPUVec<int> col_ind;
        Py_ssize_t operator_rows;
        Py_ssize_t blocks;
        std::string name;

        SizedArray<double> *apply(SizedArray<double> &);
        void vectorized_scale(SizedArray<double> &vector);

        _CSRBandedOperator(
            GPUVec<double> &data,
            GPUVec<int> &row_ptr,
            GPUVec<int> &row_ind,
            GPUVec<int> &col_ind,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            std::string name = "<UNKNOWN CSR>"
            );

    private:
        Py_ssize_t block_len;
        Py_ssize_t nnz;
        cusparseHandle_t handle;
        cusparseStatus_t status;

        friend std::ostream& operator<<(
                std::ostream &,
                const _CSRBandedOperator&);
};

std::ostream & operator<<(std::ostream & os, _CSRBandedOperator const &B);
#endif
