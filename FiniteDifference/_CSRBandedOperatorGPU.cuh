#ifndef _CSRBandedOperatorGPU_cuh
#define _CSRBandedOperatorGPU_cuh

#include <cusparse_v2.h>

#include "backtrace.h"
#include "common.h"
#include "VecArray.h"


class _CSRBandedOperator {

    public:
        SizedArray<double> data;
        SizedArray<int> row_ptr;
        SizedArray<int> row_ind;
        SizedArray<int> col_ind;
        Py_ssize_t operator_rows;
        Py_ssize_t blocks;
        std::string name;

        void apply(SizedArray<double> &);
        void vectorized_scale(SizedArray<double> &vector);

        _CSRBandedOperator(
            SizedArray<double> &data,
            SizedArray<int> &row_ptr,
            SizedArray<int> &row_ind,
            SizedArray<int> &col_ind,
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


_CSRBandedOperator * mixed_for_vector(
        SizedArray<double> &v0, SizedArray<double> &v1);

std::ostream & operator<<(std::ostream & os, _CSRBandedOperator const &B);

#endif
