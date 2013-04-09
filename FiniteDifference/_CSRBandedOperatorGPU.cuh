#ifndef _CSRBandedOperatorGPU_cuh
#define _CSRBandedOperatorGPU_cuh

#include <cusparse_v2.h>

#include "backtrace.h"
#include "common.h"
#include "VecArray.h"

extern std::string const FOLDED;
extern std::string const CAN_FOLD;
extern std::string const CANNOT_FOLD;


class _CSRBandedOperator {

    public:
        SizedArray<double> data;
        SizedArray<int> row_ptr;
        SizedArray<int> row_ind;
        SizedArray<int> col_ind;
        SizedArray<double> R;
        SizedArray<double> high_dirichlet;
        SizedArray<double> low_dirichlet;
        SizedArray<double> top_factors;
        SizedArray<double> bottom_factors;
        Py_ssize_t operator_rows;
        Py_ssize_t blocks;
        bool has_high_dirichlet;
        bool has_low_dirichlet;
        std::string top_fold_status;
        std::string bottom_fold_status;
        bool has_residual;
        std::string name;

        void apply(SizedArray<double> &);
        void solve(SizedArray<double> &);
        void fake_solve(SizedArray<double> &);
        void vectorized_scale(SizedArray<double> &vector);
        bool is_folded();
        void fold_vector(SizedArray<double> &vector, bool=false);

        _CSRBandedOperator(
            SizedArray<double> &data,
            SizedArray<int> &row_ptr,
            SizedArray<int> &row_ind,
            SizedArray<int> &col_ind,
            SizedArray<double> &R,
            SizedArray<double> &high_dirichlet,
            SizedArray<double> &low_dirichlet,
            SizedArray<double> &top_factors,
            SizedArray<double> &bottom_factors,
            unsigned int axis,
            Py_ssize_t operator_rows,
            Py_ssize_t blocks,
            bool has_high_dirichlet,
            bool has_low_dirichlet,
            std::string top_fold_status,
            std::string bottom_fold_status,
            bool has_residual,
            std::string name = "<UNKNOWN CSR>"
            );

    private:
        unsigned int axis;
        Py_ssize_t block_len;
        Py_ssize_t nnz;
        cusparseHandle_t handle;
        cusparseStatus_t status;
        cusparseMatDescr_t mat_description;
        cusparseSolveAnalysisInfo_t analysis_info;

        friend std::ostream& operator<<(
                std::ostream &,
                const _CSRBandedOperator&);
};


_CSRBandedOperator * mixed_for_vector(
        SizedArray<double> &v0, SizedArray<double> &v1);

std::ostream & operator<<(std::ostream & os, _CSRBandedOperator const &B);

#endif
