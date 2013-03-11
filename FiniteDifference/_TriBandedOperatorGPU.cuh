
#ifndef _TriBandedOperatorGPU_cuh
#define _TriBandedOperatorGPU_cuh

#include <thrust/tuple.h>
#include <thrust/device_ptr.h>

#include <cusparse_v2.h>

#include "common.h"
#include "backtrace.h"
#include "VecArray.h"


typedef GPUVec<REAL_t>::iterator VI;
typedef thrust::tuple<REAL_t,REAL_t,REAL_t> Triple;
typedef thrust::tuple<REAL_t, REAL_t, REAL_t, REAL_t,
        REAL_t, REAL_t, REAL_t> Septuple;

static const std::string FOLDED = "FOLDED";
static const std::string CAN_FOLD = "CAN_FOLD";
static const std::string CANNOT_FOLD = "CANNOT_FOLD";

class _TriBandedOperator {
    public:
        SizedArray<double> diags;
        SizedArray<double> R;
        SizedArray<double> high_dirichlet;
        SizedArray<double> low_dirichlet;
        SizedArray<double> top_factors;
        SizedArray<double> bottom_factors;

        SizedArray<double> &apply(SizedArray<double> &);
        SizedArray<double> &solve(SizedArray<double> &);
        bool is_folded();
        void DMVPY(SizedArray<double> &V, char operation, SizedArray<double> &Y, SizedArray<double> &out);
        void add_operator(_TriBandedOperator &other);
        void add_scalar(double val);
        void diagonalize();
        void fold_bottom(bool=false);
        void fold_top(bool=false);
        void fold_vector(GPUVec<double> &vector, bool=false);
        void undiagonalize();
        void vectorized_scale(SizedArray<double> &vector);
        void verify_diag_ptrs();

        // This is public to get to and from python
        bool has_residual;
        bool has_high_dirichlet;
        bool has_low_dirichlet;
        std::string top_fold_status;
        std::string bottom_fold_status;
        Py_ssize_t operator_rows;
        Py_ssize_t blocks;
        Py_ssize_t block_len;

        _TriBandedOperator(
            SizedArray<double> &data,
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
            bool has_residual
            );

    private:
        unsigned int axis;
        Py_ssize_t main_diag;
        thrust::device_ptr<double> sup, mid, sub;
        cusparseHandle_t handle;
        cusparseStatus_t status;
};

#endif
