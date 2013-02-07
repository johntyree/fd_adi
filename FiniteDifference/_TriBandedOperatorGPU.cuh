
#ifndef _TriBandedOperatorGPU_cuh
#define _TriBandedOperatorGPU_cuh

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <thrust/tuple.h>

#include <cusparse_v2.h>
#include <cmath>

#include "common.h"
#include "backtrace.h"
#include "VecArray.h"


typedef thrust::tuple<REAL_t,REAL_t,REAL_t> Triple;

void transposeDiagonal(REAL_t *odata, REAL_t *idata, int width, int height);
void transposeNoBankConflicts(REAL_t *odata, REAL_t *idata, int width, int height);
void transposeNaive(REAL_t *odata, REAL_t *idata, int width, int height);


class _TriBandedOperator {
    public:
        SizedArray<double> diags;
        SizedArray<double> R;
        SizedArray<double> high_dirichlet;
        SizedArray<double> low_dirichlet;
        SizedArray<double> top_factors;
        SizedArray<double> bottom_factors;
        SizedArray<int> offsets;

        void verify_diag_ptrs();
        bool is_folded();
        SizedArray<double> *apply(SizedArray<double> &);
        int solve(SizedArray<double> &);
        void add_scalar(double val);
        void vectorized_scale(SizedArray<double> &vector);
        void add_operator(_TriBandedOperator &other);

        _TriBandedOperator(
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
            );

    private:
        unsigned int axis;
        Py_ssize_t main_diag;
        Py_ssize_t operator_rows;
        Py_ssize_t blocks;
        Py_ssize_t block_len;
        thrust::device_ptr<double> sup, mid, sub;
        bool has_high_dirichlet;
        bool has_low_dirichlet;
        bool has_residual;
        bool is_tridiagonal;
        cusparseHandle_t handle;
        cusparseStatus_t status;
};

#endif
