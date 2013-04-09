#include "GNUC_47_compat.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>

#include <thrust/adjacent_difference.h>
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
#include "tiled_range.h"
#include "strided_range.h"

#include "_CSRBandedOperatorGPU.cuh"

typedef thrust::device_ptr<double> Dptr;
typedef thrust::detail::normal_iterator<thrust::device_ptr<double> > DptrIterator;

using namespace thrust::placeholders;
using thrust::counting_iterator;

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
        std::string name
        ) :
    data(data, true),
    row_ptr(row_ptr, true),
    row_ind(row_ind, true),
    col_ind(col_ind, true),
    R(R, true),
    high_dirichlet(high_dirichlet, true),
    low_dirichlet(low_dirichlet, true),
    top_factors(top_factors, true),
    bottom_factors(bottom_factors, true),
    axis(axis),
    operator_rows(operator_rows),
    blocks(blocks),
    block_len(operator_rows / blocks),
    has_high_dirichlet(has_high_dirichlet),
    has_low_dirichlet(has_low_dirichlet),
    top_fold_status(top_fold_status),
    bottom_fold_status(bottom_fold_status),
    has_residual(has_residual),
    nnz(data.size),
    name(name)
    {
        status = cusparseCreate(&handle);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            DIE("CUSPARSE Library initialization failed (handle).");
        }
        status = cusparseCreateSolveAnalysisInfo(&analysis_info);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            DIE("CUSPARSE Library initialization failed (analysis).");
        }

        status = cusparseCreateMatDescr(&mat_description);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            DIE("CUSPARSE Library initialization failed (mat_descr).");
        }
        status = cusparseSetMatDiagType(mat_description, CUSPARSE_DIAG_TYPE_NON_UNIT);
    }


bool _CSRBandedOperator::is_folded() {
    return (top_fold_status == FOLDED || bottom_fold_status == FOLDED);
}


template <typename Tuple, typename Result>
struct add_multiply3_ : public thrust::unary_function<Tuple, Result> {
    Result direction;
    add_multiply3_(Result x) : direction(x) {}
    __host__ __device__
    Result operator()(Tuple t) {
        using thrust::get;
        return  get<0>(t) + direction * get<1>(t) * get<2>(t);
    }
};

void _CSRBandedOperator::fold_vector(SizedArray<double> &vector, bool unfold) {
    FULLTRACE;

    typedef thrust::tuple<REAL_t,REAL_t,REAL_t> REALTuple;

    strided_range<DptrIterator> u0(vector.data, vector.data + vector.size, block_len);
    strided_range<DptrIterator> u1(vector.data+1, vector.data + vector.size, block_len);

    strided_range<DptrIterator> un(vector.data+block_len-1, vector.data + vector.size, block_len);
    strided_range<DptrIterator> un1(vector.data+block_len-2, vector.data + vector.size, block_len);

    // Top fold
    if (top_fold_status == FOLDED) {
        /* LOG("Folding top. direction("<<unfold<<") top_factors("<<top_factors<<")"); */
        thrust::transform(
            make_zip_iterator(make_tuple(u0.begin(), u1.begin(), top_factors.data)),
            make_zip_iterator(make_tuple(u0.end(), u1.end(), top_factors.data + top_factors.size)),
            u0.begin(),
            add_multiply3_<REALTuple, REAL_t>(unfold ? -1 : 1));
    }

    if (bottom_fold_status == FOLDED) {
        /* LOG("Folding bottom. direction("<<unfold<<") bottom_factors("<<bottom_factors<<")"); */
        thrust::transform(
            make_zip_iterator(make_tuple(un.begin(), un1.begin(), bottom_factors.data)),
            make_zip_iterator(make_tuple(un.end(), un1.end(), bottom_factors.data + bottom_factors.size)),
            un.begin(),
            add_multiply3_<REALTuple, REAL_t>(unfold ? -1 : 1));
    }

    FULLTRACE;
}



void _CSRBandedOperator::solve(SizedArray<double> &V) {
    FULLTRACE;

    /* if (has_low_dirichlet) { */
        /* LOG("has_low_dirichlet " << has_low_dirichlet); */
        /* thrust::copy(low_dirichlet.data, */
                /* low_dirichlet.data + low_dirichlet.size, */
                /* V.data); */
    /* } */
    /* if (has_high_dirichlet) { */
        /* LOG("has_high_dirichlet " << has_high_dirichlet); */
        /* thrust::copy(high_dirichlet.data, */
                /* high_dirichlet.data + high_dirichlet.size, */
                /* V.data + V.size - V.shape[1]); */
    /* } */

    /* if (axis == 0) { */
        /* V.transpose(1); */
    /* } */

    /* if (has_residual) { */
        /* LOG("has_residual " << has_residual); */
        /* thrust::transform(V.data, V.data + V.size, */
                /* R.data, */
                /* V.data, */
                /* thrust::minus<double>()); */
    /* } */

    /* if (is_folded()) { */
        /* LOG("is_folded()"); */
        /* fold_vector(V); */
    /* } */

    double const one = 1;
    thrust::copy(V.data, V.data + V.size, V.tempspace);
    status = cusparseDcsrsv_analysis(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            V.size,
            data.size,
            mat_description,
            data.data.get(),
            row_ptr.data.get(),
            col_ind.data.get(),
            analysis_info
            );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        DIE("CUSPARSE Dcsrsv operation failed analysis.");
    }

    status = cusparseDcsrsv_solve(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            V.size,
            &one,
            mat_description,
            data.data.get(),
            row_ptr.data.get(),
            col_ind.data.get(),
            analysis_info,
            V.tempspace.get(),
            V.data.get()
            );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        DIE("CUSPARSE Dcsrsv operation failed solve.");
    }

    /* if (axis == 0) { */
        /* V.transpose(1); */
    /* } */

    FULLTRACE;
    return;
}


void _CSRBandedOperator::fake_solve(SizedArray<double> &V) {
    FULLTRACE;
    double const one = 1;
    thrust::copy(V.data, V.data + V.size, V.tempspace);
    status = cusparseDcsrsv_analysis(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            operator_rows,
            nnz,
            mat_description,
            data.data.get(),
            row_ptr.data.get(),
            col_ind.data.get(),
            analysis_info
            );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        DIE("CUSPARSE Dcsrsv operation failed analysis.");
    }

    status = cusparseDcsrsv_solve(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            operator_rows,
            &one,
            mat_description,
            data.data.get(),
            row_ptr.data.get(),
            col_ind.data.get(),
            analysis_info,
            V.tempspace.get(),
            V.data.get()
            );

    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        DIE("CUSPARSE Dcsrsv operation failed solve.");
    }
    FULLTRACE;
    return;
}



void _CSRBandedOperator::apply(SizedArray<double> &V) {
    FULLTRACE;

    if (V.size != operator_rows) {
        DIE(V.name << ": Dimension mismatch. Got V(" <<V.size<<") Expected "<<operator_rows);
    }

    status = cusparseCreateMatDescr(&mat_description);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        DIE("CUSPARSE matrix description init failed.");
    }

    strided_range<DptrIterator> u0(V.data, V.data+V.size, block_len);
    strided_range<DptrIterator> u1(V.data+block_len-1, V.data+V.size, block_len);

    if (axis == 1) {
        if (has_low_dirichlet) {
            thrust::copy(
                low_dirichlet.data,
                low_dirichlet.data + low_dirichlet.size,
                u1.begin()
                );
        }
        if (has_high_dirichlet) {
            thrust::copy(
                high_dirichlet.data,
                high_dirichlet.data + high_dirichlet.size,
                u0.begin()
                );
        }
    } else {
        if (has_low_dirichlet) {
            thrust::copy(low_dirichlet.data,
                    low_dirichlet.data + low_dirichlet.size,
                    V.data);
        }
        if (has_high_dirichlet) {
            thrust::copy(high_dirichlet.data,
                    high_dirichlet.data + high_dirichlet.size,
                    V.data + V.size - V.shape[1]);
        }
        V.transpose(1);
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

    if (is_folded()) {
        fold_vector(V, true);
    }

    if (has_residual) {
        thrust::transform(
                V.data,
                V.data + V.size,
                R.data,
                V.data,
                thrust::plus<double>());
    }

    if (axis == 0) {
        V.transpose(1);
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

struct first_deriv_ {
    template <typename Tuple>
    __host__ __device__
    /* (sup, mid, sub, deltas+1, deltas+2) */
    void operator()(Tuple t) {
        using thrust::get;
        get<0>(t) = get<3>(t)                / (get<4>(t) * (get<3>(t) + get<4>(t)));
        get<1>(t) = (-get<3>(t) + get<4>(t)) / (get<3>(t) * get<4>(t));
        get<2>(t) = -get<4>(t)               / (get<3>(t) * (get<3>(t) + get<4>(t)));
    }
};

struct multiply3x1 {
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        using thrust::get;
        get<0>(t) *= get<3>(t);
        get<1>(t) *= get<3>(t);
        get<2>(t) *= get<3>(t);
    }
};

_CSRBandedOperator * mixed_for_vector(SizedArray<double> &v0,
                                      SizedArray<double> &v1) {

    Py_ssize_t operator_rows = v0.size * v1.size;
    Py_ssize_t blocks = 1;

    int n0 = v0.size;
    int n1 = v1.size;
    int dlen = (n1-2) * (n0-2);

    GPUVec<double> d0(n0);
    GPUVec<double> d1(n1);

    thrust::adjacent_difference(v0.data, v0.data + n0, d0.begin());
    thrust::adjacent_difference(v1.data, v1.data + n1, d1.begin());

    GPUVec<double> sup(n1-2);
    GPUVec<double> mid(n1-2);
    GPUVec<double> sub(n1-2);

    thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    sup.begin(),
                    mid.begin(),
                    sub.begin(),
                    d1.begin()+1,
                    d1.begin()+2
                    )
                ),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    sup.end(),
                    mid.end(),
                    sub.end(),
                    d1.end()-1,
                    d1.end()
                    )
                ),
            first_deriv_()
            );

    SizedArray<double> data(dlen * 9, "coo_data");


    thrust::device_ptr<double> supsup(data.data+8*dlen);
    thrust::device_ptr<double> supmid(data.data+7*dlen);
    thrust::device_ptr<double> supsub(data.data+6*dlen);

    thrust::device_ptr<double> midsup(data.data+5*dlen);
    thrust::device_ptr<double> midmid(data.data+4*dlen);
    thrust::device_ptr<double> midsub(data.data+3*dlen);

    thrust::device_ptr<double> subsup(data.data+2*dlen);
    thrust::device_ptr<double> submid(data.data+1*dlen);
    thrust::device_ptr<double> subsub(data.data);

    tiled_range<DptrIterator> sups(sup.begin(), sup.end(), n0-2);
    tiled_range<DptrIterator> mids(mid.begin(), mid.end(), n0-2);
    tiled_range<DptrIterator> subs(sub.begin(), sub.end(), n0-2);

    thrust::copy(sups.begin(), sups.end(), supsup);
    thrust::copy(mids.begin(), mids.end(), supmid);
    thrust::copy(subs.begin(), subs.end(), supsub);

    thrust::copy(sups.begin(), sups.end(), midsup);
    thrust::copy(mids.begin(), mids.end(), midmid);
    thrust::copy(subs.begin(), subs.end(), midsub);

    thrust::copy(sups.begin(), sups.end(), subsup);
    thrust::copy(mids.begin(), mids.end(), submid);
    thrust::copy(subs.begin(), subs.end(), subsub);


    sup.resize(n0-2);
    mid.resize(n0-2);
    sub.resize(n0-2);
    thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    sup.begin(),
                    mid.begin(),
                    sub.begin(),
                    d0.begin()+1,
                    d0.begin()+2
                    )
                ),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    sup.end(),
                    mid.end(),
                    sub.end(),
                    d0.end()-1,
                    d0.end()
                    )
                ),
            first_deriv_()
            );

    repeated_range<DptrIterator> dsup(sup.begin(), sup.end(), n1-2);
    repeated_range<DptrIterator> dmid(mid.begin(), mid.end(), n1-2);
    repeated_range<DptrIterator> dsub(sub.begin(), sub.end(), n1-2);

    thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    supsup,
                    supmid,
                    supsub,
                    dsup.begin()
                    )
                ),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    supsup+dlen,
                    supmid+dlen,
                    supsub+dlen,
                    dsup.end()
                    )
                ),
            multiply3x1()
            );

    thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    midsup,
                    midmid,
                    midsub,
                    dmid.begin()
                    )
                ),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    midsup+dlen,
                    midmid+dlen,
                    midsub+dlen,
                    dmid.end()
                    )
                ),
            multiply3x1()
            );


    thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    subsup,
                    submid,
                    subsub,
                    dsub.begin()
                    )
                ),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    subsup+dlen,
                    submid+dlen,
                    subsub+dlen,
                    dsub.end()
                    )
                ),
            multiply3x1()
            );

    /* row = np.tile(np.arange(1, n1-1), n0-2) */
    tiled_range<counting_iterator<int> > row0(
            counting_iterator<int>(1),
            counting_iterator<int>(n1-1),
            n0-2
            );
    /* row += np.repeat(np.arange(1, n0-1), n1-2) * n1 */
    repeated_range<counting_iterator<int> > row1(
            counting_iterator<int>(1),
            counting_iterator<int>(n0-1),
            n1-2
            );
    GPUVec<int> rowr(dlen);
    thrust::transform(
            row0.begin(),
            row0.end(),
            row1.begin(),
            rowr.begin(),
            _1 + _2 * n1
            );

    /* row = np.tile(row, 9) */
    tiled_range<GPUVec<int>::iterator> rowrt(
            rowr.begin(),
            rowr.end(),
            9);

    int offsets0[] = {-n1-1, -n1, -n1+1, -1, 0, 1, n1-1, n1, n1+1};
    GPUVec<int> o(offsets0+0, offsets0+9);
    repeated_range<GPUVec<int>::iterator> offsets(o.begin(), o.end(), dlen);

    SizedArray<int> row_ind(rowrt.end() - rowrt.begin(), "row_ind");
    thrust::copy(rowrt.begin(), rowrt.end(), row_ind.data);
    SizedArray<int> row_ptr(operator_rows+1, "row_ptr");
    SizedArray<int> col_ind(row_ind.size, "col_ind");
    thrust::transform(
        row_ind.data,
        row_ind.data + row_ind.size,
        offsets.begin(),
        col_ind.data,
        _1 + _2
        );

    cusparseHandle_t handle;
    cusparseStatus_t status;

    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE Library initialization failed." << std::endl;
        assert(false);
    }

    thrust::stable_sort_by_key(
        row_ind.data,
        row_ind.data + row_ind.size,
        make_zip_iterator(
            make_tuple(
                data.data,
                col_ind.data
                )
            )
        );

    int * cooRowInd = row_ind.data.get();
    int * csrRowPtr = row_ptr.data.get();
    int m = operator_rows;
    cusparseXcoo2csr(handle,
        cooRowInd, row_ind.size, m, csrRowPtr,
        CUSPARSE_INDEX_BASE_ZERO);

    std::string name = "mixed_for_vector CSR";

    SizedArray<double> R(1, "R");
    SizedArray<double> high_dirichlet(1, "high_dirichlet");
    SizedArray<double> low_dirichlet(1, "low_dirichlet");
    SizedArray<double> top_factors(1, "top_factors");
    SizedArray<double> bottom_factors(1, "bottom_factors");
    bool has_high_dirichlet = false;
    bool has_low_dirichlet = false;
    std::string top_fold_status = CANNOT_FOLD;
    std::string bottom_fold_status = CANNOT_FOLD;
    bool has_residual = false;

    int axis = 1;

    return new _CSRBandedOperator(
        data,
        row_ptr,
        row_ind,
        col_ind,
        R,
        high_dirichlet,
        low_dirichlet,
        top_factors,
        bottom_factors,
        axis,
        operator_rows,
        blocks,
        has_high_dirichlet,
        has_low_dirichlet,
        top_fold_status,
        bottom_fold_status,
        has_residual,
        name);
}
