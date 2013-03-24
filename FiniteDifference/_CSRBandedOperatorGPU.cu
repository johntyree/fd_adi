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

    /* GPUVec<double> d0(n0); */
    /* GPUVec<double> d1(n1); */

    /* thrust::adjacent_difference(v0.data, v0.data + n0, d0.begin()); */
    /* thrust::adjacent_difference(v1.data, v1.data + n1, d1.begin()); */

    /* GPUVec<double> sup(n1-2); */
    /* GPUVec<double> mid(n1-2); */
    /* GPUVec<double> sub(n1-2); */

    /* thrust::for_each( */
            /* thrust::make_zip_iterator( */
                /* thrust::make_tuple( */
                    /* sup.begin(), */
                    /* mid.begin(), */
                    /* sub.begin(), */
                    /* d1.begin()+1, */
                    /* d1.begin()+2 */
                    /* ) */
                /* ), */
            /* thrust::make_zip_iterator( */
                /* thrust::make_tuple( */
                    /* sup.end(), */
                    /* mid.end(), */
                    /* sub.end(), */
                    /* d1.end()-1, */
                    /* d1.end() */
                    /* ) */
                /* ), */
            /* first_deriv_() */
            /* ); */

    /* SizedArray<double> data(dlen * 9, "data"); */


    /* thrust::device_ptr<double> ssup(data.data); */
    /* thrust::device_ptr<double> smid(data.data+1*dlen); */
    /* thrust::device_ptr<double> ssub(data.data+2*dlen); */

    /* thrust::device_ptr<double> msup(data.data+3*dlen); */
    /* thrust::device_ptr<double> mmid(data.data+4*dlen); */
    /* thrust::device_ptr<double> msub(data.data+5*dlen); */

    /* thrust::device_ptr<double> bsup(data.data+6*dlen); */
    /* thrust::device_ptr<double> bmid(data.data+7*dlen); */
    /* thrust::device_ptr<double> bsub(data.data+8*dlen); */


    /* tiled_range<DptrIterator> supsup(ssup, ssup+dlen, n0-2); */
    /* tiled_range<DptrIterator> supmid(smid, smid+dlen, n0-2); */
    /* tiled_range<DptrIterator> supsub(ssub, ssub+dlen, n0-2); */

    /* tiled_range<DptrIterator> midsup(msup, msup+dlen, n0-2); */
    /* tiled_range<DptrIterator> midmid(mmid, mmid+dlen, n0-2); */
    /* tiled_range<DptrIterator> midsub(msub, msub+dlen, n0-2); */

    /* tiled_range<DptrIterator> subsup(bsup, bsup+dlen, n0-2); */
    /* tiled_range<DptrIterator> submid(bmid, bmid+dlen, n0-2); */
    /* tiled_range<DptrIterator> subsub(bsub, bsub+dlen, n0-2); */


    /* sup.resize(n0-2); */
    /* mid.resize(n0-2); */
    /* sub.resize(n0-2); */
    /* thrust::for_each( */
            /* thrust::make_zip_iterator( */
                /* thrust::make_tuple( */
                    /* sup.begin(), */
                    /* mid.begin(), */
                    /* sub.begin(), */
                    /* d0.begin()+1, */
                    /* d0.begin()+2 */
                    /* ) */
                /* ), */
            /* thrust::make_zip_iterator( */
                /* thrust::make_tuple( */
                    /* sup.end(), */
                    /* mid.end(), */
                    /* sub.end(), */
                    /* d0.end()-1, */
                    /* d0.end() */
                    /* ) */
                /* ), */
            /* first_deriv_() */
            /* ); */

    /* repeated_range<DptrIterator> dsup(sup.begin(), sup.end(), n1-2); */
    /* repeated_range<DptrIterator> dmid(mid.begin(), mid.end(), n1-2); */
    /* repeated_range<DptrIterator> dsub(sub.begin(), sub.end(), n1-2); */


    /* thrust::for_each( */
            /* thrust::make_zip_iterator( */
                /* thrust::make_tuple( */
                    /* supsup.begin(), */
                    /* supmid.begin(), */
                    /* supsub.begin(), */
                    /* dsup.begin() */
                    /* ) */
                /* ), */
            /* thrust::make_zip_iterator( */
                /* thrust::make_tuple( */
                    /* supsup.end(), */
                    /* supmid.end(), */
                    /* supsub.end(), */
                    /* dsup.end() */
                    /* ) */
                /* ), */
            /* multiply3x1() */
            /* ); */

    /* thrust::for_each( */
            /* thrust::make_zip_iterator( */
                /* thrust::make_tuple( */
                    /* midsup.begin(), */
                    /* midmid.begin(), */
                    /* midsub.begin(), */
                    /* dmid.begin() */
                    /* ) */
                /* ), */
            /* thrust::make_zip_iterator( */
                /* thrust::make_tuple( */
                    /* midsup.end(), */
                    /* midmid.end(), */
                    /* midsub.end(), */
                    /* dmid.end() */
                    /* ) */
                /* ), */
            /* multiply3x1() */
            /* ); */

    /* thrust::for_each( */
            /* thrust::make_zip_iterator( */
                /* thrust::make_tuple( */
                    /* subsup.begin(), */
                    /* submid.begin(), */
                    /* subsub.begin(), */
                    /* dsub.begin() */
                    /* ) */
                /* ), */
            /* thrust::make_zip_iterator( */
                /* thrust::make_tuple( */
                    /* subsup.end(), */
                    /* submid.end(), */
                    /* subsub.end(), */
                    /* dsub.end() */
                    /* ) */
                /* ), */
            /* multiply3x1() */
            /* ); */

    /* [> row = np.tile(np.arange(1, n1-1), n0-2) <] */
    /* tiled_range<counting_iterator<int> > row0( */
            /* counting_iterator<int>(1), */
            /* counting_iterator<int>(n1-1), */
            /* n0-2 */
            /* ); */
    /* [> row += np.repeat(np.arange(1, n0-1), n1-2) * n1 <] */
    /* repeated_range<counting_iterator<int> > row1( */
            /* counting_iterator<int>(1), */
            /* counting_iterator<int>(n0-1), */
            /* n1-2 */
            /* ); */
    /* GPUVec<int> rowr(dlen); */
    /* thrust::transform( */
            /* row1.begin(), */
            /* row1.end(), */
            /* rowr.begin(), */
            /* _1 * n1 */
            /* ); */

    /* [> row = np.tile(row, 9) <] */
    /* tiled_range<GPUVec<int>::iterator> rowrt( */
            /* rowr.begin(), */
            /* rowr.end(), */
            /* 9); */

    /* int offsets0[] = {-n1-1, -n1, -n1+1, -1, 0, 1, n1-1, n1, n1+1}; */
    /* GPUVec<int> o(offsets0+0, offsets0+9); */
    /* repeated_range<GPUVec<int>::iterator> offsets(o.begin(), o.end(), (n1-2) * (n0-2)); */

    /* GPUVec<int> row(rowrt.begin(), rowrt.end()); */
    /* GPUVec<int> rowptr(row.size()); */
    /* SizedArray<int> col(row.size(), "col"); */
    /* thrust::transform( */
        /* row.begin(), */
        /* row.end(), */
        /* offsets.begin(), */
        /* col.data, */
        /* thrust::plus<int>() */
        /* ); */

    SizedArray<double> data(dlen*9, "row_ptr");
    SizedArray<int> row_ptr(dlen*9, "row_ptr");
    SizedArray<int> row_ind(dlen*9, "row_ind");
    SizedArray<int> col_ind(dlen*9, "col_ind");

    thrust::fill_n(data.data, data.size, 0);
    thrust::fill_n(row_ptr.data, row_ptr.size, 0);
    thrust::fill_n(row_ind.data, row_ind.size, 0);
    thrust::fill_n(col_ind.data, col_ind.size, 0);

    cusparseHandle_t handle;
    cusparseStatus_t status;

    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE Library initialization failed." << std::endl;
        assert(false);
    }

    int * cooRowInd = row_ind.data.get();
    int * csrRowPtr = row_ptr.data.get();
    int m = operator_rows;
    cusparseXcoo2csr(handle,
        cooRowInd, row_ind.size, m, csrRowPtr,
        CUSPARSE_INDEX_BASE_ZERO);

    /* SizedArray<int> row_ptr(rowptr.raw(), rowptr.size(), "row_ptr", false); */
    /* SizedArray<int> row_ind(row.raw(), row.size(), "row_ind", false); */

    std::string name = "mixed_for_vector CSR";

    return new _CSRBandedOperator(data, row_ptr, row_ind,
            col_ind, operator_rows, blocks, name);
}
