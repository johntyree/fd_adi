#include "GNUC_47_compat.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include <sys/select.h>

#include <cusparse_v2.h>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/version.h>

#include "tiled_range.h"
#include "strided_range.h"

#include "_TriBandedOperatorGPU.cuh"

using thrust::make_constant_iterator;
using thrust::make_counting_iterator;
using thrust::make_tuple;
using thrust::make_zip_iterator;


template <typename T, typename U>
int find_index(T haystack, U needle, int max) {
    FULLTRACE;
    int idx;
    for (idx = 0; idx < max; ++idx) {
        if (haystack[idx] == needle) break;
    }
    if (idx >= max) {
        /* LOG("Did not find "<<needle<<" before reaching index "<<max<<"."); */
        /* std::cout << '\t'; */
        /* std::cout << "Haystack [ "; */
        /* for (idx = 0; idx < max; ++idx) { */
            /* std::cout << haystack[idx] << " "; */
        /* } */
        /* std::cout << "]"; ENDL; */
        idx = -1;
    }
    FULLTRACE;
    return idx;
}


_TriBandedOperator::_TriBandedOperator(
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
        ) :
    diags(data, true),
    R(R, true),
    high_dirichlet(high_dirichlet, true),
    low_dirichlet(low_dirichlet, true),
    top_factors(top_factors, true),
    bottom_factors(bottom_factors, true),
    axis(axis),
    main_diag(1),
    operator_rows(operator_rows),
    blocks(blocks),
    block_len(operator_rows / blocks),
    sup(diags.data),
    mid(diags.data + operator_rows),
    sub(diags.data + 2*operator_rows),
    has_high_dirichlet(has_high_dirichlet),
    has_low_dirichlet(has_low_dirichlet),
    top_fold_status(top_fold_status),
    bottom_fold_status(bottom_fold_status),
    has_residual(has_residual)
    {
        /* LOG("TriBandedOperator initialize START."); */
        verify_diag_ptrs();
        status = cusparseCreate(&handle);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            DIE("CUSPARSE Library initialization failed.");
        }
        /* LOG("TriBandedOperator initialize END."); */
    }


void _TriBandedOperator::verify_diag_ptrs() {
    FULLTRACE;

    if (sup.get() == 0 || mid.get() == 0 || sub.get() == 0) {
        DIE("Diag pointers aren't non-null");
    }

    FULLTRACE;
}


#if 0
struct DMVPY_f : thrust::unary_function<const Septuple &, REAL_t> {

    int op;

    DMVPY_f(char o) : op(o == 'p' ? 1 : -1) {}

    __host__ __device__
    REAL_t operator()(const Septuple &t) {
        using thrust::get;
        const REAL_t &a = get<0>(t);
        const REAL_t &b = get<1>(t);
        const REAL_t &c = get<2>(t);
        const REAL_t &x0 = get<3>(t);
        const REAL_t &x1 = get<4>(t);
        const REAL_t &x2 = get<5>(t);
        const REAL_t &y = get<6>(t);
        return y + op*(a*x0 + b*x1 + c*x2);
    }
};
void _TriBandedOperator::DMVPY(SizedArray<double> &V, char operation, SizedArray<double> &Y,
        SizedArray<double> &out) {
    FULLTRACE;
    const unsigned N = V.size;

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

    if (axis == 0) {
        V.transpose(1);
    }
    switch (operation) {
        case 'm':
        case 'p':
            out.data[0] = Y.data[0]
                - (mid[0]*V.data[0] + sup[0]*V.data[1]);

            thrust::transform(
                make_zip_iterator(make_tuple(
                    sub+1, mid+1, sup+1,
                    V.data, V.data+1, V.data+2,
                    Y.data+1)),
                make_zip_iterator(make_tuple(
                    sub+N-1, mid+N-1, sup+N-1,
                    V.data+N-2, V.data+N-1, V.data+N,
                    Y.data+N-1)),
                out.data+1,
                DMVPY_f(operation));

            out.data[N-1] = Y.data[N-1]
                - (sub[N-1]*V.data[N-2] + mid[N-1]*V.data[N-1]);

            break;
        default:
            DIE("Unknown operation in DMVPY '"<<operation<<"'");
    }
    return;
}
#endif


struct zipdot3 {
    template <typename T>
    __host__ __device__
    REAL_t operator()(const T &diags, const T &x) {
        using thrust::get;
        const REAL_t a = get<0>(diags);
        const REAL_t b = get<1>(diags);
        const REAL_t c = get<2>(diags);
        const REAL_t x0 = get<0>(x);
        const REAL_t x1 = get<1>(x);
        const REAL_t x2 = get<2>(x);
        return a*x0 + b*x1 + c*x2;
    }
};
struct zipdotTopAndBottom {
    template <typename T>
    __host__ __device__
    void operator()(T t) {
        using thrust::get;

        REAL_t &ret = get<0>(t);
        const REAL_t &a  = get<1>(t);
        const REAL_t &x0 = get<2>(t);
        const REAL_t &b  = get<3>(t);
        const REAL_t &x1 = get<4>(t);

        ret = a*x0 + b*x1;

        REAL_t &retN = get<5>(t);
        const REAL_t &b1  = get<6>(t);
        const REAL_t &xN1 = get<7>(t);
        const REAL_t &c1  = get<8>(t);
        const REAL_t &xN  = get<9>(t);

        retN = b1*xN1 + c1*xN;
    }
};


void _TriBandedOperator::apply(SizedArray<double> &V) {
    FULLTRACE;
    if (top_fold_status == CAN_FOLD || bottom_fold_status == CAN_FOLD) {
        DIE("Must be tridiagonal to apply operator on GPU.");
    }
    const unsigned N = V.size;

    // TODO: This Iterator works, but is it right?
    typedef GPUVec<REAL_t>::iterator Iterator;
    strided_range<Iterator> u0(V.data, V.data+V.size, block_len);
    strided_range<Iterator> u1(V.data+block_len-1, V.data+V.size, block_len);

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


    // Upper and lower edge cases simultaneously
    thrust::for_each(
        make_zip_iterator(make_tuple(V.tempspace, mid, V.data, sup, V.data+1,
                V.tempspace+N-1, sub+N-1, V.data+N-2, mid+N-1, V.data+N-1)
            ),
        make_zip_iterator(make_tuple(V.tempspace+1, mid+1, V.data+1, sup+1,
                V.data+2, V.tempspace+N, sub+N, V.data+N-1, mid+N, V.data+N)
            ),
        zipdotTopAndBottom()
    );

    // Regular dot product
    thrust::transform(
        make_zip_iterator(make_tuple(sub+1, mid+1, sup+1)),
        make_zip_iterator(make_tuple(sub+N-1, mid+N-1, sup+N-1)),
        make_zip_iterator(make_tuple(V.data, V.data+1, V.data+2)),
        V.tempspace+1,
        zipdot3()
    );


    if (is_folded()) {
        std::swap(V.tempspace, V.data);
        fold_vector(V, true);
        std::swap(V.tempspace, V.data);
    }


    if (has_residual) {
        thrust::transform(
                V.tempspace,
                V.tempspace + V.size,
                R.data,
                V.data,
                thrust::plus<double>());
    } else {
        if (V.owner) {
            std::swap(V.tempspace, V.data);
        } else {
            thrust::copy(V.tempspace, V.tempspace + V.size, V.data);
        }
    }


    if (axis == 0) {
        V.transpose(1);
    }
    FULLTRACE;
    return;
}


struct periodic_from_to_mask : thrust::unary_function<int, bool> {
    int begin;
    int end;
    int period;

    periodic_from_to_mask(int begin, int end, int period)
        : begin(begin-1), end(end+1), period(period) {}

    __host__ __device__
    bool operator()(int idx) {
        return (idx % period != begin && idx % period != end);
    }
};

/*
* Add a second BandedOperator to this one.
* Does not alter self.R, the residual vector.
*/
void _TriBandedOperator::add_operator(_TriBandedOperator &other) {
    FULLTRACE;
    if (is_folded() || other.is_folded()) {
        DIE("Cannot add folded (diagonalized) operators");
    }
    int begin = has_low_dirichlet;
    int end = block_len-1 - has_high_dirichlet;
    int o, to, fro;
    for (int i = 0; i < 3; ++i) {
        fro = to = i;
        o = 1-i;
        if (o == 0) {
            thrust::transform_if(
                    &diags.data[diags.idx(to, 0)],
                    &diags.data[diags.idx(to, 0)] + operator_rows,
                    &other.diags.data[diags.idx(fro, 0)],
                    make_counting_iterator(0),
                    &diags.data[diags.idx(to, 0)],
                    thrust::plus<double>(),
                    periodic_from_to_mask(begin, end, block_len));
        } else {
            thrust::transform(
                    &other.diags.data[diags.idx(fro, 0)],
                    &other.diags.data[diags.idx(fro, 0)] + other.diags.shape[1],
                    &diags.data[diags.idx(to, 0)],
                    &diags.data[diags.idx(to, 0)],
                    thrust::plus<double>());
        }
    }

    if (other.top_fold_status == CAN_FOLD) {
        int them = other.top_factors.size;
        int us = top_factors.size;
        if (them != us) {
            DIE("Bottom_factors are different sizes:" << us << ", " << them);
        }
        thrust::transform(
            top_factors.data,
            top_factors.data + top_factors.size,
            other.top_factors.data,
            top_factors.data,
            thrust::plus<double>());
    }

    if (other.bottom_fold_status == "CAN_FOLD") {
        int them = other.bottom_factors.size;
        int us = bottom_factors.size;
        if (them != us) {
            DIE("Bottom_factors are different sizes:" << us << ", " << them);
        }
        thrust::transform(
            bottom_factors.data,
            bottom_factors.data + bottom_factors.size,
            other.bottom_factors.data,
            bottom_factors.data,
            thrust::plus<double>());
    }

    thrust::transform(
            R.data,
            R.data + R.size,
            other.R.data,
            R.data,
            thrust::plus<double>());
    FULLTRACE;
}


/* Add a scalar to the main diagonal.
* Does not alter the residual vector.
*/
void _TriBandedOperator::add_scalar(double val) {
    FULLTRACE;

    int begin = has_low_dirichlet;
    int end = block_len-1 - has_high_dirichlet;

    thrust::transform_if(
            &diags.data[diags.idx(main_diag, 0)],
            &diags.data[diags.idx(main_diag, 0)] + operator_rows,
            make_constant_iterator(val),
            make_counting_iterator(0),
            &diags.data[diags.idx(main_diag, 0)],
            thrust::plus<double>(),
            periodic_from_to_mask(begin, end, block_len));
    FULLTRACE;
}


bool _TriBandedOperator::is_folded() {
    return (top_fold_status == FOLDED || bottom_fold_status == FOLDED);
}


void _TriBandedOperator::solve(SizedArray<double> &V) {
    FULLTRACE;
    if (top_fold_status == CAN_FOLD || bottom_fold_status == CAN_FOLD) {
        DIE("Must be tridiagonal to apply inverse operator on GPU.");
    }
    const unsigned N = V.size;

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

    if (axis == 0) {
        V.transpose(1);
    }

    if (has_residual) {
        thrust::transform(V.data, V.data + V.size,
                R.data,
                V.data,
                thrust::minus<double>());
    }

    if (is_folded()) {
        fold_vector(V);
    }

    status = cusparseDgtsvStridedBatch(handle, N,
                sub.get(), mid.get(), sup.get(),
                V.data.get(),
                1, N);

    /*
     * if (block_len > 256) {
     *     DIE("Block_len Too big!");
     * }
     * if (blocks * block_len > V.size) {
     *     DIE("Indexing is wrong. Too large.");
     * }
     * triDiagonalSystemSolve<<<blocks, block_len>>>(V.size, sub.get(), mid.get(), sup.get(), V.data.get());
     * cudaDeviceSynchronize();
     */

    if (status != CUSPARSE_STATUS_SUCCESS) {
        DIE("CUSPARSE tridiag system solve failed.");
    }

    if (axis == 0) {
        V.transpose(1);
    }
    FULLTRACE;
    return;
}


template <typename Tuple, typename OP>
struct curry :
    public thrust::unary_function<Tuple, typename OP::result_type> {

    OP f;

    __host__ __device__
    typename OP::result_type operator()(Tuple t) {
        using thrust::get;
        return  f(get<0>(t), get<1>(t));
    }
};

template <typename Tuple, typename Result>
struct add_multiply3 : public thrust::unary_function<Tuple, Result> {
    Result direction;
    add_multiply3(Result x) : direction(x) {}
    __host__ __device__
    Result operator()(Tuple t) {
        using thrust::get;
        return  get<0>(t) + direction * get<1>(t) * get<2>(t);
    }
};


void _TriBandedOperator::fold_vector(SizedArray<double> &vector, bool unfold) {
    FULLTRACE;

    typedef GPUVec<REAL_t>::iterator Iterator;
    typedef thrust::tuple<REAL_t,REAL_t,REAL_t> REALTuple;

    strided_range<Iterator> u0(vector.data, vector.data + vector.size, block_len);
    strided_range<Iterator> u1(vector.data+1, vector.data + vector.size, block_len);

    strided_range<Iterator> un(vector.data+block_len-1, vector.data + vector.size, block_len);
    strided_range<Iterator> un1(vector.data+block_len-2, vector.data + vector.size, block_len);

    // Top fold
    if (top_fold_status == FOLDED) {
        /* LOG("Folding top. direction("<<unfold<<") top_factors("<<top_factors<<")"); */
        thrust::transform(
            make_zip_iterator(make_tuple(u0.begin(), u1.begin(), top_factors.data)),
            make_zip_iterator(make_tuple(u0.end(), u1.end(), top_factors.data + top_factors.size)),
            u0.begin(),
            add_multiply3<REALTuple, REAL_t>(unfold ? -1 : 1));
    }

    if (bottom_fold_status == FOLDED) {
        /* LOG("Folding bottom. direction("<<unfold<<") bottom_factors("<<bottom_factors<<")"); */
        thrust::transform(
            make_zip_iterator(make_tuple(un.begin(), un1.begin(), bottom_factors.data)),
            make_zip_iterator(make_tuple(un.end(), un1.end(), bottom_factors.data + bottom_factors.size)),
            un.begin(),
            add_multiply3<REALTuple, REAL_t>(unfold ? -1 : 1));
    }

    FULLTRACE;
}


void _TriBandedOperator::diagonalize() {
    FULLTRACE;
    /* LOG("Before folding: " << diags); */
    if (bottom_fold_status == CAN_FOLD) {
        /* LOG("Bottom:" << bottom_fold_status); */
        fold_bottom();
        /* LOG("Bottom:" << bottom_fold_status); */
    }
    if (top_fold_status == CAN_FOLD) {
        /* LOG("Top:" << top_fold_status); */
        fold_top();
        /* LOG("Top:" << top_fold_status); */
    }
    /* LOG("After folding: " << diags); */
    FULLTRACE;
}


void _TriBandedOperator::undiagonalize() {
    FULLTRACE;
    if (bottom_fold_status == FOLDED) {
        /* LOG("Bottom:" << bottom_fold_status); */
        fold_bottom(true);
        /* LOG("Bottom:" << bottom_fold_status); */
    }
    if (top_fold_status == FOLDED) {
        /* LOG("Top:" << top_fold_status); */
        fold_top(true);
        /* LOG("Top:" << top_fold_status); */
    }
    FULLTRACE;
}


/* These fold the third element of first row and third from last element of last
 * row into the neighboring row, resulting in a tridiagonal system.
 */
template <typename Tuple>
struct fold_operator : public thrust::unary_function<Tuple, void> {
    __host__ __device__
    void operator()(Tuple t) {
        using thrust::get;
        get<5>(t) = get<1>(t) == 0 ? 0 : -get<5>(t) / get<1>(t);
        get<0>(t) += get<3>(t) * get<5>(t);
        get<2>(t) += get<4>(t) * get<5>(t);
    }
};
template <typename Tuple>
struct unfold_operator : public thrust::unary_function<Tuple, void> {
    __host__ __device__
    void operator()(Tuple t) {
        using thrust::get;
        get<0>(t) -= get<3>(t) * get<5>(t);
        get<2>(t) -= get<4>(t) * get<5>(t);
        get<5>(t) *= -get<1>(t);
    }
};

void _TriBandedOperator::fold_top(bool unfold) {
    FULLTRACE;
    typedef thrust::tuple<REAL_t&, REAL_t&, REAL_t&, REAL_t&, REAL_t&, REAL_t&> REALTuple;
    typedef thrust::device_ptr<REAL_t> Ptr;

    strided_range<Ptr> c0 (sup  , sup+operator_rows, block_len);
    strided_range<Ptr> c1 (sup+1, sup+operator_rows, block_len);
    strided_range<Ptr> b0 (mid  , mid+operator_rows, block_len);
    strided_range<Ptr> b1 (mid+1, mid+operator_rows, block_len);
    strided_range<Ptr> a1 (sub+1, sub+operator_rows, block_len);

    if (unfold) {
        thrust::for_each(
            make_zip_iterator(
                make_tuple(
                    c0.begin(), c1.begin(),
                    b0.begin(), b1.begin(),
                                a1.begin(),
                    top_factors.data
                )
            ),
            make_zip_iterator(
                make_tuple(
                    c0.end(), c1.end(),
                    b0.end(), b1.end(),
                            a1.end(),
                    top_factors.data + top_factors.size
                )
            ),
            unfold_operator <REALTuple>()
        );
    } else {
        thrust::for_each(
            make_zip_iterator(
                make_tuple(
                    c0.begin(), c1.begin(),
                    b0.begin(), b1.begin(),
                                a1.begin(),
                    top_factors.data
                )
            ),
            make_zip_iterator(
                make_tuple(
                    c0.end(), c1.end(),
                    b0.end(), b1.end(),
                            a1.end(),
                    top_factors.data + top_factors.size
                )
            ),
            fold_operator<REALTuple>()
        );
    }
    if (unfold) top_fold_status = CAN_FOLD;
    else top_fold_status = FOLDED;
    FULLTRACE;
}


void _TriBandedOperator::fold_bottom(bool unfold) {
    FULLTRACE;
    typedef thrust::tuple<REAL_t&, REAL_t&, REAL_t&, REAL_t&, REAL_t&, REAL_t&> REALTuple;
    typedef thrust::device_ptr<REAL_t> Ptr;

    strided_range<Ptr> cn1(sup+(block_len-2)  , sup+operator_rows, block_len);
    strided_range<Ptr> bn (mid+(block_len-1)  , mid+operator_rows, block_len);
    strided_range<Ptr> bn1(mid+(block_len-1)-1, mid+operator_rows, block_len);
    strided_range<Ptr> an (sub+(block_len-1), sub+operator_rows, block_len);
    strided_range<Ptr> an1(sub+(block_len-1)-1, sub+operator_rows, block_len);

    if (unfold) {
        thrust::for_each(
            make_zip_iterator(
                make_tuple(
                    an.begin(), an1.begin(),
                    bn.begin(), bn1.begin(),
                                cn1.begin(),
                    bottom_factors.data
                )
            ),
            make_zip_iterator(
                make_tuple(
                    an.end(), an1.end(),
                    bn.end(), bn1.end(),
                            cn1.end(),
                    bottom_factors.data + bottom_factors.size
                )
            ),
            unfold_operator<REALTuple>()
        );
    } else {
        thrust::for_each(
            make_zip_iterator(
                make_tuple(
                    an.begin(), an1.begin(),
                    bn.begin(), bn1.begin(),
                                cn1.begin(),
                    bottom_factors.data
                )
            ),
            make_zip_iterator(
                make_tuple(
                    an.end(), an1.end(),
                    bn.end(), bn1.end(),
                            cn1.end(),
                    bottom_factors.data + bottom_factors.size
                )
            ),
            fold_operator<REALTuple>()
        );
    }

    if (unfold) bottom_fold_status = CAN_FOLD;
    else bottom_fold_status = FOLDED;
    FULLTRACE;
}


void _TriBandedOperator::vectorized_scale(SizedArray<double> &vector) {
    FULLTRACE;
    Py_ssize_t vsize = vector.size;
    Py_ssize_t block_len = operator_rows / blocks;

    typedef thrust::device_vector<REAL_t>::iterator Iterator;
    tiled_range<Iterator> v(vector.data, vector.data + vector.size, block_len);
    typedef tiled_range<Iterator>::iterator TiledIterator;

    strided_range<TiledIterator> u0(v.begin(), v.end(), block_len);
    strided_range<TiledIterator> u1(v.begin()+block_len-1, v.end(), block_len);

    /*
     * LOG("op_rows("<<operator_rows<<") vsize("<<vsize<<") "
     *     "v.d.size("<<vector.size<<") "
     *     "v.size()("<<v.end()-v.begin()<<") "
     *     "diags.shape("<<diags.shape[0]<<","<<diags.shape[1]<<") "
     *     "diags.idx(1,0)("<<diags.idx(1,0)<<") "
     *     );
     * LOG("diags.name("<<diags.name<<")");
     * LOG("diags.idx(0,op)("<<diags.idx(0,0)+operator_rows<<")");
     */

    if (is_folded()) {
        DIE("Cannot scale diagonalized operator.");
    }

    if (operator_rows % vsize != 0) {
        DIE("Vector length does not divide "
            "evenly into operator size. Cannot scale."
            << "\n vsize("<<vsize<<") operator_rows("<<operator_rows<<")");
    }

    if (vsize == 0) {DIE("vsize == 0")}

    for (Py_ssize_t row = 0; row < 3; ++row) {
        int o = 1 - row;
        if (o >= 0) { // upper diags
            thrust::transform(diags.data + diags.idx(row, 0),
                    diags.data + diags.idx(row, 0) + operator_rows - o,
                    v.begin(),
                    diags.data + diags.idx(row, 0),
                    thrust::multiplies<REAL_t>());
        } else { // lower diags
            thrust::transform(diags.data + diags.idx(row, -o),
                    diags.data + diags.idx(row, 0) + operator_rows,
                    v.begin() + -o,
                    diags.data + diags.idx(row, -o),
                    thrust::multiplies<REAL_t>());
        }
    }

    /* We check dirichlet to avoid multiplying by 1 */
    if (top_fold_status == CAN_FOLD && !has_low_dirichlet) {
        thrust::transform(
            top_factors.data,
            top_factors.data+top_factors.size,
            u0.begin(),
            top_factors.data,
            thrust::multiplies<REAL_t>());
    }
    if (bottom_fold_status == CAN_FOLD && !has_high_dirichlet) {
        thrust::transform(
            bottom_factors.data,
            bottom_factors.data+bottom_factors.size,
            u1.begin(),
            bottom_factors.data,
            thrust::multiplies<REAL_t>());
    }

    /* LOG("Scaled data."); */
    thrust::transform(R.data, R.data + R.size,
            v.begin(),
            R.data,
            thrust::multiplies<REAL_t>());
    /* LOG("Scaled R."); */
    FULLTRACE;
    return;
}
