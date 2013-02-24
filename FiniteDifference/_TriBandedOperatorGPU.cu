
#include "GNUC_47_compat.h"

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/version.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include "tiled_range.h"
#include "strided_range.h"

#include <algorithm>
#include <cstdlib>

#include <iostream>
#include <vector>
#include <iterator>
#include <cassert>
#include <stdexcept>

#include <sys/select.h>

#include <cusparse_v2.h>

#include "_TriBandedOperatorGPU.cuh"

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
        bool has_top_factors,
        bool has_bottom_factors,
        bool has_residual
        ) :
    diags(data),
    R(R),
    offsets(offsets),
    high_dirichlet(high_dirichlet),
    low_dirichlet(low_dirichlet),
    top_factors(top_factors),
    bottom_factors(bottom_factors),
    axis(axis),
    main_diag(find_index(offsets.data, 0, offsets.size)),
    operator_rows(operator_rows),
    blocks(blocks),
    block_len(operator_rows / blocks),
    sup(diags.data.ptr()),
    mid(diags.data.ptr() + operator_rows),
    sub(diags.data.ptr() + 2*operator_rows),
    has_high_dirichlet(has_high_dirichlet),
    has_low_dirichlet(has_low_dirichlet),
    has_top_factors(has_top_factors),
    has_bottom_factors(has_bottom_factors),
    has_residual(has_residual),
    is_tridiagonal(offsets.size == 3 && main_diag != -1)
    {
        verify_diag_ptrs();
        status = cusparseCreate(&handle);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            DIE("CUSPARSE Library initialization failed.");
        }
    }

void _TriBandedOperator::verify_diag_ptrs() {
    FULLTRACE;
    if (sup.get() == 0 || mid.get() == 0 || sub.get() == 0) {
        DIE("Diag pointers aren't non-null");
    }
    if (main_diag == -1) {
        /* LOG("No main diag means not tridiagonal, hopefully."); */
        return;
    }
    if (offsets.size != 3) {
        /* LOG("Not tridiagonal. Skipping diag ptrs check."); */
        return;
    }
    int idx;
    /* LOG("main_diag("<<main_diag<<")"); */
    idx = diags.idx(main_diag-1, 0);
    if (*sup != diags.data[idx]
            || (sup.get() != (&diags.data[diags.idx(0,0)]).get())) {
        DIE("sup[0] = " << *sup << " <->  " << diags.get(0,0)
            << "\n\tsup = " << sup.get() << " <->  "
                << (&diags.data[diags.idx(0, 0)]).get());
    }
    if (*mid != diags.data[diags.idx(main_diag, 0)]
            || (mid.get() != (&diags.data[diags.idx(main_diag, 0)]).get())) {
        DIE("mid[0] = " << *mid << " !=  " << diags.get(main_diag,0)
            << "\n\tmid = " << mid.get() << " <->  "
                << (&diags.data[diags.idx(main_diag, 0)]).get());
    }
    if (*sub != diags.data[diags.idx(main_diag+1, 0)]
            || (sub.get() != (&diags.data[diags.idx(main_diag+1, 0)]).get())) {
        DIE("sub[0] = " << *sub << " !=  " << diags.get(main_diag+1,0)
            << "\n\tsub = " << sub.get() << " <->  "
                << (&diags.data[diags.idx(main_diag+1, 0)]).get());
    }
    FULLTRACE;
}


struct zipdot3 : thrust::binary_function<const Triple &, const Triple &, REAL_t> {
    __host__ __device__
    REAL_t operator()(const Triple &diags, const Triple &x) {
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

SizedArray<double> *_TriBandedOperator::apply(SizedArray<double> &V) {
    FULLTRACE;
    using std::cout;
    using std::endl;
    const unsigned N = V.size;
    GPUVec<REAL_t> out(N);

    GPUVec<REAL_t> &in = V.data;

    if (!is_tridiagonal) {
        DIE("Can only apply tridiagonal operators when on the GPU.");
    }


    if (has_low_dirichlet) {
        /* print "Setting V[0,:] to", self.dirichlet[0] */
        // Some kind of thrust thing?
        /* V[...,0] = low_dirichlet[i] */
        thrust::copy(low_dirichlet.data.begin(),
                low_dirichlet.data.end(),
                in.begin());
    }
    if (has_high_dirichlet) {
        /* print "Setting V[0,:] to", self.dirichlet[0] */
        // Some kind of thrust thing?
        /* V[...,-1] = high_dirichlet[i] */
        thrust::copy(high_dirichlet.data.begin(),
                high_dirichlet.data.end(),
                in.end() - V.shape[1]);
    }

    if (axis == 0) {
        V.transpose(1);
    }



    GPUVec<REAL_t> a(sub, sub+N);
    GPUVec<REAL_t> b(mid, mid+N);
    GPUVec<REAL_t> c(sup, sup+N);

    out[0] = b[0]*in[0] + c[0]*in[1];
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(a.begin()+1, b.begin()+1, c.begin()+1)),
        thrust::make_zip_iterator(thrust::make_tuple(a.end()-1, b.end()-1, c.end()-1)),
        thrust::make_zip_iterator(thrust::make_tuple(in.begin(), in.begin()+1, in.begin()+2)),
        out.begin()+1,
        zipdot3()
    );
    out[N-1] = a[N-1]*in[N-2] + b[N-1]*in[N-1];

    SizedArray<double> *U = new SizedArray<double>(out,
            V.ndim, V.shape, "CPP Solve U from V");

    /* if (has_residual) { */
        /* thrust::transform(U->data.begin(), U->data.end(), */
                /* R.data.begin(), */
                /* U->data.begin(), */
                /* thrust::plus<double>()); */
    /* } */

    /* ret = ret.reshape(V.shape) */

    /* t = range(V.ndim) */
    /* utils.rolllist(t, V.ndim-1, self.axis) */

    /* if (axis == 0) { */
        /* U.transpose(); */
    /* } */
    /* return ret; */
    FULLTRACE;
    return U;
}


struct periodic_from_to_mask : thrust::unary_function<int, bool> {
    int begin;
    int end;
    int period;

    periodic_from_to_mask(int begin, int end, int period)
        : begin(begin-1), end(end+1), period(period) {
        }

    __host__ __device__
    bool operator()(int idx) {
        return (idx % period != begin && idx % period != end);
    }
};

void _TriBandedOperator::add_operator(_TriBandedOperator &other) {
    /* LOG("Adding operator @ " << &other << " to this one @ " << this); */
    /*
    * Add a second BandedOperator to this one.
    * Does not alter self.R, the residual vector.
    */
    FULLTRACE;
    int begin = has_low_dirichlet;
    int end = block_len-1 - has_high_dirichlet;
    int o, to, fro;
    for (int i = 0; i < other.offsets.size; i++) {
        fro = i;
        o = other.offsets.get(i);
        to = find_index(offsets.data, o, offsets.size);
        if (offsets.get(to) != o) {
            std::cout << std::endl;
            std::cout << "to: " << to << "(";
            /* print_array(&offsets(0), offsets.size); */
            std::cout << offsets.data;
            std::cout << ")";
            std::cout << "fro: " << fro << "(";
            std::cout << other.offsets.data;
            /* print_array(&other.offsets(0), other.offsets.size); */
            std::cout << ")" << std::endl;
            assert(offsets.get(to) == o);
        }
        /* LOG("Adding offset " << o << "."); */
        if (o == 0) {
            thrust::transform_if(
                    &diags.data[diags.idx(to, 0)],
                    &diags.data[diags.idx(to, 0)] + operator_rows,
                    &other.diags.data[diags.idx(fro, 0)],
                    thrust::make_counting_iterator(0),
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
    /* LOG("Adding R."); */
    thrust::transform(
            R.data.begin(),
            R.data.end(),
            other.R.data.begin(),
            R.data.begin(),
            thrust::plus<double>());
    FULLTRACE;
}



void _TriBandedOperator::add_scalar(double val) {
    FULLTRACE;
    /* Add a scalar to the main diagonal.
     * Does not alter the residual vector.
     */
    // We add it to the main diagonal.

    int begin = has_low_dirichlet;
    int end = block_len-1 - has_high_dirichlet;

    /* LOG("has_low("<<has_low_dirichlet<<") " */
        /* "has_high("<<has_high_dirichlet<<") " */
        /* "blocklen("<<block_len<<") "); */

    thrust::transform_if(
            &diags.data[diags.idx(main_diag, 0)],
            &diags.data[diags.idx(main_diag, 0)] + operator_rows,
            thrust::make_constant_iterator(val),
            thrust::make_counting_iterator(0),
            &diags.data[diags.idx(main_diag, 0)],
            thrust::plus<double>(),
            periodic_from_to_mask(begin, end, block_len));
    FULLTRACE;
}

bool _TriBandedOperator::is_folded() {
    return has_top_factors || has_bottom_factors;
}



int _TriBandedOperator::solve(SizedArray<double> &V) {
    FULLTRACE;
    if (!is_tridiagonal) {
        DIE("Can only solve tridiagonal systems when on the GPU.");
    }
    verify_diag_ptrs();

    /* std::cout << "Begin C Solve\n"; */
    /* std::cout << "Copy Host->Dev... " << V.data << ' '; */
    GPUVec<double> d_V(V.data);
    GPUVec<double> d_sup(sup, sup+V.size);
    GPUVec<double> d_mid(mid, mid+V.size);
    GPUVec<double> d_sub(sub, sub+V.size);
    /* std::cout << "OK\n"; */

    /* std::cout << "CUSPARSE... "; */
    status = cusparseDgtsvStridedBatch(handle, V.size,
            d_sub.raw(), d_mid.raw(), d_sup.raw(),
            d_V.raw(),
            1, V.size);
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE tridiag system solve failed." << std::endl;
        return 1;
    }
    /* std::cout << "OK\n"; */

    /* std::cout << "Copy Dev->Host... " << d_V << ' '; */
    thrust::copy(d_V.begin(), d_V.end(), V.data.begin());
    /* std::cout << "OK\n"; */
    /* std::cout << "End C Solve\n"; */
    FULLTRACE;
    return 0;
}

/*
 * cpdef fold_vector(self, double[:] v, unfold=False):
 *     cdef int direction, u0, u1, un ,un1
 *     blocks = self.blocks
 *     block_len = self.shape[0] // blocks
 *
 *     for b in range(blocks):
 *         u0 = b*block_len
 *         u1 = u0 + 1
 *         un = (b+1)*block_len - 1
 *         un1 = un - 1
 *         # print u0, u1, un1, un
 *         # print "[%f, %f .. %f, %f]" % (v[u0], v[u1], v[un1], v[un])
 *         direction = -1 if unfold else 1
 *         if self.top_factors is not None:
 *             v[u0] += direction * v[u1]  * self.top_factors[b]
 *         if self.bottom_factors is not None:
 *             v[un] += direction * v[un1] * self.bottom_factors[b]
 *     return np.asarray(v)
 */

template <typename Tuple, typename OP>
struct curry : public thrust::unary_function<Tuple, typename OP::result_type> {

    OP f;

    __host__ __device__
    typename OP::result_type operator()(Tuple t) {
        using thrust::get;
        return  f(get<0>(t), get<1>(t));
    }
};

template <typename Tuple, typename Result>
struct fold_it : public thrust::unary_function<Tuple, Result> {

    Result direction;

    fold_it(Result x) : direction(x) {}

    __host__ __device__
    Result operator()(Tuple t) {
        using thrust::get;
        return  get<0>(t) + direction * get<1>(t) * get<2>(t);
    }
};


void _TriBandedOperator::fold_vector(GPUVec<double> &vector, bool unfold) {
    FULLTRACE;

    typedef thrust::device_vector<REAL_t>::iterator Iterator;

    strided_range<Iterator> u0(vector.begin(), vector.end(), block_len);
    strided_range<Iterator> u1(vector.begin()+1, vector.end(), block_len);

    strided_range<Iterator> un(vector.begin()+block_len-1, vector.end(), block_len);
    strided_range<Iterator> un1(vector.begin()+block_len-2, vector.end(), block_len);

    using thrust::make_zip_iterator;
    using thrust::make_tuple;
    using thrust::get;

    typedef thrust::tuple<REAL_t,REAL_t,REAL_t> REALTuple;
    typedef thrust::device_vector<REAL_t>::iterator Iterator;

    LOG("has_top_factors("<<has_top_factors<<") has_bottom_factors("<<has_bottom_factors<<")");

    // Top fold
    if (has_top_factors) {
        /* LOG("Folding top. direction("<<unfold<<") top_factors("<<top_factors<<")"); */
        thrust::transform(
            make_zip_iterator(make_tuple(u0.begin(), u1.begin(), top_factors.data.begin())),
            make_zip_iterator(make_tuple(u0.end(), u1.end(), top_factors.data.end())),
            u0.begin(),
            fold_it<REALTuple, REAL_t>(unfold ? -1 : 1));
    }

    if (has_bottom_factors) {
        /* LOG("Folding bottom. direction("<<unfold<<") bottom_factors("<<bottom_factors<<")"); */
        thrust::transform(
            make_zip_iterator(make_tuple(un.begin(), un1.begin(), bottom_factors.data.begin())),
            make_zip_iterator(make_tuple(un.end(), un1.end(), bottom_factors.data.end())),
            un.begin(),
            fold_it<REALTuple, REAL_t>(unfold ? -1 : 1));
    }

    FULLTRACE;
}



void _TriBandedOperator::vectorized_scale(SizedArray<double> &vector) {
    FULLTRACE;
    Py_ssize_t vsize = vector.size;
    Py_ssize_t noffsets = offsets.size;
    Py_ssize_t block_len = operator_rows / blocks;

    typedef thrust::device_vector<REAL_t>::iterator Iterator;
    tiled_range<Iterator> v(vector.data.begin(), vector.data.end(), operator_rows / vsize);
    /*
     * LOG("op_rows("<<operator_rows<<") vsize("<<vsize<<") "
     *     "v.d.size("<<vector.data.size()<<") "
     *     "v.size()("<<v.end()-v.begin()<<") "
     *     "diags.shape("<<diags.shape[0]<<","<<diags.shape[1]<<") "
     *     "diags.idx(1,0)("<<diags.idx(1,0)<<") "
     *     );
     * LOG("diags.name("<<diags.name<<")");
     * LOG("diags.idx(0,op)("<<diags.idx(0,0)+operator_rows<<")");
     */

    if (operator_rows % vsize != 0) {
        DIE("Vector length does not divide "
            "evenly into operator size. Cannot scale."
            << "\n vsize("<<vsize<<") operator_rows("<<operator_rows<<")");
    }
    if ((size_t)vsize != vector.data.size()) {DIE("vsize != vector.data.size()")}
    if (vsize == 0) {DIE("vsize == 0")}

    if (has_low_dirichlet) {
        for (Py_ssize_t b = 0; b < blocks; ++b) {
            vector.data[vector.idx(b*block_len % vsize)] = 1;
        }
    }

    if (has_high_dirichlet) {
        for (Py_ssize_t b = 0; b < blocks; ++b) {
            vector.data[vector.idx((b+1)*block_len - 1 % vsize)] = 1;
        }
    }

    for (Py_ssize_t row = 0; row < noffsets; ++row) {
        int o = offsets.get(row);
        if (o >= 0) { // upper diags
            thrust::transform(diags.data.begin() + diags.idx(row, 0),
                    diags.data.begin() + diags.idx(row, 0) + operator_rows - o,
                    v.begin(),
                    diags.data.begin() + diags.idx(row, 0),
                    thrust::multiplies<REAL_t>());
        } else { // lower diags
            thrust::transform(diags.data.begin() + diags.idx(row, -o),
                    diags.data.begin() + diags.idx(row, 0) + operator_rows,
                    v.begin() + -o,
                    diags.data.begin() + diags.idx(row, -o),
                    thrust::multiplies<REAL_t>());
        }
    }
    /* LOG("Scaled data."); */
    thrust::transform(R.data.begin(), R.data.end(),
            v.begin(),
            R.data.begin(),
            thrust::multiplies<REAL_t>());
    /* LOG("Scaled R."); */
    FULLTRACE;
    return;
}

int main () {

    thrust::host_vector<double> a(10);
    int block_len = 5;
    int begin = 1;
    int end = block_len-1 - 1;

    thrust::transform_if(
            a.begin(),
            a.end(),
            thrust::make_constant_iterator(2),
            thrust::make_counting_iterator(0),
            a.begin(),
            thrust::plus<double>(),
            periodic_from_to_mask(begin, end, block_len));

    printf("\n");
    print_array(a.data(), a.size());
    return 0;
}
