#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iomanip>

/* #include <thrust/host_vector.h> */
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <cusparse_v2.h>
#include <cuda_runtime.h>

#include "strided_range.h"
#include "tiled_range.h"

typedef thrust::device_vector<double> Vec;
typedef thrust::device_ptr<double> Dptr;
typedef thrust::detail::normal_iterator<thrust::device_ptr<double> > DptrIterator;

using namespace thrust::placeholders;

using std::cout;
using std::endl;
using std::setw;
const double NaN = std::numeric_limits<double>::quiet_NaN();

double* raw(Vec &v) {
   return thrust::raw_pointer_cast(v.data());
}

struct first_deriv {
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        using thrust::get;
        get<0>(t) = get<3>(t)                / (get<4>(t) * (get<3>(t) + get<4>(t)));
        get<1>(t) = (-get<3>(t) + get<4>(t)) / (get<3>(t) * get<4>(t));
        get<2>(t) = -get<4>(t)               / (get<3>(t) * (get<3>(t) + get<4>(t)));
    }
};
struct second_deriv {
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        using thrust::get;
        const double x = get<3>(t) + get<4>(t);
        get<0>(t) =  2. / (get<4>(t) * x);
        get<1>(t) = -2. / (get<4>(t)*get<3>(t));
        get<2>(t) =  2. / (get<3>(t) * x);
        /* get<0>(t) =  2.; */
        /* get<1>(t) = -2.; */
        /* get<2>(t) =  2.; */
    }
};

struct dirichlet_boundary {

    dirichlet_boundary(double val) : val(val) {}

    double val;

    template <typename Tuple>
    __host__ __device__
    // (sup, mid, sub, dirichlet)
    void operator()(Tuple t) {
        using thrust::get;
        get<0>(t) =  0;
        get<1>(t) =  1;
        get<2>(t) =  0;
        get<3>(t) = val;
    }
    /* # Dirichlet boundary. No derivatives, but we need to preserve the */
    /* # value we get, because we will have already forced it. */
    /* Bdata[m, 0] = 1 */
    /* B.dirichlet[0] = lower_val */
};

struct von_neumann_boundary {
// thrust::fill...
    von_neumann_boundary(double val) : val(val) {}

    double val;

    template <typename Tuple>
    __host__ __device__
    // (sup, mid, sub, residual)
    void operator()(Tuple t) {
        using thrust::get;
        get<0>(t) = 0;
        get<1>(t) = 0;
        get<2>(t) = 0;
        get<3>(t) = val;
    }
};

struct free_boundary_first {
    template <typename Tuple>
    __host__ __device__
    // (sup, mid, sub, d[1])
    // (mid, sub, sup, d[-1])
    void operator()(Tuple t) {
        using thrust::get;
        get<0>(t) =  1 / get<3>(t);
        get<1>(t) = -1 / get<3>(t);
        get<2>(t) = 0;
    }
    /* # Try first order to preserve tri-diag */
    /* Bdata[m - 1, 1] =  1 / d[1] */
    /* Bdata[m,     0] = -1 / d[1] */
    /* # First order backward */
    /* Bdata[m, -1]     =  1.0 / d[-1] */
    /* Bdata[m + 1, -2] = -1.0 / d[-1] */
};


struct free_boundary_second_with_first_derivative_one {
    template <typename Tuple>
    __host__ __device__
    // (sup, mid, sub, d[1], R[0])
    // (sub, mid, sup, d[-1], R[-1]) yes mid is still neg
    void operator()(Tuple t) {
        using thrust::get;
        const double x = get<3>(t)*get<3>(t);
        const double fst_deriv = 1;
        get<0>(t) =  2 / x;
        get<1>(t) = -2 / x;
        get<2>(t) = 0;
        get<4>(t) = -fst_deriv*2 / get<3>(t);
    }
    /* Bdata[m-1, 1] =  2 / d[1]**2 */
    /* Bdata[m,   0] = -2 / d[1]**2 */
    /* R[0]         =  -fst_deriv * 2 / d[1] */
    /* Bdata[m,   -1] = -2 / d[-1]**2 */
    /* Bdata[m+1, -2] =  2 / d[-1]**2 */
    /* R[-1]          =  fst_deriv * 2 / d[-1] */
};

struct free_boundary_second {
    template <typename Tuple>
    __host__ __device__
    // (supsup, sup, mid, sub, d[1], d[2])
    // (subsub, sub, mid, sup, d[-1], d[-2])
    void operator()(Tuple t) {
        using thrust::get;
        const double recip_denom =
            1.0 / (0.5 * (get<4>(t)+get<5>(t))*get<4>(t)*get<5>(t));
        get<0>(t) = get<4>(t)              * recip_denom;
        get<1>(t) = -(get<4>(t)+get<5>(t)) * recip_denom;
        get<2>(t) = get<5>(t)              * recip_denom;
        get<3>(t) = 0;
    }
    /* recip_denom = 1.0 / (0.5*(d[2]+d[1])*d[2]*d[1]); */
    /* Bdata[m-2,2] = d[1]         * recip_denom */
    /* Bdata[m-1,1] = -(d[2]+d[1]) * recip_denom */
    /* Bdata[m,0]   = d[2]         * recip_denom */
    /* recip_denom = 1.0 / (0.5*(d[-2]+d[-1])*d[-2]*d[-1]); */
    /* Bdata[m+2,-3] = d[-1]          * recip_denom */
    /* Bdata[m+1,-2] = -(d[-2]+d[-1]) * recip_denom */
    /* Bdata[m,-1]   = d[-2]          * recip_denom */
};


struct Diags {
    thrust::device_ptr<double> MEM;

    // Full Vectors
    thrust::device_ptr<double> sup, mid, sub, deltas, residual;

    // Vector tips
    thrust::device_ptr<double> low_dirichlet, high_dirichlet,
        top_factors, bottom_factors;

    int sz;
    int blksz;
    int blks;

    Diags(Vec v, int blocks) {
        int i = 0;
        sz = v.size();
        blks = blocks;
        blksz = sz / blks;            //   5 vecs      4 tips
        MEM = thrust::device_new<double>(v.size()*5 + 4*blksz);
        thrust::fill(MEM, MEM+sz*5 + 2, 0.);
        sup = MEM + (i++) * v.size();
        mid = MEM + (i++) * v.size();
        sub = MEM + (i++) * v.size();
        deltas = MEM + (i++) * v.size();
        residual = MEM + (i++) * v.size();
        high_dirichlet = MEM + i * v.size();
        low_dirichlet = MEM + i * v.size() + blksz;
        top_factors = MEM + i * v.size() + blksz*2;
        bottom_factors = MEM + i * v.size() + blksz*2;
        thrust::adjacent_difference(v.begin(), v.end(), deltas);
        strided_range<Vec::iterator> d(deltas, deltas+sz, blksz);
        thrust::fill(d.begin(), d.end(), NaN);
    }
};

template <typename T>
void printvec(const char *c, thrust::device_ptr<T> const &v, int size) {
    cout << c << ": ";
    cout << "DEVICE addr(" << v.get() << ") size(" << size << ")  [ ";
    std::ostream_iterator<T> out = std::ostream_iterator<T>(cout, "\t");
    std::copy(v, v + size, out);
    cout << "]" << endl;
}

int spot_first(Dptr &sup, Dptr &mid, Dptr &sub, Dptr &deltas,
        Dptr &high_dirichlet, Dptr &residual, int sz, int blksz) {
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(sup+1, mid+1, sub+1, deltas+1, deltas+2)),
            thrust::make_zip_iterator(thrust::make_tuple(sup+sz-1, mid+sz-1, sub+sz-1, deltas+sz-1, deltas+sz)),
            first_deriv()
            );
    strided_range<Vec::iterator> topsup(sup, sup+sz, blksz);
    strided_range<Vec::iterator> topmid(mid, mid+sz, blksz);
    strided_range<Vec::iterator> topsub(sub, sub+sz, blksz);
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(topsup.begin(), topmid.begin(), topsub.begin(), high_dirichlet)),
            thrust::make_zip_iterator(thrust::make_tuple(topsup.end(), topmid.end(), topsub.end(), high_dirichlet+1)),
            dirichlet_boundary(0)
            );
    strided_range<Vec::iterator> botsup(sup+blksz-1, sup+sz, blksz);
    strided_range<Vec::iterator> botmid(mid+blksz-1, mid+sz, blksz);
    strided_range<Vec::iterator> botsub(sub+blksz-1, sub+sz, blksz);
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(botsup.begin(),
                    botmid.begin(), botsub.begin(), residual+blksz-1)),
            thrust::make_zip_iterator(thrust::make_tuple(botsup.end(),
                    botmid.end(), botsub.end(), residual+sz)),
            von_neumann_boundary(1)
            );
    return 0;
}

int spot_second(Dptr &sup, Dptr &mid, Dptr &sub, Dptr &deltas,
        Dptr &high_dirichlet, Dptr &residual, int sz, int blksz) {
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(sup+1, mid+1, sub+1, deltas+1, deltas+2)),
            thrust::make_zip_iterator(thrust::make_tuple(sup+sz-1, mid+sz-1, sub+sz-1, deltas+sz-1, deltas+sz)),
            second_deriv()
            );
    strided_range<Vec::iterator> topsup(sup, sup+sz, blksz);
    strided_range<Vec::iterator> topmid(mid, mid+sz, blksz);
    strided_range<Vec::iterator> topsub(sub, sub+sz, blksz);
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(topsup.begin(), topmid.begin(), topsub.begin(), high_dirichlet)),
            thrust::make_zip_iterator(thrust::make_tuple(topsup.end(), topmid.end(), topsub.end(), high_dirichlet+1)),
            dirichlet_boundary(0)
            );
    strided_range<Vec::iterator> botsup(sup+blksz-1, sup+sz, blksz);
    strided_range<Vec::iterator> botmid(mid+blksz-1, mid+sz, blksz);
    strided_range<Vec::iterator> botsub(sub+blksz-1, sub+sz, blksz);
    strided_range<Vec::iterator> botdel(deltas+blksz-1, deltas+sz, blksz);
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(
                    botsup.begin(), botmid.begin(), botsub.begin(),
                    botdel.begin(), residual+blksz-1)),
            thrust::make_zip_iterator(thrust::make_tuple(
                    botsup.end(), botmid.end(), botsup.end(),
                    botdel.end(), residual+sz)),
            free_boundary_second_with_first_derivative_one()
            );
    return 0;
}

int var_first(Dptr &sup, Dptr &mid, Dptr &sub, Dptr &deltas, int sz, int blksz) {
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(sup+1, mid+1, sub+1, deltas+1, deltas+2)),
            thrust::make_zip_iterator(thrust::make_tuple(sup+sz-1, mid+sz-1, sub+sz-1, deltas+sz-1, deltas+sz)),
            first_deriv()
            );
    strided_range<Vec::iterator> topsup(sup, sup+sz, blksz);
    strided_range<Vec::iterator> topmid(mid, mid+sz, blksz);
    strided_range<Vec::iterator> topsub(sub, sub+sz, blksz);
    strided_range<Vec::iterator> topdel(deltas+1, deltas+sz, blksz);
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(topsup.begin(), topmid.begin(), topsub.begin(), topdel.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(topsup.end(), topmid.end(), topsub.end(), topdel.end())),
            free_boundary_first()
            );
    strided_range<Vec::iterator> botsup(sup+blksz-1, sup+sz, blksz);
    strided_range<Vec::iterator> botmid(mid+blksz-1, mid+sz, blksz);
    strided_range<Vec::iterator> botsub(sub+blksz-1, sub+sz, blksz);
    strided_range<Vec::iterator> botdel(deltas+blksz-1, deltas+sz, blksz);
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(botmid.begin(), botsub.begin(), botsup.begin(), botdel.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(botmid.end(), botsub.end(), botsup.end(), botdel.end())),
            free_boundary_first()
            );
    return 0;
}

int var_second(Dptr &sup, Dptr &mid, Dptr &sub, Dptr &deltas,
        Dptr &top_factors, Dptr &bottom_factors,
        int sz, int blksz) {
    int blks = sz / blksz;
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(sup+1, mid+1, sub+1, deltas+1, deltas+2)),
            thrust::make_zip_iterator(thrust::make_tuple(sup+sz-1, mid+sz-1, sub+sz-1, deltas+sz-1, deltas+sz)),
            second_deriv()
            );
    strided_range<Vec::iterator> topsup(sup, sup+sz, blksz);
    strided_range<Vec::iterator> topmid(mid, mid+sz, blksz);
    strided_range<Vec::iterator> topsub(sub, sub+sz, blksz);
    strided_range<Vec::iterator> topdel(deltas+1, deltas+sz, blksz);
    strided_range<Vec::iterator> topdel2(deltas+2, deltas+sz, blksz);
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(top_factors,
                    topsup.begin(), topmid.begin(), topsub.begin(),
                    topdel.begin(), topdel2.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(top_factors + blks,
                    topsup.end(), topmid.end(), topsub.end(),
                    topdel.end(), topdel2.end())),
            free_boundary_second()
            );
    strided_range<Vec::iterator> botsup(sup+blksz-1, sup+sz, blksz);
    strided_range<Vec::iterator> botmid(mid+blksz-1, mid+sz, blksz);
    strided_range<Vec::iterator> botsub(sub+blksz-1, sub+sz, blksz);
    strided_range<Vec::iterator> botdel(deltas+blksz-1, deltas+sz, blksz);
    strided_range<Vec::iterator> botdel2(deltas+blksz-2, deltas+sz, blksz);
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(bottom_factors,
                    botsub.begin(), botmid.begin(), botsup.begin(),
                    botdel.begin(), botdel2.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(bottom_factors + blks,
                    botsub.end(), botmid.end(), botsup.end(),
                    botdel.end(), botdel2.end())),
            free_boundary_second()
            );
    return 0;
}

int mixed(Diags &d) {

    return 0;
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


void plus_ident(Diags &d, int high, int low) {
    /* thrust::transform(d.mid+1, d.mid+d.sz, d.mid, _1 + 1); */
    high = d.blksz-1 - high;
    thrust::transform_if(
            d.mid,
            d.mid+d.sz,
            thrust::make_counting_iterator(0),
            d.mid,
            _1 + 1,
            periodic_from_to_mask(high, low, d.blksz));
}

int main(void) {
    //cudaError_t err;
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE Library initialization failed." << std::endl;
        return 1;
    }

    Vec vec_(5);
    thrust::sequence(vec_.begin(), vec_.end());
    tiled_range<Vec::iterator> v(vec_.begin(), vec_.end(), 3);
    Vec vec(v.begin(), v.end());

    Diags d = Diags(vec, 3);

    Vec tst(vec.size());

    thrust::counting_iterator<int> count_begin(0), count_end(tst.size());
    thrust::transform(v.begin(), v.end(), v.begin(), tst.begin(),
            thrust::multiplies<double>());

    switch (2) {
        case 1:
            spot_first(d.sup, d.mid, d.sub, d.deltas,
                    d.high_dirichlet, d.residual, d.sz, d.blksz);
            break;
        case 2:
            spot_second(d.sup, d.mid, d.sub, d.deltas,
                    d.high_dirichlet, d.residual, d.sz, d.blksz);
            break;
        case 3:
            var_first(d.sup, d.mid, d.sub, d.deltas, d.sz, d.blksz);
            break;
        case 4:
            var_second(d.sup, d.mid, d.sub, d.deltas,
                   d.top_factors, d.bottom_factors, d.sz, d.blksz);
            break;
    }

    /* plus_ident(d, -2, -2); */

    printvec("del", d.deltas, d.sz);
    printvec("sup", d.sup, d.sz);
    printvec("mid", d.mid, d.sz);
    printvec("sub", d.sub, d.sz);
    cout << "high_dirichlet: " << *d.high_dirichlet;
    printvec("tst", &tst[0], d.sz);

    /* thrust::transform(tst.begin(), tst.end(), d.residual, tst.begin(), _1 - _2); */
    status = cusparseDgtsvStridedBatch(handle, tst.size(),
            d.sub.get(), d.mid.get(), d.sup.get(),
            raw(tst),
            1, tst.size());
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE tridiag system solve failed." << std::endl;
        return 1;
    }

    printf("\nSolved.\n");
    printvec("sup", d.sup, d.sz);
    printvec("mid", d.mid, d.sz);
    printvec("sub", d.sub, d.sz);
    printvec("res", d.residual, d.sz);
    printvec("tst", &tst[0], d.sz);

    std::cout << "=======" << std::endl;
    return 0;
}
/*
 *
 *         # This expects only 2 dimensions
 *         self.check_mixed_derivative_parameters(mixed_derivs.keys())
 *         for d in mixed_derivs.keys():
 *             d0_size = len(self.grid.mesh[d[0]])
 *             d1_size = len(self.grid.mesh[d[1]])
 *
 *             # TODO: We'll need to do complicated transposing for this in the
 *             # general case
 *             Bs = BO.for_vector(self.grid.mesh[d[0]], "center", 1, 2, None, None, 0)
 *             Bm1 = BO.for_vector(self.grid.mesh[d[1]], "center", 1, 2, None, None, 1)
 *             Bb1 = Bm1.copy()
 *             Bp1 = Bm1.copy()
 *
 *             # TODO: Hardcoding in for centered differencing
 *             Bps = [Bp1 * 0, Bp1 * 0] + replicate(d0_size-2, Bp1)
 *             Bbs = [Bb1 * 0] + replicate(d0_size-2, Bb1) +  [Bb1 * 0]
 *             Bms = replicate(d0_size-2, Bm1) + [Bm1 * 0, Bm1 * 0]
 *
 *             offsets = Bs.D.offsets
 *             data = [Bps, Bbs, Bms]
 *             for row, o in enumerate(offsets):
 *                 if o >= 0:
 *                     for i in range(Bs.shape[0]-o):
 *                         a = (np.array(self.grid.mesh[d[0]][i]).repeat(d1_size),)
 *                         # vec = self.evalvectorfunc(coeffs[d], a, 1)
 *                         # data[row][i+o].vectorized_scale(vec)
 *                         data[row][i+o] *= Bs.D.data[row, i+o]
 *                 else:
 *                     for i in range(abs(o), Bs.shape[0]):
 *                         # a = (np.array(self.grid.mesh[d[0]][i]).repeat(d1_size),)
 *                         # vec = self.evalvectorfunc(coeffs[d], a, 1)
 *                         # data[row][i-abs(o)].vectorized_scale(vec)
 *                         data[row][i-abs(o)] *= Bs.D.data[row, i-abs(o)]
 *
 *             # We flatten here because it's faster
 *             # Check is set to False because we're only faking that the offsets.
 *             # The resulting operator will take the offsets from only the first
 *             # in the list.
 *             Bps[0].D.offsets += d1_size
 *             Bms[0].D.offsets -= d1_size
 *             BP = flatten_tensor_aligned(Bps, check=False)
 *             BB = flatten_tensor_aligned(Bbs, check=False)
 *             BM = flatten_tensor_aligned(Bms, check=False)
 *             templates[d] = BP + BM + BB
 *             templates[d].is_mixed_derivative = True
 */

