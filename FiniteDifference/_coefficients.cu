#include "_coefficients.cuh"
#include "VecArray.h"
#include "common.h"

#include "tiled_range.h"
#include "repeated_range.h"

#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_ptr.h>

#include <cstdio>

using thrust::make_zip_iterator;

typedef thrust::device_ptr<double> Dptr;
typedef thrust::detail::normal_iterator<thrust::device_ptr<double> > DptrIterator;

struct scale_functor_0 {
    template <typename Tuple>
    __host__ __device__
    /* f(t, r, *spot, *vars, *scaled_vec) */
    /* option.interest_rate.value * dim[0] */
    void operator()(Tuple tup) {
        using thrust::get;
        REAL_t const &t = get<0>(tup);
        REAL_t const &r = get<1>(tup);
        REAL_t const &spot = get<2>(tup);
        REAL_t const &var = get<3>(tup);
        REAL_t &val = get<4>(tup);
        val = r * spot;
    }
};

struct scale_functor_00 {
    template <typename Tuple>
    __host__ __device__
    /* f(t, r, *spot, *vars, *scaled_vec) */
    /* 0.5 * dim[1] * dim[0]**2 */
    void operator()(Tuple tup) {
        using thrust::get;
        REAL_t const &t = get<0>(tup);
        REAL_t const &r = get<1>(tup);
        REAL_t const &spot = get<2>(tup);
        REAL_t const &var = get<3>(tup);
        REAL_t &val = get<4>(tup);
        val = 0.5 * var * spot * spot;
    }
};

struct scale_functor_1 {
    template <typename Tuple>
    __host__ __device__
    /* f(t, r, *spot, *vars, reversion, mean_variance, *scaled_vec) */
    /* if dim[0] == 0: return 0 */
    /* ret = option.variance.reversion * (option.variance.mean - dim[1]) */
    /* ret[dim[0]==0] = 0 */
    void operator()(Tuple tup) {
        using thrust::get;
        REAL_t const &t = get<0>(tup);
        REAL_t const &r = get<1>(tup);
        REAL_t const &spot = get<2>(tup);
        REAL_t const &var = get<3>(tup);
        REAL_t const &reversion = get<4>(tup);
        REAL_t const &mean = get<5>(tup);
        REAL_t &val = get<6>(tup);
        if (spot == 0) {
            val = 0;
        } else {
            val = reversion * (mean - var);
        }
    }
};


struct scale_functor_11 {
    template <typename Tuple>
    __host__ __device__
    /* f(t, r, *spot, *vars, vol_of_var, *scaled_vec) */
    /* def gamma2_v(t, *dim): */
    /* if np.isscalar(dim[0]): */
    /* if dim[0] == 0: */
    /* return 0 */
    /* ret = 0.5 * option.variance.volatility**2 * dim[1] */
    /* ret[dim[0]==0] = 0 */
    void operator()(Tuple tup) {
        using thrust::get;
        REAL_t const &t = get<0>(tup);
        REAL_t const &r = get<1>(tup);
        REAL_t const &spot = get<2>(tup);
        REAL_t const &var = get<3>(tup);
        REAL_t const &vol_of_var = get<4>(tup);
        REAL_t &val = get<5>(tup);
        if (spot == 0) {
            val = 0;
        } else {
            val = 0.5 * vol_of_var * vol_of_var * var;
        }

    }
};

struct scale_functor_01 {
    template <typename Tuple>
    __host__ __device__
    /* f(t, r, *spot, *vars, vol_of_var, correlation, *scaled_vec) */
    /* option.correlation * option.variance.volatility * dim[0] * dim[1] */
    void operator()(Tuple tup) {
        using thrust::get;
        REAL_t const &t = get<0>(tup);
        REAL_t const &r = get<1>(tup);
        REAL_t const &spot = get<2>(tup);
        REAL_t const &var = get<3>(tup);
        REAL_t const &vol_of_var = get<4>(tup);
        REAL_t const &correlation = get<5>(tup);
        REAL_t &val = get<6>(tup);
        val = correlation * vol_of_var * spot * var;
    }
};


void scale_0(
          double t
        , double r
        , SizedArray<double> &spots
        , SizedArray<double> &vars
        , SizedArray<double> &vec
        ) {

    if (spots.size * vars.size != vec.size) {
        DIE("scaling_vec is not the right size. spots("<<spots.size <<")"
            << " vars("<<vars.size<<") "
            << " vec("<<vec.size<<") "
           );
    }

    tiled_range<DptrIterator> s(spots.data, spots.data + spots.size, vars.size);
    repeated_range<DptrIterator> v(vars.data, vars.data + vars.size, spots.size);
    thrust::for_each(
        make_zip_iterator(
            make_tuple(
                make_constant_iterator(t),
                make_constant_iterator(r),
                s.begin(),
                v.begin(),
                vec.data
                )
            ),
        make_zip_iterator(
            make_tuple(
                make_constant_iterator(t) + vec.size,
                make_constant_iterator(r) + vec.size,
                s.end(),
                v.end(),
                vec.data + vec.size
                )
            ),
        scale_functor_0()
        );
}

void scale_00(
          double t
        , double r
        , SizedArray<double> &spots
        , SizedArray<double> &vars
        , SizedArray<double> &vec
        ) {

    if (spots.size * vars.size != vec.size) {
        DIE("scaling_vec is not the right size. spots("<<spots.size <<")"
            << " vars("<<vars.size<<") "
            << " vec("<<vec.size<<") "
           );
    }

    tiled_range<DptrIterator> s(spots.data, spots.data + spots.size, vars.size);
    repeated_range<DptrIterator> v(vars.data, vars.data + vars.size, spots.size);
    thrust::for_each(
        make_zip_iterator(
            make_tuple(
                make_constant_iterator(t),
                make_constant_iterator(r),
                s.begin(),
                v.begin(),
                vec.data
                )
            ),
        make_zip_iterator(
            make_tuple(
                make_constant_iterator(t) + vec.size,
                make_constant_iterator(r) + vec.size,
                s.end(),
                v.end(),
                vec.data + vec.size
                )
            ),
        scale_functor_00()
        );
}


void scale_1(
          double t
        , double r
        , SizedArray<double> &spots
        , SizedArray<double> &vars
        , double reversion
        , double mean_variance
        , SizedArray<double> &vec
        ) {

    if (spots.size * vars.size != vec.size) {
        DIE("scaling_vec is not the right size. spots("<<spots.size <<")"
            << " vars("<<vars.size<<") "
            << " vec("<<vec.size<<") "
           );
    }

    repeated_range<DptrIterator> s(spots.data, spots.data + spots.size, vars.size);
    tiled_range<DptrIterator> v(vars.data, vars.data + vars.size, spots.size);
    thrust::for_each(
        make_zip_iterator(
            make_tuple(
                make_constant_iterator(t),
                make_constant_iterator(r),
                s.begin(),
                v.begin(),
                make_constant_iterator(reversion),
                make_constant_iterator(mean_variance),
                vec.data
                )
            ),
        make_zip_iterator(
            make_tuple(
                make_constant_iterator(t) + vec.size,
                make_constant_iterator(r) + vec.size,
                s.end(),
                v.end(),
                make_constant_iterator(reversion) + vec.size,
                make_constant_iterator(mean_variance) + vec.size,
                vec.data + vec.size
                )
            ),
        scale_functor_1()
        );
}


void scale_11(
          double t
        , double r
        , SizedArray<double> &spots
        , SizedArray<double> &vars
        , double vol_of_var
        , SizedArray<double> &vec
        ) {

    if (spots.size * vars.size != vec.size) {
        DIE("scaling_vec is not the right size. spots("<<spots.size <<")"
            << " vars("<<vars.size<<") "
            << " vec("<<vec.size<<") "
           );
    }

    repeated_range<DptrIterator> s(spots.data, spots.data + spots.size, vars.size);
    tiled_range<DptrIterator> v(vars.data, vars.data + vars.size, spots.size);
    thrust::for_each(
        make_zip_iterator(
            make_tuple(
                make_constant_iterator(t),
                make_constant_iterator(r),
                s.begin(),
                v.begin(),
                make_constant_iterator(vol_of_var),
                vec.data
                )
            ),
        make_zip_iterator(
            make_tuple(
                make_constant_iterator(t) + vec.size,
                make_constant_iterator(r) + vec.size,
                s.end(),
                v.end(),
                make_constant_iterator(vol_of_var) + vec.size,
                vec.data + vec.size
                )
            ),
        scale_functor_11()
        );
}


void scale_01(
          double t
        , double r
        , SizedArray<double> &spots
        , SizedArray<double> &vars
        , double vol_of_var
        , double correlation
        , SizedArray<double> &vec
        ) {

    if (spots.size * vars.size != vec.size) {
        DIE("scaling_vec is not the right size. spots("<<spots.size <<")"
            << " vars("<<vars.size<<") "
            << " vec("<<vec.size<<") "
           );
    }

    repeated_range<DptrIterator> s(spots.data, spots.data + spots.size, vars.size);
    tiled_range<DptrIterator> v(vars.data, vars.data + vars.size, spots.size);
    thrust::for_each(
        make_zip_iterator(
            make_tuple(
                make_constant_iterator(t),
                make_constant_iterator(r),
                s.begin(),
                v.begin(),
                make_constant_iterator(vol_of_var),
                make_constant_iterator(correlation),
                vec.data
                )
            ),
        make_zip_iterator(
            make_tuple(
                make_constant_iterator(t) + vec.size,
                make_constant_iterator(r) + vec.size,
                s.end(),
                v.end(),
                make_constant_iterator(vol_of_var) + vec.size,
                make_constant_iterator(correlation) + vec.size,
                vec.data + vec.size
                )
            ),
        scale_functor_01()
        );
}
