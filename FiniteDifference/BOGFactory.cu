#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <algorithm>
#include <cstdlib>
#include <cstdio>

#include <thrust/host_vector.h>
/* #include <thrust/device_vector.h> */
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#include <cusparse_v2.h>
#include <cuda_runtime.h>

/* #include "funcs.cuh" */

/* typedef thrust::device_vector<double> Vec; */
typedef thrust::host_vector<double> Vec;

int sortvec(void);

// The right way...
/* template<template<typename ...> class V, typename R> */
/* R *raw(V<R> &v) { */
    /* return thrust::raw_pointer_cast(v.data()); */
/* } */

double* raw(Vec &v) {
   return thrust::raw_pointer_cast(v.data());
}

template <typename T>
void printvec(T v) { print_range("", v.begin(), v.end()); }
template <typename T>
void printvec(const std::string &name, T v) { print_range(name, v.begin(), v.end()); }

struct make_diags {
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        using thrust::get;
        get<0>(t) = get<3>(t)                / (get<4>(t) * (get<3>(t) + get<4>(t)));
        get<1>(t) = 1 + (-get<3>(t) + get<4>(t)) / (get<3>(t) * get<4>(t));
        get<2>(t) = -get<4>(t)               / (get<3>(t) * (get<3>(t) + get<4>(t)));
    }
};

struct Diags {
    Vec sup, mid, sub, deltas;
    Diags(Vec v) {
        sup.resize(v.size());
        mid.resize(v.size());
        sub.resize(v.size());
        deltas.resize(v.size());

        thrust::adjacent_difference(v.begin(), v.end(), deltas.begin());
        /* deltas[0] = 0; */
        /* thrust::sequence(deltas.begin(), deltas.end(), 1); */
        /* deltas[0] = std::numeric_limits<double>::quiet_NaN(); */

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(sup.begin(), mid.begin()+1, sub.begin()+2, deltas.begin()+1, deltas.begin()+2)),
            thrust::make_zip_iterator(thrust::make_tuple(sup.end()-2, mid.end()-1, sub.end(), deltas.end()-1, deltas.end())),
            make_diags()
        );
        mid[0] = mid[mid.size()-1] = 1;
        sup.back() = 0;
        sub.front() = 0;
    }

};

int main(void) {
    //cudaError_t err;
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE Library initialization failed." << std::endl;
        return 1;
    }

    Vec vec(5);
    thrust::sequence(vec.begin(), vec.end());

    Diags d = Diags(vec);

    Vec res(vec.size());

    thrust::counting_iterator<int> count_begin(0), count_end(res.size());
    thrust::transform(count_begin, count_end, count_begin, res.begin(),
            thrust::multiplies<double>());


    printvec("del", d.deltas);
    printvec("sup", d.sup);
    printvec("mid", d.mid);
    printvec("sub", d.sub);
    printvec("res", res);

    status = cusparseDgtsvStridedBatch(handle, res.size(),
            raw(d.sub), raw(d.mid), raw(d.sup),
            raw(res),
            1, res.size());
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CUSPARSE tridiag system solve failed." << std::endl;
        return 1;
    }

    printf("\nSolved.\n");
    printvec("sup", d.sup);
    printvec("mid", d.mid);
    printvec("sub", d.sub);
    printvec("res", res);

    std::cout << "=======" << std::endl;
    return 0;
}
