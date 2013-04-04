#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>

#include <thrust/device_free.h>
#include <thrust/device_reference.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

#include "common.h"
#include "_kernels.h"
#include "VecArray.h"

using thrust::device_malloc;
using thrust::device_free;
using thrust::make_constant_iterator;
using thrust::placeholders::_1;
using thrust::placeholders::_2;


namespace impl {

    /* Vector Vector double */
    void pluseq(
        thrust::device_ptr<double> &us,
        thrust::device_ptr<double> &them,
        Py_ssize_t size) {
        thrust::transform(
                us, us+size,
                them,
                us,
                _1 + _2);
    }


    void minuseq(
        thrust::device_ptr<double> &us,
        thrust::device_ptr<double> &them,
        Py_ssize_t size) {
        thrust::transform(
                us, us+size,
                them,
                us,
                _1 - _2);
    }


    void minuseq_over2(
        thrust::device_ptr<double> &us,
        thrust::device_ptr<double> &them,
        Py_ssize_t size) {
        thrust::transform(
                us, us+size,
                them,
                us,
                (_1 - _2)*0.5);
    }


    void timeseq(
        thrust::device_ptr<double> &us,
        thrust::device_ptr<double> &them,
        Py_ssize_t size) {
        thrust::transform(
                us, us+size,
                them,
                us,
                _1 * _2);
    }



    /* Vector Vector int */

    void pluseq(
        thrust::device_ptr<int> &us,
        thrust::device_ptr<int> &them,
        Py_ssize_t size) {
        thrust::transform(
                us, us+size,
                them,
                us,
                _1 + _2);
    }


    void minuseq(
        thrust::device_ptr<int> &us,
        thrust::device_ptr<int> &them,
        Py_ssize_t size) {
        thrust::transform(
                us, us+size,
                them,
                us,
                _1 - _2);
    }


    void timeseq(
        thrust::device_ptr<int> &us,
        thrust::device_ptr<int> &them,
        Py_ssize_t size) {
        thrust::transform(
                us, us+size,
                them,
                us,
                _1 * _2);
    }



    /* Vector Scalar double */

    void pluseq(
        thrust::device_ptr<double> &data,
        Py_ssize_t size,
        thrust::device_reference<double> x) {
        thrust::transform(
                data, data+size,
                thrust::make_constant_iterator(x),
                data,
                _1 + _2);
    }


    void minuseq(
        thrust::device_ptr<double> &data,
        Py_ssize_t size,
        thrust::device_reference<double> x) {
        thrust::transform(
                data, data+size,
                thrust::make_constant_iterator(x),
                data,
                thrust::minus<double>());
    }


    void timeseq(
        thrust::device_ptr<double> &data,
        Py_ssize_t size,
        thrust::device_reference<double> x) {
        thrust::transform(
                data, data+size,
                thrust::make_constant_iterator(x),
                data,
                _1 * _2);
    }



    /* Vector Scalar int */

    void pluseq(
        thrust::device_ptr<int> &data,
        Py_ssize_t size,
        thrust::device_reference<int> x) {
        thrust::transform(
                data, data+size,
                thrust::make_constant_iterator(x),
                data,
                _1 + _2);
    }


    void minuseq(
        thrust::device_ptr<int> &data,
        Py_ssize_t size,
        thrust::device_reference<int> x) {
        thrust::transform(
                data, data+size,
                thrust::make_constant_iterator(x),
                data,
                _1 - _2);
    }


    void timeseq(
        thrust::device_ptr<int> &data,
        Py_ssize_t size,
        thrust::device_reference<int> x) {
        thrust::transform(
                data, data+size,
                thrust::make_constant_iterator(x),
                data,
                _1 * _2);
    }

} // namespace impl
