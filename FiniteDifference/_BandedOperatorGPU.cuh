#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/version.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

template <typename T> using SizedArray = std::pair<size_t, T*>;

typedef Vector thrust::host_vector

template <typename T>
class NDArray {
    public:
        Vector<T> data;
}

namespace CPU {

void print_array(double *a, size_t len);

void vectorized_scale(
          SizedArray<double> vector
        , SizedArray<double> data
        , SizedArray<double> R
        , SizedArray<int> offsets
        , size_t operator_rows
        , size_t blocks
        , bool low_dirichlet
        , bool high_dirichlet
        );



class BandedOperator {
    public:
        Vector


}
