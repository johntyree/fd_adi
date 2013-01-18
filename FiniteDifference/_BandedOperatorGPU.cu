#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <vector>
#include <iterator>
#include <cassert>

#include "_BandedOperatorGPU.cuh"

namespace CPU {

void print_array(double *a, size_t len) {
    auto out = std::ostream_iterator<double>(std::cout, " ");
    std::copy(a, a+len, out);
    std::cout << std::endl;
}

void vectorized_scale(
          SizedArray<double> vector
        , SizedArray<double> data
        , SizedArray<double> R
        , SizedArray<int> offsets
        , size_t operator_rows
        , size_t blocks
        , bool low_dirichlet
        , bool high_dirichlet
        ) {
    size_t vsize = vector.first;
    size_t noffsets = offsets.first;
    int o;
    size_t block_len = operator_rows / blocks;

    assert(operator_rows % vsize == 0);

    if (low_dirichlet) {
        for (size_t b = 0; b < blocks; ++b) {
            vector.second[b*block_len % vsize] = 1;
        }
    }

    if (high_dirichlet) {
        for (size_t b = 0; b < blocks; ++b) {
            vector.second[(b+1)*block_len - 1 % vsize] = 1;
        }
    }

    for (size_t row = 0; row < noffsets; ++row) {
        o = offsets.second[row];
        if (o >= 0) { // upper diags
            for (int i = 0; i < (int)operator_rows - o; ++i) {
                data.second[row * vsize + i+o] *= vector.second[i % vsize];
            }
        } else { // lower diags
            for (int i = -o; i < (int)operator_rows; ++i) {
                data.second[row * vsize + i+o] *= vector.second[i % vsize];
            }
        }
    }

    for (size_t i = 0; i < operator_rows; ++i) {
        R.second[i] *= vector.second[i % vsize];
    }
    return;
}

} // namespace CPU
