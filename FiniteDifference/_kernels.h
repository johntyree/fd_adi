#ifndef KERNELS_H
#define KERNELS_H

#include <GNUC_47_compat.h>

#include "common.h"

__global__
void d_transposeNoBankConflicts(REAL_t *odata, REAL_t *idata,
        int height, int width);

void transposeNoBankConflicts(REAL_t *odata, REAL_t *idata,
        int height, int width);

#endif /* end of include guard */
