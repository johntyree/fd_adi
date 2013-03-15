#ifndef KERNELS_H
#define KERNELS_H

#include <GNUC_47_compat.h>

#include "common.h"

__global__
void d_transposeNoBankConflicts(REAL_t *odata, REAL_t *idata,
        int height, int width);

void transposeNoBankConflicts(REAL_t *odata, REAL_t *idata,
        int height, int width);

__device__
void _triDiagonalSystemSolve(
     size_t dim   //the dimension of the tridiagonal system
    ,int rank     //thread index,within the block
    ,REAL_t *l //lowerdiagonal, destroyed at exit
    ,REAL_t *d //diagonal, destroyed at exit
    ,REAL_t *u //upperdiagonal, destroyed at exit
    ,REAL_t *h //righthand side and solution at exit
    );

__global__
void triDiagonalSystemSolve(size_t dim, REAL_t *l, REAL_t *d, REAL_t *u,
                            REAL_t *h);

#endif /* end of include guard */
