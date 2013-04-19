#include "_kernels.h"

/* Parameters to tweak for the transpose kernel. */
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__
void d_transposeNoBankConflicts(
        REAL_t *odata, REAL_t *idata, int height, int width) {
    __shared__ REAL_t tile[TILE_DIM][TILE_DIM+1];
    int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (index_in + i * width < width*height) {
            tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
        }
    }
    __syncthreads();
    for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (index_out + i * height < width*height
                && threadIdx.x + TILE_DIM * blockIdx.y < height) {
            odata[index_out+i*height] =
                tile[threadIdx.x][threadIdx.y+i];
        }
    }
}

void transposeNoBankConflicts(REAL_t *odata, REAL_t *idata,
        int height, int width) {
    dim3 grid(ceil((float)width/TILE_DIM), ceil((float)height/TILE_DIM));
    dim3 threads(TILE_DIM,BLOCK_ROWS);
    d_transposeNoBankConflicts<<<grid, threads>>>(odata, idata, height, width);
}


__device__
void _triDiagonalSystemSolve(
     size_t dim   //the dimension of the tridiagonal system
    ,REAL_t *_l //lowerdiagonal, destroyed at exit
    ,REAL_t *_d //diagonal, destroyed at exit
    ,REAL_t *_u //upperdiagonal, destroyed at exit
    ,REAL_t * const h //righthand side and solution at exit
    ) {

    size_t rank = threadIdx.x;
    REAL_t lTemp;
    REAL_t uTemp;
    REAL_t hTemp;

    /* This algorithm has data-dependent modifications to the diagonals
     * We need to build an explicit copy
     */
    __shared__ REAL_t l[256];
    __shared__ REAL_t d[256];
    __shared__ REAL_t u[256];

    /* Building the tridiagonal matrix */
    /* Note the lower diagonal starts at index 1
     * xxx_
     * xxxx
     * _xxx
     */
    l[rank] = _l[rank];
    d[rank] = _d[rank];
    u[rank] = _u[rank];

    __syncthreads();
    for (int span = 1; span < dim; span *= 2) {
         if (rank < dim) {
            if (rank >= span) {
                lTemp = d[rank-span] != 0 ? -l[rank]/d[rank-span] : 0;
            } else {
                lTemp = 0;
            }
            if(rank+span < dim) {
                uTemp = d[rank+span] != 0 ? -u[rank]/d[rank+span] : 0;
            } else {
                uTemp = 0;
            }
            hTemp = h[rank];
         }
        __syncthreads();
        if (rank < dim) {
            if (rank >= span) {
                d[rank] += lTemp * u[rank-span];
                hTemp += lTemp * h[rank-span];
                lTemp *= l[rank-span];
            }
            if (rank+span < dim) {
                d[rank] += uTemp * l[rank+span];
                hTemp += uTemp * h[rank+span];
                uTemp *= u[rank+span];
            }
        }
        __syncthreads();
        if (rank < dim) {
            l[rank] = lTemp;
            u[rank] = uTemp;
            h[rank] = hTemp;
        }
        __syncthreads();
    }
    if (rank < dim) h[rank] /= d[rank];
    __syncthreads();
}

__global__
void triDiagonalSystemSolve(size_t dim, REAL_t *l, REAL_t *d, REAL_t *u,
                            REAL_t *h){
    _triDiagonalSystemSolve(dim, l + blockIdx.x * gridDim.x, d + blockIdx.x * gridDim.x, u + blockIdx.x * gridDim.x, h + blockIdx.x * gridDim.x);
}
