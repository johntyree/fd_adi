#include "_kernels.h"

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
            // printf("Thread %i ((%i, %i) -> %i)\n", index_in, threadIdx.y+i, threadIdx.x, index_in+i*width);
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
    // printf("Grid: %i %i  Threads: %i %i", grid.x, grid.y, threads.x, threads.y);
    /* std::cout << "Grid: "<<grid.y<<", "<<grid.x */
        /* << " Threads: "<<threads.x<<", "<<threads.y<<"\n"; */
    d_transposeNoBankConflicts<<<grid, threads>>>(odata, idata, height, width);
}


