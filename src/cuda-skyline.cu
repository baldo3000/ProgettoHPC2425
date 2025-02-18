/****************************************************************************
 *
 * cuda-skyline.cu - Cuda implementaiton of the skyline operator
 *
 * Copyright (C) 2024 Andrea Baldazzi
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 *
 * Questo programma calcola lo skyline di un insieme di punti in D
 * dimensioni letti da standard input. Per una descrizione completa
 * si veda la specifica del progetto sulla piattaforma "Virtuale".
 *
 * Per compilare:
 *
 *      nvcc -Wno-deprecated-gpu-targets cuda-skyline.cu -o cuda-skyline -lm
 *
 * Per eseguire il programma:
 *
 *      ./cuda-skyline < input > output
 *
 ****************************************************************************/

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

#define BLKDIM_1D 1024
#define BLKDIM_2D 32

typedef struct
{
    float *P; /* coordinates P[i][j] of point i               */
    int N;    /* Number of points (rows of matrix P)          */
    int D;    /* Number of dimensions (columns of matrix P)   */
} points_t;

/**
 * Read input from stdin. Input format is:
 *
 * d [other ignored stuff]
 * N
 * p0,0 p0,1 ... p0,d-1
 * p1,0 p1,1 ... p1,d-1
 * ...
 * pn-1,0 pn-1,1 ... pn-1,d-1
 *
 */
void read_input(points_t *points)
{
    char buf[1024];
    int N, D;
    float *P;

    if (1 != scanf("%d", &D))
    {
        fprintf(stderr, "FATAL: can not read the dimension\n");
        exit(EXIT_FAILURE);
    }
    assert(D >= 2);
    if (NULL == fgets(buf, sizeof(buf), stdin))
    { /* ignore rest of the line */
        fprintf(stderr, "FATAL: can not read the first line\n");
        exit(EXIT_FAILURE);
    }
    if (1 != scanf("%d", &N))
    {
        fprintf(stderr, "FATAL: can not read the number of points\n");
        exit(EXIT_FAILURE);
    }
    P = (float *)malloc(D * N * sizeof(*P));
    assert(P);
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < D; k++)
        {
            if (1 != scanf("%f", &(P[i * D + k])))
            {
                fprintf(stderr, "FATAL: failed to get coordinate %d of point %d\n", k, i);
                exit(EXIT_FAILURE);
            }
        }
    }
    points->P = P;
    points->N = N;
    points->D = D;
}

void free_points(points_t *points)
{
    free(points->P);
    points->P = NULL;
    points->N = points->D = -1;
}

/* Returns 1 if |p| dominates |q| */
__device__ int dominates(const float *p, const float *q, int D)
{
    /* The following loops could be merged, but the keep them separated
       for the sake of readability */
    for (int k = 0; k < D; k++)
    {
        if (p[k] < q[k])
        {
            return 0;
        }
    }
    for (int k = 0; k < D; k++)
    {
        if (p[k] > q[k])
        {
            return 1;
        }
    }
    return 0;
}

/**
 * CUDA kernel to compute the skyline of points in a 1D space.
 *
 * @param d_P Pointer to the array of points in device memory.
 * @param d_s Pointer to the array in device memory where the skyline results will be stored.
 * @param d_r Pointer to the variable in device memory where number of found points will be stored.
 * @param N The number of points in the array.
 * @param D The number of dimensions for each point.
 */
__global__ void skyline_kernel_1d(float *d_P, int *d_s, int *d_r, int N, int D)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        for (int j = 0; j < N; j++)
        {
            if (dominates(&(d_P[j * D]), &(d_P[idx * D]), D))
            {
                d_s[idx] = 0;
                j = N; // Force loop exit
                atomicSub(d_r, 1);
            }
        }
    }
}

/**
 * CUDA kernel to initialize the array.
 *
 * This kernel initializes the elements of an array `d_s` of size `N` with value 1.
 *
 * @param d_s Pointer to the device array to be initialized.
 * @param N The number of elements in the array.
 */
__global__ void init_kernel(int *d_s, int N)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        d_s[idx] = 1;
    }
}

/**
 * CUDA kernel to compute the skyline of points in a 1D space.
 *
 * @param d_P Pointer to the array of points in device memory.
 * @param d_s Pointer to the array in device memory where the skyline results will be stored.
 * @param N The number of points in the array.
 * @param D The number of dimensions for each point.
 */
__global__ void skyline_kernel(float *d_P, int *d_s, int N, int D)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx < N && idy < N)
    {
        if (dominates(&(d_P[idy * D]), &(d_P[idx * D]), D))
        {
            atomicExch(&d_s[idx], 0);
        }
    }
}

/**
 * CUDA kernel to compute the sum of the elements of an array.
 *
 * @param d_s Pointer to the array to sum.
 * @param d_r Pointer to the variable where the result will be stored.
 * @param N The number of elements in the array.
 */
__global__ void sum_kernel(int *d_s, int *d_r, int N)
{
    __shared__ int temp[BLKDIM_1D];
    const int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    const int lindex = threadIdx.x;
    int bsize = blockDim.x / 2;
    temp[lindex] = (gindex < N) ? d_s[gindex] : 0;
    __syncthreads();
    while (bsize > 0)
    {
        if (lindex < bsize)
        {
            temp[lindex] += temp[lindex + bsize];
        }
        bsize /= 2;
        __syncthreads();
    }
    if (lindex == 0)
    {
        atomicAdd(d_r, temp[lindex]);
    }
}

/**
 * Compute the skyline of `points`. At the end, `s[i] == 1` iff point
 * `i` belongs to the skyline. The function returns the number `r` of
 * points that belongs to the skyline. The caller is responsible for
 * allocating the array `s` of length at least `points->N`.
 */
int skyline(const points_t *points, int *s)
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
#if 0 // 1 to test 1d kernel
    int r = N;
#else
    int r = 0;
#endif

    dim3 block_1d(BLKDIM_1D);
    dim3 grid_1d((N + BLKDIM_1D - 1) / BLKDIM_1D);
    dim3 block_2d(BLKDIM_2D, BLKDIM_2D);
    dim3 grid_2d((N + BLKDIM_2D - 1) / BLKDIM_2D, (N + BLKDIM_2D - 1) / BLKDIM_2D);

    int *d_s, *d_r;
    float *d_P;
    cudaSafeCall(cudaMalloc((void **)&d_s, N * sizeof(s[0])));
    cudaSafeCall(cudaMalloc((void **)&d_r, sizeof(int)));
    cudaSafeCall(cudaMalloc((void **)&d_P, N * D * sizeof(P[0])));
    cudaSafeCall(cudaMemcpy(d_r, &r, sizeof(int), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_P, P, N * D * sizeof(P[0]), cudaMemcpyHostToDevice));
#if 0 // 1 to test 1d kernel
    init_kernel<<<grid_1d, block_1d>>>(d_s, N);
    cudaCheckError();
    skyline_kernel_1d<<<grid_1d, block_1d>>>(d_P, d_s, d_r, N, D);
    cudaCheckError();
#else
    init_kernel<<<grid_1d, block_1d>>>(d_s, N);
    cudaCheckError();
    skyline_kernel<<<grid_2d, block_2d>>>(d_P, d_s, N, D);
    cudaCheckError();
    sum_kernel<<<grid_1d, block_1d>>>(d_s, d_r, N);
    cudaCheckError();
#endif
    /* Copio indietro i valori */
    cudaSafeCall(cudaMemcpy(s, d_s, N * sizeof(s[0]), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(&r, d_r, sizeof(int), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaFree(d_s));
    cudaSafeCall(cudaFree(d_r));
    cudaSafeCall(cudaFree(d_P));
    return r;
}

/**
 * Print the coordinates of points belonging to the skyline `s` to
 * standard ouptut. `s[i] == 1` iff point `i` belongs to the skyline.
 * The output format is the same as the input format, so that this
 * program can process its own output.
 */
void print_skyline(const points_t *points, const int *s, int r)
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;

    printf("%d\n", D);
    printf("%d\n", r);
    for (int i = 0; i < N; i++)
    {
        if (s[i])
        {
            for (int k = 0; k < D; k++)
            {
                printf("%f ", P[i * D + k]);
            }
            printf("\n");
        }
    }
}

int main(int argc, char *argv[])
{
    points_t points;

    if (argc != 1)
    {
        fprintf(stderr, "Usage: %s < input_file > output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    read_input(&points);
    int *s = (int *)malloc(points.N * sizeof(*s));
    assert(s);
    const double tstart = hpc_gettime();
    const int r = skyline(&points, s);
    const double elapsed = hpc_gettime() - tstart;
    print_skyline(&points, s, r);

    fprintf(stderr, "\n\t%d points\n", points.N);
    fprintf(stderr, "\t%d dimensions\n", points.D);
    fprintf(stderr, "\t%d points in skyline\n\n", r);
    fprintf(stderr, "Execution time (s) %f\n", elapsed);

    free_points(&points);
    free(s);
    return EXIT_SUCCESS;
}
