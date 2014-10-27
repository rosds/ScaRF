#ifndef CUDA_UTILS_HH
#define CUDA_UTILS_HH

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define HANDLE_ERROR(err) (cudaHandleError(err, __FILE__, __LINE__))
inline void cudaHandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

extern void cudaSumFeatureVector(float *d_data, size_t ne, size_t nf, size_t pitch);

#endif
