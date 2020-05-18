/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#ifndef HMC_CUDA
#define HMC_CUDA

#ifdef CUDA_ENABLED
#include "cuda_runtime.h"
#define CUDACHECK(cmd) do {                                         \
        cudaError_t e = cmd;                                        \
        if( e != cudaSuccess && e != cudaErrorCudartUnloading ) {   \
            HMC_ERR("Cuda failure %s:%d '%s'\n",                    \
                    __FILE__,__LINE__,cudaGetErrorString(e));       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
} while(0)

typedef struct hmc_cuda_module_t {
    int             enabled;
    void           *handle;
    cudaStream_t    stream;
    cudaError_t (*streamCreateWithFlags)(cudaStream_t* pStream, unsigned int flags);
    cudaError_t (*memcpyAsync)(void *dst, const void *src, size_t count,
                               enum cudaMemcpyKind kind, cudaStream_t stream);
    cudaError_t (*streamSynchronize)(cudaStream_t stream);
    cudaError_t (*hostAlloc)(void **ptr, size_t size, unsigned int flags);
    cudaError_t (*hostFree)(void *ptr);
    cudaError_t (*pointerGetAttributes)(struct cudaPointerAttributes* attributes, const void* ptr);
} hmc_cuda_module_t;

int hmc_cuda_init(hmc_cuda_module_t *cuda);
#endif

struct app_context;
int hmc_cuda_copy_h2d_async(void *dst, void *src, size_t len, struct app_context *ctx);
int hmc_cuda_copy_h2d(void *dst, void *src, size_t len, struct app_context *ctx);
void* hmc_cuda_host_alloc(size_t len, struct app_context *ctx);
void hmc_cuda_host_free(void *ptr, struct app_context *ctx);
int hmc_gpu_synchronize(struct app_context *ctx);
int hmc_mem_type(struct app_context *ctx, void *ptr);

#endif
