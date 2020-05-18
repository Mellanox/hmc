/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include "hmc_mcast.h"
#include "dlfcn.h"

#ifdef CUDA_ENABLED
int hmc_cuda_init(hmc_cuda_module_t *cuda) {
    cuda->stream = 0;
    cuda->handle = dlopen("libcudart.so", RTLD_LAZY);
    if (!cuda->handle) {
        goto err;
    }
    cuda->streamCreateWithFlags = dlsym(cuda->handle, "cudaStreamCreateWithFlags");
    if (dlerror()) {
        goto clean_handle;
    }
    cuda->memcpyAsync = dlsym(cuda->handle, "cudaMemcpyAsync");
    if (dlerror()) {
        goto clean_handle;
    }
    cuda->streamSynchronize = dlsym(cuda->handle, "cudaStreamSynchronize");
    if (dlerror()) {
        goto clean_handle;
    }
    cuda->hostAlloc = dlsym(cuda->handle, "cudaHostAlloc");
    if (dlerror()) {
        goto clean_handle;
    }
    cuda->hostFree = dlsym(cuda->handle, "cudaFreeHost");
    if (dlerror()) {
        goto clean_handle;
    }
    cuda->pointerGetAttributes = dlsym(cuda->handle, "cudaPointerGetAttributes");
    if (dlerror()) {
        goto clean_handle;
    }
    cuda->enabled = 1;
    return HMC_SUCCESS;
clean_handle:
    dlclose(cuda->handle);
err:
    cuda->enabled = 0;
    return HMC_ERROR;
}


#define HMC_CUDA_CHECK_STREAM(_cuda) do {                               \
        if (!cuda->enabled) {                                           \
            return HMC_ERROR;                                           \
        }                                                               \
        if (!cuda->stream) {                                            \
            CUDACHECK(_cuda->streamCreateWithFlags(&cuda->stream,       \
                                                   cudaStreamNonBlocking)); \
        }                                                               \
    } while(0)

int hmc_cuda_copy_h2d_async(void *dst, void *src, size_t len, struct app_context *ctx) {
    hmc_cuda_module_t *cuda = &ctx->cuda;
    HMC_CUDA_CHECK_STREAM(cuda);
    CUDACHECK(cuda->memcpyAsync(dst, src, len, cudaMemcpyHostToDevice, cuda->stream));
    return HMC_SUCCESS;
}

int hmc_cuda_copy_h2d(void *dst, void *src, size_t len, struct app_context *ctx) {
    hmc_cuda_copy_h2d_async(dst, src, len, ctx);
    CUDACHECK(ctx->cuda.streamSynchronize(ctx->cuda.stream));
    return HMC_SUCCESS;
}

void* hmc_cuda_host_alloc(size_t len, struct app_context *ctx) {
    void *ptr = NULL;
    hmc_cuda_module_t *cuda = &ctx->cuda;
    if (!cuda->enabled) {
        return NULL;
    }
    CUDACHECK(cuda->hostAlloc(&ptr, len, cudaHostAllocDefault));
    return ptr;
}

void hmc_cuda_host_free(void *ptr, struct app_context *ctx) {
    hmc_cuda_module_t *cuda = &ctx->cuda;
    if (!cuda->enabled) {
        return;
    }
    cuda->hostFree(ptr);
}

int hmc_gpu_synchronize(struct app_context *ctx) {
    hmc_cuda_module_t *cuda = &ctx->cuda;
    HMC_CUDA_CHECK_STREAM(cuda);
    CUDACHECK(cuda->streamSynchronize(cuda->stream));
    return HMC_SUCCESS;
}

#ifdef HAVE_UCM_MEM_TYPE_T
typedef ucm_mem_type_t hmc_ucx_mem_type_t;
#define HMC_UCX_MEM_TYPE_CUDA UCM_MEM_TYPE_CUDA
#else
typedef ucs_memory_type_t hmc_ucx_mem_type_t;
#define HMC_UCX_MEM_TYPE_CUDA UCS_MEMORY_TYPE_CUDA
#endif

int hmc_mem_type(struct app_context *ctx, void *ptr) {
    hmc_cuda_module_t *cuda = &ctx->cuda;
    int memtype = HMC_MEM_TYPE_HOST;
    if (!cuda->enabled) {
        return HMC_MEM_TYPE_HOST;
    }
    if (ctx->config.memtype_cache_enabled) {
        hmc_ucx_mem_type_t mem_type;
        if (ucs_memtype_cache_lookup(ctx->memtype_cache, ptr, 1, &mem_type) == UCS_OK) {
            if (mem_type == HMC_UCX_MEM_TYPE_CUDA) {
                memtype = HMC_MEM_TYPE_GPU;
            } else {
                memtype = HMC_MEM_TYPE_HOST;
            }
        }
    } else {
        struct cudaPointerAttributes attr;
        cudaError_t err = cuda->pointerGetAttributes(&attr, ptr);
        if (err != cudaSuccess) {
            cudaGetLastError();
            memtype = HMC_MEM_TYPE_HOST;
        }
#if CUDART_VERSION >= 10000
        if (attr.type == cudaMemoryTypeDevice) {
#else
        if (attr.memoryType == cudaMemoryTypeDevice) {
#endif
            memtype = HMC_MEM_TYPE_GPU;
        } else {
            memtype = HMC_MEM_TYPE_HOST;
        }
    }
    return memtype;
}

#else
int hmc_cuda_copy_h2d_async(void *dst, void *src, size_t len, struct app_context *ctx) {
    return HMC_ERROR;
}

int hmc_cuda_copy_h2d(void *dst, void *src, size_t len, struct app_context *ctx) {
    return HMC_ERROR;
}

void* hmc_cuda_host_alloc(size_t len, struct app_context *ctx) {
    return NULL;
}

void hmc_cuda_host_free(void *ptr, struct app_context *ctx) {}

int hmc_gpu_synchronize(struct app_context *ctx) {
    return HMC_ERROR;
}

int hmc_mem_type(struct app_context *ctx, void *ptr) {
    return HMC_MEM_TYPE_HOST;
}
#endif

