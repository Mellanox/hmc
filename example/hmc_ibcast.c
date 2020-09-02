#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <hmc.h>

#define MAX_TEST_SIZE 4194304
#define MIN_TEST_SIZE 4
#define ITERS         8
#define NSIZES        128

#ifdef CUDA_ENABLED
#include "cuda_runtime.h"
#define CUDACHECK(cmd) do {                                         \
        cudaError_t e = cmd;                                        \
        if( e != cudaSuccess && e != cudaErrorCudartUnloading ) {   \
            fprintf(stderr, "Cuda failure %s:%d '%s'\n",            \
                    __FILE__,__LINE__,cudaGetErrorString(e));       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    }while(0)
#endif

int hmc_allgather(void *sbuf, void *rbuf, size_t local_len, void *runtime_communicator) {
    MPI_Comm comm = (MPI_Comm)runtime_communicator;
    MPI_Allgather(sbuf, (int)local_len, MPI_BYTE, rbuf, (int)local_len, MPI_BYTE, comm);
    return 0;
}

static int hmc_test_rank_mapper(int rank, void *mapper_ctx) {
    MPI_Comm comm = (MPI_Comm) mapper_ctx;
    MPI_Group group, world_group;
    int wrank;
    MPI_Comm_group(comm, &group);
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_translate_ranks(group, 1, &rank, world_group, &wrank);
    MPI_Group_free(&group);
    MPI_Group_free(&world_group);
    return wrank;
}

int main (int argc, char **argv) {
    int rank, size;
    hmc_ctx_h hmc_context;
    hmc_comm_h hmc_comm;
    hmc_ctx_config_h ctx_config;
    hmc_ctx_params_t ctx_params;
    hmc_comm_params_t comm_params;
    hmc_bcast_args_t args;
    int status = 0, global_status, ii, ss, root, completed;
    void *buf, *cpu_buf, *check_buf, *cuda_buf, *req;
    size_t test_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ctx_params.field_mask = HMC_CTX_PARAMS_FIELD_WORLD_SIZE ||
                            HMC_CTX_PARAMS_FIELD_ALLGATHER  ||
                            HMC_CTX_PARAMS_FIELD_OOB_CONTEXT;
    ctx_params.allgather   = hmc_allgather;
    ctx_params.oob_context = (void*)MPI_COMM_WORLD;
    ctx_params.world_size  = size;
    hmc_context_config_read(&ctx_config);
    if (HMC_SUCCESS != hmc_init(&ctx_params, ctx_config, &hmc_context)) {
        fprintf(stderr, "Failed to init HMC ctx\n");
        goto error;
    }
    hmc_context_config_release(ctx_config);
    comm_params.field_mask = HMC_COMM_PARAMS_FIELD_COMM_SIZE        ||
                             HMC_COMM_PARAMS_FIELD_COMM_RANK        ||
                             HMC_COMM_PARAMS_FIELD_COMM_RANK_TO_CTX ||
                             HMC_COMM_PARAMS_FIELD_RANK_MAPPER_CTX  ||
                             HMC_COMM_PARAMS_FIELD_COMM_OOB_CONTEXT;
    comm_params.comm_oob_context = (void*)MPI_COMM_WORLD;
    comm_params.comm_size        = size;
    comm_params.comm_rank        = rank;
    comm_params.rank_mapper_ctx  = (void*)MPI_COMM_WORLD;
    comm_params.comm_rank_to_ctx = hmc_test_rank_mapper;

    if (HMC_SUCCESS != hmc_comm_create(hmc_context, &comm_params, &hmc_comm)) {
        fprintf(stderr, "Failed to init HMC comm\n");
        goto error;
    }

    args.field_mask = HMC_BCAST_ARGS_FIELD_ADDRESS ||
                      HMC_BCAST_ARGS_FIELD_SIZE ||
                      HMC_BCAST_ARGS_FIELD_ROOT ||
                      HMC_BCAST_ARGS_FIELD_COMM;
    for (ss = 0; ss < NSIZES; ss++) {
        test_size = (size_t)(MIN_TEST_SIZE + rand() % (MAX_TEST_SIZE - MIN_TEST_SIZE));
        root = rand() % size;
        if (rank == 0) {
            printf("size %10zd, root %3d ...\n", test_size ,root);
        }
        for (ii=0; ii < ITERS; ii++) {
            cpu_buf = malloc(test_size);
            check_buf = malloc(test_size);
            memset(check_buf, 0x123, test_size);
            if (root == rank) {
                memset(cpu_buf, 0x123, test_size);
            } else {
                memset(cpu_buf, 0, test_size);
            }

            buf = cpu_buf;
#if CUDA_ENABLED
            cuda_buf = NULL;
            CUDACHECK(cudaSetDevice(0));
            CUDACHECK(cudaMalloc(&cuda_buf, test_size));
            CUDACHECK(cudaMemcpy(cuda_buf, cpu_buf, test_size, cudaMemcpyHostToDevice));
            CUDACHECK(cudaDeviceSynchronize());
            buf = cuda_buf;
#endif
            completed = 0;
            args.address = buf;
            args.size    = test_size;
            args.root    = root;
            args.comm    = hmc_comm;
            hmc_ibcast(&args, &req);
            while (!completed) {
                if (HMC_SUCCESS == hmc_req_test(req)) {
                    completed = 1;
                }
            }
            hmc_req_free(req);
            if (root != rank) {
#if CUDA_ENABLED
                CUDACHECK(cudaMemcpy(cpu_buf, cuda_buf, test_size, cudaMemcpyDeviceToHost));
                CUDACHECK(cudaDeviceSynchronize());
#endif
                if (0 != memcmp(cpu_buf, check_buf, test_size)) {
                    status = 1;
                }
            }
#if CUDA_ENABLED
            if (cuda_buf) CUDACHECK(cudaFree(cuda_buf));
#endif
            free(cpu_buf);
            free(check_buf);
            MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (global_status != 0) {
                goto testend;
            }
        }
    }

testend:
    hmc_comm_destroy(hmc_comm);
    hmc_finalize(hmc_context);
    if (0 == rank) {
        printf("TEST: %s\n", global_status == 0 ? "SUCCESS" : "FAIL" );
    }
error:
    MPI_Finalize();
    return 0;
}
