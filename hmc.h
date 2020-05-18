/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef HMC_H_
#define HMC_H_
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HMC_SUCCESS    =  0,
    HMC_INPROGRESS =  1,
    HMC_ERROR      = -1
} hmc_status_t;

typedef void* hmc_ctx_h;
typedef void* hmc_comm_h;
typedef void* hmc_ctx_config_h;

enum hmc_ctx_params_field {
    HMC_CTX_PARAMS_FIELD_MT_ENABLED        = (1 << 0),
    HMC_CTX_PARAMS_FIELD_WORLD_SIZE        = (1 << 1),
    HMC_CTX_PARAMS_FIELD_ALLGATHER         = (1 << 2),
    HMC_CTX_PARAMS_FIELD_OOB_CONTEXT       = (1 << 3),
    HMC_CTX_PARAMS_FIELD_RUNTIME_PROGRESS  = (1 << 4),
};

typedef struct hmc_ctx_params {
    uint64_t field_mask;
    int      mt_enabled;
    int      world_size;
    int      (*allgather)(void* sbuf, void* rbuf,
                          size_t local_len, void *oob_context);
    void     *oob_context;
    void     (*runtime_progress)(void);
} hmc_ctx_params_t;

enum hmc_comm_params_field {
    HMC_COMM_PARAMS_FIELD_COMM_RANK        = (1 << 0),
    HMC_COMM_PARAMS_FIELD_COMM_SIZE        = (1 << 1),
    HMC_COMM_PARAMS_FIELD_COMM_OOB_CONTEXT = (1 << 2),
    HMC_COMM_PARAMS_FIELD_COMM_RANK_TO_CTX = (1 << 3),
    HMC_COMM_PARAMS_FIELD_RANK_MAPPER_CTX  = (1 << 4),
};

typedef struct hmc_comm_params {
    uint64_t field_mask;
    int      comm_rank;
    int      comm_size;
    void     *comm_oob_context;
    int      (*comm_rank_to_ctx)(int comm_rank, void *rank_mapper_ctx);
    void     *rank_mapper_ctx;
} hmc_comm_params_t;

hmc_status_t hmc_context_config_read(hmc_ctx_config_h *config_p);
void hmc_context_config_release(hmc_ctx_config_h *config);
hmc_status_t hmc_init(hmc_ctx_params_t *params, hmc_ctx_config_h config, hmc_ctx_h *ctx);
hmc_status_t hmc_finalize(hmc_ctx_h ctx);

hmc_status_t hmc_comm_create(hmc_ctx_h ctx, hmc_comm_params_t *params, hmc_comm_h *comm);
hmc_status_t hmc_comm_destroy(hmc_comm_h comm);

enum hmc_bcast_args_field {
    HMC_BCAST_ARGS_FIELD_ADDRESS = (1 << 0),
    HMC_BCAST_ARGS_FIELD_SIZE    = (1 << 1),
    HMC_BCAST_ARGS_FIELD_ROOT    = (1 << 2),
    HMC_BCAST_ARGS_FIELD_COMM    = (1 << 3),
    HMC_BCAST_ARGS_FIELD_MR      = (1 << 4),
};

typedef struct hmc_bcast_args {
    uint64_t    field_mask;
    void        *address;
    size_t      size;
    int         root;
    hmc_comm_h  comm;
    void        *mr;
} hmc_bcast_args_t;

hmc_status_t hmc_bcast(hmc_bcast_args_t *args);
hmc_status_t hmc_ibcast(hmc_bcast_args_t *args, void **request);
hmc_status_t hmc_req_wait(void* req);
hmc_status_t hmc_req_test(void* req);
void         hmc_req_free(void *request);

hmc_status_t hmc_mem_register(hmc_ctx_h ctx, void *data, size_t data_size, void **mr);
hmc_status_t hmc_mem_deregister(hmc_ctx_h ctx, void *mr);

void hmc_progress(hmc_ctx_h ctx);

#ifdef __cplusplus
}
#endif

#endif
