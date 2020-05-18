/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include "config.h"
#include "hmc.h"
#include "hmc_mcast.h"

HMC_EXPORT
hmc_status_t hmc_context_config_read(hmc_ctx_config_h *config_p)
{
    char *var;
    hmc_context_config_t *config = malloc(sizeof(*config));
    hmc_context_config_t default_config = {
        .devices               = NULL,
        .timeout               = 10000,
        .print_nack_stats      = false,
        .memtype_cache_enabled = 1,
        .sx_depth              = 128,
        .rx_depth              = 1024,
        .sx_sge                = 1,
        .rx_sge                = 2,
        .sx_inline             = 128,
        .post_recv_thresh      = 32,
        .scq_moderation        = 64,
        .wsize                 = 64,
        .cu_stage_thresh       = 4000,
        .max_eager             = 65536,
        .verbose               = 0,
    };

    if (!config) {
        return HMC_ERROR;
    }
    /* Default settings */
    *config = default_config;

    var = getenv("HMC_NET_DEVICES");
    if (var) {
        config->devices = var;
    }

    var = getenv("HMC_TIMEOUT");
    if (var) {
        config->timeout = atoi(var);
    }

    var = getenv("HMC_NACK_STATS");
    if (var) {
        config->print_nack_stats = atoi(var);
    }

    var = getenv("HMC_MEMTYPE_CACHE");
    if (var) {
        config->memtype_cache_enabled = atoi(var);
    }

    var = getenv("HMC_SX_DEPTH");
    if (var) {
        config->sx_depth = atoi(var);
    }

    var = getenv("HMC_SX_SGE");
    if (var) {
        config->sx_sge = atoi(var);
    }

    var = getenv("HMC_RX_DEPTH");
    if (var) {
        config->rx_depth = atoi(var);
    }

    var = getenv("HMC_RX_SGE");
    if (var) {
        config->rx_sge = atoi(var);
    }

    var = getenv("HMC_SX_INLINE");
    if (var) {
        config->sx_inline = atoi(var);
    }

    var = getenv("HMC_POST_RECV_THRESH");
    if (var) {
        config->post_recv_thresh = atoi(var);
    }

    var = getenv("HMC_SCQ_MODERATION");
    if (var) {
        config->scq_moderation = atoi(var);
    }

    var = getenv("HMC_WINDOW_SIZE");
    if (var) {
        config->wsize = atoi(var);
    }

    var = getenv("HMC_CUDA_STAGE_THRESHOLD");
    if (var) {
        config->cu_stage_thresh = atoi(var);
    }

    var = getenv("HMC_MAX_EAGER");
    if (var) {
        config->max_eager = atoi(var);
    }

    var = getenv("HMC_VERBOSE");
    if (var) {
        config->verbose = atoi(var);
    }

    *config_p = (void*)config;
    return HMC_SUCCESS;
}

HMC_EXPORT
void hmc_context_config_release(hmc_ctx_config_h *config)
{
    free(config);
}
