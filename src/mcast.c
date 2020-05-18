/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include "config.h"
#include "hmc_mcast.h"
#include <rdma/rdma_verbs.h>

static inline int join_mcast(struct app_context *ctx, struct sockaddr_in6 *net_addr,
                             struct rdma_cm_event **event, int is_root) {
    int err;
    char buf[40];
    inet_ntop(AF_INET6, net_addr, buf, 40);
    HMC_VERBOSE(ctx, 10, "joining addr: %s", buf);
    pthread_mutex_lock(&ctx->lock);
    if (rdma_join_multicast(ctx->id, (struct sockaddr*)net_addr, NULL)) {
        return HMC_ERROR;
    }

    while (1) {
        if ((err = rdma_get_cm_event(ctx->channel, event)) < 0) {
            if (errno != EINTR) {
                HMC_ERR("rdma_get_cm_event failed, errno %d %s", errno, strerror(errno));
                pthread_mutex_unlock(&ctx->lock);
                return HMC_ERROR;
            } else {
                continue;
            }
        }
        if ((*event)->event != RDMA_CM_EVENT_MULTICAST_JOIN) {
            HMC_ERR("HMC Failed to join multicast, is_root %d. Unexpected event was received: event=%d, str=%s, status=%d",
                      is_root, (*event)->event, rdma_event_str((*event)->event), (*event)->status);
            pthread_mutex_unlock(&ctx->lock);
            return HMC_ERROR;
        } else {
            break;
        }
    }
    pthread_mutex_unlock(&ctx->lock);
    inet_ntop(AF_INET6, (*event)->param.ud.ah_attr.grh.dgid.raw, buf, 40);
    HMC_VERBOSE(ctx, 10, "is_root %d: joined dgid: %s, mlid 0x%x, sl %d", is_root, buf,
                (*event)->param.ud.ah_attr.dlid, (*event)->param.ud.ah_attr.sl);

    return HMC_SUCCESS;
}

static void hmc_runtime_bcast(void *buf, size_t len, int root, hmc_comm_t *comm) {
    void *tmp_buf = malloc(len*comm->commsize);
    comm->ctx->params.allgather(buf, tmp_buf, len, comm->params.comm_oob_context);
    if (comm->rank != root) {
        memcpy(buf, (void*)((ptrdiff_t)tmp_buf + root*len), len);
    }
    free(tmp_buf);
}
int setup_mcast_group(struct app_context *ctx, hmc_comm_t *comm, int ctx_id, uint16_t *mlid) {
    int status = 0;
    int ret = HMC_SUCCESS;
    struct rdma_cm_event tmp, *event = NULL;
    struct sockaddr_in6 net_addr = {0,};
    size_t mgid_s = sizeof(tmp.param.ud.ah_attr.grh.dgid);
    size_t mlid_s = sizeof(tmp.param.ud.ah_attr.dlid);
    void *data = alloca(sizeof(int) + mgid_s + mlid_s);
    if (comm->rank == 0) {
        net_addr.sin6_family = AF_INET6;
        net_addr.sin6_flowinfo = ctx_id;
        if (HMC_SUCCESS == join_mcast(ctx, &net_addr, &event, 1)) {
            memcpy((void*)((ptrdiff_t)data + sizeof(int)),&event->param.ud.ah_attr.grh.dgid, mgid_s);
            memcpy((void*)((ptrdiff_t)data + sizeof(int) + mgid_s), &event->param.ud.ah_attr.dlid, mlid_s);
        } else {
            status = -1;
        }
        memcpy(data, &status, sizeof(int));
    }
    hmc_runtime_bcast(data, sizeof(int) + mgid_s + mlid_s, 0, comm);
    status = *(int *) data;
    if (status != 0) {
        ret = HMC_ERROR;
        goto cleanup;
    }

    if (comm->rank != 0) {
        memcpy(&net_addr.sin6_addr, (void*)((ptrdiff_t)data + sizeof(int)), sizeof(struct in6_addr));
        net_addr.sin6_family = AF_INET6;
        if (HMC_SUCCESS != join_mcast(ctx, &net_addr, &event, 0)) {
            ret = HMC_ERROR;
            goto cleanup;
        }
    }
    *mlid = *((uint16_t*)((ptrdiff_t)data + sizeof(int) + mgid_s));
    comm->mcast_addr = net_addr;
    memcpy((void*)&comm->mgid, (void*)((ptrdiff_t)data + sizeof(int)), mgid_s);
cleanup:
    if (event) rdma_ack_cm_event(event);
    return ret;
}

int fini_mcast_group(struct app_context *ctx, hmc_comm_t *comm)
{

    char buf[40];
    inet_ntop(AF_INET6, &comm->mcast_addr, buf, 40);
    HMC_VERBOSE(ctx, 3, "Mcast leave: ctx %p, comm %p, dgid: %s", ctx, comm, buf);
    pthread_mutex_lock(&ctx->lock);
    if (rdma_leave_multicast(ctx->id, (struct sockaddr*)&comm->mcast_addr)) {
        HMC_ERR("ERROR: HMC rmda_leave_multicast failed");
        pthread_mutex_unlock(&ctx->lock);
        return HMC_ERROR;
    }
    pthread_mutex_unlock(&ctx->lock);
    return HMC_SUCCESS;
}


