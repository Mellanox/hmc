/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#ifndef MCAST_H_
#define MCAST_H_
#include "reliable.h"
#include <ucs/datastruct/list.h>
#include "config.h"
static inline void process_packet(hmc_comm_t *comm,
                                  hmc_coll_req_t *req, struct pp_packet* pp)
{
    assert(pp->psn >= req->start_psn &&
           pp->psn < req->start_psn + req->num_packets*req->num_roots);

    assert(pp->length == PSN_TO_RECV_LEN(pp->psn, req, comm));
    assert(pp->context == 0);

    if (pp->length >0 ) {
        void *dest;
        if (req->num_roots > 1) {
            dest = PSN_TO_RECV_OFFSET_MULTIROOT(pp->psn, req, comm);
        } else {
            dest = req->ptr + PSN_TO_RECV_OFFSET(pp->psn, req, comm);
        }
        if (HMC_MEM_TYPE_GPU == req->buf_mem_type) {
            if (req->use_cuda_staging) {
                ptrdiff_t offset = (pp->psn - MAX(comm->last_acked, req->start_psn))*comm->max_per_packet;
                memcpy((void*)((ptrdiff_t)comm->cu_stage_buf + offset), (void*) pp->buf, pp->length);
                req->already_staged += pp->length;
            } else {
                hmc_cuda_copy_h2d_async(dest, (void*) pp->buf, pp->length, comm->ctx);
            }
        } else {
            memcpy(dest, (void*) pp->buf, pp->length);
        }
    }
    comm->r_window[pp->psn & (comm->wsize-1)] = pp;
    if (!comm->ctx->params.mt_enabled) {
        check_nack_requests(comm, pp->psn);
    }
    req->to_recv--;
    comm->psn++;
}

static inline int mcast_poll_send(hmc_comm_t *comm)
{
    struct ibv_wc wc;
    int num_comp = ibv_poll_cq(comm->scq, 1, &wc);
    HMC_VERBOSE_DBG_ONLY(comm->ctx, 100, "Polled send completions: %d", num_comp);
    if (num_comp < 0) {
        HMC_ERR("send queue poll completion failed %d", num_comp);
        exit(-1); // Fatal error ?
    } else if (num_comp > 0) {
        if (wc.status != IBV_WC_SUCCESS) {
            HMC_ERR("SEND completion failure: %s, psn %d", ibv_wc_status_str(wc.status), (int)wc.wr_id);
            exit(-1);
        }
        comm->pending_send -= num_comp;
    }
    return num_comp;
}

static inline int mcast_send(hmc_comm_t* comm, hmc_coll_req_t *req, int num_packets, const int zcopy)
{
    struct ibv_send_wr *bad_wr;
    struct ibv_send_wr* swr = &comm->mcast.swr;
    struct ibv_sge* ssg = &comm->mcast.ssg;
    int rc;
    int offset = req->offset, i;
    int max_per_packet = comm->max_per_packet;

    for (i = 0; i < num_packets; i++) {
        struct pp_packet *pp = buf_get_free(comm);
        assert(pp->context == 0);
        __builtin_prefetch((void*) pp->buf);
        __builtin_prefetch(req->ptr + offset);

        int length = req->to_send == 1 ? req->last_pkt_len : max_per_packet;
        pp->length = length;
        pp->psn = req->num_roots > 1 ?
            req->first_send_psn + req->num_packets - req->to_send : comm->psn;
        ssg[0].addr = (uintptr_t)req->ptr + offset;
        if (!zcopy && (length > comm->max_inline)) {
            /*GPU buffers either use zcopy or staging_to_host protocol. */
            assert(HMC_MEM_TYPE_GPU != req->buf_mem_type);
            memcpy((void*) pp->buf, req->ptr + offset, length);
            ssg[0].addr = (uint64_t) pp->buf;
        } else {
            pp->context = (uintptr_t)req->ptr + offset;
        }
        ssg[0].length = length;
        ssg[0].lkey = req->mr->lkey;
        swr[0].wr_id = pp->psn;
        swr[0].imm_data = htonl(pp->psn);
        swr[0].send_flags =
            (length <= comm->max_inline && HMC_MEM_TYPE_GPU != req->buf_mem_type) ? IBV_SEND_INLINE : 0;
        comm->r_window[pp->psn & (comm->wsize-1)] = pp;
        comm->psn++;
        req->to_send--;
        offset += length;
        comm->tx++;
        if (comm->tx == comm->ctx->config.scq_moderation) {
            swr[0].send_flags |= IBV_SEND_SIGNALED;
            comm->pending_send++;
            comm->tx = 0;
        }
        while (comm->ctx->config.sx_depth <=
               (comm->pending_send * comm->ctx->config.scq_moderation + comm->tx)) {
            mcast_poll_send(comm);
        }
        HMC_VERBOSE_DBG_ONLY(comm->ctx, 40,"post_send, psn %d, length %d, zcopy %d, signaled %d, gpu %d",
                             pp->psn, pp->length, zcopy, swr[0].send_flags & IBV_SEND_SIGNALED,
                             HMC_MEM_TYPE_GPU == req->buf_mem_type);
        if (0 != (rc = ibv_post_send(comm->mcast.qp, &swr[0], &bad_wr))) {
            HMC_ERR("Post send failed: ret %d, start_psn %d, to_send %d, to_recv %d, length %d, psn %d, inline %d",
                      rc, req->start_psn, req->to_send, req->to_recv,
                      length, pp->psn, length <= comm->max_inline);
            return HMC_ERROR;
        }
        if (!comm->ctx->params.mt_enabled) {
            check_nack_requests(comm, pp->psn);
        }
    }
    req->offset = offset;

    return HMC_SUCCESS;
}

static inline int mcast_recv(hmc_comm_t *comm, hmc_coll_req_t *req,
                             int num_left)
{
    struct pp_packet *pp, *next;
    ucs_list_for_each_safe(pp, next, &comm->pending_q, super) {
        if (PSN_IS_IN_RANGE(pp->psn, req, comm)) {
            __builtin_prefetch(req->ptr + PSN_TO_RECV_OFFSET(pp->psn, req, comm));
            __builtin_prefetch((void*) pp->buf);
            ucs_list_del(&pp->super);
            process_packet(comm, req, pp);
            num_left --;
        } else if (pp->psn < comm->last_acked){
            ucs_list_del(&pp->super);
            ucs_list_add_tail(&comm->bpool, &pp->super);
        }
    };

    while (num_left > 0) {
        struct ibv_wc wc[POLL_PACKED];
        int num_comp = ibv_poll_cq(comm->rcq, POLL_PACKED, &wc[0]);

        int i;
        if (num_comp < 0) {
            HMC_ERR("recv queue poll completion failed %d", num_comp);
            exit(-1); // Fatal error ?
        } else if (num_comp == 0) {
            break;
        }
        int real_num_comp = num_comp;

        for (i = 0; i < real_num_comp; i++) {

#if 0
            if (rand() % 10000 < 500) {
                HMC_VERBOSE_DBG_ONLY(comm->ctx, 50, "DROP: comm_id %d, psn %d",
                                     comm->comm_id, ntohl(wc[i].imm_data));
                num_comp --;
                continue;
            }
#endif

            assert(wc[i].status == IBV_WC_SUCCESS);
            uint64_t id = wc[i].wr_id;
            struct pp_packet* pp = (struct pp_packet*) (id);
            pp->length = wc[i].byte_len - GRH_LENGTH;
            pp->psn = ntohl(wc[i].imm_data);

            HMC_VERBOSE_DBG_ONLY(comm->ctx, 40,"completion: psn %d, length %d, already_received %d,"
                                 " psn in req %d, req_start %d, req_num packets %d, to_send %d, "
                                 "to_recv %d, num_left %d", pp->psn,
                                 pp->length, PSN_RECEIVED(pp->psn, comm) > 0,
                                 PSN_IS_IN_RANGE(pp->psn, req, comm), req->start_psn, req->num_packets,
                                 req->to_send, req->to_recv, num_left);
            if (PSN_RECEIVED(pp->psn, comm) || pp->psn < comm->last_acked) {
                /* This psn was already received */
                assert(pp->context == 0);
                ucs_list_add_tail(&comm->bpool, &pp->super);
            } else {
                if (num_left > 0 && PSN_IS_IN_RANGE(pp->psn, req, comm)) {
                    __builtin_prefetch(req->ptr + PSN_TO_RECV_OFFSET(pp->psn, req, comm));
                    __builtin_prefetch((void*) pp->buf);
                    process_packet(comm, req, pp);
                    num_left--;
                } else {
                    ucs_list_add_tail(&comm->pending_q, &pp->super);
                }
            }
        }
        comm->pending_recv -= num_comp;
        post_recv_buffers(comm);
    }
    return num_left;
}


static inline int mcast_poll_recv(hmc_comm_t *comm)
{
    struct ibv_wc wc;
    int num_comp;
    do {
        num_comp = ibv_poll_cq(comm->rcq, 1, &wc);

        if (num_comp < 0) {
            HMC_ERR("recv queue poll completion failed %d", num_comp);
            exit(-1); // Fatal error ?
        }

        if (num_comp > 0) {
            // Make sure we received all in order.
            uint64_t id = wc.wr_id;
            int length = wc.byte_len - GRH_LENGTH;
            uint32_t psn = ntohl(wc.imm_data);
            struct pp_packet* pp = (struct pp_packet*) id;
            if (psn >= comm->psn) {
                assert(!PSN_RECEIVED(psn, comm));
                pp->psn = psn;
                pp->length = length;
                ucs_list_add_tail(&comm->pending_q, &pp->super);
            } else {
                assert(pp->context == 0);
                ucs_list_add_tail(&comm->bpool, &pp->super);
            }
            comm->pending_recv--;
            post_recv_buffers(comm);
        }
    } while (num_comp);
    return num_comp;
}

static inline int mcast_reliable(hmc_comm_t *comm)
{
    int racks, sacks, nacks;
    HMC_VERBOSE(comm->ctx, 15, "REL: comm %d, rank %d, psn %d",
                comm->comm_id, comm->rank, comm->psn);
    if (comm->parent_n) {
        reliable_send(comm);
    }
    hmc_p2p_progress(comm->ctx);

    racks = comm->racks_n;
    sacks = comm->sacks_n;
    nacks = comm->nack_requests;
    while (comm->racks_n != comm->child_n || comm->sacks_n != comm->parent_n ||
           comm->nack_requests) {
        if (comm->pending_send)
            mcast_poll_send(comm);
        mcast_poll_recv(comm);
        if (comm->ctx->params.mt_enabled) {
            check_nack_requests_all_mt(comm->ctx);
        }
        hmc_p2p_progress(comm->ctx);
        if (racks == comm->racks_n && sacks == comm->sacks_n && nacks == comm->nack_requests) {
            /* No progress on P2P */
            comm->ctx->params.runtime_progress();
        } else {
            racks = comm->racks_n;
            sacks = comm->sacks_n;
            nacks = comm->nack_requests;
        }
    }

    // Reset for next round.
    memset(comm->parents, 0, sizeof(comm->parents));
    memset(comm->children, 0, sizeof(comm->children));
    comm->racks_n = comm->child_n = 0;
    comm->sacks_n = comm->parent_n = 0;
    comm->reliable_ready_multiroot = 0;
    return 1;
}

static inline void mcast_flush(hmc_comm_t* comm)
{
    while (comm->pending_send) {
        mcast_poll_send(comm);
    }
    mcast_poll_recv(comm);
}

#endif
