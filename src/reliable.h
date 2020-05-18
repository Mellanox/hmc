/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#ifndef RELIABLE_H_
#define RELIABLE_H_
#include "hmc_p2p.h"
#define TO_VIRTUAL(_rank, _size, _root) ((_rank + _size - _root) % _size)
#define TO_ORIGINAL(_rank, _size, _root) ((_rank + _root) % _size)

#define ACK 1
static void recv_completion_req(void *hmc_p2p_req, ucs_status_t status, ucp_tag_recv_info_t *info);
static void send_completion_req(void *hmc_p2p_req, ucs_status_t status);
static inline void hmc_p2p_wait_completion(hmc_comm_t *comm, hmc_p2p_req_t *req);
static inline void recv_completion(hmc_comm_t *comm, int pkt_id);
static inline void send_completion(hmc_comm_t *comm);

static void send_completion_release_req(void *hmc_p2p_req, ucs_status_t status) {
    hmc_p2p_req_t *req = (hmc_p2p_req_t*)hmc_p2p_req;
    req->comm = NULL;
    ucp_request_free(hmc_p2p_req);
}

static void recv_completion_release_req(void *hmc_p2p_req, ucs_status_t status, ucp_tag_recv_info_t *info) {
    hmc_p2p_req_t *req = (hmc_p2p_req_t*)hmc_p2p_req;
    req->comm = NULL;
    ucp_request_free(hmc_p2p_req);
}

static void send_completion_dummy(void *hmc_p2p_req, ucs_status_t status) {
}

static void recv_completion_dummy(void *hmc_p2p_req, ucs_status_t status, ucp_tag_recv_info_t *info) {
}

static inline int
hmc_p2p_send(hmc_comm_t *comm, void* buf, size_t size, int rank,  int tag,
             bool is_blocking, bool need_completion, int pkt_id) {
    ucp_ep_h ep = hmc_p2p_get_ep(comm, rank);
    ucp_tag_t ucp_tag = HMC_P2P_MAKE_SEND_TAG(tag, comm->rank, comm->mcast_lid);
    hmc_p2p_req_t *req = (hmc_p2p_req_t *) ucp_tag_send_nb(ep, buf, 1, ucp_dt_make_contig(size), ucp_tag,
                                                           is_blocking ? send_completion_dummy :
                                                           (need_completion ? send_completion_req : send_completion_release_req));
    HMC_P2P_CHECK_REQ_STATUS();
    if (need_completion) {
        if (req) {
            if (!__sync_bool_compare_and_swap(&req->comm, NULL, comm)) {
                assert(req->comm == (void*)0x1);
                send_completion(comm);
                req->comm = NULL;
                ucp_request_free(req);
                req = NULL;
            }
        } else {
            send_completion(comm);
        }
    }
    if (req && is_blocking) {
        hmc_p2p_wait_completion(comm, req);
    }
    return HMC_SUCCESS;
}

static inline int
hmc_p2p_recv(hmc_comm_t *comm, void* buf, size_t size, int rank,  int tag,
             bool is_blocking, bool need_completion, int pkt_id) {
    ucp_tag_t ucp_tag, ucp_tag_mask;
    HMC_P2P_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, rank, comm->mcast_lid);
    hmc_p2p_req_t *req = (hmc_p2p_req_t *) ucp_tag_recv_nb(comm->ctx->ucp_worker, buf, 1,
                                                           ucp_dt_make_contig(size), ucp_tag, ucp_tag_mask,
                                                           is_blocking ? recv_completion_dummy :
                                                           (need_completion ? recv_completion_req : recv_completion_release_req));
    HMC_P2P_CHECK_REQ_STATUS();
    assert(req);
    req->pkt_id = pkt_id;
    if (need_completion) {
        if (!__sync_bool_compare_and_swap(&req->comm, NULL, comm)) {
            assert(req->comm == (void*)0x1);
            recv_completion(comm, pkt_id);
            req->comm = NULL;
            ucp_request_free(req);
            req = NULL;
        }
    }
    if (req && is_blocking) {
        hmc_p2p_wait_completion(comm, req);
    }

    return HMC_SUCCESS;
}

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define GET_RELIABLE_TAG(_comm) (_comm->last_acked % 1024)

static inline int resend_packet_reliable(hmc_comm_t *comm, int p2p_pkt_id) {
    uint32_t psn = comm->p2p_pkt[p2p_pkt_id].psn;
    struct pp_packet *pp = comm->r_window[psn % comm->wsize];
    assert(pp->psn == psn);
    HMC_VERBOSE_DBG_ONLY(comm->ctx, 15, "[comm %d, rank %d] Send data NACK: to %d, "
                         "psn %d, context %p, buf %p, len %d",
                         comm->comm_id, comm->rank,
                         comm->p2p_pkt[p2p_pkt_id].from, psn, pp->context,
                         (void*) (pp->context ? pp->context : pp->buf), pp->length);
    hmc_p2p_send(comm, (void*) (pp->context ? pp->context : pp->buf), pp->length,
                 comm->p2p_pkt[p2p_pkt_id].from, 2703, false, false, p2p_pkt_id);
    hmc_p2p_recv(comm, &comm->p2p_pkt[p2p_pkt_id], sizeof(struct packet), comm->p2p_pkt[p2p_pkt_id].from,
                 GET_RELIABLE_TAG(comm), false, true, p2p_pkt_id);
    return HMC_SUCCESS;
}

static inline int check_nack_requests(hmc_comm_t *comm, uint32_t psn) {
    int i;
    assert(!comm->ctx->params.mt_enabled);
    if (!comm->nack_requests)
        return HMC_SUCCESS;

    for (i=0; i<comm->child_n; i++){
        if (psn == comm->p2p_pkt[i].psn &&
            comm->p2p_pkt[i].type == HMC_P2P_NEED_NACK_SEND) {
            resend_packet_reliable(comm, i);
            comm->p2p_pkt[i].type = HMC_P2P_ACK;
            comm->nack_requests--;
        }
    }
    return HMC_SUCCESS;
}

static inline int check_nack_requests_all(hmc_comm_t *comm) {
    int i;
    assert(!comm->ctx->params.mt_enabled);
    if (!comm->nack_requests)
        return HMC_SUCCESS;

    for (i=0; i<comm->child_n; i++){
        if (comm->p2p_pkt[i].type == HMC_P2P_NEED_NACK_SEND) {
            uint32_t psn = comm->p2p_pkt[i].psn;
            struct pp_packet* pp = comm->r_window[psn % comm->wsize];
            if (psn == pp->psn) {
                resend_packet_reliable(comm, i);
                comm->p2p_pkt[i].type = HMC_P2P_ACK;
                comm->nack_requests--;
            }
        }
    }
    return HMC_SUCCESS;
}
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
static inline
int check_nack_requests_all_mt(struct app_context *ctx) {
    assert(ctx->params.mt_enabled);

    /*First check len of the list w/o acquiring the lock. This
      is fast and should return 0 most of the time */
    int len = ucs_list_length(&ctx->pending_nacks_list);
    if (0 == len)
        return HMC_SUCCESS;

    if (!pthread_mutex_trylock(&ctx->lock)) {
        //TODO all locks should be enabled in MT env only
        pthread_mutex_lock(&ctx->nacks_list_lock);
        len = ucs_list_length(&ctx->pending_nacks_list);
        pthread_mutex_unlock(&ctx->nacks_list_lock);

        if (len > 0) {
            int i;
            hmc_p2p_req_t *next, *req = ucs_list_head(&ctx->pending_nacks_list, hmc_p2p_req_t, super);
            for (i=0; i<len; i++) {
                next = ucs_list_next(&req->super, hmc_p2p_req_t, super);
                hmc_comm_t *comm = req->comm;
                int pkt_id = req->pkt_id;
                uint32_t psn = comm->p2p_pkt[pkt_id].psn;
                if (psn == comm->r_window[psn % comm->wsize]->psn) {
                    pthread_mutex_lock(&ctx->nacks_list_lock);
                    ucs_list_del(&req->super);
                    pthread_mutex_unlock(&ctx->nacks_list_lock);
                    assert(comm->p2p_pkt[pkt_id].type == HMC_P2P_NEED_NACK_SEND);
                    resend_packet_reliable(comm, pkt_id);
                    comm->p2p_pkt[pkt_id].type = HMC_P2P_ACK;
                    ucs_mpool_put(&req);
                }
                req = next;
            }
        }
        pthread_mutex_unlock(&ctx->lock);
    }
    return HMC_SUCCESS;
}

static inline int find_nack_psn(hmc_comm_t* comm, hmc_coll_req_t *req) {
    int psn = MAX(comm->last_acked, req->start_psn);
    int max_search_psn = MIN(req->start_psn + req->num_packets*req->num_roots,
                             comm->last_acked + comm->wsize + 1);
    for (; psn < max_search_psn; psn++) {
        if (!PSN_RECEIVED(psn, comm)) {
            break;
        }
    }
    assert(psn < max_search_psn);
    return psn;
}

static inline int psn2root(hmc_coll_req_t *req, int psn) {
    return (psn - req->start_psn) / req->num_packets;
}

static inline int get_nack_parent(hmc_coll_req_t *req, int psn) {
    hmc_comm_t * comm = req->comm;
    if (req->num_roots == 1) {
        return req->parent;
    } else {
        int mask = 1;
        int root = psn2root(req, psn);
        int vrank = TO_VIRTUAL(comm->rank, comm->commsize, root);
        while (mask < comm->commsize) {
            if (vrank & mask) {
                return TO_ORIGINAL((vrank ^ mask), comm->commsize, root);
            }
            mask <<= 1;
        }
    }
    HMC_ERR("Failed to compute nack parent: comm_id %d, comm_size %d, psn %d",
              comm->comm_id, comm->commsize, psn);
    return -1;
}
static inline int reliable_send_NACK(hmc_comm_t* comm, hmc_coll_req_t *req)
{
    void *dest;
    struct packet p = {
        .type = HMC_P2P_NACK,
        .psn = find_nack_psn(comm, req),
        .from = comm->rank,
#if ENABLE_DEBUG
        .comm_id = comm->comm_id,
#endif
    };
    int parent = get_nack_parent(req, p.psn);
    if (parent < 0) {
        return HMC_ERROR;
    }
    comm->nacks_counter++;

    hmc_p2p_send(comm, &p, sizeof(struct packet), parent, GET_RELIABLE_TAG(comm), true, false, 0);
    HMC_VERBOSE_DBG_ONLY(comm->ctx, 15, "[comm %d, rank %d] Sent NAK : parent %d, psn %d",
                         comm->comm_id, comm->rank, parent, p.psn);

    // Prepare to obtain the data.
    struct pp_packet* pp = buf_get_free(comm);
    pp->psn = p.psn;
    pp->length = PSN_TO_RECV_LEN(pp->psn, req, comm);

    hmc_p2p_recv(comm, (void*) pp->buf, pp->length, parent, 2703, true, false, 0);
    HMC_VERBOSE_DBG_ONLY(comm->ctx, 15, "[comm %d, rank %d] Recved data psn %d, buf %p, length %d",
                         comm->comm_id, comm->rank, p.psn, pp->buf, pp->length);

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
    req->to_recv--;
    comm->r_window[pp->psn % comm->wsize] = pp;
    if (!comm->ctx->params.mt_enabled) {
        check_nack_requests(comm, pp->psn);
    }
    comm->psn++;

    return HMC_SUCCESS;
}

static inline int reliable_send(hmc_comm_t* comm)
{
    HMC_VERBOSE_DBG_ONLY(comm->ctx, 20, "comm %p, psn %d, last_acked %d, n_parent %d",
                         comm, comm->psn, comm->last_acked, comm->parent_n);
    int i;
    for (i=0; i<comm->parent_n; i++) {
        int parent = comm->parents[i];
        comm->p2p_spkt[i].type = HMC_P2P_ACK;
        comm->p2p_spkt[i].psn = comm->last_acked + comm->wsize;
#if ENABLE_DEBUG
        comm->p2p_spkt[i].comm_id = comm->comm_id;
#endif
        HMC_VERBOSE_DBG_ONLY(comm->ctx, 30, "rank %d, Posting SEND to parent %d, n_parent %d,  psn %d",
                             comm->rank, parent, comm->parent_n, comm->psn);
        hmc_p2p_send(comm, &comm->p2p_spkt[i], sizeof(struct packet),
                     parent, GET_RELIABLE_TAG(comm), false, true, i);
    }
    return HMC_SUCCESS;
}


static inline void recv_completion(hmc_comm_t *comm, int pkt_id) {
#if ENABLE_DEBUG
    assert(comm->comm_id == comm->p2p_pkt[pkt_id].comm_id);
#endif
    if (comm->p2p_pkt[pkt_id].type != HMC_P2P_ACK) {
        assert(comm->p2p_pkt[pkt_id].type == HMC_P2P_NACK);
        if (!comm->ctx->params.mt_enabled) {
            uint32_t psn = comm->p2p_pkt[pkt_id].psn;
            struct pp_packet* pp = comm->r_window[psn % comm->wsize];
            HMC_VERBOSE_DBG_ONLY(comm->ctx, 15, "[comm %d, rank %d] Got NACK: from %d, psn %d, avail %d",
                                 comm->comm_id, comm->rank,
                                 comm->p2p_pkt[pkt_id].from, psn, pp->psn == psn);

            if (pp->psn == psn) {
                resend_packet_reliable(comm, pkt_id);
            } else {
                comm->p2p_pkt[pkt_id].type = HMC_P2P_NEED_NACK_SEND;
                comm->nack_requests++;
            }
        } else {
            HMC_VERBOSE_DBG_ONLY(comm->ctx, 15, "[comm %d, rank %d] Got NACK: from %d, "
                                 "psn %d MultiThread --> enqueue",
                                 comm->comm_id, comm->rank,
                                 comm->p2p_pkt[pkt_id].from, comm->p2p_pkt[pkt_id].psn);

            hmc_p2p_req_t *req = hmc_get_p2p_request(comm, pkt_id);
            comm->p2p_pkt[pkt_id].type = HMC_P2P_NEED_NACK_SEND;
            pthread_mutex_lock(&comm->ctx->nacks_list_lock);
            ucs_list_add_tail(&comm->ctx->pending_nacks_list, &req->super);
            pthread_mutex_unlock(&comm->ctx->nacks_list_lock);
        }
    } else {
        comm->racks_n++;
    }
}

static void recv_completion_req(void *hmc_p2p_req, ucs_status_t status, ucp_tag_recv_info_t *info) {
    hmc_p2p_req_t *req = (hmc_p2p_req_t*)hmc_p2p_req;
    if (!__sync_bool_compare_and_swap(&req->comm, NULL, (void*)0x1)) {
        assert(NULL != req->comm && (void*)0x1 != req->comm);
        recv_completion(req->comm, req->pkt_id);
        req->comm = NULL;
        ucp_request_free(hmc_p2p_req);
    }
}

static inline void send_completion(hmc_comm_t *comm) {
    comm->sacks_n++;
}

static void send_completion_req(void *hmc_p2p_req, ucs_status_t status) {
    hmc_p2p_req_t *req = (hmc_p2p_req_t*)hmc_p2p_req;
    if (!__sync_bool_compare_and_swap(&req->comm, NULL, (void*)0x1)) {
        assert(NULL != req->comm && (void*)0x1 != req->comm);
        send_completion(req->comm);
        req->comm = NULL;
        ucp_request_free(hmc_p2p_req);
    }
}

static inline void hmc_p2p_wait_completion(hmc_comm_t *comm, hmc_p2p_req_t *req) {
    while (UCS_INPROGRESS == ucp_request_check_status(req)) {
        hmc_p2p_progress(comm->ctx);
        if (!comm->ctx->params.mt_enabled) {
            check_nack_requests_all(comm);
        } else {
            check_nack_requests_all_mt(comm->ctx);
        }
        comm->ctx->params.runtime_progress();
    }
    req->comm = NULL;
    ucp_request_free(req);
}

#endif
