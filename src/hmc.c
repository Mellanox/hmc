/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include "config.h"
#include "hmc.h"
#include "mcast.h"
#include <sys/time.h>
#define NB_POLL 8
#define NB_POLL_LARGE 32

static inline
uint64_t hmc_get_timer(void) {
    struct timeval t;
    gettimeofday(&t, 0);
    return (uint64_t)(t.tv_sec*1000000 + t.tv_usec);
}


static inline int add_uniq(int *arr, int *len, int value) {
    int i;
    for (i=0; i<(*len); i++)
        if (arr[i] == value) return 0;
    arr[*len] = value;
    (*len)++;
    return 1;
}

static int prepare_reliable(hmc_comm_t *comm, hmc_coll_req_t *req, int root) {
    int mask = 1;
    int vrank = TO_VIRTUAL(comm->rank, comm->commsize, root);

    while (mask < comm->commsize) {
        if (vrank & mask) {
            req->parent = TO_ORIGINAL((vrank ^ mask), comm->commsize, root);
            add_uniq(comm->parents, &comm->parent_n, req->parent);
            break;
        } else {
            int child = vrank ^ mask;
            if (child < comm->commsize) {
                child = TO_ORIGINAL(child, comm->commsize, root);
                if (add_uniq(comm->children, &comm->child_n, child)) {
                    HMC_VERBOSE_DBG_ONLY(comm->ctx, 30,
                                         "rank %d, Posting RECV from child %d, n_child %d,  psn %d",
                                         comm->rank, child, comm->child_n, comm->psn);
                    hmc_p2p_recv(comm,&comm->p2p_pkt[comm->child_n - 1], sizeof(struct packet),
                                 child, GET_RELIABLE_TAG(comm), false, true, comm->child_n - 1);
                }
            }
        }
        mask <<= 1;
    }
    return HMC_SUCCESS;
}

static int prepare_reliable_multiroot(hmc_comm_t *comm, hmc_coll_req_t *req) {
    int i;
    if (comm->reliable_ready_multiroot == (((uint64_t)1 << req->num_roots) - 1) ) {
        return HMC_SUCCESS;
    }
    comm->n_prep_reliable_mr++;
    for (i=0; i<req->num_roots; i++) {
        if (comm->reliable_ready_multiroot & ((uint64_t)1 << i)) {
            continue;
        }
        int round_start_psn = req->start_psn + i*req->num_packets;
        if (round_start_psn + req->num_packets > comm->last_acked &&
            round_start_psn < comm->last_acked + comm->wsize) {
            prepare_reliable(comm, req, i);
            comm->reliable_ready_multiroot |= ((uint64_t)1 << i);
        }
    }
    return HMC_SUCCESS;
}
static inline void bcast_check_drop(hmc_comm_t *comm, hmc_coll_req_t *req)
{
    if (comm->timer == 0) {
        comm->timer = hmc_get_timer();
    } else {
        if (hmc_get_timer() - comm->timer >= comm->ctx->config.timeout) {
            HMC_VERBOSE_DBG_ONLY(comm->ctx, 15, "[REL] time out psn %d, timeout %llu us, current timer %llu",
                                 comm->psn, comm->ctx->config.timeout,
                                 (long long unsigned)hmc_get_timer() - comm->timer);
            reliable_send_NACK(comm, req);
            comm->timer = 0;
        }
    }
}

static inline int cuda_staged_deliver(hmc_coll_req_t *req) {
    hmc_comm_t *comm = req->comm;
    if (req->already_staged) {
        void *dest = req->ptr + req->offset;
        hmc_cuda_copy_h2d(dest, comm->cu_stage_buf, req->already_staged, comm->ctx);
        req->offset += req->already_staged;
        req->already_staged = 0;
    }
    return HMC_SUCCESS;
}

static inline void r_window_recycle(hmc_comm_t *comm, hmc_coll_req_t *req) {
    int wsize = comm->wsize;
    int num_free_win = wsize - (comm->psn - comm->last_acked);
    /* When do we need to perform reliability protocol:
       1. Always in the end of the window
       2. For the zcopy case: in the end of collective, because we can't signal completion
       before made sure that children received the data - user can modify buffer.
       3. PROTO_ML_BUFFER does not need to sync in the end of collective since there will be
       memsync barrier at ML level that will guarantee the safe buffer re-usage */
    bool req_completed = req->to_send == 0 && req->to_recv == 0;
    if (!num_free_win || (req->proto == HMC_PROTO_ZCOPY && req_completed)) {
        comm->n_mcast_reliable++;
        mcast_reliable(comm);
        for (;comm->last_acked < comm->psn; comm->last_acked++) {
            struct pp_packet* pp = comm->r_window[comm->last_acked & (wsize-1)];
            assert(pp != &comm->dummy_packet);
            comm->r_window[comm->last_acked & (wsize-1)] = &comm->dummy_packet;
            pp->context = 0;
            ucs_list_add_tail(&comm->bpool, &pp->super);
        }

        if (!req_completed) {
            if (req->num_roots > 1) {
                prepare_reliable_multiroot(comm, req);
            } else {
                prepare_reliable(comm, req, req->root);
            }
        }
        if (req->use_cuda_staging) {
            cuda_staged_deliver(req);
        }
    }
}

static inline int do_bcast(hmc_coll_req_t *req)
{
    double timer_start = 0;
    int polls = 0, num_free_win, num_sent, to_send, to_recv, to_recv_left, ts, tr;
    hmc_comm_t *comm = req->comm;
    const int zcopy = req->proto != HMC_PROTO_EAGER;
    int wsize = comm->wsize;
    pthread_mutex_lock(&comm->lock);
    hmc_p2p_progress(comm->ctx);
    while (req->to_send || req->to_recv) {
        ts = req->to_send;
        tr = req->to_recv;

        num_free_win = wsize - (comm->psn - comm->last_acked);

        /* Send data if i'm root and there is a space in the window */
        if (num_free_win && req->am_root) {
            num_sent = req->num_packets - req->to_send;
            if (req->num_roots == 1) {
                assert(req->to_send > 0);
                assert(req->first_send_psn + num_sent < comm->last_acked + wsize);
            }
            if (req->first_send_psn + num_sent < comm->last_acked + wsize &&
                req->to_send) {
                /* How many to send: either all that are left (if they fit into window) or
                   up to the window limit */
                to_send = MIN(req->to_send,
                              comm->last_acked + wsize - (req->first_send_psn + num_sent));
                mcast_send(comm, req, to_send, zcopy);
                /* Recompute num_free_win: we probably spent some space in r_window for send packets.
                   So need to update num_free_win for the next RECV step (in multiroot case only) */
                num_free_win = wsize - (comm->psn - comm->last_acked);
            }
        }
        if (req->num_roots > 1) {
            prepare_reliable_multiroot(comm, req);
        } else {
            prepare_reliable(comm, req, req->root);
        }

        /* Recv data if i'm not root OR if we have multiple roots
           and there is a space in the window */
        if (num_free_win && req->to_recv && (!req->am_root || req->num_roots > 1)) {
            /* How many to recv: either all that are left or up to the window limit. */
            to_recv = MIN(num_free_win, req->to_recv);
            to_recv_left = mcast_recv(comm, req, to_recv);
            if (to_recv == to_recv_left) {
                /* We didn't receive anything: increase the stalled counter and get ready for
                   drop event */
                if (comm->stalled++ >= 10000)
                    bcast_check_drop(comm, req);
            } else {
                comm->stalled = 0;
                comm->timer = 0;
            }
        }

        /* This function will check if we have to do a round of reliability protocol */
        r_window_recycle(comm, req);

        /* In MT case NACK requests are handled asynchronously, so we have to poke this
           function */
        if (comm->ctx->params.mt_enabled) {
            check_nack_requests_all_mt(comm->ctx);
        }

        if (tr == req->to_recv && ts == req->to_send) {
            hmc_p2p_progress(comm->ctx);
            comm->ctx->params.runtime_progress();
        }
        if (req->non_blocking &&
            (polls++ == (zcopy ? NB_POLL_LARGE : NB_POLL))) break;
    }
    pthread_mutex_unlock(&comm->lock);
    return 0;
}

HMC_EXPORT
hmc_status_t hmc_init(hmc_ctx_params_t *params, hmc_ctx_config_h config, hmc_ctx_h *ctx) {
    hmc_context_config_t *hmc_cfg = (hmc_context_config_t*)config;
    *ctx = setup_ctx(params, hmc_cfg);
    return (*ctx) ? HMC_SUCCESS : HMC_ERROR;
}

HMC_EXPORT
hmc_status_t hmc_comm_create(hmc_ctx_h ctx, hmc_comm_params_t *params, hmc_comm_h *comm) {
    hmc_comm_t *_comm = setup_comm(ctx, params);
    if (_comm == NULL) {
        return HMC_ERROR;
    }

    *comm = _comm;
    return HMC_SUCCESS;
}

static inline int hmc_init_bcast(void* buf, size_t size, int root,
                                 void *mr, hmc_comm_t *comm, hmc_coll_req_t* req) {
    req->comm = comm;
    req->ptr = buf;
    req->length = size;
    req->root = root;
    req->am_root = (root == comm->rank);
    req->mr = comm->pp_mr;
    req->rreg = NULL;
    req->num_roots = 1;
    req->use_cuda_staging = 0;
    req->buf_mem_type = hmc_mem_type(comm->ctx, buf);
    /*Only select Eager for CPU buffs. For GPU the cost of memcpy is too high. */
    req->proto = (req->length < comm->max_eager && HMC_MEM_TYPE_GPU != req->buf_mem_type)
        ? HMC_PROTO_EAGER : HMC_PROTO_ZCOPY;
    if (HMC_MEM_TYPE_GPU == req->buf_mem_type &&
        comm->ctx->config.cu_stage_thresh >= 0 &&
        size >= comm->ctx->config.cu_stage_thresh) {
        if (!comm->cu_stage_buf) {
            comm->cu_stage_buf = hmc_cuda_host_alloc(comm->wsize*comm->max_per_packet,
                                                     comm->ctx);
        }
        req->use_cuda_staging = 1;
        req->already_staged = 0;
    }
    if (req->am_root) {
        if (mr) {
            req->mr = (struct ibv_mr *)mr;
            req->proto = HMC_PROTO_ML_BUFFER;
        } else if (req->proto != HMC_PROTO_EAGER) {
            req->mr = _hmc_mem_reg_internal(comm->ctx, req->ptr, req->length, &req->rreg);
        }
    }

    prepare_reliable(comm, req, req->root);
    req->offset = 0;
    req->start_psn = comm->last_psn;
    req->num_packets = (req->length + comm->max_per_packet - 1)/comm->max_per_packet;
    if (req->num_packets == 0) req->num_packets  = 1;
    req->last_pkt_len = req->length - (req->num_packets - 1)*comm->max_per_packet;
    assert(req->last_pkt_len > 0 && req->last_pkt_len <= comm->max_per_packet);
    comm->last_psn += req->num_packets;
    req->first_send_psn = req->start_psn;
    req->to_send = req->am_root ? req->num_packets : 0;
    req->to_recv = req->am_root ? 0 : req->num_packets;
    return HMC_SUCCESS;
}

HMC_EXPORT
hmc_status_t hmc_req_test(void* request) {
    hmc_coll_req_t *req = (hmc_coll_req_t*) request;
    if (req->to_send == 0 && req->to_recv == 0) {
        if (req->rreg) {
            assert(req->mr);
            _hmc_mem_dereg_internal(req->comm->ctx, req->mr, req->rreg);
        }
        return HMC_SUCCESS;
    } else {
        if (req->comm->psn < req->start_psn) return 0; //TODO: this will break NB collectives
        do_bcast(req);
        req->comm->ctx->params.runtime_progress();
        return HMC_INPROGRESS;
    }
}

HMC_EXPORT
hmc_status_t hmc_req_wait(void* request)
{
    hmc_coll_req_t *req = (hmc_coll_req_t*)request;
    while (HMC_INPROGRESS == hmc_req_test(req)) {
        ;
    }
    return 0;
}

HMC_EXPORT
hmc_status_t hmc_ibcast(hmc_bcast_args_t *args, void **request) {
    hmc_comm_t *hmc_comm = (hmc_comm_t*)args->comm;
    assert(args->field_mask & (HMC_BCAST_ARGS_FIELD_ADDRESS ||
                               HMC_BCAST_ARGS_FIELD_SIZE ||
                               HMC_BCAST_ARGS_FIELD_ROOT ||
                               HMC_BCAST_ARGS_FIELD_COMM));
    HMC_VERBOSE_DBG_ONLY(hmc_comm->ctx, 10, "HMC ibcast start, buf %p, size %d, root %d, comm %d, "
                         "comm_size %d, am_i_root %d",
                         args->buf, args->size, args->root,
                         hmc_comm->comm_id, hmc_comm->commsize, hmc_comm->rank == root);
    hmc_coll_req_t *req = ucs_mpool_get(&hmc_comm->ctx->coll_reqs_pool);
    req->non_blocking= 1;
    hmc_init_bcast(args->address, args->size, args->root,
                   args->field_mask & HMC_BCAST_ARGS_FIELD_MR ? args->mr : NULL,
                   hmc_comm, req);
    hmc_req_test(req);
    *request = req;
    return HMC_SUCCESS;
}

HMC_EXPORT
hmc_status_t hmc_bcast(hmc_bcast_args_t *args) {
    hmc_comm_t *hmc_comm = (hmc_comm_t*)args->comm;
    hmc_coll_req_t req = {0};
    assert(args->field_mask & (HMC_BCAST_ARGS_FIELD_ADDRESS ||
                               HMC_BCAST_ARGS_FIELD_SIZE ||
                               HMC_BCAST_ARGS_FIELD_ROOT ||
                               HMC_BCAST_ARGS_FIELD_COMM));
    HMC_VERBOSE_DBG_ONLY(hmc_comm->ctx, 10, "HMC bcast start, buf %p, size %zd, root %d, comm %d, "
                         "comm_size %d, am_i_root %d",
                         args->address, args->size, args->root,
                         hmc_comm->comm_id, hmc_comm->commsize, hmc_comm->rank == root);
    req.non_blocking = 0;
    hmc_init_bcast(args->address, args->size, args->root,
                   args->field_mask & HMC_BCAST_ARGS_FIELD_MR ? args->mr : NULL,
                   hmc_comm, &req);
    do_bcast(&req);
    if (HMC_MEM_TYPE_GPU == req.buf_mem_type && !req.am_root) {
        if (req.use_cuda_staging) {
            cuda_staged_deliver(&req);
        } else {
            hmc_gpu_synchronize(hmc_comm->ctx);
        }
    }
    if (req.rreg) {
        assert(req.mr);
        _hmc_mem_dereg_internal(req.comm->ctx, req.mr, req.rreg);
    }
    return HMC_SUCCESS;
}


static inline int hmc_init_bcast_multiroot(void *sbuf, void **rbufs, int data_size, int num_roots,
                                           void *mr, hmc_comm_t *comm, hmc_coll_req_t* req) {
    int i;
    req->comm = comm;
    req->ptr = comm->rank < num_roots ? (char*)rbufs[comm->rank] : NULL;
    req->rbufs = rbufs;
    req->length = data_size;
    req->num_roots = num_roots;
    req->am_root = (comm->rank < num_roots);
    req->proto = req->length < comm->max_eager ?
        HMC_PROTO_EAGER : HMC_PROTO_ZCOPY;
    req->mr = comm->pp_mr;
    req->rreg = NULL;
    req->buf_mem_type = HMC_MEM_TYPE_HOST;
    req->use_cuda_staging = 0;
    if (req->am_root) {
        if (data_size) {
            memcpy(req->ptr, sbuf, data_size);
        }
        if (mr) {
            req->mr = (struct ibv_mr *)mr;
            req->proto = HMC_PROTO_ML_BUFFER;
        } else if (req->proto != HMC_PROTO_EAGER) {
            req->mr = _hmc_mem_reg_internal(comm->ctx, req->ptr, req->length, &req->rreg);
        }
    }

    req->offset = 0;
    req->start_psn = comm->last_psn;
    req->num_packets = (req->length + comm->max_per_packet - 1)/comm->max_per_packet;
    if (req->num_packets == 0) req->num_packets  = 1;
    req->last_pkt_len = req->length - (req->num_packets - 1)*comm->max_per_packet;
    req->first_send_psn = req->start_psn + req->num_packets*req->comm->rank;
    req->to_send = req->am_root ? req->num_packets : 0;
    req->to_recv = req->am_root  ? req->num_packets*(num_roots - 1) : req->num_packets*num_roots;
    comm->last_psn += req->num_packets*num_roots;
    assert(req->last_pkt_len <= comm->max_per_packet);

    return HMC_SUCCESS;
}

int hmc_bcast_multiroot(void *sbuf, void **rbufs, int data_size, int num_roots, void *mr, hmc_comm_h comm) {
    hmc_comm_t *hmc_comm = (hmc_comm_t*)comm;
    HMC_VERBOSE(hmc_comm->ctx, 10, "HMC bcast MULTIROOT start, size %d, am root %d, comm %d, "
                "comm_size %d, num_roots %d",
                data_size, hmc_comm->rank < num_roots, hmc_comm->comm_id,
                hmc_comm->commsize, num_roots);
    hmc_coll_req_t req = {0};
    req.non_blocking = 0;
    hmc_init_bcast_multiroot(sbuf, rbufs, data_size, num_roots, mr, comm, &req);
    do_bcast(&req);
    if (req.rreg) {
        assert(req.mr);
        _hmc_mem_dereg_internal(req.comm->ctx, req.mr, req.rreg);
    }

    return HMC_SUCCESS;
}

HMC_EXPORT
void hmc_req_free(void *request) {
    ucs_mpool_put(request);
}

void hmc_comm_flush(hmc_comm_h _comm) {
    hmc_comm_t *comm = (hmc_comm_t *) _comm;
    HMC_VERBOSE(comm->ctx, 3, " HMC comm flush: %p, id %d, mlid %x",
                comm, comm->comm_id, comm->mcast_lid);
    mcast_flush(comm);
    mcast_reliable(comm);
}

HMC_EXPORT
hmc_status_t hmc_comm_destroy(hmc_comm_h comm) {
    hmc_comm_flush(comm);
    return clean_comm(comm);
}

HMC_EXPORT
hmc_status_t hmc_finalize(hmc_ctx_h ctx) {
    return clean_ctx(ctx);
}

HMC_EXPORT
hmc_status_t hmc_mem_register(hmc_ctx_h _ctx, void *data, size_t data_size, void **mr) {
    struct app_context *ctx = (struct app_context*)_ctx;
    *mr = ibv_reg_mr(ctx->pd, data, data_size, IBV_ACCESS_LOCAL_WRITE);
    HMC_VERBOSE_DBG_ONLY(ctx, 10,"External memory register: ptr %p, len %zd, mr %p",
                         data, data_size, (*mr));
    if (!*mr) {
        HMC_ERR("Failed to register MR: errno %d", errno);
        return HMC_ERROR;
    }
    return HMC_SUCCESS;
}

HMC_EXPORT
hmc_status_t hmc_mem_deregister(hmc_ctx_h _ctx, void *mr) {
    struct app_context *ctx = (struct app_context*)_ctx;
    if(mr != NULL) {
        int rc;
        HMC_VERBOSE_DBG_ONLY(ctx, 10,"External memory deregister: mr %p", mr);
        rc = ibv_dereg_mr(mr);
        return (rc == 0) ? HMC_SUCCESS: HMC_ERROR;
    } else {
        HMC_VERBOSE_DBG_ONLY(ctx, 10,"External memory mr %p already deregistered", mr);
        return HMC_SUCCESS;
    }
}

void hmc_progress_ctx(struct app_context *ctx) {
    check_nack_requests_all_mt(ctx);
}
