/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#ifndef HMC_MCAST_H
#define HMC_MCAST_H
#include "config.h"
#include <assert.h>
#include <math.h>
#include <netinet/in.h>
#include <rdma/rdma_cma.h>
#include <stdio.h>
#include <sys/socket.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>

#include <infiniband/ib.h>
#include <infiniband/umad.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_verbs.h>

#include <ucs/datastruct/list.h>
#include <ucs/datastruct/mpool.h>
#include <ucp/api/ucp.h>
#ifdef HAVE_UCS_MEMORY_RCACHE_H
#include <ucs/memory/rcache.h>
#else
#include <ucs/sys/rcache.h>
#endif
#ifdef HAVE_UCS_MEMORY_MEMTYPE_CACHE_H
#include <ucs/memory/memtype_cache.h>
#else
#include <ucs/sys/memtype_cache.h>
#endif
#include "hmc_cuda.h"
#include <string.h>
#include "hmc.h"
#include "hmc_verbose.h"

#define POLL_PACKED 16
#define REL_DONE ((void*)-1)
#define MAX_COMM_POW2 32

enum {
    HMC_PROTO_EAGER, /*internal hmc buffers */
    HMC_PROTO_ML_BUFFER,
    HMC_PROTO_ZCOPY
};

#define MULTICAST_QPN 0xFFFFFF
#define DEF_QKEY 0x1a1a1a1a
#define DEF_PKEY 0xffff
#define DEF_PSN 0
#define DEF_SL 0
#define DEF_SRC_PATH_BITS 0
#define GRH_LENGTH 40

enum {
    HMC_P2P_NACK = 123,
    HMC_P2P_ACK,
    HMC_P2P_NEED_NACK_SEND
};

enum {
    HMC_RECV_WR = 1,
    HMC_WAIT_RECV_WR,
    HMC_SEND_WR,
    HMC_CALC_WR,
    HMC_BCASTRECV_BUFFER_WR,
    HMC_BCASTRECV_WR,
    HMC_BCASTSEND_WR,
};

struct hmc_comm;

typedef struct hmc_context_config {
    char    *devices;
    int      print_nack_stats;
    int      memtype_cache_enabled;
    uint64_t timeout;
    int      sx_depth;
    int      rx_depth;
    int      sx_sge;
    int      rx_sge;
    int      sx_inline;
    int      post_recv_thresh;
    int      scq_moderation;
    int      wsize;
    int      cu_stage_thresh;
    int      max_eager;
    int      verbose;
} hmc_context_config_t;

struct app_context {
    hmc_context_config_t              config;
    hmc_ctx_params_t                  params;
    struct ibv_context*               ctx;
    struct ibv_pd*                    pd;
    char*                             devname;
    int                               max_qp_wr;
    int                               ib_port;
    int                               pkey_index;
    int                               mtu;
    struct rdma_cm_id*                id;
    struct rdma_event_channel*        channel;
    ucs_mpool_t                       p2p_reqs_pool;
    ucs_list_link_t                   pending_nacks_list;
    pthread_mutex_t                   nacks_list_lock;
    ucs_rcache_t*                     rcache;
    ucs_memtype_cache_t*              memtype_cache;
    pthread_mutex_t                   lock;
    ucp_context_h                     ucp_context;
    ucp_worker_h                      ucp_worker;
    ucp_ep_h*                         world_eps;
    size_t                            ucp_addrlen;
    ucp_address_t*                    worker_address;
    ucs_mpool_t                       coll_reqs_pool;
#ifdef CUDA_ENABLED
    hmc_cuda_module_t                 cuda;
#endif
};

struct packet {
    int      type;
    int      from;
    uint32_t psn;
#if ENABLE_DEBUG
    int      comm_id;
#endif
};

struct mcast_ctx {
    struct ibv_qp      *qp;
    struct ibv_ah      *ah;
    struct ibv_send_wr swr;
    struct ibv_sge     ssg;
};

struct pp_packet {
    ucs_list_link_t super;
    uint32_t        psn;
    int             length;
    uintptr_t       context;
    uintptr_t       buf; /* buffer address, initialized once. */
};

struct hmc_comm { /* Stuff at a per-communicator sort of level */
    struct pp_packet    dummy_packet;
    hmc_comm_params_t   params;
    /* These can be shared between calls.*/
    int                 tx;
    struct app_context  *ctx;
    struct ibv_cq       *scq;
    struct ibv_cq       *rcq;
    int                 rank;
    int                 commsize;
    int                 comm_id;
    ucp_ep_h            *ucp_eps;
    void                *addresses_array;
    size_t              max_addrlen;
 
    char                *grh_buf;
    struct              ibv_mr *grh_mr;
    uint16_t            mcast_lid;
    union ibv_gid       mgid;

    unsigned            max_inline;
    size_t              max_eager;
    int                 max_per_packet;
    int                 pending_send;
    int                 pending_recv;

    /* These are for buffering protocol*/
    struct ibv_mr       *pp_mr;
    char                *pp_buf;
    struct pp_packet    *pp;
    uint32_t            psn, last_psn, racks_n, sacks_n, last_acked, naks_n;
    uint32_t            child_n, parent_n;
    int                 buf_n;

    struct packet       p2p_pkt[MAX_COMM_POW2];
    struct packet       p2p_spkt[MAX_COMM_POW2];
    ucs_list_link_t     bpool;
    ucs_list_link_t     pending_q;
    struct mcast_ctx    mcast;

/* The variable below will store flags indicating that the reliability protocol
   was prepared for a given root of multiroot alg. This is uint64_t value, hence the
   max number of roots that can be stored is 64. At the same time the max number of
   roots is defined in src/ hcoll/mcast/mcast.h as HCOLL_MCAST_MAX_ROOTS_NUM=64.
   Don't change that number to be larger than 64. In that case change the logic here. */

    uint64_t            reliable_ready_multiroot;
    /* These are potentially pooled among calls.*/
    struct ibv_recv_wr  *call_rwr;
    struct ibv_sge      *call_rsgs;

    /* These are for time-out and sending NACK.*/
    uint64_t            timer;
    int                 stalled;
    struct sockaddr_in6 mcast_addr;
    int                 parents[MAX_COMM_POW2];
    int                 children[MAX_COMM_POW2];
    int                 nack_requests;
    int                 nacks_counter;
    pthread_mutex_t     lock;
    int                 n_prep_reliable_mr;
    int                 n_prep_reliable;
    int                 n_mcast_reliable;
    void                *cu_stage_buf;
    int                 wsize;
    struct pp_packet    *r_window[1];
};
typedef struct hmc_comm hmc_comm_t;

enum {
    HMC_P2P_REQ_REGULAR,
    HMC_P2P_REQ_NACK,
};

typedef struct hmc_p2p_req {
    ucs_list_link_t super;
    int             pkt_id;
    hmc_comm_t      *comm;
} hmc_p2p_req_t;

#define PSN_IS_IN_RANGE(_psn, _call, _comm)                               \
    ((_psn >= _call->start_psn) &&                                        \
     (_psn < _call->start_psn + _call->num_packets * _call->num_roots) && \
     (_psn >= _comm->last_acked) &&                                       \
     (_psn < _comm->last_acked + _comm->wsize))

#define PSN_RECEIVED(_psn, _comm)  (_comm->r_window[(_psn) % _comm->wsize]->psn == (_psn))
typedef struct hmc_coll_req { /* Stuff that has to happen per call */
    hmc_comm_t          *comm;
    size_t              length;
    int                 proto;
    int                 non_blocking;
    struct ibv_mr       *mr;
    struct ibv_recv_wr  *rwr;
    struct ibv_sge      *rsgs;
    ucs_rcache_region_t *rreg;
    char*               ptr;
    int                 am_root;
    int                 root;
    int                 num_roots;
    void                **rbufs;
    int                 first_send_psn;
    int                 to_send;
    int                 to_recv;
    int                 parent;
    uint32_t            start_psn;
    int                 num_packets;
    int                 last_pkt_len;
    int                 offset;
    int                 buf_mem_type;
    int                 use_cuda_staging;
    int                 already_staged;
} hmc_coll_req_t;

int hmc_setup_qps(struct app_context *ctx, hmc_comm_t *comm);
int hmc_init_qps(struct app_context *ctx, hmc_comm_t *comm);

int setup_mcast(hmc_comm_t *comm);
int create_ah(hmc_comm_t *comm);

struct app_context *setup_ctx(hmc_ctx_params_t *params, hmc_context_config_t *hmc_cfg);
hmc_comm_t* setup_comm(struct app_context *ctx, hmc_comm_params_t *params);
int setup_mcast_group(struct app_context *ctx, hmc_comm_t *comm, int uniq_ctx_id, uint16_t *mlid);
int fini_mcast_group(struct app_context *ctx, hmc_comm_t *comm);
int clean_ctx(struct app_context *ctx);
int clean_comm(hmc_comm_t *comm);

static inline
struct pp_packet* buf_get_free(hmc_comm_t* comm) {
    return ucs_list_extract_head(&comm->bpool, struct pp_packet, super);
}

static inline
int post_recv_buffers(hmc_comm_t* comm) {
    int count = comm->ctx->config.rx_depth - comm->pending_recv;
    if (count <= comm->ctx->config.post_recv_thresh) return 0;
    struct ibv_recv_wr *bad_wr;
    struct ibv_recv_wr *rwr = comm->call_rwr;
    struct ibv_sge *sge = comm->call_rsgs;
    struct pp_packet* pp;
    int i;
    for (i = 0; i < count; i++) {
        if (NULL == (pp = buf_get_free(comm))) {
            break;
        }

        rwr[i].wr_id = ((uint64_t) pp);
        rwr[i].next = &rwr[i+1];
        sge[2*i + 1].addr = pp->buf;
    }
    if (i > 0) {
        rwr[i-1].next = NULL;
        if (ibv_post_recv(comm->mcast.qp, &rwr[0], &bad_wr)) {
            HMC_ERR("Failed to prepost recvs: errno %d", errno);
            return HMC_ERROR;
        }
        comm->pending_recv += i;
    }
    return HMC_SUCCESS;
}

#define PSN_TO_RECV_OFFSET(_psn, _call, _comm) ((ptrdiff_t)((_psn - _call->start_psn)*_comm->max_per_packet))
#define PSN_TO_RECV_LEN(_psn, _call, _comm) ((_psn - _call->start_psn + 1) % _call->num_packets == 0 ? \
                                             _call->last_pkt_len : _comm->max_per_packet)

#define PSN_TO_RECV_OFFSET_MULTIROOT(_psn, _call, _comm)                  \
    ({                                                                    \
        int rbuf = (_psn - _call->start_psn) / _call->num_packets;        \
        int psn_offset = (_psn - _call->start_psn) % _call->num_packets;  \
        ((char*)_call->rbufs[rbuf]) + psn_offset * _comm->max_per_packet; \
    })

void hmc_progress_ctx(struct app_context *ctx);
enum {
    HMC_MEM_TYPE_HOST,
    HMC_MEM_TYPE_GPU,
};
void _hmc_mem_dereg_internal(struct app_context *ctx, struct ibv_mr *mr, ucs_rcache_region_t *region);
struct ibv_mr* _hmc_mem_reg_internal(struct app_context *ctx, void *address,
                                     size_t length, ucs_rcache_region_t **region);
#define HMC_EXPORT __attribute__ ((visibility ("default")))
#endif

