/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#ifndef __HMC_P2P_H__
#define __HMC_P2P_H__
#include "hmc_mcast.h"
static inline
hmc_p2p_req_t* hmc_get_p2p_request(hmc_comm_t *comm, int pkt_id) {
    hmc_p2p_req_t *req = ucs_mpool_get(&comm->ctx->p2p_reqs_pool);
    req->comm = comm;
    req->pkt_id = pkt_id;
    return req;
}

/*
 * UCX tag structure:
 *
 *    012345678          01234567 01234567     01234567 01234567 01234567 01234567 01234567
 *                 |                          |                          |
 *   reserved (8)  |     message tag (16)     |     source rank (24)     |  context id (16)
 */

#define HMC_P2P_RESERVED_BITS          8
#define HMC_P2P_TAG_BITS              16
#define HMC_P2P_RANK_BITS             24
#define HMC_P2P_CONTEXT_BITS          16

#define HMC_P2P_TAG_BITS_OFFSET     (HMC_P2P_CONTEXT_BITS + HMC_P2P_RANK_BITS)
#define HMC_P2P_RANK_BITS_OFFSET    (HMC_P2P_CONTEXT_BITS)
#define HMC_P2P_CONTEXT_BITS_OFFSET 0

#define HMC_P2P_MAX_TAG     ((((uint64_t)1) << HMC_P2P_TAG_BITS) - 1)
#define HMC_P2P_MAX_RANK    ((((uint64_t)1) << HMC_P2P_RANK_BITS) - 1)
#define HMC_P2P_MAX_CONTEXT ((((uint64_t)1) << HMC_P2P_CONTEXT_BITS) - 1)

#define HMC_P2P_TAG_MASK              (HMC_P2P_MAX_TAG     << HMC_P2P_TAG_BITS_OFFSET)
#define HMC_P2P_RANK_MASK             (HMC_P2P_MAX_RANK    << HMC_P2P_RANK_BITS_OFFSET)
#define HMC_P2P_CONTEXT_MASK          (HMC_P2P_MAX_CONTEXT << HMC_P2P_CONTEXT_BITS_OFFSET)
#define HMC_P2P_TAG_SENDER_MASK       ((((uint64_t)1) << (HMC_P2P_CONTEXT_BITS + HMC_P2P_RANK_BITS)) - 1)

#define HMC_P2P_MAKE_TAG(_tag, _rank, _context_id)                     \
    ((((uint64_t) (_tag))        << HMC_P2P_TAG_BITS_OFFSET)  |        \
     (((uint64_t) (_rank))       << HMC_P2P_RANK_BITS_OFFSET) |        \
     (((uint64_t) (_context_id)) << HMC_P2P_CONTEXT_BITS_OFFSET))

#define HMC_P2P_MAKE_SEND_TAG(_tag, _rank, _context_id) HMC_P2P_MAKE_TAG(_tag, _rank, _context_id)

#define HMC_P2P_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _src, _context) \
    do {                                                                       \
        assert(_tag <= HMC_P2P_MAX_TAG);                                     \
        assert(_src <= HMC_P2P_MAX_RANK);                                    \
        assert(_context <= HMC_P2P_MAX_CONTEXT);                             \
        _ucp_tag_mask = (uint64_t)(-1);                                        \
        _ucp_tag = HMC_P2P_MAKE_TAG(_tag, _src, _context);                   \
    } while(0)

static inline int create_ep(hmc_comm_t *comm, int rank, ucp_ep_h *ep) {
    ucp_address_t *address = (ucp_address_t*)((ptrdiff_t)comm->addresses_array +
                                              comm->max_addrlen*rank);
    ucp_ep_params_t ep_params;
    ucs_status_t status;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = address;
    status = ucp_ep_create(comm->ctx->ucp_worker, &ep_params, ep);

    if (UCS_OK != status) {
        HMC_ERR("UCX returned connect error: %s", ucs_status_string(status));
        return HMC_ERROR;
    }
    return HMC_SUCCESS;
}

static inline ucp_ep_h
hmc_p2p_get_ep(hmc_comm_t *comm, int rank) {
    ucp_ep_h *ep = NULL;
    if (comm->params.comm_rank_to_ctx &&
        comm->ctx->world_eps) {
        int world_rank = comm->params.comm_rank_to_ctx(rank,
                                                       comm->params.rank_mapper_ctx);
        ep = &comm->ctx->world_eps[world_rank];
    } else {
        if (!comm->ucp_eps) {
            comm->ucp_eps = (ucp_ep_h *)calloc(comm->commsize, sizeof(ucp_ep_h));
        }
        ep = &comm->ucp_eps[rank];
    }
    assert(ep);
    if (!(*ep)) {
        create_ep(comm, rank, ep);
    }
    return *ep;
}

#define HMC_P2P_CHECK_REQ_STATUS() do {                                 \
        if (UCS_PTR_IS_ERR(req)) {                                      \
            HMC_ERR("Error in %s: tag %d; dest %d; worker_id %d; errmsg %s", \
                          tag, rank,                                    \
                          *((uint16_t *) &comm->ctx->ucp_worker),       \
                          ucs_status_string(UCS_PTR_STATUS(req)));      \
                                                                        \
            ucp_request_cancel(comm->ctx->ucp_worker, req);             \
            ucp_request_free(req);                                      \
            return HMC_ERROR;                                           \
        }                                                               \
    } while(0)

static inline void hmc_p2p_progress(struct app_context* ctx) {
    ucp_worker_progress(ctx->ucp_worker);
}
#endif
