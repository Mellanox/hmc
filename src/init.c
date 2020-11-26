/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include "config.h"
#include "hmc.h"
#include "hmc_mcast.h"
#include "hmc_p2p.h"
#include <limits.h>
#include <ucm/api/ucm.h>

int hmc_verbose_level = 0;
int hmc_probe_ip_over_ib(char* ib_dev_list, struct sockaddr_storage *addr);
int create_ah(hmc_comm_t *comm)
{
    struct ibv_ah_attr ah_attr = {
        .is_global     = 1,
        .grh           = {.sgid_index = 0},
        .dlid          = comm->mcast_lid,
        .sl            = DEF_SL,
        .src_path_bits = DEF_SRC_PATH_BITS,
        .port_num      = comm->ctx->ib_port};

    memcpy(ah_attr.grh.dgid.raw, &comm->mgid, sizeof(ah_attr.grh.dgid.raw));
    comm->mcast.ah = ibv_create_ah(comm->ctx->pd, &ah_attr);
    if (!comm->mcast.ah) {
        HMC_ERR("Failed to create AH");
        return HMC_ERROR;
    }
    return HMC_SUCCESS;
}

int clean_comm(hmc_comm_t *comm)
{
    int ret, i;
    HMC_VERBOSE(comm->ctx, 3, "Cleaning HMC comm: %p, id %d, mlid %x",
                comm, comm->comm_id, comm->mcast_lid);

    if (comm->mcast.qp) {
        ret = ibv_detach_mcast(comm->mcast.qp, &comm->mgid, comm->mcast_lid);
        if (ret) {
            HMC_ERR("Couldn't detach QP, ret %d, errno %d", ret, errno);
            return HMC_ERROR;
        }
    }

    if (comm->mcast.qp) {
        ret = ibv_destroy_qp(comm->mcast.qp);
        if (ret) {
            HMC_ERR("Failed to destroy QP %d", ret);
            return HMC_ERROR;
        }
    }

    if (comm->rcq) {
        ret = ibv_destroy_cq(comm->rcq);
        if (ret) {
            HMC_ERR("COuldn't destroy rcq");
            return HMC_ERROR;
        }
    }

    if (comm->scq) {
        ret = ibv_destroy_cq(comm->scq);
        if (ret) {
            HMC_ERR("Couldn't destroy scq");
            return HMC_ERROR;
        }
    }

    if (comm->grh_mr) {
        ret = ibv_dereg_mr(comm->grh_mr);
        if (ret) {
            HMC_ERR("Couldn't destroy grh mr");
            return HMC_ERROR;
        }
    }
    if (comm->grh_buf)
        free(comm->grh_buf);

    if (comm->pp)
        free(comm->pp);

    if (comm->pp_mr) {
        ret = ibv_dereg_mr(comm->pp_mr);
        if (ret) {
            HMC_ERR("Couldn't destroy pp mr");
            return HMC_ERROR;
        }
    }

    if (comm->cu_stage_buf) {
        hmc_cuda_host_free(comm->cu_stage_buf, comm->ctx);
    }
    if (comm->pp_buf)
        free(comm->pp_buf);

    if (comm->call_rwr)
        free(comm->call_rwr);

    if (comm->call_rsgs)
        free(comm->call_rsgs);

    if (comm->mcast.ah) {
        ret = ibv_destroy_ah(comm->mcast.ah);
        if (ret) {
            HMC_ERR("Couldn't destroy ah");
            return HMC_ERROR;
        }
    }

    if (comm->mcast_lid) {
        ret = fini_mcast_group(comm->ctx, comm);
        if (ret) {
            HMC_ERR("COuldn't leave mcast group");
            return HMC_ERROR;
        }
    }

    if (comm->ctx->config.print_nack_stats) {
        HMC_VERBOSE(comm->ctx, 0,"comm_id %d, comm_size %d, comm->psn %d, rank %d,"
                    " nacks counter %d, n_prep_rel_mr %d, n_mcast_rel %d",
                    comm->comm_id, comm->commsize, comm->psn, comm->rank,
                    comm->nacks_counter, comm->n_prep_reliable_mr, comm->n_mcast_reliable);
    }
    pthread_mutex_destroy(&comm->lock);
    free(comm);
    return HMC_SUCCESS;
}

int clean_ctx(struct app_context *ctx)
{
    int i;
    ucs_status_t status;
    void *close_req, *tmp;
    HMC_VERBOSE(ctx, 2, "Cleaning HMC ctx: %p", ctx);

    if (ctx->world_eps) {
        for (i=0; i<ctx->params.world_size; i++) {
            if (!ctx->world_eps[i]) {
                continue;
            }
            close_req = ucp_ep_close_nb(ctx->world_eps[i], UCP_EP_CLOSE_MODE_FLUSH);
            if (UCS_PTR_IS_ERR(close_req)) {
                HMC_ERR("failed to start ep close, ep %p", ctx->world_eps[i]);
            }
            status = UCS_PTR_STATUS(close_req);
            if (status != UCS_OK) {
                while (status != UCS_OK) {
                    ucp_worker_progress(ctx->ucp_worker);
                    status = ucp_request_check_status(close_req);
                }
                ucp_request_free(close_req);
            }
        }
        tmp = malloc(ctx->params.world_size);
        ctx->params.allgather(tmp, tmp, 1, ctx->params.oob_context);
        free(tmp);
    }
    if (ctx->rcache) {
        ucs_rcache_destroy(ctx->rcache);
    }

    ucp_worker_destroy(ctx->ucp_worker);
    ucp_cleanup(ctx->ucp_context);

    if (ctx->pd) {
        ibv_dealloc_pd(ctx->pd);
    }

    rdma_destroy_id(ctx->id);
    rdma_destroy_event_channel(ctx->channel);
    ucs_mpool_cleanup(&ctx->p2p_reqs_pool, 1);
    if (ctx->params.mt_enabled) {
        assert(ucs_list_is_empty(&ctx->pending_nacks_list));
    }
    free(ctx);
    return HMC_SUCCESS;
}

int setup_mcast(hmc_comm_t *comm)
{
    uint16_t mlid;
    if (setup_mcast_group(comm->ctx, comm, comm->comm_id, &mlid) != HMC_SUCCESS) {
        return HMC_ERROR;
    }
    comm->mcast_lid = mlid;
    return HMC_SUCCESS;
}

int hmc_init_qps(struct app_context *ctx, hmc_comm_t *comm)
{
    struct ibv_qp_init_attr qp_init_attr;

    memset(&qp_init_attr, 0, sizeof(qp_init_attr));

    qp_init_attr.qp_type = IBV_QPT_UD;
    qp_init_attr.send_cq = comm->scq;
    qp_init_attr.recv_cq = comm->rcq;
    qp_init_attr.sq_sig_all = 0;

    qp_init_attr.cap.max_send_wr     = comm->ctx->config.sx_depth;
    qp_init_attr.cap.max_recv_wr     = comm->ctx->config.rx_depth;
    qp_init_attr.cap.max_inline_data = comm->ctx->config.sx_inline;
    qp_init_attr.cap.max_send_sge    = comm->ctx->config.sx_sge;
    qp_init_attr.cap.max_recv_sge    = comm->ctx->config.rx_sge;

    comm->mcast.qp = ibv_create_qp(ctx->pd, &qp_init_attr);
    if (!comm->mcast.qp) {
        HMC_ERR("Failed to create mcast qp, errno %d", errno);
        return HMC_ERROR;
    }
    comm->max_inline = qp_init_attr.cap.max_inline_data;
    return HMC_SUCCESS;
}

int hmc_setup_qps(struct app_context *ctx, hmc_comm_t *comm)
{
    struct ibv_port_attr port_attr;
    uint16_t pkey;

    ibv_query_port(ctx->ctx, ctx->ib_port, &port_attr);
    for (ctx->pkey_index = 0; ctx->pkey_index < port_attr.pkey_tbl_len;
         ++ctx->pkey_index) {
        ibv_query_pkey(ctx->ctx, ctx->ib_port, ctx->pkey_index, &pkey);
        if (pkey == DEF_PKEY)
            break;
    }

    if (ctx->pkey_index >= port_attr.pkey_tbl_len) {
        ctx->pkey_index = 0;
        ibv_query_pkey(ctx->ctx, ctx->ib_port, ctx->pkey_index, &pkey);
        if (pkey) {
            HMC_VERBOSE(ctx, 1, "Cannot find default pkey 0x%04x on port %d, using index 0 pkey:0x%04x",
                       DEF_PKEY, ctx->ib_port, pkey);
        } else {
            HMC_ERR("Cannot find valid PKEY");
            return HMC_ERROR;
        }
    }

    struct ibv_qp_attr attr = {.qp_state = IBV_QPS_INIT,
                               .pkey_index = ctx->pkey_index,
                               .port_num = ctx->ib_port,
                               .qkey = DEF_QKEY};

    if (ibv_modify_qp(comm->mcast.qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
        HMC_ERR("Failed to move mcast qp to INIT, errno %d", errno);
        return HMC_ERROR;
    }


    if (ibv_attach_mcast(comm->mcast.qp, &comm->mgid, comm->mcast_lid)) {
        HMC_ERR("Failed to attach QP to the mcast group, errno %d", errno);
        return HMC_ERROR;
    }

    /* Ok, now cycle to RTR on everyone */
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(comm->mcast.qp, &attr, IBV_QP_STATE)) {
        HMC_ERR("Failed to modify QP to RTR, errno %d", errno);
        return HMC_ERROR;
    }

    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = DEF_PSN;
    if (ibv_modify_qp(comm->mcast.qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        HMC_ERR("Failed to modify QP to RTS, errno %d", errno);
        return HMC_ERROR;
    }

    /* Create the address handle */
    if (HMC_SUCCESS != create_ah(comm)) {
        HMC_ERR("Failed to create adress handle");
        return HMC_ERROR;
    }

    return HMC_SUCCESS;
}

void exchange_ucp_addresses(hmc_comm_t *comm) {
    uint16_t *addrlens = malloc(comm->commsize*sizeof(uint16_t));
    uint16_t addrlen = (uint16_t)comm->ctx->ucp_addrlen;
    uint16_t max_addrlen = 0;
    int i;

    comm->ctx->params.allgather(&addrlen, addrlens, sizeof(uint16_t),
                                comm->params.comm_oob_context);
    for (i=0; i<comm->commsize; i++) {
        if (addrlens[i] > max_addrlen) max_addrlen = addrlens[i];
    }
    free(addrlens);
    comm->max_addrlen = (size_t)max_addrlen;
    comm->addresses_array = malloc(comm->commsize*comm->max_addrlen);
    comm->ctx->params.allgather(comm->ctx->worker_address,
                                comm->addresses_array, comm->max_addrlen,
                                comm->params.comm_oob_context);
}


/* Set up things that would only change per-communicator and could/would need to
 * be comm for performacne */
hmc_comm_t* setup_comm(struct app_context *ctx, hmc_comm_params_t *params) {
    hmc_comm_t* comm = (hmc_comm_t*)
        malloc(sizeof(hmc_comm_t) + sizeof(struct pp_packet*)*(ctx->config.wsize-1));
    int i;
    memset(comm, 0, sizeof(hmc_comm_t));

    pthread_mutex_init(&comm->lock, NULL);
    ucs_list_head_init(&comm->bpool);
    ucs_list_head_init(&comm->pending_q);

    if (!(params->field_mask & (HMC_COMM_PARAMS_FIELD_COMM_SIZE        |
                                HMC_COMM_PARAMS_FIELD_COMM_RANK        |
                                HMC_COMM_PARAMS_FIELD_COMM_RANK_TO_CTX |
                                HMC_COMM_PARAMS_FIELD_RANK_MAPPER_CTX  |
                                HMC_COMM_PARAMS_FIELD_COMM_OOB_CONTEXT))) {
        HMC_ERR("HMC Comm Params error: comm_size, comm_rank, comm_oob_context, rank_mapper"
                " have to be provided");
        goto error;
    }
    comm->params.comm_size        = params->comm_size;
    comm->params.comm_rank        = params->comm_rank;
    comm->params.comm_rank_to_ctx = params->comm_rank_to_ctx;
    comm->params.rank_mapper_ctx  = params->rank_mapper_ctx;
    comm->params.comm_oob_context = params->comm_oob_context;

    if (!comm->params.comm_oob_context) {
        HMC_ERR("Runtime communicator is not provided for setup_comm");
        goto error;
    }
    comm->wsize   = ctx->config.wsize;
    comm->max_eager = ctx->config.max_eager;
    comm->comm_id = 0;
    comm->ctx     = ctx;
    comm->grh_buf = (char *)malloc(GRH_LENGTH * sizeof(char));
    memset(comm->grh_buf, 0, GRH_LENGTH);
    comm->grh_mr = ibv_reg_mr(ctx->pd, comm->grh_buf, GRH_LENGTH,
                              IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!comm->grh_mr) {
        HMC_ERR("Could not register memory for GRH, errno %d", errno);
        goto error;
    }

    comm->rcq = ibv_create_cq(ctx->ctx, ctx->config.rx_depth, NULL, NULL, 0);
    if (!comm->rcq) {
        HMC_ERR("Could not create recv cq, rx_depth %d, errno %d",
                  ctx->config.rx_depth, errno);
        goto error;
    }

    comm->scq = ibv_create_cq(ctx->ctx, ctx->config.sx_depth, NULL, NULL, 0);
    if (!comm->scq) {
        HMC_ERR("Could not create send cq, sx_depth %d, errno %d",
                  ctx->config.sx_depth, errno);
        goto error;
    }

    if (params->comm_size < 2) {
        HMC_ERR("Comm size < 2");
        goto error;
    }

    if (!(params->comm_rank >= 0 && params->comm_rank < params->comm_size)) {
        HMC_ERR("Incorrect value of params->rank");
        goto error;
    }
    comm->rank             = params->comm_rank;
    comm->commsize         = params->comm_size;
    comm->max_per_packet   = ctx->mtu - GRH_LENGTH;
    comm->last_acked       = comm->last_psn = 0;
    comm->racks_n          = comm->sacks_n  = 0;
    comm->child_n          = comm->parent_n = 0;
    comm->dummy_packet.psn = UINT32_MAX;

    for (i=0; i< comm->wsize; i++) {
        comm->r_window[i] = &comm->dummy_packet;
    }

    comm->cu_stage_buf = NULL;
    if (setup_mcast(comm) != HMC_SUCCESS) {
        goto error;
    }

    if (hmc_init_qps(ctx, comm) != HMC_SUCCESS) {
        goto error;
    }

    if (hmc_setup_qps(ctx, comm) != HMC_SUCCESS) {
        goto error;
    }

    HMC_VERBOSE(ctx, 3, "Creating HMC comm: %p, id %d, mlid %x",
                comm, comm->comm_id, comm->mcast_lid);

    size_t page_size = (size_t)sysconf(_SC_PAGE_SIZE);
    int buf_size = ctx->mtu;

    // Comm receiving buffers.
    posix_memalign((void**)&comm->call_rwr, page_size, sizeof(struct ibv_recv_wr) *
                   ctx->config.rx_depth);
    posix_memalign((void**)&comm->call_rsgs, page_size, sizeof(struct ibv_sge) *
                   ctx->config.rx_depth * 2);

    comm->pending_recv = 0;
    comm->buf_n = ctx->config.rx_depth * 2;

    posix_memalign((void**) &comm->pp_buf, page_size,
                   buf_size * comm->buf_n);
    memset(comm->pp_buf, 0, buf_size * comm->buf_n);
    comm->pp_mr = ibv_reg_mr(ctx->pd, comm->pp_buf, buf_size * comm->buf_n,
                             IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!comm->pp_mr) {
        HMC_ERR("Could not register pp_buf mr, errno %d", errno);
        goto error;
    }

    posix_memalign((void**) &comm->pp, page_size, sizeof(struct pp_packet) * comm->buf_n);
    for (i = 0; i < comm->buf_n; i++) {
        comm->pp[i].buf = (uintptr_t) comm->pp_buf + i * buf_size;
        comm->pp[i].context = 0;
        ucs_list_add_tail(&comm->bpool, &comm->pp[i].super);
    }

    comm->mcast.swr.wr.ud.ah          = comm->mcast.ah;
    comm->mcast.swr.num_sge           = 1;
    comm->mcast.swr.sg_list           = & comm->mcast.ssg;
    comm->mcast.swr.opcode            = IBV_WR_SEND_WITH_IMM;
    comm->mcast.swr.wr.ud.remote_qpn  = MULTICAST_QPN;
    comm->mcast.swr.wr.ud.remote_qkey = DEF_QKEY;
    comm->mcast.swr.next              = NULL;

    for (i = 0; i < ctx->config.rx_depth; i++) {
        comm->call_rwr[i].sg_list         = &comm->call_rsgs[2 * i];
        comm->call_rwr[i].num_sge         = 2;
        comm->call_rwr[i].wr_id           = HMC_BCASTRECV_WR;
        comm->call_rsgs[2 * i].length     = GRH_LENGTH;
        comm->call_rsgs[2 * i].addr       = (uintptr_t)comm->grh_buf;
        comm->call_rsgs[2 * i].lkey       = comm->grh_mr->lkey;
        comm->call_rsgs[2 * i + 1].lkey   = comm->pp_mr->lkey;
        comm->call_rsgs[2 * i + 1].length = comm->max_per_packet;
    }

    if (HMC_SUCCESS != post_recv_buffers(comm)) {
        goto error;
    }
    memset(comm->parents,  0, sizeof(comm->parents));
    memset(comm->children, 0, sizeof(comm->children));
    comm->nacks_counter            = 0;
    comm->tx                       = 0;
    comm->reliable_ready_multiroot = 0;
    comm->n_prep_reliable_mr       = 0;
    comm->n_prep_reliable          = 0;
    comm->n_mcast_reliable         = 0;

    exchange_ucp_addresses(comm);
    HMC_VERBOSE(ctx, 10, "HMC comm created SUCCESS");
    return comm;
error:
    clean_comm(comm);
    return NULL;
}

static void* hmc_rcache_ucs_get_reg_data(void *region) {
    return (void*)((ptrdiff_t)region + sizeof(ucs_rcache_region_t));
}

static void* hmc_rcache_ucs_get_reg_start_addr(void *region) {
    return (void*)(((ucs_rcache_region_t *)region)->super.start);
}

struct ibv_mr* _hmc_mem_reg_internal(struct app_context *ctx, void *address,
                                     size_t length, ucs_rcache_region_t **region) {
    struct ibv_mr *mr = NULL;
    *region = NULL;
    if (ctx->rcache) {
        ucs_status_t status = ucs_rcache_get(ctx->rcache, address, length,
                                             PROT_READ | PROT_WRITE, NULL, region);
        if (UCS_OK == status) {
            mr = *((struct ibv_mr**)hmc_rcache_ucs_get_reg_data(*region));
        }
    } else {
        hmc_mem_register(ctx, address, length, (void**)&mr);
    }
    
    return mr;
}

void _hmc_mem_dereg_internal(struct app_context *ctx, struct ibv_mr *mr, ucs_rcache_region_t *region) {
    if (region) {
        assert(ctx->rcache);
        ucs_rcache_region_put(ctx->rcache, region);
    } else {
        hmc_mem_deregister(ctx, mr);
    }
}


#if (UCP_API_VERSION >= UCP_VERSION(1,5))
static ucs_status_t hmc_rcache_ucs_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                               void *arg, ucs_rcache_region_t *rregion,
                                               uint16_t flags)
#else
static ucs_status_t hmc_rcache_ucs_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                               void *arg, ucs_rcache_region_t *rregion)
#endif
{
    struct app_context *ctx = (struct app_context*)context;
    void *addr = (void*)rregion->super.start;
    size_t length = (size_t)(rregion->super.end - rregion->super.start);
    void **mr = hmc_rcache_ucs_get_reg_data(rregion);
    if (HMC_SUCCESS != hmc_mem_register(ctx, addr, length, mr)) {
        HMC_ERR("mem_reg failed in rcache, addr %p, len %zd", addr, length);
        return UCS_ERR_NO_MESSAGE;
    }
    HMC_VERBOSE(ctx, 10, "RCACHE: mem_reg, addr %p, length %zd, rregion %p",
                addr, length, rregion);
    return UCS_OK;
}

static void hmc_rcache_ucs_mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                                        ucs_rcache_region_t *rregion) {
    struct app_context *ctx = (struct app_context*)context;
    HMC_VERBOSE(ctx, 10, "RCACHE: mem_dereg, rregion %p", rregion);
    hmc_mem_deregister(ctx, *((void**)hmc_rcache_ucs_get_reg_data(rregion)));
}

static void hmc_rcache_ucs_dump_region_cb(void *context, ucs_rcache_t *rcache,
                                           ucs_rcache_region_t *rregion, char *buf,
                                           size_t max)
{
    snprintf(buf, max, "TODO");
}

static ucs_rcache_ops_t hmc_rcache_ucs_ops = {
    .mem_reg     = hmc_rcache_ucs_mem_reg_cb,
    .mem_dereg   = hmc_rcache_ucs_mem_dereg_cb,
    .dump_region = hmc_rcache_ucs_dump_region_cb
};

static int setup_rcache(struct app_context *ctx) {
    ucs_rcache_params_t rcache_params;
    ucs_status_t status;

    rcache_params.region_struct_size = sizeof(ucs_rcache_region_t)+sizeof(struct ibv_mr *);
    rcache_params.alignment          = 64;
    rcache_params.max_alignment      = (size_t)sysconf(_SC_PAGE_SIZE);
#if HAVE_STRUCT_UCS_RCACHE_PARAMS_UCM_EVENTS
    rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED |
        UCM_EVENT_MEM_TYPE_FREE;
#endif
    rcache_params.ucm_event_priority = 1000;
    rcache_params.context            = (void*)ctx;
    rcache_params.ops                = &hmc_rcache_ucs_ops;

    status = ucs_rcache_create(&rcache_params, "HMC_Rcache",
                               ucs_stats_get_root(), &ctx->rcache);
    if (UCS_OK != status) {
        ctx->rcache = NULL;
        return HMC_ERROR;
    }
    return HMC_SUCCESS;
}
/* Low level setup to keep this standalone. Real bcast should just need a mcast
 * group and some buffers and such */
/* Basically, sets global-level values and gets ready to do a standalone bcast
 */

ucs_mpool_ops_t hmc_req_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

static void hmc_req_init(void* _req) {
    hmc_p2p_req_t *req = (hmc_p2p_req_t*)_req;
    req->comm = NULL;
}

static int setup_ucx(struct app_context *ctx) {
    ucp_params_t params;
    ucp_worker_params_t worker_params;
    ucp_ep_params_t ep_params;
    ucp_config_t *config;
    ucs_status_t status;
    char *var;
    HMC_VERBOSE(ctx, 3, "ucx_p2p_init");

    status = ucp_config_read("HMC", NULL, &config);
    if (UCS_OK != status) {
        return HMC_ERROR;
    }

    if (ctx->devname) {
        if (UCS_OK != ucp_config_modify(config, "NET_DEVICES", ctx->devname)) {
            return HMC_ERROR;
        }
    }
    params.field_mask      = UCP_PARAM_FIELD_FEATURES |
                             UCP_PARAM_FIELD_REQUEST_SIZE |
                             UCP_PARAM_FIELD_REQUEST_INIT |
                             UCP_PARAM_FIELD_REQUEST_CLEANUP |
                             UCP_PARAM_FIELD_TAG_SENDER_MASK  |
                             UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    params.features        = UCP_FEATURE_TAG;
    params.request_size    = sizeof(hmc_p2p_req_t);
    params.request_init    = hmc_req_init;
    params.request_cleanup = NULL;
    params.tag_sender_mask = HMC_P2P_TAG_SENDER_MASK;
    params.estimated_num_eps = 8; //TODO - how to set ?
    //TODO
/* #if HAVE_DECL_UCP_PARAM_FIELD_ESTIMATED_NUM_PPN */
    /* params.estimated_num_ppn = 1; */
    /* params.field_mask       |= UCP_PARAM_FIELD_ESTIMATED_NUM_PPN; */
/* #endif */
    status = ucp_init(&params, config, &ctx->ucp_context);
    ucp_config_release(config);

    if (UCS_OK != status) {
        return HMC_ERROR;
    }

    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = ctx->params.mt_enabled ? UCS_THREAD_MODE_MULTI :
        UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(ctx->ucp_context, &worker_params,
            &ctx->ucp_worker);
    if (UCS_OK != status) {
        return HMC_ERROR;
    }

    {
        ucp_worker_attr_t attr;
        attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
        status = ucp_worker_query(ctx->ucp_worker, &attr);
        if (UCS_OK != status) {
            HMC_ERR("Failed to query UCP worker thread level");
            return HMC_ERROR;
        }

        if (ctx->params.mt_enabled && attr.thread_mode != UCS_THREAD_MODE_MULTI) {
            /* UCX does not support multithreading, disqualify current PML for now */
            /* TODO: we should let OMPI to fallback to THREAD_SINGLE mode */
            HMC_ERR("UCP worker does not support MPI_THREAD_MULTIPLE");
            return HMC_ERROR;
        }
    }
    status = ucp_worker_get_address(ctx->ucp_worker,
                                    &ctx->worker_address,
                                    &ctx->ucp_addrlen);
    if(UCS_OK != status) {
        HMC_ERR("Failed to get local worker address");
        return HMC_ERROR;
    }
    if (ctx->params.world_size >= 0) {
        ctx->world_eps = (ucp_ep_h*)calloc(ctx->params.world_size, sizeof(ucp_ep_h));
    }
    return HMC_SUCCESS;
}

static void hmc_dummy_runtime_progress(void){}

struct app_context* setup_ctx(hmc_ctx_params_t *params, hmc_context_config_t *hmc_cfg)
{
    struct ibv_device **device_list, *dev;
    struct ibv_port_attr port_attr;
    struct ibv_device_attr device_attr;
    struct app_context *ctx = NULL;
    struct sockaddr_storage ip_oib_addr, dst_addr;
    char *devname = NULL, *var = NULL;
    int num_devices, i, is_ipv4 = 0;
    int active_mtu, max_mtu;

    dev = NULL;
    ctx = (struct app_context *)calloc(1, sizeof(struct app_context));
    if (!ctx) {
        goto error;
    }

    if (!(params->field_mask & (HMC_CTX_PARAMS_FIELD_WORLD_SIZE |
                                HMC_CTX_PARAMS_FIELD_ALLGATHER  |
                                HMC_CTX_PARAMS_FIELD_OOB_CONTEXT))) {
        HMC_ERR("HMC Params error: world_size, allgather, oob_context have to be provided");
        goto error;
    }
    ctx->params.world_size  = params->world_size;
    ctx->params.allgather   = params->allgather;
    ctx->params.oob_context = params->oob_context;
    ctx->params.mt_enabled  = 0;

    if (params->field_mask & HMC_CTX_PARAMS_FIELD_RUNTIME_PROGRESS) {
        ctx->params.runtime_progress = params->runtime_progress;
    } else {
        ctx->params.runtime_progress = hmc_dummy_runtime_progress;
    }
    if (params->field_mask & HMC_CTX_PARAMS_FIELD_MT_ENABLED) {
        ctx->params.mt_enabled = params->mt_enabled;
    }

    memcpy(&ctx->config, hmc_cfg, sizeof(*hmc_cfg));
    device_list = ibv_get_device_list(&num_devices);
    if (!device_list) {
        HMC_ERR("No IB devices available");
        goto error;
    }

    if (ctx->config.devices) {
        ctx->devname = ctx->config.devices;
    } else {
        dev = device_list[0];
        devname = (char *)ibv_get_device_name(dev);
        ctx->devname = malloc(strlen(devname)+16);
        strcpy(ctx->devname, devname);
        strcat(ctx->devname, ":1");
    }
    ibv_free_device_list(device_list);

    if (HMC_SUCCESS != hmc_probe_ip_over_ib(ctx->devname, &ip_oib_addr)) {
        HMC_ERR("HMC: Failed to get ipoib interface for devname %s", ctx->devname);
        goto error;
    }

    if (HMC_SUCCESS != setup_ucx(ctx)) {
        HMC_ERR("HMC: failed to setup UCP context");
        goto error;
    }

    is_ipv4 = (ip_oib_addr.ss_family == AF_INET) ? 1 : 0;

    char addrstr[128];
    struct sockaddr_in  *in_src_addr = (struct sockaddr_in*)&ip_oib_addr;
    struct sockaddr_in6 *in6_src_addr = (struct sockaddr_in6*)&ip_oib_addr;

    inet_ntop((is_ipv4) ? AF_INET : AF_INET6,
              &in_src_addr->sin_addr, addrstr, sizeof(addrstr) - 1);
    HMC_VERBOSE(ctx, 5, "HMC: devname %s, ipoib %s", ctx->devname, addrstr);

    ctx->channel = rdma_create_event_channel();
    if (!ctx->channel) {
        HMC_ERR("rdma_create_event_channel failed, errno %d", errno);
        goto error;
    }

    memset(&dst_addr, 0, sizeof(struct sockaddr_storage));
    dst_addr.ss_family = is_ipv4 ? AF_INET : AF_INET6;
    if (rdma_create_id(ctx->channel, &ctx->id, NULL, RDMA_PS_UDP)) {
        HMC_ERR("HMC: Failed to create rdma id, errno %d", errno);
        goto error;
    }

    struct rdma_cm_event *revent;
    rdma_resolve_addr(ctx->id, (struct sockaddr *)&ip_oib_addr, (struct sockaddr *) &dst_addr, 1000);
    if (rdma_get_cm_event(ctx->channel, &revent) < 0 ||
        revent->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
        HMC_ERR("HMC: Failed to resolve rdma addr, errno %d", errno);
        goto error;
    }
    rdma_ack_cm_event(revent);
    ctx->ctx = ctx->id->verbs;
    ctx->pd = ibv_alloc_pd(ctx->ctx);
    if (!ctx->pd) {
        HMC_ERR("HMC: Failed to allocate pd");
        goto error;
    }

    char *port = strstr(ctx->devname, ":");
    if (!port) {
        HMC_ERR("HMC: incorrect devname specified %s", ctx->devname);
        goto error;
    }
    ctx->ib_port = atoi(port+1);

    /* Determine MTU */
    if (ibv_query_port(ctx->ctx, ctx->ib_port, &port_attr)) {
        HMC_ERR("Couldn't query port in ctx create, errno %d", errno);
        goto error;
    }

    if (port_attr.max_mtu == IBV_MTU_256)
        max_mtu = 256;
    if (port_attr.max_mtu == IBV_MTU_512)
        max_mtu = 512;
    if (port_attr.max_mtu == IBV_MTU_1024)
        max_mtu = 1024;
    if (port_attr.max_mtu == IBV_MTU_2048)
        max_mtu = 2048;
    if (port_attr.max_mtu == IBV_MTU_4096)
        max_mtu = 4096;

    if (port_attr.active_mtu == IBV_MTU_256)
        active_mtu = 256;
    if (port_attr.active_mtu == IBV_MTU_512)
        active_mtu = 512;
    if (port_attr.active_mtu == IBV_MTU_1024)
        active_mtu = 1024;
    if (port_attr.active_mtu == IBV_MTU_2048)
        active_mtu = 2048;
    if (port_attr.active_mtu == IBV_MTU_4096)
        active_mtu = 4096;

    ctx->mtu = active_mtu;

    if (port_attr.max_mtu < port_attr.active_mtu) {
        HMC_VERBOSE(ctx, 1, "Port active MTU (%d) is smaller than port max MTU (%d)",
                    active_mtu, max_mtu);
    }
    if (ibv_query_device(ctx->ctx, &device_attr)) {
        HMC_ERR("Failed to query device in ctx create, errno %d", errno);
        goto error;
    }

    HMC_VERBOSE(ctx, 5, "MTU %d, MAX QP WR: %d, max sqr_wr: %d, max cq: %d, max cqe: %d",
                ctx->mtu, device_attr.max_qp_wr, device_attr.max_srq_wr,
                device_attr.max_cq, device_attr.max_cqe);

    ctx->max_qp_wr = device_attr.max_qp_wr;

    ctx->rcache = NULL;
    if (HMC_SUCCESS != setup_rcache(ctx)) {
        HMC_ERR("Failed to setup rcache");
        goto error;
    }
    if (ctx->config.memtype_cache_enabled) {
        if (UCS_OK != ucs_memtype_cache_create(&ctx->memtype_cache)) {
            HMC_VERBOSE(ctx, 1,"could not create memtype cache for mem_type allocations, "
                        "fallback to default memtype check");
            ctx->config.memtype_cache_enabled = 0;
        }
    }
    //TODO init all locks and lists

    ucs_status_t status;
    status = ucs_mpool_init(&ctx->p2p_reqs_pool, 0,
                            sizeof(hmc_p2p_req_t),
                            0, 128, 16, UINT_MAX,
                            &hmc_req_mpool_ops, "test_reqs");
    if (status != UCS_OK) {
        HMC_ERR("FAILED to init mpool");
        goto error;
    }

    status = ucs_mpool_init(&ctx->coll_reqs_pool, 0,
                            sizeof(hmc_coll_req_t),
                            0, 128, 16, UINT_MAX,
                            &hmc_req_mpool_ops, "coll_reqs");

    if (status != UCS_OK) {
        HMC_ERR("FAILED to init mpool");
        goto error;
    }

    if (ctx->params.mt_enabled) {
        ucs_list_head_init(&ctx->pending_nacks_list);
    }
    HMC_VERBOSE(ctx, 1, "HMC SETUP complete: ctx %p", ctx);
#ifdef CUDA_ENABLED
    if (HMC_SUCCESS != hmc_cuda_init(&ctx->cuda)) {
        HMC_VERBOSE(ctx, 1, "CUDA support is not available");
    }
#endif
    return ctx;

error:
    if (ctx->pd) ibv_dealloc_pd(ctx->pd);
    if (ctx->id) rdma_destroy_id(ctx->id);
    if (ctx->channel) rdma_destroy_event_channel(ctx->channel);
    free(ctx);
    return NULL;
}

HMC_EXPORT
void hmc_progress(hmc_ctx_h _ctx) {
    struct app_context *ctx = (struct app_context *)_ctx;
    ucp_worker_progress(ctx->ucp_worker);
    if (ctx->params.mt_enabled) {
        hmc_progress_ctx(ctx);
    }
}
