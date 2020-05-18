/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#ifndef HMC_VERBOSE_H
#define HMC_VERBOSE_H
#include <stdio.h>

#define HMC_ERR(fmt, arg...) fprintf(stderr, "HMC ERROR: [%d]: "fmt"\n", getpid(), ## arg)

#define HMC_VERBOSE(_ctx, _level, _fmt, _arg...)             \
    do {                                                \
        if ((_ctx)->config.verbose >= _level) {                     \
            fprintf(stdout, "HMC [%d]: "_fmt"\n", getpid(), ## _arg); \
        }                                               \
    } while (0)


#if ENABLE_DEBUG
#define HMC_VERBOSE_DBG_ONLY(_ctx, _level, _fmt, _arg...) HMC_VERBOSE(_ctx, _level, _fmt, ## _arg)
#else
#define HMC_VERBOSE_DBG_ONLY(_ctx, _level, _fmt, _arg...)
#endif


#endif /*BCOL_CC_VERBOSE_H*/
