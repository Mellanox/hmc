#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

bin_PROGRAMS = hmc_ibcast

hmc_ibcast_SOURCES=hmc_ibcast.c
hmc_ibcast_LDADD=$(libdir)/libhmc.la

CC=mpicc
CFLAGS+=-I${includedir} -std=c11 



if CUDA_ENABLED
CFLAGS += $(cuda_CFLAGS) $(cuda_CPPFLAGS)
LDFLAGS+= $(cuda_LDFLAGS) -lcudart
endif
