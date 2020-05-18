# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.

sources =             \
	src/init.c        \
	src/hmc.c         \
	src/mcast.c       \
	src/mcast.h       \
	src/reliable.h    \
    src/hmc_p2p.h     \
	src/hmc_mcast.h   \
	src/hmc_verbose.h \
	src/hmc_config.c  \
    src/ipoib.c       \
	src/hmc_cuda.c

inst_headers = hmc.h

comp_inst = libhmc.la
libhmc_ladir = $(includedir)
nobase_dist_libhmc_la_HEADERS = $(inst_headers)
lib_LTLIBRARIES = $(comp_inst)

libhmc_la_SOURCES  = $(sources) $(inst_headers)
libhmc_la_LDFLAGS  = -version-info $(SOVERSION) --no-undefined -fvisibility=hidden
libhmc_la_CPPFLAGS = $(AM_CPPFLAGS) -fvisibility=hidden

install-exec-hook:
	cp -f $(HMC_TOP_BUILDDIR)/libhmc.la $(libdir)/
	perl -pi -e "s/installed=no/installed=yes/" $(libdir)/libhmc.la
