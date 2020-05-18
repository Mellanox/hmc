# -*- shell-script -*-
#
# Copyright (C) Mellanox Technologies Ltd. 2015.  ALL RIGHTS RESERVED.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# HMC_CHECK_UCX(prefix, [action-if-found], [action-if-not-found])
# --------------------------------------------------------
# check if UCX support can be found.  sets prefix_{CPPFLAGS, 
# LDFLAGS, LIBS} as needed and runs action-if-found if there is
# support, otherwise executes action-if-not-found
AC_DEFUN([HMC_CHECK_UCX],[
    AC_ARG_WITH([ucx],
        [AC_HELP_STRING([--with-ucx(=DIR)],
             [Build with Unified Communication X library support])])
    HMC_CHECK_WITHDIR([ucx], [$with_ucx], [include/ucp/api/ucp.h])
    AC_ARG_WITH([ucx-libdir],
        [AC_HELP_STRING([--with-ucx-libdir=DIR],
             [Search for Unified Communication X libraries in DIR])])
    HMC_CHECK_WITHDIR([ucx-libdir], [$with_ucx_libdir], [libucp.*])

    hmc_check_ucx_$1_save_CPPFLAGS="$CPPFLAGS"
    hmc_check_ucx_$1_save_LDFLAGS="$LDFLAGS"
    hmc_check_ucx_$1_save_LIBS="$LIBS"

    AS_IF([test "$with_ucx" != "no"],
          [AS_IF([test ! -z "$with_ucx" -a "$with_ucx" != "yes"],
                 [
                    hmc_check_ucx_dir="$with_ucx"
                    hmc_check_ucx_libdir="$with_ucx/lib"
                 ],[
                    hmc_check_ucx_dir=/usr
                 ])
           AS_IF([test ! -z "$with_ucx_libdir" -a "$with_ucx_libdir" != "yes"],
                 [hmc_check_ucx_libdir="$with_ucx_libdir"])

           hmc_check_ucx_extra_libs=""
           AS_IF([test "x$hmc_check_ucx_libdir" != "x"], [hmc_check_ucx_extra_libs="-L$hmc_check_ucx_libdir"])

           HMC_CHECK_PACKAGE([$1],
                              [ucp/api/ucp.h],
                              [ucp],
                              [ucp_cleanup],
                              [$hmc_check_ucx_extra_libs -luct -lucm -lucs],
                              [$hmc_check_ucx_dir],
                              [$hmc_check_ucx_libdir],
                              [hmc_check_ucx_happy="yes"],
                              [hmc_check_ucx_happy="no"])],
          [hmc_check_ucx_happy="no"])

    CPPFLAGS="$hmc_check_ucx_$1_save_CPPFLAGS"
    LDFLAGS="$hmc_check_ucx_$1_save_LDFLAGS"
    LIBS="$hmc_check_ucx_$1_save_LIBS"

    AC_MSG_CHECKING(for UCX version compatibility)
    AC_REQUIRE_CPP
    old_CFLAGS="$CFLAGS"
    CFLAGS="$CFLAGS -I$hmc_check_ucx_dir/include"
    AC_COMPILE_IFELSE(
            [AC_LANG_PROGRAM([[#include <uct/api/version.h>]],
                [[
                ]])],
            [hmc_ucx_version_ok="yes"],
            [hmc_ucx_version_ok="no"])

    AC_MSG_RESULT([$hmc_ucx_version_ok])

    AS_IF([test "$hmc_check_ucx_happy" = "yes"],
          AC_CHECK_HEADER([$hmc_check_ucx_dir/include/ucs/memory/rcache.h],
                          [AC_DEFINE([HAVE_UCS_MEMORY_RCACHE_H], 1, [have ucs/memory/rcache.h])]
			              [AC_CHECK_MEMBERS([struct ucs_rcache_params.ucm_events],[],[],[[#include <ucs/memory/rcache.h>]])]
			              ,[AC_CHECK_MEMBERS([struct ucs_rcache_params.ucm_events],[],[],[[#include <ucs/sys/rcache.h>]])])
          AC_CHECK_HEADER([$hmc_check_ucx_dir/include/ucs/memory/memtype_cache.h],
                          [AC_DEFINE([HAVE_UCS_MEMORY_MEMTYPE_CACHE_H], 1, [have ucs/memory/memtype_cache.h])])
          AC_CHECK_TYPES([ucm_mem_type_t], [], [], [[#include <ucm/api/ucm.h>]])
          AC_CHECK_DECLS([UCP_PARAM_FIELD_ESTIMATED_NUM_PPN], [], [], [#include <ucp/api/ucp.h>])
         )

    CFLAGS=$old_CFLAGS

    AS_IF([test "$hmc_ucx_version_ok" = "no"], [hmc_check_ucx_happy="no"])

    AS_IF([test "$hmc_check_ucx_happy" = "yes"],
          [$2],
          [AS_IF([test ! -z "$with_ucx" -a "$with_ucx" != "no"],
                 [AC_MSG_ERROR([UCX support requested but not found.  Aborting])])
           $3])
])
