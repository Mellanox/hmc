# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
define([hmc_ver_major], 1)
define([hmc_ver_minor], 0)
define([hmc_ver_micro], esyscmd([sh -c "git rev-list HEAD | wc -l | sed -e 's/ *//g' | xargs -n1 printf"]))
define([hmc_git_sha], esyscmd([sh -c "git rev-parse --short HEAD"]))

# This is the API version (see libtool library versioning)
# http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
# current:rev:age
define([libhmc_so_version], 1:0:0)

AC_INIT([hmc], [hmc_ver_major.hmc_ver_minor])
AC_CONFIG_HEADERS([config.h])

HMC_TOP_BUILDDIR="`pwd`"
AC_SUBST(HMC_TOP_BUILDDIR)
cd "$srcdir"
HMC_TOP_SRCDIR="`pwd`"
AC_SUBST(HMC_TOP_SRCDIR)
cd "$HMC_TOP_BUILDDIR"

AM_INIT_AUTOMAKE([1.10 foreign tar-ustar silent-rules subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AM_MAINTAINER_MODE
AC_CONFIG_MACRO_DIR([m4])


define([hmc_prefix], [/opt/mellanox/hmc])
AC_PREFIX_DEFAULT([hmc_prefix])

RPM_RELEASE=1
MAJOR_VERSION=hmc_ver_major
MINOR_VERSION=hmc_ver_minor
MICRO_VERSION=hmc_ver_micro
HMC_GIT_SHA=hmc_git_sha
VERSION=$MAJOR_VERSION.$MINOR_VERSION.$MICRO_VERSION
SOVERSION=libhmc_so_version
AC_SUBST(MAJOR_VERSION)
AC_SUBST(MINOR_VERSION)
AC_SUBST(MICRO_VERSION)
AC_SUBST(RPM_RELEASE)
AC_SUBST(SOVERSION)
AC_SUBST(HMC_GIT_SHA)
AC_MSG_RESULT([git sha: $hmc_git_sha])

CFLAGS_save="$CFLAGS"
AC_PROG_CC
AC_PROG_CXX
AM_PROG_AS
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
CFLAGS="$CFLAGS_save"
build_modules=""
AC_ARG_WITH([ofed],
			[AC_HELP_STRING([--with-ofed(=DIR)], [Build OpenFabrics support, adding DIR/include, DIR/lib, and DIR/lib64 to the search path for headers and libraries])],
			[],
			[with_ofed=/usr]
			)

ofed_CPPFLAGS="$CFLAGS -I$with_ofed/include"
ofed_LDFLAGS="$LDFLAGS -L$with_ofed/lib64 -L$with_ofed/lib"
AC_SUBST([ofed_CPPFLAGS])
AC_SUBST([ofed_LDFLAGS])

AC_CHECK_HEADERS([infiniband/verbs.h],,[AC_MSG_ERROR([ibverbs header files not found])])
AC_CHECK_LIB([ibverbs], [ibv_get_device_list], [], [AC_MSG_ERROR([libibverbs not found])])
AC_CHECK_LIB([rdmacm], [rdma_create_event_channel], [], AC_MSG_ERROR([librdmacm-devel not found]))
CFLAGS="$CFLAGS -std=gnu99"
AC_ARG_ENABLE([debug],
	AS_HELP_STRING([--enable-debug], [Enable extra debugging code (default is NO).]),
	[], [enable_debug=no])

if test $enable_debug = yes; then
	AC_DEFINE([ENABLE_DEBUG], [1], [Enable debugging code])
    CFLAGS="$CFLAGS -O0 -g"
else
	CFLAGS="$CFLAGS -O3 -DNDEBUG"
fi

#
# UCX support
#
ucx_enabled=no
HMC_CHECK_UCX([ucx],[ucx_happy="yes"],[ucx_happy="no"])
AS_IF([test "$ucx_happy" = "yes"],
      [
       ucx_enabled=yes
       AC_SUBST([ucx_CFLAGS])
       AC_SUBST([ucx_CPPFLAGS])
       AC_SUBST([ucx_LDFLAGS])
       AC_SUBST([ucx_LIBS])
       AC_DEFINE([UCX_ENABLED],[],[UCX is available])
       ]
      )
AM_CONDITIONAL([UCX_ENABLED], [test "x$ucx_enabled" = "xyes"])
CFLAGS="$CFLAGS $ucx_CFLAGS"
CPPFLAGS="$CPPFLAGS $ucx_CPPFLAGS"
LDFLAGS="$LDFLAGS $ucx_LDFLAGS $ucx_LIBS"

cuda_enabled=no
HMC_CHECK_CUDA([cuda],[cuda_happy="yes"],[cuda_happy="no"])
AS_IF([test "$cuda_happy" = "yes"],
      [
       cuda_enabled=yes
       AC_SUBST([cuda_CFLAGS])
       AC_SUBST([cuda_CPPFLAGS])
       AC_SUBST([cuda_LDFLAGS])
       AC_SUBST([cuda_LIBS])
       AC_MSG_RESULT([cuda_CFLAGS=$cuda_CFLAGS; cuda_CPPFLAGS=$cuda_CPPFLAGS; cuda_LDFLAGS=$cuda_LDFLAGS; cuda_LIBS=$cuda_LIBS ])
       AC_DEFINE([CUDA_ENABLED],[],[specifies if we going to use CUDA bcol])
       ]
      )
AM_CONDITIONAL([CUDA_ENABLED], [test "x$cuda_enabled" = "xyes"])
CFLAGS="$CFLAGS $cuda_CFLAGS"
CPPFLAGS="$CPPFLAGS $cuda_CPPFLAGS"
LDFLAGS="$LDFLAGS "

AC_CONFIG_FILES([
                 Makefile
                 example/Makefile
                 ])
AC_OUTPUT
