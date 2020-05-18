# -*- shell-script -*-
#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# HMC_CHECK_CUDA(prefix, [action-if-found], [action-if-not-found])
# --------------------------------------------------------
# check if CUDA support can be found.  sets prefix_{CPPFLAGS,
# LDFLAGS, LIBS} as needed and runs action-if-found if there is
# support, otherwise executes action-if-not-found
AC_DEFUN([HMC_CHECK_CUDA],[
    AC_ARG_WITH([cuda], [AC_HELP_STRING([--with-cuda(=DIR)], [Build CUDA support])], [with_cuda_given=yes], [with_cuda=guess])

    hmc_check_cuda_$1_save_CPPFLAGS="$CPPFLAGS"
    hmc_check_cuda_$1_save_LDFLAGS="$LDFLAGS"
    hmc_check_cuda_$1_save_LIBS="$LIBS"

    AS_IF([test "$with_cuda" != "no"],
          [AS_IF([test ! -z "$with_cuda" -a "x$with_cuda" != "xyes" -a "x$with_cuda" != "xguess"],
                 [
                    hmc_check_cuda_dir="$with_cuda"
                    hmc_check_cuda_libdir="$with_cuda/lib64"
                 ])
           AS_IF([test ! -z "$with_cuda_libdir" -a "$with_cuda_libdir" != "yes"],
                 [hmc_check_cuda_libdir="$with_cuda_libdir"])

           hmc_check_cuda_extra_libs=""
           AS_IF([test "x$hmc_check_cuda_libdir" != "x"], [hmc_check_cuda_extra_libs="-L$hmc_check_cuda_libdir"])

           HMC_CHECK_PACKAGE([$1],
                              [cuda_runtime.h],
                              [cudart],
                              [cudaMemcpyAsync],
                              [$hmc_check_cuda_extra_libs],
                              [$hmc_check_cuda_dir],
                              [$hmc_check_cuda_libdir],
                              [hmc_check_cuda_happy="yes"],
                              [hmc_check_cuda_happy="no"])],
          [hmc_check_cuda_happy="no"])


    CPPFLAGS="$hmc_check_cuda_$1_save_CPPFLAGS"
    LDFLAGS="$hmc_check_cuda_$1_save_LDFLAGS"
    LIBS="$hmc_check_cuda_$1_save_LIBS"

    AS_IF([test "$hmc_check_cuda_happy" = "yes"],
          [$2],
          [AS_IF([test ! -z "$with_cuda" -a "$with_cuda" != "no" -a "$with_cuda_given" == "yes"],
                 [AC_MSG_ERROR([CUDA support requested but not found.  Aborting])])
           $3])
])
