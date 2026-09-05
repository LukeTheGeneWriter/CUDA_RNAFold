#
# assert() control: release builds compile them out, the verification bar keeps
# them.
#
# NDEBUG is defined NOWHERE in this build system as it stands -- not in
# configure.ac, not in ac_rna_features.m4, not in NVCC_FLAGS -- so every
# assert() in the tree is live in every build ever produced. Measured cost on a
# Colab L4 ladder (120 x 2000 nt, outputs identical, 0.9% run-to-run spread):
#
#     branch                 stock     +NDEBUG    assert cost
#     SRB  08081e6           7.12 s     6.08 s        14.6%
#     CFB  823e363           7.56 s     6.23 s        17.6%
#
# On CUDA that is not surprising: __assertfail in a kernel affects register
# allocation and can inhibit optimisation, and eight hot kernels carry an
# assert on the row index alone.
#
# NOT a blanket define, deliberately. These asserts earn their keep -- they have
# caught real bugs (the hp_mb_3p_kernel out-of-bounds read in 00d1e07, the
# unpack() trap), and during this port they turned two device-side failures into
# precise file-and-line diagnostics instead of wrong answers:
#
#     Assertion `ij>=0 && (size_t)ij < tri_off_H[H+1]-tri_off_H[H]' failed
#     Assertion `my_c[tri_off_H[H]+ij] == INF' failed
#
# Hence a split: users get the speed, the bar keeps the checks.
#
#   default            NDEBUG defined, asserts compiled out
#   --enable-asserts   NDEBUG not defined; what every verification run uses
#
# RNA_ROW_VERIFY and the other self-checks are independent of this and keep
# working either way -- they are runtime-gated, not assert-gated.

AC_DEFUN([RNA_ENABLE_ASSERTS], [

  RNA_ADD_FEATURE([asserts],
                  [keep assert() live (release builds compile them out)],
                  [no])

  ## A dedicated substituted variable rather than appending to NVCC_FLAGS.
  ## Appending there silently did nothing: NVCC_FLAGS is set inside
  ## RNA_ENABLE_CUDA, and the append landed on the wrong side of that macro's
  ## expansion, so -DNDEBUG reached the C compiler and never nvcc. Caught by
  ## checking config.status for both flag sets rather than trusting one.
  NVCC_ASSERT_FLAGS=""

  RNA_FEATURE_IF_DISABLED([asserts],[
    AX_APPEND_FLAG([-DNDEBUG], [RNA_CPPFLAGS])
    NVCC_ASSERT_FLAGS="-DNDEBUG"
  ])

  RNA_FEATURE_IF_ENABLED([asserts],[
    AC_MSG_NOTICE([assert() kept live -- this is a verification build, not a release build])
  ])

  AC_SUBST(NVCC_ASSERT_FLAGS)
  AM_CONDITIONAL(VRNA_AM_SWITCH_ASSERTS, test "x$enable_asserts" = "xyes")
])
