/*
 *  The per-record megakernel: gate, decline list, and the pointer bundle.
 *
 *  PORT_MEGAKERNEL_SCOPE.md and ~/.claude/plans/composed-puzzling-lemur.md.
 *
 *  The sweep's seven phases are one cooperative kernel per record instead of
 *  seven kernel launches per row across the whole chunk. That is not a
 *  launch-overhead play -- launches are worth ~2 s of an 87 s fold. It is the
 *  only structure in which data can stay in fast memory ACROSS ROWS, which is
 *  what modular_decomposition needs: it re-reads a whole fML column every row
 *  with no reuse inside the row, 46.8 TB at 400 x 5601, and L2 does not hold it
 *  because the other phases stream `c` through the same cache between launches
 *  (measured: 40.8 % hit at a shape whose entire live set fits in L2).
 *
 *  This header is included from C as well as CUDA, so it carries no CUDA types.
 *  The device buffers live in three .cu files that each own their own state;
 *  the bundle below is how they hand those pointers to the fused kernel, filled
 *  by one accessor per file.
 */
#ifndef RNAFOLD_MEGAKERNEL_H
#define RNAFOLD_MEGAKERNEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  Every device pointer the fused row chain touches, gathered from the three
 *  files that allocate them. Deliberately a plain struct passed BY VALUE to the
 *  kernel: it is under the 4 KB argument limit, it costs one constant-bank read
 *  per field, and it keeps ownership where it is -- no file hands out its
 *  buffers to anyone but this bundle.
 *
 *  void* stands in for the two parameter structs (cuda_param_t, cuda_param2_t),
 *  which are private to their .cu files; megakernel.cu casts them back.
 */
typedef struct {
  /* int_loop.cu */
  const void         *param;        /* cuda_param_t  */
  const char         *pair;
  const unsigned int *S;
  const unsigned int *hccc;
  const unsigned char *up_int;      /* NULL unless a record carries a depot */
  int                *my_c;
  const size_t       *tri_off_H;
  const size_t       *row_off_H;
  const size_t       *hc_off_H;
  int                *new_e;
  int                *energy_min2;
  float               lxc;
  int                 TerminalAU;
  int                 ninio2;

  /* hp_mb_loop.cu */
  const void         *param2;       /* cuda_param2_t */
  const char         *pair2;
  const short        *S2;
  const char         *sequence;
  const unsigned int *hccc_mb;
  const unsigned int *hccc_mbenc;
  const unsigned int *hccc_any;
  const unsigned int *hccc_gu;
  const size_t       *hc2_off_H;
  const size_t       *seq_off_H;
  const int          *len_H;
  const int          *salt_loop;
  const char         *up_ml_ok;
  const int          *up_hp;        /* NULL unless a record carries a depot */
  int                *energy_hp_row;
  int                *energy_mb_row;
  int                *energy_3p00_row;
  char               *gate_row;
  int                *energy_stack_row;   /* noLP only; refused in v1 */
  int                *cc;
  int                *cc1;

  /* modular_decomposition.cu */
  int                *fml_i;
  int                *fml_j;
  int                *dml;
  int                *dml1;
  int                *fml_prev;
  int                *energy_min;
  int                *fm2;          /* circular only; refused in v1 */

  /* int16 fML (RNA_FML_INT16), stage 1. All five are NULL when the gate is
   * off, which is exactly how the standalone kernels signal it -- the cells
   * switch on `fml_row` being non-NULL, so passing them unconditionally is
   * correct on both paths. The packed triangle REPLACES fml_j rather than
   * accompanying it; fml_row is the current row in full int32, which is what
   * fmli reads before the row is final. */
  int                *fml_row;
  short              *fml_j16;
  int                *fml_b;
  const size_t       *base_off_H;
  const size_t       *colb_off;

  /* Stages 1b and 2: the on-chip geometry, decided by the host because it is
   * what the shared-memory budget allows. cw_cols == 0 means both are off and
   * the fused kernel strides cells exactly as stage 0 did. */
  int                 cw_cols;      /* columns each block OWNS for the sweep */
  int                 cw_on;        /* stage 1b: the 32-row `c` ring         */
  int                 corner_k;     /* stage 2: fML entries cached per column */

  /* Stage 3, the one-row skew. `split` is how many blocks run the c chain;
   * 0 means one undivided grid on the stage 0-2 schedule. The halves own
   * different numbers of columns because each partitions the span among its
   * own blocks, and `bar` is four counters of device memory -- two per half --
   * for the barrier that covers one half only. */
  int                 split;
  int                 cw_cols_m;    /* columns per block in the fML half */
  unsigned int       *bar;
  int                 own_mask;     /* debug: which phases use ownership */

  /* device.cu -- the chunk's row tables, indexed by sweep row */
  const size_t       *rt_size;
  const size_t       *rt_side;
  const int          *rt_ih;
} rnafold_mk_ptrs_t;

/* One accessor per owning file. Valid between init_gpu*() and teardown_gpu*(). */
void int_loop_mk_ptrs(rnafold_mk_ptrs_t *p);
void hp_mb_mk_ptrs(rnafold_mk_ptrs_t *p);
void md_mk_ptrs(rnafold_mk_ptrs_t *p);

/*
 *  RNA_MEGAKERNEL=1 -- fuse the row chain. Default 0: the existing per-phase
 *  path stays the control every sha is compared against, and the fallback for
 *  everything the fused kernel refuses.
 */
int rnafold_megakernel(void);

/*
 *  Records in flight (RNA_MK_RECORDS, or AUTO). Exposed because it is the
 *  dial that decides whether the fused kernel beats the per-phase path at all.
 */
int rnafold_megakernel_records_in_flight(int total_blocks, int length);

/*
 *  Why this batch cannot use the fused kernel, or NULL if it can.
 *
 *  The refusal list is not a formality. Each item below is a real conditional
 *  in the kernels that v1 does not reproduce, and a megakernel that silently
 *  mishandled one would return a plausible structure rather than fail. There is
 *  no room for this decision in engine.c -- gate 2 answers "can the CUDA backend
 *  do this at all", and there is no way there to say "supported, but not by the
 *  megakernel" -- so it lives here and the caller falls back to the old sweep.
 */
const char *rnafold_megakernel_refuse(int nfiles, int circ, int gquad, int nolp,
                                      int uniq_ML, int depot, int dangles,
                                      int continuous_flow);

/*
 *  Fold `count` records, `G` of them resident at once, one cooperative launch
 *  each. Returns 0 on success, -1 if the launch geometry does not fit (in which
 *  case the caller must use the old sweep).
 */
int rnafold_megakernel_sweep(const int nfiles, const int *slots, const int count,
                             const int turn, const int length, const int *len_H_host,
                             const int noGUclosure,
                             const int TerminalAU, const int ninio2, const float lxc);

#ifdef __cplusplus
}
#endif

#endif /* RNAFOLD_MEGAKERNEL_H */
