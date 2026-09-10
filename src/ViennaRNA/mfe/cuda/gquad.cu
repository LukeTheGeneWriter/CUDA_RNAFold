/*
 * G-quadruplex support, stage G0: get c_gq onto the device, unchanged.
 *
 * This file does NOT make -g work. It carries the batch's c_gq matrices to the
 * device and provides the lookup the two future call sites will use, plus the
 * bar that proves the transfer is lossless. -g stays declined by the routing
 * guard until G2. See PORT_GQUAD_SPEC.md "SCOPED AGAINST THE PORT".
 *
 * WHY THIS IS A SEPARATE STAGE. The device work for -g is exactly two sites:
 * one MIN2 in the multibranch decomposition and three bounded (p,q) sweeps in
 * the interior loop. Both READ c_gq and nothing writes it. If the table arrives
 * on the device intact, a wrong answer later is a recursion bug; if it does not,
 * every later stage debugs the wrong thing. So the table crosses first, with its
 * own bar.
 *
 * THE LAYOUT, AND WHY IT IS UPSTREAM'S OWN
 *
 * vrna_smx_csr(int) is already exactly the shape we want (sparse_mx.h:11):
 *
 *     v[]        the energies
 *     col_idx[]  the j of each entry
 *     row_idx[]  ROW OFFSETS -- row i occupies [row_idx[i], row_idx[i+1])
 *
 * so there is no format conversion at all, only a flatten across the batch in
 * the same style as tri_off_H/row_off_H. Two per-record offset tables:
 *
 *     gq_ent_off_H[H]  where record H's entries start in v/col
 *     gq_row_off_H[H]  where record H's row-offset table starts in rowoff
 *
 * TWO CORRECTIONS TO PORT_GQUAD_SPEC.md, both from reading sparse_mx.c:
 *
 *  1. The spec calls vrna_smx_csr_int_get() "a binary search over a row's
 *     sorted column indices". It is a LINEAR SCAN (sparse_mx.c:70-72). So the
 *     per-lookup cost is O(entries in row) -- measured max 12-45, roughly
 *     constant in n -- not O(log). gq_get() below mirrors that scan exactly,
 *     because matching upstream's arithmetic is worth more than being clever.
 *
 *  2. row_idx is NOT usable until a lazy prefix sum has run. The get macro
 *     carries a `dirty` flag and rewrites row_idx IN PLACE on first call
 *     (sparse_mx.c:61-65). A freshly built c_gq is dirty, so reading row_idx
 *     directly yields per-row COUNTS, not offsets -- silently, and they look
 *     plausible. gq_flatten_host() forces the fixup through the public getter
 *     before touching anything.
 *
 *     That lazy mutation is also a THREAD-SAFETY hazard of the same family as
 *     upstream Defect B: two threads calling get() on one dirty matrix race on
 *     the prefix sum. It is not a live bug for us -- c_gq is per record and one
 *     thread owns a record -- but it is worth reporting upstream alongside B.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/params/basic.h"
#include "ViennaRNA/datastructures/array.h"
#include "ViennaRNA/datastructures/sparse_mx.h"
#include "ViennaRNA/datastructures/dp_matrices.h"

#include "stub2.h"

#ifndef INF
#define INF 10000000
#endif


/* ---------------------------------------------------------------- device */

static int          *d_gq_v        = NULL;   /* energies, batch-flattened     */
static unsigned int *d_gq_col      = NULL;   /* j of each entry               */
static unsigned int *d_gq_rowoff   = NULL;   /* per-record row offset tables  */
static size_t       *d_gq_ent_off  = NULL;   /* [nfiles+1] into v/col         */
static size_t       *d_gq_row_off  = NULL;   /* [nfiles+1] into rowoff        */
static int           g_gq_active   = 0;      /* did anything get uploaded?    */

/* Sizes kept for the read-back bar and for free(). */
static size_t        g_gq_entries  = 0;
static size_t        g_gq_rowslots = 0;
static int           g_gq_nfiles   = 0;


/*
 * The lookup, mirroring vrna_smx_csr_int_get() line for line.
 *
 * Returns INF for a cell with no quadruplex, which is what every upstream call
 * site passes as its default and then tests for.
 */
__device__ int
gq_get(const int                   H,
       const unsigned int          i,
       const unsigned int          j,
       const int          *__restrict__ v,
       const unsigned int *__restrict__ col,
       const unsigned int *__restrict__ rowoff,
       const size_t       *__restrict__ ent_off,
       const size_t       *__restrict__ row_off)
{
  const size_t        ro    = row_off[H];
  const size_t        base  = ent_off[H];
  const unsigned int  s     = rowoff[ro + i];
  const unsigned int  d     = rowoff[ro + i + 1];
  unsigned int        p;

  for (p = s; p < d; p++)
    if (col[base + p] == j)
      return v[base + p];

  return INF;
}


/* The read-back bar's kernel: evaluate gq_get() over an explicit list of
 * (H,i,j) probes. A list rather than the whole triangle, because the host has
 * to hold the expected values anyway and the interesting cells are few. */
__global__ void
gq_probe_kernel(const size_t              n,
                const int    *__restrict__ pH,
                const unsigned int *__restrict__ pi,
                const unsigned int *__restrict__ pj,
                int          *__restrict__ out,
                const int          *__restrict__ v,
                const unsigned int *__restrict__ col,
                const unsigned int *__restrict__ rowoff,
                const size_t       *__restrict__ ent_off,
                const size_t       *__restrict__ row_off)
{
  const size_t m = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (m >= n)
    return;

  out[m] = gq_get(pH[m], pi[m], pj[m], v, col, rowoff, ent_off, row_off);
}


/* ------------------------------------------------- device-side entry points */
/*
 * These take PLAIN ARRAYS, not fold compounds. The walk over c_gq lives in
 * gquad.c instead, for a build reason worth writing down: the ViennaRNA headers
 * carry no `extern "C"` guards, and a .cu is compiled as C++, so every RNAlib
 * call from here comes out name-mangled and fails at link
 * (`undefined reference to vrna_alloc(unsigned long)`). The other four .cu files
 * never hit this because none of them calls a single RNAlib function -- they use
 * malloc directly and take everything else as arguments. Splitting host from
 * device keeps that property rather than working around it.
 */

extern "C" void
rnafold_gq_free(void)
{
  if (d_gq_v)        cudaFree(d_gq_v);
  if (d_gq_col)      cudaFree(d_gq_col);
  if (d_gq_rowoff)   cudaFree(d_gq_rowoff);
  if (d_gq_ent_off)  cudaFree(d_gq_ent_off);
  if (d_gq_row_off)  cudaFree(d_gq_row_off);

  d_gq_v       = NULL;
  d_gq_col     = NULL;
  d_gq_rowoff  = NULL;
  d_gq_ent_off = NULL;
  d_gq_row_off = NULL;
  g_gq_active  = 0;
  g_gq_entries = 0;
  g_gq_rowslots = 0;
  g_gq_nfiles  = 0;
}


extern "C" int
rnafold_gq_active(void)
{
  return g_gq_active;
}


/* Upload already-flattened tables. Returns 1 on success, -1 on failure. */
extern "C" int
rnafold_gq_upload_flat(const int           nfiles,
                       const size_t        entries,
                       const size_t        rowslots,
                       const int          *h_v,
                       const unsigned int *h_col,
                       const unsigned int *h_rowoff,
                       const size_t       *h_ent_off,
                       const size_t       *h_row_off)
{
  /* cudaMalloc(0) is legal but returns a pointer nothing may deref; a batch can
   * legitimately have gquad on and no quadruplex anywhere, so round up to 1. */
  const size_t vb = sizeof(int) * (entries ? entries : 1);
  const size_t cb = sizeof(unsigned int) * (entries ? entries : 1);
  const size_t rb = sizeof(unsigned int) * (rowslots ? rowslots : 1);
  const size_t ob = sizeof(size_t) * (nfiles + 1);

  rnafold_gq_free();

  if ((cudaMalloc((void **)&d_gq_v, vb) != cudaSuccess) ||
      (cudaMalloc((void **)&d_gq_col, cb) != cudaSuccess) ||
      (cudaMalloc((void **)&d_gq_rowoff, rb) != cudaSuccess) ||
      (cudaMalloc((void **)&d_gq_ent_off, ob) != cudaSuccess) ||
      (cudaMalloc((void **)&d_gq_row_off, ob) != cudaSuccess)) {
    fprintf(stderr, "gquad.cu                 cudaMalloc failed for c_gq "
                    "(%zu entries, %zu row slots)\n", entries, rowslots);
    rnafold_gq_free();
    return -1;
  }

  if (((entries) && (cudaMemcpy(d_gq_v, h_v, sizeof(int) * entries,
                                cudaMemcpyHostToDevice) != cudaSuccess)) ||
      ((entries) && (cudaMemcpy(d_gq_col, h_col, sizeof(unsigned int) * entries,
                                cudaMemcpyHostToDevice) != cudaSuccess)) ||
      ((rowslots) && (cudaMemcpy(d_gq_rowoff, h_rowoff, sizeof(unsigned int) * rowslots,
                                 cudaMemcpyHostToDevice) != cudaSuccess)) ||
      (cudaMemcpy(d_gq_ent_off, h_ent_off, ob, cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpy(d_gq_row_off, h_row_off, ob, cudaMemcpyHostToDevice) != cudaSuccess)) {
    fprintf(stderr, "gquad.cu                 cudaMemcpy failed for c_gq\n");
    rnafold_gq_free();
    return -1;
  }

  g_gq_entries  = entries;
  g_gq_rowslots = rowslots;
  g_gq_nfiles   = nfiles;
  g_gq_active   = 1;

  fprintf(stderr, "gquad.cu                 c_gq uploaded: %d records, %zu "
                  "entries, %zu row slots, %.1f KiB\n",
          nfiles, entries, rowslots, (vb + cb + rb + 2.0 * ob) / 1024.0);

  return 1;
}


/*
 * THE BAR. Evaluate gq_get() on the device over a list of (H,i,j) and hand the
 * answers back, so the caller can compare them against vrna_smx_csr_int_get()
 * on the host. Returns 0 on success, -1 on failure; `out` must hold `n` ints.
 */
extern "C" int
rnafold_gq_probe(const size_t        n,
                 const int          *pH,
                 const unsigned int *pi,
                 const unsigned int *pj,
                 int                *out)
{
  int          *dH = NULL, *dout = NULL;
  unsigned int *di = NULL, *dj = NULL;
  int          rc = -1;

  if ((n == 0) || (!g_gq_active))
    return -1;

  if ((cudaMalloc((void **)&dH, sizeof(int) * n) != cudaSuccess) ||
      (cudaMalloc((void **)&di, sizeof(unsigned int) * n) != cudaSuccess) ||
      (cudaMalloc((void **)&dj, sizeof(unsigned int) * n) != cudaSuccess) ||
      (cudaMalloc((void **)&dout, sizeof(int) * n) != cudaSuccess))
    goto out;

  if ((cudaMemcpy(dH, pH, sizeof(int) * n, cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpy(di, pi, sizeof(unsigned int) * n, cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpy(dj, pj, sizeof(unsigned int) * n, cudaMemcpyHostToDevice) != cudaSuccess))
    goto out;

  {
    const int block = 256;
    const int grid  = (int)((n + block - 1) / block);

    gq_probe_kernel<<<grid, block>>>(n, dH, di, dj, dout,
                                     d_gq_v, d_gq_col, d_gq_rowoff,
                                     d_gq_ent_off, d_gq_row_off);

    if (cudaPeekAtLastError() != cudaSuccess)
      goto out;

    if (cudaDeviceSynchronize() != cudaSuccess)
      goto out;
  }

  if (cudaMemcpy(out, dout, sizeof(int) * n, cudaMemcpyDeviceToHost) != cudaSuccess)
    goto out;

  rc = 0;

out:
  if (dH)   cudaFree(dH);
  if (di)   cudaFree(di);
  if (dj)   cudaFree(dj);
  if (dout) cudaFree(dout);

  return rc;
}
