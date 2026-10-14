//WBL 11 Dec 2017 $Revision: 1.89 $ CUDA GGGP ViennaRNA-2.3.0 rf/rf/
//Helper for fill_arrays.c 
//based on fill_arrays_loop.c r1.10

//Plan is to have all CUDA code here.
//Atleast whilst only convert modular decomposition to CUDA

//LAW 14 Jul 2026 added WBL diff till 2026
//WBL  5 Jul 2026 see if can get to compile with CUDA 13.*
//WBL 15 May 2023 add use_cuda (removed with choose_gpu, cleanup C8 2026-10-12)
//WBL  4 Mar 2018 add optional pause to fake persistence mode
//WBL 17 Feb 2018 simplify GPU startup message and send to stderr
//WBL  2 Jan 2018 
//WBL 31 Dec 2017 try without atomicMin by use of one block per column
//WBL 30 Dec 2017 for debug make modular_decomposition_ij as in modular_decomposition.c r1.6
//WBL 29 Dec 2017 ../rf_cuda try tuning


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>   /* SIZE_MAX: a VRAM budget with no ceiling */
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <time.h>

#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/utils/strings.h"
#include "ViennaRNA/utils/structures.h"
#include "ViennaRNA/params/default.h"
#include "ViennaRNA/datastructures/basic.h"
#include "ViennaRNA/datastructures/basic.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/params/basic.h"
#include "ViennaRNA/constraints/basic.h"
#include "ViennaRNA/constraints/hard.h"
#include "ViennaRNA/constraints/soft.h"
#include "ViennaRNA/eval/gquad.h"
#include "ViennaRNA/mfe/gquad.h"
#include "ViennaRNA/structured_domains.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/loops/all.h"
#include "ViennaRNA/mfe/global.h"

//https://devtalk.nvidia.com/default/topic/1012969/cuda-programming-and-performance/texture-unit-in-pascal-architecture/2
#ifdef MIN2
#undef MIN2
#endif
#define MIN2(x,y) min(x,y)

//#include <assert.h>
//#ifdef STUB
#include "stub2.h"

/* The device helpers below live in md_dev.h; see the header. */
#include "md_dev.h"

//#include "stub.h"
//#endif /*STUB*/

// System includes
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

/* vrna_cuda_quiet(): engine.c. On when a library caller (the Python binding) has
 * asked for the routine diagnostics below to be left out; errors still print. */
extern "C" int vrna_cuda_quiet(void);

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h> //Commented out in 2026, nothing here is used
//#include <helper_cuda.h> //Commented out in 2026 to get started

//BLOCK_SIZE must be power of two 32 or greater
#define BLOCK_SIZE 64

//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrchk2(ans,first) { gpuAssert((ans), __FILE__, __LINE__,first); }
inline void gpuAssert(cudaError_t code, const char *file, const int line, const bool first=false, const bool abort=true)
{
   if (code != cudaSuccess) 
   {
     fprintf(stderr,"CUDA error: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
     if(first)
     fprintf(stderr,"CUDA error: on first kernel.\nWas %s compiled with correct compute levels?\n", file);
     if (abort) exit(code);
   }
}

/* choose_gpu() and use_cuda (WBL, May 2023; from the CUDA matrixMul sample) were
 * deleted in the cleanup (C8): nothing called them, and their jobs -- the device
 * probe and the driver warm-up -- are vrna_cuda_devices() and RNAfold's reach-cost
 * probe. Its commented-out --device=N is kept as a future item (multi-GPU); the
 * code is at the `pre-cleanup` tag. */

void int_Memcpy(int* out, const int* in, const size_t size, const cudaMemcpyKind dir, const int error_report) { // 32-bit signed integer overflow bug fix
  const cudaError_t error = cudaMemcpy(out, in, size*sizeof(int), dir);
  if (error != cudaSuccess)  {
    printf("cudaMemcpy(%p,%p,%zu,%d) returned error %s (code %d), %s line(%d)\n", // 32-bit signed integer overflow bug fix
	   out,in,
	   size*sizeof(int),dir,cudaGetErrorString(error), error, __FILE__, error_report);
    exit(EXIT_FAILURE);
  }
}

// Async twin of int_Memcpy for use inside captured CUDA graph regions.
// NB: a cudaSuccess return here only means the copy was recorded (or, outside
// capture, enqueued) successfully -- NOT that it will succeed when the graph
// is later replayed. Real data/runtime errors from replayed graph work only
// surface at the cudaStreamSynchronize() after cudaGraphLaunch().
void int_MemcpyAsync(int* out, const int* in, const size_t size, const cudaMemcpyKind dir, cudaStream_t stream, const int error_report) {
  const cudaError_t error = cudaMemcpyAsync(out, in, size*sizeof(int), dir, stream);
  if (error != cudaSuccess)  {
    printf("cudaMemcpyAsync(%p,%p,%zu,%d) returned error %s (code %d), %s line(%d)\n",
	   out,in,
	   size*sizeof(int),dir,cudaGetErrorString(error), error, __FILE__, error_report);
    exit(EXIT_FAILURE);
  }
}

// New Jul 2026: page-locked host allocation for buffers that are the
// direct source/destination of a cudaMemcpy every row (DMLi/DMLi1/DMLi2
// here, and fill_arrays.c's energy_min/energy_hp_row/energy_mb_row/
// energy_3p00_row/new_C) -- pageable host memory forces the CUDA driver to
// stage every one of those transfers through an internal pinned buffer
// first; allocating these ourselves as pinned skips that copy. Hard-fails
// on allocation failure via gpuErrchk, matching every other CUDA
// allocation in this codebase -- see compute_gpu_usable_bytes()'s
// safety_margin, tightened alongside this change specifically to keep
// pinned-memory pressure down at the largest batch sizes rather than
// adding a fallback-to-pageable path here.
PUBLIC int*
cuda_host_alloc_ints(const size_t n) {
  int* p;
  gpuErrchk( cudaHostAlloc((void **) &p, n*sizeof(int), cudaHostAllocDefault) );
  return p;
}

// Byte-sized twin of the above, for row buffers holding flags rather than
// energies -- see gate_row in hp_mb_loop.cu. Pinned for the same reason the
// int version is: it is copied back once per sweep row.
PUBLIC char*
cuda_host_alloc_bytes(const size_t n) {
  char* p;
  gpuErrchk( cudaHostAlloc((void **) &p, n*sizeof(char), cudaHostAllocDefault) );
  return p;
}

PUBLIC void
cuda_host_free(void* p) {
  gpuErrchk( cudaFreeHost(p) );
}

int first = 1;

// Forward declarations so init_gpu() (below) can probe fmli_kernel's/
PUBLIC void pack_fml(const int nfiles, const int i, const int turn,
                     const int length, const size_t* size_off_H);

// modular_decomposition_kernel's occupancy before they're defined further
// down this file, and so that probe runs once, up front, rather than lazily
// at first launch like this file's other kernels get further down. That's
// deliberate, not just style: both are launched from inside a CUDA-graph
// capture region (load_fML_modular_decomposition_load_min_fML() below).
// cudaOccupancyMax*() take no stream argument and don't enqueue anything, so
// they should be legal regardless of another stream's capture state -- but
// init_gpu() already runs once per batch, well before any capture begins,
// so there's no reason to lean on that instead.
__global__ void fmli_kernel(
  const int nfiles, const int i_row, const int turn, const int length,
        int* __restrict__ fml_i,
  const int* __restrict__ fml_j,
  const int* __restrict__ fml_row,   // int16 path, NULL when off
  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
  const size_t* __restrict__ side_off_H, const size_t total,
  const int* __restrict__ i_H);   // per-record row index
// TILE = how many threads cooperate on one output cell's y-reduction; see the
// kernel's own comment further down for why that split exists at all. TILE=1
// was the pre-2026-08-22 one-thread-per-cell kernel; only TILE=32 is built now.
template <int TILE>
__global__ void modular_decomposition_kernel(
  const int nfiles, const int i_row, const int turn, const int length,
  const int* __restrict__ fml_i, const int* __restrict__ fml_j,
  const short* __restrict__ fml_j16, const int* __restrict__ fml_b,
  const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
  int* __restrict__ dml,
  int* __restrict__ fm2,          // CIRCULAR: triangular fM2_real, or NULL
  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
  const size_t* __restrict__ side_off_H, const size_t total,
  const int* __restrict__ i_H);  // per-record row index

// Block sizes for the above, chosen once in init_gpu() below instead of the
// BLOCK_SIZE=64 constant these used to hardcode (tuned against one GPU, the
// L4). Neither kernel has shared memory or a reduction tying it to a
// specific size.
static int g_block_size_fmli = 0;
static int g_block_size_md   = 0;
// How many threads cooperate on one modular_decomposition_kernel output cell
// (its TILE template parameter): 32, one warp per cell, the point of the
// 22 Aug 2026 rewrite. The other widths and RNA_MD_TILE were deleted in the
// cleanup (C6, D1): dense md is the RNA_MD_SPARSE=0 fallback and the sparse
// selftest's reference, and one width serves both.
#define MD_TILE 32

//int* d_indx; //indx no longer used
int* d_energy_min;
int* d_fml_i;  //my_fML
int* d_fml_j;  //my_fML

// int16 fml_j (RNA_FML_INT16). Allocated INSTEAD of d_fml_j when the gate is on,
// so the VRAM saving is real and measurable rather than shadowed by keeping both.
//   d_fml_j16   the offsets themselves, same triangular layout as d_fml_j
//   d_fml_b     one int32 baseline per FML_BLK consecutive entries WITHIN a
//               column -- blocks must not straddle a column boundary, because
//               the end of column j and the start of column j+1 are the two
//               extremes of the whole energy range and a straddling block would
//               have to span it
//   d_fml_row   the CURRENT row's cells in full int32. Required, not an
//               optimisation: fmli_kernel reads row i between the two writers,
//               at its pre-MIN2 value, so a cell is not final until its row
//               completes and cannot be packed before then.
short*  d_fml_j16;

int*    d_fml_b;
int*    d_fml_row;
// colb_off[j] = number of baseline slots used by columns 1..j-1, so a column's
// blocks start on a baseline boundary. A pure function of j and FML_BLK, hence
// ONE table for the whole chunk rather than one per record.
size_t* d_colb_off;
size_t* d_base_off_H;   // per-record start in d_fml_b
static size_t g_base_total = 0;
static size_t g_tri_cells  = 0;   /* the chunk's triangle cells, for RNA_TRI_CHECKSUM */
int* d_dml;  //DMLi
// GPU-resident sweep, step 1. Two row-shaped buffers the device side of the
// sweep needs once new_c_host/fml_host/fml_prev_host become kernels. They are
// allocated, INF-prefilled and maintained from here on but READ BY NOTHING
// YET: this step is deliberately behaviour-neutral, so the whole verification
// matrix can confirm byte-identical output before any host loop moves.
//
//   d_dml1      the device twin of the host's DMLi1 -- row i+1's DMLi, which
//               new_c_kernel will read as e_mb's DMLi1[j-1].
//   d_fml_prev  the device twin of the host's fml_prev -- row i's final fML,
//               which fml_scan_kernel reads one row later as fML(i+1,j).
//
// The host keeps DMLi1 by ROTATING three buffers. This does not: it takes one
// device-to-device copy per row instead (md_snapshot_dml(), below). That is a
// deliberate departure. Rotating would change the pointer baked into the
// captured CUDA-graph node every single row, and the graph currently gets away
// with 16785 cheap cudaGraphExecUpdate()s against only 3 forced
// reinstantiates; a 3.3 MB D2D copy costs ~0.18 s across a whole run and
// leaves the graph's parameters untouched. Correctness before cleverness on
// the piece that is hardest to debug.
//
// There is deliberately no d_dml2: the host allocates, prefills and rotates
// DMLi2, but nothing reads it. Confirming and removing that is a separate
// cleanup and is not bundled here.
int* d_dml1;
int* d_fml_prev;
// Staggered_Row_Batching Phase 2d: own device copies (per this codebase's
// established convention of each .cu file owning independent device state)
// of tri_off_H[]/row_off_H[] -- d_fml_j's per-H triangle-block start and
// d_energy_min/d_fml_i/d_dml's per-H row-block start, replacing the
// H-tightest H+X*nfiles convention throughout this file.
static size_t* d_tri_off_H;
static size_t* d_row_off_H;
// Staggered_Row_Batching Phase 6d: row_off_H[nfiles], the real total extent of
// the row-shaped buffers here (d_energy_min, d_dml). Cached at init_gpu() time
// -- equals the old uniform nfiles*(length+1) only while chunks are
// uniform-length, and that formula over-runs the allocation once they aren't.
static size_t  g_row_total = 0;

/* RNA_MD_SPARSE (PORT_SPARSE_MD.md), defined in md_sparse.inc */
static int  md_sparse_selftest(void);
static int  md_sparse_on(void);
static void md_sparse_alloc(const int nfiles, const size_t* row_off_H);
static void md_sparse_free(void);
static void md_sparse_md(const int nfiles, const int turn, const int* fml_i, const size_t total);
static int  md_sparse_live(void);   /* RNA_MD_SPARSE=1 and this chunk's lists exist */
static size_t md_sparse_bytes_per_file(const int length);   /* the lists, for chunk admission */
static int    md_sparse_planned(void);   /* sparse md will run: no DRAM pressure from the triangle */
/* The admission cell cap (0: none). RNAfold sets it from its driver; the library path
 * (engine.c) sets the same default itself (PORT_BUDGETS.md 9.6, decision 1). */
static unsigned long long g_chunk_cells_cap     = 0;
static int                g_chunk_cells_cap_set = 0;

// Staggered_Row_Batching Phase 4: per-row "current active width" tables for
// load_fML/fmli_kernel/modular_decomposition_kernel/load_min_fML_kernel's
// flat grids -- rebuilt+reuploaded every sweep row i (see fill_arrays_loop.c),
// unlike tri_off_H/row_off_H above which are fixed for the whole chunk. Only
// two distinct shapes needed: size_off_H is load_fML's own
// (length_H[H]-i-turn); side_off_H is shared by fmli_kernel/
// modular_decomposition_kernel/load_min_fML_kernel (length_H[H]-i-2*turn-2 --
// verified algebraically identical across all three from their pre-Phase-4
// bound-check arithmetic). Both buffers are allocated once per chunk here
// (init_gpu()) and just overwritten each row, same as d_energy_min already is.
//
// Luke's Flow Batching, fix 3: all three are row i's slots in the chunk's row
// tables (device.cu), bound by bind_row_tables(i) and never written here. The
// per-row H2Ds that used to fill them -- two of them CAPTURED INTO THE GRAPH
// from host stack tables -- are gone, so the graph now holds kernels only.
static const size_t* d_size_off_H = NULL;
static const size_t* d_side_off_H = NULL;
static const int*    d_i_H        = NULL;

static void
bind_row_tables(const int i) {
  d_size_off_H = rnafold_rowtab_size(i);
  d_side_off_H = rnafold_rowtab_side(i);
  d_i_H        = rnafold_rowtab_ih(i);
}
//int* h_dml;  //DMLi
//unsigned int mem_size_buf; //bytes in h_dml and d_dml
//int* fml_j;  //my_fML

// CUDA Graph plumbing for the load_fML -> modular_decomposition -> load_min_fML
// chain (fill_arrays_loop.c has no host CPU logic between those three calls,
// which is what makes them capturable as a single graph). graph_stream must
// stay a *blocking* stream (not cudaStreamNonBlocking): that's what makes the
// legacy-default-stream ordering rule implicitly wait for int_loop_i()'s /
// load_my_c()'s still-fully-synchronous NULL-stream work before the captured
// chain runs each iteration, with no extra synchronization code needed.
cudaStream_t    graph_stream     = 0;

/* RNA_STREAM_OVERLAP (device.cu): the stream the row's cell chain runs on. The
 * graph is CAPTURED on graph_stream either way -- capture needs a stream of its
 * own -- but the executable graph is LAUNCHED wherever the schedule wants it,
 * which is what keeps it ordered behind fml_scan(i). */
extern "C" cudaStream_t rnafold_stream_cell(void);
extern "C" cudaStream_t rnafold_stream_md(void);
extern "C" int          rnafold_stream_overlap(void);
extern "C" void         rnafold_stream_md_done(void);

/* Where the md chain's kernels are ISSUED. graph_stream while capturing (the
 * capture needs it), and -- the fix -- the md stream on the RNA_CUDA_GRAPH=0
 * path when RNA_STREAM_OVERLAP is on. That path used to issue straight onto
 * graph_stream, which is ordered against the default stream only, not against
 * the cell/md streams the rest of the row runs on: every graph-off overlap arm,
 * level 1 included, returned a wrong answer, on the control binary as well
 * (found 2026-09-17 by the overlap bar's graph-off arm). */
static cudaStream_t g_issue_stream = 0;
#define ISSUE_STREAM (g_issue_stream ? g_issue_stream : graph_stream)

/* RNA_MD_TAIL (see rnafold_md_tail()). g_md_tail_now is set by the row driver
 * for the duration of the md chain and read by modular_decomposition_cuda();
 * g_md_tail_row is the row the collapsed tail CLOSED, which is how fml_prev_i()
 * and md_snapshot_dml() learn that their work is already done -- positive
 * evidence from the code that did it, not a re-evaluation of the knob. */
static int g_md_tail_now = 0;
static int g_md_tail_row = -1;

/* ---- Blocked md, stage 3a.1 (docs/archive/PORT_MD_BLOCKING_DRIVER.md section 3.1): the rotating
 * row buffers as RINGS.
 *
 * The blocked schedule visits RB rows per column block, so a buffer that row i reads
 * to the LEFT of the column it is on -- fM2 of row i+1 at j-1 (d_dml1, through
 * new_c), and row i's own fML back in earlier column blocks (d_energy_min, md's A
 * operand) -- must keep RB+1 rows, not the one or two it keeps today. Today's
 * per-row pointer swap of d_dml/d_dml1 IS a ring of depth 2, and today's single
 * d_energy_min IS a ring of depth 1, so this file now owns its rotating rows as
 * rings and the swap becomes one case of advancing them.
 *
 * DEPTH 2 / 1 IS TODAY, LITERALLY: md_ring_advance() keeps the old swap statement for
 * that case, so the default path runs the same code, not merely equivalent code.
 * RNA_MD_RING_DEPTH=N (2..MD_RING_MAX, a diagnostic) runs the ROW path with deeper
 * rings, which must be byte-identical too: rows descend, so each row's write range
 * contains every older row's, and whatever a slot held N rows ago is overwritten
 * before it is read (the argument md_snapshot_dml() already makes for depth 2). That
 * proves the allocation, prefill, teardown and VRAM accounting of N slots on the
 * path that exists today, before stage 3b's schedule needs them.
 *
 * FREE BY ALLOCATION. teardown_gpu() used to free whatever d_dml/d_dml1 pointed at,
 * which at depth 2 is always the two allocations and at depth N is two arbitrary
 * slots. The ring arrays are the allocations; d_dml, d_dml1 and d_energy_min only
 * point into them. */
#define MD_RING_MAX 130                    /* RB + 1 for RB up to 128, plus one */
static int*  g_dml_ring[MD_RING_MAX];      /* g_dml_depth allocations */
static int*  g_emin_ring[MD_RING_MAX];     /* g_emin_depth allocations */
static int   g_dml_depth  = 2;             /* this chunk's: 2 unless RNA_MD_RING_DEPTH took */
static int   g_emin_depth = 1;             /* always g_dml_depth - 1 */
static long  g_ring_k     = 0;             /* advances this chunk */
extern "C" int* rnafold_circ_fm2_device(void);

extern "C" int rnafold_md_ring_depth(void); /* requested; defined with md_tail_refuse() */
static int md_ring_depth_effective(void);   /* this chunk's, after refusals */

/* RNA_ROW_BATCH=K (2..turn+2; 0/1 = off): the fML triangle is written K rows at a time
 * from a row-major stage instead of one scattered row per sweep row. See
 * md_flush_rows_kernel. DEFAULT 5 since 2026-10-02 (see rnafold_row_batch). */
#define ROW_BATCH_MAX 5            /* turn+2 at the default turn; the stage's slot count */
int*       d_fml_stage   = NULL;   /* ROW_BATCH_MAX x row_total ints, row-major */
static int g_stage_n     = 0;      /* rows staged and not yet flushed */
static int g_stage_K     = 0;      /* the K in force for this chunk (clamped) */

static int
rnafold_row_batch(void)
{
  static int v = -1;
  if(v < 0) {
    const char* e = getenv("RNA_ROW_BATCH");
    /* DEFAULT 5 since 2026-10-02 (A100: rb5 best at 1200/2400/5601 nt per row, wall
     * -1.0..-1.8 % at 400x5601 in two notebooks, triangles identical). =0 turns it off. */
    v = (e && e[0]) ? atoi(e) : ROW_BATCH_MAX;
    if(v < 2) v = 0;
    if(v > ROW_BATCH_MAX) v = ROW_BATCH_MAX;
  }
  return v;
}
cudaGraphExec_t graph_exec       = NULL;

// T2a (device.cu): device-to-host for backtrack worker w, through w's own
// copy stream and pinned stage. w < 0 is the old blocking default-stream copy.
void rnafold_d2h_w(void* dst, const void* src, const size_t bytes, const int w);
#define d2h_w rnafold_d2h_w
int             graph_exec_valid = 0;

// Diagnostic-only counters/timer for the graph exec update-vs-reinstantiate
// path, to measure (rather than guess) how often cudaGraphExecUpdate() is
// actually cheap-patching vs. falling back to a full destroy+instantiate,
// and how much cumulative CPU time that bookkeeping (capture+update/
// instantiate+destroy, NOT the launch+sync of the real GPU work) costs
// across a whole fold. Printed once at process exit via atexit(). Remove
// once the CUDA-graph-vs-old-branch timing question is settled.
long   graph_update_success_count      = 0;
long   graph_first_instantiate_count   = 0;
long   graph_forced_reinstantiate_count = 0;
double graph_mgmt_seconds              = 0.0;

static void
print_graph_update_stats(void) {
  if (!vrna_cuda_quiet()) fprintf(stderr,
    "%-24s CUDA graph stats: %ld update() succeeded, %ld first-time instantiate, "
    "%ld forced reinstantiate (update failed), %.3f s cumulative capture/update/"
    "instantiate/destroy overhead (excludes launch+sync)\n",
    __FILE__, graph_update_success_count, graph_first_instantiate_count,
    graph_forced_reinstantiate_count, graph_mgmt_seconds);
}

static inline double
graph_now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

PUBLIC void
init_gpu(const int nfiles, const int length,
         const size_t* tri_off_H, const size_t* row_off_H) {
  if(!first) return;
  const double _t_ig1 = rnafold_now_seconds();
  if (!vrna_cuda_quiet()) fprintf(stderr,"%-24s init_gpu(%d, %d)\n",__FILE__,nfiles,length);
  g_tri_cells = tri_off_H[nfiles];   /* RNA_TRI_CHECKSUM */
  cudaError_t error;
  // graph_stream is nfiles/length-independent -- guarded on its own initial
  // value (0), not on `first`, so teardown_gpu() can reset first=1 between
  // GPU batches without this recreating (and leaking the handle to) a fresh
  // stream every batch. Created exactly once for the whole process.
  if(graph_stream == 0) gpuErrchk( cudaStreamCreate(&graph_stream) );

  TIMED_CUDAMALLOC(&d_tri_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_tri_off_H, tri_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );
  TIMED_CUDAMALLOC(&d_row_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_row_off_H, row_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );

  // Staggered_Row_Batching Phase 4: allocated here (fixed size for the whole
  // chunk), but not populated here -- unlike tri_off_H/row_off_H these change
  // every sweep row i, so the real upload happens per-row in load_fML()/
  // modular_decomposition_cuda() instead.
  // (d_size_off_H / d_side_off_H / d_i_H: the chunk's row tables, bound per row.)

  // Staggered_Row_Batching Phase 2d: allocation sizes now the real per-H sum
  // (row_off_H[nfiles]/tri_off_H[nfiles]) instead of a uniform nfiles*(...)
  // multiply -- see compute_batch_offsets(), mfe_cuda.c.
  const size_t mem_size_len = row_off_H[nfiles] * sizeof(int);
  const size_t ijsize_len   = tri_off_H[nfiles] * sizeof(int);
  // Staggered_Row_Batching Phase 6d: cached for load_fML()/
  // modular_decomposition_cuda(), which transfer whole row-shaped buffers each
  // row but never receive the offset table.
  g_row_total = row_off_H[nfiles];

  error = cudaMalloc((void **) &d_energy_min, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_energy_min %zu returned error %s (code %d), line(%d)\n", // 32-bit signed integer overflow bug fix
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  error = cudaMalloc((void **) &d_fml_i, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_fml_i %zu returned error %s (code %d), line(%d)\n", // 32-bit signed integer overflow bug fix
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  if(!rnafold_fml_int16()) {
    error = cudaMalloc((void **) &d_fml_j, ijsize_len);
    if (error != cudaSuccess)  {
        printf("cudaMalloc d_fml_j %zu returned error %s (code %d), line(%d)\n", // 32-bit signed integer overflow bug fix
  	     ijsize_len, cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
  } else {
    // colb_off is built to the LONGEST record; every shorter one indexes a
    // prefix of it, exactly as the triangular Indx() already does.
    int maxlen = 0;
    for(int H=0;H<nfiles;H++) {
      const size_t cells = tri_off_H[H+1]-tri_off_H[H];
      int n = 0; while(((size_t)(n+1)*(n+2))/2 < cells) n++;
      if(n > maxlen) maxlen = n;
    }
    size_t* colb = (size_t*)malloc((size_t)(maxlen+2)*sizeof(size_t));
    colb[0] = colb[1] = 0;
    for(int j=1;j<=maxlen;j++)
      colb[j+1] = colb[j] + (size_t)((j + FML_BLK - 1)/FML_BLK);
    TIMED_CUDAMALLOC(&d_colb_off, (size_t)(maxlen+2)*sizeof(size_t));
    gpuErrchk( cudaMemcpy(d_colb_off, colb, (size_t)(maxlen+2)*sizeof(size_t), cudaMemcpyHostToDevice) );

    size_t* boff = (size_t*)malloc((size_t)(nfiles+1)*sizeof(size_t));
    boff[0] = 0;
    for(int H=0;H<nfiles;H++) {
      const size_t cells = tri_off_H[H+1]-tri_off_H[H];
      int n = 0; while(((size_t)(n+1)*(n+2))/2 < cells) n++;
      boff[H+1] = boff[H] + colb[n+1];
    }
    g_base_total = boff[nfiles];
    TIMED_CUDAMALLOC(&d_base_off_H, (size_t)(nfiles+1)*sizeof(size_t));
    gpuErrchk( cudaMemcpy(d_base_off_H, boff, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );
    free(colb); free(boff);

    TIMED_CUDAMALLOC(&d_fml_j16, tri_off_H[nfiles]*sizeof(short));
    TIMED_CUDAMALLOC(&d_fml_b,   g_base_total*sizeof(int));
    TIMED_CUDAMALLOC(&d_fml_row, mem_size_len);
  }

  error = cudaMalloc((void **) &d_dml, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_dml %zu returned error %s (code %d), line(%d)\n", // 32-bit signed integer overflow bug fix
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  // GPU-resident sweep, step 1 -- see the declarations above. Same row shape
  // and lifetime as d_dml, so they are allocated, prefilled, sized into the
  // VRAM budget and freed alongside it.
  error = cudaMalloc((void **) &d_dml1, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_dml1 %zu returned error %s (code %d), line(%d)\n",
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  /* Stage 3a.1: the rings. Slots 0 and 1 of the dml ring and slot 0 of the energy_min
   * ring ARE the allocations above, so at the default depths nothing new is allocated.
   * Extra slots are prefilled with INF with the others, further down. */
  g_dml_depth  = md_ring_depth_effective();
  g_emin_depth = g_dml_depth - 1;
  g_ring_k     = 0;
  g_dml_ring[0]  = d_dml;
  g_dml_ring[1]  = d_dml1;
  g_emin_ring[0] = d_energy_min;
  for (int s = 2; s < g_dml_depth; s++)
    gpuErrchk( cudaMalloc((void **) &g_dml_ring[s], mem_size_len) );
  for (int s = 1; s < g_emin_depth; s++)
    gpuErrchk( cudaMalloc((void **) &g_emin_ring[s], mem_size_len) );

  error = cudaMalloc((void **) &d_fml_prev, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_fml_prev %zu returned error %s (code %d), line(%d)\n",
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  // RNA_ROW_BATCH: the stage, ROW_BATCH_MAX rows of the same row-major layout.
  g_stage_n = 0;
  if(rnafold_row_batch()) {
    error = cudaMalloc((void **) &d_fml_stage, mem_size_len * ROW_BATCH_MAX);
    if (error != cudaSuccess)  {
        printf("cudaMalloc d_fml_stage %zu returned error %s (code %d), line(%d)\n",
               mem_size_len * ROW_BATCH_MAX, cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
  }

  // See the forward declarations above for why this happens here rather
  // than lazily at first launch.
  g_block_size_fmli = rnafold_choose_block_size(fmli_kernel, BLOCK_SIZE, "RNA_FMLI_BLOCK_SIZE");
  // The occupancy probe has to name the instantiation that will launch.
  g_block_size_md = rnafold_choose_block_size(modular_decomposition_kernel<MD_TILE>, BLOCK_SIZE, "RNA_MD_BLOCK_SIZE");
  // The kernel's tile-to-shuffle-group alignment assumes whole warps per
  // block. rnafold_choose_block_size() only ever yields a power of two >= 32
  // (occupancy heuristic or env override, both validated), so this holds
  // today -- asserted so a future change there fails loudly here instead of
  // silently corrupting the reduction for TILE < 32.
  assert(g_block_size_md % 32 == 0);
  if (!vrna_cuda_quiet()) fprintf(stderr,"%-24s fmli_kernel block size %d, modular_decomposition_kernel block size %d (both were hardcoded %d), md tile %d\n",
	  __FILE__, g_block_size_fmli, g_block_size_md, BLOCK_SIZE, MD_TILE);

  md_sparse_alloc(nfiles, row_off_H);   /* RNA_MD_SPARSE_SELFTEST; no-op otherwise */

  stage_ig1_s += rnafold_now_seconds() - _t_ig1;
  first = 0;
  return;
}

// Frees the 4 nfiles/length-scaled device buffers allocated by init_gpu()
// and resets `first` so the next init_gpu() call actually re-runs (rather
// than no-op'ing) at a new batch's nfiles. graph_stream is deliberately left
// alone -- cheap to keep, no reason to tear it down between batches (see the
// comment on its creation above). Defensively destroys graph_exec if valid,
// since its captured nodes reference the buffers freed right below --
// belt-and-suspenders per the multi-batch design doc; the existing
// update-or-reinstantiate fallback in
// load_fML_modular_decomposition_load_min_fML() would likely self-heal even
// without this, but there's no reason to leave a stale exec referencing
// about-to-be-freed memory during the "GPU unloaded" window between batches.
PUBLIC void
teardown_gpu(void) {
  if(first) return; // never initialized (or already torn down) -- nothing to free
  md_sparse_free();   /* RNA_MD_SPARSE: counters first (a sync), then by the pointer */
  /* Stage 3a.1: by allocation. Slot 0 is the original d_energy_min allocation; once
   * the ring has advanced, d_energy_min may point at any slot. */
  for (int s = 0; s < g_emin_depth; s++) {
    gpuErrchk( cudaFree(g_emin_ring[s]) );
    g_emin_ring[s] = NULL;
  }
  d_energy_min = NULL;
  /*
   * FREE BY WHAT WAS ALLOCATED, NOT BY WHAT THE CONFIGURATION NOW SAYS.
   *
   * This used to branch on rnafold_fml_int16() -- and that is a question whose
   * answer CAN CHANGE between the allocation and the free, which made it a wrong
   * answer generator. rnafold_fml_int16_stand_down() flips int16 off for the rest
   * of the process when a model turns out to be unsupported (--noLP, a non-default
   * salt), so:
   *
   *   batch 1, plain    int16 ON  -> allocates d_fml_j16 / d_fml_b
   *                     teardown: int16 still on, frees them, LEAVES THE POINTERS SET
   *   batch 2, --noLP   stand-down -> int16 OFF -> allocates d_fml_j instead
   *                     md_cell selects its path on `if(fml_j16)`, which is now a
   *                     DANGLING non-NULL pointer -> reads freed device memory
   *
   * Measured through the Python binding: plain then --noLP gives a wrong structure,
   * and every later call in that process stays wrong. --noLP first is fine, because
   * int16 is off before anything is allocated. The CLI cannot reach it -- one batch
   * shape and one model per run, then exit -- which is why 45 option checks and
   * every parity bar in the project are blind to it, exactly as they were to the
   * teardown defect this same binding found in September.
   *
   * So: guard on the POINTER, free it, and NULL it. The int16 setting is not
   * consulted here at all, which is what makes this robust against it changing.
   */
  if (d_fml_i)      { gpuErrchk( cudaFree(d_fml_i) );      d_fml_i      = NULL; }
  if (d_fml_j)      { gpuErrchk( cudaFree(d_fml_j) );      d_fml_j      = NULL; }
  if (d_fml_j16)    { gpuErrchk( cudaFree(d_fml_j16) );    d_fml_j16    = NULL; }
  if (d_fml_b)      { gpuErrchk( cudaFree(d_fml_b) );      d_fml_b      = NULL; }
  /* These three are int16-path buffers too, and they were freed inside the
   * `if (d_fml_band)` block (RNA_MD_BAND, since deleted) -- a misplaced brace, so
   * with the band off (the default) they LEAKED on every batch and left three more
   * dangling pointers. */
  if (d_fml_row)    { gpuErrchk( cudaFree(d_fml_row) );    d_fml_row    = NULL; }
  if (d_colb_off)   { gpuErrchk( cudaFree(d_colb_off) );   d_colb_off   = NULL; }
  if (d_base_off_H) { gpuErrchk( cudaFree(d_base_off_H) ); d_base_off_H = NULL; }
  g_base_total = 0;
  /* Stage 3a.1: by allocation, not by what d_dml/d_dml1 point at now. At depth 2 the
   * two are the same set; at depth N they are two of N slots. */
  for (int s = 0; s < g_dml_depth; s++) {
    gpuErrchk( cudaFree(g_dml_ring[s]) );
    g_dml_ring[s] = NULL;
  }
  d_dml = d_dml1 = NULL;
  gpuErrchk( cudaFree(d_fml_prev) );
  if(d_fml_stage) { gpuErrchk( cudaFree(d_fml_stage) ); d_fml_stage = NULL; }  // by the pointer
  g_stage_n = 0;
  gpuErrchk( cudaFree(d_tri_off_H) );
  gpuErrchk( cudaFree(d_row_off_H) );
  d_size_off_H = d_side_off_H = NULL;   // borrowed from the row tables
  d_i_H        = NULL;
  if(graph_exec_valid) {
    gpuErrchk( cudaGraphExecDestroy(graph_exec) );
    graph_exec_valid = 0;
  }
  first = 1;
}

// Bytes of device memory this file needs for one additional sequence at the
// given length -- the 3 mem_size_len-scale buffers (d_energy_min, d_fml_i,
// d_dml) plus the dominant ijsize_len-scale one (d_fml_j), mirroring
// init_gpu()'s own size formulas exactly.
/* CIRCULAR: whether this run will need the fM2_real triangle. Set by the
 * driver BEFORE any chunk is admitted, because the byte model below drives
 * chunk admission and is consulted long before rnafold_circ_alloc() runs.
 * Under-counting here does not merely mis-report -- it admits a chunk that
 * does not fit, and the failure surfaces as an OOM inside par_mfe(). */
static int g_circ_expected = 0;

extern "C" void
rnafold_circ_expect(const int circ)
{
  g_circ_expected = circ ? 1 : 0;
}


static size_t md_bytes_per_file_mode(const int length, const int i16);

PUBLIC size_t
modular_decomposition_bytes_per_file(const int length) {
  return md_bytes_per_file_mode(length, rnafold_fml_int16());
}

/* The same model with the fML encoding given, so the int16 AUTO decision can price int32
 * before the gate has been decided. */
static size_t
md_bytes_per_file_mode(const int length, const int i16) {
  // x5, not x3: d_energy_min, d_fml_i, d_dml, and (GPU-resident sweep step 1)
  // d_dml1 + d_fml_prev. This number drives RNAfold.c's chunk admission, so
  // under-counting it does not merely mis-report -- it lets a chunk be
  // accepted that does not fit, and the failure surfaces as an OOM inside
  // par_mfe() rather than as a smaller batch.
  const size_t cells        = (size_t)(length+1)*(length+2)/2;
  /* RNA_ROW_BATCH's stage: ROW_BATCH_MAX more row-shaped buffers -- noise beside the
   * triangle, but this number admits chunks, so it is charged rather than assumed. */
  const size_t stage_bytes  = rnafold_row_batch() ? (size_t)(length+1) * sizeof(int) * ROW_BATCH_MAX : 0;
  /* Stage 3a.1: depth-N rings add N-2 dml rows and N-2 energy_min rows. The REQUESTED
   * depth, not this chunk's: admission runs before any chunk exists, and a refusal can
   * only make this an over-count, never an under-count. */
  const size_t ring_bytes   = (size_t)(length+1) * sizeof(int) * 2
                              * (size_t)(rnafold_md_ring_depth() - 2);
  if(i16) {
    // x6: the five above plus d_fml_row, which is row-shaped and so is noise
    // beside the triangle. The triangle halves, and the baselines add one int32
    // per FML_BLK entries -- 1.6% at B=64, against the 50% saved.
    const size_t mem_size_len = (size_t)(length+1) * sizeof(int) * 6;
    const size_t tri16        = cells * sizeof(short);
    const size_t base_bytes   = ((cells + FML_BLK - 1)/FML_BLK + (size_t)length + 2) * sizeof(int);
    return mem_size_len + tri16 + base_bytes + stage_bytes + ring_bytes
         + md_sparse_bytes_per_file(length)
         + (g_circ_expected ? cells * sizeof(int) : 0);   /* fM2_real stays int32 */
  }
  const size_t mem_size_len = (size_t)(length+1) * sizeof(int) * 5;
  const size_t ijsize_len   = cells * sizeof(int);
  /* CIRCULAR costs a SECOND full triangle -- fM2_real, the same extent as
   * d_fml_j. This is the "costs a chunk width" trade PORT_CIRC_SPEC.md names:
   * the arithmetic is free, the memory is not. */
  return mem_size_len + ijsize_len + stage_bytes + ring_bytes
       + md_sparse_bytes_per_file(length)
       + (g_circ_expected ? cells * sizeof(int) : 0);
}

// Staggered_Row_Batching Phase 6b: replaces compute_max_gpu_batch() (which
// computed a single fixed record-count "capacity" up front from one shared
// length -- correct only because every chunk was uniform-length, and about
// to stop being true once Phase 6c drops that constraint). RNAfold.c now
// tracks a running VRAM budget incrementally as records of potentially
// different lengths accumulate into a chunk, rather than committing to a
// count derived from a single representative length. These two functions
// are the building blocks for that: gpu_bytes_per_file() gives the real
// per-record cost at any length (RNAfold.c calls this once per candidate
// record, not once per chunk), and compute_gpu_usable_bytes() gives the
// VRAM budget for a fresh chunk (called once per chunk start -- the
// in-progress chunk's own buffers aren't allocated yet at accumulation
// time, that happens later inside par_mfe(), so free VRAM doesn't
// meaningfully change as records accumulate on the host).
//
// The old MAX_GPU_BATCH_GRID_LIMIT (CUDA's gridDim.y/z 65535 cap) is
// retired along with compute_max_gpu_batch() -- it existed specifically
// because nfiles was used directly as gridDim.y in this file's kernels (and
// int_loop.cu's/hp_mb_loop.cu's); Phases 4-5 flattened every kernel here to
// a 1-D grid, so that limit's rationale no longer applies (noted but not
// acted on when Phase 5 shipped -- this is where it's actually retired).
//
// Aggregates the per-file VRAM cost across all three GPU-resident files
// (this one, int_loop.cu, hp_mb_loop.cu) for one record at the given
// length -- owned here since this file already has the dominant-cost
// buffer and the `first`-guard pattern this whole feature extends.
PUBLIC size_t
gpu_bytes_per_file(const int length) {
  return modular_decomposition_bytes_per_file(length)
       + int_loop_bytes_per_file(length)
       + hp_mb_loop_bytes_per_file(length);
}

// Queries free VRAM (cudaMemGetInfo) and returns the safety-margined usable
// budget for a fresh chunk. Call once per chunk start, not per record.
// TODO: the 85% safety margin was tightened from an initial 95% after the
// row buffers in fill_arrays.c (DMLi/DMLi1/DMLi2 + 5 more) became
// page-locked (cuda_host_alloc_ints()) -- pinned host allocation is a more
// constrained resource than pageable RAM (can't be swapped, often a
// stricter OS limit) and hard-fails rather than falling back, so this
// margin is the actual safety net for that. Not derived from a systematic
// study of allocator fragmentation/CUDA context overhead across many batch
// sizes, just empirically fine so far -- revisit if a future run OOMs
// closer to the edge than any tested so far did.
// RNA_GPU_VRAM_BUDGET_MB caps the budget below what free VRAM would allow.
// It can only ever *lower* it, never raise it, so setting it can't provoke an
// OOM that wouldn't have happened anyway. Two uses: forcing multi-chunk
// behaviour on a machine whose card is big enough to swallow the whole input
// in one batch (the only way to exercise the incremental budget above --
// chunks otherwise split on length change alone), and being a polite tenant
// on a GPU shared with another process. Env var rather than a CLI flag, per
// this project's convention of avoiding gengetopt regeneration.
//
// RNA_VRAM_BUDGET (budget.c, PORT_BUDGETS.md B2) is a different thing: a ceiling on the
// whole PROCESS -- context and every buffer -- given in MB or as a fraction of the card's
// total. RNA_GPU_VRAM_BUDGET_MB stays what it was, a buffers-only test cap the harnesses
// depend on for exact chunk counts.
//
// What the process holds is measured, not modelled: the context is charged as the total
// minus what is free at the process's FIRST query (before any buffer of ours), and its
// buffers as whatever has left the free pool since -- which is how a chunk retained for
// the backtrack pipeline is charged without anyone keeping a ledger. Measured on the
// laptop under WSL (2026-10-13): every process sees 3304 of 4096 MB free whether or not
// another holds a context, so the 792 MB is this process's own reserve. On a native,
// shared GPU the charge also takes in other processes, which makes the budget stricter,
// never looser.
extern "C" size_t vrna_budget_vram(size_t device_total);   /* budget.c */
static size_t g_vram_free0 = 0;   /* free at the first query; 0 = not yet queried */
static size_t g_vram_ctx   = 0;   /* charged for the context and driver reserve */
static size_t g_vram_total = 0;
static size_t g_vram_cap   = 0;   /* the budget in force, set while one is (0 = none) */
static size_t g_vram_minfree = SIZE_MAX;   /* lowest free VRAM sampled: the peak held */

/* Bytes the next chunk may take under the VRAM budget; SIZE_MAX when there is none. */
static size_t
gpu_budget_allowed(const size_t free_bytes, const size_t total_bytes)
{
  const size_t cap  = vrna_budget_vram(total_bytes);
  if (cap == SIZE_MAX)
    return SIZE_MAX;
  const size_t ours = (g_vram_free0 > free_bytes) ? g_vram_free0 - free_bytes : 0;
  const size_t held = g_vram_ctx + ours;
  /* The SAME 0.85 the soft limit applies to free VRAM, for the same reason: the per-record
   * model (gpu_bytes_per_file) counts the buffers it knows, not allocator granularity or
   * what is allocated beside them. MEASURED 2026-10-13 at a quarter budget on the laptop:
   * without it the peak ended 8 and 14 MB over (A_longseq, F_lite), 3-6 % of the room
   * the budget left; with it every arm stays within. */
  return (cap > held) ? (size_t)((double)(cap - held) * 0.85) : 0;
}

/* budgeted: apply RNA_VRAM_BUDGET. The unbudgeted figure is the soft limit alone,
 * asked when one record is over the budget: within it, the record is folded anyway. */
static size_t
gpu_usable_bytes(const int budgeted) {
  size_t free_bytes = 0, total_bytes = 0;
  gpuErrchk( cudaMemGetInfo(&free_bytes, &total_bytes) );
  if (g_vram_free0 == 0) {
    g_vram_free0 = free_bytes;
    g_vram_ctx   = (total_bytes > free_bytes) ? total_bytes - free_bytes : 0;
  }
  const double safety_margin = 0.85; // TODO: tune further if a tighter margin ever OOMs
  size_t usable = (size_t)((double)free_bytes * safety_margin);

  const char *env_budget = getenv("RNA_GPU_VRAM_BUDGET_MB");
  if(env_budget && *env_budget) {
    const long mb = atol(env_budget);
    if(mb > 0) {
      const size_t cap = (size_t)mb * 1024u * 1024u;
      static int announced = 0;
      if(!announced) { // once per process, matching this file's other config messages
        fprintf(stderr, "%-24s RNA_GPU_VRAM_BUDGET_MB=%ld capping GPU chunk budget "
                        "(free VRAM would have allowed %.0f MB)\n",
                __FILE__, mb, (double)usable/(1024.0*1024.0));
        announced = 1;
      }
      if(cap < usable) usable = cap;
    } else {
      fprintf(stderr, "%-24s ignoring RNA_GPU_VRAM_BUDGET_MB=%s (want a positive integer)\n",
              __FILE__, env_budget);
    }
  }
  if (budgeted) {
    const size_t allowed = gpu_budget_allowed(free_bytes, total_bytes);
    static int   said    = 0;

    if (allowed != SIZE_MAX) {
      g_vram_total = total_bytes;
      g_vram_cap   = vrna_budget_vram(total_bytes);
      if (!said) {
        said = 1;
        if (!vrna_cuda_quiet())
          fprintf(stderr, "%-24s VRAM budget %.0f MB of %.0f: %.0f MB charged for the context, "
                          "chunks sized to at most %.0f MB (free VRAM would allow %.0f)\n", __FILE__,
                  (double)vrna_budget_vram(total_bytes) / 1048576.0, (double)total_bytes / 1048576.0,
                  (double)g_vram_ctx / 1048576.0, (double)allowed / 1048576.0,
                  (double)usable / 1048576.0);
      }
      if (allowed < usable)
        usable = allowed;
    }
  }
  /* int16 AUTO (mfe_cuda.c): decided on the first query, before any record is priced for
   * real. A full-cap chunk is priced in int32 at the WORST bytes per cell of three lengths
   * (row buffers weigh most at short lengths), so the estimate errs towards int16, the side
   * that cannot run out of VRAM. */
  {
    static int decided = 0;
    if (!decided) {
      decided = 1;
      double per_cell = 0.0;
      const int Ls[3] = { 600, 2000, 5601 };
      for (int t = 0; t < 3; t++) {
        const double cells = (double)(Ls[t] + 1) * (double)(Ls[t] + 2) / 2.0;
        const double b = (double)(md_bytes_per_file_mode(Ls[t], 0) + int_loop_bytes_per_file(Ls[t])
                                  + hp_mb_loop_bytes_per_file(Ls[t])) / cells;
        if (b > per_cell) per_cell = b;
      }
      const size_t full32 = (size_t)(per_cell * (double)g_chunk_cells_cap);
      rnafold_fml_int16_auto_decide(usable, g_chunk_cells_cap, full32, !md_sparse_planned());
    }
  }
  return usable;
}

PUBLIC size_t
compute_gpu_usable_bytes(void) {
  return gpu_usable_bytes(1);
}

/* The soft limit alone: free VRAM x 0.85 (and the RNA_GPU_VRAM_BUDGET_MB test cap). */
PUBLIC size_t
compute_gpu_usable_bytes_soft(void) {
  return gpu_usable_bytes(0);
}

/* A record over the VRAM budget but within the soft limit: say so, then the caller folds
 * it alone on the device (PORT_BUDGETS.md: warn, and fold it anyway). The first five are
 * named; after that only a count, at exit, so a long input cannot flood stderr. */
static int g_vram_over_n = 0;   /* records folded alone over the budget */

static void
vram_over_report(void)
{
  if (g_vram_over_n > 5)
    fprintf(stderr, "%-24s %d records in all were over the VRAM budget and folded alone "
                    "anyway\n", __FILE__, g_vram_over_n);
}

/* POSITIVE EVIDENCE that the budget held: par_mfe() samples free VRAM after each chunk's
 * allocations and again at the end of its sweep (when a retained chunk is still
 * resident), and the lowest reading is the most this process held. Only while a VRAM
 * budget is in force, so `full` pays nothing. */
static void
vram_peak_report(void)
{
  if ((g_vram_minfree == SIZE_MAX) || vrna_cuda_quiet())
    return;
  const size_t held = (g_vram_total > g_vram_minfree) ? g_vram_total - g_vram_minfree : 0;
  fprintf(stderr, "%-24s VRAM budget: peak held %.0f MB (context and reserve %.0f) of a "
                  "%.0f MB budget -- %s\n", __FILE__, (double)held / 1048576.0,
          (double)g_vram_ctx / 1048576.0, (double)g_vram_cap / 1048576.0,
          (held <= g_vram_cap) ? "within it"
          : (g_vram_over_n > 0) ? "over it: a record over the budget was folded alone, as warned"
          : "OVER IT: the device holds more than the chunk model predicted");
}

PUBLIC void
rnafold_vram_sample(void)
{
  size_t f = 0, t = 0;

  if (g_vram_cap == 0)
    return;
  if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
    cudaGetLastError();
    return;
  }
  if (g_vram_minfree == SIZE_MAX)
    atexit(vram_peak_report);
  if (f < g_vram_minfree)
    g_vram_minfree = f;
}

PUBLIC void
rnafold_vram_over_budget(const size_t need, const size_t allowed)
{
  if (g_vram_over_n++ == 0)
    atexit(vram_over_report);
  if (g_vram_over_n <= 5)
    fprintf(stderr, "%-24s WARNING: a record needs %.0f MB of VRAM, over the VRAM budget's "
                    "%.0f MB; folding it alone on the device anyway (within what is free)\n",
            __FILE__, (double)need / 1048576.0, (double)allowed / 1048576.0);
}

extern "C" void
rnafold_set_chunk_cells_cap(const unsigned long long cap_cells)
{
  g_chunk_cells_cap = cap_cells;
  g_chunk_cells_cap_set = 1;
}

/* 1 and the cap when a caller has set one (RNAfold, or an earlier library batch) */
extern "C" int
rnafold_get_chunk_cells_cap(unsigned long long *cap_cells)
{
  *cap_cells = g_chunk_cells_cap;
  return g_chunk_cells_cap_set;
}

// ---------------------------------------------------------------------------
// int16 fml_j (RNA_FML_INT16). See INT16_FML_SCOPE.md.
//
// A cell is stored as a signed 16-bit offset from a baseline shared by FML_BLK
// consecutive entries WITHIN one column, with FML_INF16 reserved for INF. The
// baseline is whichever entry of the block is packed first (the highest index,
// packed at the earliest row) -- deliberately NOT the block minimum, which
// would change as the block fills and invalidate offsets already written. What
// bounds the offset is the block's SPREAD, not the choice of origin.
// ---------------------------------------------------------------------------




// Closes row i: the two writers have both run, so d_fml_row now holds this
// row's FINAL values and they can be packed. Launch shape mirrors
// load_fML_kernel's (size_off_H), which is the superset of the two write ranges.
// The md chain cells, shared through a header: the
// kernels below are wrappers around exactly this code.
#include "md_chain_cells.inc"

__global__ void
pack_fml_kernel(const int nfiles, const int i_row, const int turn, const int length,
                const int* __restrict__ fml_row,
                      short* __restrict__ fml_j16,
                      int* __restrict__ fml_b,
                const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
                const size_t* __restrict__ size_off_H, const size_t total,
                const int* __restrict__ i_H) {
  // The arithmetic lives in md_chain_cells.inc.
  pack_fml_cell(nfiles, i_row, turn, length, fml_row, fml_j16, fml_b, tri_off_H, row_off_H, base_off_H, colb_off, size_off_H, total, i_H,
                (size_t)blockIdx.x*blockDim.x+threadIdx.x);
}

/* RNA_MD_TAIL: load_fML + load_min_fML + pack_fml + fml_prev as one launch.
 * See md_close_row_cell for why this is byte-identical. */
__global__ void
md_close_row_kernel(const int nfiles, const int i_row, const int turn,
                    const int* __restrict__ energy_min,
                    const int* __restrict__ dml,
                          int* __restrict__ fml_prev,
                          int* __restrict__ fml_j,
                          short* __restrict__ fml_j16,
                          int* __restrict__ fml_b,
                    const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                    const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
                    const size_t* __restrict__ size_off_H, const size_t total,
                    const int* __restrict__ i_H,
                          int* __restrict__ stage) {   // RNA_ROW_BATCH slot, or NULL
  md_close_row_cell(nfiles, i_row, turn, energy_min, dml, fml_prev, fml_j, fml_j16, fml_b,
                    tri_off_H, row_off_H, base_off_H, colb_off, size_off_H, total, i_H,
                    (size_t)blockIdx.x*blockDim.x+threadIdx.x, stage);
}

/* RNA_ROW_BATCH: write `npend` staged rows -- i_new (the newest, smallest i) up to
 * i_new+npend-1 -- into the triangle in ONE pass.
 *
 * WHY: the triangle is column-major (Indx(i,j) = j(j-1)/2 + i), so writing ONE row puts
 * one element in every column -- measured 31.96 sectors per store request in load_my_c,
 * 18 in md_close_row (RTX 3050, ncu): a sector per element. Here the K pending rows
 * of a column are written CONTIGUOUSLY (they share a sector), through a shared-memory
 * transpose so the stage is read coalesced as well -- see the kernel body.
 *
 * WHY THIS IS LEGAL WITHOUT TOUCHING md: md(r) reads fML rows >= r+turn+2. A row is
 * flushed at most K-1 rows after it was closed, and the driver clamps K <= turn+2, so
 * no md ever reads a row that is still only in the stage.
 *
 * The cells are row i_new's size range (the widest pending row); row r is written
 * only where it has that column (j >= r+turn+1), which is exactly the cells
 * md_close_row_cell staged for it. int16: the per-(column, 64-row block) baseline is
 * the FIRST non-INF value in SWEEP order (descending i), so each column's thread sets
 * any unset baselines oldest-row-first before the tile packs -- the same baselines
 * row-at-a-time packing would have chosen. */
template<int TJ>
__global__ void
md_flush_rows_kernel(const int nfiles, const int turn, const int i_new, const int npend,
                     const int K, const size_t row_total,
                     const int* __restrict__ stage,
                           int* __restrict__ fml_j,
                           short* __restrict__ fml_j16,
                           int* __restrict__ fml_b,
                     const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                     const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
                     const size_t* __restrict__ size_off_H, const size_t total)
{
  /* One block = TJ consecutive cells (columns) of row i_new's size range. Shared-memory
   * TRANSPOSE: the stage is row-major, the triangle column-major, so
   *   1. read the tile row by row  -- thread t reads column t of each pending row:
   *      consecutive threads, consecutive j, COALESCED;
   *   2. int16: one thread per column sets that column's unset baselines, oldest
   *      pending row first (sweep order), from shared;
   *   3. write column by column -- consecutive threads write consecutive ROWS of the
   *      same column, which Indx(i,j) = j(j-1)/2 + i makes CONTIGUOUS.
   * The first version gave lanes different rows of the stage and paid uncoalesced
   * READS for coalesced writes: a wash (K=5: 12.4 vs 11.9 us/row on an RTX 3050). */
  __shared__ int    sv[ROW_BATCH_MAX][TJ];
  __shared__ int    sH[TJ], sj[TJ];
  const int    t  = threadIdx.x;
  const size_t m  = (size_t)blockIdx.x * TJ + t;
  int H = -1, j = 0;

  if(m < total) {
    H = flatten_index_to_H(m, size_off_H, nfiles);
    j = (int)((long long)m - (long long)size_off_H[H]) + i_new + turn + 1;
    const size_t o = row_off_H[H];
    for(int q = 0; q < npend; q++) {
      const int r = i_new + q;
      sv[q][t] = (j >= r + turn + 1) ? stage[(size_t)(r % K)*row_total + o + j] : INF;
    }
  }
  sH[t] = H; sj[t] = j;
  __syncthreads();

  if(fml_j16 && H >= 0)
    for(int q = npend - 1; q >= 0; q--) {               // sweep order: oldest row first
      const int r = i_new + q, v = sv[q][t];
      if(j < r + turn + 1 || v == INF) continue;
      const size_t bidx = fml_bidx(base_off_H, colb_off, H, j, r);
      if(fml_b[bidx] == FML_BASE_UNSET) fml_b[bidx] = v;
    }
  __syncthreads();

  for(int k = t; k < TJ * npend; k += TJ) {             // row fastest: contiguous per column
    const int c = k / npend, q = k % npend;
    const int Hc = sH[c], jc = sj[c], r = i_new + q;
    if(Hc < 0 || jc < r + turn + 1) continue;
    const int v = sv[q][c];
    if(fml_j16) pack_fml_value(v, Hc, r, jc, fml_j16, fml_b, tri_off_H, base_off_H, colb_off);
    else        fml_j[tri_off_H[Hc] + Indx(r, jc)] = v;
  }
}


/* prefill matrices with init contributions */
__global__ void
init_fML_kernel(const size_t ijsize, // 32-bit signed integer overflow bug fix
		int* __restrict__ fml_j) { //out d_fml_j
  const size_t m = (size_t)blockIdx.x*blockDim.x+threadIdx.x; // 32-bit signed integer overflow bug fix
  if(m>=ijsize) return;
  fml_j[m] = INF;
}

__global__ void
init_fml16_kernel(const size_t ijsize, short* __restrict__ j16) {
  const size_t m = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(m>=ijsize) return;
  j16[m] = FML_INF16;
}

__global__ void
init_base_kernel(const size_t n, int* __restrict__ b) {
  const size_t m = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(m>=n) return;
  b[m] = FML_BASE_UNSET;
}

PUBLIC void
init_fML(const int nfiles, const int length,
         const size_t tri_off_H_total, const size_t row_off_H_total) {
  // Staggered_Row_Batching Phase 2d: init_fML() is only ever reached via
  // fill_arrays.c -> par_fill_arrays(), which always runs after par_mfe()
  // has already unconditionally called init_gpu() -- `first` is always
  // already 0 here, same reasoning as int_loop_i()'s equivalent dead-code
  // removal in Phase 2b. The old `if(first) init_gpu(...)` fallback had no
  // way to supply tri_off_H/row_off_H here anyway (only the two totals are
  // passed in, not the full tables init_gpu() needs).
  assert(!first);
  const int first_ = first;
  const size_t ijsize = tri_off_H_total;
  /* Setup execution parameters for helper kernel */
  const size_t nblocks = (ijsize + BLOCK_SIZE - 1)/BLOCK_SIZE; // 32-bit signed integer overflow bug fix
  if(!rnafold_fml_int16()) {
    init_fML_kernel<<<nblocks,BLOCK_SIZE>>>(ijsize, d_fml_j);
  } else {
    // Every cell starts INF, and every baseline starts UNSET so the first
    // non-INF packer of each block establishes it.
    init_fml16_kernel<<<nblocks,BLOCK_SIZE>>>(ijsize, d_fml_j16);
    const size_t nb = (g_base_total + BLOCK_SIZE - 1)/BLOCK_SIZE;
    if(g_base_total) init_base_kernel<<<nb,BLOCK_SIZE>>>(g_base_total, d_fml_b);
  }
  gpuErrchk2( cudaPeekAtLastError(),  first_ );

  //To aid debug etc initialise d_dml (DMLi)
  const size_t hsize = row_off_H_total;
  const size_t nblock2 = (hsize + BLOCK_SIZE - 1)/BLOCK_SIZE; // 32-bit signed integer overflow bug fix
  init_fML_kernel<<<nblock2,BLOCK_SIZE>>>(hsize, d_dml);
  gpuErrchk( cudaPeekAtLastError() );

  // Same INF prefill the host gives DMLi1/fml_prev in fill_arrays.c, and for
  // the same reason: a record that has not joined the sweep yet is never
  // written by any kernel, so the first row it does join must read INF -- the
  // state a single-sequence fold starts from. Prefilling here also means row
  // `length-turn-1` reads a defined d_dml1 on the very first iteration.
  init_fML_kernel<<<nblock2,BLOCK_SIZE>>>(hsize, d_dml1);
  gpuErrchk( cudaPeekAtLastError() );
  /* Stage 3a.1: extra ring slots start as INF, like d_dml1 -- a slot is first read as
   * "the previous row's" before any row has written it in a record that has not joined
   * the sweep. The energy_min extras need it less (each row writes its whole range
   * before md reads it), but a defined value costs one launch per chunk. */
  for (int s = 2; s < g_dml_depth; s++)
    init_fML_kernel<<<nblock2,BLOCK_SIZE>>>(hsize, g_dml_ring[s]);
  for (int s = 1; s < g_emin_depth; s++)
    init_fML_kernel<<<nblock2,BLOCK_SIZE>>>(hsize, g_emin_ring[s]);
  gpuErrchk( cudaPeekAtLastError() );
  init_fML_kernel<<<nblock2,BLOCK_SIZE>>>(hsize, d_fml_prev);
  gpuErrchk( cudaPeekAtLastError() );

#ifndef NDEBUG
  gpuErrchk2( cudaDeviceSynchronize(),first_ );
  //may pickup errors later if dont sync now
#endif
}

//perhaps this can be combined with fmli_kernel?
__global__ void
load_fML_kernel(const int nfiles, const int i_row, const int turn, const int length,
		const int* __restrict__ energy_min,
	              int* __restrict__ fml_j,
	              int* __restrict__ fml_row,   //int16 path: NULL when off
		const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
		const size_t* __restrict__ size_off_H, const size_t total,
		const int* __restrict__ i_H) {
  // The arithmetic lives in md_chain_cells.inc.
  load_fML_cell(nfiles, i_row, turn, length, energy_min, fml_j, fml_row, tri_off_H, row_off_H, size_off_H, total, i_H,
                (size_t)blockIdx.x*blockDim.x+threadIdx.x);
}

PUBLIC void
load_fML(const int nfiles,
	 const int i, const int turn, const int length,
	 const int* energy_min,
	 const size_t* size_off_H) {   //in
  //out d_fml_j
  const size_t total = size_off_H[nfiles];
  if(total==0) return;

  // NB: the old #ifdef NDEBUG pre-sync here was dead code -- this build never
  // defines NDEBUG -- and is superseded anyway: everything in this function
  // now runs on graph_stream, a *blocking* stream, so the legacy-default-
  // stream ordering rule already guarantees init_fML_kernel (NULL stream) has
  // completed before any of this is issued.
  //for simplicity transfer all energy_min, even though only need H * [start:length]
  // GPU-resident sweep: in device mode fml_scan_kernel has already written
  // d_energy_min, so this upload is the round trip being removed. Guarded
  // INSIDE the capture region on purpose -- the mode is constant for the run,
  // so exactly one topology is ever captured and the graph's cheap-update path
  // is unaffected. Check the graph-stats line: the reinstantiate count must not
  // climb.
  if(!rnafold_gpu_sweep())
    int_MemcpyAsync(d_energy_min,energy_min, g_row_total, cudaMemcpyHostToDevice, ISSUE_STREAM, __LINE__);
  bind_row_tables(i);

  /* Setup execution parameters for helper kernel */
  const int nblocks = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
  load_fML_kernel<<<nblocks,BLOCK_SIZE,0,ISSUE_STREAM>>>(nfiles, i, turn, length,
					  d_energy_min,  //in
					  d_fml_j,  //out
					  d_fml_row, //out, int16 path (NULL when off)
					  d_tri_off_H, d_row_off_H,
					  d_size_off_H, total, d_i_H);
  gpuErrchk( cudaPeekAtLastError() );
}

// Closes the row for the int16 path: both writers have run, so d_fml_row holds
// this row's final values and can be packed into the triangle. Shares
// load_fML()'s launch shape because that is the superset of the two write
// ranges. A no-op when the gate is off.
PUBLIC void
pack_fml(const int nfiles, const int i, const int turn, const int length,
         const size_t* size_off_H) {
  if(!rnafold_fml_int16()) return;
  const size_t total = size_off_H[nfiles];
  if(total==0) return;
  bind_row_tables(i);
  const size_t nblocks = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
  pack_fml_kernel<<<nblocks,BLOCK_SIZE,0,ISSUE_STREAM>>>(nfiles, i, turn, length,
                        d_fml_row, d_fml_j16, d_fml_b,
                        d_tri_off_H, d_row_off_H,
                        d_base_off_H, d_colb_off,
                        d_size_off_H, total, d_i_H);
  gpuErrchk( cudaPeekAtLastError() );
}

__global__ void
load_min_fML_kernel(const int nfiles, const int i_row, const int turn, const int length,
		    const int* __restrict__ energy_min,
		    const int* __restrict__ dml,     //in  d_dml   DMLi
		          int* __restrict__ fml_j,     //out d_fml_j my_fML
		          int* __restrict__ fml_row,   //int16 path: NULL when off
		    const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
		    const size_t* __restrict__ side_off_H, const size_t total,
		    const int* __restrict__ i_H) {
  // The arithmetic lives in md_chain_cells.inc.
  load_min_fML_cell(nfiles, i_row, turn, length, energy_min, dml, fml_j, fml_row, tri_off_H, row_off_H, side_off_H, total, i_H,
                    (size_t)blockIdx.x*blockDim.x+threadIdx.x);
}

PUBLIC void
load_min_fML(const int nfiles,
	     const int i, const int turn, const int length,
	     const size_t total) { // Staggered_Row_Batching Phase 4: side_off_H[nfiles] --
	                            // d_side_off_H already uploaded by modular_decomposition_cuda()
	                            // earlier in this row's call sequence, just reused here.
// energy_min already in d_energy_min
// DMLi       already in d_dml
// d_fml_j    out
  if(total==0) return;
  bind_row_tables(i);

/* Setup execution parameters for helper kernel */
  const int nblocks = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
  load_min_fML_kernel<<<nblocks,BLOCK_SIZE,0,ISSUE_STREAM>>>(nfiles, i, turn, length,
					  d_energy_min,  //in
					  d_dml,    //in
					  d_fml_j,  //out
					  d_fml_row, //out, int16 path (NULL when off)
					  d_tri_off_H, d_row_off_H,
					  d_side_off_H, total, d_i_H);
  gpuErrchk( cudaPeekAtLastError() );
}

//indx[n] = n*(n-1)/2
//On GTX 745 no point fml_i in texture as have unified texture/LI cache
//perhaps also fml_j access pattern may not suit texture anyway
__global__ void
fmli_kernel(
  const int nfiles, const int i_row, const int turn, const int length,
        int* __restrict__ fml_i,   //out
  const int* __restrict__ fml_j,   //In  d_fml_j
  const int* __restrict__ fml_row, //In  d_fml_row -- int16 path, NULL when off
  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
  const size_t* __restrict__ side_off_H, const size_t total,
  const int* __restrict__ i_H) {
  // The arithmetic lives in md_chain_cells.inc.
  fmli_cell(nfiles, i_row, turn, length, fml_i, fml_j, fml_row, tri_off_H, row_off_H, side_off_H, total, i_H,
            (size_t)blockIdx.x*blockDim.x+threadIdx.x);
}

//Use __restrict__ to give compiler best chance
// Rewritten 5 Aug 2026 (per Dr. Langdon's Aug 2026 main-branch work, 1893825)
// to exploit H-tightest parallelism: one thread per (H,j) output cell doing a
// serial scan over y, dropping the old block-per-j/reduction-per-block
// scheme entirely. That scheme made sense when parallelism came only from
// splitting one sequence's column sum across a block; with nfiles sequences
// batched, there's now enough independent thread-level parallelism across H
// alone that the reduction/shared-memory machinery is no longer needed. The
// *thread-grid* decomposition below (m/nfiles, H-fastest-varying flat index)
// is still H-tightest for this reason -- only the *data storage* underneath
// (fml_i[row_off_H[H]+y]/fml_j[tri_off_H[H]+yij] below, Staggered_Row_Batching
// Phase 2d) moved off H-tightest, since that's what staggering/mixed lengths
// actually breaks (see the coalescing finding in harmonic-swimming-hare.md).
// Split across TILE threads per output cell, 22 Aug 2026. The Aug 5 rewrite
// above left each thread walking `for(y=0..x)` alone, so the kernel could
// only finish when its *longest* thread did -- a serial chain as long as the
// sequence. ncu confirmed that shape rather than a work shortage: block size
// 32..640 all gave ~250 us (grid 189 -> 10) because every thread was already
// resident, duration scaled linearly with length (124/249/574 us at 300/600/
// 1200) but barely at all with nfiles (239/249/297 us for 5/20/40, i.e. 8x
// the work for +24% time), and SM throughput sat at 9%. Classic critical-path
// bound. TILE lanes now stride the same y range (lane, lane+TILE, ...) and
// combine with a warp shuffle, so the chain shortens to ~x/TILE + log2(TILE)
// while total work is unchanged.
//
// Numerically identical to TILE=1, not merely close: min is associative and
// commutative over exact ints, and a lane with no y of its own contributes
// INF -- the same value the serial loop seeds `value` with -- so the set
// being minimised is identical either way.
//
// Coalescing improves as a side effect. Adjacent threads used to hold
// adjacent j (different cells) and so read fml_j at a stride; now adjacent
// lanes hold adjacent y within one cell, making both fml_i[row_off_H[H]+y]
// and fml_j[tri_off_H[H]+y+ij0] unit-stride across the tile.
// One cell of this kernel, shared through a header (see the header).
#include "md_cell.inc"
extern "C" int rnafold_md_tile_cb(void);
extern "C" int rnafold_md_tile_rb(void);


template <int TILE>
__global__ void
modular_decomposition_kernel(
  const int nfiles, const int i_row, const int turn, const int length,
  const int* __restrict__ fml_i, const int* __restrict__ fml_j,  //In  d_dml_i, d_fml_j
  const short* __restrict__ fml_j16, const int* __restrict__ fml_b, //int16 path
  const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
  int* __restrict__ dml,                            //Out d_dml (h_dml)
  int* __restrict__ fm2,                            //Out d_fm2, or NULL
  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
  const size_t* __restrict__ side_off_H, const size_t total,
  const int* __restrict__ i_H) {
  // The cell/lane split is the only thing that stays here; the arithmetic
  // lives in md_cell.inc.
  const long long gtid = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  md_cell<TILE>(nfiles, i_row, turn, length, fml_i, fml_j, fml_j16, fml_b,
                base_off_H, colb_off, dml, fm2, tri_off_H, row_off_H,
                side_off_H, total, i_H,
                gtid / TILE, (int)(gtid & (TILE-1)));
}

/* RNA_MD_PRUNE (a (min,+) block bound that skipped fml_j reads), RNA_MD_BAND (a
 * global near-diagonal fML cache), RNA_MD_SMEM (fml_i staged in shared memory) and
 * the RNA_MD_TILE widths other than 32 were deleted in the cleanup (C6, D1). Each
 * was byte-identical and measured slower or neutral on the dense md, which sparse
 * md has replaced as the default; the code and the full diagnoses are at the
 * `pre-cleanup` tag. */

/* ===================== CIRCULAR RNA: the persistent fM2_real ==============
 *
 * postprocess_circular() (mfe/mfe.c) reads fM2_real in 13 places and the sweep
 * never filled it -- which is the entire reason -c was declined. It is not new
 * arithmetic: modular_decomposition_kernel already reduces exactly this
 * quantity into DMLi every row and the fork discarded it. See the store in
 * that kernel.
 *
 * The cost is memory, not compute: one triangular int matrix per record, the
 * same extent as fML. Allocated ONLY when md->circ, so a linear fold pays
 * nothing -- which is why this is a runtime pointer rather than a compile flag.
 */
static int    *d_fm2        = NULL;
static size_t  g_fm2_cells  = 0;

extern "C" int *
rnafold_circ_fm2_device(void)
{
  return d_fm2;
}


extern "C" void
rnafold_circ_free(void)
{
  if(d_fm2) cudaFree(d_fm2);
  d_fm2 = NULL;
  g_fm2_cells = 0;
}


/* N5a (bt_retired.h): hand this chunk's fML -- whichever layout was ALLOCATED,
 * read off the pointers, never off rnafold_fml_int16() -- and, circular, its
 * fM2_real to a retired set. NULLed here, so teardown_gpu() and the next
 * rnafold_circ_alloc() skip them. d_colb_off stays: the host decode recomputes
 * the column offsets and never reads it. */
extern "C" void
rnafold_fml_detach(rnafold_retired_t *r)
{
  r->d_fml_j      = d_fml_j;      d_fml_j      = NULL;
  r->d_fml_j16    = d_fml_j16;    d_fml_j16    = NULL;
  r->d_fml_b      = d_fml_b;      d_fml_b      = NULL;
  r->d_base_off_H = d_base_off_H; d_base_off_H = NULL;
  r->d_fm2        = d_fm2;        d_fm2        = NULL;
  g_fm2_cells = 0;
}


/* Allocate and prefill. `circ` 0 frees and returns 0; returns 1 on success,
 * -1 on failure. Prefill is INF because upstream prefills fM2_real to INF and
 * the kernel only writes cells with j >= i+2*turn+3 -- every shorter span has
 * no two-branch decomposition and must stay INF. */
extern "C" int
rnafold_circ_alloc(const int circ, const size_t tri_cells)
{
  rnafold_circ_free();

  if((!circ) || (tri_cells == 0))
    return 0;

  if(cudaMalloc((void **)&d_fm2, tri_cells * sizeof(int)) != cudaSuccess) {
    fprintf(stderr, "modular_decomposition.cu  cudaMalloc failed for fM2_real "
                    "(%zu cells, %.1f MB)\n", tri_cells,
            tri_cells * sizeof(int) / 1048576.0);
    d_fm2 = NULL;
    return -1;
  }

  g_fm2_cells = tri_cells;
  {
    const size_t nb = (tri_cells + BLOCK_SIZE - 1)/BLOCK_SIZE;
    init_fML_kernel<<<nb,BLOCK_SIZE>>>(tri_cells, d_fm2);   /* fills INF */
    gpuErrchk( cudaPeekAtLastError() );
  }

  if (!vrna_cuda_quiet()) fprintf(stderr, "modular_decomposition.cu  circular: fM2_real allocated, "
                  "%zu cells, %.1f MB\n", tri_cells,
          tri_cells * sizeof(int) / 1048576.0);

  return 1;
}


/* One record's fM2_real triangle, the twin of fetch_my_c_one(). */
extern "C" void
fetch_fm2_one_w(int *dst, const size_t tri_lo, const size_t cells, const int w)
{
  if(!d_fm2) {
    fprintf(stderr, "modular_decomposition.cu  fetch_fm2_one with no fM2 "
                    "allocated -- the circular guard has a hole\n");
    return;
  }

  d2h_w(dst, &d_fm2[tri_lo], cells*sizeof(int), w);
}

void rnafold_d2h_nb_w(void* dst, const void* src, const size_t bytes, const int w);

extern "C" void
rnafold_retired_fetch_fm2_w(const rnafold_retired_t *r, int *dst,
                            const size_t tri_lo, const size_t cells, const int w)
{
  assert(r && r->d_fm2 && (tri_lo + cells <= r->cells));
  rnafold_d2h_nb_w(dst, &r->d_fm2[tri_lo], cells*sizeof(int), w);
}

extern "C" void
fetch_fm2_one(int *dst, const size_t tri_lo, const size_t cells)
{
  fetch_fm2_one_w(dst, tri_lo, cells, -1);
}


void modular_decomposition_cuda(const int nfiles,
				const int i, const int turn, const int length,
			      //const int* indx,
			      //const int* my_fML,
				      int* DMLi,
				const size_t* row_off_H,
				const size_t* side_off_H,
				const int* i_H) { //in, nfiles entries -- per-record row index
  //printf("\nmodular_decomposition_cuda(%d,%d,%d,indx,my_fML)\n",
  //	 i,turn,length);

  // Staggered_Row_Batching Phase 4: side_off_H[nfiles] replaces the old
  // scalar side<=0 check -- numerically identical while every H shares one
  // length (today), but now table-driven like everything feeding it.
  const size_t total = side_off_H[nfiles];

  if(total == 0) {
    for(int H=0; H<nfiles;H++) {
    // Staggered_Row_Batching Phase 6d: bound this fill by H's OWN row width,
    // not the shared `length`. Each H's row slot is exactly
    // row_off_H[H+1]-row_off_H[H] == VC[H]->length+1 entries (Phase 2a built
    // the table that way), so the shared bound would run off the end of a
    // short H's row and into the next H's. VC[] isn't in scope here, so the
    // per-H length is recovered from the table itself rather than threading a
    // new parameter through modular_decomposition_i()'s public signature.
    const int len_H = (int)(row_off_H[H+1] - row_off_H[H]) - 1;
    // This record's own row (i_H, equal to i); a record with no row (i_H[H] < 1)
    // is skipped so its row buffer keeps what its last row left there.
    const int i_h = i_H[H];
    if(i_h < 1) continue;
    for (int j = i_h+turn+1; j <= len_H; j++) {
      DMLi[row_off_H[H]+j] = INF; // Staggered_Row_Batching Phase 2d: table-driven per-H row offset
    }}
    return;
  }

  // Staggered_Row_Batching Phase 4: d_side_off_H uploaded once here -- both
  // fmli_kernel below and load_min_fML_kernel (called later this row, from
  // load_min_fML()) read the same already-uploaded table, so it's not
  // re-uploaded there.
  bind_row_tables(i);

  //for simplicity transfer all to start with
  //perhaps should use cuMemsetD32
  //for (int j = 0; j <= length; j++) DMLi[j] = INF;
  //int_Memcpy(d_dml,h_dml,length+1, cudaMemcpyHostToDevice,__LINE__);


  /*now done in fmli_kernel
  //in first version just use ith column and multiple columns of my_fML
  //start off by transfering all of my_fML which we might read to GPU.
  //Could just transfer part which has changed, or get kernel to set it
  int m;
  int* fml_i = (int*)malloc(side*sizeof(int));
  for(m=0;m<side;m++) {
    const int k  = start + m;
    const int ik = indx[k] + i;
    fml_i[m] = my_fML[ik]; //ith column
    //printf("modular_decomposition_cuda(%d,%d,%d,indx,my_fML) fml_i[%d]%d <= my_fML[%d]\n",
    //	   i,turn,length,
    //	   m,fml_i[m],ik);
  }
  int_Memcpy(d_fml_i,fml_i, side, cudaMemcpyHostToDevice,__LINE__);
  */

  //these parts of my_fML re-used as i reduces to 1
  //const int ij_size  = side*(side+1)/2;
  //assert(ij_size<=ijsize_min);
  //int* d_fml_j = &d_dml[(length+1) + side];
  //int*   fml_j = &h_dml[(length+1) + side]; //(int*)malloc(ij_size*sizeof(int));
  /*
  m=0;
  for(int j=0; j <  side;j++) {
    const int ij  = indx[i + 2*(turn + 1) + 1 + j] + i;
    const int k1j = ij + turn + 2; //indx[j] + i + 1; //indx[j] + i + turn + 2;
    for(int k=0; k <= j   ;k++,m++) {
      fml_j[m] = my_fML[k+k1j]; //covers mutiple columns
      //printf("modular_decomposition_cuda(i=%d,%d,%d,indx,my_fML) j %d k %d fml_j[%d]%d <= my_fML[%d]\n",
      //     i,turn,length,
      //     j,k,
      //     m,fml_j[m],k+k1j);
      assert(m<ij_size);
//    fml_j[m] = ((fml_i[k] != INF ) &&   (my_fML[k+k1j] != INF))?   fml_i[k] + my_fML[k+k1j] : INF;
    }
  }*/
  //int_Memcpy(d_dml,h_dml, (length+1) + side + ij_size, cudaMemcpyHostToDevice,__LINE__);
  //const int ij_size = ((length+1)*(length+2)/2); //better not to transfer whole of my_fML but will do for the time being
  //int_Memcpy(d_dml,h_dml, 2*(length+1) + ij_size, cudaMemcpyHostToDevice,__LINE__);

  //Gave up trying to page lock my_fML, instead copy it directly
  /**Now keeping my_fML in sync with d_fml_j elsewhere
  {
  const int j = (i + 2*(turn+1)) + 1; //first used value of j
  const int start =      j*(     j-1)/2 + i      + (turn+1) + 1;
  const int top   = length*(length-1)/2 + length - (turn+1);
  const int size  = top - start + 1;
  int_Memcpy(&d_fml_j[start],&my_fML[start], size, cudaMemcpyHostToDevice,__LINE__);
  }
  */
//gpuErrchk( cudaDeviceSynchronize() );

  /* RNA_MD_TAIL: fml_i is not a buffer md needs. fmli_cell copies
   * fml_i[row_off+y] = energy_min[row_off + i+turn+1 + y] (through d_fml_row or
   * the triangle, both of which load_fML had just filled FROM energy_min), so in
   * lock-step -- one i for every record, which md_tail_refuse() checks -- the
   * same values are energy_min shifted by i+turn+1. md's own indexing is
   * untouched; only the base pointer moves. */
  const int* md_fml_i = g_md_tail_now ? (const int*)(d_energy_min + (i+turn+1)) : d_fml_i;
  if(!g_md_tail_now)
  { /* Setup execution parameters for helper kernel */
    // g_block_size_fmli: chosen once in init_gpu() -- see the forward
    // declarations near the top of this file for why.
    const int block_size = g_block_size_fmli;
    const int nblocks = (total + block_size - 1)/block_size;
    fmli_kernel<<<nblocks,block_size,0,ISSUE_STREAM>>>(nfiles, i, turn, length,
					d_fml_i,  //Out
					d_fml_j,  //In
					d_fml_row, //In, int16 path (NULL when off)
					d_tri_off_H, d_row_off_H,
					d_side_off_H, total, d_i_H);
    gpuErrchk( cudaPeekAtLastError() );
  }


//const int todo = side*(side+1)/2;


//  printf("modular_decomposition_cuda(i=%d,%d,%d,indx,my_fML) start %d stop %d threads needed %d\n",
//	 i,turn,length,
//	 start, stop,todo);

  //modular_decomposition_kernel(side,i,turn,length,fml_i,fml_j); //host testing

  /* Setup execution parameters */
  // g_block_size_md: chosen once in init_gpu() -- see the forward
  // declarations near the top of this file for why. (The serial per-thread
  // `for y` scan below is a separate, algorithmic imbalance this doesn't
  // address -- see the NCU report follow-up.)
  const int block_size = g_block_size_md;
  //const int nblocks = (todo + block_size - 1)/block_size; //for time being waste many blocks
  //const int nblocks = side;
  // MD_TILE threads per output cell, not one -- see the kernel comment.
  // Done in size_t because total already is: at the widest chunks measured so
  // far (nfiles ~ 2000) total x 32 comfortably exceeds 2^31.
  const size_t nthreads   = total * (size_t)MD_TILE;
  const size_t nblocks_sz = (nthreads + (size_t)block_size - 1)/(size_t)block_size;
  assert(nblocks_sz <= 2147483647u); //gridDim.x limit
  const int nblocks = (int)nblocks_sz;



  /* RNA_MD_SPARSE: candidates + a row scan in place of the dense launch (md_sparse.inc) */
  if(md_sparse_live()) md_sparse_md(nfiles, turn, md_fml_i, total);
  else
    modular_decomposition_kernel<MD_TILE><<<nblocks,block_size,0,ISSUE_STREAM>>>(
                nfiles, i, turn, length,
                md_fml_i, d_fml_j,
                d_fml_j16, d_fml_b, d_base_off_H, d_colb_off,
                d_dml,   /*Out*/
                rnafold_circ_fm2_device(), /*Out, NULL unless circular*/
                d_tri_off_H, d_row_off_H,
                d_side_off_H, total, d_i_H);

  gpuErrchk( cudaPeekAtLastError() );

  //for effiency transfer all of DMLi rather that just those part that have been calculated
  // GPU-resident sweep: DMLi's only readers were new_c_host (via DMLi1) and
  // fml_prev_host, both skipped in device mode; new_c_kernel reads d_dml1 and
  // fml_prev_kernel reads d_dml, both device-side.
  if(!rnafold_gpu_sweep())
    int_MemcpyAsync(DMLi,d_dml, g_row_total, cudaMemcpyDeviceToHost, ISSUE_STREAM, __LINE__);
  // no sync here -- the one remaining sync happens once, after the whole
  // captured chain is launched, in the new orchestration function.

  //free(fml_j);
  //free(fml_i);
}

      /* modular decomposition -------------------------------**
      const int decomp = modular_decomposition(i,ij,j,turn,Fmi,my_fML);
      ** modular decomposition -------------------------------*/
PUBLIC int
modular_decomposition_ij(const int i, const int j, const int turn, 
		      const int length,
		      const int* indx, 
		      const int ijsize,
 		      const int* fmi,   //better cache access
		      const int* my_fML) {
  {
      int ij            = indx[j]+i;
//    assert(ij>=0 && ij<ijsize);
{
  int k   =  i + turn + 1;
  int k1j = ij + turn + 2; //indx[j] + i + 1; //indx[j] + i + turn + 2;
  const int stop = j - 2 - turn;
  int decomp = INF;

  for (; k <= stop; k++, k1j++){
    //assert(k  >0 &&   k<=length); //starts at 1 not 0
    //assert(k1j>0 && k1j<ijsize ); //starts at 1 not 0

    //const int ik = indx[k] + i;
    //assert(ik>0 && ik<ijsize  ); //starts at 1 not 0
    //assert(my_fML[ik] == Fmi[k]);

  //if((my_fML[ik] != INF ) && (my_fML[k1j] != INF)){
    if((   fmi[k]  != INF ) && (my_fML[k1j] != INF)){
    //const int en = my_fML[ik] + my_fML[k1j];
      const int en = fmi[k] + my_fML[k1j];
      decomp = MIN2(decomp, en);
    }
    //if(i==2895)
    //printf("modular_decomposition_ij(%d,%d,%d,%d,indx,%d,my_fML) j=%d stop=%d k=%d fmi[%d]=%d my_fML[%d] %d decomp %d\n",
    //	   i,j,turn,length,ijsize,
    //	   j,stop,k,
    //	   k,fmi[k], k1j,my_fML[k1j], decomp);
  }
  //assert(((k+1) <= j - 2 - turn)==0);
  return decomp;
  /* End modular decomposition -------------------------------*/
}
    }
}
/**/
//C interface to CUDA code
extern "C" /*PUBLIC*/ void
modular_decomposition_i(const int nfiles,
			const int i, const int turn, const int length,
		      //const int* indx,
		      //const int ijsize,
		      //const int* my_fML, //In
			int* DMLi,         //Out
			const size_t* row_off_H,
			const size_t* side_off_H,
			const int* i_H) {  //in, nfiles entries -- per-record row index
  //const int max_ij_len = length - 2*(turn+1) - 1;
  //const int ijsize_min = max_ij_len*(max_ij_len+1)/2;
  //if(first) init_gpu(length);
  modular_decomposition_cuda(nfiles,i,turn,length, DMLi, row_off_H, side_off_H, i_H);
  return;

  /* remove debug** {
    int* Fmi   = (int *) malloc(sizeof(int)*(length + 1)); // holds row i of fML (avoids jumps in memory)
    const int start = i+turn+1;
    const int stop  = length - 2 - turn;
    int j;
    for (j = start; j <= stop; j++) {
      const int ij = indx[j] + i;
      Fmi[j] = my_fML[ij]; //ith column
    }
    int* c_DMLi = (int*)calloc(length+1,sizeof(int));
    int err = 0;
    for (j = i+turn+1; j <= length; j++) {
      c_DMLi[j] = modular_decomposition_ij(i,j,turn,length,indx,ijsize,Fmi,my_fML);
      ** remove debug**
      fprintf(stdout,"modular_decomposition_i(i=%d,%d,%d,indx,%d,my_fML,DMLi) DMLi[%d] %d v h_dml[%d] %d\n",
	      i,turn,length,ijsize,
	      j,c_DMLi[j],j,DMLi[j]);
      **
      if(c_DMLi[j]!=DMLi[j]) err = j;
    }
    if(err) {
      fprintf(stdout,"ERROR in modular_decomposition_i(i=%d,%d,%d,indx,%d,my_fML,DMLi) DMLi[%d]%d != h_dml[%d]%d\n",
	      i,turn,length,ijsize,
	      err,c_DMLi[err],err,DMLi[err]);
      fprintf(stderr,"ERROR in modular_decomposition_i(i=%d,%d,%d,indx,%d,my_fML,DMLi) DMLi[%d]%d != h_dml[%d]%d\n",
	      i,turn,length,ijsize,
	      err,c_DMLi[err],err,DMLi[err]);
      exit(1);
    } else {
      const int j = i+turn+1;
      memcpy(&DMLi[j],&c_DMLi[j],(length-j+1)*sizeof(int));
    }
    free(c_DMLi);
  }
  */
}

// Copies the GPU's my_fML triangle back into each record's own
// VC[H]->matrices->fML, once, after the whole sweep. Replaces the per-row
// host mirroring my_fml_update_host used to do (see fill_arrays_loop.c):
// nothing between rows reads the host triangle, and backtrack() -- the only
// consumer that does -- runs after this.
//
// The copy is contiguous per record, not scattered, because the two layouts
// already agree: ViennaRNA indexes fML by jindx[j]+i == j*(j-1)/2+i, which is
// exactly this file's Indx(i,j), and dp_matrices.c sizes the allocation
// (n+1)*(n+2)/2 == tri_off_H[H+1]-tri_off_H[H] (compute_batch_offsets(),
// mfe_cuda.c). So H's slice of d_fml_j maps onto its fML one-for-one.
//
// Synchronous on the NULL stream: this runs once per chunk, immediately
// before the host reads the result, so there is nothing to overlap it with.
// One record's slice, for the scratch-pool path: the caller backtracks records
// one at a time into a reusable buffer instead of holding every record's
// triangle at once. Same copy as fetch_fML() does per H, just addressable
// individually. Safe to call from several host threads -- the CUDA runtime is
// thread-safe and these serialise on the default stream.
// GPU-resident sweep, step 1: hand this file's row-shaped device buffers to
// the kernels that will live in hp_mb_loop.cu. Same shape as int_loop.cu's
// int_loop_hccc_buffers() -- the device state stays owned by the file that
// allocates it, and only a pointer crosses the translation-unit boundary.
// Valid only between init_gpu() and teardown_gpu().
extern "C" /*PUBLIC*/ void
md_row_buffers(int** dml_out, int** dml1_out, int** fml_prev_out,
               int** energy_min_out) {
  if(dml_out)        *dml_out        = d_dml;
  if(dml1_out)       *dml1_out       = d_dml1;
  if(fml_prev_out)   *fml_prev_out   = d_fml_prev;
  if(energy_min_out) *energy_min_out = d_energy_min;
}

// Publishes row i's DMLi as "the previous row's" for row i-1, replacing the
// host's DMLi1 rotation (see the declaration comment for why a copy and not a
// rotate). Called from fill_arrays_loop.c at exactly the point the host
// rotates, so the two stay in lockstep. Synchronous on the NULL stream: the
// next row's first kernel must see it, and there is nothing to overlap it
// with here.
extern "C" /*PUBLIC*/ void
md_snapshot_dml(void) {
  // The tail of the md chain. A BLOCKING cudaMemcpy here would serialise the
  // two streams every row whatever the events say, so at level 2 it is async on
  // the md stream and publishes the event new_c(i-1) waits on.
  //
  // ORDERING FIX (2026-09-29): below level 2 this used to be a plain cudaMemcpy
  // to the LEGACY default stream. At level 1 -- the default -- new_c runs on
  // g_stream_cell, created NON-BLOCKING, so it is not ordered against the legacy
  // stream at all, and a device-to-device cudaMemcpy does not block the host.
  // Only the host syncs elsewhere in the row (md's per-row stream sync, and
  // load_my_c's) kept new_c(i-1) from reading d_dml1 before this copy landed --
  // the same class of hole the g_issue_stream fix above closed for the
  // graph-off path. On rnafold_stream_md() it is ordered at every level: at 0
  // that IS the legacy stream, at 1 it is the stream new_c runs on, at 2 the
  // event covers it.
  //
  // RNA_MD_TAIL: and no copy at all -- the ROTATE the declaration comment above
  // declined. Its reason was to keep the captured graph's parameters fixed, but
  // they change every row regardless (i does), so a swapped pointer is one more
  // parameter in an update that happens anyway, never a reinstantiate. Check
  // the graph-stats line: the reinstantiate count must stay where it was.
  // Equivalence: d_dml1 must hold row i's DMLi for new_c(i-1), and md(i-1) then
  // overwrites d_dml. Swapping gives exactly that PROVIDED md(i-1) writes every
  // cell a later reader takes from d_dml, and it does: a row's side range
  // contains the previous row's, cells below it are never written (INF in both
  // buffers from init_fML), and a record outside the sweep has
  // width 0 until it joins. Readers fetch both pointers per launch through
  // md_row_buffers().
  if(g_md_tail_row >= 0) {
    if(g_dml_depth == 2) {
      /* depth 2: today's swap, the same statement as before stage 3a.1 */
      int *t = d_dml; d_dml = d_dml1; d_dml1 = t;
    } else {
      /* Stage 3a.1, depth N: row i's slot becomes "the previous row's" (d_dml1)
       * and the next slot -- last written N-1 rows ago, every cell of it inside the
       * range row i-1 is about to write -- becomes d_dml. energy_min advances with
       * it, one slot shallower, so md(i-1) reads row i-1's own fML. */
      g_ring_k++;
      d_dml1       = g_dml_ring[(g_ring_k - 1) % g_dml_depth];
      d_dml        = g_dml_ring[g_ring_k % g_dml_depth];
      d_energy_min = g_emin_ring[g_ring_k % g_emin_depth];
    }
    g_md_tail_row = -1;
  } else {
    gpuErrchk( cudaMemcpyAsync(d_dml1, d_dml, g_row_total*sizeof(int),
                               cudaMemcpyDeviceToDevice, rnafold_stream_md()) );
  }
  rnafold_stream_md_done();   // a no-op below level 2
}

// Widens on the way out when the gate is on. The host's fML is int32 and every
// consumer of it -- backtracking, subopt, the fM1 post-pass -- expects that, so
// the encoding stops at the device boundary. `H` is needed for the baseline
// offset, which the int32 path did not require, hence the extra argument.
extern "C" /*PUBLIC*/ void
fetch_fML_one_H(int* dst, const size_t tri_lo, const size_t cells, const int H);


extern "C" /*PUBLIC*/ void
fetch_fML_one(int* dst, const size_t tri_lo, const size_t cells) {
  assert(!rnafold_fml_int16());   // callers must use fetch_fML_one_H()
  d2h_w(dst, &d_fml_j[tri_lo], cells*sizeof(int), -1);
}

// Host-side decode. Deliberately NOT a kernel: it runs once per record on the
// backtrack pool, off the critical path, and a kernel would need a scratch
// int32 triangle on the device -- which is exactly the allocation this whole
// change exists to remove.
//
// N5a: the body takes its buffers and its copy as ARGUMENTS, so the live chunk
// (globals, g_xfer[]) and a retired one (bt_retired.h, g_xfer_nb[]) share every
// line of the decode. `j16` non-NULL selects the int16 layout.
typedef void (*fml_d2h_t)(void*, const void*, const size_t, const int);

static void
fml_fetch_decode(int* dst, const size_t tri_lo, const size_t cells, const int H, const int w,
                 const int* j32, const short* j16, const int* fb, const size_t* boff,
                 fml_d2h_t d2h) {
  if(!j16) { d2h(dst, &j32[tri_lo], cells*sizeof(int), w); return; }

  static __thread short*  h16  = NULL;
  static __thread size_t  h16n = 0;
  if(h16n < cells) { free(h16); h16 = (short*)malloc(cells*sizeof(short)); h16n = cells; }

  d2h(h16, &j16[tri_lo], cells*sizeof(short), w);

  // The baselines for this record, and the column offsets, both small.
  size_t base_lo = 0, base_n = 0;
  d2h(&base_lo, &boff[H], sizeof(size_t), w);
  { size_t hi; d2h(&hi, &boff[H+1], sizeof(size_t), w);
    base_n = hi - base_lo; }
  int* hb = (int*)malloc((base_n?base_n:1)*sizeof(int));
  if(base_n) d2h(hb, &fb[base_lo], base_n*sizeof(int), w);

  // n from the triangle size: cells == (n+1)(n+2)/2
  int n = 0; while(((size_t)(n+1)*(n+2))/2 < cells) n++;
  size_t* colb = (size_t*)malloc((size_t)(n+2)*sizeof(size_t));
  colb[0] = colb[1] = 0;
  for(int j=1;j<=n;j++) colb[j+1] = colb[j] + (size_t)((j + FML_BLK - 1)/FML_BLK);

  // The baseline is constant across FML_BLK consecutive i, so load it once per
  // block rather than once per cell. That one dependent load per cell WAS the
  // cost of this function: 2.46 -> 0.46 ns/cell, 5.3x, which over 400 x 5601 is
  // 15.4 -> 2.9 worker-seconds. int16 already saves ~2.1 s of PCIe by moving
  // half the bytes, so this is what turns the int16 fML fetch from more
  // expensive than the int32 one into less. See docs/archive/STRESS272_RESULTS.md 15.3.
  //
  // tools/fml_decode_equiv.c holds the old loop verbatim as the reference and
  // compares the two over 1500 shapes -- including the ragged ones where
  // `cells` cuts the last row, which is the only reason the old loop needed a
  // per-cell bound test. It goes RED (1151/1500) on a one-off block boundary.
  //
  // `i` advances only in the inner loop, so this terminates only while
  // end >= i. That holds because end comes from i's own block:
  // ((i-1)/BLK + 1)*BLK > i-1, hence >= i. A red-team mutation that shifted the
  // block index by 7 instead of 6 made this spin rather than answer wrongly --
  // do not restructure `end` without re-checking it.
  for(int j=1;j<=n;j++) {
    const size_t row = (size_t)j*(j-1)/2;
    if(row >= cells) break;              // rows only grow: nothing after is ours
    int hi = j;
    if(row + (size_t)hi >= cells) hi = (int)(cells - row) - 1;
    const int*   b   = hb + colb[j];
    const short* src = h16 + row;
    int*         out = dst + row;
    for(int i=1;i<=hi;) {
      const int blk  = (i-1)/FML_BLK;
      const int base = b[blk];
      int       end  = (blk+1)*FML_BLK;  // last i sharing this baseline
      if(end > hi) end = hi;
      for(; i<=end; i++) {
        const short o = src[i];
        out[i] = (o == FML_INF16) ? INF : base + (int)o;
      }
    }
  }
  free(hb); free(colb);
}

extern "C" /*PUBLIC*/ void
fetch_fML_one_Hw(int* dst, const size_t tri_lo, const size_t cells, const int H, const int w) {
  fml_fetch_decode(dst, tri_lo, cells, H, w, d_fml_j,
                   rnafold_fml_int16() ? d_fml_j16 : NULL, d_fml_b, d_base_off_H,
                   rnafold_d2h_w);
}

extern "C" void
rnafold_retired_fetch_fml_w(const rnafold_retired_t *r, int *dst,
                            const size_t tri_lo, const size_t cells,
                            const int H, const int w) {
  assert(r && (r->d_fml_j || r->d_fml_j16) && (tri_lo + cells <= r->cells) && (H < r->nslots));
  fml_fetch_decode(dst, tri_lo, cells, H, w, r->d_fml_j, r->d_fml_j16, r->d_fml_b,
                   r->d_base_off_H, rnafold_d2h_nb_w);
}

extern "C" /*PUBLIC*/ void
fetch_fML_one_H(int* dst, const size_t tri_lo, const size_t cells, const int H) {
  fetch_fML_one_Hw(dst, tri_lo, cells, H, -1);
}

extern "C" /*PUBLIC*/ void
fetch_fML(const int nfiles, int** fML_H, const size_t* tri_off_H) {
  for(int H=0; H<nfiles; H++) {
    const size_t n = tri_off_H[H+1] - tri_off_H[H];
    assert(n > 0);
    // Routed through the decoding form so the bulk fetch cannot quietly become
    // the one path that reads raw offsets as energies.
    fetch_fML_one_H(fML_H[H], tri_off_H[H], n, H);
  }
}

/* ===================== RNA_MD_ROW_SYNC and RNA_MD_TAIL =====================
 *
 * Two changes to the md chain's per-row shape, both answer-neutral by construction
 * (2026-09-29, from a call-order audit of the default path). BOTH DEFAULT ON SINCE
 * 2026-09-30: with RNA_LOAD_MY_C_SYNC they are -5.3 % at 400 x 5601 on an A100,
 * byte-identical in every arm against the CPU (docs/archive/PORT_LUKES_FLOW_BATCHING.md 9.12).
 * Each has an env var that restores the old path, and announces itself only when set.
 *
 * md's per-row cudaStreamSynchronize below level 2 is now OFF; RNA_MD_ROW_SYNC=1
 * restores it. Forced ON off the GPU-resident sweep, where the host reads DMLi back
 * and needs the copy to have landed.
 *
 * The collapsed tail replaces
 *     load_fML -> fmli -> md -> load_min_fML -> pack_fml     (graph)
 *     fml_prev                                                (md stream)
 *     md_snapshot_dml: a whole-row D2D copy                   (md stream)
 * with
 *     md (reading fml_i out of energy_min) -> md_close_row   (graph)
 *     a d_dml/d_dml1 pointer swap                             (host)
 * because fmli only copies energy_min, and load_min_fML and fml_prev compute the
 * same min(energy_min, DMLi). Two GPU operations instead of seven. RNA_MD_TAIL=0
 * restores the old chain; md_tail_refuse() still declines it where it cannot apply. */
extern "C" int
rnafold_md_row_sync(void)
{
  static int v = -1;

  if(v < 0) {
    const char* e = getenv("RNA_MD_ROW_SYNC");
    v = (e && e[0] == '1') ? 1 : 0;
    if(!v && !rnafold_gpu_sweep()) {
      if(e && e[0])
        fprintf(stderr,"%-24s RNA_MD_ROW_SYNC=0 ignored: off the GPU-resident sweep "
                       "the host reads DMLi back every row\n", __FILE__);
      v = 1;
    }
    if(e && e[0])
      fprintf(stderr,"%-24s RNA_MD_ROW_SYNC=%d: %s\n", __FILE__, v,
              v ? "a host sync after the md chain every sweep row (the pre-2026-09-30 default)"
                : "no host sync after the md chain (the default)");
  }
  return v;
}

extern "C" int
rnafold_md_tail(void)
{
  static int v = -1;

  if(v < 0) {
    const char* e = getenv("RNA_MD_TAIL");
    v = (e && e[0] == '0') ? 0 : 1;
    if(!v)
      fprintf(stderr,"%-24s RNA_MD_TAIL=0: the old seven-operation md tail "
                     "(the pre-2026-09-30 default)\n", __FILE__);
  }
  return v;
}

/* NULL when the collapsed tail may run this row, else why not: off the
 * GPU-resident sweep, or a schedule where records sit on different rows and a
 * single base pointer cannot stand in for fml_i. */
static const char*
md_tail_refuse(const int nfiles, const int i, const int* i_H)
{
  if(!rnafold_gpu_sweep())        return "off the GPU-resident sweep the host rotates DMLi itself";
  for(int H = 0; H < nfiles; H++)
    if(i_H[H] != i)               return "records are on different rows";
  return NULL;
}

/* Stage 3a.1: RNA_MD_RING_DEPTH=N, the depth of the dml and cc rings (energy_min's is
 * N-1). Default 2, today's swap. A diagnostic until stage 3b sets it from RB. */
extern "C" int
rnafold_md_ring_depth(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_MD_RING_DEPTH");

    v = (e && e[0]) ? atoi(e) : 2;
    if (v < 2)           v = 2;
    if (v > MD_RING_MAX) v = MD_RING_MAX;
    /* stage 3b: column-block-major order with RB rows per block-row reads row i+1's
     * dml and cc and row i's energy_min from RB+1 distinct rows -- the ring must hold them */
    if (rnafold_md_tile_cb() && v < rnafold_md_tile_rb() + 1) v = rnafold_md_tile_rb() + 1;
    if (e && e[0])
      fprintf(stderr, "%-24s RNA_MD_RING_DEPTH=%d: the dml and cc rings hold %d rows and "
                      "energy_min's %d (the row path; byte-identical by construction)\n",
              __FILE__, v, v, v - 1);
  }
  return v;
}

/* This chunk's depth, as init_gpu() settled it -- for hp_mb_loop.cu's cc ring, which
 * must rotate in step with the dml ring (init_gpu3() runs after init_gpu()). */
extern "C" int
rnafold_md_ring_depth_now(void)
{
  return g_dml_depth;
}

/* This chunk's depth: the requested one only where every reader of the rotating rows
 * fetches them per launch, which is exactly where the collapsed md tail runs. Anywhere
 * else the rows are copied rather than rotated (md_snapshot_dml()'s other branch), so
 * depth 2 is kept and said. */
static int
md_ring_depth_effective(void)
{
  const int   want = rnafold_md_ring_depth();
  const char *why  = NULL;

  if (want == 2)                                       return 2;
  if (!rnafold_md_tail())                              why = "RNA_MD_TAIL=0 copies DMLi instead of rotating it";
  else if (!rnafold_gpu_sweep())                       why = "off the GPU-resident sweep the host rotates DMLi itself";
  if (why) {
    static int said = 0;
    if (!said) {
      said = 1;
      fprintf(stderr, "%-24s RNA_MD_RING_DEPTH=%d REFUSED: %s -- depth 2\n",
              __FILE__, want, why);
    }
    return 2;
  }
  return want;
}

/* Stage 3b: select the rotating rows BY ROW rather than by advancing. In column-block-major
 * order the rows of a block-row interleave across column blocks, so "the next slot" is
 * meaningless; what each step needs is fixed by its row: row i's fM2 and fML (written),
 * row i+1's fM2 (new_c reads it at j-1). Slot = row mod depth, so the RB rows of a
 * block-row and the row above them occupy RB+1 distinct slots when depth >= RB+1, and a
 * slot's previous occupant (row i+depth) belongs to a block-row that is finished. Its
 * stale cells are all overwritten before they are read, by the argument
 * md_snapshot_dml() makes for depth 2 (row i's range contains every lower row's). */
extern "C" void
rnafold_md_ring_select(const int i)
{
  if (g_dml_depth < rnafold_md_tile_rb() + 1) {
    fprintf(stderr, "%-24s md ring depth %d < RB+1 = %d: two rows of a block-row would "
                    "share a slot\n", __FILE__, g_dml_depth, rnafold_md_tile_rb() + 1);
    exit(EXIT_FAILURE);
  }
  d_dml        = g_dml_ring[i % g_dml_depth];
  d_dml1       = g_dml_ring[(i + 1) % g_dml_depth];
  d_energy_min = g_emin_ring[i % g_emin_depth];
}

/* Is row i's fml_prev already written? fml_prev_i() (hp_mb_loop.cu) asks, and
 * skips its launch only on a yes from the code that did the work. */
extern "C" int
rnafold_md_tail_closed_row(const int i)
{
  return g_md_tail_row == i;
}

static void
md_close_row(const int nfiles, const int i, const int turn, const size_t* size_off_H)
{
  const size_t total = size_off_H[nfiles];
  if(total == 0) return;
  bind_row_tables(i);
  const int nblocks = (int)((total + BLOCK_SIZE - 1)/BLOCK_SIZE);
  int* stage = (g_stage_K >= 2) ? d_fml_stage + (size_t)(i % g_stage_K) * g_row_total : NULL;
  md_close_row_kernel<<<nblocks,BLOCK_SIZE,0,ISSUE_STREAM>>>(nfiles, i, turn,
      d_energy_min, d_dml, d_fml_prev, d_fml_j, d_fml_j16, d_fml_b,
      d_tri_off_H, d_row_off_H, d_base_off_H, d_colb_off,
      d_size_off_H, total, d_i_H, stage);
  gpuErrchk( cudaPeekAtLastError() );
}

/* Sparse multiloop decomposition, PORT_SPARSE_MD.md (stage S1: selftest only). */
#include "md_sparse.inc"

/* RNA_TRI_CHECKSUM=1: at the end of each chunk's sweep, hash the WHOLE c and fML
 * triangles and print the hashes.
 *
 * WHY: a structure-level comparison cannot see some defects. The first negative
 * control for RNA_ROW_BATCH removed the turn+2 clamp so md read fML rows still sitting
 * in the stage -- and every output still matched, because the candidates it lost
 * (fML[i][i+4..i+6], segments too short for anything but a minimal hairpin) essentially
 * never win the minimum. Equal triangles is the stronger bar: any change that only
 * reorders writes must leave every cell, winning or not, bit-identical. Compare runs
 * of the same encoding (int16 and int32 hash differently by construction). */
extern "C" void int_loop_my_c_buffers(int** my_c_out, const size_t** tri_off_H_out);

static unsigned long long
tri_hash(const void* dev, const size_t bytes, unsigned long long h)
{
  const size_t chunk = (size_t)64 << 20;
  unsigned char* buf = (unsigned char*)malloc(chunk);
  if(!buf) return 0;
  for(size_t off = 0; off < bytes; off += chunk) {
    const size_t n = (bytes - off < chunk) ? (bytes - off) : chunk;
    gpuErrchk( cudaMemcpy(buf, (const char*)dev + off, n, cudaMemcpyDeviceToHost) );
    for(size_t k = 0; k < n; k++) { h ^= buf[k]; h *= 1099511628211ULL; }   /* FNV-1a */
  }
  free(buf);
  return h;
}

static long long
tri_nearinf(const int* dev, const size_t cells)
{
  const size_t chunk = (size_t)16 << 20;                 /* ints */
  int* buf = (int*)malloc(chunk * sizeof(int));
  if(!buf) return -1;
  long long n = 0;
  for(size_t off = 0; off < cells; off += chunk) {
    const size_t m = (cells - off < chunk) ? (cells - off) : chunk;
    gpuErrchk( cudaMemcpy(buf, dev + off, m * sizeof(int), cudaMemcpyDeviceToHost) );
    for(size_t k = 0; k < m; k++) n += (buf[k] > INF/2 && buf[k] < INF);
  }
  free(buf);
  return n;
}

extern "C" void
rnafold_tri_checksum(void)
{
  static int on = -1;
  if(on < 0) { const char* e = getenv("RNA_TRI_CHECKSUM"); on = (e && e[0] == '1'); }
  if(!on || g_tri_cells == 0) return;
  gpuErrchk( cudaDeviceSynchronize() );
  int* myc = NULL; const size_t* t = NULL;
  int_loop_my_c_buffers(&myc, &t);
  const unsigned long long seed = 1469598103934665603ULL;
  const unsigned long long hc = myc ? tri_hash(myc, g_tri_cells * sizeof(int), seed) : 0;
  unsigned long long hf;
  if(d_fml_j16) hf = tri_hash(d_fml_b, g_base_total * sizeof(int),
                              tri_hash(d_fml_j16, g_tri_cells * sizeof(short), seed));
  else          hf = tri_hash(d_fml_j, g_tri_cells * sizeof(int), seed);
  fprintf(stderr, "%-24s triangle checksum: c %016llx  fML %016llx (%s, %zu cells)\n",
          __FILE__, hc, hf, d_fml_j16 ? "int16+baselines" : "int32", g_tri_cells);
  /* NEAR-INF FINITE cells, INF/2 < v < INF: the premise int16's per-block offsets
   * rest on, and the one --noLP and -C were found to break (an INF term plus a
   * finite correction, passed by a one-sided guard). Counted on the int32 encoding
   * only -- under int16 such a cell traps in pack_fml_cell() before it lands. */
  const long long nc = myc ? tri_nearinf((const int*)myc, g_tri_cells) : -1;
  const long long nf = d_fml_j16 ? -1 : tri_nearinf(d_fml_j, g_tri_cells);
  fprintf(stderr, "%-24s triangle near-INF finite cells: c %lld  fML %lld%s\n",
          __FILE__, nc, nf, d_fml_j16 ? " (fML not counted: int16)" : "");
}

/* RNA_ROW_BATCH: flush the g_stage_n staged rows, the newest being i_new, on `s`.
 * size_off_H is row i_new's, the widest of them. Issued OUTSIDE the captured graph:
 * flushing every K-th row inside it would change the graph's topology every K rows
 * and force a reinstantiate each time. Stream order keeps it after the close. */
static void
md_flush_rows(const int nfiles, const int i_new, const int turn,
              const size_t* size_off_H, cudaStream_t s)
{
  const int npend = g_stage_n;
  g_stage_n = 0;
  if(npend == 0) return;
  const size_t total = size_off_H[nfiles];
  if(total == 0) return;
  bind_row_tables(i_new);
  const int TJ = 128;                    /* columns per block = threads per block */
  const int nblocks = (int)((total + TJ - 1)/TJ);
  md_flush_rows_kernel<128><<<nblocks,TJ,0,s>>>(nfiles, turn, i_new, npend,
      g_stage_K, g_row_total, d_fml_stage, d_fml_j, d_fml_j16, d_fml_b,
      d_tri_off_H, d_row_off_H, d_base_off_H, d_colb_off, d_size_off_H, total);
  gpuErrchk( cudaPeekAtLastError() );
}

/* The md chain for one row, on whichever path the caller is capturing or
 * issuing. `tail` from md_tail_refuse(). */
static void
md_row_chain(const int nfiles, const int i, const int turn, const int length,
             const int* energy_min, int* DMLi,
             const size_t* row_off_H, const size_t* size_off_H,
             const size_t* side_off_H, const int* i_H, const int tail)
{
  if(!tail) {
    load_fML(nfiles,i,turn,length,energy_min,size_off_H);
    modular_decomposition_i(nfiles,i,turn,length,DMLi,row_off_H,side_off_H,i_H);
    md_sparse_check(nfiles, i, turn, d_fml_i);            /* RNA_MD_SPARSE_SELFTEST */
    load_min_fML(nfiles,i,turn,length,side_off_H[nfiles]);
    md_sparse_append(nfiles, i, turn, size_off_H);        /* row i is final */
    // int16: closes the row AFTER both writers, which is the whole ordering
    // constraint this design exists to respect. No-op when the gate is off.
    pack_fml(nfiles,i,turn,length,size_off_H);
    return;
  }
  g_md_tail_now = 1;
  modular_decomposition_i(nfiles,i,turn,length,DMLi,row_off_H,side_off_H,i_H);
  g_md_tail_now = 0;
  /* RNA_MD_SPARSE_SELFTEST: md read energy_min shifted by i+turn+1 on this path */
  md_sparse_check(nfiles, i, turn, d_energy_min + (i + turn + 1));
  md_close_row(nfiles,i,turn,size_off_H);
  md_sparse_append(nfiles, i, turn, size_off_H);   /* row i is final */
  g_md_tail_row = i;
}

// Captures load_fML() -> modular_decomposition_i() -> load_min_fML() as a
// single CUDA graph and replays it, instead of issuing 6 separate blocking
// driver calls (fill_arrays_loop.c calls these three back-to-back with no
// host CPU logic between them, which is what makes this legal to capture).
// Recaptured every call rather than captured once, for two reasons: (1) i
// and turn change every call, which changes grid dims (nblocks/side) for the
// underlying kernels -- cudaGraphExecUpdate() handles that as a params
// update, not a topology change; and (2) modular_decomposition_cuda() and
// load_min_fML() both take a host-only early-return branch (side<=0) for the
// first few iterations near the diagonal, which produces a *different*
// node topology (fewer GPU nodes) than later iterations -- recapturing
// means cudaGraphExecUpdate() naturally fails exactly once at that
// transition and falls back to a fresh cudaGraphInstantiate(), rather than
// needing to special-case that boundary by hand.
extern "C" /*PUBLIC*/ void
load_fML_modular_decomposition_load_min_fML(const int nfiles,
					     const int i, const int turn, const int length,
					     const int* energy_min, //in
					     int* DMLi,             //out
					     const size_t* row_off_H,
					     const size_t* size_off_H,
					     const size_t* side_off_H,
					     const int* i_H) {      //in, nfiles entries -- per-record row index
  // Row i's tables were uploaded with the chunk (device.cu), so nothing is
  // copied here and nothing host-side is captured into the graph.
  bind_row_tables(i);

  // RNA_MD_TAIL: decided per row (the lock-step test reads i_H), reported once.
  const char* tail_why = rnafold_md_tail() ? md_tail_refuse(nfiles, i, i_H) : "off";
  const int   tail     = (tail_why == NULL);
  if(rnafold_md_tail()) {
    static int said = 0;
    if(!said) {
      said = 1;
      if(tail) { if(!vrna_cuda_quiet()) fprintf(stderr,"%-24s RNA_MD_TAIL ACTIVE: md reads fml_i from energy_min, "
                              "and ONE kernel closes the row (was load_fML + fmli + "
                              "load_min_fML + pack_fml + fml_prev + a D2D snapshot)\n", __FILE__); }
      else     fprintf(stderr,"%-24s RNA_MD_TAIL REFUSED: %s -- the old md chain runs\n",
                       __FILE__, tail_why);
    }
  }

  // RNA_ROW_BATCH: part of the collapsed tail (it stages md_close_row's triangle
  // writes), so it runs only where the tail does. K is clamped to turn+2: md(r) reads
  // fML rows >= r+turn+2, so a row may wait at most turn+1 rows for its flush.
  {
    int K = (tail && d_fml_stage) ? rnafold_row_batch() : 0;
    if(K > turn + 2) K = turn + 2;
    if(K < 2)        K = 0;
    if(K == 0 && g_stage_n > 0) {
      // the stage was in use and this row cannot continue it: flush what is pending
      // (newest row i+1) before anything reads the triangle
      g_stage_K = (g_stage_K ? g_stage_K : 2);
      md_flush_rows(nfiles, i + 1, turn, rnafold_rowtab_size_host(i + 1),
                    rnafold_stream_overlap() ? rnafold_stream_md() : graph_stream);
    }
    g_stage_K = K;
    static int said = 0;
    if(rnafold_row_batch() && !said) {
      said = 1;
      if(K) { if(!vrna_cuda_quiet()) fprintf(stderr,"%-24s RNA_ROW_BATCH=%d ACTIVE: the fML triangle is written %d rows "
                           "at a time from a row-major stage\n", __FILE__, K, K); }
      else  fprintf(stderr,"%-24s RNA_ROW_BATCH REFUSED: %s\n", __FILE__,
                    tail ? "turn+2 < 2" : "it is part of the collapsed md tail, which is off here");
    }
  }

  // RNA_CUDA_GRAPH=0 disables capture/replay and just issues the same
  // (now-async, graph_stream-targeted) calls directly, with one sync at the
  // end -- lets a whole fold be re-run graph-off vs graph-on and diffed
  // end-to-end (structure/MFE output) to validate the graph path, without
  // needing a second parallel set of device buffers threaded through every
  // kernel just for this A/B check.
  static int use_graph = -1;
  if(use_graph == -1) {
    const char* env = getenv("RNA_CUDA_GRAPH");
    /* An explicit RNA_CUDA_GRAPH wins. Unset, capture is ON at every overlap level since
     * 2026-10-06 (Luke's call on the A100 N2 run): with sparse md -- three launches per row
     * where there was one -- level 2 with graphs measured -0.5 % at 400 x 5601 and -2.2 % at
     * 3000 x 1200 (beyond spread), one output across every arm. It was off at level 2 from
     * 2026-10-03, when the A100 Queue run's rule was written for level 2 without graphs and
     * graphs measured the same there. */
    use_graph = (env && env[0]) ? (env[0] != '0') : 1;
    /* The sparse md selftest synchronises and reads device symbols right
     * after md, which is illegal inside a capture ("operation not permitted when
     * stream is capturing", code 900 -- how this was found). It is a diagnostic, so
     * it turns the graph off rather than the graph turning it off. */
    if(use_graph && md_sparse_selftest()) {
      use_graph = 0;
      fprintf(stderr,"%-24s RNA_MD_SPARSE_SELFTEST forces CUDA graph capture OFF "
                     "(the selftest cannot run inside a capture)\n", __FILE__);
    }
    if (!vrna_cuda_quiet()) fprintf(stderr,"%-24s CUDA graph capture for load_fML/modular_decomposition/load_min_fML: %s\n",
	    __FILE__, use_graph ? "enabled"
	              : (env && env[0]) ? "disabled (RNA_CUDA_GRAPH=0)"
	                                : "disabled (the default at RNA_STREAM_OVERLAP=2; RNA_CUDA_GRAPH=1 turns it on)");
    if(use_graph) atexit(print_graph_update_stats);
  }

  if(!use_graph) {
    g_issue_stream = rnafold_stream_overlap() ? rnafold_stream_md() : 0;
    md_row_chain(nfiles,i,turn,length,energy_min,DMLi,row_off_H,size_off_H,side_off_H,i_H,tail);
    // RNA_ROW_BATCH: count this row into the stage; flush every K rows and at the
    // sweep's last row (i == 1), before anything after the sweep reads the triangle.
    if(g_stage_K && (++g_stage_n == g_stage_K || i == 1))
      md_flush_rows(nfiles, i, turn, size_off_H, ISSUE_STREAM);
    // Level 2 must not sync here -- that would end the overlap before it began,
    // exactly as on the graph path below. RNA_MD_ROW_SYNC=0 drops it below
    // level 2 as well; see rnafold_md_row_sync().
    if(rnafold_stream_overlap() < 2 && rnafold_md_row_sync())
      gpuErrchk( cudaStreamSynchronize(ISSUE_STREAM) );
    g_issue_stream = 0;
    return;
  }

  cudaGraph_t graph = NULL;
  gpuErrchk( cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeThreadLocal) );

  // int16: pack_fml is a captured node too, and must be inside the capture so
  // the replay reproduces the whole row. Ordering is what matters -- pack must
  // follow both writers -- and stream order inside the capture gives that.
  md_row_chain(nfiles,i,turn,length,energy_min,DMLi,row_off_H,size_off_H,side_off_H,i_H,tail);

  gpuErrchk( cudaStreamEndCapture(graph_stream, &graph) );

  const double mgmt_start = graph_now_seconds(); //diagnostic-only

  if(graph_exec_valid) {
    cudaGraphExecUpdateResultInfo update_result;
    const cudaError_t update_error = cudaGraphExecUpdate(graph_exec, graph, &update_result);
    if(update_error != cudaSuccess) {
      //topology changed (e.g. the side<=0 boundary above) -- stale exec can't
      //be patched in place, drop it and instantiate fresh below.
      //This is an *expected*, handled outcome (not a real fault), but every
      //CUDA runtime call -- success or failure -- updates the thread-local
      //"last error" slot that cudaPeekAtLastError()/cudaGetLastError() read.
      //Left uncleared, this failure would sit there and get misattributed to
      //the next unrelated cudaPeekAtLastError() check anywhere downstream
      //(e.g. inside int_loop_i() on the next iteration). cudaGetLastError()
      //(not Peek) both reads AND resets it, so call it to consume/clear this
      //one now that we've handled it.
      cudaGetLastError();
      gpuErrchk( cudaGraphExecDestroy(graph_exec) );
      graph_exec_valid = 0;
      graph_forced_reinstantiate_count++; //diagnostic-only
    } else {
      graph_update_success_count++; //diagnostic-only
    }
  } else {
    graph_first_instantiate_count++; //diagnostic-only
  }
  if(!graph_exec_valid) {
    gpuErrchk( cudaGraphInstantiate(&graph_exec, graph, 0) );
    graph_exec_valid = 1;
  }

  gpuErrchk( cudaGraphDestroy(graph) ); //transient capture object, not graph_exec (the persistent one) -- leaks every iteration if skipped

  graph_mgmt_seconds += graph_now_seconds() - mgmt_start; //diagnostic-only

  {
  cudaStream_t launch_stream = rnafold_stream_overlap() ? rnafold_stream_md() : graph_stream;
  gpuErrchk( cudaGraphLaunch(graph_exec, launch_stream) );
  // RNA_ROW_BATCH: outside the graph (a K-periodic node would change its topology),
  // same stream, so after this row's close. Every K rows and at the last row.
  if(g_stage_K && (++g_stage_n == g_stage_K || i == 1))
    md_flush_rows(nfiles, i, turn, size_off_H, launch_stream);
  //the one sync that remains: also the only point where a real runtime/data
  //error from the replayed graph (bad address, illegal access, device-side
  //assert()) becomes observable, since a graph gives no per-node attribution
  //
  //DELIBERATELY NOT GATED by step 5b, unlike the six per-row syncs in
  //int_loop.cu/hp_mb_loop.cu. Those were made redundant by stream ordering and
  //bought real run-ahead; this one is the graph's only error checkpoint, and
  //dropping it would push a graph fault out to some later row with nothing to
  //attribute it to. It also buys little: a row still contains two blocking
  //pageable H2Ds (upload_size_off_H in each file), and a pageable H2D
  //stream-syncs before it copies, so the host cannot run far ahead regardless.
  //Making those async -- which needs persistent PINNED host tables, per this
  //file's standing capture-region hazard -- is what would unlock true
  //back-to-back queueing, and it is not part of 5b.
  // AT LEVEL 2, NOT SYNCING HERE IS THE WHOLE POINT. The md chain is meant to
  // run beside row i-1's cell work, and a stream sync would end that before it
  // began. The graph's error checkpoint moves to the next synchronisation --
  // the end of the sweep, or the next row's own blocking upload -- which costs
  // attribution, not correctness.
  //
  // AND BELOW LEVEL 2 IT IS NOW OPTIONAL (RNA_MD_ROW_SYNC=0, 2026-09-29). The
  // "two blocking pageable H2Ds" above are gone -- fix 3 uploads every row
  // table with the chunk -- so this sync no longer lands on an already-blocked
  // host; it is what stops the host queueing the next row. Error attribution
  // moves to the next blocking call, exactly as at level 2.
  if(rnafold_stream_overlap() < 2 && rnafold_md_row_sync())
    gpuErrchk( cudaStreamSynchronize(launch_stream) );
  }
}


