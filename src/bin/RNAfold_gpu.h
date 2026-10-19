/*
 *  RNAfold_gpu.h -- the private seam between RNAfold.c and gpu_batch.c
 *  (cleanup C6b, decision D3). Not installed.
 */
#ifndef RNAFOLD_GPU_H
#define RNAFOLD_GPU_H

#include <stdio.h>

#include "ViennaRNA/model.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/io/commands.h"
#include "ViennaRNA/constraints/soft_special.h"
#include "ViennaRNA/datastructures/stream_output.h"

#include "input_id_helpers.h"
#include "probing_data_helpers.h"


struct options {
  int             filename_full;
  char            *filename_delim;
  int             pf;
  int             noPS;
  int             plot_layout;
  int             noDP;
  int             noconv;
  int             lucky;
  int             MEA;
  double          MEAgamma;
  double          bppmThreshold;
  int             verbose;
  char            *ligandMotif;
  vrna_cmd_t      cmds;
  vrna_md_t       md;
  dataset_id      id_control;

  char            *constraint_file;
  int             constraint_batch;
  int             constraint_enforce;
  int             constraint_canonical;

  /* structure probing data releated options */
  probing_data_t  *probing_data;

  vrna_sc_mod_param_t *mod_params;

  unsigned int    benchmark;
  FILE            *benchmark_file;
  double          benchmark_tp;
  double          benchmark_fp;
  double          benchmark_tn;
  double          benchmark_fn;

  int             jobs;
  int             tofile;
  char            *output_file;
  int             keep_order;
  FILE            *output_stream;
  unsigned int    next_record_number;
  vrna_ostream_t  output_queue;
};

struct record_data {
  unsigned int    number;
  char            *id;
  char            *sequence;
  char            *SEQ_ID;
  char            **rest;
  char            *input_filename;
  int             multiline_input;
  struct options  *options;
  int             tty;

#ifdef VRNA_WITH_CUDA
  /* CUDA chunk path: results computed in a batch by par_mfe() before this
   * record is dispatched. When set, process_record() skips its own vrna_mfe()
   * and formats these instead. Everything else about the record -- output
   * formatting, plots, ordering through vrna_ostream_t -- is unchanged, which
   * is what keeps the byte-identical bar meaningful across the two paths. */
  int             prefolded;
  float           prefolded_energy;
  char            *prefolded_structure;

  /* Held back from a GPU chunk to fold on the CPU while the device works on the
   * rest (option C). Only used to signal completion, so the slice size can adapt
   * to how much the pool actually got through. */
  int             cpu_slice;

  /* RNA_RECORD_TRACE: when this record's MATRICES were filled, in seconds since
   * the first traced event. For a GPU record the window is its CHUNK's -- the
   * device folds a batch, so a per-record device window does not exist. For a
   * CPU record process_record() stamps its own fold.
   * Zero means "not stamped", which is how the reader tells the two apart. */
  double          trace_fold_t0, trace_fold_t1;
#endif
};


/* RNAfold.c, for gpu_batch.c */
void    rnafold_dispatch_record(struct record_data *record);   /* RUN_IN_PARALLEL(process_record, .) */
double  trace_now(void);
void    trace_open(void);
void    add_ligand_motif(vrna_fold_compound_t *vc,
                         char                 *motifstring,
                         int                  verbose,
                         unsigned int         options);
void    apply_constraints(vrna_fold_compound_t  *fc,
                          const char            *constraints_file,
                          const char            **rec_rest,
                          int                   maybe_multiline,
                          int                   enforceConstraints,
                          int                   canonicalBPonly,
                          int                   quiet);

#ifdef VRNA_WITH_CUDA
/* gpu_batch.c, for RNAfold.c */
#define GPU_BATCH_NOT_TAKEN 0
#define GPU_BATCH_TAKEN     1
#define GPU_BATCH_TOO_BIG   2

void  gpu_batch_begin(struct options *opt,
                      int            interactive);   /* stdin is a terminal: hold nothing */
int   gpu_batch_offer(struct record_data *record,
                      struct options     *opt);
void  gpu_batch_end(struct options *opt);
int   output_needs_compound(struct options *opt);
void  cpu_slice_finished(void);
#endif

#endif
