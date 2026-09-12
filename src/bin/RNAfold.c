/* VRNA-PATCH-FILE(rnafold-driver, DRIVER) -- PORT_LOCAL_PATCHES.md
 *
 * THIS WHOLE FILE IS A LOCAL PATCH, declared once rather than bracketed hunk
 * by hunk: it is +1333 lines against v2.7.2 and the changes are pervasive,
 * not surgical. Marking each one would be noise pretending to be precision.
 *
 * What it adds: chunked batch folding on the GPU (gpu_path_usable(), the VRAM
 * budget, the chunk loop, the build/fold pipeline, the CPU queue) around
 * upstream own per-record processing. Everything that decides an ANSWER still
 * comes from RNAlib; this file only decides which records go to the device.
 *
 * It is NOT part of any upstream proposal. The library-side patches -- the
 * ones tools/list_local_patches.sh lists -- are what a pull request carries.
 * A driver is the caller business, and upstream RNAfold has no reason to grow
 * a CUDA chunker.
 */
/*
 *                Ineractive Access to folding Routines
 *
 *                c Ivo L Hofacker
 *                Vienna RNA package
 */

/** \file
 *  \brief RNAfold program source code
 *
 *  This code provides an interface for MFE and Partition function folding
 *  of single linear or circular RNA molecules.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <unistd.h>
#include <string.h>

#include "ViennaRNA/mfe/global.h"
#include "ViennaRNA/partfunc/global.h"
#include "ViennaRNA/eval/structures.h"
#include "ViennaRNA/fold_vars.h"
#include "ViennaRNA/plotting/probabilities.h"
#include "ViennaRNA/plotting/structures.h"
#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/utils/strings.h"
#include "ViennaRNA/utils/log.h"
#include "ViennaRNA/io/utils.h"
#include "ViennaRNA/params/io.h"
#include "ViennaRNA/mfe/global.h"
#include "ViennaRNA/structures/centroid.h"
#include "ViennaRNA/structures/mea.h"
#include "ViennaRNA/structures/pairtable.h"
#include "ViennaRNA/structures/benchmark.h"
#include "ViennaRNA/sequences/alphabet.c"
#include "ViennaRNA/params/basic.h"
#include "ViennaRNA/constraints/basic.h"
#include "ViennaRNA/probing/SHAPE.h"
#include "ViennaRNA/constraints/ligand.h"
#include "ViennaRNA/constraints/soft_special.h"
#include "ViennaRNA/structured_domains.h"
#include "ViennaRNA/unstructured_domains.h"
#include "ViennaRNA/io/file_formats.h"
#include "ViennaRNA/io/commands.h"
#include "ViennaRNA/probabilities/basepairs.h"
#include "ViennaRNA/probabilities/structures.h"
#include "ViennaRNA/sampling/basic.h"
#include "ViennaRNA/datastructures/char_stream.h"
#include "ViennaRNA/datastructures/stream_output.h"
#include "ViennaRNA/combinatorics/basic.h"
#include "ViennaRNA/intern/color_output.h"
#include "ViennaRNA/io/sanitize.h"

#ifdef VRNA_WITH_CUDA
#include "ViennaRNA/mfe/cuda/engine.h"

/* Per-chunk teardown. The device buffers are sized for one chunk's (nfiles,
 * length) by init_gpu/2/3 and are NOT reset by par_mfe() itself, so a second
 * call without these reuses dirty state -- observed as
 *   Assertion `my_c[tri_off_H[H]+ij] == INF' failed
 * on the very next chunk. The 2.3.0 driver called them at the end of every
 * chunk for exactly this reason. */
extern void teardown_gpu(void);
extern void teardown_gpu2(void);
extern void teardown_gpu3(void);

/* VRAM budgeting. Both live in the library already:
 *   gpu_bytes_per_file(len)      device bytes one record of that length needs
 *   compute_gpu_usable_bytes()   free VRAM * 0.85, capped by
 *                                RNA_GPU_VRAM_BUDGET_MB (which can only lower
 *                                it, never raise it)
 * and the slot-flow knobs the projection has to respect. */
extern size_t gpu_bytes_per_file(const int length);
/* CIRCULAR: tell the VRAM model that fM2_real will be allocated, before any
 * chunk is sized against it. See modular_decomposition.cu. */
extern void   rnafold_circ_expect(const int circ);
extern size_t compute_gpu_usable_bytes(void);
/* The four stage counters that mfe_cuda.c PRINTS but nothing incremented until
 * 2026-09-08. They cover work that happens in this file rather than in the CUDA
 * layer -- building and freeing fold compounds, device teardown, and output --
 * which is why they were never wired up: the counter lives with the reporter,
 * the work lives here, and nobody closed the gap. The deep profile found 61% of
 * wall at 120x5601 outside every timer; these are the first four candidates.
 * See PROFILE272_DEEP_RESULTS.md. */
extern double stage_build_s, stage_output_s, stage_teardown_s, stage_free_s;
extern double rnafold_now_seconds(void);
extern int    rnafold_slot_flow(void);
extern int    rnafold_slot_capacity_max(void);


/* What a chunk of these lengths would cost on the device, with `cand` added.
 *
 * Ported from the 2.3.0 driver unchanged in substance. The subtlety is slot
 * flow: under RNA_SLOT_FLOW=k, k records SHARE a slot and only the slots are
 * allocated, so the cost is not a sum over records. par_mfe() deals records
 * round-robin in descending length order, so slot s holds the s-th longest
 * record and the footprint is the sum of gpu_bytes_per_file() over the top
 * ceil(n/k) lengths. At k == 1 that degenerates to "sum over every record".
 *
 * RNA_SLOT_CAPACITY=max sizes every slot to the chunk maximum instead of its
 * own occupant's length, so the projection has to follow that too -- the two
 * disagreed when C1 shipped (budget from real lengths, allocation from the
 * capacity rule), which is the kind of mismatch that OOMs only on the inputs
 * you did not test.
 *
 * `sorted_desc` holds the accepted lengths in descending order; cand < 0 means
 * "no candidate, just price what is already here".
 */
static size_t
projected_chunk_bytes(const int *sorted_desc,
                      int        n,
                      int        cand,
                      int        k,
                      int        cap_max)
{
  const int total = n + ((cand >= 0) ? 1 : 0);
  const int slots = (k > 0) ? ((total + k - 1) / k) : total;
  size_t    sum   = 0;
  int       i     = 0;
  int       used  = (cand < 0);
  int       taken;

  if (cap_max) {
    int mx = (n > 0) ? sorted_desc[0] : -1;
    if (cand > mx)
      mx = cand;

    return (mx < 0) ? 0 : (size_t)slots * gpu_bytes_per_file(mx);
  }

  for (taken = 0; taken < slots; taken++) {
    int len;
    if ((!used) && ((i >= n) || (cand >= sorted_desc[i]))) {
      len  = cand;
      used = 1;
    } else if (i < n) {
      len = sorted_desc[i++];
    } else {
      break;
    }
    sum += gpu_bytes_per_file(len);
  }

  return sum;
}
#endif

#include "RNAfold_cmdl.h"
#include "gengetopt_helpers.h"
#include "input_id_helpers.h"
#include "modified_bases_helpers.h"
#include "parallel_helpers.h"
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
#endif
};


struct output_stream {
  vrna_cstr_t data;
  int         individual;
};


static char *
annotate_ligand_motif(vrna_fold_compound_t  *vc,
                      const char            *structure);


static void
print_ligand_motifs(vrna_fold_compound_t  *vc,
                    const char            *structure,
                    const char            *structure_name,
                    vrna_cstr_t           buf);


static void
add_ligand_motif(vrna_fold_compound_t *vc,
                 char                 *motifstring,
                 int                  verbose,
                 unsigned int         options);


static char *
annotate_ud_motif(vrna_fold_compound_t  *vc,
                  vrna_ud_motif_t       *motifs);


static void
print_ud_motifs(vrna_fold_compound_t  *vc,
                vrna_ud_motif_t       *motifs,
                const char            *structure_name,
                vrna_cstr_t           buf);


static void
add_ligand_motifs_dot(vrna_fold_compound_t  *fc,
                      vrna_ep_t             **prob_list,
                      vrna_ep_t             **mfe_list,
                      const char            *structure);


static void
add_ligand_motifs_to_list(vrna_ep_t       **list,
                          vrna_sc_motif_t *motifs);


static void
compute_MEA(vrna_fold_compound_t  *fc,
            double                MEAgamma,
            const char            *ligandMotif,
            int                   verbose,
            vrna_cstr_t           buf);


static void
compute_centroid(vrna_fold_compound_t *fc,
                 const char           *ligandMotif,
                 int                  verbose,
                 vrna_cstr_t          buf);


/* `quiet` suppresses only the WARNINGS, never the application of a constraint
 * or the hard error on an over-long one. The GPU chunk path builds its own fold
 * compounds and so must apply the same constraints process_record() will, which
 * would otherwise report every malformed record twice. Muting the logger around
 * the call is not an option: process_record() runs on the thread pool and could
 * be logging concurrently. */
static void
apply_constraints(vrna_fold_compound_t  *fc,
                  const char            *constraints_file,
                  const char            **rec_rest,
                  int                   maybe_multiline,
                  int                   enforceConstraints,
                  int                   canonicalBPonly,
                  int                   quiet);


static char *
generate_filename(const char  *pattern,
                  const char  *def_name,
                  const char  *id,
                  const char  *filename_delim);


int
process_input(FILE            *input_stream,
              const char      *input_filename,
              struct options  *opt);


static void
process_record(struct record_data *record);


/*--------------------------------------------------------------------------*/
void
flush_cstr_callback(void          *auxdata,
                    unsigned int  i,
                    void          *data)
{
  struct output_stream *s = (struct output_stream *)data;

  if (s) {
    /* flush/free/close data[k] */
    if (s->individual)
      vrna_cstr_close(s->data);
    else
      vrna_cstr_free(s->data);

    free(s);

  }
}


static void
postscript_layout(vrna_fold_compound_t  *fc,
                  const char            *orig_sequence,
                  const char            *structure,
                  const char            *SEQ_ID,
                  struct options        *opt)
{
  char      *filename_plot  = NULL;
  char      *annotation     = NULL;
  vrna_md_t *md             = &(fc->params->model_details);

  char  *ligandMotif = opt->ligandMotif;
  char  *filename_delim = opt->filename_delim;
  int   verbose = opt->verbose;

  filename_plot = generate_filename("%s%sss.ps",
                                    "rna.ps",
                                    SEQ_ID,
                                    filename_delim);

  if (ligandMotif) {
    char *annote = annotate_ligand_motif(fc, structure);
    vrna_strcat_printf(&annotation, annote);
    free(annote);
  }

  if (fc->domains_up) {
    vrna_ud_motif_t *m  = vrna_ud_motifs_MFE(fc, structure);
    char            *a  = annotate_ud_motif(fc, m);
    vrna_strcat_printf(&annotation, a);
    free(a);
    free(m);
  }

  vrna_plot_data_t  aux_data;
  aux_data.pre = annotation;
  aux_data.post = NULL;
  aux_data.md = md;
  vrna_plot_layout_t  *layout;

  layout = vrna_plot_layout(structure, opt->plot_layout);

  THREADSAFE_FILE_OUTPUT(
    vrna_plot_structure(filename_plot,
                        orig_sequence,
                        structure,
                        VRNA_FILE_FORMAT_PLOT_DEFAULT,
                        layout,
                        &aux_data));

  vrna_plot_layout_free(layout);
  free(annotation);
  free(filename_plot);
}


static void
ImFeelingLucky(vrna_fold_compound_t *fc,
               const char           *orig_sequence,
               const char           *SEQ_ID,
               int                  noPS,
               const char           *filename_delim,
               vrna_cstr_t          buf,
               int                  istty_in)
{
  vrna_md_t *md = &(fc->params->model_details);

  vrna_init_rand();

  char      *filename_plot  = NULL;
  char      *s              = vrna_pbacktrack(fc);
  float     e               = vrna_eval_structure(fc, (const char *)s);

  vrna_cstr_printf_structure(buf,
                             s,
                             (istty_in) ? "\n free energy = %6.2f kcal/mol" : " (%6.2f)",
                             e);

  if (!noPS) {
    filename_plot = generate_filename("%s%sss.ps",
                                      "rna.ps",
                                      SEQ_ID,
                                      filename_delim);

    vrna_plot_data_t  aux_data;
    aux_data.pre = NULL;
    aux_data.post = NULL;
    aux_data.md = md;
    vrna_plot_layout_t  *layout;

    layout = vrna_plot_layout(s, rna_plot_type);

    THREADSAFE_FILE_OUTPUT(
      vrna_plot_structure(filename_plot,
                          orig_sequence,
                          s,
                          VRNA_FILE_FORMAT_PLOT_DEFAULT,
                          layout,
                          &aux_data));

    vrna_plot_layout_free(layout);
  }

  free(s);
}


static char *
generate_filename(const char  *pattern,
                  const char  *def_name,
                  const char  *id,
                  const char  *filename_delim)
{
  char *filename, *ptr;

  if (id) {
    filename  = vrna_strdup_printf(pattern, id, filename_delim);
    ptr       = vrna_filename_sanitize(filename, filename_delim);
    free(filename);
    filename = ptr;
  } else {
    filename = strdup(def_name);
  }

  return filename;
}


static char **
collect_unnamed_options(struct RNAfold_args_info  *ggostruct,
                        int                       *num_files)
{
  char  **input_files = NULL;
  int   i;

  *num_files = 0;

  /* collect all unnamed options */
  if ((ggostruct->inputs_num > 0) && (!sanitize_input(ggostruct->inputs[0]))) {
    input_files = (char **)vrna_realloc(input_files, sizeof(char *) * ggostruct->inputs_num);
    for (i = 0; i < ggostruct->inputs_num; i++)
      input_files[(*num_files)++] = strdup(ggostruct->inputs[i]);
  }

  return input_files;
}


static char **
append_input_files(struct RNAfold_args_info *ggostruct,
                   char                     **files,
                   int                      *numfiles)
{
  int i;

  if (ggostruct->infile_given) {
    files = (char **)vrna_realloc(files, sizeof(char *) * (*numfiles + ggostruct->infile_given));
    for (i = 0; i < ggostruct->infile_given; i++)
      files[(*numfiles)++] = strdup(ggostruct->infile_arg[i]);
  }

  return files;
}


void
init_default_options(struct options *opt)
{
  opt->filename_full  = 0;
  opt->filename_delim = NULL;
  opt->pf             = 0;
  opt->noPS           = 0;
  opt->plot_layout    = rna_plot_type;
  opt->noDP           = 0;
  opt->noconv         = 0;
  opt->lucky          = 0;
  opt->MEA            = 0;
  opt->MEAgamma       = 1.;
  opt->bppmThreshold  = 1e-5;
  opt->verbose        = 0;
  opt->ligandMotif    = NULL;
  opt->cmds           = NULL;
  set_model_details(&(opt->md));

  opt->constraint_file      = NULL;
  opt->constraint_batch     = 0;
  opt->constraint_enforce   = 0;
  opt->constraint_canonical = 0;

  opt->probing_data     = NULL;

  opt->mod_params         = NULL;

  opt->benchmark          = 0;
  opt->benchmark_file     = stdout;
  opt->benchmark_tp       = 0.;
  opt->benchmark_fp       = 0.;
  opt->benchmark_tn       = 0.;
  opt->benchmark_fn       = 0.;

  opt->jobs               = 1;
  opt->tofile             = 0;
  opt->output_file        = NULL;
  opt->keep_order         = 1;
  opt->output_stream      = NULL;
  opt->next_record_number = 0;
  opt->output_queue       = NULL;
}


int
main(int  argc,
     char *argv[])
{
  struct  RNAfold_args_info args_info;
  char                      **input_files;
  int                       num_input;
  struct  options           opt;

  num_input = 0;

  init_default_options(&opt);

  /*
   #############################################
   # check the command line parameters
   #############################################
   */
  if (RNAfold_cmdline_parser(argc, argv, &args_info) != 0)
    exit(1);

  /* prepare logging system and verbose mode */
  ggo_log_settings(args_info, opt.verbose);

  /* get basic set of model details */
  ggo_get_md_eval(args_info, opt.md);
  ggo_get_md_fold(args_info, opt.md);
  ggo_get_md_part(args_info, opt.md);
  ggo_get_circ(args_info, opt.md.circ);

  /* temperature */
  ggo_get_temperature(args_info, opt.md.temperature);

  /* check dangle model */
  if ((opt.md.dangles < 0) || (opt.md.dangles > 3)) {
    vrna_log_warning("Requested dangle model not implemented, falling back to default dangles=2");
    opt.md.dangles = dangles = 2;
  }

  ggo_get_id_control(args_info, opt.id_control, "Sequence", "sequence", "_", 4, 1);

  ggo_get_constraints_settings(args_info,
                               fold_constrained,
                               opt.constraint_file,
                               opt.constraint_enforce,
                               opt.constraint_batch);

  /* enforce canonical base pairs in any case? */
  if (args_info.canonicalBPonly_given)
    opt.constraint_canonical = 1;

  /* do not convert DNA nucleotide "T" to appropriate RNA "U" */
  if (args_info.noconv_given)
    opt.noconv = 1;

  /* always look on the bright side of life */
  if (args_info.ImFeelingLucky_given)
    opt.md.uniq_ML = opt.lucky = opt.pf = st_back = 1;

  /* set the bppm threshold for the dotplot */
  if (args_info.bppmThreshold_given)
    opt.bppmThreshold = MIN2(1., MAX2(0., args_info.bppmThreshold_arg));

  /* do not produce postscript output */
  if (args_info.noPS_given)
    opt.noPS = 1;

  /* do not produce dot-plot output */
  if (args_info.noDP_given)
    opt.noDP = 1;

  /* partition function settings */
  if (args_info.partfunc_given) {
    opt.pf = 1;
    if (args_info.partfunc_arg != 1)
      opt.md.compute_bpp = do_backtrack = args_info.partfunc_arg;
    else
      opt.md.compute_bpp = do_backtrack = 1;
  }

  /* MEA (maximum expected accuracy) settings */
  if (args_info.MEA_given) {
    opt.pf = opt.MEA = 1;
    if (args_info.MEA_arg != -1)
      opt.MEAgamma = args_info.MEA_arg;
  }

  if (args_info.layout_type_given)
    opt.plot_layout = rna_plot_type = args_info.layout_type_arg;

  if (args_info.outfile_given) {
    opt.tofile = 1;
    if (args_info.outfile_arg)
      opt.output_file = strdup(args_info.outfile_arg);
  }

  if (args_info.motif_given)
    opt.ligandMotif = strdup(args_info.motif_arg);

  if (args_info.commands_given)
    opt.cmds = vrna_file_commands_read(args_info.commands_arg, VRNA_CMD_PARSE_DEFAULTS);

  ggo_get_modified_base_settings(args_info, opt.mod_params, &(opt.md));

  ggo_geometry_settings(args_info, &(opt.md));

  /* collect probing data */
  ggo_get_probing_data(argc, argv, args_info, opt.probing_data);

  /* filename sanitize delimiter */
  if (args_info.filename_delim_given)
    opt.filename_delim = strdup(args_info.filename_delim_arg);
  else if (get_id_delim(opt.id_control))
    opt.filename_delim = strdup(get_id_delim(opt.id_control));

  if ((opt.filename_delim) && isspace(*opt.filename_delim)) {
    free(opt.filename_delim);
    opt.filename_delim = NULL;
  }

  /* full filename from FASTA header support */
  if (args_info.filename_full_given)
    opt.filename_full = 1;

  if (args_info.jobs_given) {
#if VRNA_WITH_PTHREADS
    int thread_max = max_user_threads();
    if (args_info.jobs_arg == 0) {
      /* use maximum of concurrent threads */
      int proc_cores, proc_cores_conf;
      if (num_proc_cores(&proc_cores, &proc_cores_conf)) {
        opt.jobs = MIN2(thread_max, proc_cores_conf);
      } else {
        vrna_log_warning("Could not determine number of available processor cores!\n"
                         "Defaulting to serial computation");
        opt.jobs = 1;
      }
    } else {
      opt.jobs = MIN2(thread_max, args_info.jobs_arg);
    }

    opt.jobs = MAX2(1, opt.jobs);
#else
    vrna_log_warning(
      "This version of RNAfold has been built without parallel input processing capabilities");
#endif

    if (args_info.unordered_given)
      opt.keep_order = 0;
  }

  if (args_info.benchmark_given) {
    opt.benchmark = 1;

    if (args_info.bm_output_given) {
      char *bfname = vrna_filename_sanitize(args_info.bm_output_arg, opt.filename_delim);

      char *mode = (args_info.bm_output_append_given) ? "a" : "w";

      if (!(opt.benchmark_file = fopen(bfname, mode))) {
        vrna_log_error("Failed to open benchmark file \"%s\" for %s",
                       bfname,
                       (args_info.bm_output_append_given) ? "appending" : "writing");
        exit(EXIT_FAILURE);
      }

      free(bfname);
    }

    if (args_info.bm_rm_pk_given)
      opt.benchmark |= 1 << 2;

    if (args_info.bm_rm_nc_given)
      opt.benchmark |= 1 << 3;

    fprintf(opt.benchmark_file,
            "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
            "ID",
            "TP",
            "FP",
            "TN",
            "FN",
            "TPR",
            "PPV",
            "MCC",
            "F1");

    fflush(opt.benchmark_file);

  }

  input_files = collect_unnamed_options(&args_info, &num_input);
  input_files = append_input_files(&args_info, input_files, &num_input);

  /* free allocated memory of command line data structure */
  RNAfold_cmdline_parser_free(&args_info);


  /*
   #############################################
   # begin initializing
   #############################################
   */
#if 0
  if (opt.md.circ && opt.md.gquad) {
    vrna_log_error("G-Quadruplex support is currently not available for circular RNA structures");
    exit(EXIT_FAILURE);
  }
#endif

  if (opt.md.circ && opt.md.noLP)
    vrna_log_warning("Depending on the origin of the circular sequence, some structures may be missed when using --noLP\n"
             "Try rotating your sequence a few times");

  if ((opt.verbose) && (opt.jobs > 1))
    vrna_log_info("Preparing %d parallel computation slots", opt.jobs);

  if (opt.keep_order)
    opt.output_queue = vrna_ostream_init(&flush_cstr_callback, NULL);

  /*
   ################################################
   # process input files or handle input from stdin
   ################################################
   */
  INIT_PARALLELIZATION(opt.jobs);

  if (num_input > 0) {
    int i, skip;
    for (skip = i = 0; i < num_input; i++) {
      if (!skip) {
        FILE *input_stream = fopen((const char *)input_files[i], "r");

        if (!input_stream) {
          vrna_log_error("Unable to open %d. input file \"%s\" for reading",
                         i + 1,
                         input_files[i]);
          exit(EXIT_FAILURE);
        }

        if (opt.verbose) {
          vrna_log_info("Processing %d. input file \"%s\"",
                        i + 1,
                        input_files[i]);
        }

        if (process_input(input_stream, (const char *)input_files[i], &opt) == 0)
          skip = 1;

        fclose(input_stream);
      }

      free(input_files[i]);
    }
  } else {
    (void)process_input(stdin, NULL, &opt);
  }

  UNINIT_PARALLELIZATION

  /*
   ################################################
   # post processing
   ################################################
   */

  if (opt.benchmark) {
    vrna_score_t scores = vrna_score_from_confusion_matrix(opt.benchmark_tp,
                                                           opt.benchmark_tn,
                                                           opt.benchmark_fp,
                                                           opt.benchmark_fn);

    THREADSAFE_FILE_OUTPUT(({
      fprintf(opt.benchmark_file,
              "%s\t%.1f\t%.1f\t%.1f\t%.1f\t%f\t%f\t%f\t%f\n",
              "TOTAL",
              scores.TP,
              scores.FP,
              scores.TN,
              scores.FN,
              scores.TPR,
              scores.PPV,
              scores.MCC,
              scores.F1);
    }))

    fflush(opt.benchmark_file);

    if (opt.benchmark_file != stdout)
      fclose(opt.benchmark_file);
  }

  /* close output stream if necessary */
  if ((opt.output_stream) && (opt.output_stream != stdout))
    fclose(opt.output_stream);

  vrna_ostream_free(opt.output_queue);

  free(input_files);
  free(opt.constraint_file);
  free(opt.ligandMotif);
  free(opt.filename_delim);
  vrna_commands_free(opt.cmds);

  if (opt.mod_params) {
    for (vrna_sc_mod_param_t *ptr = opt.mod_params; *ptr != NULL; ptr++)
      vrna_sc_mod_parameters_free(*ptr);

    free(opt.mod_params);
  }

  free_id_data(opt.id_control);

  probing_data_free(opt.probing_data);

  if (vrna_log_fp() != stderr)
    fclose(vrna_log_fp());

  return EXIT_SUCCESS;
}


struct output_stream *
get_output_stream(unsigned int    init_size,
                  struct options  *opt,
                  const char      *SEQ_ID,
                  const char      *input_filename)
{
  struct output_stream  *o_stream;
  FILE                  *output;
  int                   individual_stream;

  individual_stream = 0; /* we default to using a single output sink */

  o_stream = (struct output_stream *)vrna_alloc(sizeof(struct output_stream));

  /* in case we do parallel processing of input, let's block access to the opt->output_stream pointer */
  ATOMIC_BLOCK(({
    /* default to stream that we've already opened */
    output = opt->output_stream;

    if ((!opt->tofile) && (!output)) {
      output = stdout;
      opt->output_stream = stdout;
    } else if (opt->tofile) {
      char *filename, *tmp;

      tmp = filename = NULL;

      if ((!opt->output_file) && (SEQ_ID)) {
        /* need to open new individual output file */
        tmp = vrna_strdup_printf("%s.fold", SEQ_ID);
        individual_stream = 1;

        filename = vrna_filename_sanitize(tmp, opt->filename_delim);

        if ((input_filename) && !strcmp(input_filename, filename)) {
          vrna_log_error("Input and output file names are identical");
          exit(EXIT_FAILURE);
        }

        if (!(output = fopen(filename, "a"))) {
          vrna_log_error("Failed to open file for writing");
          exit(EXIT_FAILURE);
        }
      } else if (!output) {
        /* we need to open global output file */
        tmp = (opt->output_file) ?
              vrna_strdup_printf("%s", opt->output_file) :
              vrna_strdup_printf("RNAfold_output.fold");

        filename = vrna_filename_sanitize(tmp, opt->filename_delim);

        if ((input_filename) && !strcmp(input_filename, filename)) {
          vrna_log_error("Input and output file names are identical");
          exit(EXIT_FAILURE);
        }

        if (!(output = fopen(filename, "a"))) {
          vrna_log_error("Failed to open file for writing");
          exit(EXIT_FAILURE);
        }

        opt->output_stream = output;
      }

      free(tmp);
      free(filename);
    }

    /* actually initialize vrna_cstr_t of the stream */
    o_stream->data = vrna_cstr(init_size, output);
    o_stream->individual = (individual_stream) ? 1 : 0;
  }));

  return o_stream;
}


#ifdef VRNA_WITH_CUDA
/* Fold one accumulated chunk on the GPU, then dispatch its records normally.
 *
 * The records are dispatched AFTER folding, each carrying its result, so all
 * output formatting stays in process_record() and is shared with the
 * per-record path. Nothing about ordering changes: the ostream slots were
 * requested at read time.
 */
/* May this RUN use the GPU chunk path at all?
 *
 * This is what makes the CUDA build a strict accelerator rather than a variant
 * of RNAfold with a different feature set. Anything the device path does not
 * reproduce exactly is not refused and is not approximated -- it simply goes
 * down upstream's own per-record path, which is the validated CPU code. The
 * user sees ViennaRNA 2.7.2's answers for every option; the GPU only ever
 * accelerates the cases where it agrees.
 *
 * Two layers, deliberately:
 *   here            a RUN-level decision from the model details and the
 *                   options, made once, before any record is committed;
 *   par_fill_arrays a per-batch backstop that ERRORS if an unsupported model
 *                   reaches the sweep anyway. It should now be unreachable,
 *                   and is kept precisely so that "unreachable" is enforced
 *                   rather than assumed.
 *
 * Note what is NOT excluded: the partition function, MEA, centroid, plots and
 * probability output all keep working, because process_record() still does
 * every one of them itself on its own fold compound. The chunk path supplies
 * only the MFE energy and structure. That falls out of not duplicating
 * process_record(), and it is why -p needs no special handling here.
 */
static int
gpu_path_usable(struct options *opt,
                const char    **why)
{
  vrna_md_t *md = &(opt->md);

#define NO(msg) do { if (why) *why = (msg); return 0; } while (0)

  /* Model details the sweep does not implement. Mirrors
   * vrna_cuda_engine_supports() in mfe/cuda/engine.c; kept in step with it. */
  /* DANGLE MODEL 0 ACCEPTED 2026-09-11, alongside the long-standing 2.
   *
   * d0 and d2 share the RECURSION -- mfe_multibranch.c:686 dispatches
   * ml_pair_d0 or ml_pair_d2 and BOTH read only dmli1 -- so no new DP state was
   * needed.
   *
   * AND IT NEEDED EXACTLY ONE DEVICE CHANGE, which is not where I first looked.
   * Upstream ZEROES P->mismatchM when dangles == 0 (params.c:644-646), so
   * E_MLstem() already returns the bare-stem energy under d0 and both
   * multibranch sites were ALREADY correct. Measured: 0 nonzero mismatchM
   * entries at d0 against 175 at d2, and red-teaming those two sites changes
   * nothing. The one term that genuinely differed is
   * vrna_mfe_gquad_internal_loop()'s mismatchI (mfe_gquad.c:306, `if
   * (dangles)`) -- mismatchI is NOT zeroed (158 nonzero at both models), so
   * -d0 -g was the only combination actually returning a wrong answer.
   * Red-team: 20 differing lines with that gate removed.
   *
   * d1 AND d3 STAY DECLINED, and not for want of effort: ml_pair_d1 needs
   * dmli2 as WELL as dmli1 -- a second DMLi generation the sweep does not
   * carry -- plus the `dangle_model % 2` terms at :1015 and :1376, and d3 adds
   * coaxial stacking at :1501. That is new DP state, not new arithmetic.
   *
   * The exterior loop needed nothing: vrna_mfe_exterior_f5() is upstream's own
   * and runs on the host, as does vrna_mfe_multibranch_m1() for fM1 under
   * uniq_ML, so both are dangle-correct for every model by construction.
   */
  if ((md->dangles != 0) && (md->dangles != 2))
    NO("dangle model 1 or 3 (0 and 2 are accelerated)");
  /* G-QUADRUPLEXES ACCEPTED 2026-09-10 (G3).

   * The sweep now scores quadruplexes into c and fML: c_gq is carried to the
   * device (G0), extend_fm_3p()'s multibranch term is in fml_scan_kernel (G1),
   * and vrna_mfe_gquad_internal_loop()'s three bounded (p,q) sweeps are in
   * gq_internal_kernel (G2).
   *
   * The last blocker was NOT the recursion. With the energy already byte-exact,
   * the STRUCTURE still came out with a single '+' where a quadruplex belongs,
   * because vrna_backtrack_from_intervals() discards a gquad's layout when it
   * downconverts to the legacy vrna_bp_stack_t. Fixed by
   * VRNA-PATCH(bps-backtrack): the port now backtracks into a vrna_bps_t and
   * renders with vrna_db_from_bps().
   *
   * BAR: byte-identical to pristine 2.7.2 on the frozen tests/gquad/ set (6
   * records, 80-1200 nt, every one containing a quadruplex), and GPU==CPU on 20
   * further G-rich records at 70-1100 nt, alone and combined with --noLP,
   * --noGU, -4, --salt and -T. Regression test: tests/mfe_cuda_gquad.ts.
   */
  /* CIRCULAR RNA ACCEPTED 2026-09-11.

   * The sweep now persists fM2_real. It was never new arithmetic:
   * modular_decomposition_kernel already reduces min_k(fML[i,k]+fML[k+1,j])
   * into DMLi every row -- exactly what upstream's mfe_multibranch_m2_fast()
   * computes -- and the fork discarded it one row later. It now also stores it
   * into a persistent triangle, and backtrack_one_slot() calls
   * VRNA-PATCH(circular-postprocess) on it.
   *
   * COSTS A CHUNK WIDTH. fM2_real is a second full int32 triangle per record,
   * counted in modular_decomposition_bytes_per_file() under
   * rnafold_circ_expect(), so a circular batch admits proportionally fewer
   * records rather than OOMing.
   *
   * BAR: byte-identical to pristine 2.7.2 on the frozen tests/circ/ set (6
   * records, 60-600 nt, every circular energy differing from its linear one),
   * and GPU==CPU on 20 further records at 45-570 nt -- plain, chunked, int16,
   * --noLP, --noGU, --salt, -T and -4. tests/mfe_cuda_circ.ts is the
   * regression bar; RNA_CIRC_VERIFY=1 re-checks fM2_real against its own
   * definition, O(n^3), for small inputs.
   */
  /* --noClosingGU ACCEPTED 2026-09-11.

   * It was the last multi-gate option, and it was half implemented: the
   * hairpin/multibranch half already worked (rnafold_hc_opt() drops HP_LOOP and
   * MB_LOOP for a GU/UG pair exactly as hard.c:786-791 does, and new_c_kernel
   * skips both terms on gate bit 1), while the interior-loop half did nothing
   * at all -- so c was internally inconsistent rather than merely suboptimal.
   *
   * The missing half is two rules, both from mfe/mfe_internal.c: a GU/UG pair
   * may neither CLOSE nor BE ENCLOSED BY a bulge or interior loop. STACKS ARE
   * EXEMPT -- mfe_stacks() carries no such test, and vrna_E_internal() returns
   * the stack energy before consulting no_close. Energy() now skips those
   * candidates, and gq_internal_kernel returns early for a GU closing pair
   * because vrna_mfe_gquad_internal_loop() is called from inside upstream's
   * own `if (!noclose)` block.
   *
   * BAR: byte-identical to pristine 2.7.2 with --noClosingGU across the frozen
   * fixtures; RNA_HC_VERIFY=1 proves the mask half against the host's own
   * hc->mx under this flag. tests/mfe_cuda_noclosinggu.ts is the regression
   * bar.
   */

  /* NOT rejected, and each for a reason that was measured rather than assumed:
   *
   *   uniq_ML  the MFE recursion never reads fM1, so the ANSWER is right.
   *            Note the sweep leaves fM1 entirely INF (fill_arrays.c:271);
   *            RNAfold never reads it, but the LIBRARY guard in
   *            mfe/cuda/engine.c still declines uniq_ML for that reason, since
   *            a vrna_mfe_batch() caller might go on to call vrna_subopt().
   *   (logML is NOT here: it stays declined. See mfe/cuda/engine.c -- 12/12 at
   *    160 nt is a sample, and RNAfold has no --logML flag to bar it with.)
   *   noGU     reaches the sweep through the uploaded hc->mx bitmasks, which
   *            upstream has already populated.
   *
   * All three verified byte-identical against the CPU route over the full
   * reference set before the rejection was lifted; tools/verify_option_parity.sh
   * covers them.
   *
   * noLP is now ACCEPTED (2026-09-07). The earlier note here ("disagreed on 3
   * of 12 records") understated the defect badly, because it compared ENERGIES.
   * Diagnosed
   * properly 2026-09-07 by lifting both guards and re-evaluating each returned
   * structure with RNAeval -- the check that needs no oracle:
   *
   *   CPU  rec 0: reports -295.60, structure re-evaluates to -295.60, 274 pairs
   *   GPU  rec 0: reports -295.60, structure re-evaluates to  -77.18,  90 pairs
   *   GPU  rec 1: reports -294.94, structure re-evaluates to   +5.54,   6 pairs
   *
   * 8 of 8 sampled records inconsistent by 87 to 300 kcal. The matrix fill and
   * the backtrack do not agree WITH EACH OTHER -- the same failure shape as the
   * hard-constraint bug, and invisible to an energy comparison because the f5
   * value is roughly right while the structure is not.
   *
   * Cause, and it is not "ptype is not enough". noLP changes what the
   * recursion WRITES: at mfe/mfe.c:4413 upstream stores cc1[j-1]+stackEnergy
   * into c[ij] rather than new_c, carrying the unconstrained value sideways in
   * the cc/cc1 row buffers. That is what forbids a helix of length one. The
   * sweep stores new_c, so upstream's backtrack -- which branches on noLP at
   * mfe/mfe.c:4289 and calls vrna_bt_stacked_pairs() -- walks a matrix built
   * under a different convention than the one it assumes.
   *
   * fill_arrays_loop.c:215 still carries the deleted machinery as a comment,
   * marked "gcov says not used". It said that because noLP was never exercised.
   *
   * FIXED and the guard lifted: a cc/cc1 row-buffer pair, a stack_row_kernel
   * that computes upstream's vrna_eval_stack() per row, and new_c_kernel
   * writing the stacked value into c while carrying the unconstrained new_c in
   * cc. The BACKTRACK needed nothing -- vrna_backtrack_from_intervals() wraps
   * upstream's own backtrack(), which already handles noLP; it was mis-walking
   * our matrix precisely because the matrix was wrong.
   *
   * Verified: 60/60 records byte-identical to upstream --noLP (was 1/60), every
   * structure re-evaluates to its own reported energy, zero lonely pairs, seven
   * reference workloads including F_extreme and a 1600-record batch, and
   * identical across 2/3/4/7 chunks. --noLP + RNA_FML_INT16 is REFUSED at init
   * (fill_arrays.c): noLP puts near-INF FINITE values into fML that the 16-bit
   * per-block offsets cannot represent. */
  if (md->energy_set != 0)      NO("non-default energy set");
  /* salt is no longer barred: the multibranch kernel inherits it through the
   * parameter tables, and the hairpin/internal kernels each add one term from
   * a host-built table. tools/verify_salt_parity.sh is the bar. */

  /* Per-record data that reaches the recursion as constraints.
   *
   * Hard constraints were MEASURED here, not assumed, and they do not work: on
   * a 12x80 nt batch with a forced-unpaired block the sweep returned -14.30 for
   * a structure worth -5.40. See the long note in mfe/cuda/engine.c, which
   * declines them at the library level -- that is the check that actually keeps
   * a vrna_mfe_batch() caller safe. This one is a routing decision on top:
   * barring the run here keeps -C on upstream's PARALLEL per-record path,
   * whereas letting the chunk build and then be declined would fold it serially
   * inside vrna_mfe_batch().
   *
   * flush_gpu_chunk() still applies constraints to the compounds it builds. It
   * is not dead code: it is what makes the declined-batch fallback fold the
   * right thing, and it has to be in place before this bar can lift. */
  if (fold_constrained)         NO("structure constraints");
  if (opt->constraint_file)     NO("a constraint file");
  if (opt->probing_data)        NO("probing/SHAPE data");
  if (opt->cmds)                NO("a command file");
  if (opt->mod_params)          NO("modified bases");
  if (opt->ligandMotif)         NO("a ligand motif");

#undef NO

  if (why)
    *why = NULL;

  return 1;
}


/* Below this, a chunk is not worth a GPU batch: the per-chunk init_gpu/teardown
 * cycle alone measured ~1.6 s, which dwarfs folding a handful of records.
 *
 * 10 is the 2.3.0 value and is carried over deliberately unchanged, but it is
 * KNOWN to be length-dependent rather than a constant -- break-even is ~65
 * records at 300 nt, ~10 at 600 nt, and 1 at >=1200 nt. So 10 is right only
 * near 600-700 nt and too small below that. The proper fix is a length guard,
 * not a better constant; carried over as-is so the port stays behaviour-
 * preserving and the guard lands as its own measured change. */
#define VRNA_MIN_GPU_BATCH 10


/* The CPU-fallback threshold, overridable so it can be MEASURED.
 *
 * Measured 2026-09-07 on 60 x 900 nt: this constant is what made int16 look
 * like a REGRESSION in benchmark v3's arm F. Chunk capacity comes from the
 * VRAM budget, so the final chunk holds a quantisation remainder -- and
 * int16's larger chunks leave a LARGER remainder. At a 48 MB budget int32
 * packed 14/chunk and left 4 records behind (2.6 s on the CPU) while int16
 * packed 18/chunk and left 6 (4.0 s). Both did ~2.7 s of GPU work, so the
 * whole apparent 0.86x regression was the CPU tail -- and nothing could see
 * it, because the benchmark's validity gate tested `sweeps == 0`, which
 * catches only a TOTAL fallback and never a partial one.
 *
 * Per-chunk GPU cost measured at 900 nt is 0.36 s + 0.025 s/record against
 * 0.65 s/record on the CPU, so break-even there is under ONE record: at that
 * length the fallback can never win. That is consistent with the length
 * dependence noted above -- the CPU fold and the GPU marginal cost are both
 * O(L^3) while the per-chunk fixed cost is not, so the break-even count goes
 * as L^-3. That is also why those three documented points line up: 65 at
 * 300 nt gives 8.1 at 600 and 1.0 at 1200.
 *
 * An env override rather than a new constant, deliberately: the default stays
 * behaviour-preserving, and ONE binary can be folded both ways and compared
 * against itself -- the shape that settled RNA_GPU_SWEEP and RNA_FML_INT16.
 * It announces itself for the reason the int16 gate does: a run that did not
 * apply the setting must not be able to pass for one that did.
 *
 * Deliberately NOT used at the other VRNA_MIN_GPU_BATCH site below, which
 * asks a different question -- "can free VRAM hold a worthwhile batch at this
 * length?". Setting that to 1 would admit a GPU path where a single record
 * fills VRAM, i.e. one chunk per record, which is the worst case for chunk
 * count and so the worst case for wall clock.
 */
static int
rnafold_min_gpu_batch(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_MIN_GPU_BATCH");

    v = VRNA_MIN_GPU_BATCH;

    if (e && *e) {
      const long n = atol(e);

      if (n >= 1) {
        v = (int)n;
        fprintf(stderr,
                "%-24s RNA_MIN_GPU_BATCH=%d (default %d): chunks smaller than "
                "this fold on the CPU\n",
                "bin/RNAfold.c", v, VRNA_MIN_GPU_BATCH);
      } else {
        fprintf(stderr,
                "%-24s ignoring RNA_MIN_GPU_BATCH=%s (want a positive integer)\n",
                "bin/RNAfold.c", e);
      }
    }
  }

  return v;
}


/* Will process_record() have to construct a fold compound for an
 * already-folded record? ONE predicate with two users, because they must not
 * drift:
 *
 *   1. process_record() itself, to skip the construction (that skip is the whole
 *      of stage_output -- 124.8 s of a 669 s run before it landed);
 *   2. pipeline_flush(), which is only race-free while the answer is NO.
 *
 * (2) is the subtle one. The builder thread calls vrna_fold_compound() ->
 * vrna_params(), whose SPEEDUP_PARAMS cache is unsynchronised (Defect B). The
 * fold side is clean -- vrna_mfe_batch() reaches vrna_fold_compound_prepare()
 * with VRNA_OPTION_MFE, and vrna_params_prepare() only touches the already-built
 * fc->params on that path (params.c:428-441) -- but process_record() is
 * dispatched to the -j pool from inside the fold, and if IT builds compounds
 * they race the builder. So the pipeline declines whenever this returns true.
 *
 * Conservative in the same direction as process_record()'s own use: anything not
 * enumerated counts as needing a compound, which costs speed, not correctness.
 */
static int
output_needs_compound(struct options *opt)
{
  return !(opt->noPS && !opt->pf && !opt->MEA && !opt->lucky &&
           !opt->verbose && !opt->benchmark &&
           !fold_constrained && !opt->constraint_file &&
           !opt->probing_data && !opt->ligandMotif &&
           !opt->cmds && !opt->mod_params);
}


/* A chunk between its two halves.
 *
 * flush_gpu_chunk() used to do both -- build every fold compound, then fold them
 * -- in one serial pass, which means the GPU sits idle for the whole build and
 * the cores sit idle for the whole fold. At 400 x 5601 that idle is 33.9% of
 * wall and `build` is 72% of it (STRESS272_RESULTS.md §14).
 *
 * Splitting it here is a PURE REFACTOR and changes nothing on its own; it exists
 * so the build of one chunk can later be overlapped with the fold of the
 * previous one. `built` is what separates a batch that owns compounds from one
 * that does not, so the teardown path can tell them apart.
 */
struct gpu_batch {
  struct record_data    **chunk;
  int                     n;
  vrna_fold_compound_t  **VC;
  char                  **Str;
  float                  *EN;
  int                     built;
};


/* How many threads build one chunk's fold compounds (option A).
 *
 * DEFAULT "auto" (nproc) SINCE 2026-09-12. It shipped serial as a new,
 * unproven knob, and `MERGING.md` has listed it under "Default ON" that whole
 * time -- the doc was describing the intent and the code never got there.
 *
 * WHAT CHANGED IS THE SIZE OF THE PRIZE, not the confidence. `build` was 24.5%
 * of wall on a T4. On an A100 the GPU phases are 3-5x faster and `build` does
 * not move at all -- 121.4 s of a 224.9 s wall, **54%**
 * (STRESS272_RESULTS.md 23.5). Measured here at 60 x 2400 on 12 cores:
 * build 1.684 -> 0.260 s, **6.47x**, byte-identical at 2/3/4/6/8/12/auto.
 *
 * Scaling flattens past ~4-6 threads, which is what an allocation- and
 * bandwidth-heavy O(n^2) table build should do; the remaining gain to 12 is
 * real but shallow. `auto` rather than a tuned constant because the right
 * number is a property of the HOST, and this project has been wrong before
 * about numbers that are (see the heterogeneous scope note).
 *
 * ON OVERSUBSCRIPTION WITH -j: the output pool and these builders can overlap
 * on the pipeline path, so a `-j nproc` run briefly has 2x nproc runnable
 * threads. Left alone deliberately -- the builders are short-lived and the
 * fold they overlap is GPU-bound, so the cost is context switches rather than
 * contention, and capping it would need a shared thread budget this driver
 * does not have. RNA_BACKTRACK_THREADS subtracts `cpu_queue_threads` for the
 * same reason, but that queue is retired on this branch and is always 0 here.
 *
 * "0" or "1" still forces serial, which is how the A/B is run.
 *
 * SAFE ONLY BECAUSE Defect B IS FIXED. vrna_fold_compound() reaches
 * vrna_params(), whose SPEEDUP_PARAMS cache was shared mutable state with no
 * lock; params.c now guards it. Before that fix this knob was undefined
 * behaviour that happened to look fine whenever every record shared one model
 * -- measured 0 of 160 000 tables corrupted with identical model details,
 * 25 140 of 160 000 (15.7 %) when they differ. RNAfold always passes one md, so
 * the failure would never have shown up here and would have shown up in a
 * library caller doing a temperature sweep.
 */
static int
rnafold_build_threads(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_BUILD_THREADS");

    if ((!e) || (!e[0]) || (!strcmp(e, "auto"))) {
      long hw = sysconf(_SC_NPROCESSORS_ONLN);
      v = (hw > 1) ? (int)hw : 1;
    } else {
      v = atoi(e);
      if (v < 1)
        v = 1;
    }
  }

  return v;
}


struct build_range {
  struct gpu_batch  *b;
  struct options    *opt;
  int                lo, hi;      /* [lo, hi) */
};


static void build_one(struct gpu_batch *b, struct options *opt, int i);


static void *
build_range_main(void *p)
{
  struct build_range *r = (struct build_range *)p;
  int                 i;

  for (i = r->lo; i < r->hi; i++)
    build_one(r->b, r->opt, i);

  return NULL;
}


/* Build every fold compound in the batch. Host-only; touches no device state,
 * which is what makes it safe to run beside another batch's GPU work. */
static void
build_gpu_batch(struct gpu_batch *b,
                struct options   *opt)
{
  struct record_data  **chunk = b->chunk;
  int                   n     = b->n;
  int                   i;

  vrna_fold_compound_t  **VC;
  char                  **Str;

  b->VC  = (vrna_fold_compound_t **)vrna_alloc(sizeof(void *) * n);
  b->Str = (char **)vrna_alloc(sizeof(char *) * n);
  b->EN  = (float *)vrna_alloc(sizeof(float) * n);
  VC     = b->VC;
  Str    = b->Str;

  /* stage_build_s was DECLARED and PRINTED but never incremented, so the stage
   * line reported build=0.000 for the life of the project -- which reads as
   * "this costs nothing" when it means "nobody measured it". The deep profile
   * on 2026-09-08 found 61% of wall at 120x5601 outside every timer, and this
   * is one of the four counters that should have been covering it.
   * vrna_fold_compound() builds the ptype and hard-constraint tables, which are
   * O(n^2) per record, so this is a genuine candidate for that remainder. */
  const double t_build = rnafold_now_seconds();
  const int    nthr    = rnafold_build_threads();

  if ((nthr <= 1) || (n < 2)) {
    for (i = 0; i < n; i++)
      build_one(b, opt, i);
  } else {
    /* Records are independent: each writes only VC[i] and Str[i], and the only
     * shared thing any of them touches is the parameter cache inside
     * vrna_params(), which params.c now locks. Static contiguous ranges rather
     * than a work queue -- the records in one chunk are close in length, so the
     * imbalance is small and a queue would add a second shared structure for no
     * measured gain. */
    const int  t = (nthr < n) ? nthr : n;
    pthread_t  *th = (pthread_t *)vrna_alloc(sizeof(pthread_t) * t);
    struct build_range *rg =
      (struct build_range *)vrna_alloc(sizeof(struct build_range) * t);
    int        k, started = 0;

    for (k = 0; k < t; k++) {
      rg[k].b   = b;
      rg[k].opt = opt;
      rg[k].lo  = (int)((long)n * k / t);
      rg[k].hi  = (int)((long)n * (k + 1) / t);

      if (pthread_create(&th[k], NULL, build_range_main, &rg[k]) == 0)
        started++;
      else
        break;              /* fall through: this range is built inline below */
    }

    for (k = started; k < t; k++)
      build_range_main(&rg[k]);

    for (k = 0; k < started; k++)
      pthread_join(th[k], NULL);

    free(th);
    free(rg);
  }

  /* WALL time, not summed worker time. Threading a phase once made its timer go
   * NEGATIVE on this project by subtracting worker-seconds from wall-seconds
   * (trap 8, step 2a); this stays a plain wall-clock span so it keeps meaning
   * the same thing whether or not the loop above threaded. */
  stage_build_s += rnafold_now_seconds() - t_build;
  b->built       = 1;
}


/* One record's fold compound. Split out so the serial and threaded paths cannot
 * drift -- they are the same code, called from two loops. */
static void
build_one(struct gpu_batch *b,
          struct options   *opt,
          int               i)
{
  struct record_data **chunk = b->chunk;

  b->VC[i] = vrna_fold_compound(chunk[i]->sequence, &(opt->md), VRNA_OPTION_DEFAULT);

  /* The chunk path builds its OWN fold compounds, so it has to apply every
   * per-record constraint that process_record() would apply to its own. Skip
   * this and the batch folds unconstrained and returns a plausible, wrong
   * structure -- the precise failure mode the routing guard exists to
   * prevent, arriving through the driver rather than through the device.
   *
   * process_record() still applies constraints to the compound it builds for
   * the partition function, MEA and the no-solution check; these two
   * compounds are separate objects, so nothing is constrained twice. */
  if (fold_constrained)
    apply_constraints(b->VC[i],
                      opt->constraint_file,
                      (const char **)chunk[i]->rest,
                      chunk[i]->multiline_input,
                      opt->constraint_enforce,
                      opt->constraint_canonical,
                      1 /* quiet: process_record() reports each record once */);

  b->Str[i] = (char *)vrna_alloc(sizeof(char) * (strlen(chunk[i]->sequence) + 1));
}


/* Fold a built batch, hand the answers back to the records, release the device
 * and dispatch the output. Everything here that touches the GPU lives on ONE
 * thread; build_gpu_batch() is the half that may run beside it. */
static void
fold_gpu_batch(struct gpu_batch *b,
               struct options   *opt)
{
  struct record_data    **chunk = b->chunk;
  int                     n     = b->n;
  vrna_fold_compound_t  **VC    = b->VC;
  char                  **Str   = b->Str;
  float                  *EN    = b->EN;
  int                     i;

  /* THE SEAM. Not par_mfe(): the driver asks the LIBRARY to fold a batch, and
   * the library uses whatever backend is registered -- the CUDA one here, or
   * a plain loop over vrna_mfe() if none is. Nothing below this line, and
   * nothing in this function, is CUDA-specific.
   *
   * That is what makes the diff presentable: the accelerator is a backend, not
   * a fork of the driver, and removing it leaves a correct program. */
  vrna_mfe_batch(VC, (size_t)n, Str, EN);

  /* stage_free_s: also a dead counter until 2026-09-08. Freeing a fold
   * compound releases the same O(n^2) tables build allocated. */
  const double t_free = rnafold_now_seconds();
  for (i = 0; i < n; i++) {
    chunk[i]->prefolded           = 1;
    chunk[i]->prefolded_energy    = EN[i];
    chunk[i]->prefolded_structure = Str[i];   /* handed over; freed with the record */
    vrna_fold_compound_free(VC[i]);
  }
  stage_free_s += rnafold_now_seconds() - t_free;

  /* Release the device state this chunk sized, before the next chunk sizes its
   * own. Without this the second chunk inherits dirty buffers. */
  const double t_teardown = rnafold_now_seconds();
  teardown_gpu();
  teardown_gpu2();
  teardown_gpu3();
  stage_teardown_s += rnafold_now_seconds() - t_teardown;

  /* dispatch only once every record in the chunk has its answer.
   * stage_output_s covers process_record(): formatting, the PS/DP plots when
   * they are not suppressed, and everything the partition function does when
   * -p is set. It is the fourth dead counter, and on a --noPS MFE-only run it
   * should be small -- if it is NOT, that is the finding.
   *
   * CAVEAT, and it is the trap that once made a phase timer report a NEGATIVE
   * number here: RUN_IN_PARALLEL is thpool_add_work() whenever max_threads > 1
   * (parallel_helpers.h:63), so with -j this measures DISPATCH, not the work.
   * Single-threaded it calls fun(data) inline and the number is real. Read
   * stage_output_s only from a run without -j until someone joins the pool
   * inside the timed region. */
  const double t_output = rnafold_now_seconds();
  for (i = 0; i < n; i++)
    RUN_IN_PARALLEL(process_record, chunk[i]);
  stage_output_s += rnafold_now_seconds() - t_output;

  free(VC);
  free(Str);
  free(EN);
  b->VC = NULL; b->Str = NULL; b->EN = NULL; b->built = 0;
}


/* ===================== the build/fold pipeline (option B) =====================
 *
 * One chunk deep: while the GPU folds chunk N, ONE host thread builds chunk
 * N+1's fold compounds. Prize is (chunks-1)/chunks x build -- ~19.6% of wall at
 * 400 x 5601, and exactly ZERO on a single-chunk run, because there is nothing
 * to overlap with. PORT_HETEROGENEOUS_SCOPE.md option B.
 *
 * WHY ONE THREAD AND NOT A POOL. vrna_fold_compound() reaches vrna_params(),
 * whose SPEEDUP_PARAMS cache is four unsynchronised file-scope statics
 * (params.c:100-106, our Defect B). A pool of builders is a data race on it; a
 * single builder never calls it concurrently with itself. The FOLD side must
 * therefore never call vrna_params() either, or the race comes back through the
 * other door -- that is asserted below rather than assumed, because it is the
 * one property this whole design rests on.
 *
 * OFF BY DEFAULT. RNA_BUILD_PIPELINE=1 enables it. A pipelined run holds two
 * chunks of compounds at once (~47 MB per record of ptype + hc->mx), so it
 * trades host RAM for wall clock and the trade is the user's to make until it
 * is measured at scale.
 */
static int
rnafold_build_pipeline(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_BUILD_PIPELINE");
    v = (e && e[0] && e[0] != '0') ? 1 : 0;
  }

  return v;
}


struct builder_arg {
  struct gpu_batch  *b;
  struct options    *opt;
  double             span;      /* builder wall time, for the overlap report */
};


static void *
builder_main(void *p)
{
  struct builder_arg *a  = (struct builder_arg *)p;
  const double        t0 = rnafold_now_seconds();

  build_gpu_batch(a->b, a->opt);
  a->span = rnafold_now_seconds() - t0;
  return NULL;
}


/* How much of the build actually ran BESIDE the GPU.
 *
 * Wall clock alone cannot answer that on a machine whose ceiling is ~4% -- it is
 * under run-to-run noise -- and `accounted - wall` only mirrors wall, so it is
 * not independent evidence. min(builder span, fold span) per chunk is: it is
 * bounded by both and is zero if either did not run. Reported once at the end so
 * a pipelined run says what it achieved rather than leaving it to be inferred. */
static double g_overlap_s   = 0.0;
static double g_builder_s   = 0.0;
static int    g_pipelined_n = 0;


/* The original entry point, now just the two halves back to back. Behaviour is
 * unchanged: the split exists so a caller that wants to overlap them can. */
static void
flush_gpu_chunk(struct record_data **chunk,
                int                  n,
                struct options      *opt)
{
  struct gpu_batch b;
  int              i;

  if (n <= 0)
    return;

  if (n < rnafold_min_gpu_batch()) {
    /* CPU fallback. Not a separate worker queue: upstream's driver already has
     * a per-record parallel path, so an undersized chunk simply goes down it.
     * That is the fork's RNAfold_cpu_queue.c retired rather than ported --
     * MERGING.md flagged it as largely redundant once upstream grew its own
     * thread pool and vrna_ostream_t, and this is where that pays off. The
     * records keep their ostream slots, so output order is unaffected. */
    for (i = 0; i < n; i++)
      RUN_IN_PARALLEL(process_record, chunk[i]);

    return;
  }

  memset(&b, 0, sizeof(b));
  b.chunk = chunk;
  b.n     = n;

  build_gpu_batch(&b, opt);
  fold_gpu_batch(&b, opt);
}


/* ================= CPU slice: fold some records on the cores (option C) ======
 *
 * The accelerated path uses ~ONE core of twelve (measured: 98% of a possible
 * 1200%, and 107% with A and B both on). Everything else is waiting on the
 * device. So hold `m` records back from each chunk and hand them to the -j pool
 * before folding the rest: the cores work through them while the GPU works
 * through the chunk. PORT_CPU_QUEUE_SCOPE.md.
 *
 * NO NEW FOLDING CODE. A record that never entered a chunk has prefolded == 0,
 * so process_record() folds it itself -- the same path an undersized chunk has
 * always taken. What is new is only the decision to send it there.
 *
 * REQUIRES -j. Without a pool RUN_IN_PARALLEL() runs inline, so the slice would
 * be folded SERIALLY BEFORE the GPU starts, which is strictly worse than not
 * slicing at all. m is forced to 0 in that case, and that is asserted rather
 * than assumed because the failure mode is a silent slowdown, not an error.
 *
 * SAFE ONLY BECAUSE Defect B IS FIXED: several pool threads now call
 * vrna_fold_compound() at once. Before params.c was locked this was UB.
 */
static int
rnafold_cpu_slice(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_CPU_SLICE");
    v = (e && e[0] && e[0] != '0') ? 1 : 0;
  }

  return v;
}


static pthread_mutex_t  g_slice_mtx  = PTHREAD_MUTEX_INITIALIZER;
static long             g_slice_out  = 0;   /* dispatched but not yet finished */
static long             g_slice_done = 0;   /* finished, for the report */
static int              g_slice_m    = 0;   /* current slice size */


/* Called by process_record() when a sliced record is finished. */
static void
cpu_slice_finished(void)
{
  pthread_mutex_lock(&g_slice_mtx);
  g_slice_out--;
  g_slice_done++;
  pthread_mutex_unlock(&g_slice_mtx);
}


static long
cpu_slice_outstanding(void)
{
  long v;

  pthread_mutex_lock(&g_slice_mtx);
  v = g_slice_out;
  pthread_mutex_unlock(&g_slice_mtx);

  return v;
}


/* Move the slice out of `chunk` and onto the pool. Returns how many records are
 * left for the GPU, compacted to the front.
 *
 * WHICH records: the SHORTEST ones. CPU fold cost goes as ~n^3 while the slice's
 * whole job is to finish before the GPU does, so a single long record in the
 * slice is the tail-imbalance failure -- it can still be folding after the device
 * has finished everything, which turns C from a disappointment into a
 * regression. Selection is a partial sort by length, not a shuffle.
 */
static int
cpu_slice_take(struct record_data **chunk,
               int                  n,
               struct options      *opt)
{
  int   m, i, k;
  long  behind;

  if ((!rnafold_cpu_slice()) || (opt->jobs <= 1) || (n < 2))
    return n;                   /* no pool: RUN_IN_PARALLEL would be inline */

  /* Adapt on OUTCOME, not on a model of the two throughputs: the ratio depends
   * on the card's clock state, which moves. If the pool cleared its last slice
   * before the device finished, it had spare capacity -- grow. If work is still
   * outstanding, it did not -- shrink by the backlog. */
  behind = cpu_slice_outstanding();

  /* Additive increase, multiplicative decrease. The first version subtracted the
   * backlog outright and OSCILLATED 0 -> jobs -> 0, averaging half the slice it
   * should have run and converging on 1. AIMD is stable for the same reason it
   * is everywhere else. */
  if (behind > 0) {
    g_slice_m /= 2;
  } else {
    const int step = (opt->jobs > 2) ? opt->jobs / 2 : 1;
    g_slice_m += step;
  }

  if (g_slice_m < 0)
    g_slice_m = 0;

  /* THE CAP, and it was set wrong the first time.
   *
   * Every record handed to the cores is a record the GPU does not get, and the
   * first version capped the slice at n/8 to protect BATCH WIDTH. That reasoning
   * came from MIN_GPU_BATCH and the chunking work -- but the stress runs
   * measured the opposite at scale: going from 5 chunks to 14 (2.8x NARROWER
   * batches) cost 0.6% of wall. Batch width is nearly free to lose once the
   * device is saturated, so n/8 was throttling the slice for a cost that is not
   * there, and the measured result was a ~4% REGRESSION with only 5% of records
   * offloaded.
   *
   * The real constraint is the TAIL: a slice still folding after the device has
   * finished everything sets the wall. That is what the outcome feedback above
   * is for, so the cap only has to keep the GPU above MIN_GPU_BATCH and leave
   * the controller room to find the balance.
   *
   * RNA_CPU_SLICE_CAP is the maximum percent of a chunk, so the trade can be
   * measured rather than argued. */
  {
    int         cap_pct = 33;
    const char *e       = getenv("RNA_CPU_SLICE_CAP");
    const int   floor_n = rnafold_min_gpu_batch();
    int         width_cap;

    if ((e) && (e[0])) {
      cap_pct = atoi(e);
      if (cap_pct < 0)   cap_pct = 0;
      if (cap_pct > 90)  cap_pct = 90;
    }

    width_cap = (int)((long)n * cap_pct / 100);

    if (g_slice_m > width_cap)
      g_slice_m = width_cap;

    if (n - g_slice_m < floor_n)
      g_slice_m = (n > floor_n) ? (n - floor_n) : 0;
  }

  m = g_slice_m;

  if (m <= 0)
    return n;

  /* Partial selection sort: pull the m shortest to the end of the array. */
  for (k = 0; k < m; k++) {
    int  best = 0;
    size_t best_len = (size_t)-1;

    for (i = 0; i < n - k; i++) {
      size_t l = strlen(chunk[i]->sequence);

      if (l < best_len) {
        best_len = l;
        best     = i;
      }
    }

    {
      struct record_data *t = chunk[best];
      chunk[best]  = chunk[n - k - 1];
      chunk[n - k - 1] = t;
    }
  }

  pthread_mutex_lock(&g_slice_mtx);
  g_slice_out += m;
  pthread_mutex_unlock(&g_slice_mtx);

  for (i = n - m; i < n; i++) {
    chunk[i]->cpu_slice = 1;
    RUN_IN_PARALLEL(process_record, chunk[i]);
  }

  return n - m;
}


/* One-deep pipeline state. Static because main()'s three flush sites share it. */
static struct gpu_batch *g_pending = NULL;


/* Fold and release whatever is waiting. Safe to call when nothing is. */
static void
pipeline_drain(struct options *opt)
{
  if (g_pending) {
    struct gpu_batch *b = g_pending;

    g_pending = NULL;          /* clear FIRST: fold_gpu_batch() dispatches output */
    fold_gpu_batch(b, opt);
    free(b->chunk);
    free(b);
  }
}


/* Pipelined replacement for flush_gpu_chunk(). Identical behaviour when
 * RNA_BUILD_PIPELINE is unset. */
static void
pipeline_flush(struct record_data **chunk,
               int                  n,
               struct options      *opt)
{
  struct gpu_batch   *b;
  struct builder_arg  a;
  pthread_t           th;
  int                 i;

  if (n <= 0)
    return;

  /* Hand part of the chunk to the cores first, so they are already working by
   * the time the device starts. Returns the number of records still bound for
   * the GPU, with those records compacted to the front of `chunk`. */
  n = cpu_slice_take(chunk, n, opt);

  if (n <= 0)
    return;                     /* the whole chunk went to the cores */

  /* Declined when the output path builds compounds of its own: those run on the
   * -j pool from inside the fold and would race the builder thread through
   * vrna_params(). Serial is slower, not wrong. */
  if ((!rnafold_build_pipeline()) || (output_needs_compound(opt))) {
    flush_gpu_chunk(chunk, n, opt);
    return;
  }

  if (n < rnafold_min_gpu_batch()) {
    /* Too small for the device, exactly as flush_gpu_chunk() decides. Dispatch
     * down the per-record path; ordering is by ostream slot (requested at READ
     * time, main loop), not by dispatch order, so this may precede a pending
     * chunk's output without reordering the file. */
    for (i = 0; i < n; i++)
      RUN_IN_PARALLEL(process_record, chunk[i]);

    return;
  }

  b        = (struct gpu_batch *)vrna_alloc(sizeof(*b));
  memset(b, 0, sizeof(*b));
  b->n     = n;
  /* The caller REUSES its gpu_chunk[] array as soon as this returns, so the
   * batch has to own a copy of the pointers -- not the records, which outlive
   * it either way. */
  b->chunk = (struct record_data **)vrna_alloc(sizeof(void *) * n);
  memcpy(b->chunk, chunk, sizeof(void *) * n);

  a.b   = b;
  a.opt = opt;

  if (pthread_create(&th, NULL, builder_main, &a) != 0) {
    /* No thread available: degrade to the serial shape rather than fail. */
    build_gpu_batch(b, opt);
    pipeline_drain(opt);
    g_pending = b;
    return;
  }

  /* THE OVERLAP. The previous chunk folds on the GPU while `b` builds on the
   * builder thread. Note the device is IDLE again by the time this returns --
   * fold_gpu_batch() tears it down -- which is what keeps main()'s
   * compute_gpu_usable_bytes() query correct without any change there. */
  {
    const double t_fold = rnafold_now_seconds();
    double       fold_span;

    pipeline_drain(opt);
    fold_span = rnafold_now_seconds() - t_fold;

    pthread_join(th, NULL);   /* `a` is a stack local; joining keeps it alive */

    /* Zero on the first chunk, where pipeline_drain() had nothing to fold --
     * which is exactly the (chunks-1)/chunks the prize is bounded by. */
    g_overlap_s += (a.span < fold_span) ? a.span : fold_span;
    g_builder_s += a.span;
    g_pipelined_n++;
  }

  g_pending = b;
}


#endif


/* main loop that processes an input stream */
int
process_input(FILE            *input_stream,
              const char      *input_filename,
              struct options  *opt)
{
  int           ret       = 1;
  int           istty_in  = isatty(fileno(input_stream));
  int           istty_out = isatty(fileno(stdout));

  unsigned int  read_opt = 0;

#ifdef VRNA_WITH_CUDA
  /* The GPU chunk accumulator: a THIRD dispatch path alongside the serial and
   * per-record-parallel ones. Records are held back until a chunk's worth have
   * been read, folded together by par_mfe(), then dispatched normally with
   * their results attached.
   *
   * Ordering is unaffected: vrna_ostream_request() is still called at READ
   * time, in input order, so the output queue is unchanged no matter when a
   * record is actually dispatched. That is what lets the byte-identical bar
   * compare this path against the per-record path directly.
   *
   * Deliberately simple for now -- a fixed chunk size, uniform lengths, no
   * VRAM budgeting and no CPU-queue fallback. The chunker, the mixed-length
   * join mask and the slot machinery are all still in mfe/cuda and can be
   * reconnected once this path is proven correct. */
  struct record_data  **gpu_chunk     = NULL;
  int                   gpu_chunk_n   = 0;
  int                   gpu_chunk_cap = 0;   /* allocated slots in gpu_chunk[] */
  int                   gpu_enabled   = 0;
  int                   gpu_hard_cap  = 0;   /* RNA_GPU_CHUNK, testing override */
  unsigned int          gpu_chunk_len = 0;

  /* The VRAM budget, replacing the fixed record count.
   *
   * chunk_usable_bytes is queried ONCE PER CHUNK, after the previous chunk's
   * teardown, so it sees genuinely free VRAM rather than counting the last
   * chunk's still-resident buffers as unavailable. len_desc holds the accepted
   * lengths in descending order, which is what projected_chunk_bytes() needs
   * to price slot flow. */
  size_t                chunk_usable_bytes = 0;
  int                   chunk_started      = 0;
  int                  *len_desc           = NULL;
  int                   len_desc_n         = 0;
  int                   len_desc_cap       = 0;
  const int             slot_flow_k        = (rnafold_slot_flow() >= 1) ? rnafold_slot_flow() : 1;
  const int             slot_cap_max       = rnafold_slot_capacity_max();

  {
    const char *e   = getenv("RNA_GPU_CHUNK");
    const char *why = NULL;

    gpu_enabled = (vrna_cuda_devices() > 0) && (e) && (e[0]);

    if ((gpu_enabled) && (!gpu_path_usable(opt, &why))) {
      /* Not an error: this run just folds on the CPU, exactly as stock
       * ViennaRNA would. Announced only in verbose mode -- a user who asked
       * for --gquad wants their answer, not a lecture about the accelerator. */
      if (opt->verbose)
        vrna_log_info("CUDA backend not used for this run (%s); "
                      "folding on the CPU path", why);

      gpu_enabled = 0;
    }

    if (gpu_enabled) {
      gpu_hard_cap = atoi(e);            /* 0 or less: budget alone decides */
      if (gpu_hard_cap < 0)
        gpu_hard_cap = 0;

      /* One of only TWO places this driver names CUDA -- the other is the
       * vrna_cuda_devices() probe 20 lines up, which is why this comment used
       * to say "the one and only" and was wrong. After this call the driver
       * asks for batches through vrna_mfe_batch() and the library decides;
       * every other line of the chunking machinery is backend-agnostic, which
       * is the claim that actually matters and the one MERGING.md repeats. */
      /* CIRCULAR: the VRAM model must know BEFORE the first chunk is sized,
       * because fM2_real is a second full triangle per record. */
      rnafold_circ_expect(opt->md.circ);
      vrna_cuda_register_batch_backend();
    }
  }
#endif

  /* print user help if we get input from tty */
  if (istty_in && istty_out) {
    if (fold_constrained) {
      vrna_message_constraint_options_all();
      vrna_message_input_seq("Input sequence (upper or lower case) followed by structure constraint");
    } else {
      vrna_message_input_seq_simple();
    }
  }

  /* set options we wanna pass to vrna_file_fasta_read_record() */
  if (istty_in)
    read_opt |= VRNA_INPUT_NOSKIP_BLANK_LINES;

  if ((!fold_constrained) &&
      (!opt->benchmark))
    read_opt |= VRNA_INPUT_NO_REST;

  /* main loop that processes each record obtained from input stream */
  do {
    char          *rec_sequence, *rec_id, **rec_rest;
    unsigned int  rec_type;
    int           maybe_multiline;

    rec_id          = NULL;
    rec_rest        = NULL;
    maybe_multiline = 0;

    rec_type = vrna_file_fasta_read_record(&rec_id,
                                           &rec_sequence,
                                           &rec_rest,
                                           input_stream,
                                           read_opt);

    if (rec_type & (VRNA_INPUT_ERROR | VRNA_INPUT_QUIT))
      break;

    /*
     ########################################################
     # init everything according to the data we've read
     ########################################################
     */
    if (rec_id) {
      maybe_multiline = 1;
      /* remove '>' from FASTA header */
      rec_id = memmove(rec_id, rec_id + 1, strlen(rec_id));
    }

    /* construct the sequence ID */
    set_next_id(&rec_id, opt->id_control);

    struct record_data *record = (struct record_data *)vrna_alloc(sizeof(struct record_data));

    record->number          = opt->next_record_number;
    record->sequence        = rec_sequence;
    record->SEQ_ID          = fileprefix_from_id(rec_id, opt->id_control, opt->filename_full);
    record->id              = rec_id;
    record->rest            = rec_rest;
    record->multiline_input = maybe_multiline;
    record->options         = opt;
    record->tty             = istty_in && istty_out;
    record->input_filename  = (input_filename) ? strdup(input_filename) : NULL;

    if (opt->output_queue)
      vrna_ostream_request(opt->output_queue, opt->next_record_number++);

#ifdef VRNA_WITH_CUDA
    if (gpu_enabled) {
      /* A chunk must be UNIFORM LENGTH.
       *
       * The sweep shares one triangular layout across the batch, sized from the
       * chunk, and the mixed-length join mask that would relax that lives in
       * the chunker which is not reconnected yet. Feeding it a mixed chunk
       * reaches the device and fails there -- observed as
       *   Assertion `ij>=0 && (size_t)ij < tri_off_H[H+1]-tri_off_H[H]' failed
       *   Assertion `my_c[tri_off_H[H]+ij] == INF' failed
       *   CUDA error: an illegal memory access was encountered
       * on the mixed-length reference inputs. Flushing on a length change is
       * what the 2.3.0 driver did for the same reason, and it keeps this simple
       * path honest until the real chunker is wired back in.
       *
       * Note those were device-side ASSERTS, not silent corruption: this build
       * has no NDEBUG, and they turned an out-of-bounds read into a precise
       * diagnostic. Worth remembering when NDEBUG becomes the release default
       * (plan item I1) that the verification build must keep them. */
      const unsigned int this_len = (unsigned int)strlen(record->sequence);

      /* RNA_GPU_UNIFORM_CHUNKS=1 restores flushing on a length change.
       *
       * Kept as an A/B switch rather than deleted, because the reason mixed
       * chunks first failed here turned out to be the MISSING PER-CHUNK
       * TEARDOWN, not mixed lengths: the 2.3.0 driver dropped its
       * `vc->length != chunk_length` clause deliberately, so records of
       * different lengths accumulate into one batch and the sweep's join mask
       * handles the short ones. Its only flush trigger is the VRAM budget.
       * Having the switch means the claim can be re-tested rather than
       * believed. */
      size_t bytes_projected;
      int    need_flush;

      if ((gpu_chunk_n > 0) && (this_len != gpu_chunk_len) &&
          (getenv("RNA_GPU_UNIFORM_CHUNKS"))) {
        pipeline_flush(gpu_chunk, gpu_chunk_n, opt);
        gpu_chunk_n = len_desc_n = 0;
        chunk_started = 0;
      }

      /* Would this record still fit the chunk's VRAM budget? */
      bytes_projected = projected_chunk_bytes(len_desc, len_desc_n, (int)this_len,
                                              slot_flow_k, slot_cap_max);

      need_flush = (!chunk_started) || (bytes_projected > chunk_usable_bytes);

      if ((gpu_hard_cap > 0) && (gpu_chunk_n >= gpu_hard_cap))
        need_flush = 1;

      if (need_flush) {
        pipeline_flush(gpu_chunk, gpu_chunk_n, opt);
        gpu_chunk_n = len_desc_n = 0;

        /* Query AFTER the flush: flush_gpu_chunk() tears the device state down,
         * so this sees free VRAM rather than counting the previous chunk's
         * buffers as unavailable. */
        chunk_usable_bytes = compute_gpu_usable_bytes();
        chunk_started      = 1;

        /* Degraded case: free VRAM cannot hold even a minimum batch of this
         * length. Folding it on the GPU one chunk at a time would be slower
         * than not using the GPU at all, so this record and its like go down
         * the per-record path. */
        if (gpu_bytes_per_file((int)this_len) * (size_t)VRNA_MIN_GPU_BATCH
            > chunk_usable_bytes) {
          RUN_IN_PARALLEL(process_record, record);
          continue;
        }
      }

      /* grow both arrays as needed -- the chunk size is decided by bytes now,
       * so it is not known in advance */
      if (gpu_chunk_n == gpu_chunk_cap) {
        gpu_chunk_cap = gpu_chunk_cap ? gpu_chunk_cap * 2 : 64;
        gpu_chunk     = (struct record_data **)vrna_realloc(gpu_chunk,
                                                            sizeof(void *) * gpu_chunk_cap);
      }
      if (len_desc_n == len_desc_cap) {
        len_desc_cap = len_desc_cap ? len_desc_cap * 2 : 64;
        len_desc     = (int *)vrna_realloc(len_desc, sizeof(int) * len_desc_cap);
      }

      /* keep len_desc sorted descending: projected_chunk_bytes() walks it in
       * that order to decide which records own slots */
      {
        int p = len_desc_n++;
        while ((p > 0) && (len_desc[p - 1] < (int)this_len)) {
          len_desc[p] = len_desc[p - 1];
          p--;
        }
        len_desc[p] = (int)this_len;
      }

      gpu_chunk_len            = this_len;
      gpu_chunk[gpu_chunk_n++] = record;
    } else {
      RUN_IN_PARALLEL(process_record, record);
    }
#else
    RUN_IN_PARALLEL(process_record, record);
#endif

    if ((opt->probing_data) ||
        (opt->constraint_file && (!opt->constraint_batch))) {
      ret = 0;
      break;
    }

    /* print user help for the next round if we get input from tty */
    if (istty_in && istty_out) {
      if (fold_constrained) {
        vrna_message_constraint_options_all();
        vrna_message_input_seq(
          "Input sequence (upper or lower case) followed by structure constraint");
      } else {
        vrna_message_input_seq_simple();
      }
    }
  } while (1);

#ifdef VRNA_WITH_CUDA
  /* whatever is left over at EOF. A short final chunk is the normal case, and
   * flush_gpu_chunk() sends it down the per-record path if it is below
   * VRNA_MIN_GPU_BATCH -- which is exactly the tail-case fallback the 2.3.0
   * driver had, now expressed once rather than at each call site. */
  if (gpu_enabled) {
    pipeline_flush(gpu_chunk, gpu_chunk_n, opt);
    /* The pipeline leaves the last chunk built but unfolded by construction --
     * there was no following chunk to overlap it with. Nothing else drains it,
     * and skipping this loses the tail silently rather than loudly. */
    pipeline_drain(opt);

    if (g_pipelined_n > 0)
      fprintf(stderr,
              "%-24s build pipeline: %d chunks, builder %.3f s, "
              "OVERLAPPED %.3f s (%.0f%% of builder time hidden behind the GPU)\n",
              "RNAfold.c", g_pipelined_n, g_builder_s, g_overlap_s,
              (g_builder_s > 0.0) ? 100.0 * g_overlap_s / g_builder_s : 0.0);

    if ((rnafold_cpu_slice()) || (g_slice_done > 0))
      fprintf(stderr,
              "%-24s cpu slice: %ld records folded on the cores, "
              "final slice size %d%s\n",
              "RNAfold.c", g_slice_done, g_slice_m,
              (opt->jobs > 1) ? "" : "  <-- -j not set, so the slice stayed 0");

    free(gpu_chunk);
    free(len_desc);
  }
#endif

  return ret;
}


static void
process_record(struct record_data *record)
{
  unsigned int          length;
  struct options        *opt;
  char                  *rec_sequence, *mfe_structure, *ref_structure;
  double                min_en, energy;
  vrna_fold_compound_t  *vc;
  struct output_stream  *o_stream;
  size_t                **mod_positions;
  size_t                mod_param_sets;

  opt = record->options;

  rec_sequence  = strdup(record->sequence);
  ref_structure = NULL;

  mod_positions   = mod_positions_seq_prepare(rec_sequence,
                                              opt->mod_params,
                                              opt->verbose,
                                              &mod_param_sets);

  /* convert DNA alphabet to RNA if not explicitely switched off */
  if (!opt->noconv) {
    vrna_seq_toRNA(rec_sequence);
    vrna_seq_toRNA(record->sequence);
  }

  /* convert sequence to uppercase letters only */
  vrna_seq_toupper(rec_sequence);

  /* THE SECOND FOLD COMPOUND, and why it is now conditional.
   *
   * process_record() is upstream's per-record path: build a compound, fold it,
   * print it. The GPU chunker pre-folds instead and sets record->prefolded --
   * and until 2026-09-09 that skipped only the vrna_mfe() call. The compound
   * was still constructed, rebuilding ptype and the dense (n+1)^2
   * hard-constraint matrix (31.4 MB per record at 5601 nt) for a fold that had
   * already happened, and then freeing them unused.
   *
   * That WAS the whole of stage_output: 124.8 s of a 669 s run at 400 x 5601,
   * and within 3% of stage_build in all twelve stress arms across two GPUs --
   * because it is the same call flush_gpu_chunk() already made and freed.
   *
   * MEASURED, not assumed. Setting vc = NULL on this path and rebuilding took
   * stage_output from 0.130 s to 0.000 s on a 12-record mixed-length batch with
   * NOTHING dereferencing it. See PORT_HOST_WALL_SCOPE.md.
   *
   * The comment on the prefolded branch below used to say the compound was
   * "still used for everything downstream". It is not, once everything
   * downstream is switched off -- and that claim had never been checked.
   *
   * THE PREDICATE IS DELIBERATELY CONSERVATIVE. It enumerates every remaining
   * consumer of `vc` in this function, and anything NOT listed still gets a
   * compound, so a consumer added later is slow rather than wrong. Note
   * !opt->verbose, which is not about speed: that branch dereferences
   * vc->domains_up. mod_bases_apply() is safe unguarded -- it touches fc only
   * when param_set_num > 0 (modified_bases_helpers.c:101), which !opt->mod_params
   * already excludes.
   */
  {
    int need_vc = 1;

#ifdef VRNA_WITH_CUDA
    need_vc = (!record->prefolded) || output_needs_compound(opt);

    /* The fast path needs `length` without a compound. rec_sequence is a strdup
     * of record->sequence put through toRNA and toupper, neither of which
     * changes its length, and flush_gpu_chunk() sized prefolded_structure by
     * strlen(sequence) as well -- so the two agree by construction. ASSERT it
     * anyway and fall back to building on disagreement: the step-2b notes
     * record strlen(seq) != vc->length under whitespace, and if that ever bites
     * here the structure would be printed against the wrong length. Slow rather
     * than wrong. */
    if ((!need_vc) &&
        (strlen(record->prefolded_structure) != strlen(rec_sequence)))
      need_vc = 1;
#endif

    vc = (need_vc)
         ? vrna_fold_compound(rec_sequence, &(opt->md), VRNA_OPTION_DEFAULT)
         : NULL;

    if ((need_vc) && (!vc)) {
      vrna_log_warning("Skipping computations for \"%s\"",
                       (record->id) ? record->id : "identifier unavailable");
      return;
    }

    length = (vc) ? vc->length : (unsigned int)strlen(rec_sequence);
  }

  if ((opt->md.circ) && (vrna_rotational_symmetry(rec_sequence) > 1))
    vrna_log_warning("Input sequence %ld is rotationally symmetric! "
                     "Symmetry correction might be required to compute actual MFE and equilibrium properties!",
                     record->number);

  /* retrieve string stream, 6*length should be enough memory to start with */
  o_stream = get_output_stream(6 * length,
                               opt,
                               record->SEQ_ID,
                               record->input_filename);

  if (record->tty)
    vrna_log_info("length = %d\n", length);

  mfe_structure = (char *)vrna_alloc(sizeof(char) * (length + 1));

  /* parse the rest of the current dataset to obtain a structure constraint */
  if (fold_constrained) {
    apply_constraints(vc,
                      opt->constraint_file,
                      (const char **)record->rest,
                      record->multiline_input,
                      opt->constraint_enforce,
                      opt->constraint_canonical,
                      0 /* the one place constraint problems are reported */);
  }

  if (opt->probing_data)
    apply_probing_data(vc,
                       opt->probing_data);

  if (opt->benchmark)
    ref_structure = vrna_extract_record_rest_structure((const char **)record->rest,
                                                       0,
                                                       VRNA_OPTION_MULTILINE);

#if 0
  if (opt->shape) {
    constraints_add_SHAPE(vc,
                               opt->shape_file,
                               opt->shape_method,
                               opt->shape_conversion,
                               opt->verbose,
                               VRNA_OPTION_DEFAULT);
  }
#endif

  if (opt->ligandMotif) {
    add_ligand_motif(vc,
                     opt->ligandMotif,
                     opt->verbose,
                     VRNA_OPTION_MFE | ((opt->pf) ? VRNA_OPTION_PF : 0));
  }

  if (opt->cmds)
    vrna_commands_apply(vc,
                        opt->cmds,
                        VRNA_CMD_PARSE_DEFAULTS);

  /* apply modified base support if requested */
  mod_bases_apply(vc,
                  mod_param_sets,
                  mod_positions,
                  opt->mod_params);

  /*
   ########################################################
   # begin actual computations
   ########################################################
   */


  /* put header + sequence into output string stream */
  if (!opt->benchmark) {
    vrna_cstr_print_fasta_header(o_stream->data, record->id);
    vrna_cstr_printf(o_stream->data, "%s\n", record->sequence);
  }

#ifdef VRNA_WITH_CUDA
  if (record->prefolded) {
    /* Already folded, as part of a GPU chunk, so the MFE call is skipped.
     *
     * Everything downstream -- constraint checks, plots, partition function,
     * evaluation -- still runs against the fold compound and still formats
     * identically to the per-record path by construction rather than by two
     * implementations agreeing. What changed 2026-09-09 is that the compound is
     * only BUILT when one of those consumers is actually enabled; see the long
     * note at its construction above. When none is, vc is NULL here and this
     * branch needs nothing from it. */
    strncpy(mfe_structure, record->prefolded_structure, strlen(record->sequence) + 1);
    min_en = (double)record->prefolded_energy;
  } else {
    min_en = (double)vrna_mfe(vc, mfe_structure);
  }
#else
  min_en = (double)vrna_mfe(vc, mfe_structure);
#endif

  /* check whether the constraint allows for any solution */
  if ((fold_constrained) || (opt->cmds)) {
    if (min_en == (double)(INF / 100.)) {
      vrna_log_error(
        "Supplied structure constraints create empty solution set for sequence:\n%s",
        record->sequence);
      exit(EXIT_FAILURE);
    }
  }

  if (opt->benchmark) {
    short *pt = vrna_ptable(mfe_structure);
    short *pt_gold = vrna_ptable_from_string(ref_structure, VRNA_BRACKETS_ANY);

    if (opt->benchmark & (1 << 2)) {
      /* remove pseudoknots */
      short *tmp = vrna_pt_pk_remove(pt_gold, 0);
      free(pt_gold);
      pt_gold = tmp;
    }

    if (opt->benchmark & (1 << 3)) {
      /* remove non-canonical basepairs */
      for (unsigned int i = 1; i <= vc->length; ++i) {
        if (pt_gold[i] > i) {
          unsigned int j = pt_gold[i];

          /* remove too-short hairpins */
          if ((j - i) > 3) {
            unsigned int type = vrna_get_ptype_md(vc->sequence_encoding2[i],
                                                  vc->sequence_encoding2[j],
                                                  &(vc->params->model_details));
            if ((type == 0) ||
                (type == 7)) {
              pt_gold[j] = pt_gold[i] = 0;
            }
          } else {
            pt_gold[j] = pt_gold[i] = 0;
          }
        }
      }
    }

    vrna_score_t scores = vrna_compare_structure_pt(pt_gold, pt, 0);

    ATOMIC_BLOCK({
      opt->benchmark_tp += scores.TP;
      opt->benchmark_fp += scores.FP;
      opt->benchmark_tn += scores.TN;
      opt->benchmark_fn += scores.FN;
    })

    if (opt->benchmark_file != stdout) {
      vrna_cstr_print_fasta_header(o_stream->data, record->id);
      vrna_cstr_printf(o_stream->data, "%s\n", record->sequence);

      char *ref_struct_processed = vrna_db_from_ptable(pt_gold);
      vrna_cstr_printf_structure(o_stream->data,
                                 mfe_structure,
                                 " (%6.2f) [predition]",
                                 min_en);
      vrna_cstr_printf_structure(o_stream->data,
                                 ref_struct_processed,
                                 " (%6.2f) [reference]",
                                 vrna_eval_structure(vc, ref_struct_processed));
      free(ref_struct_processed);
    }

    THREADSAFE_FILE_OUTPUT(({
      fprintf(opt->benchmark_file,
              "%s\t%.1f\t%.1f\t%.1f\t%.1f\t%f\t%f\t%f\t%f\n",
              record->id,
              scores.TP,
              scores.FP,
              scores.TN,
              scores.FN,
              scores.TPR,
              scores.PPV,
              scores.MCC,
              scores.F1);
    }))

    fflush(opt->benchmark_file);

    free(pt);
    free(pt_gold);
    goto record_end;
  } else if (!opt->lucky) {
    vrna_cstr_printf_structure(o_stream->data,
                               mfe_structure,
                               record->tty ?  "\n minimum free energy = %6.2f kcal/mol" : " (%6.2f)",
                               min_en);

    if (opt->verbose) {
      if (opt->ligandMotif)
        print_ligand_motifs(vc, mfe_structure, "MFE", o_stream->data);

      if (vc->domains_up) {
        vrna_ud_motif_t *m = vrna_ud_motifs_MFE(vc, mfe_structure);
        print_ud_motifs(vc, m, "MFE", o_stream->data);
        free(m);
      }
    }

    if (!opt->noPS) {
      postscript_layout(vc,
                        record->sequence,
                        mfe_structure,
                        record->SEQ_ID,
                        opt);
    }
  }

  /* vc is NULL on the prefolded fast path -- there are no MFE matrices to
   * release because there is no compound. */
  if ((vc) && (length > 2000))
    vrna_mx_mfe_free(vc);

  if (opt->pf) {
    char *pf_struc = (char *)vrna_alloc(sizeof(char) * (length + 1));
    if (vc->params->model_details.dangles % 2) {
      int dang_bak = vc->params->model_details.dangles;
      vc->params->model_details.dangles = 2;   /* recompute with dangles as in pf_fold() */
      min_en                            = vrna_eval_structure(vc, mfe_structure);
      vc->params->model_details.dangles = dang_bak;
    }

    vrna_exp_params_rescale(vc, &min_en);

    if (length > 2000)
      vrna_log_info("scaling factor %f", vc->exp_params->pf_scale);

    energy = (double)vrna_pf(vc, pf_struc);

    /* in case we abort because of floating point errors */
    if (length > 1600)
      vrna_log_info("free energy = %8.2f", energy);

    if (opt->lucky) {
      ImFeelingLucky(vc,
                     record->sequence,
                     record->SEQ_ID,
                     opt->noPS,
                     opt->filename_delim,
                     o_stream->data,
                     record->tty);
    } else if (opt->md.compute_bpp) {
      vrna_cstr_printf_structure(o_stream->data,
                                 pf_struc,
                                 record->tty ? "\n free energy of ensemble = %6.2f kcal/mol" : " [%6.2f]",
                                 energy);

      if (!opt->noDP) {
        char  *filename_dotplot;
        plist *pl1, *pl2;

        filename_dotplot = NULL;

        /* generate initial element probability lists for dot-plot */
        pl1 = vrna_plist_from_probs(vc, opt->bppmThreshold);
        pl2 = vrna_plist(mfe_structure, 0.95 * 0.95);

        /* add ligand motif annotation if necessary */
        if (opt->ligandMotif)
          add_ligand_motifs_dot(vc, &pl1, &pl2, mfe_structure);

        filename_dotplot = generate_filename("%s%sdp.ps",
                                             "dot.ps",
                                             record->SEQ_ID,
                                             opt->filename_delim);

        if (filename_dotplot) {
          THREADSAFE_FILE_OUTPUT(
            vrna_plot_dp_EPS(filename_dotplot,
                             record->sequence,
                             pl1,
                             pl2,
                             NULL,
                             VRNA_PLOT_PROBABILITIES_DEFAULT));
        }

        free(filename_dotplot);
        free(pl2);

        /* compute stack probabilities and generate dot-plot */
        if (opt->md.compute_bpp == 2) {
          char *filename_stackplot = generate_filename("%s%sdp2.ps",
                                                       "dot2.ps",
                                                       record->SEQ_ID,
                                                       opt->filename_delim);

          pl2 = vrna_stack_prob(vc, 1e-5);

          if (filename_stackplot) {
            THREADSAFE_FILE_OUTPUT(
              PS_dot_plot_list(record->sequence, filename_stackplot, pl1, pl2,
                               "Probabilities for stacked pairs (i,j)(i+1,j-1)"));
          }

          free(pl2);
          free(filename_stackplot);
        }

        free(pl1);
      }

      /* compute centroid structure */
      compute_centroid(vc, opt->ligandMotif, opt->verbose, o_stream->data);

      /* compute MEA structure */
      if (opt->MEA) {
        compute_MEA(vc,
                    opt->MEAgamma,
                    opt->ligandMotif,
                    opt->verbose,
                    o_stream->data);
      }

      vrna_cstr_printf_structure(o_stream->data,
                                 NULL,
                                 " frequency of mfe structure in ensemble %g"
                                 "; ensemble diversity %-6.2f",
                                 vrna_pr_energy(vc, min_en),
                                 vrna_mean_bp_distance(vc));
    } else {
      vrna_cstr_printf_structure(o_stream->data,
                                 NULL,
                                 " free energy of ensemble = %6.2f kcal/mol\n"
                                 " frequency of mfe structure in ensemble %g;",
                                 energy,
                                 vrna_pr_energy(vc, min_en));
    }

    free(pf_struc);
  }

record_end:

  if (opt->output_queue) {
    if (o_stream->individual) {
      /* output immediately */
      ATOMIC_BLOCK(flush_cstr_callback(NULL, record->number, (void *)o_stream));

      /* use dummy element for insert into queue */
      o_stream = NULL;
    }

    vrna_ostream_provide(opt->output_queue, record->number, (void *)o_stream);
  } else {
    ATOMIC_BLOCK(flush_cstr_callback(NULL, record->number, (void *)o_stream));
  }

  /* clean up */
  vrna_fold_compound_free(vc);
  free(record->id);
  free(record->SEQ_ID);
  free(record->sequence);
  free(rec_sequence);
  free(mfe_structure);
  free(ref_structure);

  /* free the rest of current dataset */
  if (record->rest) {
    for (int i = 0; record->rest[i]; i++)
      free(record->rest[i]);
    free(record->rest);
  }

  free(record->input_filename);

#ifdef VRNA_WITH_CUDA
  free(record->prefolded_structure);   /* handed over by flush_gpu_chunk() */

  /* Signal AFTER the work, not after the dispatch: the slice controller asks
   * "did the pool actually clear the last slice", and a counter decremented at
   * dispatch would answer a different question and always say yes. */
  if (record->cpu_slice)
    cpu_slice_finished();
#endif

  free(record);
}


static void
apply_constraints(vrna_fold_compound_t  *fc,
                  const char            *constraints_file,
                  const char            **rec_rest,
                  int                   maybe_multiline,
                  int                   enforceConstraints,
                  int                   canonicalBPonly,
                  int                   quiet)
{
  if (constraints_file) {
    /** [Adding hard constraints from file] */
    vrna_constraints_add(fc, constraints_file, VRNA_OPTION_DEFAULT);
    /** [Adding hard constraints from file] */
  } else {
    char          *cstruc   = NULL;
    unsigned int  length    = fc->length;
    unsigned int  coptions  = (maybe_multiline) ? VRNA_OPTION_MULTILINE : 0;
    cstruc = vrna_extract_record_rest_structure((const char **)rec_rest, 0, coptions);
    unsigned int  cl = (cstruc) ? strlen(cstruc) : 0;

    if (cl == 0) {
      if (!quiet)
        vrna_log_warning("structure constraint is missing");
    } else if (cl < length) {
      if (!quiet)
        vrna_log_warning("structure constraint is shorter than sequence");
    } else if (cl > length) {
      vrna_log_error("structure constraint is too long");
      exit(EXIT_FAILURE);
    }

    if (cstruc) {
      /** [Adding hard constraints from pseudo dot-bracket] */
      unsigned int constraint_options = VRNA_CONSTRAINT_DB_DEFAULT;

      if (enforceConstraints)
        constraint_options |= VRNA_CONSTRAINT_DB_ENFORCE_BP;

      if (canonicalBPonly)
        constraint_options |= VRNA_CONSTRAINT_DB_CANONICAL_BP;

      vrna_constraints_add(fc, (const char *)cstruc, constraint_options);
      /** [Adding hard constraints from pseudo dot-bracket] */

      free(cstruc);
    }
  }
}


static void
compute_MEA(vrna_fold_compound_t  *fc,
            double                MEAgamma,
            const char            *ligandMotif,
            int                   verbose,
            vrna_cstr_t           rec_output)
{
  char  *structure;
  float mea, mea_en;
  /*  this is a hack since vrna_plist_from_probs() always resolves g-quad pairs,
   *  while MEA_seq() still expects unresolved gquads */
  int   gq = fc->exp_params->model_details.gquad;

  fc->exp_params->model_details.gquad = 0;
  plist *pl = vrna_plist_from_probs(fc, 1e-4 / (1 + MEAgamma));

  fc->exp_params->model_details.gquad = gq;

  structure = vrna_MEA(fc, MEAgamma, &mea);

  mea_en = vrna_eval_structure(fc, (const char *)structure);

  vrna_cstr_printf_structure(rec_output, structure, " {%6.2f MEA=%.2f}", mea_en, mea);

  if ((ligandMotif) && (verbose))
    print_ligand_motifs(fc, structure, "MEA", rec_output);

  if ((fc->domains_up) && (verbose)) {
    vrna_ud_motif_t *m = vrna_ud_motifs_MEA(fc, structure, pl);
    print_ud_motifs(fc, m, "MEA", rec_output);
    free(m);
  }

  free(pl);
  free(structure);
}


static void
compute_centroid(vrna_fold_compound_t *fc,
                 const char           *ligandMotif,
                 int                  verbose,
                 vrna_cstr_t          rec_output)
{
  char    *cent;
  double  cent_en, dist;

  cent    = vrna_centroid(fc, &dist);
  cent_en = vrna_eval_structure(fc, (const char *)cent);

  vrna_cstr_printf_structure(rec_output, cent, " {%6.2f d=%.2f}", cent_en, dist);

  if ((ligandMotif) && (verbose))
    print_ligand_motifs(fc, cent, "centroid", rec_output);

  if ((fc->domains_up) && (verbose)) {
    vrna_ud_motif_t *m = vrna_ud_motifs_centroid(fc, cent);
    print_ud_motifs(fc, m, "centroid", rec_output);
    free(m);
  }

  free(cent);
}


static void
add_ligand_motif(vrna_fold_compound_t *vc,
                 char                 *motifstring,
                 int                  verbose,
                 unsigned int         options)
{
  int   r, l, error;
  char  *seq, *str, *ptr;
  float energy;

  l   = strlen(motifstring);
  seq = vrna_alloc(sizeof(char) * (l + 1));
  str = vrna_alloc(sizeof(char) * (l + 1));

  error = 1;

  if (motifstring) {
    error = 0;
    /* parse sequence */
    for (r = 0, ptr = motifstring; *ptr != '\0'; ptr++) {
      if (*ptr == ',')
        break;

      seq[r++] = toupper(*ptr);
    }
    seq[r]  = '\0';
    seq     = vrna_realloc(seq, sizeof(char) * (strlen(seq) + 1));

    for (ptr++, r = 0; *ptr != '\0'; ptr++) {
      if (*ptr == ',')
        break;

      str[r++] = *ptr;
    }
    str[r]  = '\0';
    str     = vrna_realloc(str, sizeof(char) * (strlen(seq) + 1));

    ptr++;
    if (!(sscanf(ptr, "%f", &energy) == 1)) {
      vrna_log_warning("Energy contribution in ligand motif missing!");
      error = 1;
    }

    if (strlen(seq) != strlen(str)) {
      vrna_log_warning("Sequence and structure length in ligand motif have unequal lengths!");
      error = 1;
    }

    if (strlen(seq) == 0) {
      vrna_log_warning("Sequence length in ligand motif is zero!");
      error = 1;
    }

    if (!error && verbose)
      vrna_log_info("Read ligand motif: %s, %s, %f", seq, str, energy);
  }

  if (error || (!vrna_sc_add_hi_motif(vc, seq, str, energy, options)))
    vrna_log_warning("Malformatted ligand motif! Skipping stabilizing motif.");

  free(seq);
  free(str);
}


static char *
annotate_ligand_motif(vrna_fold_compound_t  *vc,
                      const char            *structure)
{
  char            *annote;
  vrna_sc_motif_t *motifs, *m_ptr;

  annote  = NULL;
  motifs  = vrna_sc_ligand_detect_motifs(vc, structure);

  if (motifs) {
    for (m_ptr = motifs; m_ptr->i; m_ptr++) {
      char *tmp_string, *annotation;
      annotation  = NULL;
      tmp_string  = annote;

      if (m_ptr->i != m_ptr->k) {
        annotation = vrna_strdup_printf(" %d %d %d %d 1. 0 0 BFmark",
                                        m_ptr->i,
                                        m_ptr->j,
                                        m_ptr->k,
                                        m_ptr->l);
      } else {
        annotation = vrna_strdup_printf(" %d %d 1. 0 0 Fomark",
                                        m_ptr->i,
                                        m_ptr->j);
      }

      if (tmp_string)
        annote = vrna_strdup_printf("%s %s", tmp_string, annotation);
      else
        annote = strdup(annotation);

      free(tmp_string);
      free(annotation);
    }
  }

  free(motifs);

  return annote;
}


static void
print_ligand_motifs(vrna_fold_compound_t  *vc,
                    const char            *structure,
                    const char            *structure_name,
                    vrna_cstr_t           buf)
{
  vrna_sc_motif_t *motifs, *m_ptr;

  motifs = vrna_sc_ligand_detect_motifs(vc, structure);

  if (motifs) {
    for (m_ptr = motifs; m_ptr->i; m_ptr++) {
      if (m_ptr->i != m_ptr->k) {
        /* put annotation into output vrna_cstr_t */
        vrna_cstr_message_info(buf,
                               "specified motif detected in %s structure: [%d:%d] & [%d:%d]",
                               structure_name,
                               m_ptr->i,
                               m_ptr->k,
                               m_ptr->l,
                               m_ptr->j);
      } else {
        /* put annotation into output vrna_cstr_t */
        vrna_cstr_message_info(buf,
                               "specified motif detected in %s structure: [%d:%d]",
                               structure_name,
                               m_ptr->i,
                               m_ptr->j);
      }
    }
  }

  free(motifs);
}


static char *
annotate_ud_motif(vrna_fold_compound_t  *vc,
                  vrna_ud_motif_t       *motifs)
{
  int   m, i, size;
  char  *annote;

  m       = 0;
  annote  = NULL;

  if (motifs) {
    while (motifs[m].start != 0) {
      char  *tmp_string = annote;
      i     = motifs[m].start;
      size  = vc->domains_up->motif_size[motifs[m].number];
      char  *annotation;

      annotation = vrna_strdup_printf(" %d %d 12 0.4 0.65 0.95 omark", i, i + size - 1);

      if (tmp_string)
        annote = vrna_strdup_printf("%s %s", tmp_string, annotation);
      else
        annote = strdup(annotation);

      free(tmp_string);
      free(annotation);
      m++;
    }
  }

  return annote;
}


static void
print_ud_motifs(vrna_fold_compound_t  *vc,
                vrna_ud_motif_t       *motifs,
                const char            *structure_name,
                vrna_cstr_t           buf)
{
  int m, i, size;

  m = 0;

  if (motifs) {
    while (motifs[m].start != 0) {
      i     = motifs[m].start;
      size  = vc->domains_up->motif_size[motifs[m].number];

      /* put annotation into output vrna_cstr_t */
      vrna_cstr_message_info(buf,
                             "ud motif %d detected in %s structure: [%d:%d]",
                             motifs[m].number,
                             structure_name,
                             i,
                             i + size - 1);
      m++;
    }
  }
}


static void
add_ligand_motifs_dot(vrna_fold_compound_t  *fc,
                      vrna_ep_t             **prob_list,
                      vrna_ep_t             **mfe_list,
                      const char            *structure)
{
  vrna_sc_motif_t *motifs;

  /* append motif positions to the plists of base pair probabilities */
  motifs = vrna_sc_ligand_get_all_motifs(fc);
  if (motifs) {
    add_ligand_motifs_to_list(prob_list, motifs);
    free(motifs);
  }

  /* now scan for the motif in MFE structure again */
  motifs = vrna_sc_ligand_detect_motifs(fc, structure);
  if (motifs) {
    add_ligand_motifs_to_list(mfe_list, motifs);
    free(motifs);
  }
}


static void
add_ligand_motifs_to_list(vrna_ep_t       **list,
                          vrna_sc_motif_t *motifs)
{
  unsigned int    cnt, add, size;
  vrna_ep_t       *ptr;
  vrna_sc_motif_t *m_ptr;

  cnt = 0;
  add = 10;

  /* get current size of list */
  for (size = 0, ptr = (*list); ptr->i; size++, ptr++);

  /* increase length of list */
  (*list) = vrna_realloc((*list), sizeof(vrna_ep_t) * (size + add + 1));

  for (m_ptr = motifs; m_ptr->i; m_ptr++) {
    if (m_ptr->i == m_ptr->k) {
      /* hairpin motif */
      (*list)[size + cnt].i     = m_ptr->i;
      (*list)[size + cnt].j     = m_ptr->j;
      (*list)[size + cnt].p     = 0.95 * 0.95;
      (*list)[size + cnt].type  = VRNA_PLIST_TYPE_H_MOTIF;
      cnt++;
      if (cnt == add) {
        add += 10;
        /* increase length of (*prob_list) */
        (*list) = vrna_realloc((*list), sizeof(vrna_ep_t) * (size + add + 1));
      }
    } else {
      /* internal loop motif */
      (*list)[size + cnt].i     = m_ptr->i;
      (*list)[size + cnt].j     = m_ptr->j;
      (*list)[size + cnt].p     = 0.95 * 0.95;
      (*list)[size + cnt].type  = VRNA_PLIST_TYPE_I_MOTIF;
      cnt++;
      (*list)[size + cnt].i     = m_ptr->k;
      (*list)[size + cnt].j     = m_ptr->l;
      (*list)[size + cnt].p     = 0.95 * 0.95;
      (*list)[size + cnt].type  = VRNA_PLIST_TYPE_I_MOTIF;
      cnt++;
      if (cnt == add) {
        add += 10;
        /* increase length of (*prob_list) */
        (*list) = vrna_realloc((*list), sizeof(vrna_ep_t) * (size + add + 1));
      }
    }
  }

  /* resize pl1 to actual needs */
  (*list)               = vrna_realloc((*list), sizeof(vrna_ep_t) * (size + cnt + 1));
  (*list)[size + cnt].i = 0;
  (*list)[size + cnt].j = 0;
}
