# A one-command build for Ubuntu, Debian and WSL: `tools/setup_ubuntu.sh`

*For the ViennaRNA maintainers (TBI). A short companion to `Note_to_TBI.md`, 2026-10-04.*

## What it is

```
git clone https://github.com/lukethegenewriter/CUDA_RNAFold.git
cd CUDA_RNAFold
tools/setup_ubuntu.sh --install
```

The script does, in order:

1. **Prerequisites.** Installs upstream's Debian prerequisite list, using current package
   names, plus what a *git* build needs that the release tarball does not.
2. **SWIG.** Installs a SWIG new enough for the Python interface when the distribution's is
   older.
3. **CUDA.** If an NVIDIA GPU is visible and no CUDA toolkit is installed, adds one.
4. **Build.** `autoreconf -i`, `./configure`, `make`.
5. **Self-check.** Checks the GPU's answer against the CPU's (`RNA_GPU=0`) byte for byte, and
   checks that the GPU path actually ran. A build without the GPU path gives the same answer
   silently, so a matching answer alone proves nothing. Also checks that `import RNA` works.
6. **Install** (with `--install`). Runs `make install`, then checks that the installed
   `RNAfold` and `import RNA` work from a new shell.

It asks before installing packages (`--yes` skips the question), keeps every log in
`setup-logs/`, and stops with one sentence saying what to do, not a page of compiler
output. Options: `--no-cuda`, `--cuda` (install a toolkit on a machine with no GPU),
`--prefix DIR`, `--jobs N`, `--fresh`.

## Why it exists

We followed `README.md`'s "Install from git repository" instructions literally on a **blank
Ubuntu 24.04 WSL distribution**. Seven separate things stopped a newcomer before a working
accelerated build with Python. **Five of them are in upstream 2.7.2 itself**, independent of
CUDA, so they may be worth fixing upstream whatever happens to the rest:

| # | what happens | where | here |
|---|---|---|---|
| 1 | `apt-get install … liblapacke liblapack …` fails: `liblapack` is not a package on current Ubuntu, and apt installs **nothing** | README's list | `liblapack-dev`, `liblapacke-dev` |
| 2 | `make` stops at `RNAlib.info`, Error 127: **texinfo** is needed by a git build but not listed | README's list | listed and installed |
| 3 | `make` stops at `No rule to make target 'xml/*'`: `doc/doxygen/Makefile.am` puts `noinst_DATA = $(REFERENCE_MANUAL_FILES_XML)` outside the `WITH_REFERENCE_MANUAL_BUILD` guard, so a git build needs **doxygen** even with `--without-doc` | build defect | doxygen listed and installed; the rule is untouched |
| 4 | the Python interface is silently off: **python3-dev** is not listed | README's list | listed and installed |
| 5 | still off: configure needs **SWIG ≥ 4.3**, and Ubuntu 24.04 ships 4.2.0 (22.04: 4.0.2). Only a WARNING says so, and the summary just says `no`. A fresh 24.04 has no pip, and `pip install` there fails under PEP 668 | requirement vs distributions | `pipx install swig`, plus `pipx ensurepath` so a later `./configure` in a new shell does not fall back to 4.2 |
| 6 | no hint that a GPU is present but no toolkit is | ours | a configure hint (below) |
| 7 | `sudo make install` puts the module in **`/usr/local/local/lib/pythonX.Y/dist-packages`**, which is not on `sys.path`, so `import RNA` fails at the **default** prefix | **upstream defect** in `ax_python3_devel.m4` | fixed, as a marked DEFECT patch (below) |

## Item 7 in detail

On Debian and Ubuntu the default `sysconfig` scheme is `posix_local`. Its purelib is
`{base}/local/lib/pythonX.Y/dist-packages`: it assumes `base=/usr`, and moves "/usr"
installs to `/usr/local`. `ax_python3_devel.m4` evaluates that template with
`base=${prefix}`, so at the default prefix `/usr/local` the result is `/usr/local/local/…`.

**Reproducer:** `./configure && make && sudo make install && python3 -c 'import RNA'` on a
fresh Ubuntu 24.04.

**The fix:** when the prefix already ends in `/local`, the scheme's extra `/local` is
dropped. Any other prefix, and an explicit `PYTHON3_DIR` / `PYTHON3_EXECDIR`, is unchanged.
It is marked `debian-local-scheme` / `-exec` in both copies of the macro (`m4/` and
`src/RNAxplorer/m4/`).

## What configure now says

Nothing is decided differently; the summary only gets more specific:

* **CUDA backend: no**, when an NVIDIA GPU is visible (`nvidia-smi`) but no `nvcc` exists:
  a `to enable it` line with the toolkit advice. Under WSL it says to install the toolkit
  **only**, never a Linux driver: WSL's driver comes from Windows, and Linux driver libraries
  inside WSL can shadow its `libcuda`.
* **Python 3.x: no** now says why, and what fixes it: `needs SWIG >= 4.3.0, found 4.2.0`, or
  `Python.h missing: install python3-dev`.
* **Install Directories**, when the Python module's directory is not on the interpreter's
  `sys.path` (a custom `--prefix`): the `export PYTHONPATH=…` line to add.

## Two choices worth knowing about

* **The CUDA toolkit has to match the driver, not just the GPU.** A toolkit newer than the
  driver builds fine and then fails at run time. The script reads the driver's CUDA version
  from `nvidia-smi` and installs the newest `cuda-toolkit-X-Y` that fits, from NVIDIA's
  repository: the `wsl-ubuntu` one under WSL, which carries no driver. On the test laptop
  (Windows driver 566.14, CUDA 12.7) it chose 12.6; the repository's newest, 13.x, would have
  failed. Ubuntu's own `nvidia-cuda-toolkit` package also works, but under WSL it pulls in
  `libnvidia-compute-*` driver libraries. They are harmless only because WSL's library
  directory comes first in the loader's search order.
* **It is deliberately narrow:** Debian and Ubuntu (including WSL), amd64 for the automatic
  CUDA step. Anywhere else it stops and says what to install by hand.

## How it was verified

* **A blank Ubuntu 24.04 WSL distribution**, twice:
  - `--no-cuda`: a CPU build with Python, and the new configure hint.
  - `--install`: toolkit 12.6 chosen; configure turned CUDA on; the GPU's answer equalled the
    CPU's; and after install, a new shell with **nothing set** had `RNAfold` on `PATH` and
    `import RNA` working. No driver packages were installed.
* **`Finished_Port`, end to end**, the same way.
* **`tests/zeroconf_configure.sh`**, which builds with the toolkit hidden, all seven cases.
  The CPU-only build is the bioconda/PyPI case.
