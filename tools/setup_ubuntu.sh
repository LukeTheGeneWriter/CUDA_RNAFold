#!/bin/bash
# tools/setup_ubuntu.sh -- build this tree on Ubuntu or Debian (including WSL), from a git
# clone, with GPU acceleration when the machine has an NVIDIA GPU.
#
#   git clone --branch Lukes_Flow_Batching https://github.com/lukethegenewriter/CUDA_RNAFold.git && cd CUDA_RNAFold
#   tools/setup_ubuntu.sh            # prerequisites, CUDA if a GPU is visible, build, self-check
#   tools/setup_ubuntu.sh --install  # ... and `sudo make install`, then check `import RNA`
#
# WHY THIS EXISTS. Following the README on a blank Ubuntu 24.04 WSL (2026-10-04) hit seven
# separate stops before a working accelerated build, any one of which ends a newcomer's
# attempt: upstream's apt list names a package that does not exist (`liblapack`, so apt
# installs NOTHING); texinfo and doxygen are needed by a git build but not listed; python3-dev
# is not listed; the Python interface needs SWIG >= 4.3 and 24.04 ships 4.2.0; nothing said how
# to get a CUDA toolkit on WSL; and `sudo make install` put the module where Python does not
# look. This script does the steps that are the same for everyone, and stops with a
# sentence, not a stack of errors, at the ones it cannot do.
#
# Options:
#   --cuda       install a CUDA toolkit even if no GPU is visible (a build-only machine)
#   --no-cuda    never install a toolkit; build CPU-only unless one is already there
#   --install    run `sudo make install` after the build, then check the installed result
#   --prefix DIR configure --prefix=DIR (implies nothing about --install)
#   --jobs N     make -jN (default: all cores)
#   --fresh      re-run autoreconf and configure even if they have run before
#   --yes        do not ask before installing packages
#
# Every command's output goes to setup-logs/ in the tree; the terminal gets one line per step.

set -u
CUDA_MODE=auto; DO_INSTALL=0; PREFIX=""; JOBS=$(nproc 2>/dev/null || echo 4); FRESH=0; YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cuda)     CUDA_MODE=yes ;;
    --no-cuda)  CUDA_MODE=no ;;
    --install)  DO_INSTALL=1 ;;
    --prefix)   PREFIX=$2; shift ;;
    --prefix=*) PREFIX=${1#--prefix=} ;;
    --jobs)     JOBS=$2; shift ;;
    --jobs=*)   JOBS=${1#--jobs=} ;;
    --fresh)    FRESH=1 ;;
    --yes|-y)   YES=1 ;;
    -h|--help)  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
[ -f configure.ac ] && [ -d src/ViennaRNA ] || { echo "run this from a clone of the repository" >&2; exit 2; }
LOGS="$ROOT/setup-logs"; mkdir -p "$LOGS"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mSTOPPED:\033[0m %s\n' "$*" >&2; exit 1; }
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"

# ------------------------------------------------------------------ 0. the machine
say "Checking the machine"
. /etc/os-release 2>/dev/null || die "no /etc/os-release: this script is for Ubuntu and Debian"
case " ${ID:-} ${ID_LIKE:-} " in
  *" ubuntu "*|*" debian "*) ok "system: ${PRETTY_NAME:-$ID}" ;;
  *) die "this script is for Ubuntu and Debian (found ${PRETTY_NAME:-$ID}). On other systems install
the equivalents of the packages listed in step 1 below, then: autoreconf -i && ./configure && make" ;;
esac
WSL=0
grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null && WSL=1
[ $WSL = 1 ] && ok "running under WSL"

GPU=0; DRIVER_CUDA=""
if nvidia-smi -L > "$LOGS/nvidia-smi.log" 2>&1; then
  GPU=1
  DRIVER_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
  ok "GPU: $(head -1 "$LOGS/nvidia-smi.log")"
  ok "the driver supports CUDA up to ${DRIVER_CUDA:-unknown}"
else
  ok "no NVIDIA GPU visible (nvidia-smi did not run)"
  [ $WSL = 1 ] && ok "under WSL the GPU driver comes from WINDOWS: install NVIDIA's Windows driver, not a Linux one"
fi

# ------------------------------------------------------------------ 1. packages
# Upstream's README list, corrected for current Debian/Ubuntu package names, plus what a
# git build needs that the release tarball does not: texinfo (RNAlib.info), doxygen (the
# doc/doxygen xml rule fires even with --without-doc), python3-dev (Python.h), pipx (a SWIG
# new enough for the Python interface when the distribution's is not).
PKGS="build-essential autoconf automake libtool pkg-config gengetopt help2man bison flex
      vim-common texinfo doxygen gfortran liblapacke-dev liblapack-dev python3-dev pipx swig
      ca-certificates wget"
MISSING=""
for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"; done
say "Step 1: build prerequisites"
if [ -n "$MISSING" ]; then
  ok "to install:$MISSING"
  if [ $YES = 0 ] && [ -t 0 ]; then
    read -r -p "    install them with apt now? [Y/n] " a; case "$a" in [nN]*) die "prerequisites not installed" ;; esac
  fi
  $SUDO apt-get update > "$LOGS/apt-update.log" 2>&1 || die "apt-get update failed -- see $LOGS/apt-update.log"
  UPDATED=1
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y $MISSING > "$LOGS/apt-install.log" 2>&1 \
    || die "apt-get install failed -- see $LOGS/apt-install.log"
  ok "installed"
else
  ok "all present"
fi

# xxd builds the PostScript templates. It was part of vim-common up to Ubuntu 22.04 and is its
# OWN package from 24.04 (and Debian 12) -- so on Colab's 24.04 image, where vim-common was
# already installed, configure stopped on "Can't find the postscript hex template" (2026-10-05).
if ! command -v xxd > /dev/null; then
  [ -n "${UPDATED:-}" ] || $SUDO apt-get update > "$LOGS/apt-update.log" 2>&1
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y xxd >> "$LOGS/apt-install.log" 2>&1 \
    && ok "installed xxd (its own package on this release)"
fi

# The packages are a means; these COMMANDS are what the build runs. Checking them by name is
# what catches a package split like xxd's here, instead of 270 lines into configure.log.
TMISS=""
for t in gcc g++ make autoreconf libtoolize pkg-config gengetopt help2man bison flex xxd \
         makeinfo doxygen gfortran python3 wget; do
  command -v "$t" > /dev/null || TMISS="$TMISS $t"
done
[ -z "$TMISS" ] || die "these commands are still missing after apt:$TMISS
Install whatever provides them on this system (apt-file search bin/<name>), then re-run."
ok "every command the build needs is present"

# ------------------------------------------------------------------ 2. SWIG >= 4.3
say "Step 2: SWIG for the Python interface (needs >= 4.3)"
export PATH="$HOME/.local/bin:$PATH"
swig_ver() { swig -version 2>/dev/null | sed -n 's/.*SWIG Version \([0-9.]*\).*/\1/p' | head -1; }
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }
SV=$(swig_ver)
if [ -n "$SV" ] && ver_ge "$SV" 4.3; then
  ok "SWIG $SV ($(command -v swig))"
else
  # Three routes, in order, because no single one works everywhere (2026-10-04):
  #   pipx            plain Ubuntu 24.04 (pip itself refuses there: PEP 668)
  #   python3 -m pip  where Python is not externally managed. Colab: its python3 cannot
  #                   create a venv (ensurepip fails), so pipx fails there
  #   from source     SWIG's own release, into ~/.local; needs only what step 1 installed
  ok "SWIG ${SV:-none} is too old; installing a current one"
  swig_ok() { hash -r; SV=$(swig_ver); [ -n "$SV" ] && ver_ge "$SV" 4.3; }
  ROUTE=""
  if pipx install swig > "$LOGS/swig-pipx.log" 2>&1 || pipx upgrade swig >> "$LOGS/swig-pipx.log" 2>&1; then
    swig_ok && ROUTE=pipx
  fi
  if [ -z "$ROUTE" ]; then
    ok "pipx could not install it (see $LOGS/swig-pipx.log); trying pip"
    if python3 -m pip install swig > "$LOGS/swig-pip.log" 2>&1; then swig_ok && ROUTE=pip; fi
  fi
  if [ -z "$ROUTE" ]; then
    SWIG_TAG=v4.5.0      # the version this tree's interfaces are verified with
    ok "pip could not either (see $LOGS/swig-pip.log); building SWIG $SWIG_TAG from source into ~/.local"
    tmp=$(mktemp -d)
    { wget -q -O "$tmp/swig.tar.gz" "https://github.com/swig/swig/archive/refs/tags/$SWIG_TAG.tar.gz" \
      && tar -xzf "$tmp/swig.tar.gz" -C "$tmp" \
      && cd "$tmp"/swig-* && ./autogen.sh && ./configure --prefix="$HOME/.local" --without-pcre \
      && make -j"$JOBS" && make install; } > "$LOGS/swig-source.log" 2>&1
    cd "$ROOT"; rm -rf "$tmp"
    swig_ok && ROUTE=source
  fi
  [ -n "$ROUTE" ] || die "no route gave SWIG >= 4.3 -- see $LOGS/swig-pipx.log, swig-pip.log and swig-source.log.
The C library and RNAfold do not need it; re-run with SWIG >= 4.3 on PATH to get the Python interface."
  ok "SWIG $SV ($(command -v swig), via $ROUTE)"
  # A later ./configure in a NEW shell would find the distribution's old SWIG again and turn
  # Python off without an error, so ~/.local/bin goes on PATH for future shells too.
  case "$(command -v swig)" in
    "$HOME/.local/bin/"*)
      if command -v pipx > /dev/null && pipx ensurepath >> "$LOGS/swig-path.log" 2>&1; then
        ok "~/.local/bin added to PATH for new shells (pipx ensurepath)"
      elif ! grep -qs '\.local/bin' "$HOME/.bashrc"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc" && ok "~/.local/bin added to PATH in ~/.bashrc"
      fi ;;
  esac
fi

# ------------------------------------------------------------------ 3. CUDA toolkit
# The toolkit must not be NEWER than the driver: a CUDA 13 runtime against a driver that
# supports 12.7 builds fine and then fails at run time. So the newest cuda-toolkit-X-Y that
# the driver supports is chosen, from NVIDIA's repository -- the wsl-ubuntu one under WSL,
# which carries the toolkit only. Under WSL a Linux DRIVER must never be installed: the
# driver is Windows', and Linux driver libraries inside WSL can shadow it.
find_nvcc() {
  command -v nvcc 2>/dev/null && return
  for d in ${CUDA_HOME:-} ${CUDA_PATH:-} /usr/local/cuda $(ls -d /usr/local/cuda-* 2>/dev/null | sort -Vr); do
    [ -x "$d/bin/nvcc" ] && { echo "$d/bin/nvcc"; return; }
  done
}
say "Step 3: CUDA toolkit"
NVCC=$(find_nvcc)
if [ -n "$NVCC" ]; then
  ok "already installed: $NVCC ($("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'))"
elif [ $CUDA_MODE = no ] || { [ $CUDA_MODE = auto ] && [ $GPU = 0 ]; }; then
  ok "skipped ($( [ $CUDA_MODE = no ] && echo '--no-cuda' || echo 'no GPU visible; --cuda installs one anyway')): the build will be CPU-only"
else
  arch=$(dpkg --print-architecture)
  [ "$arch" = amd64 ] || die "automatic CUDA install is only set up for amd64 here (found $arch).
Install a CUDA toolkit by hand, then re-run this script."
  if [ $WSL = 1 ]; then
    REPO=wsl-ubuntu
  else
    case "${ID}-${VERSION_ID:-}" in
      ubuntu-22.04) REPO=ubuntu2204 ;; ubuntu-24.04) REPO=ubuntu2404 ;;
      debian-12) REPO=debian12 ;;
      *) REPO="" ;;
    esac
  fi
  if [ -z "$REPO" ]; then
    ok "no NVIDIA repository mapping for ${PRETTY_NAME:-$ID}; using the distribution's nvidia-cuda-toolkit"
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-cuda-toolkit > "$LOGS/cuda.log" 2>&1 \
      || die "apt-get install nvidia-cuda-toolkit failed -- see $LOGS/cuda.log"
  else
    URL="https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-keyring_1.1-1_all.deb"
    ok "adding NVIDIA's $REPO repository"
    tmp=$(mktemp -d)
    wget -q -O "$tmp/keyring.deb" "$URL" > "$LOGS/cuda.log" 2>&1 || die "could not download $URL"
    $SUDO dpkg -i "$tmp/keyring.deb" >> "$LOGS/cuda.log" 2>&1 || die "could not install the CUDA keyring -- see $LOGS/cuda.log"
    rm -rf "$tmp"
    $SUDO apt-get update >> "$LOGS/cuda.log" 2>&1 || die "apt-get update failed -- see $LOGS/cuda.log"
    # newest cuda-toolkit-X-Y the driver supports (all of them on a GPU-less build machine)
    PICK=""
    for p in $(apt-cache pkgnames cuda-toolkit- 2>/dev/null | grep -E '^cuda-toolkit-[0-9]+-[0-9]+$' | sort -t- -k3,3n -k4,4n); do
      v=$(echo "$p" | sed 's/^cuda-toolkit-\([0-9]*\)-\([0-9]*\)$/\1.\2/')
      if [ -z "$DRIVER_CUDA" ] || ver_ge "$DRIVER_CUDA" "$v"; then PICK=$p; fi
    done
    [ -n "$PICK" ] || die "no cuda-toolkit-X-Y package in the repository fits a driver that supports CUDA $DRIVER_CUDA.
Update the GPU driver$( [ $WSL = 1 ] && echo ' (on WINDOWS)'), then re-run this script."
    ok "installing $PICK (the newest the driver supports)$( [ $WSL = 1 ] && echo ' -- toolkit only, no driver')"
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$PICK" >> "$LOGS/cuda.log" 2>&1 \
      || die "apt-get install $PICK failed -- see $LOGS/cuda.log"
  fi
  NVCC=$(find_nvcc)
  [ -n "$NVCC" ] || die "a toolkit was installed but no nvcc was found -- see $LOGS/cuda.log"
  ok "nvcc: $NVCC ($("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'))"
fi

# ------------------------------------------------------------------ 4. build
say "Step 4: build"
for t in src/libsvm-*.tar.gz src/dlib-*.tar.bz2; do
  [ -f "$t" ] || continue
  d=src/$(basename "$t" | sed -e 's/\.tar\.gz$//' -e 's/\.tar\.bz2$//')
  [ -d "$d" ] || tar -xf "$t" -C src/ || die "could not unpack $t"
done
ok "bundled libsvm and dlib unpacked"
if [ $FRESH = 1 ] || [ ! -x configure ]; then
  autoreconf -i > "$LOGS/autoreconf.log" 2>&1 || die "autoreconf failed -- see $LOGS/autoreconf.log"
  ok "autoreconf done"
fi
CONF_ARGS=""; [ -n "$PREFIX" ] && CONF_ARGS="--prefix=$PREFIX"
if [ $FRESH = 1 ] || [ ! -f config.status ] || [ -n "$NVCC" -a "$(grep -c 'CUDA backend not built' config.log 2>/dev/null)" != 0 ]; then
  ./configure $CONF_ARGS > "$LOGS/configure.log" 2>&1 || die "configure failed -- see the end of $LOGS/configure.log"
  ok "configured"
fi
awk '/^GPU Acceleration/{f=1} f&&/^Features/{f=0} f' "$LOGS/configure.log" | sed -n '3,9p' | sed 's/^/    /'
grep -E '^\s*\* Python 3.x  ' "$LOGS/configure.log" | sed 's/^ */    /'
t0=$(date +%s)
make -j"$JOBS" > "$LOGS/make.log" 2>&1 || die "make failed -- the first errors:
$(grep -n -E '\berror\b|No rule to make' "$LOGS/make.log" | head -8)
(full log: $LOGS/make.log)"
ok "built in $(( $(date +%s) - t0 )) s"

# ------------------------------------------------------------------ 5. self-check
# The GPU's answer must be the CPU's, byte for byte, and the GPU path must actually have run:
# a GPU-less build gives the same answer silently, so "it matched" alone proves nothing.
say "Step 5: self-check"
FA="$LOGS/selfcheck.fa"
python3 -c "
import random; r = random.Random(7)
for k in range(24): print('>s%d' % k); print(''.join(r.choice('ACGU') for _ in range(600)))" > "$FA"
BIN=src/bin/RNAfold
RNA_GPU=0 $BIN --noPS -i "$FA" > "$LOGS/selfcheck.cpu" 2>/dev/null || die "RNAfold failed on the CPU"
RNA_GPU_WORK_FLOOR=0 $BIN --noPS -i "$FA" > "$LOGS/selfcheck.gpu" 2> "$LOGS/selfcheck.err" || die "RNAfold failed -- see $LOGS/selfcheck.err"
cmp -s "$LOGS/selfcheck.cpu" "$LOGS/selfcheck.gpu" || die "the accelerated answer DIFFERS from the CPU's -- please report this, with $LOGS/"
if grep -q "sweep shape:" "$LOGS/selfcheck.err"; then
  ok "RNAfold: the GPU ran, and its answer equals the CPU's"
elif [ -n "$NVCC" ]; then
  ok "RNAfold: answer correct, but the GPU did NOT run:"; grep -i "gpu\|cuda" "$LOGS/selfcheck.err" | head -3 | sed 's/^/      /'
else
  ok "RNAfold: answer correct (CPU-only build)"
fi
if grep -q 'Python 3.x  *: yes' "$LOGS/configure.log"; then
  if (cd /tmp && PYTHONPATH="$ROOT/interfaces/Python" python3 -c "import RNA; print(RNA.cuda_devices())") > "$LOGS/selfcheck.py" 2>&1; then
    ok "Python: import RNA works from the build tree (cuda_devices = $(tail -1 "$LOGS/selfcheck.py"))"
  else
    ok "Python: import RNA FAILED from the build tree -- see $LOGS/selfcheck.py"
  fi
fi

# ------------------------------------------------------------------ 6. install
if [ $DO_INSTALL = 1 ]; then
  say "Step 6: install"
  INST_SUDO=$SUDO
  [ -n "$PREFIX" ] && [ -w "$(dirname "$PREFIX")" ] && INST_SUDO=""
  $INST_SUDO make install > "$LOGS/install.log" 2>&1 || die "make install failed -- see $LOGS/install.log"
  hash -r
  ok "installed; RNAfold: $(command -v RNAfold || echo "not on PATH -- add ${PREFIX:-/usr/local}/bin")"
  if grep -q 'Python 3.x  *: yes' "$LOGS/configure.log"; then
    if (cd /tmp && python3 -c "import RNA") > /dev/null 2>&1; then
      ok "Python: import RNA works"
    else
      line=$(grep -o 'export PYTHONPATH=[^ ]*' "$LOGS/configure.log" | head -1)
      ok "Python: import RNA does not find the module yet. Add this to ~/.bashrc:"
      ok "    ${line:-export PYTHONPATH=<the Python install directory in $LOGS/configure.log>}"
    fi
  fi
fi

say "Done"
ok "logs: $LOGS"
[ $DO_INSTALL = 1 ] || ok "to install: tools/setup_ubuntu.sh --install   (or: sudo make install)"
