#!/usr/bin/env bash
# One-time board setup: install Miniforge (aarch64) if no conda is present,
# then create/update the `rcdl` conda env from env/environment.yml. Idempotent.
# Run from the workstation:  scripts/bootstrap_board.sh   (after scripts/sync.sh)
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "${PROJ}/scripts/local.env" ] && source "${PROJ}/scripts/local.env"
BOARD="${RCDL_BOARD:-rk3588}"
DEST="${RCDL_BOARD_DEST:-projects/rcdl}"

echo ">> bootstrapping conda env on ${BOARD}"
ssh "$BOARD" "DEST='$DEST' bash -s" <<'REMOTE'
set -euo pipefail
# A non-interactive ssh shell does not source ~/.bashrc, so probe the known
# install dirs directly and prefer an EXISTING conda over installing a second one.
CONDA_SH=""
for _c in "$HOME/conda" "$HOME/miniforge3" "$HOME/miniconda3" /opt/conda; do
  if [ -f "$_c/etc/profile.d/conda.sh" ]; then CONDA_SH="$_c/etc/profile.d/conda.sh"; break; fi
done
if [ -z "$CONDA_SH" ]; then
  echo ">> no conda found; installing Miniforge3 (aarch64) to \$HOME/miniforge3"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
  tmp="$(mktemp "${TMPDIR:-/tmp}/miniforge.XXXXXX.sh")"
  curl -fsSL "$url" -o "$tmp"
  bash "$tmp" -b -p "$HOME/miniforge3"
  rm -f "$tmp"
  CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
fi
echo ">> using conda at $(dirname "$(dirname "$(dirname "$CONDA_SH")")")"
# shellcheck disable=SC1091
source "$CONDA_SH"

spec="$HOME/${DEST}/env/environment.yml"
if conda env list | grep -qE '^\s*rcdl\s'; then
  echo ">> env 'rcdl' exists; updating"
  [ -f "$spec" ] && conda env update -n rcdl -f "$spec" --prune || true
else
  if [ -f "$spec" ]; then
    conda env create -f "$spec"
  else
    echo ">> spec not synced yet; creating minimal env"
    conda create -y -n rcdl -c conda-forge python=3.12 cmake ninja nanobind numpy pytest
  fi
fi
conda activate rcdl
echo ">> ready:  python=$(python --version 2>&1)  cmake=$(cmake --version | head -1)"
python -c "import nanobind, numpy; print('nanobind', nanobind.__version__, '| numpy', numpy.__version__)"
REMOTE
echo ">> done. Next: scripts/sync.sh && scripts/board_build.sh"
