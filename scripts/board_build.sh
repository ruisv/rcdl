#!/usr/bin/env bash
# Configure + build RCDL on the board inside the `rcdl` conda env (falls back to
# the board's system toolchain when no conda env exists).
#   scripts/board_build.sh              # build
#   scripts/board_build.sh --run        # build, then run model_info ($MODEL on the board)
#   scripts/board_build.sh --clean      # wipe build/ first
#   scripts/board_build.sh --no-python  # skip the nanobind module
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "${PROJ}/scripts/local.env" ] && source "${PROJ}/scripts/local.env"
BOARD="${RCDL_BOARD:-rk3588}"
DEST="${RCDL_BOARD_DEST:-projects/rcdl}"
MODEL="${MODEL:-}"                 # path on board to a .rknn for --run
RUN=0; CLEAN=0; PY=ON
for a in "$@"; do
  case "$a" in
    --run) RUN=1 ;;
    --clean) CLEAN=1 ;;
    --no-python) PY=OFF ;;
    *) echo "unknown arg: $a" >&2; exit 1 ;;
  esac
done

ssh "$BOARD" "RUN=$RUN CLEAN=$CLEAN PY=$PY MODEL='$MODEL' DEST='$DEST' bash -s" <<'REMOTE'
set -euo pipefail
for _c in "$HOME/conda" "$HOME/miniforge3" "$HOME/miniconda3" /opt/conda; do
  if [ -f "$_c/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$_c/etc/profile.d/conda.sh"
    if conda env list | grep -qE '^\s*rcdl\s'; then conda activate rcdl; fi
    break
  fi
done
cd "$HOME/$DEST"
[ "$CLEAN" = "1" ] && rm -rf build
if [ "$PY" = "ON" ] && ! python -c "import nanobind" 2>/dev/null; then
  echo ">> nanobind not importable in this env; building without the Python module"
  PY=OFF
fi
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRCDL_BUILD_PYTHON="$PY"
cmake --build build -j"$(nproc)"
echo ">> build ok: $(ls -1 build/*.so build/model_info 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
if [ "$RUN" = "1" ]; then
  if [ -z "$MODEL" ]; then
    echo ">> --run needs MODEL=/path/to/model.rknn on the board" >&2; exit 1
  fi
  ./build/model_info "$MODEL"
fi
REMOTE
