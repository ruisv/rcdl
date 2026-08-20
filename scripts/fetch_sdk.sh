#!/usr/bin/env bash
# Fetch the RKNPU2 runtime headers (and, optionally, librknnrt.so) from the
# public rknn-toolkit2 repository into third_party/rknpu2 (gitignored).
#
# The headers are not part of the board image's packages and are NOT
# redistributed by this repo (they carry Rockchip's own license), so every
# checkout fetches them once:
#
#   scripts/fetch_sdk.sh              # headers for RKNN_VERSION (default 2.3.2)
#   scripts/fetch_sdk.sh --lib        # also librknnrt.so (Linux aarch64) — for
#                                     # a board whose image lacks /usr/lib/librknnrt.so
#   RKNN_VERSION=2.3.0 scripts/fetch_sdk.sh
#
# Match RKNN_VERSION to the board's runtime (model_info prints it) — a model
# converted with toolkit X.Y needs runtime >= X.Y, and the header must match
# the runtime you link.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="${RKNN_VERSION:-2.3.2}"
BASE="${RKNN_SDK_BASE:-https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v${VER}/rknpu2/runtime/Linux/librknn_api}"
DEST="${PROJ}/third_party/rknpu2"
WANT_LIB=0
[ "${1:-}" = "--lib" ] && WANT_LIB=1

mkdir -p "${DEST}/include" "${DEST}/lib"
for h in rknn_api.h rknn_matmul_api.h rknn_custom_op.h; do
  echo ">> ${h}"
  curl -fsSL -o "${DEST}/include/${h}" "${BASE}/include/${h}"
done
if [ "$WANT_LIB" = "1" ]; then
  echo ">> librknnrt.so (aarch64)"
  curl -fsSL -o "${DEST}/lib/librknnrt.so" "${BASE}/aarch64/librknnrt.so"
fi
echo "${VER}" > "${DEST}/VERSION"
echo ">> rknpu2 ${VER} -> ${DEST}"
