#!/usr/bin/env bash
# Mirror the board's Rockchip dev headers + libs into third_party/sysroot
# (gitignored) so a workstation editor can resolve includes (IntelliSense /
# clangd). Nothing is built against this on the workstation.
#
#   scripts/fetch_sysroot.sh
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "${PROJ}/scripts/local.env" ] && source "${PROJ}/scripts/local.env"
BOARD="${RCDL_BOARD:-rk3588}"
DEST="${PROJ}/third_party/sysroot"

pull() {  # pull <board_dir> <local_dir> <names...>
  local board_dir="$1" dest="$2"; shift 2
  mkdir -p "${dest}"
  ssh "${BOARD}" "cd '${board_dir}' && tar -cf - $*" | tar -xpf - -C "${dest}"
}
echo ">> sysroot from ${BOARD} -> ${DEST}"
rm -rf "${DEST}"
pull /usr/include "${DEST}/usr/include" rga rockchip
pull /usr/lib/aarch64-linux-gnu "${DEST}/usr/lib/aarch64-linux-gnu" 'librga.so*' 'librockchip_mpp.so*'
pull /usr/lib "${DEST}/usr/lib" 'librknnrt.so'
echo ">> done"
