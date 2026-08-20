#!/usr/bin/env bash
# Push the working tree workstation -> board. Edit locally, sync, then board_build.sh.
#
# What is ignored by git is not synced either (rsync reads .gitignore), with
# one exception: third_party/rknpu2 — the RKNPU2 headers the board image lacks.
# Deletes on the far side, so everything fetch_models.sh puts in models/ is
# PROTECTED rather than excluded by extension (new asset types would otherwise
# get silently deleted on the next sync).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "${PROJ}/scripts/local.env" ] && source "${PROJ}/scripts/local.env"
BOARD="${RCDL_BOARD:-rk3588}"
DEST="${RCDL_BOARD_DEST:-projects/rcdl}"   # relative to board $HOME

ssh "$BOARD" "mkdir -p ${DEST}"
echo ">> sync ${PROJ} -> ${BOARD}:${DEST}"
rsync -az --delete \
  --exclude '.git/' \
  --include 'third_party/' \
  --include 'third_party/rknpu2/***' \
  --filter ':- .gitignore' \
  --filter 'protect models/*' \
  --filter 'protect build/' \
  --filter 'protect build-*/' \
  "${PROJ}/" "${BOARD}:${DEST}/"
echo ">> synced"
