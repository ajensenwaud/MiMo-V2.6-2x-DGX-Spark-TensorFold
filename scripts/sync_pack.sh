#!/usr/bin/env bash
# Put the selected pack ($PACK) on both nodes at $MODEL_DIR. Packs are built on the worker (gx11, where the source
# checkpoint lives), so the copy runs worker -> head over the CX7 link, then checks the sha256 manifest on both
# nodes (MANIFEST.sha256 in the pack dir; written here once if absent).
#   PACK=2.27 scripts/sync_pack.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

ssh "$WORKER" "test -f $MODEL_DIR/model.safetensors.index.json" </dev/null || die "no pack at $WORKER:$MODEL_DIR"
if ! ssh "$WORKER" "test -f $MODEL_DIR/MANIFEST.sha256" </dev/null; then
  log "writing $MODEL_DIR/MANIFEST.sha256 on the worker"
  ssh "$WORKER" "cd $MODEL_DIR && find . -type f ! -path './.cache/*' ! -name MANIFEST.sha256 -printf '%P\n' |
    sort | xargs -P 8 -I{} sha256sum {} | sort -k2 > MANIFEST.sha256" </dev/null
fi
log "copying $PACK pack to the head ($MODEL_DIR)"
mkdir -p "$MODEL_DIR"
rsync -a --partial --exclude .cache "$WORKER:$MODEL_DIR/" "$MODEL_DIR/"
(cd "$MODEL_DIR" && sha256sum --quiet -c MANIFEST.sha256) || die "head copy fails its manifest"
log "head: $(wc -l < "$MODEL_DIR/MANIFEST.sha256") files match MANIFEST.sha256"
