#!/usr/bin/env bash
# Download the published pack instead of quantising it (~17 h): Hugging Face on the head, pinned to its revision, run
# in the serving image (no host installs) as you; verified; then copied to the worker over the CX7 link and verified
# there. Only PACK=3.0 (the default) is published by this recipe; 4.0: scripts/quantise.sh 4.0.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
[[ "$PACK" == 3.0 ]] || die "PACK=$PACK is not published: scripts/quantise.sh $PACK (on the worker), then scripts/sync_pack.sh"
[[ -n "$PACK_REV" ]] || die "no PACK_REV pinned in scripts/config.sh"
docker image inspect "$IMAGE:latest" >/dev/null 2>&1 || die "no $IMAGE image: run scripts/prepare.sh first"
mkdir -p "$MODEL_DIR"
log "$PACK_REPO @ ${PACK_REV:0:12} -> $MODEL_DIR (113 GB)"
docker run --rm --network host --user "$(id -u):$(id -g)" -e HF_HOME=/out/.hf-home -v "$MODEL_DIR:/out" \
  --entrypoint python3 "$IMAGE:latest" -c \
  "from huggingface_hub import snapshot_download; snapshot_download('$PACK_REPO', revision='$PACK_REV', local_dir='/out')"
rm -rf "$MODEL_DIR/.hf-home" "$MODEL_DIR/.cache"
python3 tools/verify_pack.py "$MODEL_DIR" 36316 || die "the download does not verify"
log "copying to $WORKER over the CX7 link"
ssh "$WORKER" "mkdir -p '$MODEL_DIR'" </dev/null
rsync -a --delete "$MODEL_DIR/" "$WORKER:$MODEL_DIR/"
ssh "$WORKER" python3 - "$MODEL_DIR" 36316 < tools/verify_pack.py || die "the worker's copy does not verify"
log "PACK=$PACK ready on both nodes: ./start.sh"
