#!/usr/bin/env bash
# Quantise our own EXL3 packs from the RL source with exllamav3 (in the serving image), one after another, on the
# node that holds the source (the worker). Resumable: rerun the same command and each job continues from its
# work dir (exllamav3 -r). Run detached:
#   ssh -n gx11 'cd <repo> && setsid nohup scripts/quantise.sh 3.0 4.0 > ~/mimo/quantise.log 2>&1 </dev/null & disown'
# Flags follow the 2.27 pack's own conversion (-hq -cr 250 -cc 2048, mul1 codebook) with the head at 8 bits and the
# MTP heads at 4. The 4 bpw DFlash drafter (dflash/) is copied from the 2.27 pack: same RL source, already EXL3.
# tools/convert.py wraps exllamav3's converter (CuTe DSL argv trap); the TF32 cuBLAS override is removed for the run.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
[[ $# -ge 1 ]] || die "usage: scripts/quantise.sh <bits> [<bits> ...]"
WORK_ROOT="${WORK_ROOT:-$HOME/mimo/quant-work}"
BASELINE_DIR="$MODELS_ROOT/MiMo-V2.6-Flash-RL-exl3-2.27bpw"
docker image inspect "$IMAGE:latest" >/dev/null 2>&1 || die "no $IMAGE image here: run scripts/prepare.sh on the head"
[[ -f "$SRC_DIR/config.json" ]] || die "no source checkpoint at $SRC_DIR"
# EXL3 matrices a complete text pack holds: 47 MoE layers x 256 experts x 3, 48 x 4 attention, 3 dense, the head,
# and the MTP layers' 3 x (4 attention + 3 MLP + eh_proj). The 2.27 pack holds exactly this many (36,316).
EXPECT_MATS=$(( 47 * 256 * 3 + 48 * 4 + 3 + 1 + 3 * 8 ))

for bits in "$@"; do
  out="$MODELS_ROOT/MiMo-V2.6-Flash-RL-exl3-${bits}bpw"
  work="$WORK_ROOT/$bits"
  if [[ -f "$out/model.safetensors.index.json" && -f "$out/quantization_config.json" ]]; then
    log "$bits bpw: already done ($out)"; continue
  fi
  mkdir -p "$work" "$out"
  resume=(); [[ -f "$work/args.json" ]] && resume=(-r)
  log "$bits bpw: converting ${resume[*]:+(resuming) }-> $out"
  docker run --rm --name "mimo-quant-$bits" --gpus all --ipc=host --shm-size 16g \
    -v "$SRC_DIR:/src:ro" -v "$work:/work" -v "$out:/out" -v "$CACHE_DIR:/root/.cache" -v "$PWD/tools:/tools:ro" \
    --entrypoint env "$IMAGE:latest" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 /tools/convert.py \
    -i /src -w /work -o /out -b "$bits" -hb 8 -mb 4 -hq -cr 250 -cc 2048 -cb mul1 "${resume[@]}"
  [[ -f "$out/model.safetensors.index.json" ]] || die "$bits bpw: the converter exited without writing a pack"
  mats=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['weight_map']; print(sum(k.endswith('.trellis') for k in m))" \
    "$out/model.safetensors.index.json")
  [[ "$mats" == "$EXPECT_MATS" ]] || die "$bits bpw: $mats EXL3 matrices, expected $EXPECT_MATS"
  cp -r "$BASELINE_DIR/dflash" "$out/"
  for f in chat_template.jinja preprocessor_config.json; do
    [[ -f "$out/$f" ]] || cp "$SRC_DIR/$f" "$out/" 2>/dev/null || true
  done
  log "$bits bpw: done; $mats EXL3 matrices, $(du -sh "$out" | cut -f1)"
done
log "all requested packs done"
