#!/usr/bin/env bash
# Z1.1 answer key (+ Z1.3 Triton launches with RECORD=1): tools/mimo_zig_capture.py on both ranks, rank 1 on gx11
# over the CX7 link, rank 0 here. Run on gx10 with the serve stopped. Outputs in $OUT on each node:
# rank{R}.safetensors/.json, and with RECORD=1 launches-r{R}.json, jit-r{R}.json and the Triton cache triton-r{R}/.
# RECORD needs TensorFold's tools/zig (triton_aot_manifest.py) at $ZIG_TOOLS on both nodes.
set -u
OUT=${OUT:-$HOME/mimo/oracle}
ZIG_TOOLS=${ZIG_TOOLS:-$HOME/mimo/zig-engine/tools/zig}
WORKER=aj@10.100.128.2
args() {   # TOOLS_DIR RANK
  local rec=""
  [ "${RECORD:-0}" = 1 ] && rec="-v $ZIG_TOOLS:/zigtools:ro -e TRITON_CACHE_DIR=/out/triton-r$2"
  echo "--rm --name mimo-capture --gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband" \
       "--cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864" \
       "-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3" \
       "-e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e TF_ROCE_MAX_KB=512 $rec" \
       "-v $HOME/models/mimo-v2.6/MiMo-V2.6-Flash-RL-exl3-3.0bpw:/model:ro -v $HOME/.cache/mimo-tf:/root/.cache" \
       "-v $OUT:/out -v $1:/tools:ro --entrypoint env mimo-tf:latest -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3" \
       "/tools/mimo_zig_capture.py /model /out --tp 2 --rank $2 ${EXTRA:-}" \
       "$([ "${RECORD:-0}" = 1 ] && echo --tools /zigtools)"
}
mkdir -p "$OUT"
ssh -n "$WORKER" "mkdir -p $OUT; docker rm -f mimo-capture >/dev/null 2>&1; docker run $(args '$HOME/mimo/recipe-tools' 1) > $OUT/capture-r1.log 2>&1" </dev/null &
docker rm -f mimo-capture >/dev/null 2>&1
docker run $(args "$HOME/mimo/recipe/tools" 0) > "$OUT/capture-r0.log" 2>&1
rc=$?
wait
echo "rank 0 rc $rc"
