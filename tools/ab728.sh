#!/usr/bin/env bash
# A/B on the 16-bit cache: the rebased PR build vs the recipe's release build, decode bench each, same session; Muse back.
set -u
G=$HOME/mimo/gate728; L=$G/logs/ab; mkdir -p "$L"
W=aj@10.100.128.2
IMG=muse-zig-build:0.17.0
MODEL=$HOME/models/mimo-v2.6/MiMo-V2.6-Flash-RL-exl3-3.0bpw
PORT=8891
say() { echo "[ab $(date +%H:%M:%S)] $*"; }
stop_mimo() { docker rm -f mimo-zig >/dev/null 2>&1; ssh -n "$W" "docker rm -f mimo-zig-follow >/dev/null 2>&1"; }
restore() {
  say "restoring Muse"; stop_mimo
  ssh -n "$W" "docker start muse-zig-follow >/dev/null"; docker start muse-zig >/dev/null
  for _ in $(seq 1 120); do
    curl -fsS -m 3 localhost:8888/health 2>/dev/null | grep -q '"ok"' && { say "Muse up on :8888"; return; }
    sleep 10
  done
  say "Muse NOT up after 20 min"
}
trap restore EXIT
say "stopping Muse"; docker stop muse-zig >/dev/null; ssh -n "$W" "docker stop muse-zig-follow >/dev/null"

base() {   # ZIG_OUT_ON_GX10 ZIG_OUT_ON_GX11 ENV
  echo "--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK" \
       "--ulimit memlock=-1 --ulimit stack=67108864" \
       "-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3" \
       "-e TF_TP_WORLD=2 -e TF_TP_MASTER=10.100.128.1 -e TF_TP_PORT=29661 -e TF_COMM_BACKEND=roce" \
       "-e TF_TP_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e TF_TP_ROCE_FAST=1 $3 -v $MODEL:/model:ro"
}
arm() {   # NAME ZIG_OUT_GX10 ZIG_OUT_GX11 ENV
  local name=$1 z0=$2 z1=$3 env=$4
  stop_mimo
  ssh -n "$W" "docker run -d --name mimo-zig-follow $(base x x "$env") -v $z1:/zig:ro -e TF_TP_RANK=1 --entrypoint /zig/bin/tf-mimo $IMG follow /model /zig/aot >/dev/null"
  # shellcheck disable=SC2046
  docker run -d --name mimo-zig $(base x x "$env") -v "$z0:/zig:ro" -e TF_TP_RANK=0 --entrypoint /zig/native/bin/tensorfold-native "$IMG" \
    serve /model --host 0.0.0.0 --port $PORT --name MiMo-V2.6-Flash --parallel 4 --context 608256 \
    --max-tokens 32768 --top-k 20 --top-p 0.95 >/dev/null
  local up=0
  for _ in $(seq 1 180); do
    curl -fsS -m 3 localhost:$PORT/health 2>/dev/null | grep -q '"ok"' && { up=1; break; }
    docker inspect -f '{{.State.Running}}' mimo-zig 2>/dev/null | grep -q true || break
    sleep 5
  done
  [[ $up == 1 ]] || { say "$name: not up"; docker logs --tail 30 mimo-zig; return 1; }
  say "$name: up; $(docker logs mimo-zig 2>&1 | grep -m1 'mimo costs'); GPU $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)"
  (cd "$G/spark-bench" && /usr/bin/python3 scripts/bench.py --base-url http://localhost:$PORT --preset full --suites decode \
     --label "AB $name" > "$L/bench-$name.log" 2>&1)
  cp "$(ls -td "$G"/spark-bench/results/*/ | head -1)summary.json" "$L/bench-$name.json"
  grep -E "decode (chat|code|agent|report)" "$L/bench-$name.log"
  say "$name: GPU after $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)"
  docker logs mimo-zig > "$L/serve-$name.log" 2>&1
}
arm rebased-fp16 "$HOME/mimo/zig-rebase/zig-out" /home/aj/mimo/zig-rebase/zig-out "-e TENSORFOLD_CUDA_KERNELS=/zig/aot"
arm release-fp16 "$HOME/mimo/rel/zig-out" /home/aj/mimo/tensorfold-zig/zig-out "-e TF_MIMO_AOT=/zig/aot -e TF_MIMO_KV8=0"
say done
