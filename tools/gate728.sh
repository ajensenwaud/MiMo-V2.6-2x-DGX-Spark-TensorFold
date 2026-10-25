#!/usr/bin/env bash
# PR #728 served gate on the rebased branch (main 2ff01910 + 3 commits): stop Muse, build with nvcc, serve MiMo-V2.6-Flash
# TP=2 on :8891 under each cache, run the lane contracts, the decode bench and (8-bit cache) passkey + quality, and
# restore Muse whatever happens. Run on gx10, detached.
set -u
G=$HOME/mimo/gate728; L=$G/logs; mkdir -p "$L"
SRC=$HOME/mimo/zig-rebase
W=aj@10.100.128.2
IMG=muse-zig-build:0.17.0
MODEL=$HOME/models/mimo-v2.6/MiMo-V2.6-Flash-RL-exl3-3.0bpw
PORT=8891
say() { echo "[gate $(date +%H:%M:%S)] $*"; }

stop_mimo() { docker rm -f mimo-zig >/dev/null 2>&1; ssh -n "$W" "docker rm -f mimo-zig-follow >/dev/null 2>&1"; }
restore() {
  say "restoring Muse"
  docker logs mimo-zig > "$L/serve-last.log" 2>&1
  stop_mimo
  ssh -n "$W" "docker start muse-zig-follow >/dev/null"; docker start muse-zig >/dev/null
  for _ in $(seq 1 120); do
    curl -fsS -m 3 localhost:8888/health 2>/dev/null | grep -q '"ok"' && { say "Muse up on :8888"; return; }
    sleep 10
  done
  say "Muse NOT up after 20 min: docker logs muse-zig"
}
trap restore EXIT

say "stopping Muse"
docker stop muse-zig >/dev/null; ssh -n "$W" "docker stop muse-zig-follow >/dev/null"

say "building (tf-mimo + tensorfold-native, sm_121)"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e ZIG_GLOBAL_CACHE_DIR=/work/.zig-global -v "$SRC:/work" -w /work \
  "$IMG" bash -c 'NVCC=$(which nvcc); zig build tf-mimo native -j8 -Dnvcc=$NVCC -Dtp-nvcc=$NVCC -Dsm=121 --prefix /work/zig-out' \
  > "$L/build.log" 2>&1 || { say "build failed"; tail -30 "$L/build.log"; exit 1; }
[[ -x $SRC/zig-out/native/bin/tensorfold-native && -x $SRC/zig-out/bin/tf-mimo ]] || { say "build incomplete"; exit 1; }
rm -rf "$SRC/zig-out/aot"; cp -r "$G/aot" "$SRC/zig-out/aot"
ssh -n "$W" "mkdir -p mimo/zig-rebase"
rsync -a --delete "$SRC/zig-out/" "$W:mimo/zig-rebase/zig-out/" || { say "rsync to worker failed"; exit 1; }
say "built $(cat "$G/commit.txt")"

ARGS="--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK
 --ulimit memlock=-1 --ulimit stack=67108864
 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3
 -e TF_TP_WORLD=2 -e TF_TP_MASTER=10.100.128.1 -e TF_TP_PORT=29661 -e TF_COMM_BACKEND=roce
 -e TF_TP_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e TF_TP_ROCE_FAST=1 -e TENSORFOLD_CUDA_KERNELS=/zig/aot
 -v $MODEL:/model:ro -v $SRC/zig-out:/zig:ro"
ARGS=$(echo $ARGS)

serve() {   # extra serve flags
  stop_mimo
  ssh -n "$W" "docker run -d --name mimo-zig-follow $ARGS -e TF_TP_RANK=1 --entrypoint /zig/bin/tf-mimo $IMG follow /model /zig/aot >/dev/null"
  # shellcheck disable=SC2086
  docker run -d --name mimo-zig $ARGS -e TF_TP_RANK=0 --entrypoint /zig/native/bin/tensorfold-native "$IMG" \
    serve /model --host 0.0.0.0 --port $PORT --name MiMo-V2.6-Flash --parallel 4 --context 608256 \
    --max-tokens 32768 --top-k 20 --top-p 0.95 "$@" >/dev/null
  for i in $(seq 1 180); do
    curl -fsS -m 3 localhost:$PORT/health 2>/dev/null | grep -q '"ok"' && { say "serving after $((i * 5)) s: $*"; return 0; }
    docker inspect -f '{{.State.Running}}' mimo-zig 2>/dev/null | grep -q true || { say "rank 0 exited"; docker logs --tail 40 mimo-zig; return 1; }
    sleep 5
  done
  say "not up after 15 min"; docker logs --tail 40 mimo-zig; return 1
}

bench() {   # ARM SUITES LABEL [extra]
  local arm=$1 suites=$2 label=$3; shift 3
  (cd "$G/spark-bench" && /usr/bin/python3 scripts/bench.py --base-url http://localhost:$PORT --preset full --suites "$suites" \
     --label "$label" "$@" > "$L/bench-$arm-$suites.log" 2>&1)
  cp "$(ls -td "$G"/spark-bench/results/*/ | head -1)summary.json" "$L/bench-$arm-$suites.json"
  grep -E "decode (chat|code|agent|report)|quality (gsm8k|mmlu)" "$L/bench-$arm-$suites.log"
}

for arm in fp16 fp8; do
  extra=(); [[ $arm == fp8 ]] && extra=(--kv-cache fp8)
  serve "${extra[@]}" || exit 1
  curl -s localhost:$PORT/health > "$L/health-$arm.json"
  say "$arm: lane contracts"
  python3 "$G/tools/upstream_contracts.py" --base http://localhost:$PORT --src "$G/tools" --out "$L/contracts-$arm.json" \
    > "$L/contracts-$arm.log" 2>&1; grep -cE ": PASS" "$L/contracts-$arm.log"; grep -E "FAIL" "$L/contracts-$arm.log"
  say "$arm: decode bench"
  bench "$arm" decode "PR 728 rebased, $arm cache"
  if [[ $arm == fp8 ]]; then
    say "$arm: passkey"
    python3 "$G/tools/mimo_niah.py" http://localhost:$PORT --contexts 32000,120000 --depths 0.1,0.3,0.5,0.7,0.9 \
      --out "$L/niah-$arm.json" > "$L/niah-$arm.log" 2>&1; tail -3 "$L/niah-$arm.log"
    say "$arm: quality (GSM8K 500 + MMLU 228)"
    bench "$arm" quality "PR 728 rebased, $arm cache" --gsm8k-n 500
  fi
  docker logs mimo-zig > "$L/serve-$arm.log" 2>&1
done
say "done"
