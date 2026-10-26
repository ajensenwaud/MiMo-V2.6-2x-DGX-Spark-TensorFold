#!/usr/bin/env bash
# PR #728 re-gate after a rebase (16-bit cache only): stop Muse, build with nvcc, serve MiMo-V2.6-Flash TP=2 on :8891,
# run the lane contracts, compare each solo token SHA with the previous gate's, run the decode bench, restore Muse.
# Run on gx10, detached:  TAG=e9b69ed7 setsid nohup bash gate-rebase.sh > gate-$TAG.log 2>&1 < /dev/null &
set -u
TAG=${TAG:?set TAG to the main commit the branch is rebased on}
G=$HOME/mimo/gate728; L=$G/logs-$TAG; mkdir -p "$L"
SRC=$HOME/mimo/zig-$TAG
PREV=${PREV:-$G/logs/contracts-fp16.json}
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

say "building (tf-mimo, tensorfold-native, CUDA tests; sm_121)"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e ZIG_GLOBAL_CACHE_DIR=/work/.zig-global -v "$SRC:/work" -w /work \
  "$IMG" bash -c 'NVCC=$(which nvcc); zig build tf-mimo native compile-cuda-tests -j8 -Dnvcc=$NVCC -Dtp-nvcc=$NVCC -Dsm=121 \
  --prefix /work/zig-out --summary all' > "$L/build.log" 2>&1 || { say "build failed"; tail -30 "$L/build.log"; exit 1; }
grep -m1 "Build Summary" "$L/build.log"
[[ -x $SRC/zig-out/native/bin/tensorfold-native && -x $SRC/zig-out/bin/tf-mimo ]] || { say "build incomplete"; exit 1; }
rm -rf "$SRC/zig-out/aot"; cp -r "$G/aot" "$SRC/zig-out/aot"
ssh -n "$W" "mkdir -p mimo/zig-$TAG"
rsync -a --delete "$SRC/zig-out/" "$W:mimo/zig-$TAG/zig-out/" || { say "rsync to worker failed"; exit 1; }
say "built $(cat "$SRC/COMMIT" 2>/dev/null)"

ARGS="--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK
 --ulimit memlock=-1 --ulimit stack=67108864
 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3
 -e TF_TP_WORLD=2 -e TF_TP_MASTER=10.100.128.1 -e TF_TP_PORT=29661 -e TF_COMM_BACKEND=roce
 -e TF_TP_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e TF_TP_ROCE_FAST=1 -e TENSORFOLD_CUDA_KERNELS=/zig/aot
 -v $MODEL:/model:ro"
ARGS=$(echo $ARGS)

stop_mimo
ssh -n "$W" "docker run -d --name mimo-zig-follow $ARGS -v \$HOME/mimo/zig-$TAG/zig-out:/zig:ro -e TF_TP_RANK=1 \
  --entrypoint /zig/bin/tf-mimo $IMG follow /model /zig/aot >/dev/null"
# shellcheck disable=SC2086
docker run -d --name mimo-zig $ARGS -v "$SRC/zig-out:/zig:ro" -e TF_TP_RANK=0 --entrypoint /zig/native/bin/tensorfold-native \
  "$IMG" serve /model --host 0.0.0.0 --port $PORT --name MiMo-V2.6-Flash --parallel 4 --context 608256 \
  --max-tokens 32768 --top-k 20 --top-p 0.95 >/dev/null
up=0
for i in $(seq 1 180); do
  curl -fsS -m 3 localhost:$PORT/health 2>/dev/null | grep -q '"ok"' && { say "serving after $((i * 5)) s"; up=1; break; }
  docker inspect -f '{{.State.Running}}' mimo-zig 2>/dev/null | grep -q true || { say "rank 0 exited"; docker logs --tail 40 mimo-zig; exit 1; }
  sleep 5
done
[[ $up == 1 ]] || { say "not up after 15 min"; docker logs --tail 40 mimo-zig; exit 1; }
curl -s localhost:$PORT/health > "$L/health-fp16.json"
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader > "$L/gpu-start.txt"

say "lane contracts"
python3 "$G/tools/upstream_contracts.py" --base http://localhost:$PORT --src "$G/tools" --out "$L/contracts-fp16.json" \
  > "$L/contracts-fp16.log" 2>&1; grep -cE ": PASS" "$L/contracts-fp16.log"; grep -E "FAIL" "$L/contracts-fp16.log"
python3 - "$PREV" "$L/contracts-fp16.json" <<'EOF'
import json, sys
a, b = (json.load(open(p))["solo"] for p in sys.argv[1:3])
same = [k for k in a if k in b and a[k]["key"] == b[k]["key"]]
print(f"solo SHAs equal to the previous gate: {len(same)}/{len(a)}")
for k in a:
    if k not in same: print("  differs:", k, a[k]["key"], b.get(k, {}).get("key"))
EOF

say "decode bench"
(cd "$G/spark-bench" && /usr/bin/python3 scripts/bench.py --base-url http://localhost:$PORT --preset full --suites decode \
   --label "PR 728 on main $TAG, fp16 cache" > "$L/bench-fp16-decode.log" 2>&1)
cp "$(ls -td "$G"/spark-bench/results/*/ | head -1)summary.json" "$L/bench-fp16-decode.json"
grep -E "decode (chat|code|agent|report)" "$L/bench-fp16-decode.log"
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader > "$L/gpu-end.txt"
docker logs mimo-zig > "$L/serve-fp16.log" 2>&1
say "done"
