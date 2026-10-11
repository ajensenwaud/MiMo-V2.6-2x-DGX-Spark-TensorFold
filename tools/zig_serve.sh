#!/usr/bin/env bash
# The Zig engine's server (ZIG-PLAN Z5) on :8892: tensorfold-native serve on gx10 (TP rank 0), `tf-mimo follow` on gx11
# (rank 1). Run on gx10 with the Python serve stopped. `zig_serve.sh stop` stops both.
set -u
WORKER=aj@10.100.128.2
PORT=${PORT:-8892}
CONTEXT=${CONTEXT:-65536}
TP_PORT=${TP_PORT:-29661}
AOT=${AOT:-/mimo/oracle/aot4-sm121}
MODEL=$HOME/models/mimo-v2.6/MiMo-V2.6-Flash-RL-exl3-3.0bpw
common() {   # RANK
  echo "--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK" \
       "--ulimit memlock=-1 --ulimit stack=67108864" \
       "-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3" \
       "-e TF_TP_WORLD=2 -e TF_TP_RANK=$1 -e TF_TP_MASTER=10.100.128.1 -e TF_TP_PORT=$TP_PORT -e TF_COMM_BACKEND=nccl" \
       "-e TF_MIMO_AOT=$AOT -v $MODEL:/model:ro -v $HOME/mimo:/mimo"
}
if [ "${1:-}" = stop ]; then
  docker rm -f mimo-zig >/dev/null 2>&1
  ssh -n "$WORKER" "docker rm -f mimo-zig-follow >/dev/null 2>&1" </dev/null
  echo "[zig_serve] stopped"; exit 0
fi
mkdir -p "$HOME/mimo/logs"
docker rm -f mimo-zig >/dev/null 2>&1
ssh -n "$WORKER" "docker rm -f mimo-zig-follow >/dev/null 2>&1; docker run -d --name mimo-zig-follow $(common 1) --entrypoint /mimo/m1/tf-mimo mimo-tf:latest follow /model $AOT >/dev/null" </dev/null
docker run -d --name mimo-zig $(common 0) --entrypoint /mimo/zig-engine/zig-out/native/bin/tensorfold-native mimo-tf:latest \
  serve /model --host 0.0.0.0 --port "$PORT" --name mimo-zig --parallel 4 --context "$CONTEXT" --max-tokens 32768 >/dev/null
for i in $(seq 1 120); do
  sleep 5
  if curl -s -m 2 "localhost:$PORT/health" | grep -q '"ok"'; then echo "[zig_serve] up on :$PORT after $((i * 5))s"; exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -q '^mimo-zig$'; then echo "[zig_serve] rank 0 exited:"; docker logs mimo-zig 2>&1 | tail -15; exit 1; fi
done
echo "[zig_serve] not up after 600s"; docker logs mimo-zig 2>&1 | tail -15; exit 1
