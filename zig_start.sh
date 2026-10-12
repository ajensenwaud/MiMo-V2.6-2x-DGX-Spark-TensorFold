#!/usr/bin/env bash
# Start MiMo-V2.6-Flash on the Zig engine: rank 1 on the worker, then the server here (the API on :$PORT).
# Stop the Python server first (./stop.sh): both use the same port and the whole of both GPUs.
# ./zig_start.sh restart bounces both ranks.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/config.sh

[[ "${1:-}" == restart ]] && ./zig_stop.sh
[[ -x "$ZIG_DIR/zig-out/native/bin/tensorfold-native" ]] || die "no Zig build in $ZIG_DIR: run scripts/zig_prepare.sh"
[[ "$TP" == 2 ]] || die "the Zig engine recipe runs on two Sparks (TP=2)"
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then die "the Python server is running: ./stop.sh first"; fi

zig_args() {   # RANK: docker run arguments for one rank
  echo "--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK" \
       "--ulimit memlock=-1 --ulimit stack=67108864" \
       "-e NCCL_SOCKET_IFNAME=$NCCL_IFNAME -e NCCL_IB_HCA=$NCCL_HCAS -e NCCL_IB_GID_INDEX=$NCCL_GID" \
       "-e TF_TP_WORLD=2 -e TF_TP_RANK=$1 -e TF_TP_MASTER=$MASTER -e TF_TP_PORT=$ZIG_TP_PORT" \
       "-e TF_COMM_BACKEND=roce -e TF_TP_ROCE_HCA=$NCCL_HCAS -e TF_TP_ROCE_FAST=1 -e TF_MIMO_AOT=/zig/aot ${ZIG_ENV:-}" \
       "-v $MODEL_DIR:/model:ro -v $ZIG_DIR/zig-out:/zig:ro"
}

log "rank 1 on $WORKER"
ssh "$WORKER" "docker rm -f $ZIG_CONTAINER-follow >/dev/null 2>&1;
  docker run -d --name $ZIG_CONTAINER-follow $(zig_args 1) --entrypoint /zig/bin/tf-mimo $IMAGE:latest follow /model /zig/aot >/dev/null" </dev/null
log "rank 0 here"
docker rm -f "$ZIG_CONTAINER" >/dev/null 2>&1 || true
# shellcheck disable=SC2046
docker run -d --name "$ZIG_CONTAINER" $(zig_args 0) --entrypoint /zig/native/bin/tensorfold-native "$IMAGE:latest" \
  serve /model --host "$HOST" --port "$PORT" --name "$SERVED_NAME" --parallel "$PARALLEL" --context "$ZIG_CONTEXT" \
  --max-tokens "$MAX_TOKENS" --top-k 20 --top-p 0.95 >/dev/null
log "loading (about 4 minutes): docker logs -f $ZIG_CONTAINER"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    log "serving $SERVED_NAME (Zig engine) at http://$(hostname):$PORT/v1"
    exit 0
  fi
  docker inspect -f '{{.State.Running}}' "$ZIG_CONTAINER" 2>/dev/null | grep -q true || {
    docker logs --tail 30 "$ZIG_CONTAINER"; die "rank 0 exited"; }
  sleep 5
done
die "not up after 15 min: docker logs $ZIG_CONTAINER"
