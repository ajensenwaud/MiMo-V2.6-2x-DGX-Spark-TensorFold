#!/usr/bin/env bash
# Start MiMo-V2.6-Flash: rank 1 on the worker, then rank 0 here (the API on :$PORT). ./start.sh restart bounces both.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/config.sh

[[ "${1:-}" == restart ]] && ./stop.sh
docker image inspect "$IMAGE:latest" >/dev/null 2>&1 || die "no $IMAGE image: run scripts/prepare.sh"
mkdir -p "$CACHE_DIR"
if [[ "$TP" == 2 ]]; then
  log "rank 1 on $WORKER"
  ssh "$WORKER" "mkdir -p $CACHE_DIR; docker rm -f $CONTAINER >/dev/null 2>&1;
    docker run -d --name $CONTAINER $(run_args) $IMAGE:latest $(serve_args 1) >/dev/null" </dev/null
fi
log "rank 0 here"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
# shellcheck disable=SC2046
docker run -d --name "$CONTAINER" $(run_args) "$IMAGE:latest" $(serve_args 0) >/dev/null
log "loading (~2 min; first start also compiles kernels, ~5 min): docker logs -f $CONTAINER"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    log "warming up (first start: kernel JIT and decode graph captures)"
    curl -fsS -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":16,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
      -o /dev/null || die "warm-up request failed: docker logs $CONTAINER"
    log "serving $SERVED_NAME at http://$(hostname):$PORT/v1 (TP=$TP, $PARALLEL streams)"
    exit 0
  fi
  docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true || {
    docker logs --tail 30 "$CONTAINER"; die "rank 0 exited"; }
  sleep 5
done
die "not up after 15 min: docker logs $CONTAINER"
