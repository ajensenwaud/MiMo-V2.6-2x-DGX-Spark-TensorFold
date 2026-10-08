#!/usr/bin/env bash
# Stop both ranks.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/config.sh

mkdir -p logs   # keep the server's per-request stats (tok/s, prefill, cache, drafting) before the container goes
docker logs "$CONTAINER" > "logs/$CONTAINER-$(date +%Y%m%d-%H%M%S).log" 2>&1 || true
docker rm -f "$CONTAINER" >/dev/null 2>&1 && log "stopped rank 0" || true
if [[ "$TP" == 2 ]]; then
  ssh "$WORKER" "docker rm -f $CONTAINER >/dev/null 2>&1" </dev/null && log "stopped rank 1 on $WORKER" || true
fi
