#!/usr/bin/env bash
# Stop the Zig engine on both Sparks.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/config.sh
docker rm -f "$ZIG_CONTAINER" >/dev/null 2>&1 || true
[[ "$TP" == 2 ]] && ssh "$WORKER" "docker rm -f $ZIG_CONTAINER-follow >/dev/null 2>&1" </dev/null || true
log "stopped"
