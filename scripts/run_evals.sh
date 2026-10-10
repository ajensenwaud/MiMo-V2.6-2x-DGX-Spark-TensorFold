#!/usr/bin/env bash
# Task evals for each pack in turn on the served engine (TP=2, DFlash, 4 streams): GSM8K and MMLU-Pro, 500 items
# each, thinking off, greedy. Detached: ssh -n gx10 'cd <repo> && setsid nohup scripts/run_evals.sh 2.27 3.0 4.0
# > ~/mimo-evals.log 2>&1 </dev/null & disown'. Resumable (eval_tasks.py skips items already answered).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
for pack in "$@"; do
  log "pack $pack: starting the server"
  ./stop.sh >/dev/null 2>&1 || true
  PACK="$pack" DRAFTER=1 PARALLEL=4 ./start.sh
  python3 scripts/eval_tasks.py "http://127.0.0.1:$PORT" "$SERVED_NAME" results/evals --tag "$pack" --concurrency 4
  log "pack $pack: done"
done
./stop.sh >/dev/null 2>&1 || true
log "all packs evaluated"
