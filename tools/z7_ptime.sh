#!/usr/bin/env bash
# Z7: prompt chunk costs per arm (tf-mimo ptiming), then Python restarts.  Usage: z7_ptime.sh NAME "ENV" [NAME "ENV" ...]
set -u
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $HOME/mimo/logs/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
while [ $# -ge 2 ]; do
  a=$1; x=$2; shift 2
  BIN=${BIN:-/mimo/zig-out-z7c/bin} AOT=aot5-sm121 MODE=ptiming COMM=roce EXTRA_ENV="-e TF_TP_ROCE_FAST=1 $x" timeout 900 bash ~/mimo/m1/m1_run.sh >/dev/null 2>&1
  cp ~/mimo/m1/ptiming-r0.log $HOME/mimo/logs/ptime-$a.log
  grep -a "^\[ptiming\]" ~/mimo/m1/ptiming-r0.log | sed "s/^/$a /"
done
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
