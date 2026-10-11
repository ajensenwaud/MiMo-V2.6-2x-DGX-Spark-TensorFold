#!/usr/bin/env bash
# Fixed-text decode A/B (tools/mimo_fixed_decode.py, greedy, 3 tags a workload, 1 stream) on Zig arms, then the
# Python serve restarts. Usage: zig_fixed.sh NAME "EXTRA_ENV" [NAME "EXTRA_ENV" ...]
set -u
T=$HOME/mimo/recipe/tools
L=$HOME/mimo/logs
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $L/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
while [ $# -ge 2 ]; do
  a=$1; x=$2; shift 2
  docker rm -f mimo-zig >/dev/null 2>&1; ssh -n aj@10.100.128.2 "docker rm -f mimo-zig-follow" >/dev/null 2>&1 </dev/null
  CONTEXT=${CONTEXT:-608256} BIN=${BIN:-/mimo/zig-out-z7} AOT=${AOT:-/mimo/oracle/aot4-sm121} EXTRA_ENV="$x" bash $T/zig_serve.sh >/dev/null 2>&1 || { echo "$a: serve failed"; continue; }
  (cd /tmp && /usr/bin/python3 $T/mimo_fixed_decode.py http://localhost:8892 --reps 3) > $L/fixed-$a.txt 2>&1
  docker logs mimo-zig > $L/fixed-$a-serve.log 2>&1
  grep "done " $L/fixed-$a-serve.log | grep "tokens=512" | tail -12 | \
    sed "s/.*prompt=\([0-9]*\).*sha=\([0-9a-f]*\).*rounds=\([0-9]*\) accepted=\([0-9\/]*\)/\1 \2 \3 \4/" > $L/fixed-$a-done.txt
  bash $T/zig_serve.sh stop >/dev/null 2>&1
  echo "== $a"; paste $L/fixed-$a.txt $L/fixed-$a-done.txt
done
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
