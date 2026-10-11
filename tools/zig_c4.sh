#!/usr/bin/env bash
# 4-stream decode cells (spark-bench decode, concurrency 4, twice) per Zig arm (Z6 build), then Python restarts.
# Usage: zig_c4.sh ARM "EXTRA_ENV" [ARM "EXTRA_ENV" ...]
set -u
T=$HOME/mimo/recipe/tools
L=$HOME/mimo/logs
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $L/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
while [ $# -ge 2 ]; do
  a=$1; x=$2; shift 2
  CONTEXT=608256 BIN=${BIN:-/mimo/zig-out-z6} AOT=/mimo/oracle/aot5-sm121 EXTRA_ENV="$x" bash $T/zig_serve.sh >/dev/null 2>&1 || { echo "$a: serve failed"; continue; }
  echo "== $a"
  for i in 1 2; do
    (cd /tmp/spark-bench && /usr/bin/python3 scripts/bench.py --base-url http://localhost:8892 --suites decode --concurrency 4 \
       --decode-tokens 512 --label "c4 $a" 2>&1 | grep -E "decode (chat|code|agent|report)")
  done
  docker logs mimo-zig > $L/c4-$a-serve.log 2>&1
  bash $T/zig_serve.sh stop >/dev/null 2>&1
done
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
