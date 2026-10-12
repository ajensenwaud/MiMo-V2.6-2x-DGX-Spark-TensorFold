#!/usr/bin/env bash
# Z7: decode at depth (64K, 128K; 3 samples) per Zig arm, then Python restarts.  Usage: z7_depth.sh ARM "ENV" ...
set -u
T=$HOME/mimo/recipe/tools
L=$HOME/mimo/logs
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $L/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
while [ $# -ge 2 ]; do
  a=$1; x=$2; shift 2
  CONTEXT=608256 BIN=${BIN:-/mimo/zig-out-z7d} AOT=/mimo/oracle/aot5-sm121 EXTRA_ENV="-e TF_TP_ROCE_FAST=1 $x" bash $T/zig_serve.sh >/dev/null 2>&1 || { echo "$a: serve failed"; continue; }
  echo "== $a"
  (cd /tmp/spark-bench && /usr/bin/python3 scripts/bench.py --base-url http://localhost:8892 --suites depth --depths 65536,131072 \
     --depth-samples 3 --label "depth $a" 2>&1 | grep -E "depth +[0-9]")
  docker logs mimo-zig > $L/depth-$a-serve.log 2>&1
  bash $T/zig_serve.sh stop >/dev/null 2>&1
done
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
