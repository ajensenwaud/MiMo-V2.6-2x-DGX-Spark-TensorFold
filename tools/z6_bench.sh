#!/usr/bin/env bash
# Z6 full spark-bench (decode + depth) on the Zig server (prompt path, Python's lone-stream depth rule), then Python restarts.
set -u
T=$HOME/mimo/recipe/tools
L=$HOME/mimo/logs
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $L/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
CONTEXT=608256 BIN=/mimo/zig-out-z6 AOT=/mimo/oracle/aot5-sm121 EXTRA_ENV="-e TF_TP_ROCE_FAST=1 ${XENV:--e TF_MIMO_DEPTH=python}" bash $T/zig_serve.sh || exit 1
(cd /tmp/spark-bench && /usr/bin/python3 scripts/bench.py --base-url http://localhost:8892 --preset full --suites decode,depth \
   --label "Zig Z6 ${NAME:-python-rule}" 2>&1 | grep -vE "^\s*$" | tee $L/z6-${NAME:-python-rule}.log | grep -E "decode|depth|prefill|done in")
cp "$(ls -td /tmp/spark-bench/results/*/ | head -1)summary.json" $L/z6-${NAME:-python-rule}.json 2>/dev/null
docker logs mimo-zig > $L/z6-${NAME:-python-rule}-serve.log 2>&1
bash $T/zig_serve.sh stop
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
