#!/usr/bin/env bash
# Z7 kv8 gate: per arm, passkey retrieval (32K / 120K x 5 depths) and spark-bench quality (+ decode, depth) on the
# kv8 build; Python restarts.  Usage: kv8_gate.sh ARM "ENV" [ARM "ENV" ...]
set -u
T=$HOME/mimo/recipe/tools
L=$HOME/mimo/logs
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $L/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
while [ $# -ge 2 ]; do
  a=$1; x=$2; shift 2
  CONTEXT=608256 BIN=${BIN:-/mimo/zig-out-kv8} AOT=/mimo/oracle/aot5-sm121 EXTRA_ENV="-e TF_TP_ROCE_FAST=1 $x" bash $T/zig_serve.sh >/dev/null 2>&1 || { echo "$a: serve failed"; continue; }
  echo "== $a"
  [ "${NIAH:-1}" = 1 ] && python3 $T/mimo_niah.py http://localhost:8892 --out $L/niah-$a.json | tail -11
  (cd /tmp/spark-bench && /usr/bin/python3 scripts/bench.py --base-url http://localhost:8892 --preset full --suites ${SUITES:-decode,depth,quality} ${BENCH_EXTRA:-} \
     --label "kv8 gate $a" 2>&1 | grep -E "decode (chat|code|agent|report)|depth +[0-9]|quality (gsm8k|mmlu)")
  cp "$(ls -td /tmp/spark-bench/results/*/ | head -1)summary.json" $L/kv8gate-$a.json 2>/dev/null
  docker logs mimo-zig > $L/kv8gate-$a-serve.log 2>&1
  bash $T/zig_serve.sh stop >/dev/null 2>&1
done
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
