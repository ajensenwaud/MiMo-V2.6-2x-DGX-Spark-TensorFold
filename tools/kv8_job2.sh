#!/usr/bin/env bash
# Z7 kv8: kernel check against Triton, then step costs at 9,000 / 128,000 with the 8-bit cache; Python restarts.
set -u
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $HOME/mimo/logs/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
B=/mimo/zig-out-kv8b/bin
BIN=$B AOT=aot5-sm121 MODE=kv8check COMM=roce EXTRA_ENV="-e TF_TP_ROCE_FAST=1" timeout 900 bash ~/mimo/m1/m1_run.sh >/dev/null 2>&1
cp ~/mimo/m1/kv8check-r0.log $HOME/mimo/logs/kv8check-r0.log
grep -a "kv8check\]\|error\|panic" ~/mimo/m1/kv8check-r0.log | head -12
for pos in 128000; do
  BIN=$B AOT=aot5-sm121 MODE=timing COMM=roce EXTRA_ENV="-e TF_TP_ROCE_FAST=1 -e TF_MIMO_KV8=1 -e TF_MIMO_TIMING_CONTEXT=135168 -e TF_MIMO_TIMING_POS=$pos" timeout 900 bash ~/mimo/m1/m1_run.sh >/dev/null 2>&1
  cp ~/mimo/m1/timing-r0.log $HOME/mimo/logs/kv8-timing-$pos.log
  echo "== kv8 pos $pos"; grep -a "^\[timing\]" ~/mimo/m1/timing-r0.log | grep "drafter off"; grep -a "error\|panic" ~/mimo/m1/timing-r0.log | head -3
done
./start.sh 2>&1 | tail -1
