#!/usr/bin/env bash
# Z7 long context: per-step cost (1/4/8 rows) and the eager breakdown at position 9,000 vs 128,000 (Zig), Python's
# window costs at 128,000 (random tokens); Python restarts.
set -u
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $HOME/mimo/logs/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
for pos in 9000 128000; do
  BIN=/mimo/zig-out-z7c/bin AOT=aot5-sm121 MODE=timing COMM=roce EXTRA_ENV="-e TF_TP_ROCE_FAST=1 -e TF_MIMO_TIMING_CONTEXT=135168 -e TF_MIMO_TIMING_POS=$pos" timeout 900 bash ~/mimo/m1/m1_run.sh >/dev/null 2>&1
  cp ~/mimo/m1/timing-r0.log $HOME/mimo/logs/z7-timing-$pos.log
  echo "== zig pos $pos"; grep -a "^\[timing\]" ~/mimo/m1/timing-r0.log
done
OUT=$HOME/mimo/oracle SCRIPT=mimo_profile.py ORACLE_ARG=1 EXTRA="--tp 2 --widths 1,4,8 --steps 30 --random --prompt 128000" timeout 1500 bash ~/mimo/recipe/tools/mimo_zig_capture.sh >/dev/null 2>&1
cp ~/mimo/oracle/capture-r0.log $HOME/mimo/logs/py-widths-128k.log
echo "== python pos 128000"; grep -a "window_ms" ~/mimo/oracle/capture-r0.log | tail -1; grep -aiE "error|Traceback" ~/mimo/oracle/capture-r0.log | tail -3
./start.sh 2>&1 | tail -1
