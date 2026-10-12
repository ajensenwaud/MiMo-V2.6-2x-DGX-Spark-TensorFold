#!/usr/bin/env bash
# Z7 pgemm: prompt answer key (closeness, greedy), prompt chunk costs per arm; Python restarts.
set -u
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $HOME/mimo/logs/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
B=/mimo/zig-out-pg/bin
ORACLE=/mimo/oracle/prompt/prompt-r BIN=$B AOT=aot5-sm121 MODE=prompt COMM=roce EXTRA_ENV="-e TF_TP_ROCE_FAST=1 -e TF_MIMO_KV8=0" timeout 1200 bash ~/mimo/m1/m1_run.sh >/dev/null 2>&1
cp ~/mimo/m1/prompt-r0.log $HOME/mimo/logs/pg-prompt-r0.log
echo "== prompt gate (pgemm, fp16 cache)"; grep -a "greedy\|PASS\|FAIL\|error\|panic" ~/mimo/m1/prompt-r0.log | head -24
for arm in "base:-e TF_MIMO_PGEMM=0" "pg:" "pg256:-e TF_MIMO_PROMPT_EXPERTS_FROM=256" "pg0:-e TF_MIMO_PROMPT_EXPERTS_FROM=1"; do
  a=${arm%%:*}; x=${arm#*:}
  BIN=$B AOT=aot5-sm121 MODE=ptiming COMM=roce EXTRA_ENV="-e TF_TP_ROCE_FAST=1 $x" timeout 900 bash ~/mimo/m1/m1_run.sh >/dev/null 2>&1
  cp ~/mimo/m1/ptiming-r0.log $HOME/mimo/logs/pg-ptime-$a.log
  grep -a "^\[ptiming\]" ~/mimo/m1/ptiming-r0.log | sed "s/^/$a /"; grep -a "error\|panic" ~/mimo/m1/ptiming-r0.log | head -3
done
./start.sh 2>&1 | tail -1
