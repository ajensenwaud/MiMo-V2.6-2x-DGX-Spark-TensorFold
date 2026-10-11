#!/usr/bin/env bash
# Z7 window 1: prompt chunk costs under nsys (ptiming), then the z7b server: 4-stream cells and the fixed-text set.
set -u
T=$HOME/mimo/recipe/tools
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $HOME/mimo/logs/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
NSYS=ptime BIN=/mimo/zig-out-z7b/bin AOT=aot5-sm121 MODE=ptiming COMM=roce EXTRA_ENV="-e TF_TP_ROCE_FAST=1" timeout 900 bash ~/mimo/m1/m1_run.sh
grep -a "ptiming" ~/mimo/m1/ptiming-r0.log
BIN=/mimo/zig-out-z7b bash $T/zig_c4.sh z7b "-e TF_TP_ROCE_FAST=1" 2>&1 | grep -v start.sh
BIN=/mimo/zig-out-z7b AOT=/mimo/oracle/aot5-sm121 bash $T/zig_fixed.sh z7b "-e TF_TP_ROCE_FAST=1" | grep -v "^chat\|^code\|^agent\|^report"
echo "z7b $(awk '{s[$1]+=$4; n[$1]++} END {for (k in s) printf "%s %.1f  ", k, s[k]/n[k]}' $HOME/mimo/logs/fixed-z7b.txt)"
