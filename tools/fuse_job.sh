#!/usr/bin/env bash
# Z7 fusions: step costs fused vs not (9,000), then the fused build served: decode + quality; Python restarts.
set -u
T=$HOME/mimo/recipe/tools
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $HOME/mimo/logs/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
for arm in fused:1 plain:0; do
  a=${arm%%:*}; v=${arm##*:}
  BIN=/mimo/zig-out-z7f/bin AOT=aot5-sm121 MODE=timing COMM=roce EXTRA_ENV="-e TF_TP_ROCE_FAST=1 -e TF_MIMO_FUSE=$v" timeout 900 bash ~/mimo/m1/m1_run.sh >/dev/null 2>&1
  cp ~/mimo/m1/timing-r0.log $HOME/mimo/logs/fuse-timing-$a.log
  echo "== $a"; grep -a "^\[timing\]" ~/mimo/m1/timing-r0.log | grep "drafter off"; grep -a "error\|panic" ~/mimo/m1/timing-r0.log | head -3
done
BIN=/mimo/zig-out-z7f NIAH=0 SUITES=decode,quality bash $T/kv8_gate.sh fused ""
