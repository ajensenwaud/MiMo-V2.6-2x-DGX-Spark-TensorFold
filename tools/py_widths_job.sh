#!/usr/bin/env bash
# Python verify-window cost curve on the 3.0 pack at position 9,000: prompt text, then pseudo-random tokens; restart.
set -u
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])"); [ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > ~/mimo/logs/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd ~/mimo/recipe && ./stop.sh >/dev/null 2>&1
for r in "" "--random"; do
  OUT=$HOME/mimo/oracle SCRIPT=mimo_profile.py ORACLE_ARG=1 EXTRA="--tp 2 --widths 1,2,3,4,5,6,8 --steps 30 $r" bash ~/mimo/recipe/tools/mimo_zig_capture.sh >/dev/null 2>&1
  echo "== text${r}"; grep -a "window_ms" ~/mimo/oracle/capture-r0.log | tail -1
done
./start.sh 2>&1 | tail -1
