#!/usr/bin/env bash
# Python engine kernel profile on the 3.0 pack (tools/mimo_profile.py, torch.profiler, both ranks), then restart the serve.
set -u
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])"); [ "$h" = 0 ] || { echo busy; exit 1; }
cd ~/mimo/recipe && ./stop.sh >/dev/null 2>&1
OUT=$HOME/mimo/oracle SCRIPT=mimo_profile.py ORACLE_ARG=1 EXTRA="--json /out/profile-3.0-r0.json --steps 20" bash ~/mimo/recipe/tools/mimo_zig_capture.sh
grep -v "^\[1/1\]" ~/mimo/oracle/capture-r0.log | tail -40
./start.sh 2>&1 | tail -1
