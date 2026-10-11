#!/usr/bin/env bash
# A served chat request under nsys on rank 0 (Z6 build, Python's depth rule): where a narrow verify's time goes.
set -u
T=$HOME/mimo/recipe/tools
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $HOME/mimo/logs/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
t0=$(date +%s)
NSYS=serve1 NSYS_DELAY=300 NSYS_DUR=45 CONTEXT=608256 BIN=/mimo/zig-out-z6 AOT=/mimo/oracle/aot5-sm121 EXTRA_ENV="-e TF_TP_ROCE_FAST=1 -e TF_MIMO_DEPTH=python -e TF_MIMO_PROFILE=1" bash $T/zig_serve.sh
up=$(( $(date +%s) - t0 )); echo "up after ${up}s"
while [ $(( $(date +%s) - t0 )) -lt 305 ]; do sleep 1; done
(cd /tmp && timeout 40 /usr/bin/python3 $T/mimo_fixed_decode.py http://localhost:8892 --reps 3 2>&1 | head -3)
sleep 15
bash $T/zig_serve.sh stop >/dev/null 2>&1
ls -la $HOME/mimo/nsys/serve1* 2>&1
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
