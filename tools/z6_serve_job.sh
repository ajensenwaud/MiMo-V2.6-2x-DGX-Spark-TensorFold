#!/usr/bin/env bash
# Z6 serving check: the Zig server on the prompt path; startup facts, the replies gate, then the fixed-text decode set
# (shas compare with Python's ~/mimo/logs/fixed-python-done.txt), then the Python serve restarts.
set -u
T=$HOME/mimo/recipe/tools
L=$HOME/mimo/logs
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $L/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
CONTEXT=${CONTEXT:-608256} BIN=/mimo/zig-out-z6 AOT=/mimo/oracle/aot5-sm121 EXTRA_ENV="-e TF_TP_ROCE_FAST=1 ${XENV:-}" bash $T/zig_serve.sh > $L/z6-serve-start.log 2>&1
echo "serve rc $?"
docker logs mimo-zig 2>&1 | grep -E "prompt path|mimo costs|CUDA sm_121|error|Error" | head
docker run --rm --network host --user 1000:1000 -v $HOME/mimo:/mimo --entrypoint python3 mimo-tf:latest \
  /mimo/recipe/tools/mimo_zig_replies.py http://localhost:8892 /mimo/oracle/rank0.safetensors 2>&1 | tail -2
(cd /tmp && /usr/bin/python3 $T/mimo_fixed_decode.py http://localhost:8892 --reps 3) > $L/fixed-z6.txt 2>&1
docker logs mimo-zig > $L/fixed-z6-serve.log 2>&1
grep "done " $L/fixed-z6-serve.log | grep "tokens=512" | tail -12 | \
  sed "s/.*prompt=\([0-9]*\).*sha=\([0-9a-f]*\).*rounds=\([0-9]*\) accepted=\([0-9\/]*\)/\1 \2 \3 \4/" > $L/fixed-z6-done.txt
bash $T/zig_serve.sh stop >/dev/null 2>&1
paste $L/fixed-z6.txt $L/fixed-z6-done.txt
echo "== sha equal to Python:"; paste <(awk '{print $2}' $L/fixed-python-done.txt) <(awk '{print $2}' $L/fixed-z6-done.txt) | awk '{n++; if ($1==$2) e++} END {print e"/"n}'
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
