#!/usr/bin/env bash
# kv8c: kernel arms (kv8check, rank 1's log), the step at 128,000, then the gates: kv8 (passkey, full bench, GSM8K 500)
# and the fp16 cache's GSM8K 500 on the same build.
set -u
T=$HOME/mimo/recipe/tools
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $HOME/mimo/logs/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
B=/mimo/zig-out-kv8c/bin
BIN=$B AOT=aot5-sm121 MODE=kv8check COMM=roce EXTRA_ENV="-e TF_TP_ROCE_FAST=1" timeout 900 bash ~/mimo/m1/m1_run.sh >/dev/null 2>&1
ssh aj@10.100.128.2 cat mimo/m1/kv8check-r1.log | grep -a "kv8check\]" | tee $HOME/mimo/logs/kv8check-c.log
BIN=$B AOT=aot5-sm121 MODE=timing COMM=roce EXTRA_ENV="-e TF_TP_ROCE_FAST=1 -e TF_MIMO_KV8=1 -e TF_MIMO_TIMING_CONTEXT=135168 -e TF_MIMO_TIMING_POS=128000" timeout 900 bash ~/mimo/m1/m1_run.sh >/dev/null 2>&1
echo "== kv8c pos 128000"; grep -a "^\[timing\]" ~/mimo/m1/timing-r0.log | grep "drafter off"
BIN=/mimo/zig-out-kv8c BENCH_EXTRA="--gsm8k-n 500" bash $T/kv8_gate.sh kv8c "-e TF_MIMO_KV8=1"
BIN=/mimo/zig-out-kv8c NIAH=0 SUITES=quality BENCH_EXTRA="--gsm8k-n 500" bash $T/kv8_gate.sh fp16c ""
