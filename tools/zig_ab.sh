#!/usr/bin/env bash
# Same-night A/B with repeats (spark-bench decode cells, 512 tokens): 3 runs at 1 stream + 1 at 4 streams for each
# Zig arm, then the Python serve (left running). Results: ~/mimo/logs/ab-<arm>.log
set -u
T=$HOME/mimo/recipe/tools
bench() {   # ARM URL
  for c in 1 1 1 4; do
    (cd /tmp/spark-bench && /usr/bin/python3 scripts/bench.py --base-url "$2" --suites decode --concurrency $c --decode-tokens 512 \
       --label "AB $1 c=$c" 2>&1 | grep -E "decode (chat|code|agent|report)") >> $HOME/mimo/logs/ab-$1.log
  done
}
zig_arm() {   # ARM, then env for zig_serve.sh
  local a=$1; shift
  : > $HOME/mimo/logs/ab-$a.log
  env "$@" CONTEXT=608256 BIN=/mimo/zig-out-z7 bash $T/zig_serve.sh >> $HOME/mimo/logs/ab-$a.log 2>&1 || { bash $T/zig_serve.sh stop; return; }
  docker run --rm --network host --user 1000:1000 -v $HOME/mimo:/mimo --entrypoint python3 mimo-tf:latest \
    /mimo/recipe/tools/mimo_zig_replies.py http://localhost:8892 /mimo/oracle/rank0.safetensors 2>&1 | tail -1 >> $HOME/mimo/logs/ab-$a.log
  bench "$a" http://localhost:8892
  docker logs mimo-zig > $HOME/mimo/logs/ab-$a-serve.log 2>&1
  bash $T/zig_serve.sh stop
}
docker rm -f mimo-zig >/dev/null 2>&1; ssh -n aj@10.100.128.2 "docker rm -f mimo-zig-follow" >/dev/null 2>&1 </dev/null
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
zig_arm geometric EXTRA_ENV="-e TF_TP_ROCE_FAST=1 -e TF_MIMO_DEPTH=geometric"
zig_arm lanes EXTRA_ENV="-e TF_TP_ROCE_FAST=1"
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
for i in $(seq 1 60); do curl -s -m 2 localhost:8891/health | grep -q '"ok": true' && break; sleep 5; done
: > $HOME/mimo/logs/ab-python.log
bench python http://localhost:8891
echo "=== done"
