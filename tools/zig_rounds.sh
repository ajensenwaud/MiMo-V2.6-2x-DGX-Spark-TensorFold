#!/usr/bin/env bash
# Round-phase comparison at 1 stream: Zig (TF_MIMO_PROFILE=1) then Python (TF_MIMO_PROFILE_ROUNDS=1), spark-bench
# decode cells at concurrency 1, twice each; then the Python serve restarts clean. Logs: ~/mimo/logs/rounds-*.log
set -u
T=$HOME/mimo/recipe/tools
L=$HOME/mimo/logs
bench() {
  for i in 1 2; do
    (cd /tmp/spark-bench && /usr/bin/python3 scripts/bench.py --base-url "$1" --suites decode --concurrency 1 \
       --decode-tokens 512 --label "rounds" 2>&1 | grep -E "decode (chat|code|agent|report)")
  done
}
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
docker rm -f mimo-zig >/dev/null 2>&1; ssh -n aj@10.100.128.2 "docker rm -f mimo-zig-follow" >/dev/null 2>&1 </dev/null
echo "== zig"
CONTEXT=608256 BIN=/mimo/zig-out-z7 EXTRA_ENV="-e TF_TP_ROCE_FAST=1 -e TF_MIMO_PROFILE=1" bash $T/zig_serve.sh >/dev/null 2>&1 \
  && bench http://localhost:8892
docker logs mimo-zig > $L/rounds-zig.log 2>&1
bash $T/zig_serve.sh stop >/dev/null 2>&1
echo "== python"
cd $HOME/mimo/recipe && TF_MIMO_PROFILE_ROUNDS=1 ./start.sh 2>&1 | tail -1
bench http://localhost:8891
docker logs mimo-tf > $L/rounds-python.log 2>&1
./start.sh restart 2>&1 | tail -1
echo "=== done"
