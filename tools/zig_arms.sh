#!/usr/bin/env bash
# Z7 arms, one change each, on gx10 with the Python serve stopped: start the Zig server, gate (answer key replies),
# spark-bench decode (full preset's decode cells), stop. Logs and summaries under ~/mimo/logs/z7-<arm>.*
set -u
T=$HOME/mimo/recipe/tools
arm() {   # NAME, then env for zig_serve.sh
  local name=$1; shift
  echo "=== $name ($*)"
  env "$@" CONTEXT=608256 bash $T/zig_serve.sh || { echo "$name: no serve"; bash $T/zig_serve.sh stop; return; }
  docker logs mimo-zig 2>&1 | grep "mimo costs"
  docker run --rm --network host --user 1000:1000 -v $HOME/mimo:/mimo --entrypoint python3 mimo-tf:latest \
    /mimo/recipe/tools/mimo_zig_replies.py http://localhost:8892 /mimo/oracle/rank0.safetensors 2>&1 | tail -1
  (cd /tmp/spark-bench && /usr/bin/python3 scripts/bench.py --base-url http://localhost:8892 --preset full --suites decode \
     --label "Zig Z7 arm $name" 2>&1 | grep -E "decode|done in" | tee $HOME/mimo/logs/z7-$name.log)
  cp "$(ls -td /tmp/spark-bench/results/*/ | head -1)summary.json" $HOME/mimo/logs/z7-$name.json 2>/dev/null
  docker logs mimo-zig > $HOME/mimo/logs/z7-$name-serve.log 2>&1
  bash $T/zig_serve.sh stop
}
arm device-argmax BIN=/mimo/zig-out-z7 COMM=nccl
arm device-argmax-roce BIN=/mimo/zig-out-z7 COMM=roce
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
echo "=== done"
