#!/usr/bin/env bash
# Start-to-start variance of the Zig server (Z6, Python's depth rule): N starts per arm, the fixed-text decode set each,
# with GPU clocks / power / throttle reasons sampled on both nodes every 500 ms during the decode set.
# Usage: zig_starts.sh N ARM "EXTRA_ENV" [ARM "EXTRA_ENV" ...]   (arms alternate start by start)
set -u
T=$HOME/mimo/recipe/tools
L=$HOME/mimo/logs
N=$1; shift
h=$(curl -s -m 5 localhost:8891/health | python3 -c "import json,sys;print(json.load(sys.stdin)['requests_running'])" 2>/dev/null || echo 0)
[ "$h" = 0 ] || { echo busy; exit 1; }
docker logs mimo-tf > $L/mimo-tf-$(date +%Y%m%d-%H%M%S).log 2>&1
cd $HOME/mimo/recipe && ./stop.sh >/dev/null 2>&1
Q="--query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv,noheader -lms 500"
for i in $(seq 1 $N); do
  set -- "${@}"
  args=("$@")
  for ((j = 0; j < ${#args[@]}; j += 2)); do
    a="${args[j]}-$i"; x="${args[j+1]}"
    CONTEXT=608256 BIN=/mimo/zig-out-z6 AOT=/mimo/oracle/aot5-sm121 EXTRA_ENV="$x" bash $T/zig_serve.sh >/dev/null 2>&1 || { echo "$a: serve failed"; continue; }
    nvidia-smi $Q > $L/clk-$a-r0.csv 2>&1 & c0=$!
    ssh -n aj@10.100.128.2 "timeout 300 nvidia-smi $Q" > $L/clk-$a-r1.csv 2>&1 </dev/null & c1=$!
    (cd /tmp && /usr/bin/python3 $T/mimo_fixed_decode.py http://localhost:8892 --reps 3) > $L/fixed-$a.txt 2>&1
    kill $c0 $c1 2>/dev/null
    docker logs mimo-zig > $L/fixed-$a-serve.log 2>&1
    bash $T/zig_serve.sh stop >/dev/null 2>&1
    echo "== $a: $(awk '{s[$1]+=$4; n[$1]++} END {for (k in s) printf "%s %.1f  ", k, s[k]/n[k]}' $L/fixed-$a.txt)"
    for r in r0 r1; do
      awk -F', ' -v r=$r '{gsub(/ MHz/,"",$2); gsub(/ W/,"",$3); c+=$2; p+=$3; t=$4; n++; if ($5 != "0x0000000000000000") th++} END {printf "   %s: sm %.0f MHz, %.1f W, %s C, throttled %d/%d samples\n", r, c/n, p/n, t, th, n}' $L/clk-$a-$r.csv
    done
    grep -h "mimo costs" $L/fixed-$a-serve.log | sed 's/.*window ms /   costs: /'
  done
done
cd $HOME/mimo/recipe && ./start.sh 2>&1 | tail -1
