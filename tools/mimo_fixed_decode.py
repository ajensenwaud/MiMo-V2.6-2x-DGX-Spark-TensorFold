#!/usr/bin/env python3
"""spark-bench's four decode workloads with fixed tags (no random cache-bust), greedy, 512 tokens, one stream at a
time: the same texts on every engine, so each engine's log lines (sha, rounds, accepted/drafted) compare request for
request. Prints tok/s a request.

  python3 mimo_fixed_decode.py BASE_URL [--bench /tmp/spark-bench/scripts] [--reps 2]
"""
import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--bench", default="/tmp/spark-bench/scripts")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=512)
    a = ap.parse_args()
    sys.path.insert(0, a.bench)
    import bench  # spark-bench

    import json
    import urllib.request

    model = json.load(urllib.request.urlopen(bench.root_url(a.base) + "/v1/models"))["data"][0]["id"]
    ep = bench.Endpoint(a.base, model, 600)
    for wl, prompts in bench.WORKLOADS.items():
        ep.stream(prompts[0][:200], 16)
        for i in range(a.reps):
            s = ep.stream(f"[run {i:08x}] " + prompts[0], a.tokens, force_length=True)
            print(f"{wl:<6} rep {i}: {s['decode_tps']:.2f} tok/s, {s['completion_tokens']} tokens", flush=True)


if __name__ == "__main__":
    main()
