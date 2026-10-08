#!/usr/bin/env python3
"""Prefill (prompt processing) rate, one stream or several at once: cold prompts of about L tokens, 1 new token each.

  python3 scripts/concurrent_prefill.py http://127.0.0.1:8891 MiMo-V2.6-Flash --streams 1 4 --lengths 8000 32000
Per (streams, length): aggregate tok/s = all prompt tokens / wall time from the first send to the last reply, and the
median time to first token; median of 2 repetitions. Every prompt starts with a unique line, so nothing is cached.
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PARA = ("The committee reviewed the quarterly figures line by line, noting where the regional offices had exceeded "
        "their targets and where supply delays had pushed revenue into the following period. ")


def one(url, model, n_tokens, tag):
    text = f"Request {tag} {time.time_ns()}.\n" + PARA * max(1, n_tokens // 34) + "\nReply with one word."
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        j = json.load(r)
    return j["usage"]["prompt_tokens"], time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("model")
    p.add_argument("--streams", type=int, nargs="+", default=[1, 4])
    p.add_argument("--lengths", type=int, nargs="+", default=[8000, 32000])
    p.add_argument("--reps", type=int, default=2)
    a = p.parse_args()
    url = a.url.rstrip("/")
    for n in a.streams:
        for length in a.lengths:
            agg, ttft = [], []
            for rep in range(a.reps):
                t0 = time.perf_counter()
                with ThreadPoolExecutor(n) as ex:
                    res = list(ex.map(lambda i: one(url, a.model, length, f"{rep}-{i}"), range(n)))
                wall = time.perf_counter() - t0
                agg.append(sum(t for t, _ in res) / wall)
                ttft += [s for _, s in res]
            print(json.dumps({"streams": n, "prompt_tokens": res[0][0], "aggregate_prefill_tps": round(
                statistics.median(agg)), "ttft_median_s": round(statistics.median(ttft), 2)}), flush=True)


if __name__ == "__main__":
    main()
