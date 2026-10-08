#!/usr/bin/env python3
"""Decode speed at long context: one stream, a prompt of about N tokens (repeated prose), 256 new tokens, t=0.

  python3 scripts/long_decode.py http://127.0.0.1:8891 MiMo-V2.6-Flash 8000 32000 64000
Per length: decode tok/s = completion tokens / the server's own decode seconds (``tensorfold.decode_s``), median of 3;
prefill tok/s likewise from ``tensorfold.prefill_s``. Each repetition changes the first line so no prefix is reused.
"""

import json
import statistics
import sys
import time
import urllib.request

url, model = sys.argv[1].rstrip("/"), sys.argv[2]
PARA = ("The committee reviewed the quarterly figures line by line, noting where the regional offices had exceeded "
        "their targets and where supply delays had pushed revenue into the following period. ")


def one(n_tokens: int, rep: int) -> dict:
    text = f"Run {rep} at {time.time()}.\n" + PARA * max(1, n_tokens // 34)
    body = {"model": model, "messages": [{"role": "user", "content": text + "\nSummarise the above in detail."}],
            "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    tf, u = j.get("tensorfold") or {}, j["usage"]
    return {"prompt": u["prompt_tokens"], "out": u["completion_tokens"],
            "decode_tps": u["completion_tokens"] / tf["decode_s"] if tf.get("decode_s") else 0.0,
            "prefill_tps": u["prompt_tokens"] / tf["prefill_s"] if tf.get("prefill_s") else 0.0}


for n in map(int, sys.argv[3:]):
    runs = [one(n, i) for i in range(3)]
    print(json.dumps({"context": runs[0]["prompt"], "completion": runs[0]["out"],
                      "decode_tps_median": round(statistics.median(r["decode_tps"] for r in runs), 1),
                      "prefill_tps_median": round(statistics.median(r["prefill_tps"] for r in runs), 1)}), flush=True)
