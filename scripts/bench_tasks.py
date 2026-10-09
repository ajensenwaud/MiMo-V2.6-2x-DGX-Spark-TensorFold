#!/usr/bin/env python3
"""Decode speed by task through the server: coding, prose, reasoning, code edit (thinking off, t=0, 512 tokens).

  python3 scripts/bench_tasks.py http://127.0.0.1:8891 MiMo-V2.6-Flash [--streams 1] [--reps 3] [--save out.json]
                                 [--compare earlier.json]

Per task: decode tok/s = completion tokens / the server's own decode seconds (``tensorfold.decode_s``), median of
``--reps``; with ``--streams N`` each rep sends N copies at once and reports the per-stream median and the aggregate.
``--save`` keeps each reply's text; ``--compare`` checks the replies are byte-identical to an earlier run's (greedy
decoding with and without the drafter must agree).
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/tools")
from mimo_prompts import PROMPTS  # noqa: E402  (the same prompts as the engine-level check)


def call(url: str, model: str, text: str, tokens: int, draft: bool = True) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if not draft:
        body["draft"] = False                              # the server's serial reference: no drafting at all
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    wall = time.perf_counter() - t
    tf, u = j.get("tensorfold") or {}, j["usage"]
    text = j["choices"][0]["message"].get("content") or ""
    return {"out": u["completion_tokens"], "decode_s": tf.get("decode_s"), "wall_s": wall, "text": text,
            "tps": u["completion_tokens"] / tf["decode_s"] if tf.get("decode_s") else 0.0, "tf": tf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--no-draft", action="store_true", help="send draft: false (plain decode, the baseline)")
    ap.add_argument("--mixed", action="store_true", help="one stream a task, the four at once (realistic batching)")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    earlier = json.load(open(a.compare))["texts"] if a.compare else {}
    out = {"streams": a.streams, "tasks": {}, "texts": {}}
    call(url, a.model, "Say hello.", 8)                    # warm
    if a.mixed:                                            # every stream a different task, all at once
        names = list(PROMPTS)
        agg, per, texts = [], {n: [] for n in names}, {}
        for _ in range(a.reps):
            t = time.perf_counter()
            with cf.ThreadPoolExecutor(len(names)) as ex:
                rs = list(ex.map(lambda n: call(url, a.model, PROMPTS[n], a.tokens, not a.no_draft), names))
            span = time.perf_counter() - t
            agg.append(sum(r["out"] for r in rs) / span)
            for n, r in zip(names, rs):
                per[n].append(r["tps"])
                texts[n] = r["text"]
        row = {"aggregate_tps_median": round(statistics.median(agg), 1),
               "per_stream_tps_median": {n: round(statistics.median(v), 1) for n, v in per.items()}}
        if earlier:
            row["matches_compare"] = {n: texts[n] == earlier.get(n) for n in names}
        out["mixed"], out["texts"] = row, texts
        print(json.dumps({"mixed": row}), flush=True)
        if a.save:
            json.dump(out, open(a.save, "w"), indent=1)
        return
    for name, text in PROMPTS.items():
        per, agg, texts = [], [], []
        for _ in range(a.reps):
            t = time.perf_counter()
            with cf.ThreadPoolExecutor(a.streams) as ex:
                rs = list(ex.map(lambda _: call(url, a.model, text, a.tokens, not a.no_draft), range(a.streams)))
            span = time.perf_counter() - t
            per += [r["tps"] for r in rs]
            agg.append(sum(r["out"] for r in rs) / span)
            texts += [r["text"] for r in rs]
        same = len({hashlib.sha256(x.encode()).hexdigest() for x in texts}) == 1
        out["texts"][name] = texts[0]
        row = {"decode_tps_median": round(statistics.median(per), 1), "aggregate_tps_median": round(statistics.median(agg), 1),
               "completion_tokens": rs[0]["out"], "replies_identical": same}
        if name in earlier:
            row["matches_compare"] = texts[0] == earlier[name]
        if rs[0]["tf"]:
            row["server"] = {k: v for k, v in rs[0]["tf"].items() if k not in ("decode_s", "prefill_s")}
        out["tasks"][name] = row
        print(json.dumps({name: row}), flush=True)
    if a.save:
        json.dump(out, open(a.save, "w"), indent=1)


if __name__ == "__main__":
    main()
