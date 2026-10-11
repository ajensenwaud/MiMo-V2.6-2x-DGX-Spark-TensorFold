#!/usr/bin/env python3
"""Z5 replies: the answer key's 8 prompts (tools/mimo_zig_capture.py, as token ids) through a server's /v1/completions,
greedy, as many tokens as the key's reply; every reply's text must equal the key's greedy reply, detokenized by the
same server. Serially, then all 8 at once (the server takes 4 streams at a time; shared windows, drafting).

  python3 tools/mimo_zig_replies.py http://localhost:8892 /oracle/rank0.safetensors
"""

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from safetensors import safe_open

PROMPTS = ["en_story", "zh_explain", "code_py", "math", "tool_json", "edit", "list", "short"]
EOS = {151645, 151643}


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    base, oracle = sys.argv[1].rstrip("/"), sys.argv[2]
    o = safe_open(oracle, framework="pt")
    cases = {}
    for p in PROMPTS:
        ids = o.get_tensor(f"{p}.ids").tolist()
        greedy = o.get_tensor(f"{p}.greedy").tolist()
        body = [t for t in greedy if t not in EOS]
        want = post(base, "/detokenize", {"tokens": body})["prompt"] if body else ""
        cases[p] = (ids, len(greedy), want, greedy[-1] in EOS)

    def one(p):
        ids, n, want, eos = cases[p]
        t0 = time.perf_counter()
        r = post(base, "/v1/completions", {"model": "x", "prompt": ids, "max_tokens": n, "temperature": 0})
        text = r["choices"][0]["text"]
        return p, text == want, text, want, r["usage"]["completion_tokens"], time.perf_counter() - t0

    ok_all = True
    for mode in ("serial", "parallel"):
        if mode == "serial":
            res = [one(p) for p in PROMPTS]
        else:
            with ThreadPoolExecutor(8) as ex:
                res = list(ex.map(one, PROMPTS))
        good = sum(r[1] for r in res)
        ok_all &= good == len(res)
        for p, same, text, want, toks, secs in res:
            if not same:
                i = next((j for j in range(min(len(text), len(want))) if text[j] != want[j]), min(len(text), len(want)))
                print(f"  DIFF {mode} {p}: at char {i}: got {text[i:i + 60]!r} want {want[i:i + 60]!r}")
        toks = sum(r[4] for r in res)
        print(f"{mode}: {good}/{len(res)} replies equal to the answer key; {toks} tokens; "
              f"{', '.join(f'{r[0]} {r[4] / r[5]:.0f} tok/s' for r in res)}")
    print("PASS replies" if ok_all else "FAIL replies")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
