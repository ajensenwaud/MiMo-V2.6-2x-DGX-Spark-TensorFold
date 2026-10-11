#!/usr/bin/env python3
"""Z5 qualification checks against a server (Jay Leaton's list, MiMo edition).

  sampled  --out F       record sampled replies (seeded and seedless, top_k on and off, chat and completions)
  sampled  --against F   the same requests; replies must equal the recorded ones
  cancel                 a streamed reply dropped after 16 tokens frees its stream: /health back to 0 running, and a
                         greedy canary afterwards still equals its solo reply
  long --out/--against   a 6K-token prompt, greedy 64 tokens (prompt seams across 128-row windows)

  python3 tools/mimo_zig_qual.py http://localhost:8892 sampled --against results/zig-z5/sampled-python.json
"""

import argparse
import http.client
import json
import sys
import time
import urllib.parse
import urllib.request

SAMPLED = {
    "k20_seed": {"messages": [{"role": "user", "content": "Write a short poem about the sea."}], "temperature": 0.7, "top_k": 20, "seed": 1},
    "k40_topp_seed": {"messages": [{"role": "user", "content": "Name five unusual fruits and describe one."}], "temperature": 1.0, "top_k": 40, "top_p": 0.9, "seed": 2},
    "k0_nucleus_seed": {"messages": [{"role": "user", "content": "Tell me a fact about octopuses."}], "temperature": 0.8, "top_k": 0, "top_p": 0.95, "seed": 3},
    "k20_noseed": {"messages": [{"role": "user", "content": "Suggest a name for a cafe."}], "temperature": 0.6, "top_k": 20},
    "minp_seed": {"messages": [{"role": "user", "content": "Describe Sydney in two sentences."}], "temperature": 0.9, "top_k": 50, "min_p": 0.05, "seed": 4},
    "completion_seed": {"prompt": "Once upon a time in a small village,", "temperature": 0.7, "top_k": 20, "seed": 5},
}
for v in SAMPLED.values():
    v.setdefault("max_tokens", 128)
    v.setdefault("chat_template_kwargs", {"enable_thinking": False})


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def text_of(r):
    c = r["choices"][0]
    return c["message"]["content"] if "message" in c else c["text"]


def run_sampled(base):
    out = {}
    for k, body in SAMPLED.items():
        b = dict(body, model="x")
        path = "/v1/completions" if "prompt" in b else "/v1/chat/completions"
        if "prompt" in b:
            b.pop("chat_template_kwargs")
        out[k] = text_of(post(base, path, b))
    return out


def health(base):
    with urllib.request.urlopen(base + "/health", timeout=10) as r:
        h = json.loads(r.read())
    if "requests_running" not in h and "live" in h:          # the Zig server's /health
        h["requests_running"] = h["live"].get("connections", 0) + h["live"].get("waiting", 0)
    return h


def cancel(base):
    canary = {"model": "x", "messages": [{"role": "user", "content": "Count from 1 to 30 separated by commas."}],
              "max_tokens": 120, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    solo = text_of(post(base, "/v1/chat/completions", canary))
    u = urllib.parse.urlparse(base)
    for _ in range(3):
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=120)
        body = {"model": "x", "messages": [{"role": "user", "content": "Write a long essay about rivers."}],
                "max_tokens": 2000, "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = conn.getresponse()
        chunks = 0
        while chunks < 16:
            line = r.fp.readline()
            if line.startswith(b"data: {"):
                chunks += 1
        conn.close()
    t0 = time.time()
    while time.time() - t0 < 60:
        if health(base).get("requests_running", 1) == 0:
            break
        time.sleep(0.5)
    freed = health(base).get("requests_running")
    after = text_of(post(base, "/v1/chat/completions", canary))
    ok = freed == 0 and after == solo
    print(f"{'PASS' if ok else 'FAIL'} cancel: 3 streams dropped after 16 chunks; running afterwards {freed}; "
          f"canary {'equal' if after == solo else 'DIFFERENT'} to its solo reply")
    return ok


def long_prompt(base):
    words = " ".join(f"item{i} is followed by item{i + 1}." for i in range(1200))
    body = {"model": "x", "messages": [{"role": "user", "content": words + "\nWhat follows item731? Answer briefly."}],
            "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = post(base, "/v1/chat/completions", body)
    return {"text": text_of(r), "prompt_tokens": r["usage"]["prompt_tokens"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("what", choices=["sampled", "cancel", "long"])
    ap.add_argument("--out", default="")
    ap.add_argument("--against", default="")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    if a.what == "cancel":
        sys.exit(0 if cancel(base) else 1)
    got = run_sampled(base) if a.what == "sampled" else long_prompt(base)
    if a.out:
        json.dump(got, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"recorded {a.what} -> {a.out}")
    if a.against:
        want = json.load(open(a.against))
        if a.what == "sampled":
            bad = [k for k in want if want[k] != got.get(k)]
            for k in bad:
                w, g = want[k], got.get(k, "")
                i = next((j for j in range(min(len(w), len(g))) if w[j] != g[j]), min(len(w), len(g)))
                print(f"  DIFF {k} at char {i}: got {g[i:i + 50]!r} want {w[i:i + 50]!r}")
            print(f"{'PASS' if not bad else 'FAIL'} sampled: {len(want) - len(bad)}/{len(want)} replies equal")
            sys.exit(1 if bad else 0)
        same = want["text"] == got["text"]
        print(f"long ({got['prompt_tokens']} prompt tokens): reply {'equal to' if same else 'differs from'} the reference"
              f"\n  got  {got['text'][:120]!r}\n  want {want['text'][:120]!r}")


if __name__ == "__main__":
    main()
