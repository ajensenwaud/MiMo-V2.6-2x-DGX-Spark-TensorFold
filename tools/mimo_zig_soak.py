#!/usr/bin/env python3
"""Z5 soak: 4 clients for DURATION seconds against a server: greedy and sampled chat, streamed replies (one in five
dropped early), completions, 2-6K-token prompts; a greedy canary every 5 minutes must equal its first reply. Then,
with --kill-follower CMD, the follower is killed and the server must fail (an error or a dropped connection) within
FAILFAST seconds instead of hanging. Writes a JSON summary to --out.

  python3 tools/mimo_zig_soak.py http://localhost:8892 --duration 3600 --out soak.json \
      --kill-follower "ssh -n aj@10.100.128.2 docker kill mimo-zig-follow"
"""

import argparse
import http.client
import json
import random
import subprocess
import threading
import time
import urllib.parse
import urllib.request

TOPICS = ["rivers", "compilers", "the Roman empire", "graph theory", "coffee", "volcanoes", "jazz", "sorting algorithms",
          "Sydney", "photosynthesis", "chess openings", "bridges"]


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def text_of(r):
    c = r["choices"][0]
    return c["message"]["content"] if "message" in c else c["text"]


def stream(base, body, drop_after):
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    conn.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True)), {"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        raise RuntimeError(f"stream status {r.status}")
    chunks = 0
    while True:
        line = r.fp.readline()
        if not line:
            break
        if line.startswith(b"data: [DONE]"):
            break
        if line.startswith(b"data: {"):
            chunks += 1
            if drop_after and chunks >= drop_after:
                break
    conn.close()
    return chunks


def request(base, rng):
    kind = rng.choice(["greedy", "sampled", "stream", "completion", "long"])
    topic = rng.choice(TOPICS)
    msgs = [{"role": "user", "content": f"Explain {topic} in about {rng.choice([50, 120, 250])} words."}]
    nothink = {"enable_thinking": rng.random() < 0.5}
    if kind == "greedy":
        return kind, text_of(post(base, "/v1/chat/completions", {"model": "x", "messages": msgs, "max_tokens": rng.choice([64, 256, 512]), "temperature": 0, "chat_template_kwargs": nothink}))
    if kind == "sampled":
        return kind, text_of(post(base, "/v1/chat/completions", {"model": "x", "messages": msgs, "max_tokens": 256, "temperature": rng.choice([0.6, 0.8, 1.0]), "top_p": 0.95, "top_k": 20, "seed": rng.randrange(1 << 30), "chat_template_kwargs": nothink}))
    if kind == "stream":
        return kind, stream(base, {"model": "x", "messages": msgs, "max_tokens": 300, "temperature": 0, "chat_template_kwargs": nothink}, rng.choice([0, 0, 0, 0, 10]))
    if kind == "completion":
        return kind, text_of(post(base, "/v1/completions", {"model": "x", "prompt": f"A short history of {topic}:", "max_tokens": 128, "temperature": 0}))
    doc = " ".join(f"Fact {i}: {rng.choice(TOPICS)} relates to {rng.choice(TOPICS)}." for i in range(rng.randrange(250, 700)))
    return kind, text_of(post(base, "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": doc + "\nSummarise in one sentence."}], "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--duration", type=int, default=3600)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--out", default="soak.json")
    ap.add_argument("--kill-follower", default="")
    ap.add_argument("--failfast", type=int, default=60)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    canary_body = {"model": "x", "messages": [{"role": "user", "content": "List the planets of the solar system in order, comma separated."}],
                   "max_tokens": 80, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    canary = text_of(post(base, "/v1/chat/completions", canary_body))
    stats = {"requests": 0, "errors": [], "by_kind": {}, "canaries": 0, "canary_mismatch": 0}
    lock = threading.Lock()
    end = time.time() + a.duration

    def worker(i):
        rng = random.Random(1000 + i)
        while time.time() < end:
            try:
                kind, _ = request(base, rng)
                with lock:
                    stats["requests"] += 1
                    stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
            except Exception as e:  # noqa: BLE001
                with lock:
                    stats["errors"].append(f"{time.strftime('%H:%M:%S')} {type(e).__name__}: {str(e)[:200]}")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.clients)]
    for t in threads:
        t.start()
    t0 = time.time()
    while time.time() < end:
        time.sleep(min(300, max(1, end - time.time())))
        try:
            same = text_of(post(base, "/v1/chat/completions", canary_body)) == canary
        except Exception as e:  # noqa: BLE001
            same = False
            stats["errors"].append(f"canary {type(e).__name__}: {e}")
        stats["canaries"] += 1
        stats["canary_mismatch"] += 0 if same else 1
        print(f"[soak] {time.time() - t0:.0f}s: {stats['requests']} requests, {len(stats['errors'])} errors, canary "
              f"{'ok' if same else 'MISMATCH'}", flush=True)
    for t in threads:
        t.join(timeout=900)
    stats["seconds"] = time.time() - t0
    if a.kill_follower:
        res = {}
        box = {}

        def victim():
            t1 = time.time()
            try:
                post(base, "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "Write a long essay on rivers."}], "max_tokens": 4000, "temperature": 0}, timeout=a.failfast * 3)
                box["outcome"] = "completed"
            except Exception as e:  # noqa: BLE001
                box["outcome"] = f"{type(e).__name__}: {str(e)[:160]}"
            box["seconds"] = time.time() - t1

        th = threading.Thread(target=victim)
        th.start()
        time.sleep(5)
        subprocess.run(a.kill_follower, shell=True, check=False)
        th.join(timeout=a.failfast * 3)
        res["outcome"] = box.get("outcome", "still waiting (hung)")
        res["seconds_after_kill"] = round(box.get("seconds", float("nan")) - 5, 1) if "seconds" in box else None
        res["fails_fast"] = "seconds" in box and box.get("outcome") != "completed" and box["seconds"] - 5 <= a.failfast
        stats["failfast"] = res
        print(f"[soak] follower killed: {res}", flush=True)
    ok = not stats["errors"] and stats["canary_mismatch"] == 0 and (not a.kill_follower or stats["failfast"]["fails_fast"])
    stats["pass"] = ok
    json.dump(stats, open(a.out, "w"), indent=1)
    print(f"{'PASS' if ok else 'FAIL'} soak: {stats['requests']} requests in {stats['seconds']:.0f}s "
          f"({stats['by_kind']}), {len(stats['errors'])} errors, canaries {stats['canaries'] - stats['canary_mismatch']}/{stats['canaries']}")


if __name__ == "__main__":
    main()
