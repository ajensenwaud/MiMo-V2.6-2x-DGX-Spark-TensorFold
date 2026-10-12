#!/usr/bin/env python3
"""TensorFold's lane contracts against a served engine (greedy):
  1. drafted == plain ("draft": false), token for token, solo;
  2. each of N concurrent replies == its solo run (drafted and plain);
Prompts cover the three prompt paths: short (decode rows), medium (grouped MoE), long (unpack + GEMM).
Compares the server's token SHA when it reports one, else the text."""
import argparse, concurrent.futures as cf, hashlib, json, sys, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8891")
ap.add_argument("--model", default="MiMo-V2.6-Flash")
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--src", default=str(Path(__file__).resolve().parent))
ap.add_argument("--out", required=True)
a = ap.parse_args()

code = "".join(p.read_text() for p in sorted(Path(a.src).glob("*.py")))
words = code.split()
PROMPTS = {
    "chat": "Explain how a heat pump works to a twelve-year-old, then list three common myths about them.",
    "code": "Write a Python function that merges overlapping intervals, with type hints, a docstring and three tests.",
    "agent": 'You can call tools. Reply with JSON only: {"tool": name, "args": {...}}. Tools: search(query), '
             'read(path), write(path, text). Task: find the newest log file under /var/log/app and summarise its errors.',
    "report": "Write a short report on the trade-offs of 8-bit versus 16-bit key-value caches in LLM serving, "
              "with a summary, three sections and a conclusion.",
    "medium": "Here is some code:\n\n" + " ".join(words[:1100]) + "\n\nList every function above and what it does.",
    "long": "Here is some code:\n\n" + " ".join(words[:4200]) + "\n\nSummarise what this code is for in five bullets.",
}


def ask(name, draft):
    body = {"model": a.model, "messages": [{"role": "user", "content": PROMPTS[name]}], "max_tokens": a.tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    text = r["choices"][0]["message"].get("content") or ""
    tf = r.get("tensorfold") or {}
    spec = r.get("speculative") or {}
    key = tf.get("token_sha") or hashlib.sha256(text.encode()).hexdigest()[:12]
    return {"key": key, "sha_from": "server" if tf.get("token_sha") else "text", "drafts": tf.get("drafts"),
            "rounds": spec.get("rounds"), "drafted": spec.get("drafted"), "accepted": spec.get("accepted"),
            "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
            "s": round(time.time() - t, 1), "text": text[:120]}


res = {"solo": {}, "concurrent": {}}
fails = []
for draft in (False, True):
    for n in PROMPTS:
        res["solo"][f"{n}/{'drafted' if draft else 'plain'}"] = r = ask(n, draft)
        print(f"solo {n:7s} {'drafted' if draft else 'plain  '} p={r['prompt_tokens']:5d} c={r['completion_tokens']} "
              f"{r['key']} ({r['sha_from']}) rounds={r['rounds']} drafted={r['drafted']} {r['s']}s", flush=True)
for n in PROMPTS:
    p, d = res["solo"][f"{n}/plain"], res["solo"][f"{n}/drafted"]
    ok = p["key"] == d["key"]
    print(f"[1] drafted == plain  {n:7s}: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        fails.append(f"drafted!=plain {n}: plain '{p['text'][:60]}' / drafted '{d['text'][:60]}'")

groups = [["chat", "code", "agent", "report"], ["medium", "long", "chat", "code"], ["agent", "agent", "report", "long"]]
for draft in (True, False):
    mode = "drafted" if draft else "plain"
    for gi, g in enumerate(groups):
        with cf.ThreadPoolExecutor(len(g)) as ex:
            out = list(ex.map(lambda n: ask(n, draft), g))
        for i, (n, r) in enumerate(zip(g, out)):
            res["concurrent"][f"{mode}/g{gi}/{i}/{n}"] = r
            ok = r["key"] == res["solo"][f"{n}/{mode}"]["key"]
            print(f"[2] concurrent == solo {mode:7s} group {gi} {n:7s}: {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                fails.append(f"concurrent!=solo {mode} g{gi} {n}: solo '{res['solo'][f'{n}/{mode}']['text'][:60]}' "
                             f"/ conc '{r['text'][:60]}'")

res["fails"] = fails
Path(a.out).write_text(json.dumps(res, indent=1))
print(f"RESULT: {len(fails)} failures", flush=True)
for f in fails:
    print("  ", f)
sys.exit(1 if fails else 0)
