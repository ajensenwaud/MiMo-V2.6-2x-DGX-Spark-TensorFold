#!/usr/bin/env python3
"""Long-context retrieval check (passkey in a haystack): contexts x depths, greedy, passkey found in the reply.

  python3 tools/mimo_niah.py http://localhost:8892 --contexts 32000,120000 --depths 0.1,0.3,0.5,0.7,0.9 --out F.json
"""
import argparse
import json
import random
import time
import urllib.request

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
          "A river runs past the old mill, and the town wakes slowly. Nothing here is out of the ordinary. ")
TOKENS_PER_WORD = 1.408   # spark-bench's calibration on this model


def post(base: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def model_of(base: str) -> str:
    with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--contexts", default="32000,120000")
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model = model_of(a.base)
    rng = random.Random(a.seed)
    words = FILLER.split()
    rows, hits = [], 0
    for ctx in [int(x) for x in a.contexts.split(",")]:
        n_words = int(ctx / TOKENS_PER_WORD)
        hay = (words * (n_words // len(words) + 1))[:n_words]
        for depth in [float(x) for x in a.depths.split(",")]:
            key = str(rng.randint(10000, 99999))
            at = int(len(hay) * depth)
            text = " ".join(hay[:at]) + f" The secret passkey is {key}. Remember it: {key}. " + " ".join(hay[at:])
            body = {"model": model, "temperature": 0, "max_tokens": 256, "messages": [
                {"role": "user", "content": text + "\n\nWhat is the secret passkey in the text above? "
                                                   "Answer with the number only."}]}
            t0 = time.time()
            r = post(a.base, body)
            reply = r["choices"][0]["message"].get("content") or ""
            ok = key in reply
            hits += ok
            rows.append({"context": ctx, "depth": depth, "key": key, "ok": ok, "s": round(time.time() - t0, 1),
                         "prompt_tokens": r.get("usage", {}).get("prompt_tokens"), "reply": reply[-120:]})
            print(f"ctx {ctx:7d} depth {depth:.1f}: {'PASS' if ok else 'FAIL'} ({rows[-1]['s']} s) {reply[-60:]!r}",
                  flush=True)
    print(f"niah: {hits}/{len(rows)}")
    if a.out:
        json.dump({"hits": hits, "n": len(rows), "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
