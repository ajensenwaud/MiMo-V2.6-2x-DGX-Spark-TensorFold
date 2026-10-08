"""Serving smoke against a running endpoint: one request, N concurrent, a long prompt. Stdlib only."""

import json
import sys
import threading
import time
import urllib.request

base, model = sys.argv[1], sys.argv[2]
streams = int(sys.argv[3]) if len(sys.argv) > 3 else 4


def ask(prompt, tokens=64, out=None, i=0):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    res = (j["usage"]["completion_tokens"], j["choices"][0]["finish_reason"], time.time() - t)
    if out is not None:
        out[i] = res
    return res


print("single:", ask("Hello there, tell me about Paris."))
out = [None] * streams
ts = [threading.Thread(target=ask, args=(f"Request {i}: count to twenty in German.", 128, out, i)) for i in range(streams)]
t0 = time.time()
for t in ts:
    t.start()
for t in ts:
    t.join()
total = sum(o[0] for o in out)
print(f"{streams} concurrent: {out}; aggregate {total / (time.time() - t0):.1f} tok/s")
long = " ".join(f"word{i}" for i in range(900))
print("long prompt:", ask("Summarise: " + long, 32))
print("single again:", ask("Hello there, tell me about Paris."))
