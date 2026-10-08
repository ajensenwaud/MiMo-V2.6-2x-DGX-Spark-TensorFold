#!/usr/bin/env python3
"""Summarise results/<label>/ into one markdown table: decode (one stream), concurrency, cold prefill."""

import json
import statistics
import sys
from pathlib import Path


def load(p):
    if not p.exists():
        return []
    t = p.read_text().strip()
    try:
        j = json.loads(t)
        if isinstance(j, dict) and "cells" in j:
            return j["cells"]
        if isinstance(j, dict) and "summary" in j:
            return j["summary"]
        return j if isinstance(j, list) else j.get("results", [j])
    except json.JSONDecodeError:
        return [json.loads(x) for x in t.splitlines() if x.strip().startswith("{")]


root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
labels = sorted(d.name for d in root.iterdir() if d.is_dir())
rows = []
for lab in labels:
    d = root / lab
    dec = {}
    for f, tok in (("decode-64.json", 64), ("decode-256.json", 256)):
        for r in load(d / f):
            dec[(tok, r["prompt"], r["temperature"])] = r["decode_tps_median"]
    conc = {}
    for r in load(d / "concurrent.json"):
        conc.setdefault(r["streams"], []).append(r["aggregate_tps"])
    eq = [(r.get("alone") or {}) for r in load(d / "concurrent.json")]
    pre = {r["length"]: r["tok_s"] for r in load(d / "prefill.json") if "tok_s" in r}
    rows.append((lab, dec, conc, eq, pre))

print("| Arm | decode 64 tok, t=0 (code / chat) | decode 256 tok, t=0 (code / chat) | decode t=1.0 (median of 4) "
      "| 2 streams agg | 4 streams agg | alone-equal | prefill 2k / 8k / 32k / 64k |")
print("|---|---|---|---|---|---|---|---|")
for lab, dec, conc, eq, pre in rows:
    def g(k):
        v = dec.get(k)
        return f"{v:.1f}" if v else "-"
    t1 = [v for (tok, _, t), v in dec.items() if t == 1.0]
    c = {k: f"{statistics.median(v):.1f}" for k, v in conc.items()}
    eqs = f"{sum(e.get('equal', 0) for e in eq)}/{sum(e.get('equal', 0) + e.get('unequal', 0) for e in eq)}" if eq else "-"
    p = " / ".join(f"{pre[n]:.0f}" if n in pre else "-" for n in (2048, 8192, 32768, 65536))
    print(f"| {lab} | {g((64, 'fibonacci-raw', 0.0))} / {g((64, 'gpu-chat-no-think', 0.0))} "
          f"| {g((256, 'fibonacci-raw', 0.0))} / {g((256, 'gpu-chat-no-think', 0.0))} "
          f"| {statistics.median(t1):.1f} | {c.get(2, '-')} | {c.get(4, '-')} | {eqs} | {p} |" if t1 else f"| {lab} | incomplete |")
