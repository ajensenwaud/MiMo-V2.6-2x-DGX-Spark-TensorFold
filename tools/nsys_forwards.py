#!/usr/bin/env python3
"""Per-kernel time of eager Zig forwards from an nsys (cuda-sw) sqlite export: split the kernel stream into forwards at
each embed launch, keep runs of `--want` same-shaped forwards, and sum kernel time by name per forward.

  python3 nsys_forwards.py zig3.sqlite [--list] [--group G] [--json out.json]
"""
import argparse
import collections
import json
import sqlite3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--group", type=int, default=-1, help="index of the forward group to break down")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    c = sqlite3.connect(a.db)
    rows = c.execute("select k.start, k.end, s.value, k.gridX, k.gridY from CUPTI_ACTIVITY_KIND_KERNEL k "
                     "join StringIds s on s.id = k.shortName order by k.start").fetchall()
    fwd, cur = [], None
    for st, en, name, gx, gy in rows:
        if "embed" in name and "dr_" not in name:
            cur = []
            fwd.append(cur)
        if cur is not None:
            cur.append((st, en, name, gx, gy))
    # group consecutive forwards with the same kernel count
    groups = []
    for f in fwd:
        sig = (len(f), f[-1][3], f[-1][4], sum(x[3] for x in f))
        if groups and groups[-1][0] == sig:
            groups[-1][1].append(f)
        else:
            groups.append((sig, [f]))
    if a.list or a.group < 0:
        for i, (sig, fs) in enumerate(groups):
            if len(fs) < 5:
                continue
            busy = sum(sum(e - s for s, e, *_ in f) for f in fs) / len(fs) / 1e6
            span = sum(f[-1][1] - f[0][0] for f in fs) / len(fs) / 1e6
            print(f"group {i}: {len(fs)} forwards of {sig} kernels, kernel ms {busy:.2f}, span ms {span:.2f}")
        return
    sig, fs = groups[a.group]
    fs = fs[1:]   # the first may be a warm-up
    by = collections.defaultdict(float)
    for f in fs:
        for s, e, name, gx, gy in f:
            by[name] += (e - s) / 1e6
    total = sum(by.values()) / len(fs)
    span = sum(f[-1][1] - f[0][0] for f in fs) / len(fs) / 1e6
    out = {"forwards": len(fs), "kernels": sig, "kernel_ms": round(total, 3), "span_ms": round(span, 3),
           "by_kernel_ms": {k: round(v / len(fs), 3) for k, v in sorted(by.items(), key=lambda x: -x[1])}}
    print(json.dumps(out, indent=1)[:4000])
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
