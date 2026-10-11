#!/usr/bin/env python3
"""A served rank 0's GPU timeline from an nsys (cuda-sw) sqlite: graph replays by graph id (count, mean/median ms),
eager kernels and copies, and the GPU's idle time between activities (where the host or the peer holds it up).

  python3 nsys_serve.py serve1.sqlite
"""
import collections
import sqlite3
import statistics as st
import sys


def main() -> None:
    c = sqlite3.connect(sys.argv[1])
    ev = []   # (start, end, kind, label)
    for s, e, g in c.execute("select start, end, graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        ev.append((s, e, "graph", g))
    for s, e, name in c.execute("select k.start, k.end, n.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                                "join StringIds n on n.id = k.shortName where k.graphId is null or k.graphId = 0"):
        ev.append((s, e, "kernel", name))
    for s, e, kind in c.execute("select start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        ev.append((s, e, "copy", kind))
    ev.sort()
    if not ev:
        print("no GPU activity")
        return
    by = collections.defaultdict(list)
    for s, e, k, l in ev:
        by[(k, l)].append((e - s) / 1e6)
    print("activity: count, mean ms, median ms, total ms")
    for (k, l), d in sorted(by.items(), key=lambda x: -sum(x[1]))[:25]:
        print(f"  {k:6} {str(l)[:40]:40} {len(d):6} {st.mean(d):8.3f} {st.median(d):8.3f} {sum(d):10.1f}")
    span = (ev[-1][1] - ev[0][0]) / 1e6
    busy, end = 0.0, ev[0][0]
    gaps = []
    for s, e, k, l in ev:
        if s > end:
            gaps.append((s - end) / 1e6)
        busy += max(0, e - max(s, end)) / 1e6
        end = max(end, e)
    print(f"span {span:.1f} ms, GPU busy {busy:.1f} ms ({busy / span:.0%}); gaps: {len(gaps)}, "
          f"total {sum(gaps):.1f} ms, >1 ms: {sum(1 for g in gaps if g > 1)} totalling {sum(g for g in gaps if g > 1):.1f} ms")
    big = sorted(gaps)[-10:]
    print("largest gaps ms:", [round(g, 2) for g in big])
    # what precedes and follows each graph replay: the gap before it
    pre = collections.defaultdict(list)
    prev_end = None
    for s, e, k, l in ev:
        if k == "graph" and prev_end is not None:
            pre[l].append((s - prev_end) / 1e6)
        prev_end = e if prev_end is None else max(prev_end, e)
    for g, d in sorted(pre.items(), key=lambda x: -len(x[1]))[:8]:
        print(f"  graph {g}: {len(d)} replays, gap before median {st.median(d):.3f} ms, mean {st.mean(d):.3f}")


if __name__ == "__main__":
    main()
