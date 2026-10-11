#!/usr/bin/env python3
"""Fixed-text decode, request for request: Python vs a Zig arm (rounds, drafted, accepted, ms a round, tok/s).

  python3 fixed_compare.py LOGDIR ARM [ARM ...]
"""
import sys
from pathlib import Path


def load(d: Path, arm: str):
    sp = [float(l.split()[3]) for l in open(d / f"fixed-{arm}.txt") if "tok/s" in l]
    dn = [l.split() for l in open(d / f"fixed-{arm}-done.txt")]
    return sp, dn


def main() -> None:
    d = Path(sys.argv[1])
    P = load(d, "python")
    kinds = ["chat"] * 3 + ["code"] * 3 + ["agent"] * 3 + ["report"] * 3
    for arm in sys.argv[2:]:
        Z = load(d, arm)
        print(f"{arm}: python / zig")
        for i, w in enumerate(kinds):
            ps, pd, zs, zd = P[0][i], P[1][i], Z[0][i], Z[1][i]
            pr, zr = int(pd[2]), int(zd[2])
            pa, pdr = map(int, pd[3].split("/"))
            za, zdr = map(int, zd[3].split("/"))
            same = "=" if pd[1] == zd[1] else "X"
            print(f"  {w:6} sha{same} rounds {pr:4d}/{zr:4d} drafted {pdr:4d}/{zdr:4d} accepted {pa:4d}/{za:4d}"
                  f"  ms/round {512 / ps / pr * 1e3:5.1f}/{512 / zs / zr * 1e3:5.1f}  tok/s {ps:5.1f}/{zs:5.1f}")


if __name__ == "__main__":
    main()
