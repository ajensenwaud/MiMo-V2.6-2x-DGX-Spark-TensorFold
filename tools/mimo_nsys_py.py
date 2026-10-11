#!/usr/bin/env python3
"""Kernel-level reference for the Zig engine (Z7): the Python engine's graphed decode forward at TP=2, under nsys on
rank 0. 1-row and 4-row windows (distinct tokens) at position 9,000 of a 32K context, 40 replays each after capture,
drafter attached (its context update rides in the forward, as served). Pair with `tf-mimo timing` under nsys.

  python3 tools/mimo_nsys_py.py /model /out --tp 2 --rank R
"""

import argparse
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="10.100.128.1")
    ap.add_argument("--port", type=int, default=29671)
    ap.add_argument("--tools", default="")
    a = ap.parse_args()
    from pathlib import Path

    from tensorfold.families.mimo_v2.cuda.forward import Chain, Model
    from tensorfold.families.mimo_v2.cuda.graphs import Graphs
    from tensorfold.families.mimo_v2.cuda.weights import load

    torch.cuda.set_device(0)
    from tensorfold.cuda.comm import open_comm

    comm = open_comm(a.rank, 2, a.master, a.port)
    if os.environ.get("TF_MIMO_COMM", "roce") == "roce":
        from tensorfold.cuda.roce import RoceComm

        comm = RoceComm(comm, a.rank, 2)
    w = load(a.model, rank=a.rank, world=2)
    model = Model(w, 32768, 4, comm=comm)
    model.attach_drafter(str(Path(a.model) / "dflash"))
    comm.barrier()
    if hasattr(comm, "settle"):
        comm.settle()
    graphs = Graphs(model)
    g = torch.Generator().manual_seed(7)
    for n in (1, 4):
        def chains():
            toks = torch.randint(1000, 141000, (n,), generator=g).tolist()
            return [Chain(0, 9000, toks)]
        graphs.forward(chains())
        graphs.forward(chains())
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"py-{n}rows")
        t0 = time.perf_counter()
        for _ in range(40):
            graphs.forward(chains())
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print(f"[nsys-py] rank {a.rank}: {n} rows: {(time.perf_counter() - t0) / 40 * 1e3:.2f} ms a forward "
              f"(captures {graphs.captures}, replays {graphs.replays})", flush=True)
    s = [1000]
    model.draft_runner(s, [9000], [0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(40):
        model.draft_runner([1000 + 7919 * i % 140000], [9000], [0])
    torch.cuda.synchronize()
    print(f"[nsys-py] rank {a.rank}: draft pass 1 stream {(time.perf_counter() - t0) / 40 * 1e3:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
