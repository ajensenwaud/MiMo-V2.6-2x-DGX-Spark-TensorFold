#!/usr/bin/env python3
"""Where a graphed decode step's time goes (TP=2 or one GPU): torch.profiler over replays of captured decode graphs,
kernels grouped into categories, for 1-row and 8-row (draft window) steps.

  python3 tools/mimo_profile.py /model /results/ref/ref.safetensors [--tp 2 --rank R ...] [--json out]

Reports, per shape: wall time a step (host-timed, synchronised), GPU kernel time a step by category, and the share
of wall time with no kernel running (gaps: launch latency inside the graph, comm waits, host syncs).
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

import torch

CATS = [
    ("comm", r"nccl|roce|gather|allgather|AllGather|ncclDevKernel|rdma"),
    ("experts", r"exl3_moe|experts|routed|moe_|grouped"),
    ("exl3 gemv/gemm", r"exl3|gemv|gemm|trellis|matmul"),
    ("attention", r"_rows|_merge|_prompt|attn|flash"),
    ("qkv/rope", r"_qkv|rope"),
    ("router/topk", r"_topk|router|topk|sigmoid"),
    ("norm/add/act", r"_rms|_add_rms|rms|_silu|silu"),
    ("copies/misc", r"copy|Memcpy|memset|elementwise|index|cat|fill|reduce|argmax|softmax|vectorized"),
]


def cat(name: str) -> str:
    for c, pat in CATS:
        if re.search(pat, name, re.I):
            return c
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="10.100.128.1")
    ap.add_argument("--port", type=int, default=29571)
    ap.add_argument("--prompt", type=int, default=9000)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--json")
    ap.add_argument("--widths", default="", help="timing only: comma-separated window widths (e.g. 1,2,4,8,16)")
    ap.add_argument("--grid", default="", help="timing only: STREAMS:WIDTHS, e.g. 1,2,4:1,2,3,4,6,8 (a different text "
                                               "a stream)")
    ap.add_argument("--prefill", type=int, default=0, help="profile one prompt forward of this many rows instead")
    a = ap.parse_args()

    from safetensors import safe_open
    from torch.profiler import ProfilerActivity, profile

    from tensorfold.families.mimo_v2.cuda.forward import Chain, Model
    from tensorfold.families.mimo_v2.cuda.graphs import Graphs
    from tensorfold.families.mimo_v2.cuda.weights import load

    say = (lambda *x: print(*x, flush=True)) if a.rank == 0 else (lambda *x: None)
    torch.cuda.set_device(0)
    comm = None
    if a.tp == 2:
        from tensorfold.cuda.comm import open_comm

        comm = open_comm(a.rank, 2, a.master, a.port)
        if os.environ.get("TF_MIMO_COMM", "roce") == "roce":
            from tensorfold.cuda.roce import RoceComm

            comm = RoceComm(comm, a.rank, 2)
    model = Model(load(a.model, rank=a.rank, world=a.tp), 32768, 4 if a.grid else 1, comm=comm)
    if comm is not None:
        comm.barrier()
        if hasattr(comm, "settle"):
            comm.settle()
    ref = safe_open(a.ref, framework="pt")
    pool = [t for k in sorted(ref.keys()) if k.endswith(".ids") for t in ref.get_tensor(k).tolist()]
    prompt = (pool * (a.prompt // len(pool) + 1))[:a.prompt]
    if a.prefill:                                          # one prompt chunk of a.prefill rows, profiled
        rows = (pool * (a.prefill // len(pool) + 1))[:a.prefill]
        for _ in range(2):                                 # compile, warm
            model.forward([Chain(0, 0, rows)], prompt=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.forward([Chain(0, 0, rows)], prompt=True)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t) * 1e3
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model.forward([Chain(0, 0, rows)], prompt=True)
            torch.cuda.synchronize()
        by, top = defaultdict(float), defaultdict(float)
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                us = e.device_time if hasattr(e, "device_time") else e.cuda_time
                by[cat(e.name)] += us
                top[e.name] += us
        report = {"tp": a.tp, "rows": a.prefill, "wall_ms": round(wall, 1), "tok_s": round(a.prefill / wall * 1e3, 1),
                  "kernel_ms": round(sum(by.values()) / 1e3, 1),
                  "by_category_ms": {k: round(v / 1e3, 1) for k, v in sorted(by.items(), key=lambda x: -x[1])},
                  "top_kernels_ms": {k[:100]: round(v / 1e3, 1) for k, v in sorted(top.items(), key=lambda x: -x[1])[:15]}}
        say(f"[profile] {json.dumps(report)}")
        if a.json and a.rank == 0:
            with open(a.json, "w") as f:
                json.dump(report, f, indent=1)
        return
    if a.grid:                                             # cost of S streams x w rows, each stream its own text
        texts = [ref.get_tensor(k).tolist() for k in sorted(ref.keys()) if k.endswith(".ids")]
        texts = [(t * (a.prompt // len(t) + 1))[:a.prompt] for t in texts]
        for s, t in enumerate(texts[:4]):
            model.prefill(t, slot=s)
        graphs = Graphs(model)
        ss, ws = a.grid.split(":")
        grid, p = {}, a.prompt
        for S in [int(x) for x in ss.split(",")]:
            for w in [int(x) for x in ws.split(",")]:
                chains = [Chain(s, p, texts[s][300 + 37 * s:300 + 37 * s + w]) for s in range(S)]
                for _ in range(3):
                    graphs.forward(chains)
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(a.steps):
                    graphs.forward(chains)
                torch.cuda.synchronize()
                grid[f"{S}x{w}"] = round((time.perf_counter() - t) / a.steps * 1e3, 3)
        base = grid.get("1x1")
        report = {"tp": a.tp, "position": p, "grid_ms": grid,
                  "grid_cost_vs_1": {k: round(v / base, 3) for k, v in grid.items()} if base else None}
        say(f"[profile] {json.dumps(report)}")
        if a.json and a.rank == 0:
            with open(a.json, "w") as f:
                json.dump(report, f, indent=1)
        return
    model.prefill(prompt)
    graphs = Graphs(model)
    p = len(prompt)
    report = {"tp": a.tp, "position": p}
    if a.widths:                                           # verification cost curve: wall ms a window of w rows
        curve = {}
        for wdt in [int(x) for x in a.widths.split(",")]:
            chains = [Chain(0, p, prompt[200:200 + wdt])]
            for _ in range(3):
                graphs.forward(chains)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(a.steps):
                graphs.forward(chains)
            torch.cuda.synchronize()
            curve[wdt] = round((time.perf_counter() - t) / a.steps * 1e3, 3)
        base = curve.get(1)
        report["window_ms"] = curve
        report["window_cost_vs_1"] = {k: round(v / base, 3) for k, v in curve.items()} if base else None
        say(f"[profile] {json.dumps(report)}")
        if a.json and a.rank == 0:
            with open(a.json, "w") as f:
                json.dump(report, f, indent=1)
        return
    for rows in (1, 8):
        chains = [Chain(0, p, prompt[200:200 + rows])]
        for _ in range(3):                                 # capture + warm
            graphs.forward(chains)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(a.steps):
            graphs.forward(chains)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t) / a.steps * 1e3
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(a.steps):
                graphs.forward(chains)
            torch.cuda.synchronize()
        by, top, kn = defaultdict(float), defaultdict(float), defaultdict(int)
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                us = e.device_time if hasattr(e, "device_time") else e.cuda_time
                by[cat(e.name)] += us
                top[e.name] += us
                kn[e.name] += 1
        gpu = sum(by.values()) / a.steps / 1e3
        launches = sum(kn.values()) / a.steps
        report[f"{rows}_rows"] = {
            "wall_ms": round(wall, 3), "kernel_ms": round(gpu, 3), "idle_ms": round(wall - gpu, 3),
            "kernels_per_step": round(launches),
            "by_category_ms": {k: round(v / a.steps / 1e3, 3) for k, v in sorted(by.items(), key=lambda x: -x[1])},
            "top_kernels_ms": {k[:90]: round(v / a.steps / 1e3, 3)
                               for k, v in sorted(top.items(), key=lambda x: -x[1])[:15]},
        }
        say(f"[profile] {rows} row(s): {json.dumps(report[f'{rows}_rows'])}")
    if a.json and a.rank == 0:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
