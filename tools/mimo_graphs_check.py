#!/usr/bin/env python3
"""CUDA-graph decode vs the eager decode forward, bit for bit, then graph decode speed (one GPU or --tp 2).

  python3 tools/mimo_graphs_check.py /model /results/ref/ref.safetensors [--tp 2 --rank R --master IP --port P]

Each case runs the eager forward, then the graph replay on the same chains: the replay rewrites the same cache rows
with the same bits (idempotent), so both see one state and their logits must be torch.equal. Cases: single rows
across the 8192-key bucket edge, 8- and 16-row windows, two streams in one forward. Then greedy decode timed for
``--steps`` tokens, eager and graphed. Both ranks run the same calls; rank 0 reports.
"""

import argparse
import json
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="10.100.128.1")
    ap.add_argument("--port", type=int, default=29571)
    ap.add_argument("--prompt", type=int, default=9000)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--json")
    a = ap.parse_args()

    from safetensors import safe_open

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
    t0 = time.perf_counter()
    w = load(a.model, rank=a.rank, world=a.tp)
    torch.cuda.empty_cache()
    model = Model(w, a.context, 2, comm=comm)
    if comm is not None:
        comm.barrier()
        if hasattr(comm, "settle"):
            comm.settle()
    say(f"[graphs] loaded in {time.perf_counter() - t0:.0f}s; {torch.cuda.memory_allocated() / 2**30:.1f} GiB "
        f"allocated on this rank")
    ref = safe_open(a.ref, framework="pt")
    pool = []
    for k in sorted(ref.keys()):
        if k.endswith(".ids"):
            pool += ref.get_tensor(k).tolist()
    prompt = (pool * (a.prompt // len(pool) + 1))[:a.prompt]
    for s in (0, 1):
        logits = model.prefill(prompt, slot=s)
    graphs = Graphs(model)
    report = {"prompt": len(prompt), "cases": {}, "mismatches": 0}

    def case(name: str, chains) -> None:
        n = sum(len(c.tokens) for c in chains)
        e = model.forward(chains, prompt=False, rows=range(n)).clone()
        g = graphs.forward(chains).clone()
        same = bool(torch.equal(e, g))
        diff = 0.0 if same else float((e - g).abs().max())
        r = report["cases"].setdefault(name, {"runs": 0, "equal": 0, "max_abs_diff": 0.0})
        r["runs"] += 1
        r["equal"] += same
        r["max_abs_diff"] = max(r["max_abs_diff"], diff)
        report["mismatches"] += not same

    p = len(prompt)
    tok = int(logits[0].argmax())
    for i in range(16):                                   # single rows, both slots, past the 8192 bucket edge
        case("1 row", [Chain(0, p + i, [tok])])
        case("1 row", [Chain(1, p + i, [tok])])
        tok = int(model.forward([Chain(0, p + i, [tok])], prompt=False)[0].argmax())
        model.forward([Chain(1, p + i, [tok])], prompt=False)
    p += 16
    for width in (8, 16):                                 # draft windows: the pending token plus width-1 guesses
        win = prompt[100:100 + width]
        case(f"{width}-row window", [Chain(0, p, win)])
    case("2 streams x 1 row", [Chain(0, p, [tok]), Chain(1, p, [tok])])
    case("2 streams x 4 rows", [Chain(0, p, prompt[:4]), Chain(1, p, prompt[4:8])])

    for mode in ("eager", "graphs"):                      # greedy decode speed, slot 0, from position p
        tok = int(model.forward([Chain(0, p, [tok])], prompt=False)[0].argmax())
        torch.cuda.synchronize()
        t = time.perf_counter()
        for i in range(1, a.steps + 1):
            c = [Chain(0, p + i, [tok])]
            out = graphs.forward(c) if mode == "graphs" else model.forward(c, prompt=False)
            tok = int(out[0].argmax())
        torch.cuda.synchronize()
        report[f"{mode}_decode_tok_s"] = round(a.steps / (time.perf_counter() - t), 2)
    report.update(captures=graphs.captures, replays=graphs.replays, eager_fallbacks=graphs.eager,
                  gib_allocated=round(torch.cuda.memory_allocated() / 2**30, 2), tp=a.tp,
                  context=a.context, decode_position=p)
    say("[graphs] " + json.dumps(report))
    say("[graphs] " + ("PASS: every graph replay bit-equal to eager" if report["mismatches"] == 0
                       else f"FAIL: {report['mismatches']} mismatching cases"))
    if a.json and a.rank == 0:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
