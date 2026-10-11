#!/usr/bin/env python3
"""Z3's Python reference: decode windows of several streams and several rows a stream, eager and graphed, on both TP
ranks. Each row is teacher-forced from the answer key's sequence (tools/mimo_zig_capture.py: prompt + greedy), so
every row's logits must equal the serial answer key's row (Python's decode kernels are row-invariant: the claim this
records). Writes /out/windows-r<R>.json: the schedule (rounds of chains: slot, p0, prompt) and per row the argmax
and sha256 of its fp32 logits; with --tools also the Triton launches (launches/jit-windows-r<R>.json).

  python3 tools/mimo_zig_windows.py /model /oracle/rank<R>.safetensors /out --tp 2 --rank R [--tools DIR]
"""

import argparse
import hashlib
import json
import time

import torch

PROMPTS = ["en_story", "zh_explain", "code_py", "math", "tool_json", "edit", "list", "short"]
# widths a round cycles through (rows a stream contributes); 4 streams => windows of 4 to 64 rows
PATTERNS = [[1, 1, 1, 1], [2, 1, 3, 1], [8, 1, 1, 1], [9, 9, 9, 9], [16, 1, 1, 1], [4, 4, 4, 4], [1, 2, 1, 2],
            [32, 1, 1, 1], [3, 3, 3, 3], [1, 1, 1, 1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("oracle")
    ap.add_argument("out")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="10.100.128.1")
    ap.add_argument("--port", type=int, default=29631)
    ap.add_argument("--tools", default="")
    a = ap.parse_args()
    import os
    import sys
    from pathlib import Path

    rec_aot = None
    if a.tools:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec_aot = aot.Recorder().install()
    from safetensors import safe_open

    from tensorfold.families.mimo_v2.cuda.forward import Chain, Model
    from tensorfold.families.mimo_v2.cuda.graphs import Graphs
    from tensorfold.families.mimo_v2.cuda.weights import load

    torch.cuda.set_device(0)
    comm = None
    if a.tp == 2:
        from tensorfold.cuda.comm import open_comm

        comm = open_comm(a.rank, 2, a.master, a.port)
        if os.environ.get("TF_MIMO_COMM", "roce") == "roce":
            from tensorfold.cuda.roce import RoceComm

            comm = RoceComm(comm, a.rank, 2)
    w = load(a.model, rank=a.rank, world=a.tp)
    model = Model(w, a.context, 4, comm=comm)
    if comm is not None:
        comm.barrier()
        if hasattr(comm, "settle"):
            comm.settle()
    graphs = Graphs(model)
    eos = set(w.config.eos)
    ora = safe_open(a.oracle, framework="pt")
    seqs = {}
    for p in PROMPTS:
        ids = ora.get_tensor(f"{p}.ids").tolist()
        greedy = [t for t in ora.get_tensor(f"{p}.greedy").tolist() if t not in eos]
        seqs[p] = ids + greedy
    rounds, rows_out = [], []
    t0 = time.perf_counter()
    for group in (PROMPTS[:4], PROMPTS[4:]):
        at = {p: 0 for p in group}
        k = 0
        while any(at[p] < len(seqs[p]) for p in group):
            widths = PATTERNS[k % len(PATTERNS)]
            graphed = (k // len(PATTERNS)) % 2 == 0            # alternate: a pattern cycle graphed, the next eager
            chains, meta = [], []
            for slot, p in enumerate(group):
                n = min(widths[slot], len(seqs[p]) - at[p])
                if n <= 0:
                    continue
                chains.append(Chain(slot, at[p], seqs[p][at[p]:at[p] + n]))
                meta.append({"slot": slot, "prompt": p, "p0": at[p], "n": n})
                at[p] += n
            total = sum(m["n"] for m in meta)
            if graphed and graphs.eligible(chains):
                lg = graphs.forward(chains)
                mode = "graph"
            else:
                lg = model.forward(chains, prompt=False, rows=range(total))
                mode = "eager"
            torch.cuda.synchronize()
            lg = lg.float().cpu()
            r = 0
            for m in meta:
                for j in range(m["n"]):
                    row = lg[r].contiguous()
                    rows_out.append({"prompt": m["prompt"], "pos": m["p0"] + j, "argmax": int(row.argmax()),
                                     "sha256": hashlib.sha256(row.numpy().tobytes()).hexdigest()})
                    r += 1
            rounds.append({"mode": mode, "chains": meta})
            k += 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"windows-r{a.rank}.json").write_text(json.dumps({"context": a.context, "slots": 4, "rounds": rounds,
                                                              "rows": rows_out}))
    print(f"[windows] rank {a.rank}: {len(rounds)} rounds, {len(rows_out)} rows, graphs {graphs.captures} captured "
          f"/ {graphs.replays} replayed / {graphs.eager} eager, {time.perf_counter() - t0:.0f}s", flush=True)
    if rec_aot is not None:
        from triton.runtime.jit import JITFunction
        import gc

        rec_aot.dump(out / f"launches-windows-r{a.rank}.json")
        jit = {}
        for fn in gc.get_objects():
            if isinstance(fn, JITFunction):
                jit[f"{fn.fn.__module__}.{fn.fn.__qualname__}"] = {
                    "params": [q.name for q in fn.params],
                    "do_not_specialize": [q.name for q in fn.params if q.do_not_specialize],
                    "no_align": [q.name for q in fn.params if q.do_not_specialize_on_alignment]}
        (out / f"jit-windows-r{a.rank}.json").write_text(json.dumps(jit, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
