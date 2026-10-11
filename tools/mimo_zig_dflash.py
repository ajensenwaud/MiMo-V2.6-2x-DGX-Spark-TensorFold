#!/usr/bin/env python3
"""Z4's Python reference: the DFlash drafter's proposals under a fixed protocol, on both TP ranks.

For each answer-key prompt (tools/mimo_zig_capture.py), one target decode row a step (each row feeds the drafter's
context through Model.forward's taps), then a draft block from the next token (DraftRunner, graphed as served). Per
step it records the 7 drafts and how many lead the answer key's continuation (the tokens a verify would accept).
Writes /out/dflash-r<R>.json; with --tools the Triton launches (launches/jit-dflash-r<R>.json).

  python3 tools/mimo_zig_dflash.py /model /oracle/rank<R>.safetensors /out --tp 2 --rank R [--tools DIR]
"""

import argparse
import json
import time

import torch

PROMPTS = ["en_story", "zh_explain", "code_py", "math", "tool_json", "edit", "list", "short"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("oracle")
    ap.add_argument("out")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="10.100.128.1")
    ap.add_argument("--port", type=int, default=29641)
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
    model = Model(w, a.context, 1, comm=comm)
    model.attach_drafter(str(Path(a.model) / "dflash"))
    if comm is not None:
        comm.barrier()
        if hasattr(comm, "settle"):
            comm.settle()
    eos = set(w.config.eos)
    ora = safe_open(a.oracle, framework="pt")
    out = {"block": model.drafter.block, "prompts": {}}
    t0 = time.perf_counter()
    for p in PROMPTS:
        ids = ora.get_tensor(f"{p}.ids").tolist()
        seq = ids + [t for t in ora.get_tensor(f"{p}.greedy").tolist() if t not in eos]
        steps = []
        for pos in range(len(seq) - 1):
            model.forward([Chain(0, pos, [seq[pos]])], prompt=False)       # feeds the drafter's context
            d = model.draft_runner([seq[pos + 1]], [pos + 1], [0])[0]
            want = seq[pos + 2:pos + 2 + len(d)]
            acc = 0
            while acc < len(want) and d[acc] == want[acc]:
                acc += 1
            steps.append({"drafts": d, "accepted": acc})
        torch.cuda.synchronize()
        out["prompts"][p] = {"len": len(seq), "steps": steps}
        mean = sum(s["accepted"] for s in steps) / max(1, len(steps))
        print(f"[dflash] rank {a.rank} {p}: {len(steps)} drafts, mean accepted {mean:.2f}", flush=True)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"dflash-r{a.rank}.json").write_text(json.dumps(out))
    print(f"[dflash] rank {a.rank}: {time.perf_counter() - t0:.0f}s", flush=True)
    if rec_aot is not None:
        from triton.runtime.jit import JITFunction
        import gc

        rec_aot.dump(od / f"launches-dflash-r{a.rank}.json")
        jit = {}
        for fn in gc.get_objects():
            if isinstance(fn, JITFunction):
                jit[f"{fn.fn.__module__}.{fn.fn.__qualname__}"] = {
                    "params": [q.name for q in fn.params],
                    "do_not_specialize": [q.name for q in fn.params if q.do_not_specialize],
                    "no_align": [q.name for q in fn.params if q.do_not_specialize_on_alignment]}
        (od / f"jit-dflash-r{a.rank}.json").write_text(json.dumps(jit, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
