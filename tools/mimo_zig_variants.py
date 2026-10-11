#!/usr/bin/env python3
"""Every Triton specialisation the decode path can launch, recorded from the Python engine (both TP ranks).

Triton compiles a variant per class of each integer argument (== 1, divisible by 16, other) and per constexpr, so a
captured set only covers the windows that ran. This runs eager decode windows of every row count 1 .. 128 (rows
spread over 4 slots) at low positions, and at positions past 15 * CK (full layers' 16-chunk class), then the DFlash
draft pass for 1 .. 4 streams; with --tools it records the launches (launches/jit-variants-r<R>.json). Outputs are not
checked: only the kernels launched matter.

  python3 tools/mimo_zig_variants.py /model /out --tp 2 --rank R --tools DIR
"""

import argparse
import json
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="10.100.128.1")
    ap.add_argument("--port", type=int, default=29651)
    ap.add_argument("--tools", default="")
    a = ap.parse_args()
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, a.tools)
    import triton_aot_manifest as aot

    rec = aot.Recorder().install()
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
    model.attach_drafter(str(Path(a.model) / "dflash"))
    if comm is not None:
        comm.barrier()
        if hasattr(comm, "settle"):
            comm.settle()
    t0 = time.perf_counter()
    tok = 1000
    for base in (0, 15 * 256 + 64):                          # chunks 1 .. 2 and 16 (context 4096, CK 256)
        for n in range(1, 129):
            per = [n // 4 + (1 if i < n % 4 else 0) for i in range(4)]
            chains = [Chain(s, base, [tok] * k) for s, k in enumerate(per) if k]
            model.forward(chains, prompt=False, rows=range(n))
    graphs = Graphs(model)                                   # graphed windows: bucket chunks
    for n in (1, 4, 16, 32, 33, 64, 128):
        per = [n // 4 + (1 if i < n % 4 else 0) for i in range(4)]
        chains = [Chain(s, 100, [tok] * k) for s, k in enumerate(per) if k]
        if graphs.eligible(chains):
            graphs.forward(chains)
    for s in range(1, 5):
        model.draft_runner([tok] * s, [200] * s, list(range(s)))
    torch.cuda.synchronize()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    rec.dump(od / f"launches-variants-r{a.rank}.json")
    from triton.runtime.jit import JITFunction
    import gc

    jit = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            jit[f"{fn.fn.__module__}.{fn.fn.__qualname__}"] = {
                "params": [q.name for q in fn.params],
                "do_not_specialize": [q.name for q in fn.params if q.do_not_specialize],
                "no_align": [q.name for q in fn.params if q.do_not_specialize_on_alignment]}
    (od / f"jit-variants-r{a.rank}.json").write_text(json.dumps(jit, indent=1, sort_keys=True) + "\n")
    print(f"[variants] rank {a.rank}: {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
