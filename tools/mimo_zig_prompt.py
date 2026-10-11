#!/usr/bin/env python3
"""Z6 answer key: the Python engine's PROMPT path (prompt GEMM, full-attention prompt kernel, per-expert prompt MoE),
recorded on both TP ranks, then a prompt-shape variant sweep for the Zig kernel set.

  python3 tools/mimo_zig_prompt.py /model /out --tp 2 --rank R [--tools DIR]

Scenarios (one slot, context 32768): a deterministic text (the installed TensorFold sources) cut to N tokens, fed as
prompt chunks exactly as given, then --steps greedy decode rows. Per scenario and chunk it records, in
/out/prompt-r<R>.safetensors:

  <s>.ids                  the scenario's prompt ids            <s>.chunks         chunk sizes
  <s>.c<j>.logits          fp32 logits of chunk j's last row
  <s>.c<j>.fp.<i>          per-row fingerprint int64 of the residual after layer i's MLP (sum_c bits[r, c] * (2c + 1))
  <s>.c<j>.fpa.<i>         ... after layer i's attention
  <s>.argmax               greedy ids of the decode rows after the prompt (first from the last chunk's logits)

and /out/prompt-r<R>.json with each tensor's sha256 per layer. Capture disables the drafter (as in the decode key);
the sweep afterwards runs prompt forwards with the drafter attached (its prompt-update path) at chunk sizes of every
Triton integer class, plus decode windows, recording launches with --tools.
"""

import argparse
import hashlib
import inspect
import json
import time

import torch

SCENARIOS = [            # name, prompt tokens, chunks
    ("short", 40, [40]),
    ("mid", 700, [700]),
    ("multi", 2600, [1024, 1024, 552]),
    ("mixed", 1224, [1024, 200]),
    ("big", 9000, [9000]),
]


def fingerprint(x: torch.Tensor) -> torch.Tensor:
    """Per row: sum over columns of the fp32 bits (as int32) times (2c + 1), in int64 (no overflow at 4096 wide)."""

    bits = x.contiguous().view(torch.int32).long()
    w = torch.arange(1, 2 * bits.shape[1], 2, device=x.device, dtype=torch.int64)
    return (bits * w[None, :]).sum(1)


class Hashing(dict):
    """A forward's capture dict that keeps per-row fingerprints and a sha256, not the tensors."""

    def __init__(self):
        super().__init__()
        self.fp, self.sha = {}, {}

    def __setitem__(self, key, value):
        self.fp[key] = fingerprint(value).cpu()
        self.sha[key] = hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def corpus() -> str:
    import tensorfold.families.mimo_v2.cuda.attention as a
    import tensorfold.families.mimo_v2.cuda.decoder as d
    import tensorfold.families.mimo_v2.cuda.dflash as f
    import tensorfold.families.mimo_v2.cuda.forward as fw
    import tensorfold.families.mimo_v2.cuda.moe as m
    import tensorfold.families.mimo_v2.cuda.weights as w

    return "\n\n".join(inspect.getsource(x) for x in (d, fw, a, m, f, w)) * 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="10.100.128.1")
    ap.add_argument("--port", type=int, default=29681)
    ap.add_argument("--tools", default="")
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args()
    import os
    import sys
    from pathlib import Path

    rec_aot = None
    if a.tools:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec_aot = aot.Recorder().install()
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

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
    model = Model(w, a.context, 4, comm=comm)
    model.attach_drafter(str(Path(a.model) / "dflash"))
    if comm is not None:
        comm.barrier()
        if hasattr(comm, "settle"):
            comm.settle()
    eos = set(w.config.eos)
    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    text_ids = tok.encode(corpus()).ids
    print(f"[prompt] rank {a.rank}: corpus {len(text_ids)} tokens", flush=True)
    tensors, report = {}, {}
    for name, n, chunks in SCENARIOS:
        assert sum(chunks) == n and n <= len(text_ids)
        ids = text_ids[:n]
        t = time.perf_counter()
        p0, sha, lg = 0, {}, None
        for j, c in enumerate(chunks):
            cap = Hashing()
            lg = model.forward([Chain(0, p0, ids[p0:p0 + c])], prompt=True, capture=cap)
            tensors[f"{name}.c{j}.logits"] = lg[0].detach().float().cpu().contiguous()
            for key, fp in cap.fp.items():
                kind, i = ("fpa", key.split(".")[0]) if key.endswith(".attn") else ("fp", key)
                tensors[f"{name}.c{j}.{kind}.{i}"] = fp.contiguous()
            sha[f"c{j}"] = cap.sha
            p0 += c
        seq = list(ids)
        nxt = int(lg[0].argmax())
        out = [nxt]
        while len(out) < a.steps and nxt not in eos:
            seq.append(nxt)
            lg = model.forward([Chain(0, len(seq) - 1, [nxt])], prompt=False)
            nxt = int(lg[0].argmax())
            out.append(nxt)
        torch.cuda.synchronize()
        tensors[f"{name}.ids"] = torch.tensor(ids, dtype=torch.int32)
        tensors[f"{name}.chunks"] = torch.tensor(chunks, dtype=torch.int32)
        tensors[f"{name}.argmax"] = torch.tensor(out, dtype=torch.int32)
        report[name] = {"tokens": n, "chunks": chunks, "seconds": round(time.perf_counter() - t, 1),
                        "reply": tok.decode(out)[:120], "sha": sha}
        print(f"[prompt] rank {a.rank} {name}: {n} tokens in {chunks}, {report[name]['seconds']}s: "
              f"{report[name]['reply'][:60]!r}", flush=True)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(od / f"prompt-r{a.rank}.safetensors"),
              metadata={"engine": "tensorfold-python mimo_v2 prompt path", "rank": str(a.rank)})
    (od / f"prompt-r{a.rank}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    if not a.no_sweep:   # every prompt-chunk integer class, drafter attached (prompt update), several positions
        t = time.perf_counter()
        for p0 in (0, 15 * 256 + 64, 20000):
            for c in (1, 2, 15, 16, 17, 100, 128, 129, 255, 256, 257, 1000, 1024, 1025, 4096, 4111):
                if p0 + c <= a.context:
                    model.forward([Chain(1, p0, text_ids[:c])], prompt=True)
        torch.cuda.synchronize()
        print(f"[prompt] rank {a.rank}: sweep {time.perf_counter() - t:.0f}s", flush=True)
    if rec_aot is not None:
        from triton.runtime.jit import JITFunction
        import gc

        rec_aot.dump(od / f"launches-prompt-r{a.rank}.json")
        jit = {}
        for fn in gc.get_objects():
            if isinstance(fn, JITFunction):
                jit[f"{fn.fn.__module__}.{fn.fn.__qualname__}"] = {
                    "params": [p.name for p in fn.params],
                    "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                    "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
        (od / f"jit-prompt-r{a.rank}.json").write_text(json.dumps(jit, indent=1, sort_keys=True) + "\n")
    if comm is not None:   # both ranks' replies must agree
        mine = torch.cat([tensors[f"{s[0]}.argmax"] for s in SCENARIOS]).cuda().to(torch.int32)
        both = torch.empty((2, mine.numel()), dtype=torch.int32, device="cuda")
        getattr(comm, "nccl", comm).all_gather(mine, both)
        print(f"[prompt] rank {a.rank}: replies equal on both ranks: {bool(torch.equal(both[0], both[1]))}", flush=True)


if __name__ == "__main__":
    main()
