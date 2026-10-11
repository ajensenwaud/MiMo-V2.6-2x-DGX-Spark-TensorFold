#!/usr/bin/env python3
"""Answer key for the Zig port (ZIG-PLAN Z1.1): the Python engine's decode path, recorded on both TP ranks.

  python3 tools/mimo_zig_capture.py /model /out --tp 2 --rank R --master 10.100.128.1 --port P   (one per node)

Every prompt is fed one row at a time through decode forwards (the row-invariant kernels the Zig engine's decode
windows mirror), then decoded greedily for --steps tokens (stopping at EOS). Per prompt and rank it saves, in
/out/rank<R>.safetensors:

  <p>.ids            prompt ids                         <p>.greedy        greedy ids (argmax of fp32 logits)
  <p>.argmax         argmax at every step               <p>.logits        fp32 logits at the first --logits steps
  <p>.s<j>.res.<i>   residual after layer i's MLP, steps j < --deep   (and .attn.<i>: after its attention)
  <p>.s<j>.pick.<i>  routed experts [8] int32 / .wts.<i> weights fp32 (MoE layers), steps j < --deep
  <p>.s<j>.att.<i>   attention output [heads, v_dim] (this rank's heads) at the first full and first sliding layers

Both ranks' residuals, logits and picks must be equal (the forward sums partials in rank order); the script checks
the greedy ids against the peer's at the end. Runs eager (no graphs): the Python engine's graphs == eager bit for bit.
"""

import argparse
import json
import time

import torch

PROMPTS = {
    "en_story": "Write a short story of about 300 words about a lighthouse keeper who finds a message in a bottle.",
    "zh_explain": "请用简洁的中文解释什么是量子纠缠，并举一个日常生活中的类比。",
    "code_py": "Write a Python function that parses an ISO-8601 duration string such as 'P3DT4H5M' into a "
               "datetime.timedelta, with type hints and three pytest tests.",
    "math": "A train leaves city A at 9:00 travelling at 80 km/h towards city B, 360 km away. Another leaves B at "
            "10:00 towards A at 100 km/h. When and where do they meet? Reason step by step.",
    "tool_json": "You can call tools. Respond only with a JSON object {\"tool\": name, \"arguments\": {...}}. "
                 "Tools: get_weather(city: str, unit: 'C'|'F'). User: what's the weather in Sydney in Celsius?",
    "edit": "Rename the variable `cnt` to `count` and add docstrings:\n\ndef f(xs):\n    cnt = 0\n    for x in xs:\n"
            "        if x > 0:\n            cnt += 1\n    return cnt\n",
    "list": "List the planets of the solar system in order from the sun, one per line, with one fact each.",
    "short": "Hello!",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--logits", type=int, default=16)
    ap.add_argument("--deep", type=int, default=4)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="10.100.128.1")
    ap.add_argument("--port", type=int, default=29591)
    ap.add_argument("--tools", default="", help="folder of TensorFold's tools/zig/triton_aot_manifest.py: record every "
                    "Triton launch (launches.json, jit.json) for the Zig engine's kernel set")
    a = ap.parse_args()
    import os
    import sys

    rec_aot = None
    if a.tools:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec_aot = aot.Recorder().install()
    from pathlib import Path

    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    from tensorfold.cuda.exl3 import experts as x3experts
    from tensorfold.families.mimo_v2.cuda import attention
    from tensorfold.families.mimo_v2.cuda.forward import Chain, Model
    from tensorfold.families.mimo_v2.cuda.weights import load

    torch.cuda.set_device(0)
    t0 = time.perf_counter()
    comm = None
    if a.tp == 2:
        from tensorfold.cuda.comm import open_comm

        comm = open_comm(a.rank, 2, a.master, a.port)
        if os.environ.get("TF_MIMO_COMM", "roce") == "roce":
            from tensorfold.cuda.roce import RoceComm

            comm = RoceComm(comm, a.rank, 2)
    w = load(a.model, rank=a.rank, world=a.tp)
    model = Model(w, a.context, 1, comm=comm)
    if comm is not None:
        comm.barrier()
        if hasattr(comm, "settle"):
            comm.settle()
    cfg = w.config
    eos = set(cfg.eos)
    first_full = next(i for i, f in enumerate(cfg.full) if f)
    first_swa = next(i for i, f in enumerate(cfg.full) if not f)
    print(f"[capture] rank {a.rank}: loaded in {time.perf_counter() - t0:.0f}s, first full layer {first_full}, "
          f"first sliding layer {first_swa}", flush=True)
    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))

    # hooks, active only while `rec` is a dict: routed picks / weights in layer order, attention outputs
    rec = {"on": False, "picks": [], "att": []}
    orig_routed = x3experts.routed
    orig_rows = attention.rows

    def routed(x, pick, wts, *args, **kw):
        if rec["on"]:
            rec["picks"].append((pick.detach().clone(), None if wts is None else wts.detach().clone()))
        return orig_routed(x, pick, wts, *args, **kw)

    def rows(*args, **kw):
        out = orig_rows(*args, **kw)
        if rec["on"]:
            rec["att"].append(out.detach().clone())
        return out

    x3experts.routed = routed
    import tensorfold.families.mimo_v2.cuda.moe as moe_mod
    moe_mod.x3experts.routed = routed
    attention.rows = rows
    import tensorfold.families.mimo_v2.cuda.forward as fwd_mod
    fwd_mod.attention.rows = rows

    moe_layers = [i for i, L in enumerate(w.layers) if L.experts is not None]
    tensors: dict[str, torch.Tensor] = {}
    summary = {}
    for name, text in PROMPTS.items():
        ids = tok.encode(text).ids
        argmax, greedy, logits_kept = [], [], []
        seq = list(ids)
        t = time.perf_counter()
        step = 0
        while step < len(seq):
            deep = step < a.deep
            cap = {} if deep else None
            rec["on"], rec["picks"], rec["att"] = deep, [], []
            lg = model.forward([Chain(0, step, [seq[step]])], prompt=False, capture=cap)   # [1, V] fp32
            rec["on"] = False
            nxt = int(lg[0].argmax())
            argmax.append(nxt)
            if step < a.logits:
                logits_kept.append(lg[0].detach().cpu())
            if deep:
                for i in range(len(w.layers)):
                    tensors[f"{name}.s{step}.res.{i}"] = cap[str(i)][0].cpu().contiguous()
                    tensors[f"{name}.s{step}.attn.{i}"] = cap[f"{i}.attn"][0].cpu().contiguous()
                for i, (p, wt) in zip(moe_layers, rec["picks"]):
                    tensors[f"{name}.s{step}.pick.{i}"] = p[0].cpu().contiguous()
                    tensors[f"{name}.s{step}.wts.{i}"] = wt[0].cpu().contiguous()
                for i, att in enumerate(rec["att"]):
                    if i in (first_full, first_swa):
                        tensors[f"{name}.s{step}.att.{i}"] = att[0].cpu().contiguous()
            if step == len(seq) - 1 and len(greedy) < a.steps:   # past the prompt: decode greedily
                greedy.append(nxt)
                if nxt not in eos:
                    seq.append(nxt)
            step += 1
        torch.cuda.synchronize()
        tensors[f"{name}.ids"] = torch.tensor(ids, dtype=torch.int32)
        tensors[f"{name}.greedy"] = torch.tensor(greedy, dtype=torch.int32)
        tensors[f"{name}.argmax"] = torch.tensor(argmax, dtype=torch.int32)
        tensors[f"{name}.logits"] = torch.stack(logits_kept).float()
        summary[name] = {"prompt": len(ids), "generated": len(greedy), "eos": bool(greedy and greedy[-1] in eos),
                         "seconds": round(time.perf_counter() - t, 1), "text": tok.decode(greedy)[:160]}
        print(f"[capture] rank {a.rank} {name}: {len(ids)} + {len(greedy)} tokens, "
              f"{summary[name]['seconds']}s: {summary[name]['text'][:80]!r}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / f"rank{a.rank}.safetensors"),
              metadata={"engine": "tensorfold-python mimo_v2", "tp": str(a.tp), "rank": str(a.rank),
                        "first_full": str(first_full), "first_swa": str(first_swa), "moe_layers": json.dumps(moe_layers)})
    (out / f"rank{a.rank}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    if rec_aot is not None:
        from triton.runtime.jit import JITFunction
        import gc

        rec_aot.dump(out / f"launches-r{a.rank}.json")
        jit = {}
        for fn in gc.get_objects():
            if isinstance(fn, JITFunction):
                jit[f"{fn.fn.__module__}.{fn.fn.__qualname__}"] = {
                    "params": [p.name for p in fn.params],
                    "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                    "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
        (out / f"jit-r{a.rank}.json").write_text(json.dumps(jit, indent=1, sort_keys=True) + "\n")
    if comm is not None:   # both ranks' greedy ids must agree
        mine = torch.cat([tensors[f"{n}.argmax"] for n in PROMPTS]).cuda().to(torch.int32)
        both = torch.empty((2, mine.numel()), dtype=torch.int32, device="cuda")
        getattr(comm, "nccl", comm).all_gather(mine, both)
        same = bool(torch.equal(both[0], both[1]))
        print(f"[capture] rank {a.rank}: argmax at every step equal on both ranks: {same}", flush=True)
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
