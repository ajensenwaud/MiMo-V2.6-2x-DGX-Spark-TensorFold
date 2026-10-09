#!/usr/bin/env python3
"""DFlash on the TensorFold mimo_v2 engine: acceptance, exactness and speed (one GPU or --tp 2).

  python3 tools/mimo_dflash_check.py /model [--tp 2 --rank R --master IP --port P] [--tokens 192] [--widths 3,4,6,8]

Per prompt (coding, prose, reasoning, code edit; chat template, thinking off):
  serial    greedy decode, one row a step on the graphed forward: the reference tokens and the baseline tok/s;
  accept    serial again, the drafter asked at every step: how many of its leading drafts the reference repeats;
  spec(w)   [pending, w - 1 drafts] verified a step, the greedy prefix kept: its tokens must equal the reference's
            (exactness), and its tok/s is the measured speed with the drafter.
Both ranks take the same steps from the same logits; rank 0 reports.
"""

import argparse
import json
import os
import time

import torch

from mimo_prompts import PROMPTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="10.100.128.1")
    ap.add_argument("--port", type=int, default=29571)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--widths", default="3,4,6,8")
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--json")
    a = ap.parse_args()

    from transformers import PreTrainedTokenizerFast

    from tensorfold.families.mimo_v2.cuda.forward import Chain, Model
    from tensorfold.families.mimo_v2.cuda.graphs import Graphs
    from tensorfold.families.mimo_v2.cuda.weights import load

    say = (lambda *x: print(*x, flush=True)) if a.rank == 0 else (lambda *x: None)
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.model, "tokenizer.json"))   # no repo code runs
    tok.chat_template = open(os.path.join(a.model, "chat_template.jinja")).read()
    torch.cuda.set_device(0)
    comm = None
    if a.tp == 2:
        from tensorfold.cuda.comm import open_comm

        comm = open_comm(a.rank, 2, a.master, a.port)
        if os.environ.get("TF_MIMO_COMM", "roce") == "roce":
            from tensorfold.cuda.roce import RoceComm

            comm = RoceComm(comm, a.rank, 2)
    model = Model(load(a.model, rank=a.rank, world=a.tp), a.context, 2, comm=comm)
    dr = model.attach_drafter(os.path.join(a.model, "dflash"))
    if comm is not None:
        comm.barrier()
        if hasattr(comm, "settle"):
            comm.settle()
    graphs = Graphs(model)
    dev = model.w.embed.device
    say(f"[dflash] block {dr.block}, taps {dr.taps}, window {dr.window}; "
        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on this rank")

    def one(t: torch.Tensor) -> int:
        return int(t.argmax())

    def draft(token: int, pos: int, slot: int) -> list[int]:
        return model.draft_runner([token], [pos], [slot])[0]        # graphed (TF_MIMO_DRAFT_GRAPHS=0: eager)

    widths = [int(x) for x in a.widths.split(",")]
    warm = list(range(1000, 1100))                         # captures every graph the timed runs use, untimed
    model.prefill(warm, slot=0)
    model.prefill(warm, slot=1)
    for w in [1, *widths]:
        for sl in (0, 1):
            graphs.forward([Chain(sl, 100, warm[:w])])
            draft(warm[0], 100, sl)
    report = {"tp": a.tp, "tokens": a.tokens, "prompts": {}}
    for name, text in PROMPTS.items():
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if not ids or not isinstance(ids[0], int):
            raise ValueError(f"tokenizer returned {type(ids[0])} ids")
        r = report["prompts"][name] = {"prompt_tokens": len(ids)}

        # serial reference + baseline speed (slot 0)
        logits = model.prefill(ids, slot=0)
        ref, p = [one(logits[0])], len(ids)
        torch.cuda.synchronize()
        t = time.perf_counter()
        while len(ref) < a.tokens:
            ref.append(one(graphs.forward([Chain(0, p, [ref[-1]])])[0]))
            p += 1
        torch.cuda.synchronize()
        r["serial_tok_s"] = round((len(ref) - 1) / (time.perf_counter() - t), 2)

        # acceptance: serial again, drafting at every step (slot 0, same positions; idempotent rewrites)
        model.prefill(ids, slot=0)
        p, acc, dms = len(ids), [], 0.0
        for i in range(a.tokens - dr.block):
            torch.cuda.synchronize()
            t = time.perf_counter()
            d = draft(ref[i], p + i, 0)
            torch.cuda.synchronize()
            dms += time.perf_counter() - t
            k = 0
            while k < len(d) and d[k] == ref[i + 1 + k]:
                k += 1
            acc.append(k)
            graphs.forward([Chain(0, p + i, [ref[i]])])
        r["accepted_mean"] = round(sum(acc) / len(acc), 3)
        r["accepted_hist"] = [acc.count(k) for k in range(dr.block)]
        r["draft_ms"] = round(dms / len(acc) * 1e3, 2)

        # speculative decode at fixed widths (slot 1): exactness and speed
        for w in widths:
            logits = model.prefill(ids, slot=1)
            out, p = [one(logits[0])], len(ids)
            steps = 0
            torch.cuda.synchronize()
            t = time.perf_counter()
            while len(out) < a.tokens:
                d = draft(out[-1], p, 1)[:w - 1]
                lg = graphs.forward([Chain(1, p, [out[-1], *d])])
                g = lg[:len(d) + 1].argmax(-1).tolist()
                k = 0
                while k < len(d) and d[k] == g[k]:
                    k += 1
                new = d[:k] + [g[k]]
                out += new
                p += len(new)
                steps += 1
            torch.cuda.synchronize()
            secs = time.perf_counter() - t
            out = out[:a.tokens]
            first = next((i for i, (x, y) in enumerate(zip(out, ref)) if x != y), None)
            r[f"spec_w{w}"] = {"tok_s": round((len(out) - 1) / secs, 2), "tokens_per_step": round((len(out) - 1) / steps, 3),
                               "exact": first is None, "first_divergence": first}
        r["text"] = tok.decode(ref[:64])
        say(f"[dflash] {name}: {json.dumps(r)}")
    report["graphs"] = {"captures": graphs.captures, "replays": graphs.replays, "eager": graphs.eager}
    say(f"[dflash] {json.dumps(report['graphs'])}")
    if a.json and a.rank == 0:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
