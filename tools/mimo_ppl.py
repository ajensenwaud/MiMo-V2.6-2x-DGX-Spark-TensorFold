#!/usr/bin/env python3
"""Wikitext-2 perplexity on the TensorFold mimo_v2 engine, built exactly as exllamav3's eval/ppl.py builds it (the
2.27 pack's published 5.4003): the test split joined with "\\n\\n", exllamav3's tokenizer, the first ``--rows``
non-overlapping rows of ``--len`` tokens; perplexity = exp(mean NLL of each row's next tokens within the row).

  python3 tools/mimo_ppl.py /model [--rows 64 --len 2048] [--tp 2 --rank R --master IP --port P] [--json out]

Both ranks tokenise and run the same forwards; rank 0 reports. Per-row NLL is kept so packs compare row by row.
"""

import argparse
import json
import math
import os
import time

import torch


def wiki_tokens(model_dir: str) -> torch.Tensor:
    from datasets import load_dataset
    from exllamav3 import Config, Tokenizer

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return Tokenizer(Config.from_directory(model_dir)).encode(text)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="10.100.128.1")
    ap.add_argument("--port", type=int, default=29571)
    ap.add_argument("--json")
    a = ap.parse_args()

    from tensorfold.families.mimo_v2.cuda.forward import Chain, Model
    from tensorfold.families.mimo_v2.cuda.weights import load

    say = (lambda *x: print(*x, flush=True)) if a.rank == 0 else (lambda *x: None)
    tokens = wiki_tokens(a.model)
    rows = [tokens[s:s + a.len].tolist() for s in range(0, tokens.numel() - a.len, a.len)][:a.rows]
    say(f"[ppl] {tokens.numel()} tokens; {len(rows)} rows of {a.len}")
    torch.cuda.set_device(0)
    comm = None
    if a.tp == 2:
        from tensorfold.cuda.comm import open_comm

        comm = open_comm(a.rank, 2, a.master, a.port)
        if os.environ.get("TF_MIMO_COMM", "roce") == "roce":
            from tensorfold.cuda.roce import RoceComm

            comm = RoceComm(comm, a.rank, 2)
    w = load(a.model, rank=a.rank, world=a.tp)
    torch.cuda.empty_cache()
    model = Model(w, max(4096, a.len), 1, comm=comm)
    if comm is not None:
        comm.barrier()
        if hasattr(comm, "settle"):
            comm.settle()
    nll_rows, t0 = [], time.perf_counter()
    for i, ids in enumerate(rows):
        logits = model.forward([Chain(0, 0, ids)], prompt=True, rows=range(len(ids) - 1))
        target = torch.tensor(ids[1:], device=logits.device)
        nll = -torch.log_softmax(logits.float(), -1).gather(1, target[:, None]).squeeze(1)
        nll_rows.append(float(nll.mean()))
        if i % 16 == 15:
            say(f"[ppl] {i + 1}/{len(rows)} rows, running ppl {math.exp(sum(nll_rows) / len(nll_rows)):.4f}")
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    out = {"model": a.model, "tp": a.tp, "rows": len(rows), "len": a.len,
           "ppl": math.exp(sum(nll_rows) / len(nll_rows)), "nll_rows": nll_rows,
           "prefill_tok_s": round(len(rows) * a.len / secs, 1)}
    say(f"[ppl] {json.dumps({k: v for k, v in out.items() if k != 'nll_rows'})}")
    if a.json and a.rank == 0:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
