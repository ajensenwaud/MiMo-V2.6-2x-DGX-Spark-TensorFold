#!/usr/bin/env python3
"""Reference logits from exllamav3 for the quality gate: the same pack, the same token ids, every position.

  python3 tools/mimo_ref.py /model /out/ref.safetensors [--greedy 32]

For each text in tools/mimo_texts.json: its token ids (tokenized once, here, so every engine scores the same ids),
fp16 logits at every position, and a greedy continuation of ``--greedy`` tokens from full forwards (no cache, so
no cache code path is part of the reference). Runs inside the mimo-tf image on one GPU.
"""

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--texts", default=str(Path(__file__).with_name("mimo_texts.json")))
    ap.add_argument("--greedy", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    a = ap.parse_args()

    from exllamav3 import Config, Model, Tokenizer
    from safetensors.torch import save_file

    t0 = time.perf_counter()
    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    model.load()
    tok = Tokenizer(config)
    print(f"[ref] loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    texts = json.loads(Path(a.texts).read_text())
    out, meta = {}, {}
    params = {"attn_mode": "flash_attn_nc"}
    for name, text in texts.items():
        ids = tok.encode(text, add_bos=False, encode_special_tokens=True)[:, :a.max_len]
        with torch.inference_mode():
            logits = model.forward(ids, params)[0].float()             # [T, V]
            seq = ids
            gen = []
            for _ in range(a.greedy):
                nxt = int(model.forward(seq, params)[0, -1].float().argmax())
                gen.append(nxt)
                seq = torch.cat([seq, torch.tensor([[nxt]])], 1)
        out[f"{name}.ids"] = ids[0].to(torch.int32).contiguous()
        out[f"{name}.logits"] = logits.to(torch.float16).contiguous()
        out[f"{name}.greedy"] = torch.tensor(gen, dtype=torch.int32)
        meta[name] = {"tokens": int(ids.shape[1]), "greedy_text": tok.decode(torch.tensor([gen]))[0]}
        print(f"[ref] {name}: {ids.shape[1]} tokens; greedy: {meta[name]['greedy_text'][:80]!r}", flush=True)
    save_file(out, a.out, metadata={"engine": "exllamav3", "model": a.model, "texts": json.dumps(meta)})
    print(f"[ref] wrote {a.out} in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
