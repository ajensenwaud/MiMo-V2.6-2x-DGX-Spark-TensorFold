#!/usr/bin/env python3
"""exllamav3's own noise floor: logits of a text's first k positions from a forward over the first k tokens versus
the forward over the whole text (identical maths in a causal model; only the row count, hence kernel tiling and
routing paths, differs). Reports the same metrics as mimo_check.py so the two read side by side.

  python3 tools/mimo_noise.py /model /results/ref/ref.safetensors [--json out.json]
"""

import argparse
import json

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref")
    ap.add_argument("--json")
    a = ap.parse_args()

    from exllamav3 import Config, Model
    from safetensors import safe_open

    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    model.load()
    ref = safe_open(a.ref, framework="pt")
    params = {"attn_mode": "flash_attn_nc"}
    report = {}
    for name in sorted({k.split(".")[0] for k in ref.keys()}):
        ids = ref.get_tensor(f"{name}.ids").long()[None]
        full = ref.get_tensor(f"{name}.logits").float()
        n = ids.shape[1]
        rows = {}
        for k in (n // 2, n - 17):
            with torch.inference_mode():
                part = model.forward(ids[:, :k], params)[0].float().cpu()
            p = torch.log_softmax(full[:k], -1)
            q = torch.log_softmax(part[:, :full.shape[1]], -1)
            kl = (p.exp() * (p - q)).sum(-1)
            top = float((full[:k].argmax(-1) == part.argmax(-1)).float().mean())
            rows[f"prefix_{k}"] = {"kl": float(kl.mean()), "kl_max": float(kl.max()), "top1": top}
        report[name] = rows
        print(f"[noise] {name}: {json.dumps(rows)}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
