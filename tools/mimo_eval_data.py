#!/usr/bin/env python3
"""Export the evaluation items once (run in the image: it has ``datasets``): GSM8K test and MMLU-Pro test, ``--n``
items each sampled with ``random.Random(seed)`` (benthecarman's pack evals: n=500, seed 1234; his exact sampler is
not published, so the subsets are comparable in size and kind, not item for item).

  python3 tools/mimo_eval_data.py /results/evals [--n 500 --seed 1234]
"""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    from datasets import load_dataset

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g = load_dataset("openai/gsm8k", "main", split="test")
    pick = sorted(random.Random(a.seed).sample(range(len(g)), a.n))
    items = [{"id": int(i), "question": g[i]["question"], "gold": g[i]["answer"].split("####")[-1].strip().replace(",", "")}
             for i in pick]
    (out / "items-gsm8k.json").write_text(json.dumps(items, indent=0))
    m = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    pick = sorted(random.Random(a.seed).sample(range(len(m)), a.n))
    items = [{"id": int(i), "question": m[i]["question"], "options": list(m[i]["options"]), "gold": m[i]["answer"],
              "category": m[i]["category"]} for i in pick]
    (out / "items-mmlu_pro.json").write_text(json.dumps(items, indent=0))
    print(f"wrote {a.n} GSM8K and {a.n} MMLU-Pro items to {out}")


if __name__ == "__main__":
    main()
