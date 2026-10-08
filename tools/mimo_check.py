#!/usr/bin/env python3
"""Score the TensorFold mimo_v2 engine against exllamav3's reference logits (tools/mimo_ref.py), one GPU.

  python3 tools/mimo_check.py /model /out/ref-2.27.safetensors [--json out.json]

Per text: (1) prompt path, every position: KL(ref || engine) and top-1 agreement; (2) decode path: the first half
as a prompt, the rest fed one row at a time (decode kernels), same metrics on those rows; (3) greedy 32 tokens from
the full text via decode steps, compared with the reference's greedy continuation (matching prefix length).
"""

import argparse
import json
import time

import torch


def kl_top1(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float, float]:
    """mean KL(ref || got), max KL, top-1 agreement over rows (fp32, the reference's vocab width)."""

    v = ref.shape[1]
    lp = torch.log_softmax(ref.float(), -1)
    lq = torch.log_softmax(got[:, :v].float(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top = (ref.float().argmax(-1) == got[:, :v].argmax(-1)).float().mean()
    return float(kl.mean()), float(kl.max()), float(top)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref")
    ap.add_argument("--json")
    ap.add_argument("--context", type=int, default=4096)
    a = ap.parse_args()

    from safetensors import safe_open

    from tensorfold.families.mimo_v2.cuda.forward import Chain, Model
    from tensorfold.families.mimo_v2.cuda.weights import load

    torch.cuda.set_device(0)
    t0 = time.perf_counter()
    w = load(a.model)
    model = Model(w, a.context, 1)
    print(f"[check] loaded in {time.perf_counter() - t0:.0f}s; "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated", flush=True)
    ref = safe_open(a.ref, framework="pt")
    names = sorted({k.split(".")[0] for k in ref.keys()})
    report = {}
    for name in names:
        ids = ref.get_tensor(f"{name}.ids").tolist()
        rl = ref.get_tensor(f"{name}.logits").cuda()
        greedy_ref = ref.get_tensor(f"{name}.greedy").tolist()
        n = len(ids)
        t = time.perf_counter()
        got = model.forward([Chain(0, 0, ids)], prompt=True, rows=range(n))
        torch.cuda.synchronize()
        tp = time.perf_counter() - t
        p_kl, p_max, p_top = kl_top1(rl, got)
        v = rl.shape[1]
        lp = torch.log_softmax(rl.float(), -1)
        by_pos = (lp.exp() * (lp - torch.log_softmax(got[:, :v].float(), -1))).sum(-1)
        worst = torch.topk(by_pos, min(8, n))
        worst = [(int(i), round(float(k), 3), int(ids[i])) for k, i in zip(worst.values, worst.indices)]
        half = n // 2
        model.forward([Chain(0, 0, ids[:half])], prompt=True)
        dec = []
        for i in range(half, n):
            dec.append(model.forward([Chain(0, i, [ids[i]])], prompt=False))
        dec = torch.cat(dec)
        d_kl, d_max, d_top = kl_top1(rl[half:], dec)
        logits = model.forward([Chain(0, 0, ids)], prompt=True)
        gen, t = [], time.perf_counter()
        for i in range(len(greedy_ref)):
            nxt = int(logits[0].argmax())
            gen.append(nxt)
            logits = model.forward([Chain(0, n + i, [nxt])], prompt=False)
        torch.cuda.synchronize()
        td = time.perf_counter() - t
        match = next((i for i, (x, y) in enumerate(zip(gen, greedy_ref)) if x != y), len(gen))
        report[name] = {"tokens": n, "prompt_kl": p_kl, "prompt_kl_max": p_max, "prompt_top1": p_top,
                        "decode_kl": d_kl, "decode_kl_max": d_max, "decode_top1": d_top,
                        "greedy_match": f"{match}/{len(gen)}", "prompt_s": round(tp, 3),
                        "eager_decode_tok_s": round(len(gen) / td, 2),
                        "kl_first_half": float(by_pos[:n // 2].mean()), "kl_second_half": float(by_pos[n // 2:].mean()),
                        "worst_positions": worst}
        print(f"[check] {name}: {json.dumps(report[name])}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
