#!/usr/bin/env python3
"""Layer-by-layer residual comparison, exllamav3 vs the TensorFold mimo_v2 engine, on one text of a reference file.

  python3 tools/mimo_layers.py exl3 /model /results/ref/ref.safetensors agent /results/ref/layers-exl3.safetensors
  python3 tools/mimo_layers.py tf   /model /results/ref/ref.safetensors agent /results/ref/layers-exl3.safetensors

exl3: run exllamav3 on the text's ids, save every block's output residual [T, D]. tf: run ours (prompt path), then
print per layer the relative error of our residual against exllamav3's, overall and at the worst positions.
"""

import sys

import torch


def main() -> None:
    mode, model_dir, ref, name, path = sys.argv[1:6]
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    ids = safe_open(ref, framework="pt").get_tensor(f"{name}.ids")
    if mode == "exl3":
        from exllamav3 import Config, Model
        from exllamav3.modules.transformer import TransformerBlock

        cap = {}
        orig = TransformerBlock.forward

        def fwd(self, x, params, out_dtype=None):
            y = orig(self, x, params, out_dtype)
            if self.layer_idx is not None:
                cap[str(self.layer_idx)] = y[0].float().cpu().clone()
            return y

        TransformerBlock.forward = fwd
        config = Config.from_directory(model_dir)
        model = Model.from_config(config)
        model.load()
        with torch.inference_mode():
            model.forward(ids[None].long(), {"attn_mode": "flash_attn_nc"})
        save_file(cap, path)
        print(f"saved {len(cap)} layers of {tuple(next(iter(cap.values())).shape)}")
        return

    from tensorfold.families.mimo_v2.cuda.forward import Chain, Model
    from tensorfold.families.mimo_v2.cuda.weights import load

    torch.cuda.set_device(0)
    model = Model(load(model_dir), 4096, 1)
    cap = {}
    ref_cap = load_file(path)
    local = mode == "tfl"          # teacher-forced: each layer from exllamav3's previous residual
    model.forward([Chain(0, 0, ids.tolist())], prompt=True, capture=cap, teacher=ref_cap if local else None)
    watch = [int(p) for p in sys.argv[6].split(",")] if len(sys.argv) > 6 else []
    for i in range(len(model.w.layers)):
        a = cap[str(i)].float().cpu()
        b = ref_cap[str(i)]
        err = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)
        if i > 0:
            d = (b - ref_cap[str(i - 1)]).norm(dim=-1).clamp_min(1e-6)
            derr = (a - b).norm(dim=-1) / d
            at_d = f" | vs layer delta mean {derr.mean():.4f} max {derr.max():.4f} at {int(derr.argmax())}"
        else:
            at_d = ""
        top = torch.topk(err, 4)
        at = " ".join(f"p{p}={float(err[p]):.4f}" for p in watch)
        print(f"layer {i:2d} {'full' if model.cfg.full[i] else 'swa '} rel err mean {err.mean():.4f} "
              f"max {err.max():.4f} at {[int(p) for p in top.indices]} | |res| {b.norm(dim=-1).mean():.1f} "
              f"max {b.norm(dim=-1).max():.1f} | {at}{at_d}", flush=True)
    if watch:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        rl = safe_open(ref, framework="pt").get_tensor(f"{name}.logits").float()
        got = model.forward([Chain(0, 0, ids.tolist())], prompt=True, rows=watch).float().cpu()
        for j, p in enumerate(watch):
            ctx = tok.decode(ids[max(0, p - 6):p + 1].tolist())
            r = torch.softmax(rl[p], -1).topk(4)
            g = torch.softmax(got[j][:rl.shape[1]], -1).topk(4)
            fmt = lambda t: [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(t.values, t.indices)]
            print(f"pos {p} after {ctx!r}: exl3 {fmt(r)} | ours {fmt(g)}", flush=True)


if __name__ == "__main__":
    main()
