#!/usr/bin/env python3
"""Census of an EXL3 MiMo-V2.6 pack from safetensors headers only (stdlib; never reads tensor data).

  python3 tools/mimo_inspect.py <pack_dir> [--json out.json]

Per module class: bits per weight (from the trellis' last dim: 16 x K words per 16x16 tile), bytes, tensor counts;
plus suffix census (each EXL3 matrix must carry exactly one trellis/suh/svh/mul1 or mcg) and a TP=2 per-rank budget.
"""

import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "I8": 1, "U8": 1, "F8_E4M3": 1, "I64": 8}
EXL3_PARTS = {"trellis", "suh", "svh", "mul1", "mcg", "su", "sv"}


def headers(pack: Path) -> dict[str, dict]:
    index = json.loads((pack / "model.safetensors.index.json").read_text())
    out = {}
    for shard in sorted(set(index["weight_map"].values())):
        with open(pack / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            out[k] = {**v, "shard": shard, "nbytes": v["data_offsets"][1] - v["data_offsets"][0]}
    missing = set(index["weight_map"]) - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} indexed tensors not in any shard header, e.g. {sorted(missing)[:3]}")
    return out


def klass(name: str) -> str:
    if name.startswith(("visual.", "model.visual.")):
        return "vision"
    if name.startswith("model.mtp."):
        return "mtp"
    if ".mlp.experts." in name:
        return "routed_experts"
    if ".mlp.gate." in name:
        return "router"
    if re.search(r"\.self_attn\.(q|k|v|o|qkv)_proj\.", name):
        return "attention"
    if ".self_attn.attention_sink_bias" in name:
        return "sinks"
    if re.search(r"layers\.\d+\.mlp\.(gate|up|down)_proj\.", name):
        return "dense_mlp"
    if name.startswith("lm_head."):
        return "lm_head"
    if "embed_tokens" in name:
        return "embeddings"
    if "norm" in name:
        return "norms"
    return "other"


def main() -> None:
    pack = Path(sys.argv[1])
    h = headers(pack)
    by_class = defaultdict(lambda: {"tensors": 0, "bytes": 0, "bits": Counter()})
    mats = defaultdict(set)                       # matrix prefix -> suffixes
    expert_bits = defaultdict(Counter)            # layer -> Counter(bits)
    for name, t in h.items():
        c = klass(name)
        by_class[c]["tensors"] += 1
        by_class[c]["bytes"] += t["nbytes"]
        stem, _, suffix = name.rpartition(".")
        if suffix in EXL3_PARTS:
            mats[stem].add(suffix)
        if suffix == "trellis":
            bits = t["shape"][-1] / 16
            by_class[c]["bits"][bits] += 1
            m = re.search(r"layers\.(\d+)\.mlp\.experts\.", name)
            if m and c == "routed_experts":
                expert_bits[int(m.group(1))][bits] += 1
    sig = Counter(tuple(sorted(s)) for s in mats.values())
    total = sum(t["nbytes"] for t in h.values())
    text = total - by_class["vision"]["bytes"]
    # TP=2: experts, attention, dense MLP, lm_head split across ranks; embeddings, norms, router, sinks, MTP replicated
    split = sum(by_class[c]["bytes"] for c in ("routed_experts", "attention", "dense_mlp", "lm_head"))
    per_rank = split / 2 + (text - split)
    gib = 2 ** 30
    report = {
        "pack": str(pack), "tensors": len(h), "bytes_total": total, "bytes_text": text,
        "classes": {c: {"tensors": v["tensors"], "GiB": round(v["bytes"] / gib, 3),
                        "bits": {str(k): n for k, n in sorted(v["bits"].items())}} for c, v in sorted(by_class.items())},
        "exl3_matrices": len(mats), "exl3_suffix_signatures": {"+".join(k): n for k, n in sig.items()},
        "expert_bits_by_layer": {str(k): {str(b): n for b, n in sorted(v.items())} for k, v in sorted(expert_bits.items())},
        "tp2_text_GiB_per_rank": round(per_rank / gib, 2), "tp1_text_GiB": round(text / gib, 2),
    }
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(report, indent=1))
    print(f"{pack.name}: {len(h)} tensors, {total / gib:.2f} GiB ({text / gib:.2f} GiB text); "
          f"TP=2 text per rank ~{per_rank / gib:.2f} GiB")
    for c, v in report["classes"].items():
        print(f"  {c:15s} {v['tensors']:7d} tensors {v['GiB']:8.2f} GiB  bits {v['bits']}")
    print(f"  EXL3 matrices {len(mats)}; suffix signatures {report['exl3_suffix_signatures']}")
    runs, prev = [], None
    for layer, cnt in sorted(expert_bits.items()):
        key = tuple(sorted(cnt.items()))
        if key != prev:
            runs.append([layer, layer, dict(cnt)])
            prev = key
        else:
            runs[-1][1] = layer
    for a, b, cnt in runs:
        print(f"  experts L{a}-L{b}: {cnt}")


if __name__ == "__main__":
    main()
