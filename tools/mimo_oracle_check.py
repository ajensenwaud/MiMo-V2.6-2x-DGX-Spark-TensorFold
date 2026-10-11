"""Z1.1 checks on the captured answer key: run-to-run determinism and rank agreement.

  python3 oracle_check.py RUN1/rank0.safetensors RUN2/rank0.safetensors RUN2/rank1.safetensors
"""
import sys

from safetensors import safe_open

a, b, r1 = (safe_open(p, framework="pt") for p in sys.argv[1:4])
shared = sorted(set(a.keys()) & set(b.keys()))
diff = [k for k in shared if not a.get_tensor(k).equal(b.get_tensor(k))]
only = sorted(set(b.keys()) - set(a.keys()))
print(f"run1 vs run2 (rank 0): {len(shared)} shared tensors, {len(diff)} differ; {len(only)} new in run 2")
for k in diff[:10]:
    print("  DIFF", k)
# every tensor both ranks hold the same way: residuals, logits, picks, weights, tokens (not .att: a rank's own heads)
same = [k for k in sorted(b.keys()) if ".att." not in k]
rd = [k for k in same if not b.get_tensor(k).equal(r1.get_tensor(k))]
print(f"rank 0 vs rank 1 (run 2): {len(same)} rank-shared tensors, {len(rd)} differ")
for k in rd[:10]:
    print("  DIFF", k)
sys.exit(1 if diff or rd else 0)
