# PR #728: native kernels and shared EXL3 (plan)

Ash's review of #728 (26 Oct 2026) holds the family until it runs without the Python-captured Triton set and with
every launch registered in the kernel contract. Phase 1 (rebase onto main, the 8-bit cache removed) is pushed as
`12d9d644`. This is phase 2. Template for each slice: Muse Spark #1295 (`~/code/TensorFold-muse-1295`, commits
`5348282d` for the family on shared operators and `15c96f23` for the contract registrations and tests).

## Rules (upstream's)

- One arithmetic per operation: a row's bits never depend on batch size, window width or prompt chunking. Row-count
  switches stay only as performance choices with identical bits, tested at each edge (16/17 router rows, 255/256
  prompt tokens, 3,071-3,073 expert rows).
- Main's shared EXL3 operators define the reference (agreed with Ash on 24 Oct). Regenerate the oracle fixtures
  against them; keep a full-model gate against exllamav3 (per-text KL vs the noise floor, NLL band) plus GSM8K/MMLU.
- Every new kernel is registered with the kernel contract (gates 1-4 against float64 or a stated rule, launch
  records for gate 5), as `zig/src/core/contract/muse_cuda_records.zig` and `zig/tests/cuda/contract_muse_*.zig`.
- No lossy formats (fp16 KV only). No env knobs; constants at release values. Lean: files <= 600 lines, one-line
  comments <= 120 columns; `zig build/cuda.zig` must stay at 600 lines.

## Slices (each: kernel + contract registration + host tests, then a device check against the Triton output)

| # | Replaces (Triton) | Native kernel | Device check |
|---|---|---|---|
| 1 | `_rms`, `_add_rms` | norms in `mimo_v2/glue.cu` (fp32 sums, fp16 out) | bytes vs Triton on captured rows; contract gate vs float64 |
| 2 | `_silu_mul`, `_silu_mul_scale` | glue.cu | same |
| 3 | `_router`, `_topk` | one fixed-tile fp32 sigmoid router + bias + top-8 (no row-count tile switch) | routing ids equal on 1/16/17/128 rows |
| 4 | `_qkv`, `_rows`, `_merge` | decode attention: qkv + partial RoPE + cache write, chunked rows with sinks and sliding window, merge | KL vs noise floor (mimo_check), drafted == plain |
| 5 | `_prompt` | prompt attention, bit-equal to the decode path's rows | 255/256-token edge equal |
| 6 | `_gemm` + MiMo EXL3 copies | `cuda.exl3.Linear`, `exl3_experts`, main's prompt kernels (`Dense.prompt`) | prompt rows == decode rows; grouped == per-expert at 3,071-3,073 |

Then drop `TENSORFOLD_CUDA_KERNELS` from the family (it must start with its own kernels), delete
`cuda/mimo_exl3.zig` and the vendored `.cu` copies, regenerate fixtures, and gate:
`zig build kernel-contract` SUMMARY on GB10; lane contracts (drafted == plain, concurrent == solo); resumed == fresh
across 256 tokens; TP=2 == TP=1 on the 2.27 bpw pack; full-model KL/NLL vs exllamav3; GSM8K 500 + MMLU 228; decode
and prefill against the release numbers. Muse must be stopped for device runs (`docker stop muse-zig` on gx10,
`muse-zig-follow` on gx11; `docker start` them in an EXIT trap, as `tools/gate-rebase.sh`).

## Environment

- Branch: `~/code/TensorFold-mimo-rebase` `mimo-v2-main` -> fork `mimo-v2`. Check conflicts vs `up/main` before
  every push (`git merge-tree --write-tree --name-only up/main HEAD`), then `gh pr view 728 --json mergeable`.
- Host builds: `W=<worktree> ~/.hermes/cache/scratch/mbuild.sh <steps>` (approved Zig 0.17.0 toolchain in a capped
  ubuntu:26.04 container; no nvcc, no python). nvcc builds and GPU runs: `muse-zig-build:0.17.0` on gx10/gx11.
- Pack on both nodes: `~/models/mimo-v2.6/MiMo-V2.6-Flash-RL-exl3-3.0bpw` (2.27 bpw only on gx11). Captured
  Triton set (until slice 6 lands): `gx10:~/mimo/gate728/aot`.
