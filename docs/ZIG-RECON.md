# Zig port: recon (2026-10-11)

Facts for porting the MiMo-V2.6-Flash recipe from TensorFold's Python engine (0.6.6) to the Zig engine, following
the approach Jay Leaton used for DeepSeek-V4.1-Flash. Read with [ZIG-PLAN.md](ZIG-PLAN.md). Read-only recon: no GPU
was used and no engine code was changed.

## 1. Starting point: the Python recipe (measured)

Release image `mimo-tf:4df8c51482`, 3.0 bpw pack, DFlash, TP=2, `:8891`.

| Measurement | Value | Source |
|---|---:|---|
| Code, 1 stream, decode tok/s | 83.0 | spark-bench full, 2026-10-10 23:36, idle server |
| Chat / agent / report, 1 stream | 52.9 / 60.6 / 50.2 | same |
| 4 streams, aggregate decode tok/s (code / chat) | 91.4 / 85.1 | same |
| Prefill tok/s at 4K / 16K / 64K / 128K (cold) | 764 / 1,179 / 1,141 / 1,022 | same |
| Decode at 64K / 128K context | 41.8 / 33.4 | same |
| GSM8K (n=500) / MMLU-Pro (n=500) | 96.0% / 73.4% | README, `results/evals/` |
| One decode row (4.0 pack) | 24.5 ms: experts 10.6, dense + head 9.8, exchanges 2.0, attention/routing/norms 1.9 | README |
| One 16K prompt chunk (4.0 pack) | 13.1 s: prompt GEMM 4.9, exchanges 2.7, weight decoding 1.6, rest 3.9 | README |

The Python family: `src/tensorfold/families/mimo_v2/cuda/` from patches 0024-0030, about 2,000 added lines.

| File | Lines (approx.) | Triton kernels | Leans on |
|---|---:|---:|---|
| `attention.py` | 236 | 6 | sliding-window ring + sinks, full attention, partial RoPE |
| `decoder.py` | 310 | 0 | `cuda.scheduler`, `streams`, `sampling`, qwen3_5 copy drafting |
| `dflash.py` | 200 | 0 | target taps at layers 0/11/23/35/47, 8-token draft block |
| `forward.py` | 206 | 0 | `exl3.prefill`, layer loop |
| `moe.py` | 127 | 2 | `exl3.experts` (grouped mul1 kernel), sigmoid + bias top-8 of 256 |
| `ops.py` | 76 | 3 | norms, router glue |
| `weights.py` | 203 | 0 | `exl3.linear.Exl3Linear`, `direct_read.SafeTensors` |
| `graphs.py`, `tp.py`, `engine.py` | 275 | 0 | CUDA graphs keyed by row count, TP mirror, `RoceComm` |

Python-only dependencies to replace in Zig: torch tensors and allocator, `torch.cuda.CUDAGraph`, a handful of torch
ops (`softmax`, `einsum`, `matmul`, `bincount`, `argsort`, `cat`, `rsqrt`), the exllamav3-derived CUDA extension
behind `exl3.linear`/`experts`/`prefill`, and the Python scheduler, sampler and server.

MiMo features with no Zig counterpart anywhere: hybrid attention (39 sliding-window layers of 128 keys with sink
logits, 9 full layers), q/k 192 and v 128 head dims, partial NEOX RoPE (64 of 192), GQA 64 q / 4 kv heads,
sigmoid + correction-bias top-8 routing over 256 experts, a dense layer 0 MLP, the DFlash drafter, the
sliding-window ring KV, and MiMo's chat template and tool markup.

## 2. TensorFold today

- `python-0.6` @ `ed78d6f` (0.6.6) is the head of the Python line; the recipe is on it. The line is frozen.
- `main` is the Zig engine: v1.0.5 (2026-10-10) plus unreleased work. On a GB10 it serves Nemotron only. No family
  loads EXL3 natively (#496 is a reference decoder). There is no CUDA tensor parallelism upstream. Families:
  flashnext, glm, kimi_k3, nemotron, qwen27, qwen3_5.
- Jay is upstreaming his DeepSeek port as a 14-PR stack. #634-#638 (stack 1-5) are open: server JSON fix, keyed graph
  cache + device arena (#635), TP across CUDA processes over NCCL/mailbox/RoCE (#636), sessions (#637), grammar
  (#638). Stack 6-14, which include the EXL3 kernels, are not submitted yet.

## 3. Jay's DeepSeek-V4.1 Zig port

Repo `jayleaton/deepseek-v41-tensorfold-spark` @ `199ea20` (2026-10-10), Apache-2.0. `engine/zig` is TensorFold
1.0.2 (`f8fe17d`) plus the DeepSeek family (snapshot `8e96f99`). Upstream `main` is 368 commits and 300 files ahead
of that base.

**Method (what "his approach" means):**
1. He built the family on the Python engine first (`dsv41-060`, milestones M1-M3, tuning windows G1-G19).
2. He then wrote the Zig family so that every launch mirrors the Python binding it replaces: same grid, block,
   shared memory and instance, so the same SASS runs on the same work. The header of `kernels_exl3.zig` states it.
3. Triton kernels are not rewritten: they are compiled ahead of time from the Python reference
   (`tools/zig/triton_fill.py`, `aot-needs`), per rank, and loaded by `zig/src/cuda/triton.zig`.
4. Hand CUDA kernels live in `zig/kernels/cuda/<family>/*.cu`, built into sm_121 fatbins (`build-kit.sh`,
   `kit-fatbins.sh`).
5. Python captures are the oracle (`dsv41_m1_capture.py`, `m1_check.zig`): greedy reference parity before drafting,
   then drafted == plain, concurrent == solo, resumed == fresh, prompt seams, cancellation, TP failure.
6. Every new path is default-off behind an environment switch; the serving profile opts in.

**Reusable as is (generic):** `zig/src/cuda` (driver, module, AOT/Triton loader, graph cache, device arena, NCCL,
cuBLASLt, streams), `zig/src/fabric` (verbs/RDMA, mailbox, collectives, TP bootstrap), `zig/src/sessions`,
`zig/src/server` (lane, OpenAI/Anthropic routes, tool calls, drain, keepalive), `zig/kernels/cuda/torch_ops`
(argmax, top-k, norm + rotary, pointwise, nucleus, log-probs, indexing, movement).

**Reusable with adaptation (from the DeepSeek family):** EXL3 kernels for the mul1 codebook, which our packs also
use (`x3ld` decode experts, `x3pf`/`x3gm` prefill experts, upstream's grouped kernel, `linear.cu`, `lin2.cu`, the
fused-trellis prefill GEMM `pfdense_k12.cu`), the L2 prefetch (`l2pf.cu`), the router-as-GEMV pattern, the
draft-window/row/calibration machinery (`dspark_rows`, `calib_gpu`), the paged-KV pieces, `load.zig`/`prepare.zig`.
DSpark's drafter forward itself is DeepSeek-specific.

**Measured gain, Zig over Python, his model, one night:** 1 stream +1.0-2.6%, 4 streams +5.6-8.6%, long prompts
+10%, 131K decode +3.3%. Replies identical.

**Caveats:** the export carries 5,949 lint findings against upstream's zero-added convention; device gates were not
rerun for the export; serving needs per-rank Triton AOT assets generated with Python, Torch, Triton 3.7.1 and
ptxas 13.3.73; the DeepSeek family is about 60,000 lines including tests and benches.

## 4. Toolchain and cluster

- The recipe's base image (GLM TensorFold 0.6.0, pinned digest) already has nvcc 13.3.73, ptxas 13.3, Triton 3.7.1
  and torch 2.13 nv26.07: exactly the AOT fill requirements. Zig 0.17.0 is not present anywhere; it would be
  added inside a build image, never on a host.
- gx10: 177 GB free, MiMo serving (`mimo-tf`) + `sparkDash`. gx11: 2.3 TB free, `mimo-tf` + `meter-proxy`.
  Both nodes hold the 3.0 pack at `~/models/mimo-v2.6/`. MemAvailable while serving: ~4.3 GB a node, so no GPU work
  runs beside the Python serve.
- Hermes' desktop session uses the MiMo serve; taking it down for GPU windows takes that model away.

## 5. TensorFold v1.0.5 as the base (decision D1)

`v1.0.5` = `bec00ae` (2026-10-10). Already upstream: `zig/src/fabric` (verbs/RDMA, mailbox, collectives,
sendrecv), `zig/src/cluster`, `cuda/aot.zig`, `triton.zig`, `nccl.zig`, `qlinear.zig` (one weight view per format:
MLX affine 4-bit, block-FP8), `grouped.zig` (shared routed-expert plan), `experts.zig`, and `core/exl3_format.zig`
(EXL3 trellis reference decoder, no CUDA model path). Missing for MiMo: the keyed graph cache and device arena (#635),
CUDA tensor parallelism / TP bootstrap (#636), EXL3 CUDA kernels (in Jay's export, not yet submitted), paged-KV
sessions (#637). #635-#638 are open against `main` and all four report merge conflicts, so they are rebased by us
(gate Z0B). #634 merges cleanly.
