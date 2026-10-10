# MiMo-V2.6-Flash on 2x DGX Spark (GB10) with TensorFold

Xiaomi's MiMo-V2.6-Flash-RL (309B MoE, 15B active; 48 layers, 39 of them 128-token sliding-window attention with
sink logits, 9 full) served across two DGX Sparks with tensor parallelism, from our own EXL3 quantisation, on
TensorFold's Python engine (`python-0.6`) plus this recipe's patches, with Xiaomi's DFlash drafter.

Status: measured, text only (no vision, no audio). Default: the 3.0 bpw pack with DFlash.

## What this recipe adds to TensorFold

TensorFold had no MiMo family. Everything below is new, in `src/tensorfold/families/mimo_v2/` (patches 0024-0030; 0001-0023
are the Kolibri-1 recipe's base: two-rank serving, RoCE all-gathers, graphs, the KL tooling).

| Change | Where | Effect on bits |
|---|---|---|
| MiMo-V2 on CUDA from EXL3 packs: q/k 192, v 128 (read at the pack's 192-wide padding), partial NEOX RoPE (64 of 192), sink logits on the sliding layers, sigmoid + correction-bias top-8 routing on fp32 logits (as the model's own code), layer 47's interm_div compensation | `weights.py`, `attention.py`, `moe.py`, `forward.py` | matches exllamav3 within exllamav3's own run-to-run noise (below) |
| Two ranks: half the query/KV heads, half of every expert's width, half the vocabulary, all cut on EXL3's 128-wide Hadamard blocks; partials all-gathered and summed in rank order | `weights.py`, `forward.py`, `tp.py` | ranks bit-equal to each other |
| Decode as CUDA graphs keyed by (row count, key bucket): any split of rows into streams replays the same graph | `graphs.py` | none: graphs equal eager bit for bit (36/36 cases a pack) |
| DFlash drafter: the target's residual after layers 0/11/23/35/47 feeds the drafter's keys inside every target forward on both ranks (no extra exchange); a block of 8 drafted in one graphed pass through the target's head; window widths chosen jointly across streams from a measured streams x rows cost grid | `dflash.py`, `decoder.py`, `tp.py` | none: verified drafting; replies byte-identical to plain decode |
| Prompt rows' experts on the EXL3 prompt GEMM (each expert decoded once a chunk, expert-sorted contiguous slices) instead of the decode kernel, which re-decodes per row group; 16K prompt chunks | `moe.py`, `forward.py` | prompt rows only; wikitext perplexity unchanged (4.9803 vs 4.9794) |
| Input-side TP cuts copied, not viewed (a view kept every full down_proj alive: 57.1 -> 45.4 GiB a rank on 2.27) | `weights.py` | none |

## Run it

Two GB10 Sparks joined by their CX7 ports, Docker with the NVIDIA runtime on both, SSH from the head (gx10, rank 0) to
the worker (gx11, rank 1). Ports: 8891 (API), 29571 (rendezvous).

```bash
scripts/prepare.sh                 # TensorFold python-0.6 @ ed78d6f + patches/, the image built on the head and loaded on the worker
scripts/quantise.sh 3.0            # on the worker: our 3.0 bpw pack from the RL source (~17 h on one GB10); 4.0 likewise
PACK=3.0 scripts/sync_pack.sh      # the pack to the head over CX7, sha256-checked
./start.sh                         # rank 1 on the worker, rank 0 here; OpenAI API on :8891, model id MiMo-V2.6-Flash
./stop.sh
```

Settings in `scripts/config.sh` (override in `scripts/local.sh`): `PACK=3.0|4.0|2.27`, `DRAFTER=1|0`, `PARALLEL=4`
streams, `TP=2`. A request can send `"draft": false` for plain decode. Pins: source `XiaomiMiMo/MiMo-V2.6-Flash-RL`
@ 5711b268, exllamav3 v1.6.0, release image `mimo-tf:4df8c51482` (sha256:93c36168635a).

## Packs

Same RL source and exllamav3 settings (`-hq -cr 250 -cc 2048`, mul1 codebook, head 8 bits, MTP 4 bits), only the
bitrate differs; each carries the DFlash drafter. 2.27 is benthecarman's published pack, the baseline.

| Pack | Size | Weights a Spark (TP=2) | Plain decode, 1 stream (engine, 9K context, graphed) |
|---|---|---|---|
| 2.27 bpw | 89 GB | 45.4 GiB | 54.6 tok/s |
| **3.0 bpw (default)** | 113 GB | 60.4 GiB | 46.3 tok/s (served: 45.4) |
| 4.0 bpw | 149 GB | 76.8 GiB | 40.7 tok/s |

## Quality (served, TP=2, the same 500 items a task for every pack, thinking off, greedy)

| Pack | GSM8K | MMLU-Pro | Wikitext-2 perplexity (64 x 2048) |
|---|---|---|---|
| 2.27 | 95.4% | 70.0% | 5.402 (exllamav3, published: 5.400) |
| **3.0** | **96.0%** | **73.4%** | 5.069 |
| 4.0 | 96.6% | 73.6% | 4.979 |

Paired on identical items: MMLU-Pro 3.0 vs 2.27 +3.4 points (42 items only 3.0 right vs 25, McNemar p = 0.04); 4.0 vs
3.0 +0.2 (26 vs 25, p = 0.89). GSM8K differences are within noise (p >= 0.11). So 3.0 recovers what 2.27 loses, and
4.0 adds nothing measurable while 3.0 decodes 14% faster. Perplexity separates all three (3.0 better than 2.27 on 64/64 rows, 4.0
better than 3.0 on 57/64), a smaller effect than the tasks can resolve. Raw replies: `results/evals/`.

Engine parity: on the 2.27 pack our engine's KL against exllamav3's logits equals exllamav3's own KL against itself
when only the row count changes (agent 0.138 vs 0.128, code 0.034 vs 0.020, prose 0.0038 vs 0.0035); true-token NLL
within 0.1-1.5% of exllamav3's. Exact-token parity across engines does not exist for this model: near-tied top-8
routing flips under any last-bit change, exllamav3 included.

Reported, not measured here (benthecarman, on his own 500-item samples, exllamav3, 2.27 pack): GSM8K 95.2% (FP8
model: 96.4%), MMLU-Pro 73.8% (FP8: 76.4%). His samples differ from ours, so only the gaps within each set compare.

Not run: HumanEval+/MBPP+ (they execute model-written code), GPQA-Diamond (gated), anything with thinking on.

## Measured speed (3.0 pack, release image, `results/release-3.0/`)

Decode tok/s = completion tokens / the server's own decode seconds, median of 3; 512-token replies, greedy, thinking
off; prompts in `tools/mimo_prompts.py`.

| Task | Plain decode | DFlash | Speed-up |
|---|---|---|---|
| coding | 45.4 | 77.6 | 1.71x |
| code edit | 44.7 | 80.8 | 1.81x |
| reasoning | 45.3 | 72.8 | 1.61x |
| prose | 45.5 | 46.0 | 1.01x (the window chooser stops drafting) |
| 4 streams, one task each (aggregate) | 71.5 | 77.4 | 1.08x |

Every DFlash reply was byte-identical to the plain one. With several streams the gain shrinks: every drafted row
routes to its own experts, and rows from different streams share few (the cost grid in `results/tp2/grid-cost-4.0.json`).

| Prompt | 7.3K | 29K | 58K |
|---|---|---|---|
| Prefill tok/s | 1018 | 1231 | 1174 |
| Decode after it (DFlash, summarising) tok/s | 51.8 | 48.1 | 48.8 |

Context: 4 streams of 555K tokens on the 3.0 pack (MiMo keeps full keys in 9 of 48 layers only).

Reported, not measured here (benthecarman, one Spark, 2.27 pack, exllamav3, his prompts): plain decode 31.0-31.5
tok/s; with DFlash coding 49.5, prose 35.4, reasoning 61.1, code edit 80.7; a 250K prompt prefills in about 11
minutes. Different prompts and a lower-quality pack, so read these as orientation, not a matched A/B.

## Where the time goes

Decode (4.0, one row, 24.5 ms): experts 10.6 ms, dense projections and the head 9.8 ms, the ranks' exchanges 2.0 ms,
attention, routing and norms 1.9 ms. Experts and projections read their weights at about 215-225 GB/s, so decode
kernels have about 10% left; the levers are fewer bytes (the pack) and drafting. A 16K prompt chunk (4.0, 13.1 s):
the prompt GEMM 4.9 s, the exchanges 2.7 s, weight decoding 1.6 s, the rest 3.9 s.

Measured and rejected: fp16 prompt partials (exchange -0.65 s, casts and overflow guard +0.9 s a 16K chunk); copy
drafting from the context (no gain on this model).

## Files

`scripts/` serve, quantise, sync, benchmark and evaluate; `tools/` engine-level checks (parity, layers, graphs,
DFlash, profiles, perplexity); `patches/` the TensorFold changes; `results/` every number above as raw JSON;
`docs/RECON.md`, `docs/PLAN.md` the facts and plan the work followed.
