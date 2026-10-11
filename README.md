# MiMo-V2.6-Flash on 2x DGX Spark (GB10) with TensorFold

Xiaomi's MiMo-V2.6-Flash-RL (309B MoE, 15B active; 48 layers, 39 of them 128-token sliding-window attention with
sink logits, 9 full) served across two DGX Sparks with tensor parallelism, from our own EXL3 quantisation, on
TensorFold's Python engine (`python-0.6`) plus this recipe's patches, with Xiaomi's DFlash drafter.

Status: measured, text only (no vision, no audio). Default: the 3.0 bpw with DFlash.

## What this recipe adds to TensorFold

TensorFold had no MiMo family. Everything below is new, in `src/tensorfold/families/mimo_v2/` (patches 0024-0030).
Patches 0001-0023 are carried over from earlier TensorFold work; MiMo uses only their shared engine changes (the RoCE
exchange between the two Sparks and a server fix), none of their model code.

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
scripts/fetch_pack.sh              # the 3.0 pack from Hugging Face (pinned revision) on the head, copied to the worker, both verified
./start.sh                         # rank 1 on the worker, rank 0 here; OpenAI API on :8891, model id MiMo-V2.6-Flash
./stop.sh
```

The model: [ajensenwaud/MiMo-V2.6-Flash-RL-exl3-3.0bpw](https://huggingface.co/ajensenwaud/MiMo-V2.6-Flash-RL-exl3-3.0bpw).
To make it yourself instead: `scripts/quantise.sh 3.0` on the worker (it holds the RL source; ~17 h on one
GB10), then `PACK=3.0 scripts/sync_pack.sh`.

Settings in `scripts/config.sh` (override in `scripts/local.sh`): `PACK=3.0|4.0|2.27`, `DRAFTER=1|0`, `PARALLEL=4`
streams, `TP=2`.

## Models

Same RL source and exllamav3 settings (`-hq -cr 250 -cc 2048`, mul1 codebook, head 8 bits, MTP 4 bits), only the
bitrate differs; each carries the DFlash drafter. 2.27 is benthecarman's published model, our baseline.

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

Not run: HumanEval+/MBPP+ (they execute model-written code), GPQA-Diamond (gated), anything with thinking on.

## Speed (3.0 model)

Decode tok/s = completion tokens / the server's own decode seconds, median of 3; 512-token replies, greedy, thinking
off; prompts in `tools/mimo_prompts.py`.

| Task | Plain decode | DFlash | Speed-up |
|---|---|---|---|
| coding | 45.4 | 77.6 | 1.71x |
| code edit | 44.7 | 80.8 | 1.81x |
| reasoning | 45.3 | 72.8 | 1.61x |
| prose | 45.5 | 46.0 | 1.01x (the window chooser stops drafting) |
| 4 streams, one task each (aggregate) | 71.5 | 77.4 | 1.08x |

## Credits and licence

Thanks to Xiaomi MiMo for the model and its DFlash drafter; Ash Hart ([@ashxhart](https://x.com/ashxhart)) for
[TensorFold](https://github.com/ashhart/TensorFold), the engine this recipe patches; Jürgen Schmied
([@jschmied](https://github.com/jschmied)), author of patches 0001-0006, including the server fix this recipe relies on; turboderp for
[exllamav3](https://github.com/turboderp-org/exllamav3); benthecarman for the 2.27 bpw model, and the conversion settings and
the repaired DFlash drafter. 
