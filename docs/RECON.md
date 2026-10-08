# Recon — MiMo-V2.6-Flash on 2x DGX Spark via TensorFold (2026-10-09, read-only)

Everything here was read from the hosts, the Hub packs or the TensorFold tree on 2026-10-09. Nothing was executed on a GPU.

## Hosts

| | gx10 (head, rank 0) | gx11 (worker, rank 1) |
|---|---|---|
| CX7 rails | 10.100.128.1 / 10.100.129.1 | 10.100.128.2 / 10.100.129.2 |
| Unified memory | 121.6 GiB | 121.6 GiB |
| Disk free | **530 GB** (916 GB root; was 223 GB, cleaned 2026-10-09 — see below) | 2.7 TB |
| Live GPU containers | `glm53-flash-tf` (TensorFold v0.6.0, GLM-5.3 TP=2 rank 0, :8888), `sparkDash` (non-GPU) | `glm53-flash-tf` (rank 1), `meter-proxy` (non-GPU) |
| MemAvailable while GLM serves | ~13 GB | ~10 GB |
| IOMMU | `iommu.passthrough=1` since 2026-10-06 (RDMA 18.7 GB/s) | same |

gx10 cleanup 2026-10-09 (Anders approved each item): deleted the retired vLLM GLM stack's TR3 checkpoint (164 GB) and `:exl3` image, the Kolibri checkpoint `~/kolibri/ckpt` (74 GB; `scripts/prepare.sh` in the Kolibri repo re-downloads it), the Kolibri NVFP4 expert cache `~/.cache/kolibri-tf/kolibri-quant` (60 GB, rebuilt at the next `PROFILE=fast` start), 8 superseded `kolibri-tf:*` tags + `kolibri-dev:pr328`, 22 exited DeepSeek/h4a/bench/kolibri-dev containers, pip/uv caches. Kept: `kolibri-tf:latest`, `tensorfold-glm53:v0.6.0` and the GLM TensorFold checkpoint (84 shards, verified), `~/harness-levers` + `levers-*` images (another session's live work). The old vLLM GLM rollback no longer exists on gx10. GLM verified serving afterwards (`finish_reason stop`).

Other recipe repos on gx10: `~/models/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` (GLM, :8888), `~/models/Kolibri-1-2x-DGX-Sparks-TensorFold` (Kolibri, :8890, image `kolibri-tf:latest`; checkpoint deleted, re-fetched by its `prepare.sh`). Ports/rendezvous in use: 8888/29551 (GLM), 8890/29561 (Kolibri). This recipe takes **8891/29571**.

## Checkpoints already on gx11 (`~/models/mimo-v2.6/`, pulled by `~/mimo-prestage.sh` 00:10–01:32 today)

### Source: `XiaomiMiMo/MiMo-V2.6-Flash-RL` @ `5711b268169967567844e1e560e8a3966da959b1` — 166 GB, 65 shards
- `model.safetensors.index.json`: `save_format: mxfp4`, `total_size: 172,923,364,096`, `tp_size: 4`, 73,081 tensors.
- `config.json` quantization block: `quant_method: fp8`, `store_dtype: mxfp4`, `mxfp4_block_size: 32`, `weight_block_size [128,128]`, `o_proj` of every layer ignored (bf16). So the "original" is itself quantised (FP8 attention/dense, MXFP4-stored experts). Any "lossless vs original" claim is relative to this, not to bf16 training weights.
- Tensor families (text): `embed_tokens`, `layers.N.self_attn.{qkv_proj (fused), o_proj, attention_sink_bias}`, `layers.N.mlp.{gate.weight, gate.e_score_correction_bias, experts.N.{gate,up,down}_proj(+weight_scale)}`, layer 0 dense `mlp.{gate,up,down}_proj`, `mtp.layers.N.{eh_proj, enorm, hnorm, input_layernorm, pre_mlp_layernorm, final_layernorm, self_attn.*, mlp.*}` (3 MTP layers, in `model_mtp.safetensors`), `lm_head`, `norm`. Multimodal: `visual.*` (28 blocks), `speech_embeddings.*`, `audio_tokenizer/` (1.9 GB). `dflash/` = BF16 5-layer drafter (2.9 GB) + `dflash.py` + `mask_embedding.pt`.

### Baseline pack: `benthecarman/MiMo-V2.6-Flash-RL-exl3` @ `5ae87b830eb950c705967aea6130601a8e1f74a5` — 89 GB (83.5 GiB), 12 shards
- `quantization_config.json`: `quant_method exl3`, `version 1.5.2`, `bits 2.27`, `head_bits 6`, `codebook mul1`, `mtp_bits 4`, calibration 250 rows x 2048, `out_scales always`.
- Per-module bpw (from the README): routed experts layers 12–35 → 2.0, layers 1–11 and 36–47 → 2.5; attention 4.0; dense MLP (layer 0) 3.0; `lm_head` 6.0; MTP 4.0; embeddings/norms/router/vision bf16.
- Tensor layout: 145,877 tensors, 105 name patterns. Attention is **split** `q_proj/k_proj/v_proj/o_proj` (each `trellis/suh/svh/mul1`), not the source's fused `qkv_proj`. Experts: `layers.N.mlp.experts.N.{gate,up,down}_proj.{trellis,suh,svh,mul1}`. Router `gate.weight` + `gate.e_score_correction_bias` bf16. MTP layers quantised the same way. Sample shapes: `q_proj.trellis [256,768,64]` (4096→12288, 4 bit), `k_proj.trellis [256,48,64]` (4096→768 = 4 KV heads x 192), expert `up/gate.trellis [256,128,40]` (4096→2048, 2.5 bit), `down.trellis [128,256,40]`, `lm_head.trellis [256,9536,96]` (6 bit). All `mul1_multiplier 2212286765`.
- **Layer 47 `interm_div`**: some experts overflow fp16; the pack scales `up_proj` of layer 47 down by 128 and the engine must restore x128 in fp32 after the activation. The README says the scale is "folded into the weights" — verify by reading exllamav3's MiMo module before assuming where the restore happens.
- `dflash/`: 4 bpw EXL3 drafter (`model.safetensors` 735 MB, `config.json` repaired). `dflash-bf16/`: the original. `eval/`: ppl + benchmark artifacts (`eval/bench/`).
- Reported by the author (exllamav3 1.5.2, one GB10, batch 1, greedy, 512 tok, median of 3): no drafter 31.5 / 31.3 / 31.2 / 31.0 tok/s (coding / prose / reasoning / code-edit); DFlash 49.5 / 35.4 / 61.1 / 80.7; MTP 40.7 / 31.8 / 45.1 / 51.3. 2K prefill 593 tok/s, 32K 826 tok/s. Long context with drafter: 64.6k → 43.8 tok/s, 249k → 33.6 tok/s. Quality vs vendor FP8 endpoint: HumanEval+ 88.4 vs 90.2, MBPP+ 77.2 vs 77.2, GSM8K 95.2 vs 96.4, MMLU-Pro 73.8 vs 76.4, GPQA-D (8K cap) 50.0 vs 56.2. wikitext-2 ppl 5.4003.

### Other public packs (not downloaded)
- `vcruz305/MiMo-V2.6-Flash-MOPD-EXL3`: RL 2.20 (86.9 GB, top-1 82.16 %, KLD 0.193) and 2.50 bpw (98.5 GB, top-1 83.76 %, KLD 0.206); MOPD 2.20/2.50. Serving recipe exists for exllamav3 (RTX PRO 6000: 49.6 tok/s single, 116 tok/s aggregate @8).
- `MikeRoz/MiMo-V2.6-Flash-MOPD-4.03bpw-h8-exl3`: 148.5 GiB (MOPD lineage), 2.52bpw-h6 94.4 GiB.

## Model architecture (from `config.json` + `modeling_mimo_v2.py`)

| Item | Value |
|---|---|
| `model_type` / arch | `mimo_v2` / `MiMoV2ForCausalLM` |
| Layers | 48; `hybrid_layer_pattern` 1 = SWA (39 layers), 0 = full (9 layers: 0, 5, 11, 17, 23, 29, 35, 41, 47) |
| Hidden / vocab | 4096 / 152,576 |
| Full attention | 64 Q heads, **4** KV heads, qk head_dim 192, **v_head_dim 128**, rope_theta 1e7 |
| SWA | window 128 (`sliding_window`, `attention_chunk_size 128`), 64 Q / **8** KV heads, qk 192, v 128, `swa_rope_theta 1e4`, `add_swa_attention_sink_bias true` (per-head sink logit appended to the softmax), `add_full_attention_sink_bias false` |
| RoPE | `partial_rotary_factor 0.334` → rope_dim = int(192 x 0.334) = 64 of 192 dims rotated |
| Value scale | `attention_value_scale 0.707` applied to V |
| Projection layout | source: `fused_qkv`; EXL3 pack: split q/k/v |
| MoE | layers 1–47 (`moe_layer_freq`), 256 routed experts, top-8, `moe_intermediate_size 2048`, `scoring_func sigmoid`, `topk_method noaux_tc` (scores + `e_score_correction_bias` pick the top-8; the *uncorrected* sigmoid scores are the weights), `norm_topk_prob true`, `n_group 1`, `topk_group 1`, `routed_scaling_factor null`, `n_shared_experts null`, `moe_router_dtype bfloat16` |
| Dense MLP | layer 0 only, `intermediate_size 16384` |
| MTP | `num_nextn_predict_layers 3`; each: enorm/hnorm → `eh_proj` (concat embed+hidden) → one SWA-style decoder layer (dense MLP) → final_layernorm → shared `lm_head` |
| Norm | RMSNorm eps 1e-6 |
| Context | `max_position_embeddings 1,048,576` |
| Tokens | eos 151645, pad 151643; DFlash `mask_token_id 151675` |

### DFlash drafter (`dflash/config.json`)
5 SWA layers (window 1024, `is_causal false`, bidirectional block), hidden 4096, 64 Q / 8 KV heads, head_dim 128, `partial_rotary_factor 0.5`, rope_theta 1e4, `block_size 8`, `target_layer_ids [0, 11, 23, 35, 47]` (hidden states tapped after those target layers, concatenated 5x4096 → `fc`), `num_anchors 4096`, `attention_value_scale 0.612`, `attention_sink_bias true`. Draft loop in `dflash/dflash.py::spec_generate` (block of 8 mask tokens, one drafter forward per block, target verifies). TensorFold has the equivalent machinery for GLM in `families/glm5_next/cuda/dflash2.py` (taps, block compute, capture, chain, propose).

## TensorFold

- GitHub `ashhart/TensorFold`. `main` = 1.0.2 (Zig binary, 7 Oct 2026). The Python engine is branch **`python-0.6`** @ `ed78d6fc204d89d90b045bf033d6551e7714f3a1` (0.6.6, 7 Oct 2026). GLM serves on image `tensorfold-glm53:v0.6.0`; Kolibri built on 0.6.5 + PR #328.
- **No MiMo family.** `src/tensorfold/families/`: bonsai, deepseek_v4, gemma4, glm5_next, nemotron_h, qwen3_5, qwen3_5_moe, qwen4_exp. Discovery is by `MODEL_TYPES` in the package `__init__.py`; a CUDA family exports `cuda_engine(model_dir, *, drafter, tp, rank, master, master_port, no_drafts, mtp_drafts, **options)` and an engine with `eos`, `generate(prompt, max_tokens, sampling, on_tokens)`, `follow()` for rank 1, cache capacity; optional `CUDA_APP`. Guide: `docs/recipes/adding-a-cuda-family.md`. Exactness bar: a row gets the same bits alone and in every verify window; graphs == eager; two ranks == that engine's serial reference.
- **Universal EXL3 kernels exist and were built on MiMo tensors**: `src/tensorfold/cuda/exl3/` — `format.py` (header parse, `scan(model_dir)`, numpy reference `dequantize/forward`), `linear.py` (`Exl3Linear.load(model_dir, prefix)`, `__call__(x, out, out_dtype)`), `experts.py` (`Exl3RoutedExperts`, `prepare(gate, up, down, codebook, …)`, `Scratch`, `routed(x, pick, wts, ex, s, …)`, `dequant`), `prefill.py`, codebooks 3inst/mcg/mul1 at 1–8 bits incl. half-bits, mixed-K within a layer. Measured on MiMo 2.50bpw (docs/recipes/exl3-universal-experts.md): 180–214 GB/s per layer, 1.5–3.6x ExLlamaV3's `exl3_moe_mixedk`; CUDA-graph replay equals eager on real MiMo layers L3/L30/L47. Dense linears within ±5 % of ExLlamaV3 (docs/recipes/exl3.md).
- Reusable family code: `glm5_next/cuda/` (`weights.py` EXL3 routed experts via the generic module, `dflash2.py`, `mtp.py`, `graphs.py`, `decode.py`, `engine.py`, `app.py`, `split.py`), `qwen4_exp/cuda/` (`multi_tp.py`, `kvcache.py`, `kvquant.py`, `ngram.py`, `prefixes.py`, `prompt_plan.py`, `copy_drafts.py`).
- Kolibri patches (on gx10 `~/models/Kolibri-1-2x-DGX-Sparks-TensorFold/patches/`): 0001 decode CUDA graphs, 0002 kernel/comm benches + vendored RoCE one-shot all-gather, 0003 two ranks (`--tp 2`: heads split, expert width split, vocab split, partials all-gathered in rank order), 0004/0005 TP tests, 0006 KL quality gate vs fp32 reference, 0007 two-node TP smoke, 0008 split-K decode experts, 0010 o_proj K-slices, 0013 bf16 prompt partials, 0014/0015 NVFP4 decode experts + precision levers, 0016/0017 prefill-fp8. Kolibri's `docker/Dockerfile` overlays a patched TensorFold `src` on the GLM base image via `PYTHONPATH`; `scripts/prepare.sh` clones + `git am`s patches on both nodes and builds the image on each, checks tree hashes match.

## exllamav3 (reference implementation + quantiser)
- MiMo-V2 support merged upstream 2026-09-25 (PR #399 `mimo-v2-architecture`, "Add MiMoV2ForCausalLM"). Tags since: v1.5.2 (what the pack was built with), v1.5.3, v1.5.4, v1.6.0. No aarch64 wheels: build from source inside the container.
- Pack author's conversion flags: `convert.py -b 2.25 -hq -cr 250 -cc 2048`; MTP at 4 bits; layer-47 `interm_div` fix — check whether that is upstream or the author's branch before quantising.

## Physics / budget
- Active params ≈ 15 B. Bytes/token for the active set: 2.27 bpw ≈ 4.3 GB; 3.0 ≈ 5.6 GB; 4.0 ≈ 7.5 GB; plus lm_head (6 bit: 0.47 GB; 8 bit: 0.62 GB) and KV reads of the 9 global layers (bf16, 4 KV heads x (192+128) x 2 B = 2.5 KB/token/layer → 23 KB/token; at 64k context ≈ 1.5 GB/step).
- GB10 ≈ 273 GB/s nominal; TensorFold's EXL3 expert kernel reaches 180–214 GB/s. One-Spark decode ceiling at 2.27 bpw ≈ 50–60 tok/s (exllamav3 gets 31.5); TP=2 halves bytes/rank.
- Pack size per rank at TP=2 (experts half-width or half-count per rank, attention heads split, embeddings/norms replicated): 2.27 bpw ≈ 42 GiB; 3.0 ≈ 55 GiB; 4.0 ≈ 73 GiB. All leave ≥ 40 GiB/rank for KV, drafter (0.7 GiB), MTP, graphs and workspaces. A 3.0 pack (~108 GiB) does **not** comfortably fit one Spark with KV; 4.0 does not fit at all. So TP=2 is what makes the higher-bpw packs servable.
- TP=2 cost: ~2 collectives/layer x 48 over CX7; Kolibri measured RoCE one-shot all-gather +14 % over NCCL and CUDA graphs +2 % at TP=2.
