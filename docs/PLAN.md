# Plan — MiMo-V2.6-Flash on 2x DGX Spark via TensorFold

> **For the executing agent (Opus):** this plan was written by a separate planning session with zero GPU work done. Facts are in `docs/RECON.md`; read it first. Work gate by gate; every gate has a measured pass condition and a `results/<arm>/` directory. Do not advance on a self-report. Report the hypothesis ID with every number.

**Goal:** Serve MiMo-V2.6-Flash-RL (309B MoE, 15B active) across gx10+gx11 (TP=2) on TensorFold from an EXL3 checkpoint, faster than the single-Spark baseline and with quantisation quality gated against the FP8 reference, packaged as a recipe repo in the Kolibri-1 format.

**Architecture:** New `mimo_v2` family on TensorFold `python-0.6` built from the existing universal EXL3 kernels (dense linears + grouped mixed-K routed experts) plus the Kolibri recipe's two-rank TP / RoCE all-gather / decode-graph patches. Two Sparks buy both bandwidth (speed) and memory for a higher-bpw pack (quality): a ~3.0 bpw pack is the speed candidate, ~4.0 bpw the quality candidate; the measured KL/eval gate picks.

**Tech stack:** TensorFold `python-0.6` @ `ed78d6fc204d89d90b045bf033d6551e7714f3a1`; base image `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529` (PyTorch 2.13 nv26.07, Triton 3.7.1, NCCL, libibverbs — the Kolibri base); exllamav3 (newest tag that loads the 2.27 pack: try v1.6.0, fall back to v1.5.2) built from source inside the image; Docker on both nodes; CX7 both rails; `iommu.passthrough=1` (done).

---

## Decisions recorded (Anders, 2026-10-09)

1. Checkpoint lineage: **RL** (`XiaomiMiMo/MiMo-V2.6-Flash-RL` @ `5711b268…`). MOPD is a documented config swap, not in scope.
2. Packs: **quantise our own** 3.0 and 4.0 bpw from the RL source (`mul1`, head 8, MTP 4). Keep `benthecarman/…-exl3` 2.27 @ `5ae87b83…` as arm A and as the Gate-2 development pack.
3. GPUs: **GLM-5.3 comes down for the project's duration.** Way back: `ssh gx10 'cd ~/models/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold && ./start.sh'` (images + weights stay cached). clawdy shares that serve — tell Anders when it goes down.
4. Builds: **containers only.** One image `mimo-tf:<tree-hash>` from the Kolibri base. No host `apt/pip/npm`, no sudo, no firewall/grub/network changes. exllamav3 compiles inside the image.
5. Modalities: **text first.** Vision/audio tensors are skipped at load; vision is an optional later gate; audio out of scope.
6. AGENTS.md path fixed to `~/models/mimo-v2.6/` (done by the planning session).

## Standing rules (from the fleet skills; violations have cost real downtime before)

- Never run `start.sh`/`prepare.sh`/long jobs in a foreground ssh. Launch with `ssh -n <host> 'cd … && setsid nohup <cmd> > ~/<log> 2>&1 </dev/null & disown; echo launched' </dev/null`, put **nothing** after it in the same call, and poll in short separate calls (`tail`, `docker ps`, `pgrep -af "[c]onvert"` — bracket the first letter or pgrep matches your own ssh).
- Client check in its **own** call before stopping any serve: `ss -tn state established | grep :<port>` and `curl -s http://127.0.0.1:<port>/health` on the host. Loopback-only pairs are local pollers; any remote socket or `requests_running > 0` → ask Anders.
- Curl the endpoints **on gx10 over ssh**; cross-machine curls from the agent host are blocked by the approval gate.
- `nvidia-smi` memory reads `[N/A]` on GB10; read `/proc/meminfo` MemAvailable and `nvidia-smi --query-compute-apps=pid,used_memory`.
- Benchmarks only on an idle endpoint, medians of ≥ 3 reps, SM clock logged (`nvidia-smi --query-gpu=clocks.sm --format=csv,noheader`), raw JSON kept.
- Do not kill, modify or starve `sparkDash`, `meter-proxy` or anything you did not start. gx11 shares a ~1 Gbps uplink: downloads are background and sequential.
- No Hermes model/provider switches; hand Anders the endpoint facts when a serve is up.
- Git: commit after every task in this repo; never push without Anders' word; check `git config user.name/email` before the first commit.

## Hypothesis register (MECE — every result names the ID, the variable, the controls, the number)

| ID | Lever | Candidate(s) | Control | Gate |
|---|---|---|---|---|
| H1 | Model/structure | RL; text-only; MTP heads loaded or not | — | 1 |
| H2 | Weight representation | own EXL3 3.0 / 4.0 bpw (`mul1`, head 8, MTP 4, converter's per-layer mixed-K); 2.27 public | 2.27 pack | 3 |
| H3 | Memory placement | FP8 KV on the 9 full-attention layers; SWA ring caches of 128 | bf16 KV | 5 |
| H4 | Engine/kernels | `mimo_v2` family: universal EXL3 experts/linears, fused sigmoid-router+bias+top-8, SWA attention with sinks, decode CUDA graphs, split-K decode experts | eager, no split-K | 2, 5 |
| H5 | Generation | DFlash 4bpw drafter (block 8, dynamic); MTP 3-head chain; n-gram | `draft: false` | 6 |
| H6 | Multi-device | TP=2: heads 32/2 (full) + 32/4 (SWA) per rank, experts half-width per rank (Kolibri 0003) vs expert-parallel 128/rank; RoCE one-shot all-gather vs NCCL; bf16 prompt partials | TP=1 | 4 |
| H7 | Scheduling | `--parallel 4` lane rounds; prefix cache | 1 stream | 7 |
| H8 | Workload | code / chat / reasoning; 64 & 256 tok; 2k–64k prefill; 2k–58k decode | fixed prompt set | all |
| H9 | Platform | passthrough (done); clocks logged | — | — |

---

## Gate 0 — Scaffold and preconditions (no GPU; ~1 session)

### Task 0.1 Repo scaffold
Create in `/home/aj/projects/mimo-2.6-2dgx-spark-tensorfold/` (mirror `gx10:~/models/Kolibri-1-2x-DGX-Sparks-TensorFold/`; copy and adapt, do not re-derive):
```
AGENTS.md README.md docs/{RECON,PLAN,KERNELS}.md docker/Dockerfile patches/ results/ logs/
scripts/{config.sh,prepare.sh,bench.sh,verify.sh,smoke.py,long_decode.py,summarise.py,concurrent_prefill.py}
start.sh stop.sh .gitignore (scripts/local.sh, logs/, results/**/*.log)
```
`git init`, commit "docs: recon and plan".

### Task 0.2 `scripts/config.sh`
Adapt Kolibri's. Pins:
```
WORKER=aj@10.100.128.2  MASTER=10.100.128.1  MASTER_PORT=29571  PORT=8891  SERVED_NAME=MiMo-V2.6-Flash
TF_REPO=https://github.com/ashhart/TensorFold  TF_BRANCH=python-0.6  TF_COMMIT=ed78d6fc204d89d90b045bf033d6551e7714f3a1
TF_DIR=$HOME/mimo/TensorFold  BASE_IMAGE=<Kolibri's digest>  IMAGE=mimo-tf  CONTAINER=mimo-tf
EXL3_REPO=https://github.com/turboderp-org/exllamav3  EXL3_TAG=v1.6.0   # fallback v1.5.2
MODEL_DIR=$HOME/models/mimo-v2.6/<pack>   # same path on both nodes; PACK=2.27|3.0|4.0 selects it
SRC_DIR=$HOME/models/mimo-v2.6/MiMo-V2.6-Flash-RL   SRC_REV=5711b268169967567844e1e560e8a3966da959b1
BASELINE_PACK_REV=5ae87b830eb950c705967aea6130601a8e1f74a5
DRAFTER_DIR=$MODEL_DIR/dflash   PARALLEL=4  TP=2  MAX_TOKENS=32768  CONTEXT=   CACHE_DIR=$HOME/.cache/mimo-tf
NCCL_IFNAME=enp1s0f0np0  NCCL_HCAS=rocep1s0f0,roceP2p1s0f0  NCCL_GID=3  ROCE_MAX_KB=512
PROFILE=quality   # fast: TF_MIMO_KV=fp8 etc., each lever gated in Gate 5
```
`serve_args` as Kolibri's plus `--drafter $DRAFTER_DIR` when `DRAFTER=1`.

### Task 0.3 `docker/Dockerfile`
Kolibri's pattern (base image + `COPY src /opt/tensorfold-mimo/src`, `PYTHONPATH`), plus a stage that clones exllamav3 at `$EXL3_TAG` and `pip install --no-build-isolation .` with `TORCH_CUDA_ARCH_LIST=12.1` (GB10 is SM121) and `MAX_JOBS=8` (UMA host: bound the compiler). Build-gate inside the image, no GPU needed:
```
docker run --rm --entrypoint python3 mimo-tf:<h> -c "import exllamav3, tensorfold; from tensorfold.cuda.exl3 import experts, linear; print(exllamav3.__version__)"
```
Record `docker image inspect --format '{{.Id}} {{.Created}} {{.Size}}'` into `docs/RECON.md`.

### Task 0.4 Disk arithmetic (write into RECON.md before any download/rsync)
gx10 has 530 GB free after the 2026-10-09 cleanup (RECON.md). Needs: image delta (base layers already present) + both serving packs for the A/B (3.0 ≈ 116 GB, 4.0 ≈ 155 GB) + caches ~10 GB ≈ 290 GB, leaving ~240 GB headroom. Kolibri's checkpoint is already gone; nothing else on gx10 needs parking. gx11 (2.7 TB) holds source, all packs, conversion work dirs.

### Task 0.5 Format archaeology (no GPU; python in the image or host python3 with safetensors headers only)
Write `tools/mimo_inspect.py` (host-side, header reads only, never a full shard) that prints per-module bpw, codebook, shapes and the tensor-count census for a pack; run it on the 2.27 pack and save `results/packs/2.27bpw-census.json`. Confirm: split q/k/v, 4 KV heads full / 8 SWA, `interm_div` handling for layer 47 (read exllamav3's `models/mimo_v2.py` or equivalent to see where the x128 restore lives), MTP tensor schema, router bias dtype.

Pass: scaffold committed; image builds and the import gate passes on both nodes; RECON.md has the per-rank budget table and the disk plan.

---

## Gate 1 — Reference serve and quality harness (one Spark, exllamav3; ~half a session + bench time)

### Task 1.1 Take the GPUs (approved)
1. Own call: `ssh gx10 'ss -tn state established | grep :8888; curl -s http://127.0.0.1:8888/health'`. Read it.
2. `ssh gx10 'cd ~/models/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold && ./stop.sh'`; confirm `docker ps` on both nodes shows no `glm53-flash-tf`; MemAvailable > 100 GB on both.
3. Tell Anders GLM is down and how it comes back.

### Task 1.2 Arm A: exllamav3 serving the 2.27 pack on gx11 (text only)
Container from `mimo-tf` image, exllamav3's own OpenAI-compatible server (or TabbyAPI if the image carries it — pick one, record it), `max_seq_len 65536`, FP16 cache, batch 1, `dflash/` as the drafter in a second run. Smoke: greedy 512-token completion, `finish_reason stop`.

### Task 1.3 `scripts/bench.sh` (port Kolibri's)
TensorFold's own tools from the image (`tools/bench_openai.py`, `bench_concurrent.py --alone`, `prefill_cold.py`), prompts fixed in `results/prefill-prompts.json` + a code/chat/reasoning trio. Run arm A no-drafter and with DFlash → `results/A-exl3-1spark-2.27/{decode-64,decode-256,concurrent,prefill}.json`, `serve-config.txt`, `health-before/after.json`, `clocks.txt`.
Pass: no-drafter decode within ±10 % of 31.5 tok/s (code) and DFlash within ±15 % of 49.5; otherwise the harness is wrong — fix it, do not argue with the pack author's numbers.

### Task 1.4 Quality harness, frozen here and reused by every later gate
- `tools/mimo_kl.py` (port Kolibri 0006 `kolibri_kl.py`): KL(reference ‖ engine) and top-1 agreement over 4 texts (EN prose, ZH prose, code, agent transcript), 160 prompt rows + 96 decode rows. Reference = exllamav3 forward on the same pack with `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` (engine-parity gate). Noise floor = exllamav3 decode path vs its own prompt path on the same rows.
- `tools/mimo_eval.py`: 100-item GSM8K + 100-item HumanEval+ slice at t=0 via the OpenAI endpoint, thinking off, 2048-token cap; same items every time. Baseline numbers: run it on arm A; compare with the pack's `eval/bench/` and the FP8 figures in RECON.md.
Pass: both tools run end to end on arm A; floor and arm-A scores in `results/A-…/quality.json`.

---

## Gate 2 — `mimo_v2` family on TensorFold, one Spark, 2.27 pack (the critical path; 2–4 sessions)

Work in `$TF_DIR` (branch `mimo-recipe` on `ed78d6f`), one commit per task, exported with `git format-patch` into `patches/NNNN-*.patch`; `scripts/prepare.sh` re-applies them with `git am` on both nodes and compares tree hashes (Kolibri pattern). Tests in `tests/cuda/test_mimo_*.py`, run inside the container with `TENSORFOLD_MIMO_EXL3=<pack>`.

### Task 2.1 Family package and weights (`families/mimo_v2/{__init__,config,weights,layouts}.py`)
`MODEL_TYPES = ("mimo_v2",)`, `TITLE`, `MODELS`, `QUANT_METHODS = {"cuda": ("exl3",)}`, `EXL3_VARIANT = EXL3_VARIANT_ANY`, `cuda_engine(...)`, `check(model_dir)` refusing non-EXL3. `weights.py`: `tensorfold.cuda.exl3.format.scan()` → per-tensor `Exl3Tensor`; `Exl3Linear.load` for q/k/v/o, dense layer-0 MLP, MTP projections, `lm_head`; `experts.prepare(gate, up, down, codebook="mul1")` per MoE layer (mixed-K is native); bf16 embeddings, norms, router weight + `e_score_correction_bias`, `attention_sink_bias`; skip `visual.*`, `speech_embeddings.*`; MTP optional (`mtp_drafts`), drafter optional. Layer-47 `interm_div` restore where exllamav3 does it.
Test: every text tensor in the pack is consumed exactly once; none invented; nbytes per module summed equals the census.

### Task 2.2 Attention (`cuda/attention.py`, `cuda/kvcache.py`)
Two kernels/plans: full GQA (64/4, qk 192, v 128, theta 1e7, no sink) and SWA-128 (64/8, theta 1e4, sink bias per head appended as an extra logit column, window 128 ring cache as `qwen4_exp/cuda/kvcache.py` does). Partial RoPE on the first 64 of 192 dims (check exllamav3 for interleaved vs half layout). V scaled by 0.707 **before** the softmax-weighted sum (check where exllamav3 applies it). Prefill via the chunked prompt path; decode via a graph-safe plan with sequence lengths as masked device loads.
Test: per-layer attention output vs exllamav3's on real rows (prompt 2k, decode rows) within bf16 tolerance; window 1 row == same row in an 8-row window.

### Task 2.3 Router and MoE (`cuda/moe.py`)
Fused: `logits = x @ gate.weight.T` (bf16 router dtype), `s = sigmoid(logits)`, `pick = top8(s + bias)`, `w = s[pick] / sum(s[pick])` (norm_topk), then `experts.routed(x, pick, w, ex, scratch)`. Stable tie-breaking (lower index wins). Dense layer 0 via three `Exl3Linear` + SiLU.
Test: expert IDs exactly equal exllamav3's on 512 real rows; layer output within 2e-4 rel.

### Task 2.4 Forward, decode, engine, server (`cuda/{forward,decode,engine,app}.py`)
Prompt path (chunks of 2048–8192 rows), decode path, `generate`/`eos`/capacity, OpenAI chat with the pack's `chat_template.jinja` (thinking on by default, `enable_thinking:false` honoured, `reasoning_content` split). Serve `--host 0.0.0.0 --port 8891 --name MiMo-V2.6-Flash`.
Test (the exactness bar): greedy 256-token completions on 3 prompts equal exllamav3's token for token. Then `scripts/verify.sh`: ≥ 500-token completion, `finish_reason stop`.

### Task 2.5 Decode CUDA graphs (`cuda/graphs.py`, Kolibri 0001 pattern)
Per (lanes, key bucket); one attention plan per forward.
Test: graph == eager bit for bit after allocator churn; `TF_MIMO_GRAPHS=0` switch kept for A/B.

### Task 2.6 Profile and fix the obvious (only if below the roofline check)
`torch.profiler` one decode step, leaf kernels only. If the expert GEMV sits below ~180 GB/s or `lm_head`/o_proj dominate: Kolibri 0008 (split-K experts) / 0010 (o_proj K-slices). Record `docs/KERNELS.md` with the byte budget and the measured per-kernel times.

**Arm B** = this engine, one Spark (gx10), 2.27 pack, no drafter → `results/B-tf-1spark-2.27/`. Pass: exact tokens vs exllamav3; `mimo_kl.py` at the noise floor; decode ≥ 1.3x arm A no-drafter (≥ 41 tok/s code; roofline says ~55 is there). If exact-token parity is not reached after ~3 sessions: stop, report the partial port, and release exllamav3-on-one-Spark as the interim recipe (AGENTS.md names TensorFold, so this is a reported shortfall, not a quiet substitution).

---

## Gate 3 — Our own packs (gx11, background to Gate 2; 1 h pilot + 10–14 h per pack unattended)

### Task 3.1 Converter audit (read before running; `quantizer-tool-review` checklist)
Read exllamav3 `convert.py` at the pinned tag: resume semantics, where it writes during a dry run, how `-b`/`-hb`/`-hq`/`-cr`/`-cc` map, whether MTP bits are a flag, and whether the layer-47 overflow (`interm_div`) is handled upstream. If not upstream, port the author's fix as a patch in `patches/exllamav3/` and apply it in the Dockerfile — do not quantise without it (fp16 overflow would silently poison layer 47).

### Task 3.2 Pilot
Container on gx11, `-i $SRC_DIR -o ~/mimo/work-3.0 -cd ~/models/mimo-v2.6/MiMo-V2.6-Flash-RL-exl3-3.0bpw -b 3.0 -hb 8 -hq -cr 250 -cc 2048` limited to the first 2 layers if the tool allows; else time the first 2 layers of the full run and extrapolate. Record s/matrix and peak MemAvailable drop. Expect ≈ 1 expert matrix/s on GB10 → 36,096 expert matrices ≈ 10–12 h + attention/head.

### Task 3.3 Full runs (detached, logged, polled)
3.0 first, then 4.0 (`-b 4.0 -hb 8`). After each: converter's own wikitext ppl; `tools/mimo_inspect.py` census (every text module present, suffix census `trellis/suh/svh/mul1` one of each per matrix, tensor count predicted from the source); sha256 manifest `MANIFEST.sha256` in the pack dir; rsync the chosen pack to gx10 over `10.100.128.1` only after Task 0.4's disk plan says it fits.

### Task 3.4 Quality (needs an engine: exllamav3 for the KL reference, our engine once Gate 2 passes)
`mimo_kl.py` and `mimo_eval.py` on 2.27 / 3.0 / 4.0. Pass: ≥ one own pack with eval slice within ±1 pt of the FP8 figures and top-1 agreement above 2.27's; its per-rank bytes leave ≥ 20 GiB headroom. Table into README "Packs".

---

## Gate 4 — Two Sparks, TP=2 (1–2 sessions)

### Task 4.1 Communication microbench first
Port Kolibri 0002's `kolibri_bigcomm_bench.py` → `tools/mimo_bigcomm_bench.py`: decode-sized (4096 x 2 B per rank) and prefill-sized (8192 x 4096 x 2 B) all-gathers over NCCL and the RoCE one-shot path, both rails. Decide expert split by numbers: half-width per rank (Kolibri: 2 gathers/layer of hidden-size partials) vs expert-parallel 128/rank (1 gather/layer but routed-token imbalance). Write the decision and the numbers into `docs/KERNELS.md`.

### Task 4.2 `cuda/tp.py` (Kolibri 0003 + 0013 + 0002 vendored RoCE)
Rank 1 follows rank 0 (`follow()`), forwards described over NCCL, gathers over RoCE up to `ROCE_MAX_KB`, fp32 partials added in rank order, prompt partials in bf16. Heads: full 32 Q / 2 KV per rank, SWA 32 Q / 4 KV per rank; vocab split for `lm_head`; embeddings/norms/router replicated. `start.sh`/`stop.sh` worker-first as Kolibri.
Test: ranks bit-equal to each other; TP=2 greedy tokens equal the one-Spark engine's on the 2.27 pack (summation order differences allowed only where documented, KL at the floor).

**Arm C** = TP=2 with the Gate-3 winner → `results/C-tp2-<bpw>/`. Also run TP=2 on the 2.27 pack for a like-for-like speed number vs arm B. Pass: C(2.27) 1-stream decode ≥ 1.4x B; C(winner) reported with its quality numbers next to it; prefill 2k/8k/32k/64k reported honestly (Kolibri saw TP=2 prefill below one Spark until bf16 partials + passthrough).

---

## Gate 5 — Precision and kernel levers (each opt-in, measured alone, quality-gated; 1 session)
`TF_MIMO_KV=fp8` on the 9 full-attention layers (bf16 default); prompt-row projections on the FP8/EXL3 prefill path; split-K / o_proj slices if not already in; fused router. Keep a lever only if `mimo_kl.py` stays at the floor; rejected levers stay in the README with their numbers. `PROFILE=quality|fast` as Kolibri's config.sh.

## Gate 6 — Speculative decoding, lossless (1–2 sessions; the biggest multiplier after TP)
- DFlash: port `glm5_next/cuda/dflash2.py` to MiMo's drafter contract (RECON: 5 SWA-1024 layers, block 8, taps after target layers 0/11/23/35/47, `mask_token_id 151675`, value scale 0.612, sink bias, partial RoPE 0.5 on head_dim 128; ashhart's omlx PR #3821 is the MLX reference for the same adapter). 4 bpw EXL3 drafter weights through `Exl3Linear`. Replicated on both ranks; verified in the target's lane rounds.
- MTP: 3-layer chain via `glm5_next/cuda/mtp.py` pattern (`mtp_drafts`); n-gram floor from `qwen4_exp/cuda/ngram.py`.
- Dynamic draft window: prose acceptance was 12 % static / 50 % dynamic in the author's data — without it, prose gets slower.
Pass: every accepted token equals `draft: false` on 10 prompts x 256 tokens; acceptance and tok/s per class in `results/D-tp2-dflash/`, `results/D-tp2-mtp/`.

## Gate 7 — Concurrency and long context (1 session)
`--parallel 4` lane rounds, prefix cache; `scripts/long_decode.py` at 2k/15k/29k/58k; prefill to 64k, then the recipe's `CONTEXT` ceiling (size from free memory as Kolibri does; report the fitting capacity). Pass: 4-stream aggregate and together == alone token counts; no OOM at `CONTEXT`.

## Gate 8 — Recipe release (1 session)
`scripts/prepare.sh` (both nodes: base image by digest, TensorFold + patches with tree-hash check, exllamav3 tag, pack by sha256 manifest, rsync to worker), `start.sh`/`stop.sh`, `scripts/verify.sh`, README in the Kolibri format: what the recipe adds (table: change / where / effect on bits), run it, measured table (arms A/B/C/D, measured vs reported always labelled, date + commit + clocks), packs table with quality, precision profiles, the link, attribution (TensorFold MIT, exllamav3, b12x RoCE Apache-2.0, Xiaomi MIT, benthecarman pack, Kolibri patches), rollback (GLM `./start.sh`). Push to `github.com/ajensenwaud/MiMo-V2.6-2x-DGX-Spark-TensorFold` only when Anders says so; offer the family upstream as a PR against `python-0.6`.

---

## Verification evidence per gate (what counts)
- Startup log lines asserting: family `mimo_v2`, pack bpw map, attention plans, KV dtype, ranks, graphs captured, drafter loaded, capacity.
- `/health`, `/v1/models`, `scripts/verify.sh` (≥ 500 tokens, `finish_reason stop`) run on gx10 over ssh.
- Exact-token parity (Gates 2, 4), KL + eval slice (1, 3, 5), rank bit-equality (4), `draft: false` equality (6).
- Every `results/<arm>/` has `tensorfold-commit.txt`, `serve-config.txt`, `health-before.json`, `health-after.json`, `clocks.txt`, raw JSON; `scripts/summarise.py` builds the README table from them.

## Risks and fallbacks
- **Engine effort** is the critical path; time-box Gate 2 (~3 sessions) with the exllamav3 interim-release fallback stated above.
- **The FP8 "original" is already quantised** (mxfp4-stored experts): say so wherever "lossless" appears.
- **Layer-47 overflow** must be handled in both the quantiser and the engine; a missing restore shows up as garbage only on some prompts — test layer 47 explicitly in Task 2.3.
- **gx10 disk** (223 GB): the 4.0 pack needs the Kolibri checkpoint parked — ask first.
- **TP=2 may not beat one Spark at equal bpw** if collective latency dominates 48 layers; Task 4.1 decides before the port is finished. The quality argument (higher bpw only fits on two) stands regardless.
- **exllamav3 tag drift**: pin whichever tag loads the 2.27 pack and runs its eval; record it in `config.sh` and RECON.md.
- **DFlash tap semantics**: confirm against `dflash/dflash.py::extract_context_feature` (which hidden state — post-layer residual — and whether the tap includes the final norm) before wiring the taps.

## Effort (GPU-exclusive from Gate 1)
Gate 0: 1 session · Gate 1: ½ session + bench · Gate 3: 1 h pilot + 2 x 10–14 h unattended on gx11 while Gate 2 runs on gx10 · Gate 2: 2–4 sessions · Gates 4–7: 1–2 each · Gate 8: 1.
