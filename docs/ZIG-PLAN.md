# Zig port: plan (2026-10-11)

**Goal:** serve MiMo-V2.6-Flash on 2x DGX Spark on TensorFold's Zig engine, faster than the Python recipe on every
measured cell, with replies identical to the Python engine's. Ship it as TensorFold patches plus a recipe, the way
Jay Leaton shipped DeepSeek-V4.1-Flash.

**Read first:** [ZIG-RECON.md](ZIG-RECON.md) (facts), then this file. The Python recipe (README, `patches/0024-0030`)
is the reference implementation and stays the production serve until Z8 passes.

## Decisions

Confirmed by Anders on 2026-10-11 ("go"): D1 as below; D2-D7 as recommended.

| # | Decision | Recommendation |
|---|---|---|
| D1 | Engine base | **Decided by Anders (2026-10-11): TensorFold v1.0.5 (`bec00ae`), not Jay's 1.0.2 export.** Jay's generic pieces that 1.0.5 lacks are ported onto it in gate Z0B: the keyed graph cache and device arena (his #635), CUDA tensor parallelism (#636), the EXL3 mul1 CUDA kernels (unsubmitted part of his stack), and later sessions (#637). #635-#638 conflict with `main` today, so we rebase them ourselves. New upstream releases are taken by rebase at a gate boundary, never mid-gate. |
| D2 | Exactness bar | **Bit-exact with the Python engine** on the same pack: same logits at greedy, same SASS where kernels are shared. Then the published quality evals carry over and need no rerun. |
| D3 | GPU windows | **Anders (2026-10-11): take the GPUs down any time the serve is not in critical use** (check `/health` requests_running == 0 and no established :8891 connections first; restart with `~/mimo/recipe/start.sh` on gx10 after). |
| D4 | Where it lives | Same repo, `zig/` (family source, kernels, launcher) + `patches-zig/` (TensorFold commits as `git format-patch`), README section added at release. Published only on an explicit yes. |
| D5 | Toolchain | Zig 0.17.0 inside a build image layered on the existing base image (nvcc 13.3.73, Triton 3.7.1 already there). No host installs. |
| D6 | Upstream contact | Tell Jay (public GitHub issue or X) once Z2 passes, not before; credit him in every artefact. Contributing the family upstream needs lint cleanup and is out of scope until Z8. |
| D7 | Pack | The shipped 3.0 bpw pack (`f2d8d100101b`). No requantising during the port. |

## Standing rules (verbatim for the executor)

- Background launches: `ssh -n host 'cd ... && setsid nohup <cmd> > <log> 2>&1 </dev/null & disown; echo launched' </dev/null`, nothing chained after it; poll in short separate calls.
- `pgrep -af "[n]ame"` (bracketed) to avoid matching your own shell.
- Before stopping any serve, check clients in its own call (`/health` `requests_running`, `ss -tn state established | grep :<port>`); never in the same call as the stop.
- Curls to gx10/gx11 run over ssh on that node. `nvidia-smi` memory reads `[N/A]` on GB10: use `/proc/meminfo` and `--query-compute-apps`.
- No host installs (apt/pip/npm/zig), no sudo, no firewall, grub or network changes. Containers only.
- One commit per task in the engine tree; export with `git format-patch <base>..HEAD` into `patches-zig/`. Never push unprompted.
- Every lever is default-off behind a `TF_MIMO_*` switch until measured; the serving profile opts in.
- Speed numbers only on idle GPUs with both ranks free; never beside a build or a conversion.
- Every number cites its hypothesis ID and its `results/zig-<arm>/` directory.

## Hypothesis register

| ID | Class | Hypothesis | Measured by |
|---|---|---|---|
| H1 | Platform | Host-side Python overhead per round (scheduler, sampling, graph replay submission) costs measurable time at 1 and 4 streams; the Zig host loop removes it | round profile, Python vs Zig, same pack |
| H2 | Multi-device | The round plan as an RDMA write + flag (Jay's mailbox) cuts rank 1's entry lag and the exchange wait (2.0 ms a row today) | `TF_MIMO_PLAN_LINK=rdma` A/B |
| H3 | Kernels | Jay's `x3ld` decode experts beat the grouped kernel MiMo uses today on MiMo's shapes (top-8 of 256, 2048 wide) | per-kernel bench, cold DRAM |
| H4 | Kernels | A warp-per-expert router GEMV with top-8 fused beats the Triton router (Jay: 112 to 25 us a layer) | per-layer bench |
| H5 | Memory placement | L2 prefetch of the next dense group (12 MiB a site, paced beside exchanges) hides dense-projection latency (Jay: +2.5-3.5%) | 1-row window ms |
| H6 | Prefill | A dense EXL3 prompt GEMM with the trellis decoded inside removes the 1.6 s of weight decoding a 16K chunk | 16K chunk profile |
| H7 | Prefill | Larger, memory-adaptive prompt pieces (2,048-4,096 rows) raise prefill (Jay: 32K +16%) | cold prefill sweep |
| H8 | Generation algorithm | Boot-time calibration of every verify-row cost raises prose drafting (Jay: +2.8%); MiMo prose gains ~0 from DFlash today | prose tok/s, 1 and 4 streams |
| H9 | Scheduling | Multi-stream rounds lose to host work and per-stream serial chains; Zig row graphs and keyed graph cache close part of 91 vs 83 | 4-stream aggregate |
| H10 | Representation | bf16 exchanges where partials are already bf16-rounded halve exchange bytes without changing bits | bit-equality + exchange ms |

## Gates

Each gate lists tasks, the pass condition, and its results directory. Effort assumes one focused session a day.

### Z0. Scaffold and toolchain (1-2 sessions, no GPU)
- Z0.1 Clone `ashhart/TensorFold` into `~/mimo/zig-engine` on gx10; branch `mimo-zig` from tag `v1.0.5` (`bec00ae`). Clone Jay's repo at `199ea20` beside it as the source to port from (read-only).
- Z0.2 Build image `mimo-zig-build` on the base digest + Zig 0.17.0 (pinned tarball, checksum verified; confirm 1.0.5's `.zig-version` first).
- Z0.3 `zig build` and `zig build test` on plain v1.0.5 in the container. Pass: upstream's own suite green, recorded as the baseline step count.
- Z0.4 Fatbins for 1.0.5's CUDA kernels for sm_121 (`-Dnvcc`). Pass: Nemotron's `tf-cuda-test` host-side checks pass.
- Z0.5 Read and note the family interface on 1.0.5: `zig/src/server/engines.zig` registration, Nemotron's CUDA family as the in-tree example, `cuda/qlinear.zig` (one weight view per format), `cuda/grouped.zig` (shared routed-expert plan), `core/exl3_format.zig` (reference decoder), `cuda/aot.zig`/`triton.zig`. Write `docs/ZIG-INTERFACE.md` (one page).

### Z0B. Jay's infrastructure onto 1.0.5 (5-8 sessions, mostly no GPU)
Port by concern, one commit each, Jay credited in every commit message and file header he wrote.
- Z0B.1 Keyed graph cache + device arena: rebase #635 (`jayleaton:stack/02-graph-cache-arena`, +890) onto `v1.0.5`. Pass: its host tests.
- Z0B.2 CUDA tensor parallelism: rebase #636 (`stack/03-tp-fabric`, +6,004, 38 files) onto 1.0.5's existing `zig/src/fabric` (verbs, mailbox, collectives already upstream). Pass: its host tests; then a two-node GPU check: `fabric_sendrecv` / `linktest` between gx10 and gx11 over the CX7 link, and the RoCE one-shot all-gather matching NCCL bit for bit at decode and prompt sizes.
- Z0B.3 EXL3 on CUDA: port the mul1 kernels from Jay's export (`exl3_experts.cu`, `x3ld.cu`, `x3pf`/`x3gm`, `linear.cu`, `lin2.cu`) as an EXL3 weight view in `cuda/qlinear.zig` and an EXL3 path in `cuda/grouped.zig`, following 1.0.5's "one home per format" rule. Pass: every kernel's output equals `core/exl3_format.zig`'s reference decode on rectangular and expert tensors from our pack, and equals the Python engine's EXL3 ops on Z1's fixtures.
- Z0B.4 Sessions (#637) are deferred to Z5.2; grammar (#638, 53K lines with vendored xgrammar) is out of scope.
- Offer: once these pass, tell Jay the rebased branches exist (D6); they are most of what his stack needs to land.
- Order: Z1 (oracle capture on the Python serve) runs alongside Z0B; Z0B.3's last check needs Z1.2's fixtures.

### Z1. Oracles from the Python engine (1-2 sessions, short GPU windows)
- Z1.1 `tools/mimo_zig_capture.py`, modelled on `dsv41_m1_capture.py`: for 8 fixed prompts (EN, ZH, code, tool call, long 32K), capture at TP=2 on both ranks: embeddings, every layer's input/output residual, attention outputs, router picks and weights, expert outputs, logits, and the greedy tokens for 256 steps. Store under `fixtures/` with a manifest and hashes.
- Z1.2 Per-kernel fixtures for each of the 11 Triton kernels and each EXL3 call (inputs, outputs, launch config) from one decode row and one 2K prompt chunk.
- Z1.3 Triton AOT fill for MiMo's kernels with `triton_fill.py`: list every shape the serving profile emits (`aot-needs` equivalent). Pass: every MiMo kernel compiles to a cubin for sm_121; manifest validated.

### Z2. M1: the checkpoint runs exact on Zig, TP=2 (critical path, 4-7 sessions)
- Z2.1 Family skeleton `zig/src/families/mimo_v2/`: config (`model_type` mimo_v2), registration, `load.zig` reading the EXL3 pack (q/k/v split, 192-padded v, sinks, router bias, MTP and vision ignored).
- Z2.2 Dense EXL3 linears and routed experts through Z0B.3's EXL3 view and grouped path, mirroring the Python engine's launches. Pass: per-kernel fixtures bit-equal.
- Z2.3 Attention: the 6 Triton attention kernels via AOT (full and sliding with sinks), partial NEOX RoPE via `norm_rotary.cu` or AOT, the sliding-window ring KV (16,512 keys) and the full-attention KV. Pass: per-layer attention outputs bit-equal.
- Z2.4 Router (sigmoid + bias top-8), dense layer 0, norms, residual, lm_head. Pass: per-layer residuals bit-equal.
- Z2.5 TP=2 through Z0B.2's fabric: heads and expert width split as in `tp.py`, RoCE one-shot all-gathers in the same order. Pass: both ranks' logits bit-equal to Python at every step.
- Z2.6 Greedy decode loop, 1 stream, eager. Pass: 256 greedy tokens identical to Python on all 8 prompts.
- **Fallback:** if Z2.6 is not reached within 3 sessions after Z2.5, stop and report: which layer diverges, by how much, and whether the cause is a kernel we cannot mirror. Anders decides between a numeric gate (KL within the Python engine's own self-noise floor, as in the Python port) and pausing.

### Z3. M2: decode windows, graphs, streams (2-3 sessions)
- Z3.1 Decode rows for up to 4 streams; keyed graph cache (#635 pattern) by total rows. Pass: graphs == eager bit for bit; batched == alone.
- Z3.2 Prompt path on decode windows (correct, not fast yet). Pass: a 32K prompt gives Python's first 64 tokens.
- Z3.3 Arm `zig-plain`: spark-bench decode + depth, plain (no drafter). Compare with Python plain (README `plain-*`).

### Z4. DFlash drafting (2-3 sessions)
- Z4.1 Drafter forward in Zig: 5 layers, taps from target layers 0/11/23/35/47 inside the target forward, 8-token block in one graphed pass through the head.
- Z4.2 Verify windows and joint window widths across streams, reusing Jay's row/window machinery; boot calibration (`TF_MIMO_CALIB=zig`, H8).
- Pass: every drafted reply byte-identical to `"draft": false` on 16 prompts at 1 and 4 streams. Arm `zig-dflash`.

### Z5. M3: serving (2 sessions)
- Z5.1 Register with the shared lane server: OpenAI chat/completions, streaming, MiMo chat template, tool-call markup, thinking on/off, cancel, drain, keepalive, `/health` counters (requests, drafted/accepted, cached tokens) so spark-bench's contention and acceptance checks work.
- Z5.2 Sessions / prefix reuse (#637 pattern) for the full-attention layers; sliding layers recomputed or ring-restored.
- Z5.3 Launcher `zig/start.sh` / `stop.sh` on `:8892` (Python keeps `:8891`).
- Pass: Jay's qualification list on MiMo: greedy parity, drafted == plain, concurrent == solo, resumed == fresh, prompt seams, cancellation, TP failure fails fast. 1 h soak with no errors.

### Z6. Prefill (2-3 sessions)
- Z6.1 Native prompt path: experts on the EXL3 prompt GEMM (Python's current method), then H6 (fused trellis prompt GEMM, `pfdense`), then H7 (adaptive pieces).
- Pass per lever: prompt rows bit-equal where arithmetic is unchanged; otherwise wikitext perplexity within 0.1% and the paired evals.

### Z7. Levers (one at a time, 3-5 sessions)
Order: H2 plan link, H3 `x3ld`, H4 router GEMV, H5 L2 prefetch, H10 bf16 exchanges, H8 calibration v4, H9 multi-stream. Each: default-off switch, exactness gate, spark-bench before/after on idle GPUs, kept only if it wins. Results in `results/zig-<lever>/`. Skip what Jay measured as no gain (draft trees, PDL, lossy verify, joint depth mode 2) unless MiMo's numbers argue otherwise.

### Z8. Release (2 sessions)
- Z8.1 Same-night A/B, Python vs Zig, best of 3 warm runs each, Jay's table shape: code and prose x1/x4, 32K and 128K prefill, 32K/128K decode, mixed windows. spark-bench full on both.
- Z8.2 Quality: if bit-exact, cite the Python evals; otherwise rerun GSM8K/MMLU-Pro paired (McNemar).
- Z8.3 Export `patches-zig/`, launcher, `docs/ZIG-SERVE.md`, README section, CREDITS (Jay Leaton first among engine credits, plus everyone already credited).
- Z8.4 Publish on an explicit yes.
- Z8.5 As Jay's stack merges upstream, replace our rebased copies of #635/#636/#637 and the EXL3 port with the merged versions; at each new TensorFold release, rebase at a gate boundary and rerun Z2.6, Z4 and Z8.1.

**Release bar:** Zig >= Python on every cell of Z8.1 and replies identical. Targets from Jay's measured deltas:
1 stream +2-3%, 4 streams +6-9% (91 to about 97-100), prefill +10% (H6/H7 aim higher: 1,180 to 1,400+ at 16K).

## Effort

About 25-38 focused sessions (4-6 weeks at one a day); building on 1.0.5 adds Z0B's 5-8. Critical path: Z0B.2-Z0B.3
(TP and EXL3 on 1.0.5), then Z2 (exact attention and TP for MiMo). The rest
parallelises: code-only work (Z5 server glue, Z6 kernels) proceeds while GPU windows run Z2-Z4.

## Results layout

`results/zig-<arm>/`: `engine-commit.txt`, `serve-config.txt`, `health-before.json`, `health-after.json`,
`clocks.txt`, raw JSON, spark-bench `summary.json` and card.

## Progress log

### 2026-10-11 (session 1, no GPU used; Python serve on :8891 untouched)
- Z0.1-Z0.3 done. Engine tree: `~/mimo-zig/zig-engine` (clawdy, branch `mimo-zig` from `v1.0.5` `bec00ae`), mirrored to
  gx10 `~/mimo/zig-engine`. Build image `mimo-zig-build:0.17.0` on gx10 (base digest + Zig 0.17.0, checksum
  verified; `zig/docker/Dockerfile.build`). Plain v1.0.5: build 8/8, tests 371/371.
- Z0.4 done: 24 fatbins for sm_121 built clean.
- Z0.5 done: `docs/ZIG-INTERFACE.md`.
- Z0B.1 done: #635 rebased onto v1.0.5 (one conflict in `cuda/root.zig`, resolved: imports of files 1.0.5 lacks
  dropped). Finding: Jay's PRs carry upstream commits merged after 1.0.5; only his own commits were cherry-picked
  (`git cherry-pick -x`, authorship kept): #634 `4284048d`, #635 `986c9960`, #636 `5fa8ad0f`, `0824e393`, `db3bf32f`.
- Z0B.2 host half done: TP stack applies cleanly; build 37/37, tests 401/402 (1 skipped: needs a GPU). Two-node GPU
  check (linktest over the CX7 link, RoCE == NCCL) waits for a GPU window.
- Z0B.3 kernel half done: `zig/kernels/cuda/mimo_v2/sync.py` generates the EXL3 kernels from the Python engine's
  sources at `f3a8c11` (verified identical by hash to the files inside the running `mimo-tf` serve). `sass_check.sh`:
  **140 of 140 kernels SASS-identical to the Python build** (linear 109, experts 7, grouped mul1 24). Remaining:
  Zig launch wrappers + the EXL3 arm of `qlinear.zig`, and the GPU check against `core/exl3_format.zig`.
- Patches exported to `patches-zig/0001-0008`.

- Z0B.3 launch half done: `zig/src/cuda/exl3.zig` (the 75 mul1 instances resolved by mangled name; launches mirror
  the bindings' grids, blocks, shared memory and checks; `Kernels.routed` = experts.routed). Host tests: linear.plan
  and experts.default_config equal the Python engine's at MiMo's TP=2 shapes (`plan_vectors.py`), symbols equal the
  fatbins'. All host tests: 423/424 (1 GPU skip).
- GPU check built, not yet run: `tf-cuda-test mimo-exl3 <shard> <prefix>`: dequant (stored) and unpack (strips)
  bit-equal to `core/exl3_format.zig`, linear M=1/4/8/16 against the fp64 reference.

### 2026-10-11 (session 1, GPU window ~35 min; serve stopped idle, restarted and answering)
- **Z0B.3 passed.** `mimo-exl3` on the 3.0 pack: dequant (stored) and unpack (strips) bit-equal to
  `core/exl3_format.zig` at K2 6 / 8 / 10 (expert gate and down, dense gate, q_proj); linear M=1/4/8/16 within
  1.6-2.5e-4 relative L2 of fp64. `mimo-exl3-fixtures`: **67 of 67 cases byte-equal to the Python engine**: q, o,
  dense gate/down at rank-0 TP cuts x fp16/bf16 inputs x fp16/bf16/f32 outputs x M 1/4/8/16, and routed experts
  (layer 1, 256 experts, 8 slots, R 1/4/16). Answer key: `zig/tests/cuda/oracle/mimo_exl3_fixtures.py` run in the
  release image (the serve's own extension build).
- **Z0B.2 passed** between gx10 and gx11 over the CX7 link: roce-host 1000 gathers bit-exact; RoCE GPU suite 378/378;
  NCCL suite 189/189; fail-fast 8/8 for each backend (survivor exits 70 within 1.1-3.6 s). Needs `-Dtp-nvcc` at build
  time (else `NoKernel`).
- Logs: `results/zig-z0b/`.

### 2026-10-11 (session 2; serve down ~12 min while idle, restarted)
- **Z1.1 done.** `tools/mimo_zig_capture.py` (+ `.sh` for both ranks): 8 prompts fed one row at a time through
  decode forwards, then greedy to 256 tokens or EOS; 1,329 decode steps. Per rank: argmax at every step, fp32 logits
  for the first 16 steps, residuals after every layer's attention and MLP plus routed picks/weights for steps 0-3,
  attention outputs at the first full (0) and first sliding (1) layer. On gx10 `~/mimo/oracle/rank0.safetensors`
  (sha256 ce588994042b5a13...) and `peer/rank1.safetensors` (117297532cfeb30b...). Checks
  (`tools/mimo_oracle_check.py`): run 1 == run 2 on 6,144 tensors; rank 0 == rank 1 on all 6,112 rank-shared tensors;
  argmax equal on both ranks at every step. (Correction: 1,905 decode steps in all, prompts included, not 1,329.)
- **Z1.3 done for the decode path.** Triton launches recorded with upstream's `tools/zig/triton_aot_manifest.py`,
  packed with `aot_pack.py`: 13 specialisations of 8 functions (_rms, _add_rms, _qkv x2, _rows x3, _merge x3,
  _router, _topk, _silu_mul), identical on both ranks. Kernel set: gx10 `~/mimo/oracle/aot-sm121/` (aot.json
  bf36ead545e8ee81...). Prompt-path and DFlash kernels are captured later (Z3.2, Z4).
- Z2.1 started: `zig/src/families/mimo_v2/config.zig` (Config.read + rank cuts), 3/3 host tests (`zig build test-mimo`).
- Build containers now run as uid 1000 (`--user 1000:1000 -e ZIG_GLOBAL_CACHE_DIR=/mimo/.zig-global`); outputs of
  root containers are handed back with `docker run ... chown`, not sudo.

### 2026-10-11 (session 3; serve down ~7 min while idle, restarted)
- **Z2 / gate M1 passed on the first run.** `zig/src/families/mimo_v2/`: `load.zig` (weights.load: strips EXL3,
  stored-layout experts with pointer tables, fp32 norms, router fp16(bf16(stored)), fp32 sinks, bf16 embeddings;
  56.1 GiB a rank, 155 s), `forward.zig` (Model.forward decode rows: 13 captured Triton cubins, EXL3 via
  cuda/exl3.zig, `glue.cu` for torch's embed / rank-order add / fp64 rope tables, TP sums by all-gather + add),
  `m1_main.zig` (`tf-mimo m1`), `m1_run.sh`. Against the answer key on both ranks, TP=2 over the CX7 link (NCCL):
  **argmax 1,905/1,905 steps, fp32 logits byte-equal 128/128, 3,136 layer captures byte-equal** (residual after
  every layer's attention and MLP at steps 0-3, attention outputs at layers 0 and 1). Logs: `results/zig-m1/`.
- Known nit: rank 1 exits 70 through fail-fast when rank 0 finishes first (no stop announcement yet).

### 2026-10-11 (session 4; serve down ~12 min in two idle windows, restarted)
- **Z3 decode windows + graphs passed on the first run.** `forward.zig`: rows of up to 4 slots at their own positions
  (`Row`), staged inputs, `run` (capture-safe: no host work), `graphed` (graphs.Graphs: key (rows, bucket), warm-up
  run then stream capture, chains wider than 32 rows eager). Python reference `tools/mimo_zig_windows.py`: 260
  rounds of 4 streams teacher-forced from the answer key in widths 1-32 a stream (4-64 rows a window), alternating
  graphed / eager cycles. Python itself: batched == serial (argmax 1,905/1,905, logits hash 128/128), rank 0 == rank 1.
  **Zig (`tf-mimo windows`, both ranks): logits hash-equal to Python's window 1,905/1,905, argmax equal to the serial
  answer key 1,905/1,905; 136 graphed rounds = 17 captures + 119 replays, as Python.** Kernel set `aot2-sm121`:
  23 Triton specialisations (decode 1-row set included). Logs: `results/zig-z3/`.
- Scope change: Z3.2 (prompt path) moves to Z6. Python serves prompts on its prompt kernels (EXL3 prompt GEMM,
  full-attention prompt kernel, per-expert prompt GEMM), whose bits differ from decode rows, so an exact Zig prompt
  path means porting those, not reusing decode windows. Z3.3 (spark-bench arm) needs the server: after Z5.

### 2026-10-11 (session 5; serve down ~25 min while idle, restarted)
- **Z4 DFlash passed.** `dflash.zig` (Drafter.update / Drafter.draft: EXL3 via cuda/exl3.zig, _rms / _silu_mul from the
  captured set, target head + gather) and `dflash.cu` (block embed, per-head norm + NEOX rope, windowed attention with
  sinks, argmax; glue flags). The target forward taps layers 0/11/23/35/47 and feeds the drafter inside the graphed
  sequence, as attach_drafter does.
  - `tf-mimo dflash` vs `tools/mimo_zig_dflash.py` (Python drafter, same protocol): **draft tokens equal 13,275/13,279
    (99.97%), whole blocks 1,893/1,897; accepted 6,035 vs 6,035, equal per prompt.** Not bit-exact by design: the
    drafter's torch glue (mean, einsum, softmax) is reimplemented; it only proposes, the target verifies.
  - `tf-mimo drafted` (4 streams, verify windows graphed): **8/8 replies byte-identical to the plain answer key, both
    ranks**; 1,645 tokens in 191 forwards (prompt windows included), drafts accepted 1,166/3,257 at fixed width 8.
  - Kernel set `aot4-sm121` (28 variants): `tools/mimo_zig_variants.py` records every decode-path specialisation
    (row counts 1-128, 1/2/16-chunk classes, draft for 1-4 streams). Found when a 32-row window needed `_router`
    BM=32 with M%16==0, which no earlier run had launched.
  - Regression: Z3 windows gate re-run on this code + aot4: 1,905/1,905, both ranks.
- Open nit: rank teardown exits 70 via fail-fast when the peer finishes first; fix with the server's stop message (Z5).

### 2026-10-11 (session 6; serve down ~30 min while idle, restarted)
- **Z5 first slice: MiMo served by the Zig engine's own server (`tensorfold-native serve`), both Sparks.**
  `native.zig`: the lanes backend (prefill as 128-row decode windows, keyed host draws, held DFlash drafts, `keep` a
  no-op: rejected rows are overwritten), load-time costs (windows 1-8 rows 24.1-47.1 ms, shared 32 rows 65.8 ms, draft
  6.0 ms); rank 0 sends each GPU step over the TP plan link, rank 1 runs `tf-mimo follow`. Registered in
  `native/cuda.zig`; two upstream touches: families without snap_* serve without the prompt cache, and TP families
  admit their rank's weight share (`rankWeights`). Launch: `tools/zig_serve.sh` (:8892).
  - Text: `/tokenize` on 20 conversation cases (tools, tool round trip, thinking on/off, efforts, Chinese, emoji,
    2,901-token prompt, completions): **20/20 prompt ids equal to the Python server's.**
  - Replies (`tools/mimo_zig_replies.py`): the answer key's 8 prompts over HTTP, greedy: **8/8 equal serially and
    8/8 with 8 concurrent requests on 4 streams.**
  - First spark-bench read (quick preset, decode only, 256 tokens; not like-for-like with the Python baseline's
    preset): 1 stream code 64.0 / chat 44.4 / agent 51.8 / report 50.6 tok/s; 4 streams aggregate code 94.1 /
    chat 61.5 / agent 73.1 / report 70.3. Python baseline: code 83.0, chat 52.9, agent 60.6, report 50.2; 4 streams
    code 91.4, chat 85.1. Single-stream gap is the expected one before Z7 (host argmax/sampling over 152k logits and
    8 logits rows copied each round; lanes' draft widths not tuned to MiMo).
- Decision (Anders, 2026-10-11): no same-preset race against Python; benchmark the Zig engine as it stands.

### 2026-10-11 (session 7; Python serve down from 17:40 for the full context, bench and soak)
- Full context: the Zig server serves 608,256 tokens a stream x 4 (as Python), model + caches 100.9 GiB on rank 0.
- Qualification (`tools/mimo_zig_qual.py`, Python's answers recorded first on :8891; Python repeats itself 6/6):
  - greedy parity: answer key 8/8 serial and concurrent at full context; a 14,602-token prompt's reply equal to the
    Python server's.
  - sampled: 3/6 equal to the Python server, 6/6 repeatable on Zig. Cause: Python prefills prompts with its prompt
    kernels (different bits from decode rows), so later logits differ in the last bits; greedy argmax survives,
    top_p's cut does not always. Sampled parity moves to Z6 (exact prompt path). Zig's request defaults set to
    Python's CUDA server's (top_k 20, top_p 0.95).
  - drafted == plain, concurrent == solo: the replies gate (Z4/Z5). resumed == fresh: n/a (no prompt cache yet).
  - cancellation: 3 streamed replies dropped after 16 chunks; streams freed, canary equal to its solo reply.
- spark-bench full (`results/zig-z5/spark-bench-full.json`), tok/s a stream (aggregate):
  chat 46.1 / 26.0 (47.5) / 19.2 (68.5) at 1/2/4 streams; code 72.8 / 46.4 (89.7) / 27.2 (95.2); agent 50.0 /
  36.8 (66.5) / 21.8 (78.4); report 48.1 / 29.1 (54.6) / 19.0 (72.2). Depth (prefill, decode): 4K 418 / 37.6,
  16K 398 / 41.8, 64K 351 / 35.8, 128K 317 / 32.8. Quality: GSM8K 96.0% (192/200), MMLU 86.0% (196/228).
- Soak (`tools/mimo_zig_soak.py`, results/zig-z5/soak.json): **722 requests in 1 h (greedy, sampled, streamed and
  dropped, completions, 2-7K prompts), 0 errors, canaries 12/12.** Fail-fast: follower killed mid-reply, the request
  failed (HTTP 500) 1.4 s later and rank 0 exited 70 for a clean restart; no hang. **Z5 passes** (sessions/prefix
  reuse deferred with the prompt cache; sampled parity moves to Z6).

### Z7 arms, first two (session 7; `tools/zig_arms.sh`, spark-bench full preset's decode cells, 608K context)
| arm | chat 1 / 4 agg | code 1 / 4 agg | agent 1 / 4 agg | report 1 / 4 agg | gate |
|---|---|---|---|---|---|
| Z5 (host argmax, NCCL) | 46.1 / 68.5 | 72.8 / 95.2 | 50.0 / 78.4 | 48.1 / 72.2 | replies 8/8 |
| + device argmax | 46.8 / 72.5 | 83.4 / 112.5 | 50.1 / 88.4 | 48.8 / 79.6 | replies 8/8 |
| + device argmax + RoCE | 45.6 / 74.7 | 60.8 / 114.9 | 53.8 / 90.7 | 47.0 / 77.6 | replies 8/8 |
- **Device argmax kept (default on; TF_MIMO_HOST_ARGMAX=1 is the old path):** 4-stream aggregates +6 to +18%,
  single stream flat within noise. Single-run 1-stream cells swing +-15% (code 60.8 vs 83.4 on the same build):
  judge levers by the 4-stream aggregates and the measured window costs, or repeat runs.
- **RoCE: no gain** (1-row window 24.38 vs 24.32 ms); NCCL stays the default.
- Python (same preset, earlier night): code 83.0 / 91.4, chat 52.9 / 85.1. Zig now ahead on code and agent at 4
  streams; behind on chat (1 and 4 streams) and on prefill (Z6).

### Z7, session 8 (Python serve down in short idle windows, restarted after each)
- Round profiler (`TF_MIMO_PROFILE=1`): no time outside backend calls (0.00 ms a round); serving windows cost more
  than load-time costs said (2.8-row verify 36.2 ms vs ~29 predicted; 4 streams 16.9 rows 99 ms vs 32 rows 68 ms).
  Cause: costs were timed with one token in every row, so all rows shared 8 experts. **Fixed: distinct pseudo-random
  tokens per row** (8 rows 59.7 ms, shared 32 rows 141.8 ms). Served: 2 streams +2 to +10%, 1 and 4 within noise.
- `tf-mimo timing` (no plan link, both ranks, position 9,000, context 32K): graphed 1 row 23.99 ms (drafter +0.5 ms;
  argmax readback 0.06 ms) vs the Python engine's 21.6 ms on this pack (results/tp2/graphs-3.0.json: 46.27 tok/s).
- **The RoCE arm had been NCCL**: sessions opened without the mailbox kernel (`tp roce: device setup failed:
  NoKernel`, silent fallback). Fixed: `tp_kernel` (the mailbox fatbin, -Dtp-nvcc) shared by tf-tp-test and the MiMo
  modules. RoCE with fast pollers (TF_TP_ROCE_FAST=1): **1 row 22.43 ms (-6.5%)**. Exactness: windows gate 1,905/1,905
  on both ranks, replies 8/8.
- **Kept: RoCE fast pollers by default** (`tools/zig_serve.sh`). Served, tok/s a stream (aggregate) at 1 / 2 / 4:
  chat 49.9 / 32.0 (61.5) / 21.4 (77.9); code 75.4 / 50.4 (94.6) / 31.8 (119.5); agent 55.0 / 41.2 (74.6) / 23.4
  (84.7); report 50.5 / 32.0 (61.9) / 22.5 (85.4). Python: chat 52.9 / 85.1 agg, code 83.0 / 91.4, agent 60.6,
  report 50.2. Zig ahead: code and report at 4 streams (+31%, n/a), report at 1. Behind: chat (-6% / -8%), agent and
  code at 1 stream.
- Open: draft pass 7.9 ms on RoCE vs 6.5 on NCCL, and 32-row shared windows 165 vs 142 ms: large messages (the
  drafter's head gather, ~2 MB) look slower on the mailbox path; try TF_TP_MAILBOX_MAX_KB (Python: 512) and graphing
  the draft pass (Python: 7.0 -> 6.05 ms). Then the per-row gap to Python (22.4 vs 21.6 ms) and Z6.

### Z7, session 9: the single-stream gap (Anders: "why slower on one stream, code and chat?")
- Same-night A/B, 3 runs a cell (median tok/s, 1 stream): Python chat 49.4 / code 78.7 / agent 58.9 / report 51.0;
  Zig lanes rule 46.6 / 66.6 / 55.4 / 47.7; Zig geometric rule 45.4 / 70.7 / 51.4 / 48.4. 4 streams aggregate: Python
  81.8 / 85.7 / 80.1 / 85.1; Zig lanes 72.2 / 110.2 / 83.4 / 78.3; geometric 80.3 / 106.5 / 89.5 / 83.5.
  Single runs swing +-15%: compare medians only. The older Python baseline (code 83.0, chat 52.9) was a different night.
- Rounds (logs): Zig drafts deeper than Python (code 5.2 vs 4.0 drafts a round, chat 2.6 vs 1.0) at about equal tokens
  a round, so it pays more a round. Python's joint_widths rule ported to lanes (`geometric_depth`, TF_MIMO_DEPTH=geometric,
  lone streams only): **no gain**, kept off.
- Draft attention rewritten (a block per row and KV head, keys and values shared by the group): draft pass at position
  9,000 9.75 -> 6.26 ms (4 streams 24.4 -> 12.1); drafts 99.97% equal to Python, drafted replies 8/8. Kept. Served
  short-prompt cells unchanged (the old kernel was cheap at low positions).
- **The cause is the forward itself:** Python 3.0 pack (`results/zig-z7/python-profile-3.0.log`, torch.profiler): 1 row
  wall 21.8 ms with 24.3 ms of kernels (2.5 ms overlapped), 8 rows 59.0. Zig graphed: 1 row 22.4-24.5, 8 rows 61.0.
  Zig eager breakdown, 1 row: dense 9.2, experts 10.1, comm 1.5, attention 1.4, head 1.4, router 1.1, norm 0.6, add 0.4;
  8 rows: experts 42.7 vs Python 43.4, dense + head 11.4 vs 12.7. Kernels match; at 1 row Python overlaps ~2.5 ms
  (its RoCE gather runs alongside compute) and Zig runs everything in order.
- nsys records no CUDA trace for the Zig binaries (dynamically linked, driver by dlopen; Python traces fine): open.
- Fixed: tf-mimo announces the stop before closing the TP session (no exit 70 when a rank ends first).

**Next:** get a graphed per-kernel timeline of the Zig forward (fix nsys injection, or event nodes in the graph), find
where Python overlaps at 1 row, mirror it (likely the RoCE gather's launch/stream arrangement in roce.py).

### Z7, session 10: single stream, measured on fixed texts
- nsys: hardware CUDA trace records nothing on these GB10s (Python or Zig); `--trace=cuda-sw` works (m1_run.sh NSYS=).
  Eager 1-row kernels, Zig vs Python (torch.profiler): linear 9.23 / 9.15, grouped experts 8.84 / 8.73, the rest
  within 0.1 ms; Zig adds mimo_add2 0.11. 8 rows: experts 40.1 / 41.2, linear 9.9 / 10.2. **Kernels are level.**
- Correction to session 9: Python's "2.5 ms overlap" was its profiler (wall timed without it, kernels with it).
- Verify cost curve, position 9,000, pseudo-random tokens (Python `mimo_profile.py --widths --random`, Zig timing):
  1 / 4 / 8 rows Python 22.5 / 40.8 / 57.8 ms, Zig 22.7 / 42.6 / 60.8. Real text costs Python the same (+-2%).
  Serving context (608,256) adds 0.3 ms to Zig's 1-row step. Served Zig steps (plan link, drafter update, position
  64) time 24.4-24.7 ms a row at load.
- `tools/mimo_fixed_decode.py` + `tools/zig_fixed.sh`: spark-bench's decode texts with fixed tags, greedy, 3 a
  workload. Zig's texts repeat exactly across arms; they differ from Python's (prompt rows via the decode path, Z6),
  so Zig-vs-Python per request is approximate. Mean tok/s (chat / code / agent / report):
  Python 48.0 / 71.2 / 60.8 / 52.4; Zig lanes 48.4 / 68.4 / 54.8 / 48.8; geometric depth 43.1 / 63.1 / 54.2 / 44.1
  (rejected); plan link pinned 47.6 / 67.5 / 55.7 / 47.9 and rdma+pinned 49.7 / 69.1 / 54.1 / 48.7 (no gain,
  off); median cost table (two starts) 48.3 / 66.8 / 55.4 / 48.1 and 48.5 / 69.2 / 55.6 / 50.1 (kept for stability,
  no speed change). Code cells depend on the text: Python 65.9-74.9 across three fixed tags.
- Rounds: Python stops drafting in chat (1.5 tokens, 31 ms a round), Zig drafts (2.2 tokens, 46 ms): same speed.
  In agent Zig drafts ~4.8 rows a round at ~50% acceptance against Python's 2.4 rows.
- Remaining gap: agent -10%, report -7%, code -4%, chat level. Next: Z6 (prompt kernels), which makes Zig's texts
  equal Python's so policies compare request for request, then tune depth on matched texts and the 4-5% on
  multi-row verifies.

### Z6: the prompt path (session 11)
- Answer key `tools/mimo_zig_prompt.py` (both ranks equal): 5 scenarios as prompt chunks (40; 700; 1024+1024+552;
  1024+200; 9000 rows), per layer per row fingerprints after attention and MLP, each chunk's last-row logits, 32
  greedy decode rows after; then a prompt-shape sweep for the kernel set. `aot5-sm121`: 42 variants (aot4 + `_gemm` x9,
  `_prompt` x3, `_silu_mul_scale` x2, router tiles past 128 rows).
- Engine `prompt.zig`: prompt GEMM (rot_in + unpack + Triton `_gemm`, Hadamard bf16 table), `_prompt` full
  attention over the slot's cache, sliding layers on `_rows`, MoE per expert from 256 rows (glue.cu
  `mimo_moe_count/order/gather/combine`: torch's bincount, stable argsort, index_select, x * (1/scale), and the
  8-way sum in torch's reduce order ((x0+x4)+(x1+x5))+(x2+x6))+(x3+x7)), grouped kernel below 256 rows, drafter fed
  the chunk's last 1024 rows in 128-row decode windows, head on the last row. `prompt.plan`: Python's lone-stream
  16384-row chunks split to the engine's chunk rows (parts multiples of 16, none under 256).
- **Gate passed first run, both ranks:** fingerprints 768/768 layer captures, last-row logits 8/8 byte-equal,
  greedy 160/160; the serving plan's chunks (9000 -> 3008+3008+2984, 2600 -> one) give the same logits and replies
  (4/4; "mixed" differs by design: Python's 200-row chunk ran the grouped experts).
- Served (`Op.prompt` on the plan link, `TF_MIMO_PROMPT_ROWS`, default 16384: 7.43 GiB scratch, fits at 608,256
  context): **the twelve fixed-text 512-token greedy replies are byte-identical to the Python server's (12/12 shas)**.
  The Z5 replies gate now differs (1/8): its key fed prompts as decode rows; Python's server does not.
- Speed on identical texts (fixed set, 1 stream). Four starts (RoCE fast pollers on/off alternating; no throttling,
  SM ~2,440-2,490 MHz on both nodes) agree within 1%: Zig with Python's depth rule chat 46.7 / code 70.0 / agent 58.3
  / report 49.7 against Python 48.0 / 71.2 / 60.8 / 52.4 (-1.7 to -5%). Same rounds as Python to within ~5%, but
  10-30% more drafted rows (to diff round by round: TF_LANES_DEPTH_LOG=1 here, dev-src decoder patch for Python), and
  1-3 ms more a round. One earlier run at 40 tok/s chat did not reproduce (likely another GPU tenant).
- nsys of a served request (cuda-sw): 1-row verify graph 22.35 ms median (Python's 22.4), 2 rows 29.7, draft 5.8, the
  GPU idle ~0.13 ms before each step: no serving overhead.
- Full spark-bench (Python depth rule): prefill 4K / 16K / 64K / 128K 673 / 1,303 / 1,150 / 1,076 tok/s (Python 764 /
  1,179 / 1,141 / 1,022; Zig Z5 ~317-400); decode at 128K 30.9 (Python 33.4). 4 streams (random tags, two runs a
  cell, lanes rule): prompt path chat 62-76 / code 94-105 / agent 84-97 / report 86-90; decode-row prompts
  78-82 / 113-118 / 94-96 / 89: the prompt path's small-prompt cost shows at 4 streams (first wave worst).
- Default now: Python's lone-stream depth rule (TF_MIMO_DEPTH=lanes|geometric to change); shared rounds keep the lanes
  rule; prompt path on (TF_MIMO_PROMPT_ROWS=0 for decode-row prompts).

**Next:** 4K prefill and small-prompt cost (load every kernel at start; per-call launch overhead of the per-expert
path), decode at 128K (-8%), the drafted-rows difference, the 4-5% on multi-row verifies, then Z8.

## Z7, session 11 (speed, exactness relaxed)

- First copy window 6 rows (was 15): fixed texts chat 46.8 -> 48.2, agent 58.2 -> 60.2. Copies off was worse.
- Prompts under 256 tokens on decode rows (TF_MIMO_PROMPT_MIN): 4-stream code 94-105 -> 117-120 agg. Not
  byte-identical to Python from here; gates are quality (GSM8K/MMLU) and the bench.
- Prompt MoE: unpack + GEMM costs 2.4 s fixed + 0.57 ms a row; the grouped trellis kernel 1.26 ms a row, no fixed
  cost. Grouped up to 3,072 rows in 1,024-row windows (512 rows 2.62 -> 0.74 s). results/zig-z7/ptime.log
- Full bench z7c: 1 stream chat 43.7 / code 76.3 / agent 58.2 / report 48.0; 4 streams 80.9 / 113.4 / 84.8 / 84.9;
  prefill 4K/16K/64K/128K 772 / 1,317 / 1,210 / 1,066; decode at 128K 30.8; GSM8K 96.0%, MMLU 86.0%.

**Next:** fused trellis MMA for big prompt chunks (drop the fp16 unpack); decode at long context (-8%); 1-stream chat.
