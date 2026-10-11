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

**Next:** Z5 (server on :8892: OpenAI API, scheduler, joint_widths draft widths, Jay's test list, 1-hour stress), Z6
(prompt path exact), then speed (Z7/Z8).
