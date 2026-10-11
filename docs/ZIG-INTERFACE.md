# Zig engine interface at TensorFold v1.0.5 (`bec00ae`)

What a CUDA family plugs into, read from the source. For the MiMo port (ZIG-PLAN Z0.5).

## Registration
- `zig/src/native/cuda.zig`: `const registry = .{nemotron.native};` One entry per CUDA family. `families` (model_type
  + formats) feeds `capabilities --json`; `open()` matches `config.json`'s `model_type` and calls `openWith(F, ...)`,
  which sets up the driver, context, device (`TF_CUDA_DEVICE`), context window, segments, memory admission and the
  lane host. MiMo adds `mimo_v2.native` here (`model_type` "mimo_v2", formats e.g. `"exl3-mul1"`).
- `zig/src/server/engines.zig` passes the CLI's options (`context`, `lanes`/`--parallel`, `drafts`, `drafter`,
  `prompt_cache_gib`, `device`, `segments`, ...) to `native.open`.

## The family's `native` module (Nemotron: `families/nemotron/cuda_native.zig`)
Must export: `model_type`, `formats`, `default_context`, `max_segments`, `prompt_rows`, `Options`, the prompt-cache
snapshot vtable (`snap_bytes/save/restore/drop`), and `open(gpa, io, ctx, dir, kernels, Options) !Loaded`.
`Loaded` = the lane backend (`lanes.backend.Backend`), the round loop's model facts (`lanes.Model`), rows a round,
per-stream device bytes, optional lone-stream driver, the engine context and `deinit`, and the prompt-cache target.
Nemotron's open: engine init (weights, graphs, segments), optional MTP head, capture decode windows for greedy and
sampled modes, then hand the lanes to the server. MiMo mirrors this shape, with DFlash in place of the MTP head.

## Reusable pieces
- `cuda/qlinear.zig`: `Weight` is a union of formats (`affine4`, `fp8g`) with one call for decode rows and one for
  prompt rows. **EXL3 goes in as a new union arm** (Z0B.3), refusing formats a family did not load.
- `cuda/grouped.zig`: the shared routed-expert plan (expert grouping of a round's (row, expert) pairs) with resolved
  kernel symbols from `experts.cu`. MiMo's top-8-of-256 routing uses it; the EXL3 expert kernels plug in behind it.
- `core/exl3_format.zig`: CPU reference for EXL3 (`Codebook` inst3 / mcg / **mul1**, tile unpack, signs, Hadamard 128).
  The oracle for every EXL3 CUDA kernel port.
- `cuda/triton.zig` + `cuda/aot.zig`: load Triton 3.7 cubins + metadata compiled ahead of time and launch them as
  Triton's launcher does; `aot.Set.load(dir)` / `find(function, args, consts)` picks the variant. This is how the
  Python family's 11 Triton kernels run unchanged.
- `cuda/graph.zig` (CUDA graphs), `nccl.zig`, `cublaslt.zig`, `stream.zig`, `memory.zig`, `segments.zig`.
- `zig/src/fabric`: verbs/RDMA, mailbox, collectives, sendrecv (already upstream; #636 adds the TP bootstrap and
  CUDA-process glue on top).
- Nemotron's `cuda_triton.zig`, `cuda_torch_ops.zig`, `cuda_dump.zig`: in-tree examples of AOT Triton use, torch-op
  replacements and capture dumps.

## Not in v1.0.5 (Z0B)
Keyed graph cache + device arena (#635), CUDA TP bootstrap (#636), EXL3 CUDA kernels, paged-KV sessions (#637).
