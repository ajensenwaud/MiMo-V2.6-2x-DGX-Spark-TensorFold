# Credits

This recipe stands on other people's work. Thank you to:

- **Xiaomi MiMo** for [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (MIT), the model
  this recipe serves, and for its DFlash speculative-decoding drafter.
- **TensorFold** ([ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) by Ash Hart
  ([@ashxhart](https://x.com/ashxhart)). The engine, its two-rank serving, kernels and tooling are TensorFold's; this
  recipe is a set of patches on top of it (`patches/` for the Python engine 0.6, `patches-zig/` for the Zig engine
  1.0.5). The Zig engine's precompiled kernel pack in `zig/aot/` is TensorFold's Triton kernels, compiled for GB10.
- **Jay Leaton** ([@jayleaton](https://github.com/jayleaton)). The Zig engine port follows the method of his
  DeepSeek-V4.1 port to TensorFold's Zig engine: launch for launch against the Python engine, checked bit for bit.
  Patches 0001-0005 in `patches-zig/` are his commits (cherry-picked with `-x`), including tensor parallelism across
  CUDA processes and the one-shot RoCE exchange.
- **Jürgen Schmied** ([@jschmied](https://github.com/jschmied)), author of patches 0001-0006 (from TensorFold pull
  request [#328](https://github.com/ashhart/TensorFold/pull/328)), including the server fix in 0001 that this recipe
  relies on.
- **turboderp** for [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT): the EXL3 format and quantiser used to
  build the 3.0 and 4.0 bpw packs, its upstream MiMo-V2 support, and the reference forward and perplexity method this
  recipe's engine is checked against.
- **benthecarman** for [MiMo-V2.6-Flash-RL-exl3](https://huggingface.co/benthecarman/MiMo-V2.6-Flash-RL-exl3) (MIT): the
  2.27 bpw pack used as the baseline, the conversion settings our packs reuse, the repaired and quantised DFlash
  drafter shipped in every pack, and the single-Spark measurements quoted for orientation.
- **Mia's AI Lab** ([MiaAI-Lab](https://github.com/MiaAI-Lab)) for the two-Spark recipe format this repository follows
  and the GLM-5.3 TensorFold image used as the CUDA base.
- **b12x "RoCEnante"** one-shot RoCE all-gather (Apache-2.0), as vendored in TensorFold's GLM-5.3 image, reused for the
  two-Spark collectives.

Evaluation data:

- **GSM8K** ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), MIT), Cobbe et al., 2021.
- **MMLU-Pro** ([TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), MIT), Wang et al., 2024.
- **WikiText-2** ([Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), CC BY-SA 3.0), Merity et
  al., 2016, for perplexity.

Any errors in this recipe are mine, not theirs.
