---
license: mit
language:
- en
- zh
base_model: XiaomiMiMo/MiMo-V2.6-Flash-RL
base_model_relation: quantized
pipeline_tag: text-generation
library_name: exllamav3
tags:
- exl3
- mimo_v2
- moe
- dflash
- tensorfold
- dgx-spark
---

# MiMo-V2.6-Flash-RL, EXL3 3.0 bpw

[XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (309B MoE, 15B active)
quantised to EXL3 at 3.0 bits per weight, made to run across two NVIDIA DGX Sparks (GB10) with tensor parallelism.
The recipe that serves it, with every measurement below as raw JSON:
[ajensenwaud/MiMo-V2.6-2x-DGX-Spark-TensorFold](https://github.com/ajensenwaud/MiMo-V2.6-2x-DGX-Spark-TensorFold).

Text only: the vision tower and audio heads are not in this pack.

| | |
|---|---|
| Weights | 113 GB, 17 shards (60 GiB a Spark at TP=2) |
| Bitrate | 3.03 bpw (excluding head); head 8 bpw; MTP heads 4 bpw |
| Conversion | exllamav3 v1.6.0, `-b 3.0 -hb 8 -mb 4 -hq -cr 250 -cc 2048 -cb mul1`, source @ `5711b268` |
| Drafter | `dflash/`: Xiaomi's DFlash drafter, 4 bpw EXL3, as published in [benthecarman/MiMo-V2.6-Flash-RL-exl3](https://huggingface.co/benthecarman/MiMo-V2.6-Flash-RL-exl3) @ `5ae87b83` |

## Quality

Served on two Sparks with the recipe's engine (TensorFold `mimo_v2`), the same 500 items a task for every pack,
thinking off, greedy. The 2.27 bpw pack is benthecarman's, the 4.0 bpw pack ours with the same settings as this one.

| Pack | GSM8K | MMLU-Pro | Wikitext-2 perplexity (64 x 2048) |
|---|---|---|---|
| 2.27 bpw | 95.4% | 70.0% | 5.402 |
| **3.0 bpw (this)** | **96.0%** | **73.4%** | **5.069** |
| 4.0 bpw | 96.6% | 73.6% | 4.979 |

Paired on identical items, MMLU-Pro: 3.0 vs 2.27 +3.4 points (McNemar p = 0.04); 4.0 vs 3.0 +0.2 (p = 0.89). GSM8K
differences are within noise. Not measured: code tasks, GPQA, anything with thinking on.

## Speed (two DGX Sparks, the recipe, one stream, 512-token replies)

| Task | Plain decode | With the DFlash drafter |
|---|---|---|
| coding | 45.4 tok/s | 77.6 tok/s |
| code edit | 44.7 | 80.8 |
| reasoning | 45.3 | 72.8 |
| prose | 45.5 | 46.0 |

Drafted replies are byte-identical to plain ones. Prefill about 1,200 tok/s at 30-60K-token prompts.

## Use

With the recipe (two Sparks): `PACK=3.0 ./start.sh` after `scripts/fetch_pack.sh`. The pack is a standard exllamav3
v1.6.0 conversion of the `mimo_v2` architecture; it was evaluated only with the recipe's engine, not with exllamav3
itself (at 113 GB it does not fit one GB10 with room to run).

## Licence

MIT, as the source model. The `dflash/` drafter is Xiaomi's, taken unchanged from benthecarman's MIT-licensed pack.
