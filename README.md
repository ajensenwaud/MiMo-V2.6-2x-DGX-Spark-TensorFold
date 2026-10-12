# MiMo-V2.6-Flash on two DGX Sparks

This recipe runs Xiaomi's MiMo-V2.6-Flash on two NVIDIA DGX Sparks joined by a cable, and serves it as one
OpenAI-compatible API. MiMo is a large mixture-of-experts model: 309 billion parameters in total, of which about 15
billion work on each token. It does not fit on one Spark at a good quality, so the recipe splits it across both.

There are two engines to choose from. Both are built on [TensorFold](https://github.com/ashhart/TensorFold):

- **The Python engine**: TensorFold 0.6 plus the changes in `patches/`. This came first and is the default.
- **The Zig engine**: TensorFold 1.0.5 plus the changes in `patches-zig/`. It is a compiled port of the Python engine,
  and it is faster in most of the tests below.

Both handle text only (no images or audio). Both use Xiaomi's DFlash drafter, a small helper model that guesses the
next few words so the big model can check several at once.

## How fast is it?

Tokens per second on the 3.0-bit model, measured with [spark-bench](https://github.com/ajensenwaud/spark-bench).
Higher is better.

| Test | Zig engine | Python engine |
|---|---|---|
| One conversation: chat / code / agent / report | 52.5 / 77.6 / 58.9 / 52.7 | 49.4 / 78.7 / 58.9 / 51.0 |
| Four conversations at once, combined: chat / code / agent / report | 83.2 / 116.6 / 95.0 / 90.3 | 81.8 / 85.7 / 80.1 / 85.1 |
| Reading a prompt of 4K / 16K / 64K / 128K tokens | 771 / 1,309 / 1,183 / 1,091 | 764 / 1,179 / 1,141 / 1,022 |
| Writing after a 128K-token prompt | 40.0 | 33.4 |

The Zig engine is level with Python on a single conversation. It is about a third faster on code when four
conversations run at once, and about a fifth faster after a very long prompt. Single-conversation results move by up
to 15% between runs, so a gap of a few percent is a tie. The Python figures are the middle of three runs; the Zig
figures are one run of the release build. The raw results are in `results/`.

## Does it still answer correctly?

On the tests we ran, yes.

| Test | Zig engine | Python engine |
|---|---|---|
| GSM8K (school maths word problems), 500 questions | 95.2% | 96.0% |
| Finding a number hidden in a 32K and a 120K prompt, 5 places each | 10 of 10 | not run |

The two engines were asked different samples of 500 questions, and the gap is inside the margin of error.

The model comes in three sizes. The 3.0-bit version is the default: it scores as well as the 4.0-bit one and runs
faster. These scores are from the Python engine.

| Size | Disk space | GSM8K | MMLU-Pro (harder general knowledge) |
|---|---|---|---|
| 2.27-bit (benthecarman's) | 89 GiB | 95.4% | 70.0% |
| **3.0-bit (default)** | 113 GiB | 96.0% | 73.4% |
| 4.0-bit | 149 GiB | 96.6% | 73.6% |

## What you need

- Two DGX Sparks (or other GB10 machines) connected through their ConnectX-7 ports.
- Docker with the NVIDIA runtime on both.
- SSH from the first Spark (the head) to the second (the worker) without a password prompt.
- About 130 GB of free disk on each Spark for the default model.

Put the worker's address and the cable's network addresses in `scripts/local.sh`. Every setting and its default is in
`scripts/config.sh`.

## Run it

On the head:

```bash
scripts/prepare.sh       # builds the engine's Docker image and copies it to the worker
scripts/fetch_pack.sh    # downloads the 3.0-bit model to both Sparks and checks it
./start.sh               # starts the Python engine; the API is on port 8891
./stop.sh
```

To use the Zig engine, run `prepare.sh` and `fetch_pack.sh` as above, then:

```bash
scripts/zig_prepare.sh   # downloads TensorFold 1.0.5, applies patches-zig/ and compiles it inside a container
./stop.sh                # only one engine can run at a time
./zig_start.sh           # starts the Zig engine on the same port, 8891
./zig_stop.sh
```

Either way the model is served as `MiMo-V2.6-Flash` at `http://<head>:8891/v1`, four conversations at a time.

The model files are at
[ajensenwaud/MiMo-V2.6-Flash-RL-exl3-3.0bpw](https://huggingface.co/ajensenwaud/MiMo-V2.6-Flash-RL-exl3-3.0bpw). To
make them yourself from Xiaomi's original, run `scripts/quantise.sh 3.0` on the worker (about 17 hours), then
`PACK=3.0 scripts/sync_pack.sh`.

## How the Zig engine was built

We ported the Python engine to TensorFold's Zig engine one piece at a time, following the method Jay Leaton used for
DeepSeek-V4.1. After every step the Zig engine had to produce exactly the same numbers as the Python engine, so a
mistake showed up at once. At the end it wrote the same replies, word for word.

Then we made it faster. From this point its replies are no longer identical to the Python engine's, so each change
stayed in only if the benchmarks improved and the quality tests still passed:

- **Half the memory traffic in long conversations.** The model keeps a conversation's full history in 9 of its 48
  layers. The Zig engine stores that history at 8 bits instead of 16 and reads it with a new attention kernel. Each
  step reads half as much, so writing after a long prompt is about a fifth faster, and the history takes half the
  memory. To switch it off, start with `ZIG_ENV="-e TF_MIMO_KV8=0" ./zig_start.sh`.
- **Guessing further ahead in long conversations.** Each step costs more when the history is long, so checking more
  of the drafter's guesses in one step pays off more. The engine now takes that into account.
- **Faster short and medium prompts.** Prompts under 256 tokens skip the long-prompt path and its start-up cost.
  Prompts of up to about 3,000 tokens use a kernel that reads the compressed weights directly instead of unpacking
  them first. A 512-token prompt now takes 0.7 seconds instead of 2.6.

Two other ideas did not help, so they are switched off: merging small kernels made no measurable difference, and
reading compressed weights inside the long-prompt multiplication was slower. `docs/ZIG-PLAN.md` records everything
that was tried and measured.

## What is in this repository

- `scripts/`: setup, download, quantising, benchmarks and quality tests.
- `patches/` and `patches-zig/`: the changes to TensorFold for the Python and Zig engines.
- `zig/`: the Zig engine's build container and its ready-compiled kernels for the GB10.
- `tools/`: the checks and measurements used along the way.
- `results/`: the raw files behind every number above.
- `docs/`: the plans and notes the work followed.

## Credits and licence

Thank you to:

- Xiaomi MiMo, for the model and its DFlash drafter.
- Ash Hart ([@ashxhart](https://x.com/ashxhart)), for [TensorFold](https://github.com/ashhart/TensorFold) and both of
  its engines.
- Jay Leaton ([@jayleaton](https://github.com/jayleaton)), whose DeepSeek-V4.1 Zig port showed how to do this.
  Patches 0001-0005 in `patches-zig/` are his, including the code that runs one model across two GPUs.
- Jürgen Schmied ([@jschmied](https://github.com/jschmied)), who wrote patches 0001-0006 in `patches/`, including a
  server fix this recipe relies on.
- turboderp, for [exllamav3](https://github.com/turboderp-org/exllamav3), the format the model files use.
- benthecarman, for the 2.27-bit model, the conversion settings and the repaired DFlash drafter.
- Mia's AI Lab, for the recipe layout and the base image.

[CREDITS.md](CREDITS.md) has the full list, including the test datasets.

The code is Apache-2.0, like TensorFold. The model and its files are MIT, like Xiaomi's release.
