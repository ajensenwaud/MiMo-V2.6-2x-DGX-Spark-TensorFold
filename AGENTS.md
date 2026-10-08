The purpose of this project is to make the MiMo-V2.6-Flash model run on 2 x DGX Sparks, optimized.

The downloaded checkpoints are on gx11 under `~/models/mimo-v2.6/`:
- `MiMo-V2.6-Flash-RL/` — the source (XiaomiMiMo/MiMo-V2.6-Flash-RL @ 5711b268), for our own EXL3 quants
- `MiMo-V2.6-Flash-RL-exl3-2.27bpw/` — benthecarman's EXL3 2.27 bpw pack (single-Spark baseline, with DFlash drafter and MTP heads)

# Requirements
- We are solving for speed without sacrificing quality
- Use EXL3 as a starting point
- TensorFold as the inference engine
- Output will be a recipe, similar to https://github.com/ajensenwaud/Kolibri-1-2x-DGX-Spark-TensorFold, which I wrote

# Environment
- Run on gx10/gx11, existing models can be stopped.

# Plan and decisions
- Facts: `docs/RECON.md`. Execution plan with gates and pass conditions: `docs/PLAN.md`. Read both before doing anything.
- Decisions (2026-10-09): RL checkpoint; quantise our own 3.0 and 4.0 bpw packs; GLM-5.3 comes down for the project's duration; container builds only, no host installs; text first (vision later, audio out of scope).
