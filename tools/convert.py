#!/usr/bin/env python3
"""exllamav3's convert.py, with sys.argv cleared once its own arguments are parsed.

NVIDIA's CuTe DSL (nvidia-cutlass-dsl 4.5.2 in the base image) parses sys.argv with a default ArgumentParser when it
first compiles a kernel; it reads convert.py's ``-hb 8`` as ``-h`` and exits 0 mid-conversion, so a dead job looks
finished. Run with TORCH_ALLOW_TF32_CUBLAS_OVERRIDE unset (the base image sets it; exllamav3 warns it breaks models).
"""

import os
import sys

from exllamav3.conversion.convert_model import main, parser, prepare

if __name__ == "__main__":
    if "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ:
        sys.exit("unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE before converting (env -u ...)")
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    in_args, job_state, ok, err = prepare(args)
    if not ok:
        sys.exit(f" !! Error: {err}")
    main(in_args, job_state)
