#!/usr/bin/env bash
# Development run on the head: the dev TensorFold source mounted over the image's, a pack ($PACK, default 2.27),
# tools/ and results/. Arguments: the python command line, e.g. /tools/mimo_check.py /model /results/ref/x.safetensors
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
DEV_SRC="${DEV_SRC:-$HOME/mimo/TensorFold-dev/src}"
mkdir -p results "$CACHE_DIR"
exec docker run --rm --name "${DEV_NAME:-mimo-dev}" $(env | grep -E "^TF_MIMO_[A-Z0-9_]*=" | sed "s/^/-e /" | tr "\n" " ") --gpus all --ipc=host --shm-size 16g --network host \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NCCL_SOCKET_IFNAME="$NCCL_IFNAME" -e NCCL_IB_HCA="$NCCL_HCAS" -e NCCL_IB_GID_INDEX="$NCCL_GID" \
  -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e TF_ROCE_MAX_KB="$ROCE_MAX_KB" \
  -v "$DEV_SRC:/opt/tensorfold-mimo/src:ro" -v "$MODEL_DIR:/model:ro" \
  -v "$PWD/tools:/tools:ro" -v "$PWD/results:/results" -v "$CACHE_DIR:/root/.cache" \
  --entrypoint env "$IMAGE:latest" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 "$@"
