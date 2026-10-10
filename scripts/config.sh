# MiMo-V2.6-Flash on two DGX Sparks via TensorFold: shared settings. Override any of them in scripts/local.sh
# (gitignored).
# shellcheck disable=SC2034

# --- nodes -------------------------------------------------------------------------------------------------------
WORKER="${WORKER:-aj@10.100.128.2}"          # rank 1, reached over the CX7 link
MASTER="${MASTER:-10.100.128.1}"             # rank 0's CX7 address (rendezvous + NCCL)
MASTER_PORT="${MASTER_PORT:-29571}"          # GLM uses 29551, Kolibri 29561

# --- checkpoints -------------------------------------------------------------------------------------------------
SRC_REPO="XiaomiMiMo/MiMo-V2.6-Flash-RL"
SRC_REV="5711b268169967567844e1e560e8a3966da959b1"
BASELINE_REPO="benthecarman/MiMo-V2.6-Flash-RL-exl3"
BASELINE_REV="5ae87b830eb950c705967aea6130601a8e1f74a5"
PACK_REPO="ajensenwaud/MiMo-V2.6-Flash-RL-exl3-3.0bpw"      # our 3.0 pack (scripts/fetch_pack.sh)
PACK_REV="${PACK_REV:-f2d8d100101b5d8ed09761a33ddd28f277407b41}"  # the uploaded 3.0 pack (all 33 files, 121.7 GB)
MODELS_ROOT="${MODELS_ROOT:-$HOME/models/mimo-v2.6}"
SRC_DIR="${SRC_DIR:-$MODELS_ROOT/MiMo-V2.6-Flash-RL}"
PACK="${PACK:-3.0}"                          # 3.0 (ours, the default: 4.0's measured quality, 14% faster), 4.0, 2.27
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/MiMo-V2.6-Flash-RL-exl3-${PACK}bpw}"   # the same path on both nodes
SERVED_NAME="${SERVED_NAME:-MiMo-V2.6-Flash}"

# --- engine ------------------------------------------------------------------------------------------------------
TF_REPO="https://github.com/ashhart/TensorFold"
TF_BRANCH="python-0.6"
TF_COMMIT="ed78d6fc204d89d90b045bf033d6551e7714f3a1"   # 0.6.6; patches/ apply on top of this
TF_DIR="${TF_DIR:-$HOME/mimo/TensorFold}"
EXL3_REPO="https://github.com/turboderp-org/exllamav3"
EXL3_COMMIT="151539c77abc7ab7425d30da7a4e8e3c5c154e7b"  # v1.6.0: MiMo-V2 upstream, layer-47 interm_div
# Base image: TensorFold 0.6.0 CUDA runtime (PyTorch 2.13 nv26.07, CUDA 13.3, Triton 3.7.1, NCCL, libibverbs),
# pinned by digest; the same base as the GLM-5.3 and Kolibri-1 recipes.
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529}"
IMAGE="${IMAGE:-mimo-tf}"                    # tagged with the patched TensorFold tree hash by prepare.sh
CONTAINER="${CONTAINER:-mimo-tf}"

# --- serving -----------------------------------------------------------------------------------------------------
PORT="${PORT:-8891}"                         # GLM :8888, Kolibri :8890
HOST="${HOST:-0.0.0.0}"
PARALLEL="${PARALLEL:-4}"
TP="${TP:-2}"                                # 1: one Spark (rank 0 only)
CONTEXT="${CONTEXT:-}"                       # empty: sized from free memory
MAX_TOKENS="${MAX_TOKENS:-32768}"            # reply cap when a client sends none (thinking needs room)
DRAFTER="${DRAFTER:-1}"                      # 1: the pack's DFlash drafter (exact; 1.5-1.8x on code and reasoning)

# --- fabric (both CX7 rails, as the GLM and Kolibri recipes) -----------------------------------------------------
NCCL_IFNAME="${NCCL_IFNAME:-enp1s0f0np0}"
NCCL_HCAS="${NCCL_HCAS:-rocep1s0f0,roceP2p1s0f0}"
NCCL_GID="${NCCL_GID:-3}"
ROCE_MAX_KB="${ROCE_MAX_KB:-512}"

CACHE_DIR="${CACHE_DIR:-$HOME/.cache/mimo-tf}"   # compiled CUDA extensions, Triton cache

PROFILE="${PROFILE:-quality}"                # fast: levers that pass the Gate-5 quality gate

[[ -f "$(dirname "${BASH_SOURCE[0]}")/local.sh" ]] && source "$(dirname "${BASH_SOURCE[0]}")/local.sh"

log() { printf '[%s] %s\n' "$(basename "$0")" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

run_args() {   # docker run arguments shared by both ranks
  echo "--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK" \
       "--ulimit memlock=-1 --ulimit stack=67108864" \
       "-v $MODEL_DIR:/model:ro -v $CACHE_DIR:/root/.cache" \
       "-e NCCL_SOCKET_IFNAME=$NCCL_IFNAME -e NCCL_IB_HCA=$NCCL_HCAS -e NCCL_IB_GID_INDEX=$NCCL_GID" \
       "-e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e TF_ROCE_MAX_KB=$ROCE_MAX_KB" \
       "$(env | grep -E '^TF_MIMO_[A-Z0-9_]*=' | sed 's/^/-e /' | tr '\n' ' ')" \
       "${DEV_SRC:+-v $DEV_SRC:/opt/tensorfold-mimo/src:ro}"
}

serve_args() {  # $1 rank
  local a="serve /model --name $SERVED_NAME --parallel $PARALLEL --max-tokens $MAX_TOKENS"
  [[ -n "$CONTEXT" ]] && a+=" --context $CONTEXT"
  [[ "$DRAFTER" == 1 ]] && a+=" --drafter /model/dflash"
  if [[ "$TP" == 2 ]]; then
    a+=" --tp 2 --rank $1 --master $MASTER --master-port $MASTER_PORT"
  fi
  [[ "$1" == 0 ]] && a+=" --host $HOST --port $PORT"
  echo "$a"
}
