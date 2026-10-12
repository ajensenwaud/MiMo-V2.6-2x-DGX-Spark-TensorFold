#!/usr/bin/env bash
# Build the Zig engine: TensorFold v1.0.5 plus patches-zig/, compiled inside a build container (nothing is installed on
# the hosts), with the recipe's kernel pack (zig/aot/sm121) next to the binaries. Then copies the build to the worker.
# Run on the head after scripts/prepare.sh (the Zig server runs inside the same mimo-tf image).
set -euo pipefail
cd "$(dirname "$0")/.."
RECIPE=$(pwd)
source scripts/config.sh

docker image inspect "$IMAGE:latest" >/dev/null 2>&1 || die "no $IMAGE image: run scripts/prepare.sh first"

# 1. the source: TensorFold at the pinned release, the patches on a fresh branch
if [[ ! -d "$ZIG_DIR/.git" ]]; then
  log "cloning TensorFold into $ZIG_DIR"
  git clone -q "$TF_REPO" "$ZIG_DIR"
fi
cd "$ZIG_DIR"
git fetch -q --tags origin
[[ "$(git rev-parse "$ZIG_TF_TAG^{commit}")" == "$ZIG_TF_COMMIT" ]] || die "$ZIG_TF_TAG is not $ZIG_TF_COMMIT"
git checkout -q -B mimo-zig "$ZIG_TF_COMMIT"
log "applying $(ls "$RECIPE"/patches-zig/*.patch | wc -l) patches"
GIT_COMMITTER_NAME=recipe GIT_COMMITTER_EMAIL=recipe@localhost git am -q --3way "$RECIPE"/patches-zig/*.patch \
  || { git am --abort; die "patches-zig/ did not apply"; }

# 2. the build image: the recipe's CUDA base plus Zig (zig/docker/Dockerfile.build)
if ! docker image inspect "$ZIG_BUILD_IMAGE" >/dev/null 2>&1; then
  log "building $ZIG_BUILD_IMAGE"
  docker build -q -t "$ZIG_BUILD_IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" -f "$RECIPE/zig/docker/Dockerfile.build" \
    "$RECIPE/zig/docker" >/dev/null
fi

# 3. compile (a few minutes the first time): the CUDA kernels for GB10 (sm_121), the tools and the server
log "compiling (first build takes a few minutes)"
rm -rf "$ZIG_DIR/zig-out"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e ZIG_GLOBAL_CACHE_DIR=/work/.zig-global \
  -v "$ZIG_DIR:/work" -w /work "$ZIG_BUILD_IMAGE" bash -c '
    set -e
    NVCC=$(which nvcc)
    zig build -Dnvcc=$NVCC -Dtp-nvcc=$NVCC -Dsm=121 --prefix /work/zig-out
    zig build native -Dnvcc=$NVCC -Dtp-nvcc=$NVCC -Dsm=121 --prefix /work/zig-out'
cp -r "$RECIPE/zig/aot/sm121" "$ZIG_DIR/zig-out/aot"
[[ -x "$ZIG_DIR/zig-out/native/bin/tensorfold-native" && -x "$ZIG_DIR/zig-out/bin/tf-mimo" ]] || die "build incomplete"

# 4. the same build on the worker
if [[ "$TP" == 2 ]]; then
  log "copying the build to $WORKER:$ZIG_DIR/zig-out"
  ssh "$WORKER" "mkdir -p $ZIG_DIR" </dev/null
  rsync -a --delete "$ZIG_DIR/zig-out/" "$WORKER:$ZIG_DIR/zig-out/"
fi
log "built $(cd "$ZIG_DIR" && git rev-parse --short HEAD) in $ZIG_DIR/zig-out; start it with ./zig_start.sh"
