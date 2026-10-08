#!/usr/bin/env bash
# One-time, idempotent, no GPU needed: TensorFold python-0.6 + patches/ checked out on the head, the serving image
# built there, and the SAME image loaded on the worker over the CX7 link (one build, so both ranks run identical
# kernels). Checkpoints are separate: scripts/sync_pack.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
ROOT=$PWD

prepare_source() {   # $1 = patches dir; prints the tree hash of the patched source
  set -euo pipefail
  local patches=$1 try
  mkdir -p "$(dirname "$TF_DIR")"
  for try in 1 2 3; do
    [[ -d "$TF_DIR/.git" ]] || git clone -q "$TF_REPO" "$TF_DIR" || { sleep 5; continue; }
    git -C "$TF_DIR" fetch -q origin "$TF_BRANCH" && break
    sleep 5
  done
  cd "$TF_DIR"
  git cat-file -e "$TF_COMMIT^{commit}" || { echo "commit $TF_COMMIT not found" >&2; exit 1; }
  git checkout -q -B mimo-recipe "$TF_COMMIT"
  if compgen -G "$patches/*.patch" >/dev/null; then
    git -c user.name=recipe -c user.email=recipe@local am -q "$patches"/*.patch
  fi
  git rev-parse --short=10 'HEAD^{tree}'            # content, not commit: `git am` stamps the time
}

log "head: TensorFold $TF_BRANCH @ ${TF_COMMIT:0:7} + $(compgen -G 'patches/*.patch' | wc -l) patches"
tree=$(prepare_source "$ROOT/patches") || die "could not prepare TensorFold in $TF_DIR"
tag="$IMAGE:$tree"
if docker image inspect "$tag" >/dev/null 2>&1; then
  log "image $tag exists"
else
  docker pull -q "$BASE_IMAGE" >/dev/null
  log "building $tag (first build compiles exllamav3: ~30-60 min)"
  docker build -t "$tag" --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg TF_COMMIT="$tree" \
    --build-arg EXL3_REPO="$EXL3_REPO" --build-arg EXL3_COMMIT="$EXL3_COMMIT" \
    -f "$ROOT/docker/Dockerfile" "$TF_DIR"
fi
docker tag "$tag" "$IMAGE:latest"
docker run --rm --entrypoint python3 "$tag" -c \
  "import exllamav3, exllamav3_ext, tensorfold; from tensorfold.cuda.exl3 import experts, linear; print('import gate ok:', exllamav3.__version__, tensorfold.__file__)"
id=$(docker image inspect --format '{{.Id}}' "$tag")
log "head image $tag = $id"

if [[ "$TP" == 2 ]]; then
  if ssh "$WORKER" "docker image inspect $tag >/dev/null 2>&1" </dev/null; then
    log "worker already has $tag"
  else
    log "copying $tag to $WORKER over the CX7 link"
    docker save "$tag" | ssh "$WORKER" docker load >/dev/null
  fi
  ssh "$WORKER" "docker tag $tag $IMAGE:latest" </dev/null
  wl=$(ssh "$WORKER" "docker image inspect --format '{{range .RootFS.Layers}}{{println .}}{{end}}' $tag" </dev/null)
  hl=$(docker image inspect --format '{{range .RootFS.Layers}}{{println .}}{{end}}' "$tag")
  [[ "$wl" == "$hl" ]] || die "worker image layers differ from the head's"
  log "worker image layers match the head's ($(echo "$hl" | grep -c .) layers)"
fi
log "ready: PACK=<2.27|3.0|4.0> scripts/sync_pack.sh, then ./start.sh"
