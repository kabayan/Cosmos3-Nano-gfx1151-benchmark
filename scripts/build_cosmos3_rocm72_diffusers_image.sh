#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFUSERS_DIR="${ROOT_DIR}/third_party/diffusers"
IMAGE_TAG="${1:-cosmos3-rocm72-diffusers:local}"

if [[ ! -d "${DIFFUSERS_DIR}/.git" ]]; then
  echo "missing ${DIFFUSERS_DIR}; clone https://github.com/huggingface/diffusers.git first" >&2
  exit 1
fi

docker build \
  -f "${ROOT_DIR}/docker/cosmos3-rocm72-diffusers.Dockerfile" \
  -t "${IMAGE_TAG}" \
  "${DIFFUSERS_DIR}"

docker run --rm "${IMAGE_TAG}" python - <<'PY'
import diffusers
import torch
print("torch", torch.__version__, "hip", torch.version.hip)
print("diffusers", diffusers.__version__, diffusers.__file__)
PY
