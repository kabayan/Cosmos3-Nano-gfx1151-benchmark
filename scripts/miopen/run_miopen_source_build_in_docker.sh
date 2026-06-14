#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cosmos3-miopen-rocm72-build:latest}"
PATCH_FILE="${PATCH_FILE:-/workspace/patches/miopen/allow-large-fp16-bf16-gemm-rest-workspace-experiment.patch}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 993 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -e MIOPEN_PATCH_FILE="${PATCH_FILE}" \
  -e BUILD_JOBS="${BUILD_JOBS}" \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash -lc /workspace/scripts/miopen/build_miopen_rocm72.sh
