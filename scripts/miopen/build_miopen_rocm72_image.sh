#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cosmos3-miopen-rocm72-build:latest}"

docker build \
  -f "${ROOT_DIR}/docker/miopen-rocm72-build.Dockerfile" \
  -t "${IMAGE_NAME}" \
  "${ROOT_DIR}"

echo "${IMAGE_NAME}"
