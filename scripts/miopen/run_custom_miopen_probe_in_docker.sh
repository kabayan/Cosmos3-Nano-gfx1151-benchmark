#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
CUSTOM_MIOPEN_INSTALL="${CUSTOM_MIOPEN_INSTALL:-${ROOT_DIR}/result/miopen-build/rocm-7.2.0/install}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/result/custom-miopen-probe}"
DESCRIPTOR="${DESCRIPTOR:-policy_160_18_274_370_to_160}"
DTYPE="${DTYPE:-bf16}"
TILE="${TILE:-1x1}"
REPEATS="${REPEATS:-1}"

if [[ ! -f "${CUSTOM_MIOPEN_INSTALL}/lib/libMIOpen.so" && ! -f "${CUSTOM_MIOPEN_INSTALL}/lib64/libMIOpen.so" ]]; then
  echo "Custom libMIOpen.so not found under ${CUSTOM_MIOPEN_INSTALL}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
OUT_DIR_REL="${OUT_DIR#"${ROOT_DIR}/"}"
CONTAINER_OUT_DIR="/workspace/${OUT_DIR_REL}"

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 993 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -e MIOPEN_FIND_MODE=NORMAL \
  -e MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE \
  -e MIOPEN_USER_DB_PATH="${CONTAINER_OUT_DIR}/miopen_user_db" \
  -e MIOPEN_EXPERIMENT_LARGE_FP16_BF16_GEMM_REST=1 \
  -e MIOPEN_ENABLE_LOGGING=1 \
  -e MIOPEN_ENABLE_LOGGING_CMD=1 \
  -e MIOPEN_LOG_LEVEL=5 \
  -e LD_LIBRARY_PATH=/workspace/result/miopen-build/rocm-7.2.0/install/lib:/workspace/result/miopen-build/rocm-7.2.0/install/lib64:/opt/rocm/lib \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "python /workspace/scripts/probe_miopen_large_conv3d_descriptors.py \
    --out-dir ${CONTAINER_OUT_DIR} \
    --descriptor ${DESCRIPTOR} \
    --dtype ${DTYPE} \
    --tile ${TILE} \
    --repeats ${REPEATS} \
    > ${CONTAINER_OUT_DIR}/probe.log 2>&1"
