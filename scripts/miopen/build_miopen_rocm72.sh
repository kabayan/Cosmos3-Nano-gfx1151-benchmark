#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="${MIOPEN_SRC_DIR:-${ROOT_DIR}/third_party/rocm-libraries-rocm-7.2.0}"
MIOPEN_DIR="${SRC_DIR}/projects/miopen"
BUILD_DIR="${MIOPEN_BUILD_DIR:-${ROOT_DIR}/result/miopen-build/rocm-7.2.0/build}"
INSTALL_DIR="${MIOPEN_INSTALL_DIR:-${ROOT_DIR}/result/miopen-build/rocm-7.2.0/install}"
DEPS_DIR="${MIOPEN_DEPS_DIR:-${ROOT_DIR}/result/miopen-build/rocm-7.2.0/deps}"
PATCH_FILE="${MIOPEN_PATCH_FILE:-}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

if [[ ! -d "${MIOPEN_DIR}" ]]; then
  echo "MIOpen source not found: ${MIOPEN_DIR}" >&2
  echo "Clone with:" >&2
  echo "  git clone --depth 1 --filter=blob:none --sparse --branch rocm-7.2.0 https://github.com/ROCm/rocm-libraries.git ${SRC_DIR}" >&2
  echo "  cd ${SRC_DIR} && git sparse-checkout set projects/miopen cmake --cone" >&2
  exit 1
fi

if [[ -n "${PATCH_FILE}" ]]; then
  echo "Applying patch: ${PATCH_FILE}"
  if git -C "${SRC_DIR}" apply --check "${PATCH_FILE}"; then
    git -C "${SRC_DIR}" apply "${PATCH_FILE}"
  elif git -C "${SRC_DIR}" apply --reverse --check "${PATCH_FILE}"; then
    echo "Patch already applied; continuing."
  else
    echo "Patch cannot be applied cleanly: ${PATCH_FILE}" >&2
    exit 1
  fi
fi

mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}" "${DEPS_DIR}"

echo "Installing MIOpen dependencies into ${DEPS_DIR}"
cmake -P "${MIOPEN_DIR}/install_deps.cmake" --minimum --prefix "${DEPS_DIR}"

echo "Configuring MIOpen"
cmake -S "${MIOPEN_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DMIOPEN_BACKEND=HIP \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/hip;${DEPS_DIR}" \
  -DBUILD_TESTING=Off \
  -DBUILD_DEV=On \
  -DMIOPEN_USE_ROCBLAS=On \
  -DMIOPEN_USE_HIPBLASLT=On \
  -DMIOPEN_USE_MLIR=Off \
  -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=Off \
  -DMIOPEN_ENABLE_AI_KERNEL_TUNING=Off

echo "Building libMIOpen and MIOpenDriver"
cmake --build "${BUILD_DIR}" --config Release --target MIOpenDriver --parallel "${BUILD_JOBS}"
cmake --build "${BUILD_DIR}" --config Release --target install --parallel "${BUILD_JOBS}"

cat <<EOF
MIOpen build complete.
Source:  ${MIOPEN_DIR}
Build:   ${BUILD_DIR}
Install: ${INSTALL_DIR}
Driver:  ${BUILD_DIR}/bin/MIOpenDriver
EOF
