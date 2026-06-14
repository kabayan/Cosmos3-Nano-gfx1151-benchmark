#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/result/miopen-large-bf16-conv3d-test}"
MIOPEN_DRIVER="${MIOPEN_DRIVER:-MIOpenDriver}"
MIOPEN_LIB_DIR="${MIOPEN_LIB_DIR:-}"
MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-${OUT_DIR}/miopen_user_db}"
MIOPEN_CACHE_DIR="${MIOPEN_CACHE_DIR:-${OUT_DIR}/miopen_cache}"

mkdir -p "${OUT_DIR}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CACHE_DIR}"

if [[ -n "${MIOPEN_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${MIOPEN_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-NORMAL}"
export MIOPEN_FIND_ENFORCE="${MIOPEN_FIND_ENFORCE:-SEARCH_DB_UPDATE}"
export MIOPEN_USER_DB_PATH
export MIOPEN_CACHE_DIR
export MIOPEN_ENABLE_LOGGING="${MIOPEN_ENABLE_LOGGING:-1}"
export MIOPEN_ENABLE_LOGGING_CMD="${MIOPEN_ENABLE_LOGGING_CMD:-1}"
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-5}"

run_case() {
  local name="$1"
  shift
  local log="${OUT_DIR}/${name}.log"
  echo "# ${name}" | tee "${log}"
  echo "${MIOPEN_DRIVER} $*" | tee -a "${log}"
  "${MIOPEN_DRIVER}" "$@" 2>&1 | tee -a "${log}"
}

run_case descriptor_160_full_bf16 \
  convbfp16 \
  -n 1 -c 160 --in_d 18 -H 274 -W 370 \
  -k 160 --fil_d 3 -y 3 -x 3 \
  --pad_d 0 -p 0 -q 0 \
  --conv_stride_d 1 -u 1 -v 1 \
  --dilation_d 1 -l 1 -j 1 \
  --spatial_dim 3 -m conv -g 1 -F 1 -t 1

run_case descriptor_512_full_bf16 \
  convbfp16 \
  -n 1 -c 512 --in_d 6 -H 242 -W 322 \
  -k 256 --fil_d 3 -y 3 -x 3 \
  --pad_d 0 -p 0 -q 0 \
  --conv_stride_d 1 -u 1 -v 1 \
  --dilation_d 1 -l 1 -j 1 \
  --spatial_dim 3 -m conv -g 1 -F 1 -t 1

grep -hE "Find Start|FW Chosen Algorithm|GetWorkSpaceSize|GemmFwdRest|ConvDirectNaiveConvFwd|ConvolutionForward" "${OUT_DIR}"/*.log \
  > "${OUT_DIR}/summary.log" || true

echo "Wrote ${OUT_DIR}/summary.log"
