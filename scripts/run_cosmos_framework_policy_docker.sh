#!/usr/bin/env bash
# cosmos-framework 経路（Policy）の実行コンテナ起動ラッパー。
#
# この手順は 2026-07-27 まで repo のどこにも記録されておらず、再現時に毎回
# 再発見が必要だった（DLS-005 の誤報と同型の問題）。ここを唯一の正とする。
#
# イメージは docker/cosmos3-rocm72-framework.Dockerfile で作る:
#   docker build -f docker/cosmos3-rocm72-framework.Dockerfile \
#     -t cosmos3-rocm72-framework:local docker/
#
# 使い方（run_cosmos_framework_policy_rocm.py の引数をそのまま渡す）:
#   scripts/run_cosmos_framework_policy_docker.sh \
#     --out-dir /workspace/result/<name> --warmup-runs 2 --policy-condition-cache
#
# 追加でマウントしたいホストパスは EXTRA_MOUNTS に渡す:
#   EXTRA_MOUNTS="-v /home/kabayan/workspace/cosmos3_v1_ckpt:/v1ckpt" \
#     scripts/run_cosmos_framework_policy_docker.sh --policy-checkpoint-path /v1ckpt ...
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/kabayan/workspace/cosmos3}
FRAMEWORK_DIR=${FRAMEWORK_DIR:-/tmp/cosmos-framework}
IMAGE=${IMAGE:-cosmos3-rocm72-framework:local}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}

if [ ! -d "$FRAMEWORK_DIR/cosmos_framework" ]; then
  echo "framework が無い: $FRAMEWORK_DIR (rsync -a --exclude .git temp_src/ $FRAMEWORK_DIR/)" >&2
  exit 1
fi

# カーネルキャッシュの永続化（コールドスタート短縮）。
#
# コンテナは --rm で使い捨てのため、MIOpen のカーネル探索結果とコンパイル済み
# バイナリ、および inductor / triton のコンパイル結果が毎 run 捨てられ、
# 毎回ゼロから探索・コンパイルが走っていた。ホストへ逃がして持ち越す。
#
# いずれも「同じ計算をどう実行するか」のキャッシュであり、計算内容は変えない
# （DLS-003 の計算省略系には該当しない）。ただし find-db が有る / 無いで
# MIOpen のアルゴリズム選択が変わりうるため、bit 単位の一致は保証しない。
# 出力への影響は DLS-012 の fp32 感度実験（全系 fp32 でも golden MSE 不動）が
# 示す非感受性の範囲内に収まる想定で、有効化後の初回 run で実測確認する。
#
# CACHE_DIRS=0 で無効化できる（キャッシュ有無の対照実験用）。
MIOPEN_CACHE=${MIOPEN_CACHE:-$HOME/.cache/miopen}
KERNEL_CACHE=${KERNEL_CACHE:-$HOME/.cache/cosmos3-rocm}
CACHE_ARGS=()
if [ "${CACHE_DIRS:-1}" != "0" ]; then
  mkdir -p "$MIOPEN_CACHE" "$KERNEL_CACHE/inductor" "$KERNEL_CACHE/triton"
  CACHE_ARGS=(
    -v "$MIOPEN_CACHE:/root/.cache/miopen"
    -v "$KERNEL_CACHE:/root/.cache/cosmos3-rocm"
    -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/cosmos3-rocm/inductor
    -e TRITON_CACHE_DIR=/root/.cache/cosmos3-rocm/triton
  )
fi

# framework は /workspace/tmp/cosmos-framework に置く。過去 run のログに残る
# config_file パスと一致させ、結果を横並びで比較できるようにするため。
exec docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 993 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size=16G \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_XET=1 \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e PYTHONPATH=/workspace/tmp/cosmos-framework \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$REPO_DIR:/workspace" \
  -v "$FRAMEWORK_DIR:/workspace/tmp/cosmos-framework" \
  "${CACHE_ARGS[@]}" \
  ${EXTRA_MOUNTS:-} \
  -w /workspace/tmp/cosmos-framework \
  "$IMAGE" \
  python3 /workspace/scripts/run_cosmos_framework_policy_rocm.py "$@"
