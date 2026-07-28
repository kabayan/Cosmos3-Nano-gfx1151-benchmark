#!/bin/bash
# und branch cache 2 スロット化（DLS-017 採用案 A）の検証実測
# プロトコルは result/guidance_official_20260728/run_commands.sh と同一（DLS-007 v3 再現条件、
# イメージ cosmos3-rocm72-diffusers:local + aotriton_tuned env + TunableOp 表読み込み、pip install なし）。
# 変更点はイメージ内 /opt/diffusers の transformer_cosmos3.py が 2 スロット cache 実装である点のみ
# （third_party/diffusers と docker cp + commit で同期済み。旧イメージ sha256:eab19ad6eb66...）。
# 検証対象は --und-branch-cache を使う T2I / I2V の 2 モード（T2V は cache 不使用のため対象外）。
# 判定: (a) cache stats が read 主体（期待 2 writes / 138 reads / cached_slots 2）、
#       (b) 出力 md5 が 2 スロット化前と一致（T2I 1b5c6bfd... / I2V be7b1565...）、
#       (c) 倍率回復幅（DLS-017 概算 T2I 5.25x→≈2.4x / I2V 11.33x→≈2.6x の反証機会）
set -uo pipefail

OUT=result/guidance_2slot_20260728

DOCKER_BASE=(docker run --rm
  --device=/dev/kfd --device=/dev/dri
  --group-add 44 --group-add 993
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host
  -e HF_HOME=/root/.cache/huggingface
  -e HF_HUB_DISABLE_XET=1
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
  -e PYTORCH_TUNABLEOP_ENABLED=1
  -e PYTORCH_TUNABLEOP_TUNING=0
  -e PYTORCH_TUNABLEOP_RECORD_UNTUNED=0
  -e PYTORCH_TUNABLEOP_FILENAME=/workspace/result/rocm_speed_matrix/tunableop_results%d.csv
  -v /home/kabayan/.cache/huggingface:/root/.cache/huggingface
  -v /home/kabayan/workspace/cosmos3:/workspace
  -v /tmp/cosmos-framework:/workspace/tmp/cosmos-framework
  -w /workspace
  cosmos3-rocm72-diffusers:local bash -lc)

run_one() {
  local name="$1" inner="$2"
  echo "=== [$name] start $(date -Is)"
  "${DOCKER_BASE[@]}" "$inner" > "$OUT/$name/run.log" 2>&1
  local rc=$?
  echo "=== [$name] exit=$rc $(date -Is)"
  return 0
}

mkdir -p "$OUT"/{i2v,t2i}

run_one t2i "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2i_rocm.py \
  --out-dir $OUT/t2i --height 960 --width 960 --steps 35 --guidance 4.0 \
  --stage-profile --vae-warmup --vae-warmup-shape 1,48,1,60,60 \
  --mode-warmup-runs 1 --measured-runs 1 --und-branch-cache"

run_one i2v "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py \
  --case i2v --out-dir $OUT/i2v --height 256 --width 448 \
  --frames 24 --fps 12 --steps 35 --guidance 6.0 --stage-profile \
  --vae-warmup --vae-warmup-shape 1,48,2,16,28 \
  --mode-warmup-runs 1 --measured-runs 1 \
  --inference-mode --disable-progress-bar --und-branch-cache"

echo "ALL DONE $(date -Is)"
