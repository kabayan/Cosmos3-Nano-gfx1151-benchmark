# Cosmos3-Nano ROCm 網羅ベンチマーク計画

作成日: 2026-06-01

目的: `nvidia/Cosmos3-Nano` を ROCm/gfx1151 Docker 環境で複数条件に分けて検証し、各条件を 3 回実行して速度差・ばらつき・失敗条件を記録する。

## 前提

既存 smoke test で確認済み:

- Docker image: `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`
- GPU: AMD Radeon Graphics / `gfx1151`
- PyTorch: `2.9.1+rocm7.2.0`
- Diffusers: `0.39.0.dev0`
- `torch_dtype=torch.float16`, `device_map="cuda"` で model load 成功
- 480p / 1 frame / 4 steps: 330.2 sec
- 480p / 5 frames / 4 steps: 1245.8 sec
- Hugging Face download は `HF_HUB_DISABLE_XET=1` を付ける

## 測定方針

各ケースは 3 回実行する。

- load-only ケース: pipeline load + cleanup を 3 回
- generation ケース: pipeline を 1 回 load し、その後 generation を 3 回
- generation の 3 回は seed を `0, 1, 2` に変える
- 各 run で `seconds`, status, error, memory, output path を JSONL に記録
- ケース単位で average / min / max / stdev / CV% を CSV/JSON に集計

load 時間と generation 時間を混ぜると原因が読めなくなるため、generation ケースでは load は `load_seconds_once` として別記録にする。

## スイート構成

### Core suite

現実的な時間で回す標準セット。

| Case | 内容 | 3回実行の狙い |
| --- | --- | --- |
| `load_fp16` | FP16 model load | cache 済み load のばらつき |
| `t2i_256_fp16_s4_g1` | 256p class / 1 frame / FP16 / 4 steps | 低解像度基準 |
| `t2i_480_fp16_s4_g1` | 480p / 1 frame / FP16 / 4 steps | smoke test baseline の再現性 |
| `t2i_480_bf16_s4_g1` | 480p / 1 frame / BF16 / 4 steps | BF16 可否と品質差 |
| `t2i_480_fp16_s8_g1` | 480p / 1 frame / FP16 / 8 steps | step 増加の影響 |
| `t2v5_256_fp16_s4_g1` | 256p class / 5 frames / FP16 / 4 steps | 動画 decode の低解像度基準 |

概算時間:

- 画像系は 256p で短縮が期待できるが、480p は 1 run 約 5.5 分の実績
- 5 frame 動画 480p は 1 run 約 20.8 分なので core には入れない
- core 全体は 1.5-3 時間程度を見込む

### Extended suite

時間がかかるため、core 成功後に実行する。

| Case | 内容 | 3回実行の狙い |
| --- | --- | --- |
| `load_bf16` | BF16 model load | BF16 load 安定性 |
| `t2i_480_fp16_s4_g4` | 480p / guidance 4.0 | guidance 増加の影響 |
| `t2v5_480_fp16_s4_g1` | 480p / 5 frames / FP16 / 4 steps | smoke test video baseline の再現性 |

概算時間:

- `t2v5_480_fp16_s4_g1` だけで 3 run 約 62 分
- extended 全体は 1.5-2.5 時間以上を見込む

## 実行スクリプト

追加スクリプト:

```text
scripts/benchmark_cosmos3_rocm.py
```

出力先:

```text
result/benchmark/
```

主な出力:

| ファイル | 内容 |
| --- | --- |
| `runs.jsonl` | run 単位の詳細記録 |
| `runs.csv` | run 単位の CSV |
| `summary.json` | case 単位の集計 |
| `summary.csv` | case 単位の CSV |
| `*.jpg` | image case の出力 |
| `*.mp4` | video case の出力 |

## Docker 起動

既存 cache を使う。

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 993 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size=16G \
  -e HF_HOME=/hf-cache \
  -e HF_HUB_DISABLE_XET=1 \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -v "$HOME/.cache/huggingface:/hf-cache" \
  -v "$PWD:/workspace" \
  -w /workspace \
  rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  bash
```

依存関係を入れる。

```bash
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate \
  av \
  cosmos_guardrail \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  safetensors \
  transformers
```

## Core suite 実行

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_cosmos3_rocm.py \
  --suite core \
  --repeats 3 \
  --out-dir /workspace/result/benchmark/core
```

## Extended suite 実行

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_cosmos3_rocm.py \
  --suite extended \
  --repeats 3 \
  --out-dir /workspace/result/benchmark/extended
```

`extended` は `core` ケースも含めて実行する設計。extended 固有ケースだけを走らせたい場合は `--case` を使う。

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_cosmos3_rocm.py \
  --case load_bf16 \
  --case t2i_480_fp16_s4_g4 \
  --case t2v5_480_fp16_s4_g1 \
  --repeats 3 \
  --out-dir /workspace/result/benchmark/extended-only
```

## 単体ケース実行

重いケースを個別に確認する場合:

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_cosmos3_rocm.py \
  --case t2v5_480_fp16_s4_g1 \
  --repeats 3 \
  --out-dir /workspace/result/benchmark/t2v5_480
```

## 判定基準

### Pass

- 3 run すべて status が `passed`
- 出力ファイルが存在する
- `summary.csv` に mean/min/max/stdev/CV% が記録される
- GPU memory が各 case 後に大きくリークしない

### Warning

- 3 run 中 1 run が失敗
- CV% が 20% を超える
- 生成時間が smoke test 比で 1.5 倍以上に悪化
- GPU temperature が 98°C 以上で長時間張り付く

### Fail

- 3 run 中 2 run 以上が失敗
- `hipErrorNoBinaryForGPU`
- OOM
- decode / postprocess が 30 分以上戻らない
- 出力ファイルが壊れている

## 記録すべき観測

ベンチ中に別 terminal で確認する。

```bash
watch -n 5 rocm-smi
```

記録項目:

- GPU temperature
- GPU%
- memory pressure
- OS が thermal throttling していないか
- 実行ごとの速度差

## 実行順序

1. `load_fp16` を単体で 3 回
2. `t2i_256_fp16_s4_g1` を 3 回
3. `t2i_480_fp16_s4_g1` を 3 回
4. `t2i_480_bf16_s4_g1` を 3 回
5. `t2i_480_fp16_s8_g1` を 3 回
6. `t2v5_256_fp16_s4_g1` を 3 回
7. 必要なら extended-only

Core が通るまでは `t2v5_480_fp16_s4_g1` を回さない。動画 480p は時間と熱の負荷が大きい。

## Web 表示への反映

実行後、`result/benchmark/**` を `result/index.html` から参照できるようにリンクを追加する。

最低限追加するリンク:

- `result/benchmark/core/summary.csv`
- `result/benchmark/core/runs.csv`
- `result/benchmark/core/*.jpg`
- `result/benchmark/core/*.mp4`
- extended を実行した場合は `result/benchmark/extended/summary.csv`
