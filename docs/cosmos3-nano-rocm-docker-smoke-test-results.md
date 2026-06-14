# Cosmos3-Nano ROCm Docker smoke test 実施結果

実施日: 2026-06-01

対象: `nvidia/Cosmos3-Nano`

## 結論

AMD Ryzen AI Max+ 395 / Radeon 8060S / `gfx1151` / ROCm 7.2.0 の Docker コンテナ上で、`nvidia/Cosmos3-Nano` の Diffusers 経路 smoke test は最後まで通った。

確認できたこと:

- Docker container から `/dev/kfd` と `/dev/dri` を利用できる
- PyTorch 2.9.1 + ROCm 7.2.0 が `gfx1151` を認識する
- FP16 matmul が成功する
- `diffusers.Cosmos3OmniPipeline` を import できる
- `torch_dtype=torch.float16`, `device_map="cuda"` で model load できる
- `num_frames=1`, 480p, 4 steps の画像生成が完了する
- `num_frames=5`, 480p, 4 steps の短尺 MP4 生成が完了する

ただし、生成品質は smoke test 相当で、画像・動画ともにかなりぼやけている。品質評価ではなく、ROCm 上で実行経路が通るかの確認として扱う。

## 実行環境

Docker image:

```text
rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

主要バージョン:

```text
torch: 2.9.1+rocm7.2.0.git7e1940d4
hip: 7.2.26015-fc0010cf6a
device: AMD Radeon Graphics
diffusers: 0.39.0.dev0, commit 07de1f6fe8152d9d931bc60f3d482c7d361f33fd
transformers: 5.9.0
accelerate: 1.13.0
cosmos_guardrail: 0.3.1
```

PyTorch ROCm check:

```text
cuda available: True
device count: 1
device: AMD Radeon Graphics
bf16 supported: True
mem_get_info: (128592412672, 128849018880)
FP16 matmul: ok
```

ROCm では PyTorch API 名は `torch.cuda` のまま使われる。

## 実行結果

| Phase | 内容 | 結果 |
| --- | --- | --- |
| 0 | container device passthrough | 成功 |
| 1 | PyTorch ROCm + FP16 matmul | 成功 |
| 2 | Python dependencies install | 成功 |
| 3 | `Cosmos3OmniPipeline` import | 成功 |
| 4 | HF assets download | 成功 |
| 5 | model load | 成功 |
| 6 | `num_frames=1` image generation | 成功 |
| 7 | `num_frames=5` video generation | 成功 |

## 重要な発見

初回の model load は Hugging Face Hub の Xet 経路で停滞した。

状況:

- unauthenticated request
- `Fetching 26 files` が 14/26 付近まで進んだ後、2 つの `.incomplete` blob が更新されなくなった
- 25 分程度で手動 interrupt
- incomplete blob:
  - 約 4.0 GiB
  - 約 3.5 GiB

回避策:

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/smoke_load_cosmos3_rocm.py
```

この設定で通常 download 経路に切り替えると、残りファイル取得と model load が完了した。今後の ROCm/Docker 検証では `HF_HUB_DISABLE_XET=1` を標準で付ける。

## Model load

実行:

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/smoke_load_cosmos3_rocm.py
```

結果:

```text
loaded_seconds 41.2
before_load free_gib: 119.76 / total_gib: 120.0
after_load free_gib: 89.05 / total_gib: 120.0
after_cleanup free_gib: 119.62 / total_gib: 120.0
```

model load 後の使用メモリは約 30.7 GiB。

## Image smoke test

実行:

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/smoke_t2i_cosmos3_rocm.py
```

設定:

```text
torch_dtype: torch.float16
device_map: cuda
num_frames: 1
height: 480
width: 832
num_inference_steps: 4
guidance_scale: 1.0
seed: 0
```

結果:

```text
generate_seconds 330.2
after_load free_gib: 89.05 / total_gib: 120.0
after_generate free_gib: 86.21 / total_gib: 120.0
after_cleanup free_gib: 119.07 / total_gib: 120.0
saved /workspace/cosmos3_rocm_t2i_smoke.jpg
```

生成物:

```text
cosmos3_rocm_t2i_smoke.jpg
JPEG, 832x480, 62 KiB
```

## Video smoke test

実行:

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/smoke_t2v5_cosmos3_rocm.py
```

設定:

```text
torch_dtype: torch.float16
device_map: cuda
num_frames: 5
height: 480
width: 832
fps: 5.0
num_inference_steps: 4
guidance_scale: 1.0
seed: 1
```

結果:

```text
generate_seconds 1245.8
saved /workspace/cosmos3_rocm_t2v5_smoke.mp4
```

生成物:

```text
cosmos3_rocm_t2v5_smoke.mp4
H.264 MP4, 832x480, 5 fps, duration 1.0 sec, 5 frames, 105 KiB
```

実行中の観測:

- GPU 使用率: 100%
- GPU edge temperature: 約 97-98°C
- denoising 4 steps は数秒で完了
- 生成時間の大半は decode / 後処理側で消費されているように見える

## 出力ファイル

| ファイル | 内容 |
| --- | --- |
| `cosmos3_rocm_pip_freeze.txt` | dependency versions |
| `cosmos3_rocm_phase5_load.log` | 初回 Xet 経路で interrupt した load log |
| `cosmos3_rocm_phase5_load_retry_no_xet.log` | `HF_HUB_DISABLE_XET=1` で成功した load log |
| `cosmos3_rocm_phase6_t2i.log` | image smoke test log |
| `cosmos3_rocm_phase7_t2v5.log` | video smoke test log |
| `cosmos3_rocm_t2i_smoke.jpg` | 1 frame 生成結果 |
| `cosmos3_rocm_t2v5_smoke.mp4` | 5 frame 生成結果 |
| `cosmos3_rocm_t2v5_frame0.jpg` | video 先頭 frame 抽出 |

## 追加した実行スクリプト

| ファイル | 用途 |
| --- | --- |
| `scripts/smoke_load_cosmos3_rocm.py` | model load smoke test |
| `scripts/smoke_t2i_cosmos3_rocm.py` | 1 frame image smoke test |
| `scripts/smoke_t2v5_cosmos3_rocm.py` | 5 frame video smoke test |

## 次の改善候補

- `HF_HUB_DISABLE_XET=1` を Docker 起動時 env に含める
- Hugging Face token を設定し、download rate limit を避ける
- 生成時間の内訳を profile し、decode / VAE / audio tokenizer / CPU postprocess のどこが支配的か確認する
- `enable_safety_checker=False` 以外に不要 component を外せるか確認する
- 画質確認用に `num_inference_steps` と `guidance_scale` を増やす。ただし 5 frames で約 20.8 分なので、長尺化は慎重に行う
