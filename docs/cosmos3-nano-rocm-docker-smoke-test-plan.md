# Cosmos3-Nano ROCm Docker smoke test 計画

作成日: 2026-06-01

目的: `nvidia/Cosmos3-Nano` を AMD Ryzen AI Max+ 395 / Radeon 8060S / `gfx1151` 上の Docker コンテナで段階的に検証し、失敗箇所を切り分ける。

## 方針

最初から 720p video generation を実行しない。以下の順で、コンテナ、PyTorch ROCm、Diffusers、Cosmos3 import、モデルロード、最小生成のどこで失敗するかを記録する。

1. Docker から `/dev/kfd` と `/dev/dri` が見えること
2. PyTorch ROCm が `gfx1151` を認識すること
3. Diffusers の `Cosmos3OmniPipeline` が import できること
4. `nvidia/Cosmos3-Nano` の checkpoint を `torch.float16` でロードできること
5. `num_frames=1`, 480p, 少ステップで JPEG を生成できること
6. 成功した場合のみ、短い video smoke test に進む

Cosmos3-Nano は公式には BF16 + NVIDIA CUDA 前提。ROCm/gfx1151 では AMD 公式検証 dtype が FP16 なので、最初は `torch_dtype=torch.float16` で試す。

## 使用する Docker image

manifest 確認済み:

```text
rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

理由:

- 現在ホストは ROCm 7.2.0
- AMD Ryzen APU 向け matrix では `gfx1151` + ROCm 7.2 + PyTorch 2.9 + Python 3.12 が production support
- Docker image に PyTorch ROCm stack が入っているため、ホスト Python 環境を汚さない

必要容量の目安:

- Docker image: 約 11 GiB 以上
- Cosmos3-Nano model cache: 約 32.5 GiB
- pip 追加依存、temporary files、生成物: 10 GiB 以上
- 合計余裕: 60 GiB 以上

## ホスト事前確認

既に確認済みのホスト状態:

| 項目 | 状態 |
| --- | --- |
| ROCm | 7.2.0 |
| HIP | 7.2.26015 |
| GPU arch | `gfx1151` |
| `/dev/kfd` | 存在 |
| `/dev/dri` | 存在 |
| user groups | `video`, `render`, `docker` に所属 |
| Docker | 利用可能 |
| HF cache | `/home/kabayan/.cache/huggingface` が存在 |

追加確認:

```bash
rocm-smi
rocminfo | grep -E 'Name:|gfx'
docker manifest inspect rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1 >/dev/null
```

## コンテナ起動

Hugging Face token がある場合:

```bash
export HF_TOKEN="<your_token>"
```

Cosmos3-Nano は gated ではないが、大容量 download と rate limit 回避のため token 設定を推奨。

起動:

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size=16G \
  -e HF_HOME=/hf-cache \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -v "$HOME/.cache/huggingface:/hf-cache" \
  -v "$PWD:/workspace" \
  -w /workspace \
  rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  bash
```

`--group-add render` が group 解決で失敗する場合は numeric group id を使う。

```bash
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video | cut -d: -f3)
```

そして `--group-add "$RENDER_GID" --group-add "$VIDEO_GID"` に置き換える。

## Phase 0: container device smoke test

コンテナ内で実行:

```bash
set -euxo pipefail
whoami
groups || true
ls -l /dev/kfd /dev/dri
rocminfo | grep -E 'Name:|gfx' | head -40
rocm-smi
```

成功条件:

- `/dev/kfd` と `/dev/dri/renderD128` が見える
- `rocminfo` に `gfx1151` が出る
- `rocm-smi` が GPU を表示する

失敗時:

- `/dev/kfd` がない: Docker 起動オプション不足
- permission denied: `video` / `render` group または root/container permission を確認
- `rocminfo` がない: image 選定ミス

## Phase 1: PyTorch ROCm smoke test

コンテナ内で実行:

```bash
python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
print("mem_get_info:", torch.cuda.mem_get_info())

x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print("matmul ok:", y.shape, y.dtype, y.mean().item())
PY
```

成功条件:

- `torch.cuda.is_available()` が `True`
- device name が AMD Radeon Graphics
- FP16 matmul が完了

失敗時:

- `hipErrorNoBinaryForGPU`: image / wheel が `gfx1151` を含まない
- OOM: 他プロセスが unified memory を圧迫している
- device count 0: Docker device passthrough または permission 問題

## Phase 2: Python dependencies install

コンテナ内で実行:

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

記録:

```bash
python3 -m pip freeze | tee /workspace/cosmos3_rocm_pip_freeze.txt
```

成功条件:

- dependency install が完了
- `pip freeze` が保存される

失敗時:

- `torch` が上書きされていないか確認する
- `pip install` 後に Phase 1 を再実行し、PyTorch ROCm が壊れていないか確認する

## Phase 3: Diffusers import smoke test

コンテナ内で実行:

```bash
python3 - <<'PY'
import torch
from diffusers import Cosmos3OmniPipeline

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
print("mem:", torch.cuda.mem_get_info())
print("Cosmos3OmniPipeline import ok")
PY
```

成功条件:

- `Cosmos3OmniPipeline import ok` が出る

失敗時:

- `Cosmos3OmniPipeline` がない: diffusers の main branch が古い/変更された
- CUDA-specific import error: ROCm 非対応の dependency が混入
- `libGL`, `libxcb` 系: system package 追加が必要

必要なら:

```bash
apt-get update
apt-get install -y libxcb1 libgl1 libglib2.0-0 ffmpeg
```

## Phase 4: model asset download smoke test

まず `assets/` だけ取得し、Hugging Face 接続と cache mount を確認する。

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download nvidia/Cosmos3-Nano assets/ --local-dir /workspace/Cosmos3-Nano-assets
ls -la /workspace/Cosmos3-Nano-assets/assets | head
```

成功条件:

- prompt JSON や sample media が取得できる
- `/hf-cache` に cache が作られる

失敗時:

- network/token/rate limit の問題
- `HF_HOME` mount の permission 問題

## Phase 5: model load smoke test

この phase で初めて約 32.5 GiB の model download が発生する。

コンテナ内で `/workspace/smoke_load_cosmos3_rocm.py` を作成して実行する。

```bash
cat > /workspace/smoke_load_cosmos3_rocm.py <<'PY'
import gc
import json
import time

import torch
from diffusers import Cosmos3OmniPipeline


def mem(label):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(json.dumps({
            "label": label,
            "free_gib": round(free / 1024**3, 2),
            "total_gib": round(total / 1024**3, 2),
        }))


print("torch", torch.__version__)
print("hip", torch.version.hip)
print("device", torch.cuda.get_device_name(0))
print("bf16", torch.cuda.is_bf16_supported())
mem("before_load")

started = time.time()
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.float16,
    device_map="cuda",
    enable_safety_checker=False,
)
print("loaded_seconds", round(time.time() - started, 1))
mem("after_load")

del pipe
gc.collect()
torch.cuda.empty_cache()
mem("after_cleanup")
PY

python3 /workspace/smoke_load_cosmos3_rocm.py 2>&1 | tee /workspace/cosmos3_rocm_phase5_load.log
```

成功条件:

- `loaded_seconds` が出る
- `after_load` の memory が記録される

失敗時の分類:

- `hipErrorNoBinaryForGPU`: `gfx1151` 非対応 kernel
- dtype error: FP16/BF16 の不整合
- OOM: model load だけで不足
- CUDA-only error: Diffusers/Cosmos3 実装に CUDA 固定箇所がある

## Phase 6: minimal image generation smoke test

Phase 5 が成功した場合のみ実行する。

```bash
cat > /workspace/smoke_t2i_cosmos3_rocm.py <<'PY'
import gc
import json
import time

import torch
from diffusers import Cosmos3OmniPipeline


def mem(label):
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({
        "label": label,
        "free_gib": round(free / 1024**3, 2),
        "total_gib": round(total / 1024**3, 2),
    }))


torch.manual_seed(0)
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("device", torch.cuda.get_device_name(0))
mem("before_load")

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.float16,
    device_map="cuda",
    enable_safety_checker=False,
)
mem("after_load")

started = time.time()
result = pipe(
    prompt="A mobile robot in a clean warehouse aisle.",
    negative_prompt="blurry, distorted, low quality",
    num_frames=1,
    height=480,
    width=832,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(0),
)
print("generate_seconds", round(time.time() - started, 1))
mem("after_generate")

out = "/workspace/cosmos3_rocm_t2i_smoke.jpg"
result.video[0].save(out, format="JPEG", quality=85)
print("saved", out)

del pipe, result
gc.collect()
torch.cuda.empty_cache()
mem("after_cleanup")
PY

python3 /workspace/smoke_t2i_cosmos3_rocm.py 2>&1 | tee /workspace/cosmos3_rocm_phase6_t2i.log
```

成功条件:

- `/workspace/cosmos3_rocm_t2i_smoke.jpg` が生成される
- `generate_seconds` と memory log が残る

失敗時:

- MIOpen convolution error の場合:

```bash
export MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=1
python3 /workspace/smoke_t2i_cosmos3_rocm.py
```

- attention/Triton error の場合:

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

を確認し、Phase 1 から再実行する。

## Phase 7: short video smoke test

Phase 6 成功後のみ実行。まず 5 frames。

```bash
cat > /workspace/smoke_t2v5_cosmos3_rocm.py <<'PY'
import time

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.float16,
    device_map="cuda",
    enable_safety_checker=False,
)

started = time.time()
result = pipe(
    prompt="A mobile robot slowly moves through a clean warehouse aisle.",
    negative_prompt="blurry, distorted, low quality",
    num_frames=5,
    height=480,
    width=832,
    fps=5.0,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(1),
)
print("generate_seconds", round(time.time() - started, 1))
export_to_video(result.video, "/workspace/cosmos3_rocm_t2v5_smoke.mp4", fps=5, macro_block_size=1)
print("saved /workspace/cosmos3_rocm_t2v5_smoke.mp4")
PY

python3 /workspace/smoke_t2v5_cosmos3_rocm.py 2>&1 | tee /workspace/cosmos3_rocm_phase7_t2v5.log
```

成功条件:

- `/workspace/cosmos3_rocm_t2v5_smoke.mp4` が生成される

次に拡張する場合:

- `num_frames=25`
- `num_inference_steps=8`
- 480p のまま
- 720p / 189 frames は最後

## 記録する成果物

| ファイル | 内容 |
| --- | --- |
| `/workspace/cosmos3_rocm_pip_freeze.txt` | Python dependency versions |
| `/workspace/cosmos3_rocm_phase5_load.log` | model load log |
| `/workspace/cosmos3_rocm_phase6_t2i.log` | minimal image generation log |
| `/workspace/cosmos3_rocm_phase7_t2v5.log` | short video generation log |
| `/workspace/cosmos3_rocm_t2i_smoke.jpg` | Phase 6 output |
| `/workspace/cosmos3_rocm_t2v5_smoke.mp4` | Phase 7 output |

## 中止条件

以下の場合はそこで停止し、次 phase に進まない。

- Phase 1 で PyTorch が GPU を認識しない
- Phase 3 で `Cosmos3OmniPipeline` が import できない
- Phase 5 で model load が OOM する
- Phase 5 で CUDA-only 実装に起因する error が出る
- Phase 6 で kernel error が出て、`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` や `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=1` でも解消しない

## 実施時の判断

この smoke test は「ROCm で Cosmos3-Nano が実用速度で動くか」を測るものではない。まず判断するのは以下。

- ロード可能か
- FP16 で最低限の forward/generation path が通るか
- 失敗する場合、それが memory、dtype、kernel、CUDA 固定、dependency のどれか

Phase 6 まで通れば、ROCm/gfx1151 での追加検証価値あり。Phase 5 以前で止まる場合は、Diffusers/Cosmos3 の ROCm 対応または source build が必要。
