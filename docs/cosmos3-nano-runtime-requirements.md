# nvidia/Cosmos3-Nano 実行要件メモ

作成日: 2026-06-01

対象: https://huggingface.co/nvidia/Cosmos3-Nano

## 結論

現在の `/home/kabayan/workspace/cosmos3` 環境では、`nvidia/Cosmos3-Nano` をローカルでそのまま実行する条件を満たしていません。主な理由は、公式の実行経路が NVIDIA CUDA GPU 前提である一方、この環境では NVIDIA GPU / CUDA runtime が確認できないためです。

現実的な実行方法は、NVIDIA Ampere / Hopper / Blackwell 世代の GPU を持つ Linux 環境を別途用意し、Diffusers、vLLM-Omni、または vLLM のいずれかで起動することです。Generator 目的なら Diffusers または vLLM-Omni、Reasoner 目的なら vLLM が最短です。

## モデル概要

- モデル ID: `nvidia/Cosmos3-Nano`
- 開発元: NVIDIA
- ライセンス: OpenMDW 1.1
- パラメータ数: 16B
- Tensor type: BF16
- 公開日: Hugging Face / GitHub ともに 2026-05-31
- Hugging Face API 確認時点の revision: `138d071cee76860b0d2acd253dc6a07a11e3f3c1`
- 最終更新: `2026-06-01T04:43:55.000Z`
- ダウンロード対象ファイル合計の目安: 約 32.5 GiB

主な大容量ファイル:

| ファイル | サイズ目安 |
| --- | ---: |
| `transformer/diffusion_pytorch_model-00001..00007-of-00007.safetensors` | 合計 約 28.3 GiB |
| `sound_tokenizer/diffusion_pytorch_model.safetensors` | 約 1.76 GiB |
| `vae/diffusion_pytorch_model.safetensors` | 約 1.31 GiB |
| `vision_encoder/model.safetensors` | 約 1.07 GiB |

## 公式に示されている実行前提

公式モデルカードと NVIDIA Cosmos README で確認できる前提:

- OS: Linux
- GPU: NVIDIA Ampere / Hopper / Blackwell
- 精度: BF16 のみテスト済み
- Runtime engine:
  - PyTorch
  - Hugging Face Diffusers
  - vLLM-Omni
  - vLLM
- テストハードウェア: GB200 / H100
- vLLM-Omni の推奨例は H200 上の `nvidia/Cosmos3-Nano` serving
- CUDA は CUDA 13 または CUDA 12.8 系の組み合わせが案内されている

## 現在環境の確認結果

このマシンで確認した情報:

| 項目 | 結果 |
| --- | --- |
| OS | Ubuntu Linux, kernel `6.17.0-1011-oem`, x86_64 |
| CPU | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 32 threads |
| Memory | 124 GiB total, 49 GiB available |
| Disk | 1.2 TiB available on `/` |
| Python | 3.12.3 |
| uv | 0.10.7 |
| Docker | 29.1.2 |
| NVIDIA CLI | `nvidia-smi` が存在しない |
| Docker NVIDIA GPU | `docker run --gpus all ...` が `could not select device driver "" with capabilities: [[gpu]]` で失敗 |
| AMD/ROCm | `rocminfo` で `gfx1151` / Radeon Graphics を確認 |
| PyTorch | 未インストール |

判定:

- ディスク容量はモデル取得には十分です。
- CPU RAM は 124 GiB ありますが、公式実行経路は NVIDIA GPU 前提です。
- AMD ROCm GPU は見えていますが、Cosmos3-Nano の公式サポート対象には含まれていません。
- `uv 0.10.7` は古く、NVIDIA Cosmos README では Cosmos Framework に `uv >= 0.11.3` が必要とされています。
- Docker はありますが NVIDIA Container Toolkit / NVIDIA GPU runtime が使えない状態です。

## 現在環境で不足しているもの

ローカル実行に必要な不足条件:

1. NVIDIA GPU
   - Ampere / Hopper / Blackwell 世代。
   - 16B BF16 checkpoint なので、最低でもモデル重み 32.5 GiB に加えて実行時メモリ、VAE、vision encoder、sound tokenizer、KV/cache、activation、guardrail モデル分の余裕が必要です。
   - 公式例では H100 / GB200 / H200 クラスが前提として出ています。小さい GPU では `--enable-layerwise-offload` や tensor/context parallelism が必要になる可能性があります。

2. NVIDIA driver / CUDA
   - `nvidia-smi` で GPU と driver を確認できること。
   - CUDA 13 または CUDA 12.8 に合う PyTorch / vLLM の組み合わせを使うこと。

3. NVIDIA Container Toolkit
   - vLLM-Omni Docker 経路を使う場合、`docker run --gpus all ...` が成功すること。

4. `uv` の更新
   - 現在は `uv 0.10.7`。
   - Cosmos Framework の README では `uv >= 0.11.3` が必要。

5. Python 3.13 環境
   - 公式例は `uv venv --python 3.13 --seed --managed-python` を使う。
   - ローカル Python 3.12.3 でも、`uv` の managed Python で 3.13 を用意する想定。

## 推奨実行パス

### 1. Generator を試すだけ: Diffusers

研究・検証用途で最も単純な Python 経路です。動画生成、画像生成、image-to-video を直接呼べます。

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate
uv pip install --torch-backend=auto \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate \
  av \
  cosmos_guardrail \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  torch \
  torchvision \
  transformers
```

最小の text-to-image smoke test:

```python
import torch
from diffusers import Cosmos3OmniPipeline

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

result = pipe(
    prompt="A mobile robot in a clean warehouse aisle.",
    num_frames=1,
    height=480,
    width=832,
)
result.video[0].save("cosmos3_t2i.jpg", format="JPEG", quality=85)
```

720p / 189 frames の text-to-video はかなり重いので、最初は `num_frames=1` または低解像度・短尺で動作確認してください。

### 2. Generator を API 化: vLLM-Omni

動画、画像、音声、action 系を OpenAI-compatible API として出す経路です。公式 Docker image が案内されています。

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd):/workspace" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --port 8000
```

起動後の text-to-video リクエスト例:

```bash
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  --form-string "prompt=A small warehouse robot moves a blue box across a clean floor." \
  --form-string "negative_prompt=blurry, distorted, low quality" \
  --form-string "size=832x480" \
  --form-string "num_frames=81" \
  --form-string "fps=24" \
  --form-string "num_inference_steps=35" \
  --form-string "guidance_scale=4.0" \
  --form-string "seed=42" \
  -o cosmos3_t2v_output.mp4
```

### 3. Reasoner を API 化: vLLM

画像・動画理解、物理推論、計画生成など、テキスト出力が目的なら Reasoner path の vLLM が適します。

CUDA 13:

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate
uv pip install --torch-backend=cu130 "vllm==0.21.0" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"
```

CUDA 12.8:

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate
uv pip install --torch-backend=cu128 "vllm==0.19.1" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"
```

起動:

```bash
vllm serve nvidia/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --async-scheduling \
  --allowed-local-media-path / \
  --port 8000
```

DeepGEMM が使えないというエラーが出る場合:

```bash
export VLLM_USE_DEEP_GEMM=0
```

## 事前確認コマンド

NVIDIA 環境へ移した後、まず以下を確認します。

```bash
nvidia-smi
uv --version
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

PyTorch CUDA 確認:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Hugging Face 認証:

```bash
uvx hf@latest auth login
```

モデル assets のみ取得:

```bash
hf download nvidia/Cosmos3-Nano assets/ --local-dir Cosmos3-Nano
```

キャッシュ先を明示する場合:

```bash
export HF_HOME=/path/to/large/cache
```

## 入出力制約の要点

Generator:

- 入力: text, image, video, action trajectory
- 画像形式: jpg, png, jpeg, webp
- 動画形式: mp4
- 対応解像度: 256p / 480p / 720p
- 対応 aspect ratio: 16:9, 4:3, 1:1, 3:4, 9:16
- 入力動画: 最大 5 frames
- 出力動画: MP4、デフォルト 189 frames、5-400 frames がモデルカード上の記載
- 音声: 48 kHz stereo AAC stream

Reasoner:

- 入力: text, text + image, text + video
- context window: 最大 256K tokens
- 動画入力: 4 fps 推奨
- 出力: text
- reasoning 出力では `max_tokens=4096+` が推奨

## 注意点

- Cosmos3-Nano は物理シミュレータではありません。長尺・高解像度・複雑な物理相互作用では、時間的一貫性、物体永続性、接触、音声同期、action-state consistency に破綻が出る可能性があります。
- ロボット制御、自動運転、安全重要用途では、モデル出力を ground truth や安全認証済み判断として扱わず、外部制約、guardrail、システム検証が必要です。
- Diffusers の safety checker は `cosmos_guardrail` に依存します。NVIDIA のライセンス上、guardrail の扱いは利用条件に沿って確認してください。
- Hugging Face の Inference Providers には、このモデルは現時点でデプロイされていません。

## 参照元

- Hugging Face model card: https://huggingface.co/nvidia/Cosmos3-Nano
- Hugging Face Diffusers Cosmos 3 docs: https://huggingface.co/docs/diffusers/main/api/pipelines/cosmos3
- NVIDIA Cosmos GitHub README: https://github.com/nvidia/cosmos
- OpenMDW 1.1 license: https://openmdw.ai/license/1-1/
