# Cosmos3-Nano ROCm 作業内容と環境構築手順

作成日: 2026-06-01

対象:

```text
nvidia/Cosmos3-Nano
AMD Ryzen AI Max+ 395 / Radeon 8060S / gfx1151
ROCm 7.2
```

## 結論

この環境で `nvidia/Cosmos3-Nano` の主要な ROCm 検証は以下まで完了した。

| 項目 | 結果 |
| --- | --- |
| ROCm / PyTorch GPU 認識 | 成功 |
| Diffusers model load | 成功 |
| Text-to-image | 成功 |
| Text-to-video 5 frames | 成功 |
| Image-to-video | 成功 |
| Text-to-video + audio | 成功 |
| Core benchmark | 全ケース成功 |
| vLLM ROCm basic serving | 成功 |
| Reasoner image-to-text | 成功 |
| Reasoner video-to-text 16-frame sampling | 成功 |
| Reasoner video-to-text full frames | 成功 |
| Action forward / inverse dynamics | 未実行。vLLM-Omni ROCm 経路の追加検証が必要 |

重要な注意:

- Hugging Face Hub download は `HF_HUB_DISABLE_XET=1` を付ける。
- ROCm/gfx1151 では `torch_dtype=torch.float16` を基本にする。
- Reasoner の vLLM 起動では `--skip-mm-profiling` が重要。付けないと ViT SDPA profiling で 256 GiB allocation を試みて HIP OOM になる。
- vLLM の GPU 使用量が大きく見える主因はモデル重みではなく KV cache の事前確保。

## 成果物

主要成果物は以下。

```text
docs/
result/
result/reasoner/
result/benchmark/core/
result/omni/
result/media/
```

代表ファイル:

```text
result/index.html
result/media/cosmos3_rocm_t2i_smoke.jpg
result/media/cosmos3_rocm_t2v5_smoke.mp4
result/omni/omni_i2v_256_5f.mp4
result/omni/omni_t2vs_256_5f_muxed.mp4
result/reasoner/image_to_text_response.txt
result/reasoner/video_to_text_response.txt
result/reasoner/video_to_text_fullframes_ctx81920_response.txt
```

## Docker / ROCm 前提

Docker は以下を使った。

```text
rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
vllm/vllm-openai-rocm:latest
```

ROCm device passthrough:

```bash
--device=/dev/kfd \
--device=/dev/dri \
--group-add 44 \
--group-add 993 \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--ipc=host
```

この環境の group:

```text
video: 44
render: 993
```

## Diffusers 環境構築

ベース image:

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 993 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_XET=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD:/workspace" \
  -w /workspace \
  rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  bash
```

依存関係:

```bash
python -m pip install --upgrade pip wheel setuptools
python -m pip install \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate \
  av \
  cosmos_guardrail \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  numpy \
  pillow \
  scipy \
  safetensors \
  transformers
```

確認:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.hip)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.mem_get_info())
PY
```

## Diffusers 実行

実行スクリプト:

```text
scripts/smoke_load_cosmos3_rocm.py
scripts/smoke_t2i_cosmos3_rocm.py
scripts/smoke_t2v5_cosmos3_rocm.py
scripts/test_cosmos3_omni_rocm.py
scripts/benchmark_cosmos3_rocm.py
```

基本実行:

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/smoke_load_cosmos3_rocm.py
HF_HUB_DISABLE_XET=1 python3 scripts/smoke_t2i_cosmos3_rocm.py
HF_HUB_DISABLE_XET=1 python3 scripts/smoke_t2v5_cosmos3_rocm.py
HF_HUB_DISABLE_XET=1 python3 scripts/test_cosmos3_omni_rocm.py --case all
```

主な結果:

```text
model load: 41.2 sec, model load 後の使用メモリ 約 30.7 GiB
T2I smoke: 832x480, 4 steps, 330.2 sec
T2V smoke: 832x480, 5 frames, 4 steps, 1245.8 sec
I2V: 448x256, 5 frames, 435.872 sec
T2V + audio: 448x256, 5 frames, 92.054 sec
```

Core benchmark:

```text
result/benchmark/core/
```

初回 generation は遅く、2回目以降が大幅に速い傾向があった。cold run と warm run を分けて見る必要がある。

## vLLM Reasoner 環境構築

vLLM ROCm image:

```text
vllm/vllm-openai-rocm:latest
```

この image 単体には Cosmos3 plugin は入っていないため、起動時に以下を追加する。

```bash
python -m pip install --no-deps --no-cache-dir \
  "transformers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"
```

確認済み:

```text
transformers-cosmos3: 0.1.0
vllm-cosmos3: 0.1.0
vLLM: 0.22.0
registered architecture: Cosmos3ReasonerForConditionalGeneration
```

## Reasoner Server 起動: image-to-text / 16-frame video

小さめの context と video 16-frame sampling で使う場合:

```bash
docker rm -f cosmos3-rocm-reasoner >/dev/null 2>&1 || true

docker run -d --name cosmos3-rocm-reasoner \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 993 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -p 8012:8012 \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD:/workspace" \
  -w /workspace \
  --entrypoint bash \
  vllm/vllm-openai-rocm:latest \
  -lc 'python -m pip install --no-deps --no-cache-dir "transformers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3" "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3" && VLLM_USE_DEEP_GEMM=0 HF_HUB_DISABLE_XET=1 vllm serve nvidia/Cosmos3-Nano --hf-overrides '\''{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}'\'' --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling --allowed-local-media-path /workspace --media-io-kwargs '\''{"video": {"num_frames": 16}}'\'' --host 0.0.0.0 --port 8012 --max-model-len 8192 --max-num-seqs 1 --dtype float16 --enforce-eager --gpu-memory-utilization 0.75 --skip-mm-profiling --mm-processor-kwargs '\''{"max_pixels": 262144}'\'''
```

起動確認:

```bash
curl -s http://127.0.0.1:8012/v1/models
```

ログ上の目安:

```text
Model loading took 16.78 GiB memory
Available KV cache memory: 72.12 GiB
GPU KV cache size: 525,136 tokens
max_model_len: 8192
```

## Reasoner Server 起動: full-frame video / long context

30秒動画を全フレームで投入する場合は、context だけでなく encoder cache 側も増やす必要がある。

成功した設定:

```text
--max-model-len 81920
--gpu-memory-utilization 0.5
--media-io-kwargs '{"video": {"num_frames": -1}}'
--limit-mm-per-prompt '{"video": {"count": 1, "num_frames": 400, "width": 768, "height": 432}}'
--max-num-batched-tokens 16384
--skip-mm-profiling
```

起動コマンド:

```bash
docker rm -f cosmos3-rocm-reasoner >/dev/null 2>&1 || true

docker run -d --name cosmos3-rocm-reasoner \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 993 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -p 8012:8012 \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD:/workspace" \
  -w /workspace \
  --entrypoint bash \
  vllm/vllm-openai-rocm:latest \
  -lc 'python -m pip install --no-deps --no-cache-dir "transformers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3" "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3" && VLLM_USE_DEEP_GEMM=0 HF_HUB_DISABLE_XET=1 vllm serve nvidia/Cosmos3-Nano --hf-overrides '\''{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}'\'' --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling --allowed-local-media-path /workspace --media-io-kwargs '\''{"video": {"num_frames": -1}}'\'' --limit-mm-per-prompt '\''{"video": {"count": 1, "num_frames": 400, "width": 768, "height": 432}}'\'' --host 0.0.0.0 --port 8012 --max-model-len 81920 --max-num-batched-tokens 16384 --max-num-seqs 1 --dtype float16 --enforce-eager --gpu-memory-utilization 0.5 --skip-mm-profiling --mm-processor-kwargs '\''{"max_pixels": 262144}'\'''
```

起動後ログ:

```text
Model loading took 16.97 GiB memory
Available KV cache memory: 41.18 GiB
GPU KV cache size: 299,856 tokens
Maximum concurrency for 81,920 tokens per request: 3.66x
```

注意:

- `--max-model-len 81920` だけでは不十分。
- 全フレーム動画では `10080` video embedding tokens になり、既定 encoder cache `8192` を超える。
- `--max-num-batched-tokens 16384` を入れて encoder cache 上限を広げる必要があった。
- `--limit-mm-per-prompt` の video 設定も合わせて指定した。

## Reasoner image-to-text client

初回に `file:///home/kabayan/...` を送ると server の `--allowed-local-media-path /workspace` 外として 400 になる。container 視点の `file:///workspace/...` を使う。

```bash
python - <<'PY'
import json
import time
from pathlib import Path

import openai

prompt_data = json.loads(Path("Cosmos3-Nano-assets/assets/example_reasoning_prompt.json").read_text())
image_url = "file:///workspace/Cosmos3-Nano-assets/assets/example_reasoning_input.png"

client = openai.OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8012/v1")
model = client.models.list().data[0].id

started = time.perf_counter()
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt_data["prompt"]},
            ],
        }
    ],
    max_tokens=512,
    temperature=0.0,
    seed=0,
    extra_body={"mm_processor_kwargs": {"max_pixels": 262144}},
)

print("elapsed", round(time.perf_counter() - started, 3))
print(response.choices[0].message.content)
PY
```

結果:

```text
elapsed: 6.107 sec
Move the arm to the flower. Grasp the flower. Move the arm to the red bottle. Place the flower in the red bottle.
```

## Reasoner video-to-text client

動画取得:

```bash
mkdir -p result/reasoner
curl -L --fail \
  --output result/reasoner/car-detection.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4

ffprobe -v error \
  -show_entries format=duration,size \
  -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames \
  -of json \
  result/reasoner/car-detection.mp4
```

動画 metadata:

```text
MP4 / H.264
768x432
12.5 fps
377 frames
30.16 sec
2.7 MiB
```

client:

```bash
python - <<'PY'
import json
import time
from pathlib import Path

import openai

client = openai.OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8012/v1")
model = client.models.list().data[0].id
video_url = "file:///workspace/result/reasoner/car-detection.mp4"
prompt = "Describe the video. Focus on the visible scene, vehicles, motion, and any notable events. Answer in concise Japanese."

started = time.perf_counter()
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    max_tokens=512,
    temperature=0.0,
    seed=0,
    extra_body={"mm_processor_kwargs": {"max_pixels": 262144}},
)

content = response.choices[0].message.content
Path("result/reasoner/video_to_text_response_latest.txt").write_text(content or "")
print("elapsed", round(time.perf_counter() - started, 3))
print(content)
PY
```

16-frame sampling 結果:

```text
elapsed: 14.171 sec
駐車場の上空から撮影された映像で、白い車が画面中央に現れ、右方向へと移動していきます。その後、赤い車と青い車が画面下部から上部へと移動し、白い車が画面左上に現れます。
```

full-frame / `max_model_len=81920` 結果:

```text
elapsed: 26.469 sec
駐車場の上空から撮影された映像で、白い車が画面下部から上部へと移動し、その後に青い車と赤い車が画面下部から上部へと移動しています。
```

保存済み:

```text
result/reasoner/video_to_text_response.txt
result/reasoner/video_to_text_response.json
result/reasoner/video_to_text_fullframes_ctx81920_response.txt
result/reasoner/video_to_text_fullframes_ctx81920_response.json
```

## GPU メモリの見方

HF の `16B params` は 16GB ではなく 160億パラメータの意味。

```text
16B params * BF16 2 bytes ~= 32 GB ~= 29.8 GiB
```

HF index の実測:

```text
total_size: 31,500,114,912 bytes
~= 29.34 GiB
```

vLLM Reasoner では全 checkpoint ではなく understanding tower 側をロードしているため、ログ上の model memory は約 16.8 GiB。

```text
Model loading took 16.78 GiB memory
```

GPU 上で 90GB から 100GB 程度に見える主因は KV cache。

例:

```text
--gpu-memory-utilization 0.75:
Available KV cache memory: 72.12 GiB

--gpu-memory-utilization 0.5:
Available KV cache memory: 41.18 GiB
```

単発の Reasoner 検証だけなら `--gpu-memory-utilization` は 0.5 以下でも足りる。

## 失敗と対策

### Hugging Face Xet download 停滞

症状:

```text
Fetching 26 files の途中で .incomplete blob が更新されない
```

対策:

```bash
export HF_HUB_DISABLE_XET=1
```

### Transformers が `cosmos3_omni` を認識しない

症状:

```text
ValueError: model type `cosmos3_omni` is not recognized
```

対策:

```bash
python -m pip install --no-deps \
  "transformers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"
```

### Reasoner 起動時に 256 GiB HIP OOM

症状:

```text
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 256.00 GiB
```

原因:

```text
vLLM V1 multimodal profiling の dummy ViT SDPA
```

対策:

```text
--skip-mm-profiling
```

### 全フレーム動画で context 超過

症状:

```text
Input length (10372) exceeds model's maximum context length (8192)
```

対策:

```text
--max-model-len 81920
```

または video sampling:

```text
--media-io-kwargs '{"video": {"num_frames": 16}}'
```

### 全フレーム動画で encoder cache 超過

症状:

```text
video item with 10080 embedding tokens exceeds the pre-allocated encoder cache size 8192
```

対策:

```text
--max-num-batched-tokens 16384
--limit-mm-per-prompt '{"video": {"count": 1, "num_frames": 400, "width": 768, "height": 432}}'
```

## Action 機能の現状

Diffusers `Cosmos3OmniPipeline` には Reasoner text output や action 引数はない。

Action forward / inverse dynamics は vLLM-Omni の `/v1/videos` / `/v1/videos/sync` 経路が必要。

確認済み:

```text
vllm/vllm-omni:cosmos3       manifest exists, CUDA / NVIDIA 前提の公式 Cosmos3 all-modality image
vllm/vllm-omni-rocm:v0.20.0  manifest exists, ROCm image
vllm/vllm-omni-rocm:latest   no such manifest
```

未確認:

```text
vllm/vllm-omni-rocm:v0.20.0 に Cosmos3 action path が同等に入っているか
vllm/vllm-omni:cosmos3 相当の構成を ROCm image 上で再現できるか
```

次に Action を試すなら:

1. `vllm/vllm-omni-rocm:v0.20.0` を pull する。
2. `/v1/videos` endpoint の有無を確認する。
3. Cosmos3 package / branch を追加して `action_mode=forward_dynamics` の 1 chunk を最小設定で試す。

## 参照ドキュメント

詳細ログと個別結果:

```text
docs/cosmos3-nano-rocm-docker-smoke-test-results.md
docs/cosmos3-nano-rocm-omni-feature-test-results.md
docs/cosmos3-nano-rocm-core-benchmark-results.md
docs/cosmos3-nano-rocm-vllm-reasoner-action-investigation.md
docs/cosmos3-nano-rocm-reasoner-image-to-text-results.md
docs/cosmos3-nano-rocm-reasoner-video-to-text-results.md
```

