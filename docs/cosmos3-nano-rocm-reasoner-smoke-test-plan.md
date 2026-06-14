# Cosmos3-Nano Reasoner ROCm smoke test 計画

作成日: 2026-06-01

目的: `nvidia/Cosmos3-Nano` の Reasoner 機能、つまり image-to-text / video-to-text を ROCm/gfx1151 環境で検証する。

## 結論

Reasoner は、これまで使ってきた Diffusers `Cosmos3OmniPipeline` とは別経路で検証する必要がある。

公式モデルカードでは Reasoner の入出力は以下。

- Reasoner Input: `Text`, `Text+Image`, `Text+Video`
- Reasoner Output: `Text`
- Video input: 4 fps 推奨
- Context window: 最大 256K tokens
- `max_tokens=4096+` 推奨

公式の実行例は vLLM server を立て、OpenAI-compatible API に image/video content を送る方式。

```bash
vllm serve nvidia/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path / \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port 8000
```

ただし上記は CUDA/NVIDIA 前提。ROCm では `vllm-cosmos3` と vLLM ROCm backend がこの model architecture を扱えるかを段階的に確認する。

## 現在わかっている制約

### Diffusers では Reasoner を直接呼べない

`Cosmos3OmniPipeline.__call__` の主な引数:

```text
prompt, negative_prompt, image, num_frames, height, width, fps,
num_inference_steps, guidance_scale, enable_sound, ...
```

これは generator pipeline であり、text output を返す image/video reasoning API ではない。

### Transformers 単体では未対応

現在の Docker 検証環境で以下を試した。

```python
from transformers import AutoConfig
AutoConfig.from_pretrained("nvidia/Cosmos3-Nano", trust_remote_code=True)
```

結果:

```text
ValueError: model type `cosmos3_omni` is not recognized by transformers
```

したがって、Reasoner は `vllm-cosmos3` / Cosmos Framework 経路を使う。

## Smoke test の段階

### Phase R0: ROCm vLLM 基本確認

目的: vLLM ROCm backend がこのマシンで小型モデルを serve できるか確認する。

候補 image:

```text
vllm/vllm-openai-rocm:latest
```

起動例:

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 993 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -e HF_HOME=/hf-cache \
  -e HF_HUB_DISABLE_XET=1 \
  -v "$HOME/.cache/huggingface:/hf-cache" \
  -v "$PWD:/workspace" \
  -w /workspace \
  vllm/vllm-openai-rocm:latest \
  bash
```

確認:

```bash
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import vllm; print(vllm.__version__)"
```

小型モデル serve:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000
```

別 terminal:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

成功条件:

- vLLM ROCm container が GPU を認識
- 小型 text model が serve できる
- OpenAI-compatible API が応答する

失敗時:

- この環境の vLLM ROCm 基本動作から未成立。Cosmos Reasoner に進まない。

## Phase R1: `vllm-cosmos3` install/import

目的: Cosmos3 Reasoner 用 plugin/package が ROCm vLLM 環境に入るか確認する。

公式 CUDA 例:

```bash
uv pip install --torch-backend=cu130 "vllm==0.21.0" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3" \
  openai
```

ROCm container では torch/vLLM を壊さないため、まず追加 package だけを試す。

```bash
python -m pip install \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3" \
  openai
```

確認:

```bash
python - <<'PY'
import vllm
print("vllm", vllm.__version__)
import vllm_cosmos3
print("vllm_cosmos3 import ok")
PY
```

成功条件:

- `vllm_cosmos3` が import できる
- vLLM/torch ROCm wheel が CUDA wheel に置換されていない

失敗時:

- setup dependency が CUDA 固定の場合、ROCm では source patch または package install 回避が必要。

## Phase R2: Cosmos3 Reasoner model load

目的: Reasoner override で model load できるか確認する。

起動:

```bash
HF_HUB_DISABLE_XET=1 \
VLLM_USE_DEEP_GEMM=0 \
vllm serve nvidia/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path / \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --host 0.0.0.0 \
  --port 8000
```

ROCm で追加検討する env:

```bash
export VLLM_USE_DEEP_GEMM=0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export HIP_VISIBLE_DEVICES=0
```

成功条件:

- server が起動する
- `/v1/models` が応答する
- model load 中に `hipErrorNoBinaryForGPU`, OOM, CUDA-only import error が出ない

失敗時分類:

- `No module named vllm_cosmos3`: Phase R1 不成立
- `model type cosmos3_omni not recognized`: override/package registration 不成立
- CUDA 固定 error: vLLM-Cosmos3 側 patch が必要
- OOM: offload/quantization/tensor parallel の検討。ただし APU 120 GiB pool では load 自体は通る可能性がある

## Phase R3: image-to-text

入力:

```text
Cosmos3-Nano-assets/assets/example_reasoning_input.png
Cosmos3-Nano-assets/assets/example_reasoning_prompt.json
```

client script:

```python
import json
from pathlib import Path

import openai

example = json.load(open("Cosmos3-Nano-assets/assets/example_reasoning_prompt.json"))
image_path = Path("Cosmos3-Nano-assets/assets/example_reasoning_input.png").resolve()
image_url = image_path.as_uri()

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": example["prompt"]},
            ],
        }
    ],
    max_tokens=512,
    seed=0,
)

print(response.choices[0].message.content)
```

成功条件:

- text response が返る
- 例の期待方向: 花を掴んで赤いボトルに入れる計画が出る

記録:

```text
result/reasoner/image_to_text_response.txt
result/reasoner/image_to_text_request.json
result/reasoner/image_to_text.log
```

## Phase R4: video-to-text

入力候補:

```text
result/omni/omni_i2v_256_5f.mp4
result/benchmark/core/t2v5_256_fp16_s4_g1_run01.mp4
Cosmos3-Nano-assets/assets/example_action_id_av_0_input.mp4
```

最初は短い 5 frames / 1 sec の MP4 を使う。

prompt:

```text
Describe the video. List the main objects, motion, and any visible physical interaction.
```

client script:

```python
from pathlib import Path

import openai

video_path = Path("result/omni/omni_i2v_256_5f.mp4").resolve()
video_url = video_path.as_uri()

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": "Describe the video. List the main objects and motion."},
            ],
        }
    ],
    max_tokens=512,
    seed=0,
)

print(response.choices[0].message.content)
```

成功条件:

- video content を参照した text response が返る
- `--media-io-kwargs '{"video": {"num_frames": -1}}'` で全 frames を処理できる

記録:

```text
result/reasoner/video_to_text_response.txt
result/reasoner/video_to_text_request.json
result/reasoner/video_to_text.log
```

## Phase R5: audio-to-text は対象外

Cosmos3-Nano の Reasoner Input は `Text`, `Text+Image`, `Text+Video`。Audio-to-text は明示されていない。

Generator input には video with audio / audio があるが、これは world generation/audio generation 用であり、ASR ではない。

audio-to-text が必要なら別 ASR モデルを併用する。

候補:

- Whisper / faster-whisper
- Distil-Whisper
- SeamlessM4T 系

## 推奨実行順

1. Phase R0: vLLM ROCm small model smoke test
2. Phase R1: `vllm-cosmos3` install/import
3. Phase R2: Cosmos3 Reasoner server 起動
4. Phase R3: image-to-text
5. Phase R4: video-to-text
6. 結果を `result/reasoner/` と Web index に反映

R0/R1/R2 のどこかで失敗した場合、R3/R4 には進まない。

## 参照元

- Hugging Face model card: https://huggingface.co/nvidia/Cosmos3-Nano
- NVIDIA Cosmos GitHub: https://github.com/nvidia/cosmos
- vLLM ROCm docs: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
