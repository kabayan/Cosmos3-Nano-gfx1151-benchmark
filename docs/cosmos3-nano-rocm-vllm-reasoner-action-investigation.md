# Cosmos3-Nano ROCm vLLM Reasoner / Action 調査結果

実施日: 2026-06-01

対象: `nvidia/Cosmos3-Nano`

## 結論

Reasoner は ROCm vLLM 経路で動かせる可能性が高い。ローカルの `vllm/vllm-openai-rocm:latest` は ROCm/gfx1151 で Qwen3-0.6B を serve 済みで、`transformers-cosmos3` と `vllm-cosmos3` を追加すると `Cosmos3ReasonerForConditionalGeneration` を vLLM に登録できることを確認した。

Action は vLLM-Omni 経路が必要。公式の Cosmos3 action examples は `vllm/vllm-omni:cosmos3` を使う前提で、これは CUDA/NVIDIA runtime の例。vLLM-Omni 自体には ROCm image と ROCm platform 実装があるが、Cosmos3 action support を含む公式 `cosmos3` image と ROCm image が同一機能セットかは未確認。現時点では ROCm/gfx1151 で「動かせる」とはまだ断定しない。

## ローカル確認

既存 Docker image:

```text
vllm/vllm-openai-rocm:latest
```

起動中の小型モデル server:

```text
container: cosmos3-rocm-vllm-small
image: vllm/vllm-openai-rocm:latest
model: Qwen/Qwen3-0.6B
port: 8011
vllm: 0.22.0+rocm722
```

`/v1/models` は正常応答した。ログ上も ROCm backend で起動している。

ただし、この image には Cosmos3 plugin は未導入。

```text
vllm_cosmos3: None
vllm_omni: None
Cosmos arch in ModelRegistry: []
vllm/model_executor/models/cosmos3.py: not present
```

Transformers も単体では未対応。

```text
transformers: 5.9.0
ValueError: model type `cosmos3_omni` is not recognized
```

## Reasoner 調査

公式の Reasoner 経路は vLLM OpenAI-compatible chat completions。

必要 package:

```text
transformers-cosmos3
vllm-cosmos3
```

使い捨て container で追加 install と registration を確認した。

```bash
python -m pip install --no-deps --no-cache-dir \
  "transformers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"
```

結果:

```text
register ok
['Cosmos3ReasonerForConditionalGeneration']
AutoConfig: transformers_cosmos3.config.Cosmos3OmniConfig
model_type: cosmos3_omni
architectures: ['Cosmos3ForConditionalGeneration']
```

`vllm-cosmos3` は vLLM plugin entrypoint を持つ。

```text
[project.entry-points."vllm.general_plugins"]
register_cosmos3 = "vllm_cosmos3:register"
```

このため、次の実機検証は plugin を入れた ROCm vLLM container で Cosmos3 Reasoner server を立てること。

候補コマンド:

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD:/workspace" \
  -w /workspace \
  --entrypoint bash \
  vllm/vllm-openai-rocm:latest
```

container 内:

```bash
python -m pip install --no-deps \
  "transformers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"

VLLM_USE_DEEP_GEMM=0 HF_HUB_DISABLE_XET=1 \
vllm serve nvidia/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path /workspace \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --host 0.0.0.0 \
  --port 8012
```

注意:

- 既存の `cosmos3-rocm-vllm-small` は GPU KV cache を大きく確保しているため、Cosmos3 Reasoner load 前に停止が必要になる可能性が高い。
- 公式 README では Reasoner server 起動に約 5 分かかる。
- 失敗した場合の主な分類は、Cosmos3 weight mapping、ROCm attention kernel、Triton JIT、または memory allocation。

## Reasoner client

server 起動後の最小 image reasoning:

```python
import json
from pathlib import Path

import openai

example = json.load(open("Cosmos3-Nano-assets/assets/example_reasoning_prompt.json"))
image_path = Path("Cosmos3-Nano-assets/assets/example_reasoning_input.png").resolve()

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8012/v1")
response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path.as_uri()}},
                {"type": "text", "text": example["prompt"]},
            ],
        }
    ],
    max_tokens=example.get("max_tokens", 4096),
    seed=0,
)
print(response.choices[0].message.content)
```

## Action 調査

Cosmos3 action は Diffusers ではなく vLLM-Omni Generator API。

公式 examples:

- Forward dynamics: `/v1/videos/sync`
- Inverse dynamics: `/v1/videos`

Forward dynamics は `extra_params` に以下を渡す。

```json
{
  "action_mode": "forward_dynamics",
  "domain_name": "agibotworld",
  "action_chunk_size": 16,
  "image_size": 480,
  "view_point": "concat_view",
  "action": ["... 16-step normalized 29-D action chunk ..."]
}
```

Inverse dynamics は source video を `input_reference` として送り、`extra_params` に以下を渡す。

```json
{
  "action_mode": "inverse_dynamics",
  "domain_name": "av",
  "action_chunk_size": 60,
  "image_size": 480,
  "view_point": "ego_view",
  "raw_action_dim": 9
}
```

ローカルの `vllm/vllm-openai-rocm:latest` は通常 vLLM server で、`/v1/videos` endpoints はない。Action には vLLM-Omni が必要。

確認した image:

```text
vllm/vllm-omni:cosmos3       manifest exists, official Cosmos3 all-modality image, CUDA例
vllm/vllm-omni-rocm:v0.20.0  manifest exists, ROCm image, MI300 verified in vLLM-Omni docs
vllm/vllm-omni-rocm:latest   no such manifest
```

vLLM-Omni docs は ROCm platform を持ち、ROCm image は `vllm/vllm-omni-rocm:v0.20.0` を示す。ただし Cosmos README は、Cosmos3 Generator support は upstreaming 中で、全 modality/action 対応は `vllm/vllm-omni:cosmos3` が公式 build と説明している。

したがって ROCm Action の次ステップは次のどちらか。

1. `vllm/vllm-omni-rocm:v0.20.0` に Cosmos3 upstream PR / package を追加して `/v1/videos` action path が存在するか確認する。
2. `vllm/vllm-omni:cosmos3` の Dockerfile / installed package versions を確認し、同等構成を ROCm image 上に再現する。

## 現時点の判定

| 項目 | 判定 | 根拠 |
| --- | --- | --- |
| ROCm vLLM basic serving | 成功 | Qwen3-0.6B が `vllm/vllm-openai-rocm:latest` で起動済み |
| Cosmos3 Reasoner plugin registration | 成功 | `transformers-cosmos3` + `vllm-cosmos3` で arch 登録成功 |
| Cosmos3 Reasoner model load | 未実行 | 既存 vLLM server が GPU memory を確保中 |
| Cosmos3 Reasoner inference | 未実行 | model load 後に実施 |
| vLLM-Omni ROCm availability | あり | `vllm/vllm-omni-rocm:v0.20.0` manifest 確認 |
| Cosmos3 Action on ROCm | 未確定 | 公式 all-modality Cosmos3 image は `vllm/vllm-omni:cosmos3`; ROCm image で action path が同等か未確認 |

## 推奨順序

1. 既存 `cosmos3-rocm-vllm-small` を停止してよいタイミングを決める。
2. Reasoner server を `vllm/vllm-openai-rocm:latest` + plugins で起動し、`example_reasoning_input.png` の image-to-text を実行する。
3. Reasoner が通った後、`vllm/vllm-omni-rocm:v0.20.0` を pull して `/v1/videos` endpoint と Cosmos3 class support を確認する。
4. Action はまず 1 chunk forward dynamics を 256p / low steps 相当に落として実行する。

