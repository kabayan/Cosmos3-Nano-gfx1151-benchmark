# Cosmos3-Nano ROCm Reasoner Image-to-Text Results

実施日: 2026-06-01

対象: `nvidia/Cosmos3-Nano`

## 結論

ROCm/gfx1151 上の `vllm/vllm-openai-rocm:latest` + `transformers-cosmos3` + `vllm-cosmos3` 経路で、Cosmos3-Nano Reasoner の image-to-text が成功した。

## 起動時の重要設定

標準設定では Reasoner の `max_model_len=262144` と multimodal profiling により、ViT SDPA が 256 GiB allocation を試みて HIP OOM になった。

成功時は以下を追加した。

```text
--max-model-len 8192
--max-num-seqs 1
--dtype float16
--enforce-eager
--gpu-memory-utilization 0.75
--skip-mm-profiling
--mm-processor-kwargs '{"max_pixels": 262144}'
```

特に `--skip-mm-profiling` が必須。これにより dummy multimodal encoder profiling の 256 GiB allocation を回避できた。

## Server

```text
container: cosmos3-rocm-reasoner
image: vllm/vllm-openai-rocm:latest
port: 8012
vllm: 0.22.0
served model: nvidia/Cosmos3-Nano
max_model_len: 8192
dtype: float16
```

Model load:

```text
checkpoint size: 29.34 GiB
loaded shards: 8/8
model loading memory: 16.78 GiB
available KV cache memory: 72.12 GiB
GPU KV cache size: 525,136 tokens
```

## Request

Input image:

```text
Cosmos3-Nano-assets/assets/example_reasoning_input.png
PNG, 512x341
```

Prompt:

```text
The task is to put flower into the red bottle. Generate a plan consisting of subtasks for accomplish the task.
```

## Response

Elapsed:

```text
6.107 sec
```

Output:

```text
Move the arm to the flower. Grasp the flower. Move the arm to the red bottle. Place the flower in the red bottle.
```

## Output Files

```text
result/reasoner/image_to_text_response.txt
result/reasoner/image_to_text_response.json
result/reasoner/models.json
result/reasoner/cosmos3_reasoner_server.log
```

## Notes

- 初回の `file:///home/kabayan/...` URI は server の `--allowed-local-media-path /workspace` 外として 400 になった。
- 成功時は container 視点の `file:///workspace/Cosmos3-Nano-assets/assets/example_reasoning_input.png` を使った。
- `--skip-mm-profiling` は実リクエストの multimodal 処理を無効化するものではなく、起動時の memory profiling を飛ばす設定。

