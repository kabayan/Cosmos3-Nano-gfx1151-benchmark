# Classmethod 記事ユースケースの ROCm 実施結果

実施日: 2026-06-01

参照記事:

```text
https://dev.classmethod.jp/articles/dgx-spark-cosmos3-omni-world-model-policy/
```

参照したローカルドキュメント:

```text
docs/cosmos3-nano-rocm-work-summary-and-setup.md
docs/cosmos3-nano-rocm-docker-smoke-test-results.md
docs/cosmos3-nano-rocm-omni-feature-test-results.md
docs/cosmos3-nano-rocm-reasoner-image-to-text-results.md
docs/cosmos3-nano-rocm-reasoner-video-to-text-results.md
docs/cosmos3-nano-rocm-vllm-reasoner-action-investigation.md
```

> 追記: 速度比較用に、記事条件へ寄せた T2V / I2V を別途再実行した。詳細は `docs/cosmos3-nano-rocm-classmethod-article-speed-benchmark.md` を参照。

## 目的

Classmethod の記事で紹介されている Cosmos 3 Omni の 4 ユースケースを、こちらの AMD ROCm/gfx1151 環境で再現可能な範囲で実施する。

記事側のユースケース:

| No | ユースケース | 記事での位置づけ |
| ---: | --- | --- |
| 1 | Text-to-image | テキストからロボティクスシーン画像を生成 |
| 2 | Text-to-video | テキストから把持動作動画を生成 |
| 3 | Image-to-video | 既存画像から物理状態を保った動画を生成 |
| 4 | Policy Model | 観測動画 + タスク指示から予測動画 + action sequence を生成 |

## 実施環境

こちらの環境:

```text
AMD Ryzen AI Max+ 395 / Radeon 8060S
GPU arch: gfx1151
ROCm: 7.2
Docker
```

使用 image:

```text
rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
vllm/vllm-openai-rocm:latest
```

共通設定:

```text
HF_HUB_DISABLE_XET=1
torch_dtype=float16
device_map=cuda
```

注意:

- 記事は DGX Spark / CUDA 13.0 / BF16 前提。
- 本検証は AMD ROCm/gfx1151 での非公式移植検証。
- 生成品質比較ではなく、ユースケースの経路が通るかを主目的にした。

## 実施結果サマリ

| ユースケース | ROCm 実施結果 | 出力 |
| --- | --- | --- |
| Text-to-image | 成功 | `result/media/cosmos3_rocm_t2i_smoke.jpg` |
| Text-to-video | 成功 | `result/media/cosmos3_rocm_t2v5_smoke.mp4` |
| Image-to-video | 成功 | `result/omni/omni_i2v_256_5f.mp4` |
| Text-to-video + audio | 追加確認として成功 | `result/omni/omni_t2vs_256_5f_muxed.mp4` |
| Policy Model / Action | 未完了 | vLLM-Omni Action endpoint が未成立 |
| Reasoner image-to-text | 代替・補助確認として成功 | `result/reasoner/image_to_text_response.txt` |
| Reasoner video-to-text | 代替・補助確認として成功 | `result/reasoner/video_to_text_fullframes_ctx81920_response.txt` |

## 1. Text-to-image

記事の対応箇所:

```text
テキストから商用品質のロボティクスシーンを生成する
```

こちらでは既存の Diffusers smoke test 結果を使用した。

実行スクリプト:

```text
scripts/smoke_t2i_cosmos3_rocm.py
```

実行設定:

```text
model: nvidia/Cosmos3-Nano
pipeline: diffusers.Cosmos3OmniPipeline
dtype: float16
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
generate_seconds: 330.2
output: result/media/cosmos3_rocm_t2i_smoke.jpg
format: JPEG, 832x480, 62 KiB
```

確認:

```text
result/media/cosmos3_rocm_t2i_smoke.jpg:
JPEG image data, 832x480
```

評価:

- ROCm 上で text-to-image 経路は成功。
- 記事の 960x960 / 35 steps とは条件が異なる。
- こちらは smoke test 目的の 480p / 4 steps のため、品質評価用ではない。

## 2. Text-to-video

記事の対応箇所:

```text
テキストから把持動作の動画を生成する
```

こちらでは既存の Diffusers text-to-video smoke test 結果を使用した。

実行スクリプト:

```text
scripts/smoke_t2v5_cosmos3_rocm.py
```

実行設定:

```text
model: nvidia/Cosmos3-Nano
pipeline: diffusers.Cosmos3OmniPipeline
dtype: float16
device_map: cuda
num_frames: 5
height: 480
width: 832
fps: 5
num_inference_steps: 4
guidance_scale: 1.0
seed: 1
```

結果:

```text
generate_seconds: 1245.8
output: result/media/cosmos3_rocm_t2v5_smoke.mp4
format: H.264 MP4, 832x480, 5 fps, 5 frames, 1.0 sec, 105 KiB
```

確認:

```text
width: 832
height: 480
fps: 5
frames: 5
duration: 1.0 sec
```

評価:

- ROCm 上で text-to-video 経路は成功。
- 記事の 256p / 24 frames / 12 fps / 22 sec とは条件が異なる。
- この環境では 5 frames / 4 steps でも約 20.8 分かかっており、DGX Spark の CUDA 実行とは性能差が大きい。
- 生成時間の多くは denoising ではなく decode / postprocess 側に見える。

## 3. Image-to-video

記事の対応箇所:

```text
既存画像から物理的に保守的な動画を生成する
```

こちらでは公式サンプル資産 `example_i2v_input.jpg` と `example_i2v_prompt.json` を使用した。

実行スクリプト:

```text
scripts/test_cosmos3_omni_rocm.py --case i2v
```

入力:

```text
Cosmos3-Nano-assets/assets/example_i2v_input.jpg
Cosmos3-Nano-assets/assets/example_i2v_prompt.json
Cosmos3-Nano-assets/assets/negative_prompt.json
```

実行設定:

```text
dtype: float16
num_frames: 5
height: 256
width: 448
fps: 5
num_inference_steps: 4
guidance_scale: 1.0
seed: 11
```

結果:

```text
seconds: 435.872
output: result/omni/omni_i2v_256_5f.mp4
format: H.264 MP4, 448x256, 5 fps, 5 frames, 1.0 sec, 45 KiB
```

評価:

- ROCm 上で image-to-video 経路は成功。
- 条件画像を受け取る Diffusers path が通ることを確認できた。
- 記事のような物理状態保持の品質評価までは行っていない。

## 4. Text-to-video + audio

記事の 4 ユースケースには含まれないが、Cosmos3 Omni の追加機能として確認した。

実行スクリプト:

```text
scripts/test_cosmos3_omni_rocm.py --case t2vs
```

入力:

```text
Cosmos3-Nano-assets/assets/example_t2vs_prompt.json
Cosmos3-Nano-assets/assets/negative_prompt.json
```

実行設定:

```text
dtype: float16
enable_sound: true
num_frames: 5
height: 256
width: 448
fps: 5
num_inference_steps: 2
guidance_scale: 1.0
seed: 12
```

結果:

```text
seconds: 92.054
video: result/omni/omni_t2vs_256_5f.mp4
wav: result/omni/omni_t2vs_256_5f.wav
muxed mp4: result/omni/omni_t2vs_256_5f_muxed.mp4
```

Muxed output:

```text
video: H.264, 448x256, 5 fps, 5 frames, 1.0 sec
audio: AAC, 48000 Hz, stereo, 1.0 sec
```

評価:

- ROCm 上で sound generation と mux まで成功。
- 実行中に MIOpen workspace warning は出たが fatal ではなかった。

## 5. Policy Model / Action

記事の対応箇所:

```text
Policy Model で動画と制御指令を同時生成する
```

記事では、観測動画と自然言語指示から以下を同時生成している。

```text
予測動画: 640x480, 17 frames
action: 16 steps x 10 dims
```

こちらのローカル資産には、Action 系の公式サンプルが含まれている。

```text
Cosmos3-Nano-assets/assets/example_action_fd_agibotworld_first_frame.png
Cosmos3-Nano-assets/assets/example_action_fd_agibotworld_action_chunks.json
Cosmos3-Nano-assets/assets/example_action_fd_agibotworld_4chunk_output.mp4
Cosmos3-Nano-assets/assets/example_action_id_av_0_input.mp4
Cosmos3-Nano-assets/assets/example_action_id_av_0_output.json
Cosmos3-Nano-assets/assets/example_action_id_av_1_input.mp4
Cosmos3-Nano-assets/assets/example_action_id_av_1_output.json
```

Forward dynamics sample metadata:

```text
prompt: Pickup items in the supermarket
domain_name: agibotworld
view_point: concat_view
fps: 10
image_size: 480
action_chunk_size: 16
num_chunks: 4
action_shape_per_chunk: [16, 29]
```

Sample output video:

```text
file: Cosmos3-Nano-assets/assets/example_action_fd_agibotworld_4chunk_output.mp4
resolution: 640x720
fps: 10
frames: 64
duration: 6.4 sec
```

Inverse dynamics sample input:

```text
file: Cosmos3-Nano-assets/assets/example_action_id_av_0_input.mp4
resolution: 832x480
fps: 10
frames: 61
duration: 6.1 sec
```

現環境での実施結果:

```text
未完了
```

理由:

- Diffusers `Cosmos3OmniPipeline` には action generation 用の `action` 引数や `/v1/videos` API がない。
- 現在起動できている `vllm/vllm-openai-rocm:latest` は通常の OpenAI-compatible vLLM server。
- `/openapi.json` で確認できる route は `/v1/chat/completions` と `/v1/models` などで、Action に必要な `/v1/videos` / `/v1/videos/sync` が存在しない。

確認結果:

```text
"/v1/chat/completions"
"/v1/chat/completions/batch"
"/v1/chat/completions/render"
"/v1/models"
```

Action を実施するには vLLM-Omni 経路が必要。

確認済み Docker image:

```text
vllm/vllm-omni:cosmos3       manifest exists, CUDA/NVIDIA 前提の公式 Cosmos3 all-modality image
vllm/vllm-omni-rocm:v0.20.0  manifest exists, ROCm image
vllm/vllm-omni-rocm:latest   no such manifest
```

現時点の判定:

```text
Policy Model / Action は、ROCm vLLM-Omni 上で Cosmos3 action path を成立させる追加作業が必要。
```

## 6. Reasoner による補助ユースケース

記事は Cosmos3 の Reasoner Tower と Generator Tower の統合を説明している。こちらでは Generator 系 3 ユースケースに加え、Reasoner Tower の image/video understanding も確認した。

### Image-to-text

入力:

```text
Cosmos3-Nano-assets/assets/example_reasoning_input.png
Cosmos3-Nano-assets/assets/example_reasoning_prompt.json
```

Prompt:

```text
The task is to put flower into the red bottle. Generate a plan consisting of subtasks for accomplish the task.
```

結果:

```text
elapsed: 6.107 sec
Move the arm to the flower. Grasp the flower. Move the arm to the red bottle. Place the flower in the red bottle.
```

出力:

```text
result/reasoner/image_to_text_response.txt
result/reasoner/image_to_text_response.json
```

### Video-to-text

使用動画:

```text
https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4
```

動画 metadata:

```text
resolution: 768x432
fps: 12.5
frames: 377
duration: 30.16 sec
```

16-frame sampling 結果:

```text
elapsed: 14.171 sec
駐車場の上空から撮影された映像で、白い車が画面中央に現れ、右方向へと移動していきます。その後、赤い車と青い車が画面下部から上部へと移動し、白い車が画面左上に現れます。
```

full-frame / long-context 結果:

```text
server max_model_len: 81920
gpu_memory_utilization: 0.5
max_num_batched_tokens: 16384
elapsed: 26.469 sec
駐車場の上空から撮影された映像で、白い車が画面下部から上部へと移動し、その後に青い車と赤い車が画面下部から上部へと移動しています。
```

出力:

```text
result/reasoner/video_to_text_response.txt
result/reasoner/video_to_text_fullframes_ctx81920_response.txt
```

## 記事との差分

| 項目 | 記事 | 本検証 |
| --- | --- | --- |
| Hardware | DGX Spark / GB10 / CUDA 13.0 | Ryzen AI Max+ 395 / gfx1151 / ROCm 7.2 |
| dtype | BF16 | FP16 |
| T2I | 960x960 / 35 steps / 約22 sec | 832x480 / 4 steps / 約330 sec |
| T2V | 256p / 24 frames / 12 fps / 約22 sec | 832x480 / 5 frames / 5 fps / 約1246 sec |
| I2V | 公式条件画像 / 約17 sec | 448x256 / 5 frames / 約436 sec |
| Policy | 動画 + 16x10 action 生成成功 | vLLM-Omni Action endpoint 未成立 |
| Reasoner | 記事では主に構造説明 | image-to-text / video-to-text 成功 |

## 実行上の重要設定

### Diffusers

```text
HF_HUB_DISABLE_XET=1
torch_dtype=float16
device_map=cuda
enable_safety_checker=False
```

### vLLM Reasoner

image-to-text / short video:

```text
--max-model-len 8192
--max-num-seqs 1
--dtype float16
--enforce-eager
--gpu-memory-utilization 0.75
--skip-mm-profiling
--mm-processor-kwargs '{"max_pixels": 262144}'
```

full-frame video:

```text
--max-model-len 81920
--gpu-memory-utilization 0.5
--media-io-kwargs '{"video": {"num_frames": -1}}'
--limit-mm-per-prompt '{"video": {"count": 1, "num_frames": 400, "width": 768, "height": 432}}'
--max-num-batched-tokens 16384
--skip-mm-profiling
```

`--skip-mm-profiling` がない場合、ViT SDPA の dummy profiling で 256 GiB allocation を試みて HIP OOM になった。

## 結論

Classmethod 記事の 4 ユースケースのうち、ROCm/gfx1151 環境では以下を実施できた。

```text
Text-to-image: 成功
Text-to-video: 成功
Image-to-video: 成功
Policy Model / Action: 未完了
```

加えて、Reasoner Tower の補助確認として以下も成功した。

```text
Image-to-text: 成功
Video-to-text: 成功
Video-to-text full frames with max_model_len=81920: 成功
```

Policy Model は、通常 vLLM server ではなく vLLM-Omni の `/v1/videos` 系 API が必要。次の作業は `vllm/vllm-omni-rocm:v0.20.0` 上で Cosmos3 action path を成立させ、`example_action_fd_agibotworld_*` または記事同等の LeRobot/Bridge 系 input で forward dynamics を試すこと。
