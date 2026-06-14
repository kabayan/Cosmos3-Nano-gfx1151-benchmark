# Cosmos3-Nano ROCm Omni Feature Test Results

実施日: 2026-06-01

対象: `nvidia/Cosmos3-Nano`

## 結論

画像生成以外の omni 機能として、ROCm/gfx1151 + Diffusers 経路で以下を追加確認した。

| Feature | Path | Result |
| --- | --- | --- |
| Image-to-video | Diffusers `Cosmos3OmniPipeline(image=...)` | 成功 |
| Text-to-video + audio | Diffusers `enable_sound=True` | 成功 |
| Reasoning / text output | Transformers AutoConfig probe | 未実行。現行 `transformers==5.9.0` は `cosmos3_omni` を未認識 |
| Action forward/inverse dynamics | vLLM-Omni API examples | 未実行。Diffusers `__call__` には action 引数なし |

## 実行環境

- Docker image: `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`
- PyTorch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015`
- Device: AMD Radeon Graphics / `gfx1151`
- Diffusers: `0.39.0.dev0`
- Env: `HF_HUB_DISABLE_XET=1`

## Image-to-video

入力:

- Image: `Cosmos3-Nano-assets/assets/example_i2v_input.jpg`
- Prompt: `Cosmos3-Nano-assets/assets/example_i2v_prompt.json`
- Negative prompt: `Cosmos3-Nano-assets/assets/negative_prompt.json`

設定:

```text
torch_dtype: float16
num_frames: 5
height: 256
width: 448
fps: 5.0
num_inference_steps: 4
guidance_scale: 1.0
seed: 11
```

結果:

```text
seconds: 435.872
output: result/omni/omni_i2v_256_5f.mp4
format: H.264 MP4, 448x256, 5 fps, 5 frames, duration 1.0 sec
```

## Text-to-video + audio

入力:

- Prompt: `Cosmos3-Nano-assets/assets/example_t2vs_prompt.json`
- Negative prompt: `Cosmos3-Nano-assets/assets/negative_prompt.json`

設定:

```text
torch_dtype: float16
enable_sound: true
num_frames: 5
height: 256
width: 448
fps: 5.0
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
sound_present: true
```

生成 audio:

```text
codec: pcm_f32le
sample_rate: 48000
channels: 2
duration: 1.0 sec
```

Muxed MP4:

```text
video: H.264, 448x256, 5 fps, 1.0 sec
audio: AAC, 48000 Hz, stereo, 1.0 sec
```

実行中に MIOpen workspace warning が出たが fatal ではなく、音声生成と保存は完了した。

## Reasoning / Action の扱い

Diffusers の `Cosmos3OmniPipeline.__call__` は以下の直接引数を持つ。

```text
prompt, negative_prompt, image, num_frames, height, width, fps,
num_inference_steps, guidance_scale, enable_sound, ...
```

`action` や text reasoning 用の generation 引数はこの Diffusers call にはない。

Transformers 側 probe:

```text
AutoConfig.from_pretrained("nvidia/Cosmos3-Nano", trust_remote_code=True)
```

結果:

```text
ValueError: model type `cosmos3_omni` is not recognized by transformers
```

そのため、この環境で Reasoning / Action を続けて検証するには以下が必要。

- Reasoning: `vllm-cosmos3` / Cosmos Framework の Reasoner 経路を ROCm で検証
- Action: vLLM-Omni API の `/v1/videos/sync` または `/v1/videos` endpoint を ROCm で立てられるか検証

## 出力

| File | 内容 |
| --- | --- |
| `result/omni/i2v.log` | image-to-video log |
| `result/omni/t2vs.log` | text-to-video + audio log |
| `result/omni/omni_i2v_256_5f.mp4` | image-to-video output |
| `result/omni/omni_t2vs_256_5f.mp4` | video-only output from sound test |
| `result/omni/omni_t2vs_256_5f.wav` | generated audio |
| `result/omni/omni_t2vs_256_5f_muxed.mp4` | video + audio muxed output |

## Script

```text
scripts/test_cosmos3_omni_rocm.py
```
