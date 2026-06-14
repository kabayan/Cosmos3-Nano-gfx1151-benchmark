# Classmethod 記事データを使った 4 モード実行結果

実施日: 2026-06-01

参照記事:

```text
https://dev.classmethod.jp/articles/dgx-spark-cosmos3-omni-world-model-policy/
```

## 目的

Classmethod 記事のユースケースで使われたデータ・条件に寄せて、こちらの ROCm/gfx1151 環境で text-to-image / text-to-video / image-to-video / Policy Model の 4 モードを実行する。

前回の `docs/cosmos3-nano-rocm-classmethod-usecase-results.md` は既存 smoke test 結果との対応付けを含んでいた。この記事件では、T2V / I2V を記事条件へ合わせて再実行し、追加で T2I と Policy Model も記事データを使って実行した。

## 環境

```text
Hardware: AMD Ryzen AI Max+ 395 / Radeon 8060S / gfx1151
ROCm: 7.2
Docker image: rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
Model: nvidia/Cosmos3-Nano
Pipeline: diffusers.Cosmos3OmniPipeline
dtype: float16
device_map: cuda
HF_HUB_DISABLE_XET: 1
```

実行スクリプト:

```text
scripts/benchmark_classmethod_article_t2v_i2v_rocm.py
scripts/benchmark_classmethod_article_t2i_rocm.py
scripts/run_cosmos_framework_policy_rocm.py
```

実行コマンド:

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 993 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_DISABLE_XET=1 \
  -v /home/kabayan/.cache/huggingface:/root/.cache/huggingface \
  -v /home/kabayan/workspace/cosmos3:/workspace \
  -w /workspace \
  rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  bash -lc 'python -m pip install --quiet --no-cache-dir "diffusers @ git+https://github.com/huggingface/diffusers.git" accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg scipy transformers safetensors pillow && HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py --case both --out-dir /workspace/result/classmethod_article_benchmark --height 256 --width 448 --frames 24 --fps 12 --steps 35 --guidance 1.0'
```

## 比較条件

記事側 T2I:

```text
960x960
35 steps
モデル常駐後の生成所要時間 22 sec
```

記事側で明示されている T2V 条件:

```text
256p
24 frames
12 fps
モデル常駐後の生成所要時間 22 sec
```

記事側 I2V:

```text
公式サンプルの条件画像
公式サンプル相当の prompt
モデル常駐後の生成所要時間 17 sec
```

I2V は記事中で frame 数や fps が明記されていないため、速度比較のため T2V と同じ条件に揃えた。

```text
height: 256
width: 448
requested frames: 24
fps: 12
steps: 35
guidance_scale: 1.0
```

記事側 Policy Model:

```text
Bridge / LeRobot v3 形式のロボット観測動画
prompt: Put the pot to the left of the purple item.
出力: 640x480 x 17 frames の予測動画
出力: 16 steps x 10 dims の action sequence
モデル常駐後の生成所要時間 21 sec
```

## 使用データ

### Text-to-image

記事と同じ robotics lab scene のサンプル JSON を使用した。

```text
/tmp/cosmos-framework/inputs/omni/t2i.json
```

### Text-to-video

記事本文のユースケースに合わせ、以下の内容を prompt 化した。

```text
ロボットグリッパーが赤いキューブを掴んで、ゆっくり持ち上げる。
```

スクリプト内 prompt:

```text
A robotic gripper descends toward a red cube, makes contact, grasps it,
and slowly lifts it upward in a physically plausible sequence.
```

### Image-to-video

記事の I2V ユースケースは公式サンプル条件画像を使っているため、ローカルの Cosmos3-Nano 公式サンプルを使用した。

```text
Cosmos3-Nano-assets/assets/example_i2v_input.jpg
Cosmos3-Nano-assets/assets/example_i2v_prompt.json
Cosmos3-Nano-assets/assets/negative_prompt.json
```

prompt 内容は、左右のロボットアームと中央の木製棚・赤い球を含む実験環境で、右側のロボットハンドが赤い球へ手を伸ばして移動させる、という公式サンプルの structured prompt。

### Policy Model

記事と同じ Bridge / LeRobot 系のロボット policy サンプルを使用した。

```text
/tmp/cosmos-framework/inputs/omni/action_policy_robot.json
vision_path: https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/action/bridge_20260501_0.mp4
action_path: https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/action/bridge_20260501_0.json
prompt: Put the pot to the left of the purple item.
domain_name: bridge_orig_lerobot
fps: 5
image_size: 480
action_chunk_size: 16
```

## 実行結果

Model load:

```text
load_seconds: 13.305
after_load free_gib: 89.236 / total_gib: 120.0
```

### T2I

結果:

```text
case: article_t2i_robotics_lab
seconds: 974.611
load_seconds: 12.934
output: result/classmethod_article_benchmark/article_t2i_robotics_lab_960x960_s35.jpg
```

設定:

```text
height: 960
width: 960
frames: 1
steps: 35
guidance: 1.0
seed: 201
```

出力 JPEG:

```text
resolution: 960x960
```

記事との速度比較:

```text
記事 DGX Spark: 22 sec generation time
本 ROCm 環境: 974.611 sec
倍率: 約 44.3x 遅い
```

補足:

- denoising 35 steps は約 201 sec で完了した。
- 総時間 974.611 sec の大半は VAE decode / postprocess / export 側。

### T2V

結果:

```text
case: article_t2v_red_cube_grasp
seconds: 483.187
output: result/classmethod_article_benchmark/article_t2v_red_cube_256p_24f_s35.mp4
```

設定:

```text
height: 256
width: 448
requested frames: 24
fps: 12
steps: 35
guidance: 1.0
seed: 202
```

出力 MP4:

```text
codec: H.264
resolution: 448x256
fps: 12
encoded frames: 21
duration: 1.75 sec
size: 30 KiB
```

補足:

- denoising 35 steps は約 52 sec で完了した。
- 総時間 483.187 sec の大半は VAE decode / postprocess / export 側。
- `num_frames=24` を指定したが、出力 MP4 の `nb_frames` は 21 だった。

記事との速度比較:

```text
記事 DGX Spark: 22 sec generation time
本 ROCm 環境: 483.187 sec
倍率: 約 22.0x 遅い
```

### I2V

結果:

```text
case: article_i2v_robot_arms
seconds: 166.890
output: result/classmethod_article_benchmark/article_i2v_robot_arms_256p_24f_s35.mp4
```

設定:

```text
height: 256
width: 448
requested frames: 24
fps: 12
steps: 35
guidance: 1.0
seed: 203
```

出力 MP4:

```text
codec: H.264
resolution: 448x256
fps: 12
encoded frames: 21
duration: 1.75 sec
size: 110 KiB
```

補足:

- denoising 35 steps は約 161 sec。
- T2V と比べると I2V は decode/export の追加待ちが小さく、総時間は denoising に近い。
- `num_frames=24` を指定したが、出力 MP4 の `nb_frames` は 21 だった。

記事との速度比較:

```text
記事 DGX Spark: 17 sec generation time
本 ROCm 環境: 166.890 sec
倍率: 約 9.8x 遅い
```

注意:

- 記事の `22 sec` / `17 sec` は動画尺ではなく生成所要時間。
- 本検証の出力動画尺は T2V / I2V ともに 1.75 sec。
- `num_frames=24`, `fps=12` を指定したが、エンコード後の MP4 は 21 frames だった。

### Policy Model

結果:

```text
status: success
output video: result/classmethod_policy_framework/action_policy_robot/vision.mp4
output json: result/classmethod_policy_framework/action_policy_robot/sample_outputs.json
total wall time: 約 1965 sec
generation phase: 約 1940 sec
sampler: 30 steps / 約 80 sec
```

設定:

```text
model_mode: policy
prompt: Put the pot to the left of the purple item.
domain_name: bridge_orig_lerobot
image_size: 480
fps: 5
num_frames: 189
num_steps: 30
guidance: 1.0
shift: 10.0
```

出力:

```text
vision.mp4: 640x480, 17 frames, 5 fps, 3.4 sec, 1.8 MiB
action sequence: 16 steps x 10 dims
```

記事との速度比較:

```text
記事 DGX Spark: 21 sec generation time
本 ROCm 環境: 約 1965 sec
倍率: 約 93.6x 遅い
```

実行上の注意:

- `vllm/vllm-omni-rocm:v0.20.0` は pull 済みで、`vllm_omni` と `/v1/videos` 系 route 実装は含まれていた。
- ただし同 ROCm image 内に Cosmos3 文字列・Cosmos3 固有 action path は見つからなかったため、今回の Policy 実行は `cosmos-framework` の `model_mode=policy` 経路で行った。
- `cosmos-framework` は NVML 前提の GPU メモリ検出があり、ROCm では `scripts/run_cosmos_framework_policy_rocm.py` で 120 GiB 固定値に差し替えた。
- framework の attention backend は FlashAttention / NATTEN 前提で、ROCm/gfx1151 では未導入だったため、PyTorch SDPA fallback をローカル実行スクリプトで差し込んだ。
- この fallback は検証用で、公式性能比較に使える最適化実装ではない。Policy の約 93.6x 遅い値は、fallback と ROCm 非公式経路込みの実測として扱う。

## 結果ファイル

```text
result/classmethod_article_benchmark/summary.json
result/classmethod_article_benchmark/article_t2i_summary.json
result/classmethod_article_benchmark/article_t2i_robotics_lab_960x960_s35.jpg
result/classmethod_article_benchmark/article_t2v_red_cube_256p_24f_s35.mp4
result/classmethod_article_benchmark/article_i2v_robot_arms_256p_24f_s35.mp4
result/classmethod_policy_framework/action_policy_robot/sample_outputs.json
result/classmethod_policy_framework/action_policy_robot/vision.mp4
```

## 考察

T2V は denoising 自体よりも VAE decode / postprocess / export の比率が大きい。以前の smoke test でも、短い動画生成で denoising 後の処理が支配的になる傾向があった。

I2V は記事と同様に T2V より速かった。条件画像があるため diffusion の収束や後段処理が安定している可能性がある。ただし、本検証では品質評価はしていない。

ROCm/gfx1151 では `float16` で動かしている。記事側は DGX Spark / CUDA 13.0 / BF16 前提のため、単純なハードウェア比較ではなく、非公式 ROCm 移植経路での実測として扱う。

Policy Model は最終的に成功したが、vLLM-Omni ROCm 経路ではなく framework 経路での実行。記事のような production path に近づけるには、Cosmos3 対応済み vLLM-Omni image を ROCm で成立させるか、FlashAttention/NATTEN 相当の attention backend を ROCm/gfx1151 に用意する必要がある。

## 今後の追加比較

より厳密に記事へ合わせるには以下を確認する必要がある。

- 記事の T2V 実行時の denoising steps が 35 で確定か。
- 記事の I2V 出力 frame 数、fps、steps。
- 記事の公式サンプル JSON が現行 `cosmos-framework` の `inputs/omni/i2v.json` と完全一致するか。
- ROCm 側で `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` を有効化した場合の SDPA / decode 時間。
- VAE decode / postprocess のプロファイル。
- Policy Model の action MSE を golden action と比較する後処理。
- vLLM-Omni ROCm image で Cosmos3 action path を成立させる追加検証。
