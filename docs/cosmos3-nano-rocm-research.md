# nvidia/Cosmos3-Nano を ROCm/gfx1151 で動かすための調査メモ

作成日: 2026-06-01

対象環境: AMD Ryzen AI Max+ 395 / Radeon 8060S / gfx1151 / ROCm 7.2.0

## 結論

`nvidia/Cosmos3-Nano` は公式には NVIDIA CUDA GPU 前提のモデルです。ROCm で動かす場合は公式サポート外の移植・検証扱いになります。

ただし、このマシンの GPU である `gfx1151` / AMD Ryzen AI Max+ 395 は、AMD の ROCm 7.2 Radeon/Ryzen 向け Linux support matrix に載っており、PyTorch 2.9 + ROCm 7.2 + Python 3.12 が official production support とされています。したがって、最も現実的な検証順は以下です。

1. ROCm 7.2 + PyTorch 2.9.1 wheel で `torch.cuda.is_available()` を通す
2. Diffusers の `Cosmos3OmniPipeline` を ROCm 上で import できるか確認する
3. 最小条件の `num_frames=1`, 480p 以下, `torch_dtype=torch.float16` でロード可否を確認する
4. BF16 が必要な箇所、CUDA 固定、未対応 Triton/AOTriton kernel、メモリ不足のいずれで止まるかを切り分ける

ROCm で最初から vLLM-Omni の Cosmos3 Generator serving を狙うのはリスクが高いです。vLLM/vLLM-Omni 自体は ROCm 対応がありますが、Cosmos3 の official examples は CUDA/NVIDIA 前提で、ROCm + gfx1151 + Cosmos3 の動作実績は公式には確認できていません。

## 現在環境の ROCm 状態

確認コマンドの結果:

| 項目 | 結果 |
| --- | --- |
| GPU | AMD Radeon Graphics, gfx1151 |
| APU | AMD Ryzen AI Max+ 395 / Radeon 8060S |
| ROCm | 7.2.0 |
| HIP | 7.2.26015 |
| ROCm path | `/opt/rocm-7.2.0` |
| `rocminfo` | `gfx1151` を認識 |
| `rocm-smi` | GPU node 0 を認識 |
| `/dev/kfd` | `root:render`, `crw-rw----` |
| `/dev/dri/renderD128` | `root:render`, `crw-rw----` |
| user groups | `video`, `render`, `docker` に所属 |
| system memory | 124 GiB |
| visible GPU memory | `rocminfo` 上は約 120 GiB class の APU memory pool |
| Python | 3.12.3 |
| PyTorch | 未インストール |

`rocm-smi` は `VRAM% 96%` と表示していますが、APU unified memory 環境では一般的な dGPU の VRAM 表示と同じ意味で扱わない方が安全です。PyTorch 側で `torch.cuda.mem_get_info()` を確認して、実際に allocator が使える量を判断してください。

## AMD 公式情報から見たサポート状況

ROCm 7.2 の Ryzen Linux support matrix:

- 対応 OS: Ubuntu 24.04.3
- 対応 architecture: `gfx1150`, `gfx1151`
- 対応 hardware:
  - AMD Ryzen AI Max+ 395
  - AMD Ryzen AI Max 390
  - AMD Ryzen AI Max 385
  - AMD Ryzen AI 9 HX 375 / 370
  - AMD Ryzen AI 9 365
  - AMD Ryzen AI 9 HX 475 / 470 / 465
- PyTorch + ROCm:
  - PyTorch 2.9
  - ROCm 7.2
  - Python 3.12
  - official production support
- AI data types:
  - FP16 のみ公式検証
  - 他 dtype は動く可能性はあるが、公式検証外

重要な不整合:

- Cosmos3-Nano のモデルカードは BF16 tested と明記している
- AMD Ryzen matrix は FP16 のみ公式検証
- そのため、ROCm/gfx1151 では `torch.bfloat16` ではなく `torch.float16` へ落として試す必要がある可能性が高い
- FP16 が Cosmos3 の全コンポーネントで安全に動く保証はない

## PyTorch ROCm 7.2 の導入候補

AMD の Ryzen APU 向けドキュメントでは、ROCm 7.2 用に Python 3.12 wheel を直接入れる手順が示されています。

作業用 venv を使う場合:

```bash
python3 -m venv .venv-rocm
source .venv-rocm/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

ROCm 7.2 / PyTorch 2.9.1 wheel の取得:

```bash
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
```

install:

```bash
pip uninstall -y torch torchvision triton torchaudio
pip install \
  torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl \
  torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
  torchaudio-2.9.0+rocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
  triton-3.5.1+rocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl
```

AOTriton を有効化する場合:

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

確認:

```bash
python -c 'import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.mem_get_info())'
python -m torch.utils.collect_env
```

ROCm では PyTorch API 名が `torch.cuda` のまま使われます。これは CUDA GPU を意味するのではなく、PyTorch の GPU backend abstraction として AMD GPU にも使われます。

## Diffusers 経路の検証

Cosmos3 の公式 Diffusers docs は `device_map="cuda"` と `torch_dtype=torch.bfloat16` の例を出しています。ROCm では PyTorch 側の device string は通常 `cuda` のままで良いですが、dtype は最初から BF16 にせず FP16 で試す価値があります。

install 候補:

```bash
pip install \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate \
  av \
  cosmos_guardrail \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  transformers \
  safetensors
```

最小 import test:

```bash
python - <<'PY'
import torch
from diffusers import Cosmos3OmniPipeline

print("torch", torch.__version__)
print("hip", torch.version.hip)
print("gpu", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
print("mem:", torch.cuda.mem_get_info())
print("Cosmos3OmniPipeline import ok")
PY
```

最小ロード test:

```python
import torch
from diffusers import Cosmos3OmniPipeline

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.float16,
    device_map="cuda",
    enable_safety_checker=False,
)
print("loaded")
```

最小生成 test:

```python
import torch
from diffusers import Cosmos3OmniPipeline

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.float16,
    device_map="cuda",
    enable_safety_checker=False,
)

result = pipe(
    prompt="A mobile robot in a clean warehouse aisle.",
    negative_prompt="blurry, distorted, low quality",
    num_frames=1,
    height=480,
    width=832,
    num_inference_steps=4,
    guidance_scale=1.0,
)

result.video[0].save("cosmos3_rocm_t2i_smoke.jpg", format="JPEG", quality=85)
```

この test の目的は品質ではなく、ロード、kernel dispatch、VAE decode、基本的な scheduler path が通るかの確認です。

## 想定される failure mode と切り分け

### `hipErrorNoBinaryForGPU`

意味:

- インストールした PyTorch / Triton / 拡張ライブラリが `gfx1151` 用 code object を含んでいない

確認:

```bash
rocminfo | grep -E 'Name:|gfx'
TORCHDIR=$(dirname $(python -c 'import torch; print(torch.__file__)'))
roc-obj-ls -v "$TORCHDIR/lib/libtorch_hip.so" | grep gfx1151
```

対応:

- AMD の Ryzen APU 向け ROCm 7.2 wheel を使う
- PyTorch.org nightly wheel ではなく repo.radeon.com の wheel を優先する
- それでも駄目なら PyTorch/Triton/diffusers/vLLM の source build が必要

### BF16 関連の error または品質劣化

意味:

- Cosmos3 は BF16 tested だが、Ryzen/gfx1151 matrix は FP16 のみ公式検証

確認:

```python
import torch
print(torch.cuda.is_bf16_supported())
```

対応:

- `torch_dtype=torch.float16` で試す
- BF16 を強制している箇所があれば FP16 に落とす
- 結果品質や数値安定性は未保証

### Triton / AOTriton / attention kernel error

意味:

- Diffusion transformer の attention / GEMM / SDPA が gfx1151 で未対応または未最適化

対応候補:

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PYTORCH_TUNABLEOP_ENABLED=1
```

それでも止まる場合は、該当 operator を特定し、PyTorch eager fallback、xFormers 無効化、または source build が必要になります。

### MIOpen convolution error

AMD Ryzen limitations では、Generative AI workload で convolution error が出る場合の環境変数が案内されています。

```bash
export MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=1
```

### memory / allocator error

Cosmos3-Nano のファイル総量は約 32.5 GiB ですが、生成時は重み以外に activation、VAE、tokenizer、guardrail、scheduler、temporary buffer が必要です。

対策:

- まず `num_frames=1`
- 480p 以下
- `num_inference_steps=4`
- `enable_safety_checker=False`
- sound 無効
- `torch.cuda.mem_get_info()` を各段階で記録
- 720p / 189 frames は後回し

## vLLM / vLLM-Omni 経路

vLLM は ROCm official Docker image を提供しています。

ROCm vLLM container の基本形:

```bash
docker run --rm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai-rocm:latest \
  --model Qwen/Qwen3-0.6B
```

ただし Cosmos3-Nano については注意が必要です。

- Reasoner: `vllm-cosmos3` が CUDA 前提でないか確認が必要
- Generator: Cosmos3 examples は vLLM-Omni の `Cosmos3OmniDiffusersPipeline` を使う
- vLLM-Omni には ROCm platform 実装が存在するが、`nvidia/Cosmos3-Nano` + ROCm/gfx1151 の公式動作確認は見つかっていない
- vLLM-Omni の公式 Cosmos3 Docker image `vllm/vllm-omni:cosmos3` は NVIDIA/CUDA 前提として案内されている

したがって、ROCm での順序は以下を推奨します。

1. 通常 vLLM ROCm container で小型モデル `Qwen/Qwen3-0.6B` を起動
2. vLLM-Omni ROCm の import / platform detection を確認
3. Cosmos3 Reasoner だけ試す
4. Generator は Diffusers で成功してから vLLM-Omni へ進む

## 実行可否の判定基準

### Phase 0: ROCm/PyTorch

成功条件:

```bash
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.mem_get_info())'
```

が成功し、`True` と AMD Radeon Graphics が出る。

### Phase 1: Diffusers import

成功条件:

```bash
python -c 'from diffusers import Cosmos3OmniPipeline; print("ok")'
```

### Phase 2: model load

成功条件:

```python
Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.float16,
    device_map="cuda",
    enable_safety_checker=False,
)
```

が完了する。

### Phase 3: minimal image generation

成功条件:

- `num_frames=1`
- 480p
- `num_inference_steps=4`
- JPEG が出力される

### Phase 4: short video

成功条件:

- `num_frames=5` から開始
- 次に `num_frames=25`
- その後 81 frames / 189 frames へ拡大

## 推奨しないこと

- 最初から 720p / 189 frames / sound enabled で走らせる
- 最初から vLLM-Omni serving を狙う
- BF16 前提のまま進める
- `rocm-smi` の VRAM 表示だけで実行可能メモリを判断する
- PyTorch.org nightly wheel と AMD repo.radeon wheel を混在させる

## 参照元

- NVIDIA Cosmos3-Nano model card: https://huggingface.co/nvidia/Cosmos3-Nano
- Hugging Face Diffusers Cosmos 3 docs: https://huggingface.co/docs/diffusers/main/api/pipelines/cosmos3
- AMD ROCm 7.2 Ryzen Linux support matrix: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html
- AMD Install PyTorch for ROCm on Ryzen APUs: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installryz/native_linux/install-pytorch.html
- AMD Ryzen limitations and recommended settings: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/limitations/limitationsryz.html
- vLLM GPU installation docs: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- vLLM-Omni ROCm platform docs: https://docs.vllm.ai/projects/vllm-omni/en/stable/api/vllm_omni/platforms/rocm/
