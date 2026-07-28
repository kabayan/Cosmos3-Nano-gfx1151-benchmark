FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HUB_DISABLE_XET=1

# cosmos-framework runtime deps (pyproject [project].dependencies minus torch/CUDA-only).
# torch is provided by the ROCm base image and must NOT be replaced.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      accelerate \
      av \
      cattrs \
      diffusers \
      einops \
      hydra-core \
      imageio \
      imageio-ffmpeg \
      loguru \
      msgpack \
      nvidia-ml-py \
      obstore \
      omegaconf \
      pydantic \
      requests \
      scipy \
      termcolor \
      "transformers>=4.57.1,<5.0.0" \
      tyro \
      uv \
      websockets \
      safetensors \
      pillow \
      iopath \
      "multi-storage-client==0.44.0" \
      boto3 \
      wandb \
      qwen_vl_utils

WORKDIR /workspace
