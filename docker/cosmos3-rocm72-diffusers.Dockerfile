FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/diffusers

COPY . /opt/diffusers

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      accelerate \
      av \
      cosmos_guardrail \
      huggingface_hub \
      imageio \
      imageio-ffmpeg \
      scipy \
      transformers \
      safetensors \
      pillow && \
    python -m pip install --no-cache-dir --no-deps -e /opt/diffusers && \
    python - <<'PY'
import diffusers
print("diffusers", diffusers.__version__, diffusers.__file__)
PY

WORKDIR /workspace
