ARG BASE_IMAGE=cosmos3-rocm72-diffusers:local
FROM ${BASE_IMAGE}

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2

# Keep the stable ROCm 7.2 userspace and the validated Diffusers checkout intact;
# only replace the PyTorch pair in this disposable comparison image.
RUN python -m pip install --no-cache-dir --upgrade \
      torch==2.13.0 \
      torchvision==0.28.0 \
      --index-url "${PYTORCH_INDEX_URL}" && \
    python - <<'PY'
import torch
import torchvision

assert torch.__version__.startswith("2.13.0+rocm7.2"), torch.__version__
assert torchvision.__version__.startswith("0.28.0+rocm7.2"), torchvision.__version__
print("torch", torch.__version__, "hip", torch.version.hip)
print("torchvision", torchvision.__version__)
PY

# The ROCm 7.2 index does not publish a torchaudio 2.13 wheel. Keeping the
# base image's 2.9 binary makes Transformers discover and import an ABI-
# incompatible optional package, so omit it from this video-only comparison.
RUN python -m pip uninstall --yes torchaudio
