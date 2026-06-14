# Cosmos3 ROCm 7.2 Diffusers Runtime Image

Date: 2026-06-04

## Purpose

Avoid installing Diffusers from GitHub on every benchmark run.

Previous runner behavior:

```text
python -m pip install "diffusers @ git+https://github.com/huggingface/diffusers.git" ...
```

This made benchmarks dependent on `github.com` DNS and Git clone availability.

## Local Clone

Diffusers was cloned locally:

```text
third_party/diffusers
```

Commit:

```text
3e83f4348f0c6baea8bee3d1ff7676f50e11e74c
```

Size:

```text
156M
```

The repository `.gitignore` already excludes:

```text
third_party/
```

So this clone is not intended to be committed.

## Docker Image

Dockerfile:

```text
docker/cosmos3-rocm72-diffusers.Dockerfile
```

Build script:

```text
scripts/build_cosmos3_rocm72_diffusers_image.sh
```

Build command:

```text
scripts/build_cosmos3_rocm72_diffusers_image.sh cosmos3-rocm72-diffusers:local
```

Built image:

```text
cosmos3-rocm72-diffusers:local
sha256:fcadfc8bb6717d79c0d67dfe00d3ec9a288b64d598f57c32caccf340bf5e74fb
31.9GB
```

Installed versions:

```text
torch 2.9.1+rocm7.2.0.git7e1940d4
hip 7.2.26015-fc0010cf6a
diffusers 0.39.0.dev0 /opt/diffusers/src/diffusers/__init__.py
```

`Cosmos3OmniPipeline` import was verified.

## Runner Usage

The speed matrix runner now supports image and install override:

```text
COSMOS3_ROCM_IMAGE=cosmos3-rocm72-diffusers:local
COSMOS3_DIFFUSERS_INSTALL=true
```

Example:

```text
COSMOS3_ROCM_IMAGE=cosmos3-rocm72-diffusers:local \
COSMOS3_DIFFUSERS_INSTALL=true \
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned_streamk \
  --case i2v_article_transformer_streamk_probe \
  --execute
```

In dry-run output, the install section becomes:

```text
true && HF_HUB_DISABLE_XET=1 python3 ...
```

This confirms the GitHub Diffusers install is skipped.

## Stream-K Probe Result With Prebuilt Image

The prebuilt image successfully avoided the GitHub DNS/install problem.

Artifact:

```text
result/rocm_speed_matrix/aotriton_tuned_streamk/i2v_article_transformer_streamk_probe/summary.json
```

Result:

| Metric | Value |
|---|---:|
| Wall before allowed error | `57.350 sec` |
| `transformer_forward` | `0.178 sec` |
| Transformer calls | `1` |

Error:

```text
RuntimeError: Expected iter != ops_.end() to be true, but got false.
```

Interpretation:

- The dependency/image problem is fixed.
- Stream-K now reaches model execution without GitHub install.
- However, global `TENSILE_SOLUTION_SELECTION_METHOD=2` + `ROCBLAS_USE_HIPBLASLT=1` still triggers a PyTorch SDPA failure after entering transformer execution.
- The VAE encode workaround is not sufficient; transformer SDPA/attention also needs isolation or backend control before a valid 35-call Stream-K transformer timing can be collected.

