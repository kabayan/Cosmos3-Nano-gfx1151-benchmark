# Cosmos3-Nano custom MIOpen T2V/I2V retest results

Date: 2026-06-03

## Purpose

Re-test the article-equivalent Text-to-Video and Image-to-Video cases using the locally built custom MIOpen library.

The custom MIOpen build includes the experimental change in:

- `patches/miopen/allow-large-fp16-bf16-gemm-rest-workspace-experiment.patch`
- Environment flag: `MIOPEN_EXPERIMENT_LARGE_FP16_BF16_GEMM_REST=1`

This change was designed to validate whether large FP16/BF16 Conv3D descriptors can use the `GemmFwdRest` path instead of falling back to slower behavior.

## Test configuration

- Model: `nvidia/Cosmos3-Nano`
- Container: `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- GPU: `AMD Radeon Graphics`
- Custom MIOpen: `/workspace/result/miopen-build/rocm-7.2.0/install/lib/libMIOpen.so`
- Output directory: `result/custom-miopen-t2v-i2v-article-warm-full`
- Script: `scripts/benchmark_classmethod_article_t2v_i2v_rocm.py`

Common generation settings:

- Resolution: `448x256`
- Frames: `24`
- FPS: `12`
- Steps: `35`
- Guidance: `1.0`
- Mode warmup runs: `1`
- Measured runs: `1`
- Stage profiling: enabled
- VAE warmup: enabled

## Results

| Case | v1.0 measured | AOTriton baseline | custom MIOpen retest | vs AOTriton | vs v1.0 |
|---|---:|---:|---:|---:|---:|
| T2V red cube grasp | 56.120 sec | 46.996 sec | 46.861 sec | 0.135 sec faster, 0.3% | 9.259 sec faster, 16.5% |
| I2V robot arms | 167.219 sec | 94.219 sec | 93.976 sec | 0.243 sec faster, 0.3% | 73.243 sec faster, 43.8% |

Stage profile for the custom MIOpen measured runs:

| Case | Total | Transformer forward | VAE decode | Postprocess | Unattributed |
|---|---:|---:|---:|---:|---:|
| T2V red cube grasp | 46.861 sec | 41.514 sec | 4.127 sec | 0.019 sec | 1.201 sec |
| I2V robot arms | 93.976 sec | 88.560 sec | 4.144 sec | 0.018 sec | 1.254 sec |

VAE warmup:

| Run | VAE warmup |
|---|---:|
| v1.0 | 370.597 sec |
| AOTriton baseline | 379.856 sec |
| custom MIOpen retest | 369.272 sec |

## Article comparison

The referenced article reports approximately:

- T2V: `22 sec`
- I2V: `17 sec`

This retest result is:

- T2V: `46.861 / 22 = 2.13x` slower than the article value
- I2V: `93.976 / 17 = 5.53x` slower than the article value

## Interpretation

The custom MIOpen patch does not materially improve the T2V/I2V article-equivalent measured runs.

The observed change versus the AOTriton baseline is about `0.3%` for both T2V and I2V, which is effectively within normal run-to-run noise for this workload. The meaningful improvement from v1.0 to the current best configuration still comes from the ROCm/AOTriton-side transformer path, not from the custom MIOpen patch.

For T2V/I2V, the measured bottleneck remains transformer forward:

- T2V: `41.514 sec / 46.861 sec = 88.6%`
- I2V: `88.560 sec / 93.976 sec = 94.2%`

The VAE decode stage is only about `4.1 sec` in both measured runs. Therefore, even a large VAE-only improvement would have limited impact on T2V/I2V total latency unless the transformer path is also improved.

## Applicability of the custom MIOpen fix

The custom MIOpen change is most relevant to workloads that hit large FP16/BF16 Conv3D descriptors in VAE decode, especially the Policy Model video/action decode path where earlier profiling showed the ROCm convolution path was a dominant problem.

For the four article modes:

| Mode | Expected applicability |
|---|---|
| Text-to-Image | Low. It is not dominated by the large Conv3D VAE decode path. |
| Text-to-Video | Low to medium. It uses VAE decode, but measured decode is only about 4 sec and not the main bottleneck. |
| Image-to-Video | Low to medium. Same as T2V; transformer forward dominates. |
| Policy Model | High. This is the primary candidate for this MIOpen Conv3D descriptor fix. |

## Conclusion

The T2V/I2V retest succeeded, but the custom MIOpen fix does not provide a meaningful additional speedup for these two modes. The next T2V/I2V improvement target should remain transformer forward kernel efficiency on ROCm, while the custom MIOpen path should be evaluated primarily on Policy Model decode workloads.
