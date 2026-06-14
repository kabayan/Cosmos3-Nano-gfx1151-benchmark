# Cosmos3-Nano ROCm v2: rocprof / TunableOp / Policy action-only results

Date: 2026-06-03

## Scope

Executed the requested v2 follow-up items:

1. Policy article-equivalent video + action output with `rocprofv3`.
2. T2I article condition with `rocprofv3`.
3. TunableOp re-check limited to T2I and I2V transformer-heavy article conditions.
4. Formal Policy action-only mode in the runner.

Runtime was `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1` with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.

## Runner changes

- Added `rocprof_wrap()` to `scripts/run_rocm_speed_matrix.py`.
- Added runner cases:
  - `t2i_article_warm_full_rocprof`
  - `policy_article_rocprof`
  - `i2v_article_warm_full`
  - `policy_article_action_only`
- Added `--action-only` to `scripts/run_cosmos_framework_policy_rocm.py`.
  - It skips real vision decode by returning zero vision tensors.
  - It removes generated video files from `sample_outputs.json`.
  - It keeps `content.action` in the output JSON.

## Benchmark results

| Case | Baseline v2 AOTriton | New result | Stage detail | Result |
|---|---:|---:|---|---|
| T2I warm measured | 104.297 sec | 88.344 sec | transformer 101.961 -> 85.995 sec | 1.18x faster |
| I2V warm measured | 94.219 sec | 77.083 sec | transformer 88.753 -> 71.621 sec | 1.22x faster |
| Policy normal | 1905.674 sec | 1912.329 sec under rocprof | decode 1058.894 sec, sampling path 781.257 sec | No material speed change |
| Policy skip vision decode | 783.394 sec | 770.385 sec action-only | decode 0.00012 sec | Formal action-only output works |

TunableOp did not create `result/rocm_speed_matrix/tunableop_results.csv`, but the measured T2I/I2V runs improved. This means the benchmark result is usable, but the exact per-op tuned table was not persisted by PyTorch in this environment.

## rocprof outputs

T2I:

- `result/rocm_speed_matrix/aotriton/t2i_article_warm_full_rocprof/rocprof/profile_kernel_stats.csv`
- `result/rocm_speed_matrix/aotriton/t2i_article_warm_full_rocprof/rocprof/profile_kernel_trace.csv`
- `result/rocm_speed_matrix/aotriton/t2i_article_warm_full_rocprof/rocprof/profile_memory_copy_stats.csv`

Policy:

- `result/rocm_speed_matrix/aotriton/policy_article_rocprof/rocprof/profile_kernel_stats.csv`
- `result/rocm_speed_matrix/aotriton/policy_article_rocprof/rocprof/profile_kernel_trace.csv`
- `result/rocm_speed_matrix/aotriton/policy_article_rocprof/rocprof/profile_memory_copy_stats.csv`

`rocprofv3` emitted timestamp correction warnings, but both runs completed and wrote kernel/memory-copy CSV files.

## Top rocprof kernels

T2I profile covered VAE warmup, warmup generation, and measured generation. Therefore aggregate kernel stats are dominated by VAE decode warmup, not only by the measured transformer window.

| T2I top kernel | Time | Share | Calls | Meaning |
|---|---:|---:|---:|---|
| `naive_conv_ab_nonpacked_fwd_ncdhw_half_double_half` | 575.075 sec | 58.38% | 176 | VAE 3D conv path |
| `naive_conv_ab_nonpacked_fwd_nchw_half_double_half` | 190.419 sec | 19.33% | 80 | VAE 2D conv path |
| Tensile GEMM `Cijk_Alik_Bljk_HHS...` | 106.370 sec | 10.80% | 12600 | transformer matmul path |
| Tensile GEMM `Cijk_Alik_Bljk_HHS...` | 43.033 sec | 4.37% | 12600 | transformer matmul path |
| `attn_fwd` | 25.160 sec | 2.55% | 5040 | attention kernel |

Policy aggregate profile is dominated by the article-equivalent vision decode.

| Policy top kernel | Time | Share | Calls | Meaning |
|---|---:|---:|---:|---|
| `naive_conv_ab_nonpacked_fwd_ncdhw_ushort_double_ushort` | 1515.715 sec | 79.61% | 368 | vision VAE 3D conv path |
| `naive_conv_ab_nonpacked_fwd_nchw_ushort_double_ushort` | 317.550 sec | 16.68% | 144 | vision VAE 2D conv path |
| Tensile GEMM `Cijk_Alik_Bljk_BBS...` | 33.684 sec | 1.77% | 7560 | transformer matmul path |
| `Im3d2Col` | 9.594 sec | 0.50% | 452 | convolution lowering |
| `attn_fwd` | 2.987 sec | 0.16% | 2160 | attention kernel |

Memory-copy stats were small compared with kernel time:

- T2I H2D copies: 0.704 sec total.
- Policy H2D copies: 1.293 sec total.

## Conclusions

Policy is not primarily blocked by host/device copy or attention. The article-equivalent Policy run is dominated by vision VAE decode conv kernels. Action-only is useful when only robot actions are required, but it is not equivalent to article video+action output.

T2I/I2V transformer throughput still has improvement room. TunableOp plus AOTriton improved steady-state measured T2I and I2V, but the first tuned run has large tuning overhead and PyTorch did not persist a tunable CSV in this environment.

For a stricter T2I transformer-only rocprof, the next runner change should profile only the measured generation window, excluding synthetic VAE warmup and warmup generation.
