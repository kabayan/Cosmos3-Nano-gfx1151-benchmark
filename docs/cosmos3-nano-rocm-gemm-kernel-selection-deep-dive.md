# Cosmos3-Nano ROCm GEMM kernel selection deep dive

Date: 2026-06-03

## Scope

Investigate the T2V/I2V transformer-forward bottleneck from the GEMM kernel selection point of view.

This work added:

- `scripts/analyze_rocm_gemm_kernels.py`
- T2V-only runner cases in `scripts/run_rocm_speed_matrix.py`
- T2V transformer-only selected-region rocprof
- TunableOp transformer-only selected-region rocprof for T2V and I2V
- GEMM logging runner cases for future rocBLAS/hipBLASLt shape capture

## New runner cases

Added cases:

- `t2v_article_warm_full`
- `t2v_article_transformer_rocprof`
- `t2v_article_gemm_log`
- `i2v_article_gemm_log`

The `*_transformer_rocprof` cases use ROCTx selected regions so that the rocprof kernel CSV covers measured-run `transformer.forward` only.

The `*_gemm_log` cases enable rocBLAS/hipBLASLt logging:

- `ROCBLAS_LAYER=14`
- `ROCBLAS_LOG_BENCH_PATH`
- `ROCBLAS_LOG_PROFILE_PATH`
- `ROCBLAS_LOG_TRACE_PATH`
- `HIPBLASLT_LOG_LEVEL=5`
- `HIPBLASLT_LOG_MASK=242`
- `HIPBLASLT_LOG_FILE`

These logging cases were added but not executed in this round because selected-region rocprof already showed the kernel-selection change, while rocBLAS logging would add another full warmup run per case.

## Runs

Commands executed:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton \
  --case t2v_article_transformer_rocprof \
  --execute

python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tunable \
  --case t2v_article_transformer_rocprof \
  --case i2v_article_transformer_rocprof \
  --execute
```

Analysis command:

```bash
python3 scripts/analyze_rocm_gemm_kernels.py \
  result/rocm_speed_matrix/aotriton/t2v_article_transformer_rocprof/rocprof/profile_kernel_stats.csv \
  result/rocm_speed_matrix/aotriton_tunable/t2v_article_transformer_rocprof/rocprof/profile_kernel_stats.csv \
  result/rocm_speed_matrix/aotriton/i2v_article_transformer_rocprof/rocprof/profile_kernel_stats.csv \
  result/rocm_speed_matrix/aotriton_tunable/i2v_article_transformer_rocprof/rocprof/profile_kernel_stats.csv \
  --out result/docs/gemm_kernel_analysis_aotriton_vs_tunable_t2v_i2v.json
```

## Stage results

| Mode | Variant | Total | Transformer | VAE decode | Transformer improvement |
|---|---:|---:|---:|---:|---:|
| T2V | AOTriton | 46.609 sec | 41.238 sec | 4.125 sec | baseline |
| T2V | AOTriton + TunableOp | 32.449 sec | 27.031 sec | 4.166 sec | 1.53x |
| I2V | AOTriton | 94.515 sec | 89.021 sec | 4.175 sec | baseline |
| I2V | AOTriton + TunableOp | 77.135 sec | 71.643 sec | 4.187 sec | 1.24x |

TunableOp clearly improves the measured transformer path. VAE decode is unchanged.

## Kernel category split

| Mode | Variant | Kernel total | GEMM | Attention | Elementwise | Other/copy |
|---|---|---:|---:|---:|---:|---:|
| T2V | AOTriton | 40.773 sec | 37.596 sec / 92.21% | 1.142 sec / 2.80% | 1.728 sec / 4.24% | 0.306 sec / 0.75% |
| T2V | AOTriton + TunableOp | 26.561 sec | 23.190 sec / 87.31% | 1.256 sec / 4.73% | 1.785 sec / 6.72% | 0.330 sec / 1.24% |
| I2V | AOTriton | 88.418 sec | 69.619 sec / 78.74% | 9.702 sec / 10.97% | 7.603 sec / 8.60% | 1.494 sec / 1.69% |
| I2V | AOTriton + TunableOp | 71.048 sec | 51.852 sec / 72.98% | 9.926 sec / 13.97% | 7.743 sec / 10.90% | 1.526 sec / 2.15% |

The improvement is almost entirely GEMM-side:

| Mode | GEMM baseline | GEMM TunableOp | GEMM improvement |
|---|---:|---:|---:|
| T2V | 37.596 sec | 23.190 sec | 1.62x |
| I2V | 69.619 sec | 51.852 sec | 1.34x |

Attention and elementwise time did not materially improve. In both modes they become a larger percentage after GEMM improves.

## Kernel selection change

### T2V

Baseline dominant GEMMs:

| Kernel family | Calls | Time | Avg |
|---|---:|---:|---:|
| `Cijk_Alik_Bljk_HHS_BH_MT128x128x32...WGM4` | 6300 | 19.810 sec | 3.144 ms |
| `Cijk_Alik_Bljk_HHS_BH_MT128x128x32...WGM1` | 6300 | 15.790 sec | 2.506 ms |

TunableOp dominant GEMMs:

| Kernel family | Calls | Time | Avg |
|---|---:|---:|---:|
| `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT96x128x32...` | 6300 | 13.313 sec | 2.113 ms |
| `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x64...` | 3780 | 5.538 sec | 1.465 ms |
| `Cijk_Alik_Bljk_HHS_BH_MT128x128x32...` | 2520 | 2.349 sec | 0.932 ms |

Interpretation:

- TunableOp changed the selected GEMM kernels.
- The two baseline dominant `MT128x128x32` kernels were replaced mainly by `Bias...UserArgs` kernels with `MT96x128x32` and `MT64x64x64`.
- This reduced T2V GEMM time from `37.596 sec` to `23.190 sec`.

### I2V

Baseline dominant GEMM:

| Kernel family | Calls | Time | Avg |
|---|---:|---:|---:|
| `Cijk_Alik_Bljk_HHS_BH_MT128x128x32...WGM4` | 12600 | 66.471 sec | 5.275 ms |

TunableOp dominant GEMM:

| Kernel family | Calls | Time | Avg |
|---|---:|---:|---:|
| `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT96x128x32...` | 12600 | 49.107 sec | 3.897 ms |

Interpretation:

- I2V has one overwhelmingly dominant GEMM family.
- TunableOp selected a different dominant GEMM kernel for the same call count.
- Average time improved from `5.275 ms` to `3.897 ms`.
- This accounts for most of the end-to-end I2V speedup.

## What this means

The working hypothesis is now stronger and more specific:

> T2V/I2V are slow because PyTorch's default ROCm GEMM selection chooses suboptimal Tensile GEMM kernels for Cosmos3 transformer shapes on `gfx1151`. PyTorch TunableOp finds faster GEMM kernels for the same workload, especially `Bias...UserArgs` kernels, reducing transformer time without changing model settings.

This explains why:

- VAE/MIOpen changes did not materially affect T2V/I2V.
- AOTriton helped first by improving attention availability/performance.
- TunableOp provides the next major speedup by changing GEMM kernel selection.

## Current limitation

Earlier checks looked for `result/rocm_speed_matrix/tunableop_results.csv`, but PyTorch TunableOp inserts the device ordinal into the filename. On this single-GPU environment the persisted file is:

```text
result/rocm_speed_matrix/tunableop_results0.csv
```

The file currently has 47 lines and includes the Cosmos3 T2V/I2V transformer GEMM entries.

This means the tuning table is persisted. The runner has been updated to make this explicit:

- `aotriton_tunable`: tune and save to `/workspace/result/rocm_speed_matrix/tunableop_results%d.csv`
- `aotriton_tuned`: load `/workspace/result/rocm_speed_matrix/tunableop_results%d.csv` with tuning disabled

The remaining practical concern is operational: when model shapes, ROCm version, PyTorch version, or GPU arch change, the validator rows may reject the saved table and retuning is required.

## Recommended next work

1. Run `t2v_article_gemm_log` and `i2v_article_gemm_log` for `aotriton` and `aotriton_tunable`.
   - Purpose: capture rocBLAS/hipBLASLt function-level shapes and backend selection.
   - This should explain whether the `Bias...UserArgs` kernels are selected via hipBLASLt, rocBLASLt/Tensile, or another PyTorch ROCm path.

2. Keep `aotriton_tuned` as the no-retuning benchmark/runtime variant.
   - It uses the persisted `tunableop_results0.csv`.
   - It should be used after `aotriton_tunable` has produced or refreshed the table.

3. Make `aotriton_tunable` the default T2V/I2V performance variant only if the retuning overhead is acceptable.
   - In the measured setup, warmup absorbs the tuning overhead.
   - For a persistent WebUI/server, this is acceptable if warmup is done at startup.
   - For single-shot CLI runs, first-run overhead can hide the benefit.

4. After GEMM tuning is stable, revisit attention and elementwise.
   - T2V TunableOp: attention + elementwise is now about `3.041 sec`.
   - I2V TunableOp: attention + elementwise is now about `17.669 sec`.
   - Attention remains relevant for I2V, but GEMM selection remains the largest lever.

## Outputs

- `result/rocm_speed_matrix/aotriton/t2v_article_transformer_rocprof/`
- `result/rocm_speed_matrix/aotriton_tunable/t2v_article_transformer_rocprof/`
- `result/rocm_speed_matrix/aotriton_tunable/i2v_article_transformer_rocprof/`
- `result/docs/gemm_kernel_analysis_aotriton_vs_tunable_t2v_i2v.json`
- `result/rocm_speed_matrix/tunableop_results0.csv`
