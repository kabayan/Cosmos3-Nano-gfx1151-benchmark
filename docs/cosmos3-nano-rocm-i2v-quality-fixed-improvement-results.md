# Cosmos3-Nano ROCm I2V quality-fixed improvement results

Date: 2026-06-04

## Scope

The user rejected step/frame reduction because it changes quality and breaks article-equivalent comparison. Therefore this run kept the I2V generation settings fixed:

- Mode: I2V
- Input: Cosmos3 official sample image and prompt
- Resolution: `448x256`
- Frames: `24`
- FPS: `12`
- Steps: `35`
- Guidance: `1.0`
- Seed: `203`

Only kernel-selection tuning was changed.

## Implementation

Added TunableOp Python API controls to:

```text
scripts/benchmark_classmethod_article_t2v_i2v_rocm.py
```

New options:

```text
--tunable-max-tuning-duration
--tunable-max-tuning-iterations
--tunable-rotating-buffer-size
```

Added runner variants and case to:

```text
scripts/run_rocm_speed_matrix.py
```

Variants:

```text
aotriton_i2v_deep_tunable
aotriton_i2v_deep_tuned
```

Case:

```text
i2v_article_warm_full_deep_tune
```

The deep tuning table is saved separately from v2.1:

```text
result/rocm_speed_matrix/tunableop_i2v_deep0.csv
```

This preserves the existing v2.1 table:

```text
result/rocm_speed_matrix/tunableop_results0.csv
```

## Commands

Deep tuning run:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_i2v_deep_tunable \
  --case i2v_article_warm_full_deep_tune \
  --execute
```

No-retune run using the deep table:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_i2v_deep_tuned \
  --case i2v_article_warm_full_deep_tune \
  --execute
```

TunableOp API settings:

```text
max_tuning_duration = 100
max_tuning_iterations = 200
rotating_buffer_size = 1024
```

## Results

| Run | Total | Transformer | VAE decode | Postprocess | Notes |
|---|---:|---:|---:|---:|---|
| v2.1 tuned | 77.160 sec | 71.691 sec | 4.195 sec | 0.021 sec | Existing persisted table |
| Deep tunable measured | 77.079 sec | 71.590 sec | 4.205 sec | 0.022 sec | Tuning enabled after warmup |
| Deep tuned measured | 77.225 sec | 71.743 sec | 4.201 sec | 0.022 sec | Tuning disabled, deep table loaded |

The deep tunable measured run was only `0.081 sec` faster than v2.1 total and `0.101 sec` faster in transformer time. The no-retune deep tuned run was slower than v2.1 by `0.065 sec` total and `0.052 sec` transformer time.

Conclusion: deeper TunableOp retune did not produce a meaningful quality-fixed improvement.

## Tuning overhead

The deep tunable warmup run showed the expected tuning cost:

| Run | Total | Transformer | Notes |
|---|---:|---:|---|
| Deep tunable warmup | 273.043 sec | 211.561 sec | First transformer step paid tuning search cost |
| Deep tuned warmup | 133.791 sec | 71.863 sec | No tuning search, but first measured-style pipeline run still has normal warmup overhead |

During deep tuning, the first denoising step took about `142 sec`, then later steps returned to about `2.05 sec/step`. This confirms the extra cost is tuning search, not generation quality work.

## Table comparison

Deep table:

```text
result/rocm_speed_matrix/tunableop_i2v_deep0.csv
```

Summary:

| Table | Entries | Lines | Notes |
|---|---:|---:|---|
| v2.1 `tunableop_results0.csv` | 42 | 47 | Broader table from previous T2V/I2V/T2I tuning |
| I2V deep `tunableop_i2v_deep0.csv` | 17 | 22 | I2V-focused table |

The deep table had `17` common operation/shape entries with the v2.1 table. Some selected kernels changed, but improvements and regressions were mixed.

Examples:

| Shape | v2.1 solution/time | Deep solution/time | Direction |
|---|---:|---:|---|
| `tn_4096_1904_12288` | `Hipblaslt_6167`, 8.2349 ms | `Hipblaslt_6175`, 8.3280 ms | slower |
| `tn_12288_1904_4096` | `Hipblaslt_6253`, 6.7254 ms | `Hipblaslt_6167`, 6.6420 ms | faster |
| `tn_4096_672_12288` | `Hipblaslt_6167`, 2.7780 ms | `Hipblaslt_6253`, 2.8894 ms | slower |
| `tn_4096_1904_4096` | `Hipblaslt_6253`, 2.4893 ms | `Hipblaslt_6253`, 2.4101 ms | faster |
| `tn_1024_1904_4096` | `Rocblas_-1162085926`, 0.3958 ms | `Rocblas_-1162085922`, 0.6435 ms | slower |

This explains why end-to-end time did not improve: the deeper search did not find a globally better set of kernels for the I2V workload.

## Current decision

Do not replace v2.1 with the deep table.

Keep using:

```text
result/rocm_speed_matrix/tunableop_results0.csv
```

The quality-fixed local improvement path through PyTorch TunableOp appears exhausted for this ROCm/PyTorch/gfx1151 stack. Further quality-fixed improvement would likely require one of:

- newer ROCm / hipBLASLt / rocBLAS / PyTorch with better algorithms for these shapes;
- custom GEMM/attention kernels for the I2V shapes;
- framework-level transformer implementation changes that preserve output settings;
- CUDA/Blackwell-class kernels as used by the article environment.

## Outputs

```text
result/rocm_speed_matrix/aotriton_i2v_deep_tunable/i2v_article_warm_full_deep_tune/summary.json
result/rocm_speed_matrix/aotriton_i2v_deep_tuned/i2v_article_warm_full_deep_tune/summary.json
result/rocm_speed_matrix/tunableop_i2v_deep0.csv
result/docs/tunableop_i2v_deep0_analysis.json
```

