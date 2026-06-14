# Cosmos3-Nano ROCm v2: VAE conv and transformer-only rocprof deep dive

Date: 2026-06-03

## Scope

This note covers two follow-up investigations:

1. Information needed to improve the ROCm-side Wan VAE decode convolution path.
2. Transformer-only rocprof for T2I and I2V, excluding VAE warmup and VAE decode from aggregate kernel stats.

## Environment

Container:

- `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`
- PyTorch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- MIOpen library: `/opt/rocm/lib/libMIOpen.so.1.0.70200`
- `torch.backends.cudnn.enabled`: `True`
- `MIOPEN_FIND_MODE`: unset
- GPU: `AMD Radeon Graphics / gfx1151`

## ROCm VAE Conv Findings

Policy article-equivalent video + action output is dominated by Wan VAE decode convolution kernels.

From `result/rocm_speed_matrix/aotriton/policy_article_rocprof/rocprof/profile_kernel_stats.csv`:

| Kernel | Calls | Time | Share | Avg |
|---|---:|---:|---:|---:|
| `naive_conv_ab_nonpacked_fwd_ncdhw_ushort_double_ushort` | 368 | 1515.715 sec | 79.61% | 4118.790 ms |
| `naive_conv_ab_nonpacked_fwd_nchw_ushort_double_ushort` | 144 | 317.550 sec | 16.68% | 2205.209 ms |

From T2I full rocprof, including VAE warmup:

| Kernel | Calls | Time | Share | Avg |
|---|---:|---:|---:|---:|
| `naive_conv_ab_nonpacked_fwd_ncdhw_half_double_half` | 176 | 575.075 sec | 58.38% | 3267.474 ms |
| `naive_conv_ab_nonpacked_fwd_nchw_half_double_half` | 80 | 190.419 sec | 19.33% | 2380.235 ms |

Interpretation:

- The critical VAE path is convolution, not attention, GEMM, memory copy, or CPU/GPU transfer.
- Kernel names show a `naive_conv` path, with both 3D (`ncdhw`) and 2D (`nchw`) variants.
- Policy uses BF16-like `ushort` kernels; T2I VAE warmup uses FP16-like `half` kernels.
- Memory copy totals were small in previous rocprof runs, so copy bandwidth is not the primary issue.

## What Is Needed To Improve VAE Decode

The required improvement is to move these shapes away from `naive_conv` and onto optimized convolution solvers.

Needed information or changes:

1. Exact convolution descriptors for the Wan VAE decode path:
   - input/output tensor dimensions
   - filter dimensions
   - stride, padding, dilation
   - grouping
   - dtype
   - tensor layout/contiguity

2. MIOpen solver behavior for those descriptors:
   - whether MIOpen is called by PyTorch for these convs
   - whether FindDb/PerfDb has entries for gfx1151
   - whether `MIOPEN_FIND_MODE=NORMAL`, `HYBRID`, `FAST`, or `DYNAMIC_HYBRID` changes solver choice
   - whether `MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE` can populate a useful user PerfDb

3. Dispatch path confirmation:
   - whether `naive_conv` is an ATen fallback or a selected ROCm/MIOpen fallback kernel
   - whether changing dtype from BF16 to FP16 changes Policy VAE decode kernels
   - whether layout conversion changes solver availability

4. Practical optimization options:
   - tune MIOpen FindDb/PerfDb for these exact shapes
   - try FP16 Policy VAE decode instead of BF16 if numerically acceptable
   - test VAE tiling/chunking to alter shape and solver choice
   - test a newer ROCm/PyTorch/MIOpen stack
   - if video is not required, use Policy action-only mode

Official MIOpen docs relevant to this:

- MIOpen Find modes are controlled by `MIOPEN_FIND_MODE`; available modes include `NORMAL`, `FAST`, `HYBRID`, and `DYNAMIC_HYBRID`.
- MIOpen FindDb stores results from Find calls; user FindDb takes precedence over system FindDb.
- MIOpen environment variables include `MIOPEN_FIND_MODE`, `MIOPEN_FIND_ENFORCE`, and `MIOPEN_DEBUG_DISABLE_FIND_DB`.

References:

- https://rocm.docs.amd.com/projects/MIOpen/en/docs-6.0.0/find_and_immediate.html
- https://rocm.docs.amd.com/projects/MIOpen/en/docs-7.1.1/reference/env_variables.html
- https://rocm.docs.amd.com/projects/MIOpen/en/docs-5.7.1/finddb.html

## Transformer-only rocprof Implementation

The first selected-region attempt used `rocprofv3 --selected-regions` only and produced no CSV. A minimal matmul test showed that ROCProfiler v3 needs marker tracing enabled too.

Working rocprof setup:

```bash
rocprofv3 \
  --kernel-trace \
  --memory-copy-trace \
  --marker-trace \
  --stats \
  --selected-regions \
  ...
```

Python benchmark changes:

- Added `--rocprof-transformer-only`.
- Loaded `librocprofiler-sdk-roctx.so` or `libroctx64.so`.
- Around measured-run `transformer.forward` only:
  - `roctxProfilerResume(0)`
  - `roctxRangePushA("transformer_forward")`
  - run transformer forward
  - synchronize
  - `roctxRangePop()`
  - `roctxProfilerPause(0)`

Runner cases added:

- `t2i_article_transformer_rocprof`
- `i2v_article_transformer_rocprof`

## Transformer-only Results

T2I selected-region output:

- `result/rocm_speed_matrix/aotriton/t2i_article_transformer_rocprof/rocprof/profile_kernel_stats.csv`
- `result/rocm_speed_matrix/aotriton/t2i_article_transformer_rocprof/rocprof/profile_kernel_trace.csv`
- `result/rocm_speed_matrix/aotriton/t2i_article_transformer_rocprof/rocprof/profile_marker_api_trace.csv`

I2V selected-region output:

- `result/rocm_speed_matrix/aotriton/i2v_article_transformer_rocprof/rocprof/profile_kernel_stats.csv`
- `result/rocm_speed_matrix/aotriton/i2v_article_transformer_rocprof/rocprof/profile_kernel_trace.csv`
- `result/rocm_speed_matrix/aotriton/i2v_article_transformer_rocprof/rocprof/profile_marker_api_trace.csv`

Measured stage profile:

| Case | Total | Transformer | VAE decode | Unattributed |
|---|---:|---:|---:|---:|
| T2I | 104.498 sec | 102.096 sec | 1.769 sec | 0.624 sec |
| I2V | 94.515 sec | 89.021 sec | 4.175 sec | 1.299 sec |

Transformer-only rocprof category split:

| Case | GEMM | Attention | Elementwise | Other | Copy/fill |
|---|---:|---:|---:|---:|---:|
| T2I | 77.779 sec / 76.66% | 12.582 sec / 12.40% | 9.242 sec / 9.11% | 1.319 sec / 1.30% | 0.544 sec / 0.54% |
| I2V | 69.619 sec / 78.74% | 9.702 sec / 10.97% | 7.603 sec / 8.60% | 1.038 sec / 1.17% | 0.456 sec / 0.52% |

Top T2I transformer-only kernels:

| Kernel category | Time | Share | Calls | Avg |
|---|---:|---:|---:|---:|
| Tensile GEMM | 52.643 sec | 51.88% | 6300 | 8.356 ms |
| Tensile GEMM | 21.351 sec | 21.04% | 6300 | 3.389 ms |
| `attn_fwd` | 12.582 sec | 12.40% | 2520 | 4.993 ms |
| Tensile GEMM | 2.820 sec | 2.78% | 5040 | 0.559 ms |

Top I2V transformer-only kernels:

| Kernel category | Time | Share | Calls | Avg |
|---|---:|---:|---:|---:|
| Tensile GEMM | 66.471 sec | 75.18% | 12600 | 5.275 ms |
| `attn_fwd` | 9.702 sec | 10.97% | 2520 | 3.850 ms |
| Tensile GEMM | 1.545 sec | 1.75% | 2520 | 0.613 ms |
| Elementwise pow | 1.306 sec | 1.48% | 10150 | 0.129 ms |

## Conclusions

Policy improvement requires VAE convolution solver work. The target is to replace `naive_conv_ab_nonpacked_*` with an optimized MIOpen/CK/implicit-GEMM path, or to change dtype/layout/tiling so that an optimized solver becomes applicable.

T2I/I2V improvement requires transformer GEMM and attention optimization. The selected-region profile confirms that the true transformer bottleneck is GEMM first, attention second, and elementwise third. VAE is not part of the measured transformer kernel profile.

## MIOpen Find validation for Policy

This probe was run after the initial VAE conv rocprof result to check whether the Policy improvement hypothesis was correctly defined.

Run:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_miopen_find \
  --case policy_article_miopen_find \
  --execute
```

Environment additions:

- `MIOPEN_ENABLE_LOGGING=1`
- `MIOPEN_ENABLE_LOGGING_CMD=1`
- `MIOPEN_LOG_LEVEL=5`
- `MIOPEN_FIND_MODE=NORMAL`
- `MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE`
- `MIOPEN_USER_DB_PATH=/workspace/result/rocm_speed_matrix/miopen_user_db`

Outputs:

- Benchmark: `result/rocm_speed_matrix/aotriton_miopen_find/policy_article_miopen_find/benchmark.json`
- MIOpen log: `result/rocm_speed_matrix/aotriton_miopen_find/policy_article_miopen_find/miopen/miopen_run.log`
- User FindDb: `result/rocm_speed_matrix/miopen_user_db/gfx1151_20.HIP.3_5_1_5b515cf1bc.ufdb.txt`

Benchmark comparison:

| Case | `generate_batch` | `decode` | `generate_samples_from_batch` |
|---|---:|---:|---:|
| Baseline `aotriton/policy_article` | 1905.674 sec | 1052.101 sec | 781.507 sec |
| `aotriton_miopen_find/policy_article_miopen_find` | 1904.524 sec | 1058.860 sec | 773.255 sec |
| `aotriton_miopen_warm/policy_article` | 1909.807 sec | 1062.072 sec | 774.877 sec |

Interpretation: the cold Find run did not improve end-to-end Policy time. It slightly improved sampler time and slightly worsened decode time, but the changes are within the same order as normal run variance and Find/logging overhead. Therefore this run should be treated as a solver-dispatch probe, not a final performance benchmark.

The warmed FindDb run used the populated user FindDb with reduced logging:

- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
- `MIOPEN_FIND_MODE=NORMAL`
- `MIOPEN_USER_DB_PATH=/workspace/result/rocm_speed_matrix/miopen_user_db`

It did not improve Policy throughput either:

- `generate_batch`: 1909.807 sec, +4.133 sec / +0.22% vs baseline
- `decode`: 1062.072 sec, +9.971 sec / +0.95% vs baseline
- `generate_samples_from_batch`: 774.877 sec, -6.630 sec / -0.85% vs baseline

This means the current MIOpen user FindDb is useful for explaining solver dispatch, but not sufficient as a performance optimization for this article-equivalent Policy workload. The remaining improvement target is still the residual large 3D BF16 conv descriptors that choose `ConvDirectNaiveConvFwd`, plus any framework-level decode path changes needed to avoid those descriptors.

MIOpen Find summary from the log:

| Final solver | Count |
|---|---:|
| `GemmFwdRest` | 44 |
| `GemmFwd1x1_0_1` | 17 |
| `ConvDirectNaiveConvFwd` | 2 |

The original broad hypothesis, "Policy VAE decode is slow because BF16 conv falls to MIOpen naive conv", was directionally correct but too broad. The corrected hypothesis is:

> `MIOPEN_FIND_MODE=NORMAL` can route many Policy BF16 NCDHW/NCHW conv descriptors to GEMM solvers, but a small number of large 3D BF16 conv descriptors still have no usable optimized solver and remain on `ConvDirectNaiveConvFwd`; these remaining shapes are the main VAE/vision-decode improvement target.

The two residual naive shapes were:

| Estimated solver time | Workspace | Descriptor | Final solver |
|---:|---:|---|---|
| 19869.900 ms | 0 | `160-18-274-370-3x3x3-160-16-272-368-1-0x0x0-1x1x1-1x1x1-0-NCDHW-BF16-F` | `ConvDirectNaiveConvFwd` |
| 22193.800 ms | 0 | `512-6-242-322-3x3x3-256-4-240-320-1-0x0x0-1x1x1-1x1x1-0-NCDHW-BF16-F` | `ConvDirectNaiveConvFwd` |

Examples of large descriptors that did move to GEMM:

| Estimated solver time | Workspace | Descriptor | Final solver |
|---:|---:|---|---|
| 218.687 ms | 6918635520 | `320-18-138-186-3x3x3-320-16-136-184-1-0x0x0-1x1x1-1x1x1-0-NCDHW-BF16-F` | `GemmFwdRest` |
| 162.360 ms | 4246732800 | `1024-6-122-162-3x3x3-512-4-120-160-1-0x0x0-1x1x1-1x1x1-0-NCDHW-BF16-F` | `GemmFwdRest` |
| 123.222 ms | 4246732800 | `256-6-242-322-3x3x3-256-4-240-320-1-0x0x0-1x1x1-1x1x1-0-NCDHW-BF16-F` | `GemmFwdRest` |

Procedure to validate or explain this hypothesis:

1. Enable MIOpen logging and command logging, force `MIOPEN_FIND_MODE=NORMAL`, and write to a dedicated user FindDb path.
2. Run the article-equivalent Policy video + action output path.
3. Extract every `Find Start` descriptor and matching `FW Chosen Algorithm`.
4. Split descriptors by final solver, workspace size, dtype, layout, and estimated time.
5. Compare `benchmark.json` against the baseline run.
6. Re-run with the populated FindDb and less verbose logging to measure warmed-DB performance separately from cold Find overhead.
7. If warmed performance improves, run rocprof again and verify that the dominant kernels move away from `naive_conv_ab_nonpacked_*`.
8. If warmed performance does not improve, keep the FindDb result as diagnostic evidence and move to dtype/layout/tiling or a newer ROCm/MIOpen stack for the residual `ConvDirectNaiveConvFwd` descriptors.

Required improvement work:

- Existing FindDb tuning is not enough for the two residual workspace-0 naive descriptors.
- Practical next probes are:
  - replay the two `MIOpenDriver convbfp16 ...` commands directly
  - test FP16 equivalents of those descriptors
  - test VAE tiling/chunking so the large descriptors become GEMM-eligible
  - test a newer ROCm/PyTorch/MIOpen build
  - add a warmed-DB benchmark case with logging reduced, then rocprof it

## Residual large 3D conv descriptor handling

A direct PyTorch `conv3d` probe was added so the two residual descriptors can be tested without running full Policy inference:

- Script: `scripts/probe_miopen_large_conv3d_descriptors.py`
- Runner cases:
  - `large_conv3d_probe`
  - `large_conv3d_probe_512_bf16`

The probe exercises:

- full descriptor execution
- BF16 vs FP16
- output-space tiling with halo-correct input crops
- MIOpen solver logging for every generated tile descriptor

The first descriptor was reproduced directly:

| Descriptor | Mode | Dtype | Final solver | Workspace | Time |
|---|---|---|---|---:|---:|
| `160-18-274-370-3x3x3-160-16-272-368` | full cold | BF16 | `ConvDirectNaiveConvFwd` | 0 | 180.883 sec |
| `160-18-274-370-3x3x3-160-16-272-368` | full warmed in same process | BF16 | `ConvDirectNaiveConvFwd` | 0 | 19.985 sec |
| `160-18-138-370-3x3x3-160-16-136-368` | tile `2x1` | BF16 | `GemmFwdRest` | 6918635520 | 80.835 sec |
| `160-18-138-186-3x3x3-160-16-136-184` | tile `2x2` | BF16 | `GemmFwdRest` | 3459317760 | 40.251 sec |

FP16 was also checked for the full first descriptor. It still used workspace 0 and `ConvDirectNaiveConvFwd`, so changing only BF16 to FP16 does not make this full descriptor solver-eligible.

The second descriptor was tested with full execution skipped after confirming the full shape enters workspace 0 / `ConvDirectNaiveConvFwd`:

| Descriptor | Mode | Dtype | Final solver | Workspace | Time |
|---|---|---|---|---:|---:|
| `512-6-242-322-3x3x3-256-4-240-320` | full | BF16 | `ConvDirectNaiveConvFwd` | 0 | stopped after solver confirmation |
| `512-6-122-322-3x3x3-256-4-120-320` | tile `2x1` | BF16 | `GemmFwdRest` | 4246732800 | 90.176 sec |
| `512-6-122-162-3x3x3-256-4-120-160` | tile `2x2` | BF16 | `GemmFwdRest` | 2123366400 | 41.895 sec |

Interpretation:

- Spatial tiling successfully changes the MIOpen solver from `ConvDirectNaiveConvFwd` to `GemmFwdRest`.
- The tiled versions are still slower than the warmed full naive estimate for the first descriptor and far slower than a viable optimization target.
- FP16 does not fix solver eligibility for the full first descriptor.
- Therefore the practical response is not to add naive spatial tiling to Policy. The next useful work is kernel/library support for these full descriptors, or model/framework decode changes that avoid generating them.

Actionable next options:

1. Open a ROCm/MIOpen issue or internal request with the two exact `MIOpenDriver convbfp16` commands and logs.
2. Test a newer ROCm/PyTorch/MIOpen stack for the same direct probe before re-running full Policy.
3. Test more aggressive model-level decode chunking only if it avoids these descriptors without multiplying total work by 2-4x.
4. If correctness allows it, test layout changes at the framework level; dtype alone is not sufficient.
5. If video output is optional, continue using action-only Policy as the effective workaround.

Immediate next probes:

1. Capture MIOpen logs for Policy VAE decode with:
   - `MIOPEN_ENABLE_LOGGING=1`
   - `MIOPEN_ENABLE_LOGGING_CMD=1`
   - `MIOPEN_LOG_LEVEL=5`
2. Re-run Policy with the populated user FindDb and reduced logging to separate warmed solver dispatch from cold Find overhead.
3. Test Policy VAE decode dtype/layout changes for the two residual naive descriptors.
4. Test VAE tiling/chunking for the two residual naive descriptors.
5. Test transformer-only TunableOp with selected-region rocprof to see whether GEMM kernels change after tuning.
