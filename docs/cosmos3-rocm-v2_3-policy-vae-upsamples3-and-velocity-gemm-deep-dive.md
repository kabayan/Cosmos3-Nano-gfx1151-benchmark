# Cosmos3 ROCm v2.3 Policy VAE upsample[3] and Velocity GEMM Deep Dive

Date: 2026-06-05

## Scope

This note continues the v2.3 Policy Model bottleneck investigation.

- VAE: break down `Decoder3d.forward -> decoder.upsamples[3]`.
- Velocity: confirm that Policy network forward is GEMM/Tensile dominated, not attention dominated, and identify the dominant GEMM shapes.
- Quality-reducing changes remain out of scope. No step count, frame count, resolution, dtype, or modality reduction is used as an optimization.

## Runner Changes

Updated `scripts/run_cosmos_framework_policy_rocm.py`.

- Added `--policy-decoder-upsample-detail-index`.
- Added `--policy-roctx-vae-detail`.
- The first sync-based per-layer probe was intentionally stopped because per-op `torch.cuda.synchronize()` made VAE decode too slow for practical diagnosis.
- The final VAE probe uses ROCTX markers plus `rocprofv3 --selected-regions --kernel-trace --marker-trace --kernel-rename` so kernels are attributed to VAE ranges without forcing per-op synchronization.

## VAE Result

Run:

```text
result/policy_v2_3_speedup/vae_upsample3_roctx_rocprof_r2
```

Important files:

```text
rocprof/vae_upsample3_kernel_trace.csv
rocprof/vae_upsample3_marker_api_trace.csv
rocprof/vae_upsample3_results.json
```

The measured VAE selected-region kernel trace is consistent with the previous synchronized decode timing:

| Metric | Value |
|---|---:|
| VAE selected-region kernel total | 97.125 sec |
| Kernel dispatches | 2,788 |
| Previous sync `OmniMoTModel.decode` | 97.128 sec |

`benchmark.json` in this run reports `OmniMoTModel.decode = 0.267 sec`, but that value is not used because sync profiling was disabled. The authoritative VAE timing here is the rocprof kernel total.

### `decoder.upsamples[3]` Breakdown

`upsamples[3]` is not a spatial/temporal resample stage in this model configuration. It is the final high-resolution `Up_ResidualBlock`, made mostly of residual blocks.

Top VAE ranges from kernel trace:

| Range | Kernel Time | Dispatches | Share |
|---|---:|---:|---:|
| `vae_decoder3d_upsample_3_detail_residual_0_conv_2` | 89.540 sec | 34 | 92.19% |
| `policy_vae_wan_decode` outside detailed markers | 4.578 sec | 2,320 | 4.71% |
| `residual_2_conv_6` | 0.551 sec | 38 | 0.57% |
| `residual_2_conv_2` | 0.548 sec | 38 | 0.56% |
| `residual_1_conv_6` | 0.547 sec | 38 | 0.56% |
| `residual_1_conv_2` | 0.539 sec | 38 | 0.55% |
| `residual_0_conv_6` | 0.524 sec | 38 | 0.54% |

Dominant kernel id and symbol:

| Kernel ID | Symbol | Time |
|---:|---|---:|
| 2639 | `naive_conv_ab_nonpacked_fwd_ncdhw_ushort_double_ushort.kd` | 89.424 sec |

Interpretation:

- The VAE bottleneck is now narrowed from `decoder.upsamples[3]` to one specific conv: the first `CausalConv3d` in the first residual block of that final high-resolution block.
- This is still the same ROCm/MIOpen issue class as before: a large BF16 3D convolution descriptor falls to a naive nonpacked NCDHW forward kernel.
- The other high-resolution convs are below 0.6 sec each, so optimizing generic residual block overhead will not materially move the total unless this specific descriptor is handled.

## Velocity Result

Run:

```text
result/policy_v2_3_speedup/policy_velocity_gemm_log_action_only_warm1
```

This run uses `--action-only` to avoid VAE decode while keeping the same `_get_velocity` and Policy network forward path.

Measured sync profile:

| Stage | Time |
|---|---:|
| `generate_batch_sync` | 52.245 sec |
| `generate_samples_from_batch_sync` | 52.075 sec |
| `get_velocity_sync` | 52.039 sec |
| `velocity_denoise_sync` | 51.993 sec |
| `velocity_network_forward_sync` | 51.990 sec |
| `velocity_network_encode_vision_sync` | 1.716 sec |
| `velocity_network_encode_action_sync` | 0.127 sec |

This confirms that the Policy velocity bottleneck is still the network forward itself.

### GEMM Shapes

rocBLAS profile logs:

```text
gemm_logs/rocblas_profile.yaml
gemm_logs/rocblas_bench.log
gemm_logs/rocblas_trace.log
gemm_logs/hipblaslt_1.log
```

The dominant estimated-FLOP shapes across warmup plus measured are BF16/F32 compute TN GEMMs:

| GEMM | Calls | Estimated FLOPs | Notes |
|---|---:|---:|---|
| `TN M=12288 N=1516 K=4096` | 4,320 | 659.26 TF | largest MLP/projection shape |
| `TN M=4096 N=1516 K=12288` | 2,160 | 329.63 TF | matching expansion/down-projection pair |
| `TN M=4096 N=1516 K=4096` | 4,320 | 219.75 TF | repeated transformer projection |
| `TN M=1024 N=1516 K=4096` | 4,320 | 54.94 TF | smaller projection |
| `TN M=12288 N=111 K=4096` | 4,320 | 48.27 TF | shorter sequence branch |

This matches the previous rocprof result:

- GEMM/BLAS: 43.044 sec out of 51.428 sec kernel time.
- Top Tensile kernel: 34.156 sec over 7,560 dispatches.
- Attention: 2.993 sec, secondary.

hipBLASLt heuristic behavior:

| Event | Count |
|---|---:|
| `rocblaslt_matmul_algo_get_heuristic returnAlgoCount=0` | 240 |
| `getAllSolutions Found hardware solutions` | 240 |
| `final returnAlgoCount=1` | 240 |
| direct `returnAlgoCount=1` | 480 |

Interpretation:

- For many matmul descriptors, hipBLASLt first has no direct heuristic result, then falls back to `getAllSolutions`, finds 4 hardware solutions, and chooses one.
- `rocblas_profile.yaml` reports `solution_index: 0` for these calls; this log does not expose enough to prove a better solution was available for the dominant shapes.
- The earlier TunableOp observation remains valid: speedup can happen by changing selected GEMM backend/kernel without changing model math. But this run shows that the current default still leaves the main Policy forward dominated by a small set of repeated large BF16 TN GEMMs.

## Current Improvement Hypotheses

### VAE

Correct hypothesis:

> Policy VAE decode is dominated by one high-resolution BF16 3D convolution descriptor that falls to MIOpen `naive_conv_ab_nonpacked_fwd_ncdhw_ushort_double_ushort`.

Required improvement:

- MIOpen/full descriptor support for this exact conv class, or
- framework/model decode change that avoids this descriptor without changing output quality.

Most promising next technical work:

- capture the exact tensor/conv descriptor for `upsamples[3].residual_0.conv_2`;
- reproduce it in a minimal MIOpen/PyTorch conv probe;
- test whether layout transform, chunking, or equivalent decomposition can avoid `naive_conv_ab_nonpacked_fwd_ncdhw_ushort_double_ushort` without changing output.

### Velocity

Correct hypothesis:

> Policy `_get_velocity` is dominated by transformer GEMM/Tensile kernel selection for repeated BF16 TN projection/MLP GEMMs; attention is not the primary bottleneck.

Required improvement:

- better hipBLASLt/rocBLAS/Tensile solution selection for the dominant TN shapes;
- persistent TunableOp coverage for the Policy-specific shapes;
- possibly newer PyTorch ROCm/hipBLASLt/Tensile stack with improved gfx1151 solutions.

Most promising next technical work:

- replay the top `rocblas-bench` shapes with candidate env variants:
  - default,
  - `TENSILE_SOLUTION_SELECTION_METHOD=2`,
  - persisted TunableOp,
  - newer ROCm/PyTorch image if available;
- compare per-shape time before running another full Policy benchmark.

## Caveats

- The VAE ROCTX run's MP4 hash differs from the v2.3 baseline and action JSON differs by `max_abs_diff = 0.022763`. The ROCTX markers do not intentionally change math, but this run is treated as a profiling run, not an output-equivalence proof.
- The earlier v2.3 condition-cache run remains the output-equivalence reference: `max_abs_diff = 0.0`, MP4 hash `05cb83e594b65b34675e2ffbcec3d7807790d12d833d4d82a891c01688356cb4`.
- GPU reached 98C during the GEMM logging run; avoid unnecessary immediate repeat full runs.

## Follow-up Equivalence Check

Date: 2026-06-06

Run:

```text
result/policy_v2_3_speedup/vae_upsample3_equivalence_sync_warm1
```

Command shape:

```text
--warmup-runs 1
--policy-sync-profile
--policy-condition-cache
--policy-deep-profile
--policy-decoder-block-profile
--policy-decoder-upsample-detail-index 3
--policy-roctx-vae-detail
```

This run keeps the VAE detail monkeypatch and ROCTX marker calls, but removes `rocprofv3` and restores synchronized profiling.

Equivalence result against `condition_cache_video_action_warm1`:

| Check | Result |
|---|---:|
| action value count | 160 vs 160 |
| action `max_abs_diff` | 0.0 |
| action `mean_abs_diff` | 0.0 |
| action nonzero diffs | 0 |
| MP4 SHA-256 | `05cb83e594b65b34675e2ffbcec3d7807790d12d833d4d82a891c01688356cb4` |

Measured timing:

| Stage | Baseline v2.3 | Equivalence run |
|---|---:|---:|
| `OmniInference.generate_batch` | 147.757 sec | 150.828 sec |
| `OmniMoTModel.generate_samples_from_batch` | 51.465 sec | 52.020 sec |
| `OmniMoTModel.decode` | 96.081 sec | 98.591 sec |

Conclusion:

- The VAE detail monkeypatch and ROCTX marker calls are output-equivalent when run without `rocprofv3` and with synchronized profiling.
- The earlier mismatch in `vae_upsample3_roctx_rocprof_r2` is therefore attributed to the profiling configuration, especially `rocprofv3 --selected-regions` plus disabled sync profiling, not to the model input, seed, or VAE detail monkeypatch itself.
