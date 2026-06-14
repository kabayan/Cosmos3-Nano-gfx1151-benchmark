# Cosmos3 ROCm v2.3 Policy decoder block and velocity rocprof results

Date: 2026-06-05

## Summary

Policy condition-cache run に対して、次の 2 つを実施した。

1. `Decoder3d.forward` 内部の block profile
2. `Cosmos3VFMNetwork.forward` measured region の `rocprofv3` kernel trace

Run:

- `result/policy_v2_3_speedup/block_profile_roctx_velocity_warm1_r2`

Output validation:

- MP4 SHA256 matches v2.3 condition-cache baseline:
  - `05cb83e594b65b34675e2ffbcec3d7807790d12d833d4d82a891c01688356cb4`
- action/output JSON numeric max abs diff: `0.0`

## Implemented instrumentation

Script:

- `scripts/run_cosmos_framework_policy_rocm.py`

Added options:

- `--policy-decoder-block-profile`
- `--policy-roctx-network-forward`

`--policy-decoder-block-profile` replaces the `Decoder3d.forward` hook with block-level timing:

- `vae_decoder3d_conv1_sync`
- `vae_decoder3d_middle_{i}_sync`
- `vae_decoder3d_upsample_{i}_sync`
- `vae_decoder3d_head_{i}_sync`

`--policy-roctx-network-forward` wraps measured-phase `Cosmos3VFMNetwork.forward` with ROCTx:

- range name: `policy_velocity_network_forward`
- `rocprofv3 --selected-regions`
- `--kernel-trace`
- `--marker-trace`

The first attempt called `roctxProfilerPause(0)` during hook initialization and failed because the rocprof context was not ready. The fix was to remove the initial pause and only call resume/pause around measured network forward.

## Full measured time

| Stage | Time |
|---|---:|
| `generate_batch_sync` | 149.609 sec |
| `decode_sync` | 97.128 sec |
| `generate_samples_from_batch_sync` | 52.222 sec |
| `get_velocity_sync` | 52.185 sec |
| `velocity_network_forward_sync` | 52.132 sec |

The result is close to the previous v2.3 condition-cache baseline:

- previous: `147.757 sec`
- this profiled run: `149.609 sec`

The overhead is acceptable for attribution.

## Decoder3d block profile

Measured decode:

| Decoder block | Time |
|---|---:|
| `vae_decoder3d_forward_sync` | 97.126 sec |
| `vae_decoder3d_upsample_3_sync` | 92.527 sec |
| `vae_decoder3d_upsample_2_sync` | 2.719 sec |
| `vae_decoder3d_upsample_1_sync` | 1.145 sec |
| `vae_decoder3d_head_2_sync` | 0.434 sec |
| `vae_decoder3d_upsample_0_sync` | 0.167 sec |
| `vae_decoder3d_middle_2_sync` | 0.041 sec |
| `vae_decoder3d_middle_0_sync` | 0.040 sec |
| `vae_decoder3d_head_0_sync` | 0.028 sec |
| `vae_decoder3d_middle_1_sync` | 0.017 sec |
| `vae_decoder3d_head_1_sync` | 0.006 sec |
| `vae_decoder3d_conv1_sync` | 0.002 sec |

Warmup decode:

| Decoder block | Time |
|---|---:|
| `vae_decoder3d_forward_sync` | 1118.436 sec |
| `vae_decoder3d_upsample_2_sync` | 452.327 sec |
| `vae_decoder3d_upsample_3_sync` | 424.794 sec |
| `vae_decoder3d_upsample_1_sync` | 198.010 sec |
| `vae_decoder3d_upsample_0_sync` | 20.639 sec |
| `vae_decoder3d_head_2_sync` | 16.343 sec |
| `vae_decoder3d_middle_0_sync` | 5.136 sec |

Interpretation:

- Measured VAE decode is overwhelmingly dominated by `upsample_3`: `92.527 / 97.126 sec`.
- This points to the final/high-resolution upsampling block, not the whole decoder evenly.
- Next VAE investigation should target `Up_ResidualBlock` index 3 internals:
  - residual block convs
  - shortcut path
  - final upsample/resample path
  - any `CausalConv3d` descriptors inside that block
- The earlier hypothesis that VAE decode is broadly `Decoder3d.forward` was correct, but now the actionable target is much narrower: `decoder.upsamples[3]`.

## Velocity rocprof

rocprof outputs:

- `policy_velocity_kernel_trace.csv` about 20 MB
- `policy_velocity_results.json` about 101 MB
- `policy_velocity_marker_api_trace.csv`

The CSV kernel names were region-renamed by `--kernel-rename`, but the original kernel symbols were recovered from `policy_velocity_results.json` via `Kernel_Id`.

Kernel trace summary:

| Metric | Value |
|---|---:|
| kernel dispatches | 141,270 |
| total kernel time | 51.428 sec |
| measured network forward calls | 30 |

Kernel category breakdown:

| Category | Time | Dispatches | Share |
|---|---:|---:|---:|
| GEMM / BLAS / Tensile | 43.044 sec | 15,450 | 83.7% |
| elementwise | 3.419 sec | 73,710 | 6.6% |
| attention | 2.993 sec | 2,160 | 5.8% |
| copy / cat | 1.440 sec | 36,180 | 2.8% |
| reduce | 0.422 sec | 8,700 | 0.8% |
| index / scatter / gather | 0.110 sec | 5,040 | 0.2% |

Top kernels:

| Time | Dispatches | Kernel |
|---:|---:|---|
| 34.156 sec | 7,560 | `Cijk_Alik_Bljk_BBS_BH_MT128x128x32...WGM8.kd` |
| 6.687 sec | 5,400 | `Cijk_Alik_Bljk_BBS_BH_MT128x128x32...SIA3...WGM8.kd` |
| 3.216 sec | 60,900 | `vectorized_elementwise_kernel` |
| 2.993 sec | 2,160 | `attn_fwd.kd` |
| 1.639 sec | 120 | `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT8x8x8...` |
| 1.283 sec | 35,340 | `elementwise_kernel_manual_unroll` |
| 0.547 sec | 2,160 | `Cijk_Alik_Bljk_BBS_BH_MT128x128x32...WGM1.kd` |
| 0.422 sec | 8,700 | `reduce_kernel` |

Interpretation:

- `_get_velocity` is GEMM-dominated.
- Attention is not the main bottleneck in Policy velocity: `2.993 sec / 51.428 sec`.
- Elementwise/fusion is secondary: `3.419 sec`.
- Kernel launch count is high, but total time is mostly in Tensile GEMM kernels.
- The next transformer-side improvement should focus on GEMM selection/Tensile/hipBLASLt rather than attention backend first.

## Updated improvement priorities

### VAE decode

Highest priority:

1. Profile `decoder.upsamples[3]` internally.
2. Identify the exact conv/resample op in that block.
3. Capture Conv2d/Conv3d descriptors and selected MIOpen kernels.
4. Test equivalent shape-preserving rewrites for the final upsample block.

Likely useful probes:

- `upsample_3` submodule timing:
  - shortcut `DupUp3D`
  - each residual block
  - final resample
- MIOpen logging only during decode measured region
- temporal chunking may not help if `upsample_3` is spatial/high-resolution dominated

### `_get_velocity`

Highest priority:

1. GEMM kernel selection deep dive for the two dominant `Cijk_Alik_Bljk_BBS_BH_MT128x128x32...WGM8` variants.
2. Compare TunableOp table impact specifically for Policy network forward.
3. Test hipBLASLt / rocBLAS / Tensile settings that alter these kernels.
4. Only after GEMM: inspect elementwise fusion and attention backend.

Deprioritized:

- packed metadata cache
- text embedding cache
- attention-first optimization

## Next concrete step

Run two focused probes:

1. `decoder.upsamples[3]` internal profile.
2. Policy velocity GEMM kernel selection probe comparing current env vs TunableOp/hipBLASLt knobs, using the same ROCTx selected-region mechanism.
