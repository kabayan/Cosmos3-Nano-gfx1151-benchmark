# Cosmos3 ROCm v2.3 Policy condition cache results

Date: 2026-06-05

## Summary

Policy Model の warmup-excluded video + action 出力に対して、conditioning latent の再計算を避ける condition cache prototype を実装し、記事同等の出力を維持したまま measured time を短縮できることを確認した。

| Case | Baseline | v2.3 condition cache | Improvement | Article time | Article ratio |
|---|---:|---:|---:|---:|---:|
| Policy video + action | 232.423 sec sync no-cache | 147.757 sec | 1.57x faster | 21 sec | 7.0x slower |
| Policy action-only | 136.092 sec sync deep no-cache | 52.162 sec | 2.61x faster | N/A | N/A |
| Previous v2.3 CPU-timer baseline | 303.253 sec | 147.757 sec | 2.05x faster | 21 sec | 7.0x slower |

The video + action output is still article-equivalent: `640x480`, `17 frames`, `5 fps`, `3.4 sec`, plus `16x10` action sequence.

## Implemented changes

Target script:

- `scripts/run_cosmos_framework_policy_rocm.py`

Changes:

- Added `--policy-sync-profile` for synchronized stage timing.
- Added `--policy-condition-cache` prototype.
- Instrumented `OmniMoTModel.generate_samples_from_batch`, `decode`, `_prepare_inference_data`, `get_data_and_condition`, `_normalize_video_databatch_inplace`, `_normalize_action_databatch`, `_maybe_apply_prompt_upsampling`, and `_get_velocity`.
- Split save/postprocess timing into clamp/cast, CPU numpy conversion, and video encode.
- Cached the deterministic conditioning result produced by `get_data_and_condition` during warmup and reused it in the measured run.

This prototype keeps the measured seed, denoise path, VAE decode path, output resolution, frames, and action output unchanged. It only avoids recomputing the same conditioning latent for the identical input video/action sample.

## Measurements

### Video + action, no cache

Run:

- `result/policy_v2_3_speedup/sync_profile_video_action_warm1`

Measured profile:

| Stage | Time |
|---|---:|
| `generate_batch_sync` | 232.423 sec |
| `generate_samples_from_batch_sync` | 134.919 sec |
| `decode_sync` | 97.255 sec |
| `save_img_or_video_total` | 0.233 sec |

### Video + action, condition cache

Run:

- `result/policy_v2_3_speedup/condition_cache_video_action_warm1`

Measured profile:

| Stage | Time |
|---|---:|
| `generate_batch_sync` | 147.757 sec |
| `decode_sync` | 96.080 sec |
| `generate_samples_from_batch_sync` | 51.465 sec |
| `_get_velocity` | 51.429 sec |
| `prepare_inference_data_sync` | 0.006 sec |
| `save_img_or_video_total` | 0.204 sec |

Warmup profile:

| Stage | Time |
|---|---:|
| `generate_batch_sync` | 1891.741 sec |
| `decode_sync` | 1118.125 sec |
| `generate_samples_from_batch_sync` | 773.615 sec |
| `get_data_and_condition_sync` | 720.957 sec |
| `_get_velocity` | 52.219 sec |

### Action-only, condition cache

Run:

- `result/policy_v2_3_speedup/condition_cache_action_only_warm1`

Measured profile:

| Stage | Time |
|---|---:|
| `generate_batch_sync` | 52.162 sec |
| `generate_samples_from_batch_sync` | 52.017 sec |
| `_get_velocity` | 51.982 sec |
| `prepare_inference_data_sync` | 0.006 sec |

## Output validation

Video output SHA256:

```text
05cb83e594b65b34675e2ffbcec3d7807790d12d833d4d82a891c01688356cb4
```

The SHA256 is identical across:

- `result/policy_v2_3_speedup/condition_cache_video_action_warm1/action_policy_robot/vision.mp4`
- `result/policy_v2_3_speedup/sync_profile_video_action_warm1/action_policy_robot/vision.mp4`
- `result/policy_warm_excluded/video_action_warm1/action_policy_robot/vision.mp4`
- `result/rocm_speed_matrix/aotriton/policy_article/action_policy_robot/vision.mp4`

Action/output JSON numeric comparison:

| Pair | Numeric values | Max abs diff |
|---|---:|---:|
| video + action no-cache vs cache | 187 | 0.0 |
| action-only no-cache vs cache | 189 | 0.0 |

MP4 metadata:

```text
width=640
height=480
r_frame_rate=5/1
duration=3.400000
nb_frames=17
```

## Interpretation

The original v2.3 `303.253 sec` number was useful as a warmup-excluded framework timer, but synchronized profiling showed the no-cache measured video + action run at `232.423 sec`. The gap was caused by asynchronous GPU work attribution around decode.

After condition cache, the measured video + action run is `147.757 sec`. The reduction comes from removing the repeated conditioning encode path:

- No-cache measured `generate_samples_from_batch`: `134.919 sec`
- Cache measured `generate_samples_from_batch`: `51.465 sec`
- Removed measured cost: about `83.454 sec`

The remaining bottlenecks are:

1. VAE decode: `96.080 sec`
2. Denoise / transformer velocity: `51.429 sec`
3. Save/postprocess: only `0.204 sec`, not a bottleneck

## Productionization requirements

The current implementation is a validated prototype. To make it production-safe, cache invalidation must include:

- input video path/content hash
- action path/content hash
- condition frame indexes
- resolution and image size
- model checkpoint and config identity
- dtype/device
- tokenizer/VAE config
- action domain and raw action dimension

The cache should be scoped per model instance and sample, and disabled when any conditioning input changes.

## Next improvement candidates

1. VAE decode ROCm/MIOpen path: measured `96.080 sec`, still dominant after cache.
2. `_get_velocity` transformer kernels: measured `51.429 sec`, now the second largest component.
3. Production condition cache with deterministic invalidation and multi-sample handling.
4. Measured-only rocprof on the cached video + action run, focusing separately on VAE decode and `_get_velocity`.
