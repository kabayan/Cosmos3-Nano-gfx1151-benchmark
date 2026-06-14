# Cosmos3 ROCm v2.3 Policy deep profile results

Date: 2026-06-05

## Summary

Policy condition cache 後の残ボトルネックである VAE decode と `_get_velocity` を deep profile した。

Run:

- `result/policy_v2_3_speedup/deep_profile_condition_cache_video_action_warm1_r2`

The run completed successfully and preserved output:

- MP4 SHA256 matches v2.3 condition-cache baseline:
  - `05cb83e594b65b34675e2ffbcec3d7807790d12d833d4d82a891c01688356cb4`
- action/output JSON numeric max abs diff vs baseline: `0.0`
- MP4 metadata: `640x480`, `17 frames`, `5 fps`, `3.4 sec`

## Measured result

| Stage | Time |
|---|---:|
| `generate_batch_sync` | 147.971 sec |
| `decode_sync` | 96.054 sec |
| `generate_samples_from_batch_sync` | 51.648 sec |
| `get_velocity_sync` | 51.612 sec |
| `save_img_or_video_total` | 0.263 sec |

This is consistent with the previous v2.3 condition-cache run:

- previous full measured: `147.757 sec`
- deep-profile full measured: `147.971 sec`

The deep profile did not materially perturb the measured result.

## VAE decode breakdown

Measured phase:

| Stage | Time |
|---|---:|
| `decode_sync` | 96.054 sec |
| `vae_wan_decode_total_sync` | 96.054 sec |
| `vae_decoder3d_forward_sync` | 96.052 sec |

Warmup phase:

| Stage | Time |
|---|---:|
| `decode_sync` | 1113.347 sec |
| `vae_wan_decode_total_sync` | 1113.347 sec |
| `vae_decoder3d_forward_sync` | 1113.335 sec |

Interpretation:

- Almost all VAE decode time is inside `Decoder3d.forward`.
- Wrapper overhead, denormalize, `conv2`, and `unpatchify` are not visible as meaningful bottlenecks at this granularity.
- The next VAE work should profile or alter `Decoder3d.forward` internals:
  - `CausalConv3d`
  - `ResidualBlock`
  - `Up_ResidualBlock`
  - `Resample`
  - `AttentionBlock`
- Because `Decoder3d.forward` is called once per latent temporal chunk, temporal chunking / one-shot decode / Conv3d solver selection remain the most relevant hypotheses.

## `_get_velocity` breakdown

Measured phase:

| Stage | Time |
|---|---:|
| `get_velocity_sync` | 51.612 sec |
| `velocity_denoise_sync` | 51.564 sec |
| `velocity_network_forward_sync` | 51.562 sec |
| `velocity_network_encode_vision_sync` | 1.639 sec |
| `velocity_network_encode_action_sync` | 0.089 sec |
| `velocity_pack_input_sequence_sync` | 0.029 sec |
| `velocity_build_packed_sequence_sync` | 0.012 sec |
| `velocity_network_decode_vision_sync` | 0.010 sec |
| `velocity_network_decode_action_sync` | 0.008 sec |
| `velocity_packed_sequence_to_cuda_sync` | 0.005 sec |
| `velocity_network_encode_text_sync` | 0.003 sec |

Call counts:

| Stage | Count |
|---|---:|
| `get_velocity_sync` | 60 |
| `velocity_network_forward_sync` | 60 |
| `velocity_network_encode_vision_sync` | 60 |
| `velocity_network_encode_action_sync` | 60 |
| `velocity_build_packed_sequence_sync` | 60 |
| `velocity_pack_input_sequence_sync` | 62 |

There are 60 velocity calls because the run includes 30 warmup steps and 30 measured steps. Policy uses `guidance=1.0`, so this is not CFG double-forward.

Interpretation:

- `_get_velocity` is almost entirely `Cosmos3VFMNetwork.forward`.
- Repeated Python-side packing is not a meaningful bottleneck:
  - `_pack_input_sequence`: `0.029 sec` total measured
  - `PackedSequence.to_cuda`: `0.005 sec` total measured
  - `build_packed_sequence`: `0.012 sec` total measured
- Static packed metadata cache is therefore not worth prioritizing for this Policy case.
- `_encode_vision` is the only non-trivial substage outside the transformer body, but it is still only `1.639 sec` total measured.
- The real `_get_velocity` improvement target is the model forward internals:
  - Qwen/Cosmos transformer layers
  - attention backend
  - GEMM selection
  - RMSNorm/RoPE/MLP fusion
  - scatter/gather/indexing inside token encode only as a secondary target

## Updated hypotheses

### Confirmed

1. VAE decode is dominated by `Decoder3d.forward`.
2. `_get_velocity` is dominated by `Cosmos3VFMNetwork.forward`.
3. Pack/build/to_cuda metadata work is negligible for Policy.
4. Save/postprocess is negligible.
5. Condition cache remains valid and output-identical.

### Rejected or deprioritized

1. Static packed metadata cache is unlikely to materially speed up Policy.
2. Text embedding cache is unlikely to matter.
3. Save/encode optimization is irrelevant for this bottleneck.

## Next implementation targets

### P1: VAE decoder internal profile

Add lower-level timing inside `Decoder3d.forward`:

- `decoder.conv1`
- each `decoder.middle` module
- each `decoder.upsamples` block
- `decoder.head`

Use aggregate timing, not per-Conv3d timing first, to avoid excessive synchronization overhead.

Decision:

- If one `Up_ResidualBlock` dominates, inspect its `CausalConv3d` and `Resample` descriptors.
- If all blocks are proportionally slow, test temporal micro-batching or one-shot decode.

### P2: VAE decode shape experiment

Implement probes:

- current streaming `T=1`
- temporal micro-batch `T=2`
- temporal micro-batch `T=full` / non-streaming

Validation:

- decoded tensor max abs diff
- MP4 SHA256
- full Policy output equality

Promotion rule:

- Only exact output match is acceptable for article comparison.

### P3: `_get_velocity` model-forward rocprof

Run measured-only rocprof around `Cosmos3VFMNetwork.forward` / `language_model`.

Focus:

- hipBLASLt/rocBLAS GEMM kernels
- AOTriton SDPA kernels
- RMSNorm/RoPE kernels
- MLP activation kernels
- scatter/gather/index kernels

The previous cache/metadata hypothesis is now weak; kernel/backend optimization is the correct path.

### P4: `_encode_vision` secondary optimization

`_encode_vision` is `1.639 sec` total measured. This is not dominant, but it is the only visible non-transformer substage inside `_get_velocity`.

Candidate:

- cache clean/conditioned vision projection and recompute only noisy token projection plus timestep embedding.

Expected benefit:

- At most about `1-2 sec`, so this should come after VAE and transformer kernel work.

## Practical next step

Do P1 and P3 next:

1. Implement aggregate `Decoder3d.forward` sub-block profile.
2. Implement measured-only rocprof wrapper for `velocity_network_forward_sync`.
3. Use those two results to decide between:
   - VAE temporal batching/code rewrite
   - MIOpen Conv3d descriptor work
   - transformer attention/GEMM/kernel backend work
