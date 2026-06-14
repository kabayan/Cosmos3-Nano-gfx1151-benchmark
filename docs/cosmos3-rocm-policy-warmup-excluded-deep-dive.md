# Cosmos3 Policy Model warmup-excluded deep dive

Date: 2026-06-04

## Scope

Policy Model was re-tested with warmup excluded. The goal was to separate:

- first-run framework/model/kernel warmup
- steady-state measured generation
- article-equivalent video + action output
- action-only output

Article-equivalent condition:

- input: `inputs/omni/action_policy_robot.json`
- task: Bridge / LeRobot robot policy sample
- output video: `640x480`, `17` frames, `5 fps`, `3.4 sec`
- action output: `16x10`
- diffusion steps: `30`
- quality reduction: none

## Runner change

`scripts/run_cosmos_framework_policy_rocm.py` now supports:

```bash
--warmup-runs N
```

This is passed to the framework CLI as:

```bash
--warmup N
```

The framework already records warmup timers separately:

- `[warmup] OmniInference.generate_batch`
- `[warmup] OmniMoTModel.generate_samples_from_batch`
- `[warmup] OmniMoTModel.decode`
- measured `OmniInference.generate_batch`
- measured `OmniMoTModel.generate_samples_from_batch`
- measured `OmniMoTModel.decode`

### Warmup data mutation fix

The first attempt failed after warmup with:

```text
AssertionError: Video data is not in uint8 format.
```

Cause:

- framework warmup reuses the same `data_batch`
- Policy video normalization mutates nested data structures in-place
- the second measured run then receives already-normalized tensors while the top-level batch no longer carries the matching `is_preprocessed` flag

Fix in the runner:

- clone `data_batch` recursively only for `warmup=True`
- preserve the original batch for the measured run

## Results

### Article-equivalent video + action

Artifact:

- `result/policy_warm_excluded/video_action_warm1/benchmark.json`
- `result/policy_warm_excluded/video_action_warm1/action_policy_robot/vision.mp4`
- `result/policy_warm_excluded/video_action_warm1/action_policy_robot/sample_outputs.json`

| Stage | Warmup | Measured |
|---|---:|---:|
| `OmniInference.generate_batch` | `1835.093 sec` | **`303.253 sec`** |
| `OmniMoTModel.generate_samples_from_batch` | `782.737 sec` | `134.997 sec` |
| `OmniMoTModel.decode` | `1052.355 sec` | `0.021 sec` |
| unprofiled save/postprocess/overhead | about `0 sec` | about `168.235 sec` |

The article Policy time is `21 sec`.

- previous single-run value: `1904.524 sec`, `90.7x slower`
- warmup-excluded value: **`303.253 sec`, `14.4x slower`**
- improvement vs previous single-run value: `6.28x faster`

### Output validation

The warmup-excluded video output matches the previous article-equivalent output:

```text
05cb83e594b65b34675e2ffbcec3d7807790d12d833d4d82a891c01688356cb4  result/policy_warm_excluded/video_action_warm1/action_policy_robot/vision.mp4
05cb83e594b65b34675e2ffbcec3d7807790d12d833d4d82a891c01688356cb4  result/rocm_speed_matrix/aotriton/policy_article/action_policy_robot/vision.mp4
```

Video properties:

| File | Width | Height | FPS | Duration | Frames |
|---|---:|---:|---:|---:|---:|
| warmup-excluded | `640` | `480` | `5` | `3.4 sec` | `17` |
| previous baseline | `640` | `480` | `5` | `3.4 sec` | `17` |

Action output:

- shape: `16x10`
- status: `success`

Conclusion:

- The `303.253 sec` measured run is article-equivalent.
- The output MP4 is byte-identical to the previous full Policy output.
- No quality-reducing shortcut was used.

### Action-only

Artifact:

- `result/policy_warm_excluded/action_only_warm1_clone/benchmark.json`
- `result/policy_warm_excluded/action_only_warm1_clone/action_policy_robot/sample_outputs.json`

| Stage | Warmup | Measured |
|---|---:|---:|
| `OmniInference.generate_batch` | `776.514 sec` | **`135.830 sec`** |
| `OmniMoTModel.generate_samples_from_batch` | `776.513 sec` | `135.708 sec` |
| `OmniMoTModel.decode` | `0.000124 sec` | `0.000064 sec` |

Interpretation:

- The previous action-only value around `770 sec` was essentially first-run warmup.
- Warmup-excluded action-only is `135.830 sec`.
- This is not article-equivalent because it skips real video output.

## Updated bottleneck model

Before this test, Policy was described as "decode dominated" because the single-run timer showed:

- `generate_batch`: about `1905 sec`
- `generate_samples_from_batch`: about `781 sec`
- `decode`: about `1052 sec`

That statement is true for cold/single-run behavior, but it is incomplete for steady-state warmup-excluded behavior.

Warmup-excluded article-equivalent Policy is:

1. `generate_samples_from_batch`: `134.997 sec`
2. output save/postprocess/unprofiled overhead: about `168.235 sec`
3. framework-reported `decode`: `0.021 sec`

Important correction:

- The very large `1052 sec` VAE decode cost is first-run/warmup behavior in this runner.
- Once the full video decode path has been warmed, the measured framework `decode` timer is no longer the dominant stage.
- The new steady-state target is the `303 sec` measured full path, not the old `1905 sec` single-run path.

## Remaining gap vs article

Article:

- `21 sec`

Current warmup-excluded ROCm:

- `303.253 sec`
- `14.4x slower`

Breakdown of the remaining measured time:

- sampler/generation: `134.997 sec`
- save/postprocess/unprofiled: `168.235 sec`
- decode timer: `0.021 sec`

Therefore, the next Policy improvements should target:

1. measured `generate_samples_from_batch`
   - the denoising loop itself is only about `52 sec` from console sampling progress
   - the rest is generation preparation / condition handling / action-video packing inside the broad stage
2. video save/postprocess path
   - measured full path has about `168 sec` not covered by current framework timers
   - this includes output formatting, CPU/GPU synchronization, and MP4 encoding path
3. first-run decode warmup only if cold-start latency matters
   - still important operationally
   - not part of warmup-excluded article comparison

## Artifacts

- runner: `scripts/run_cosmos_framework_policy_rocm.py`
- failed first warmup attempt: `result/policy_warm_excluded/action_only_warm1/`
- fixed action-only warmup-excluded: `result/policy_warm_excluded/action_only_warm1_clone/`
- fixed video+action warmup-excluded: `result/policy_warm_excluded/video_action_warm1/`

