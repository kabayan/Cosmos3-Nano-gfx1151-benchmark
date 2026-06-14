# Cosmos3 ROCm v2.3 Policy Model speedup plan

Date: 2026-06-04

## v2.3 definition

v2.3 is the current article benchmark state after Policy warmup-excluded measurement:

| Mode | Article | v2.3 ROCm | Ratio | Article-equivalent |
|---|---:|---:|---:|---|
| T2I | `22 sec` | `27.136 sec` | `1.23x slower` | yes |
| T2V | `22 sec` | `32.165 sec` | `1.46x slower` | yes |
| I2V | `17 sec` | `25.045 sec` | `1.47x slower` | yes |
| Policy video + action | `21 sec` | `303.253 sec` | `14.44x slower` | yes |

Policy v2.3 artifacts:

- `result/policy_warm_excluded/video_action_warm1/benchmark.json`
- `result/policy_warm_excluded/video_action_warm1/action_policy_robot/vision.mp4`
- `result/policy_warm_excluded/video_action_warm1/action_policy_robot/sample_outputs.json`
- `docs/cosmos3-rocm-policy-warmup-excluded-deep-dive.md`

Policy output validation:

- video output SHA256 matches the previous full Policy output
- video: `640x480`, `17` frames, `5 fps`, `3.4 sec`
- action: `16x10`
- no quality reduction

## Corrected Policy bottleneck model

The old single-run model was:

- `generate_batch`: about `1905 sec`
- `generate_samples_from_batch`: about `781 sec`
- `decode`: about `1052 sec`

The warmup-excluded v2.3 measurement is:

| Stage | Time |
|---|---:|
| `OmniInference.generate_batch` | `303.253 sec` |
| `OmniMoTModel.generate_samples_from_batch` | `134.997 sec` |
| framework-reported `OmniMoTModel.decode` | `0.021 sec` |
| timer gap | `168.235 sec` |

Important correction:

- framework `TrainingTimer` uses CPU timing with `measure_cuda=False`.
- GPU decode work can be launched inside the `decode` timer and then synchronize later at `sample.cpu().numpy()` in `save_img_or_video`.
- Therefore the `168.235 sec` timer gap is not necessarily pure CPU video encoding. It may include deferred GPU decode synchronization, tensor transfer, array conversion, MP4 encoding, action JSON conversion, or guardrail no-op overhead.

The next optimization must first reclassify that `168 sec` gap with synchronized timers.

## Goal

Target the article-equivalent Policy video + action path without changing:

- model
- input sample
- output video
- action output
- image size
- frame count
- step count
- dtype/precision unless output is proven identical or explicitly accepted as numerically equivalent

Near-term success target:

- reduce Policy video + action from `303.253 sec` to below `200 sec`
- preserve MP4 SHA256 if possible; if encoder changes alter bytes, validate frame-level pixel equality/tolerance and identical `16x10` action output

Longer target:

- approach `21 sec` article time
- this likely requires kernel/backend changes beyond local Python timers

## Phase 0: measurement repair

Purpose:

- avoid optimizing a misleading timer bucket
- split `303 sec` into synchronized GPU decode, CPU transfer, NumPy conversion, MP4 encode, action serialization, and JSON write

Implementation:

1. Add Policy-specific synchronized stage timers in `scripts/run_cosmos_framework_policy_rocm.py` by monkeypatching:
   - `OmniInference.generate_batch`
   - `OmniMoTModel.generate_samples_from_batch`
   - `OmniMoTModel.decode`
   - `save_img_or_video`

2. For CUDA/ROCm stage timing:
   - call `torch.cuda.synchronize()` before timer start
   - call `torch.cuda.synchronize()` after timer end
   - record wall time to a new JSON file, for example `policy_stage_sync_profile.json`

3. Split save path:
   - `vision_guardrail`
   - `vision_cpu_transfer`
   - `vision_numpy_convert`
   - `mp4_encode`
   - `action_cpu_transfer`
   - `action_json_convert`
   - `sample_outputs_json_write`

4. Re-run:
   - warmup `1`
   - measured `1`
   - article-equivalent video + action
   - action-only

Expected output:

- exact synchronized stage table
- confirmation whether the `168 sec` timer gap is deferred GPU decode or CPU save/encode

Decision rule:

- If synchronized decode is still large, continue MIOpen/Wan VAE decode work.
- If MP4 encode/CPU transfer is large, optimize save path first.
- If `generate_samples_from_batch` dominates after sync, split sampler and preparation.

## Phase 1: save/postprocess path optimization

This phase is only valid if Phase 0 shows the timer gap is save/postprocess heavy.

Candidate optimizations:

1. Avoid unnecessary float conversion in `save_img_or_video`
   - current path does `sample.cpu().float().numpy() * 255`
   - for video output, convert to `uint8` once on GPU before CPU transfer:
     - `sample.mul(255).clamp(0,255).to(torch.uint8).permute(1,2,3,0).contiguous().cpu().numpy()`
   - expected benefit: less CPU memory bandwidth and conversion work

2. Use explicit synchronized transfer timing
   - isolate `.cpu()` from NumPy and encoder time
   - if transfer is large, test pinned-memory/non-blocking copy only if the output remains identical

3. Test encoder settings without changing visible output
   - current path uses `easy_io.dump(... format="mp4")`
   - compare:
     - current encoder
     - direct `imageio.mimsave`
     - ffmpeg with same resolution/fps/pixelformat
   - validate:
     - frame count/duration
     - action JSON shape
     - frame-level diff

Risk:

- MP4 byte SHA256 may change even when frames are identical because encoder metadata can differ.
- Use frame-level validation if encoder implementation changes.

## Phase 2: `generate_samples_from_batch` split and optimization

This phase targets the measured `134.997 sec`.

Known signal:

- console sampling progress is about `52 sec` for `30` steps
- therefore about `80 sec` is outside the visible denoise progress but inside `generate_samples_from_batch`

Add timers around:

1. `_maybe_apply_prompt_upsampling`
2. `_prepare_inference_data`
3. sampler call
4. `_get_velocity` aggregate
5. `_normalize_video_databatch_inplace`
6. `_normalize_action_databatch`
7. sequence plan / token packing

Likely candidates:

- fixed input/action preprocessing cache
- avoid repeated video/action normalization after warmup
- cache sequence plans for the same `action_policy_robot` sample
- avoid repeated CPU-to-GPU data preparation if the input sample is unchanged
- evaluate TunableOp/AOTriton for Policy transformer kernels separately from T2I/I2V

Validation:

- action output shape `16x10`
- action values identical or within strict tolerance
- MP4 frame-level equality if preprocessing is changed

## Phase 3: selected-region rocprof for warmup-excluded measured run

Purpose:

- profile only measured run, not warmup
- avoid old cold-run rocprof where VAE decode dominated

Implementation:

1. Add optional rocTX ranges around:
   - measured `generate_samples_from_batch`
   - measured sampler
   - measured decode
   - measured save/video encode

2. Run `rocprofv3 --selected-regions`.

3. Produce a post-v2.3 kernel breakdown:
   - GEMM
   - attention
   - elementwise/reduce/copy
   - naive conv
   - copy/transfer kernels
   - CPU-visible encode time from synchronized timers

Expected:

- cold/warmup naive conv should shrink or move into save synchronization if decode is async.
- if naive conv still appears during measured save, VAE decode remains a hidden GPU bottleneck.

## Phase 4: cold-start mitigation

This is operational, not article warmup-excluded speed.

Problem:

- first Policy full video+action run is about `1835 sec`
- warmup decode alone is about `1052 sec`

Mitigations:

1. Explicit service startup warmup:
   - run one full Policy warmup before accepting requests
   - keep process resident

2. Smaller synthetic VAE decode warmup:
   - find the minimal latent shape that triggers the same MIOpen solver/kernel initialization
   - avoid full `action_policy_robot` warmup if possible

3. Persist MIOpen user DB:
   - already tested as not sufficient for full speed, but still useful for reducing find overhead

4. Keep action-only warm path if video is not needed:
   - warmup-excluded action-only is `135.830 sec`
   - not article-equivalent

## Priority

1. P0: synchronized timer instrumentation and re-run
2. P1: split and optimize save/postprocess if timer gap is CPU/transfer/encode
3. P2: split `generate_samples_from_batch` and cache fixed input/action preprocessing
4. P3: selected-region rocprof on measured-only Policy
5. P4: cold-start warmup strategy for operational use

## Next concrete command target

After adding synchronized timers:

```bash
python /workspace/scripts/run_cosmos_framework_policy_rocm.py \
  --out-dir /workspace/result/policy_v2_3_speedup/sync_profile_video_action_warm1 \
  --warmup-runs 1 \
  --policy-sync-profile
```

Expected result files:

- `benchmark.json`
- `policy_stage_sync_profile.json`
- `action_policy_robot/vision.mp4`
- `action_policy_robot/sample_outputs.json`

