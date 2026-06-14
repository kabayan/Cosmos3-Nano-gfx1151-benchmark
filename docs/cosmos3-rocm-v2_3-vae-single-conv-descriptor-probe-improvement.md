# Cosmos3 ROCm v2.3 VAE Single Conv Descriptor Probe Improvement

Date: 2026-06-07

## Scope

Improved the single-conv probe for the Policy Model VAE bottleneck:

- Target: `vae_decoder3d_upsample_3_detail_residual_0_conv_2`
- Descriptor:
  - input: `[1, 512, 1, 240, 320]`
  - weight: `[256, 512, 3, 3, 3]`
  - dtype: `torch.bfloat16`
  - causal padding: `[1, 1, 1, 1, 2, 0]`
  - stride/dilation: `[1, 1, 1]`
  - groups: `1`

## Implemented Changes

### Probe Capture

Updated `scripts/run_cosmos_framework_policy_rocm.py`.

- Added capture of the actual VAE target conv payload:
  - input
  - weight
  - bias
  - causal padding
  - stride/dilation/groups
  - cache index
  - actual `feat_cache[idx]` tensor when present
- Fixed the experimental runner-side `channels_last_3d` path to preserve streaming decode cache semantics:
  - updates `feat_cache[idx]`
  - advances `feat_idx`
  - applies cached temporal prefix before causal padding

The previous runner-side target replacement was not valid for full decode because it ignored `feat_cache` and `feat_idx`.

### Standalone Probe

Updated `scripts/probe_wan_vae_conv_descriptor.py`.

- Supports cache-aware baseline and `channels_last_3d` execution.
- Separates:
  - cold reference baseline time, which includes first-use MIOpen/Tensile setup
  - warmed baseline time, which better represents steady-state kernel execution
- Added `--repeats`.
- Reports:
  - `seconds_all`
  - `seconds_median`
  - `speedup_vs_warm_baseline`
  - `speedup_vs_cold_reference`
  - `exact_match`

This avoids the misleading interpretation that a variant is faster only because the cold baseline includes first-use setup.

## Results

Input probe:

`result/policy_v2_3_speedup/vae_conv_descriptor_capture_warm1/upsample3_residual0_conv2_probe.pt`

Improved output:

`result/policy_v2_3_speedup/vae_conv_descriptor_capture_warm1/conv_descriptor_probe_results_v4_repeats3.json`

Command:

```bash
python /workspace/scripts/probe_wan_vae_conv_descriptor.py \
  --probe /workspace/result/policy_v2_3_speedup/vae_conv_descriptor_capture_warm1/upsample3_residual0_conv2_probe.pt \
  --out /workspace/result/policy_v2_3_speedup/vae_conv_descriptor_capture_warm1/conv_descriptor_probe_results_v4_repeats3.json \
  --repeats 3 \
  --variants baseline channels_last_3d
```

Measured results:

| Variant | Exact | Min sec | Median sec | Warm baseline speed | Cold reference speed |
|---|---:|---:|---:|---:|---:|
| warmed baseline | yes | 0.0600 | 0.0602 | 0.97x | 718.6x |
| `channels_last_3d` | yes | 0.3070 | 0.3083 | 0.19x | 140.6x |

Reference timings:

- cold reference baseline: `43.152 sec`
- warm baseline min: `0.058 sec`
- warm baseline median: `0.058 sec`

Chunk variants from the prior improved schema run were not exact:

- `chunk_h2`: `max_abs_diff 0.03125`
- `chunk_h4`: `max_abs_diff 0.03125`
- `chunk_w2`: `max_abs_diff 0.03125`
- `chunk_hw2x2`: `max_abs_diff 0.03125`

They are not acceptable for quality-preserving comparison.

## Interpretation

The earlier apparent `channels_last_3d` win was a cold-start artifact.

`channels_last_3d` avoids the very slow first baseline call, but after MIOpen/Tensile is warmed, the normal baseline conv is faster:

- warmed baseline: about `0.058-0.060 sec`
- `channels_last_3d`: about `0.307-0.308 sec`

Therefore `channels_last_3d` should not be enabled as the steady-state VAE conv optimization for v2.3.

The real Policy VAE issue is still that this descriptor appears many times in full decode, and first-use/profile setup can dominate unless the descriptor database and solver path are already warmed. For full benchmark comparison, the correct condition is:

- warmed MIOpen DB
- no kernel logging
- measured run after warmup
- no `channels_last_3d` target replacement unless a full output equivalence and steady-state speed win are both shown

## Full Runner Attempt

Attempted full `warmup-runs 1` with cache-aware `--policy-vae-target-conv-channels-last`.

The run was stopped during warmup because it had not reached measured phase after several minutes and GPU was at 100% utilization with 98C edge temperature. No valid full speed result was produced from that run.

This is not treated as a regression result. The standalone probe is sufficient to reject `channels_last_3d` as a warmed steady-state optimization candidate.

## Current Recommendation

Do not apply `channels_last_3d` to the target VAE conv in v2.3.

Keep the improved probe and cache-aware runner code for future experiments, but the next optimization should focus on one of:

1. Ensuring the warmed MIOpen FindDb covers this exact descriptor before measured Policy decode.
2. Reducing repeated invocation count of the target VAE conv at the model/decode scheduling level.
3. Implementing or enabling a better ROCm/MIOpen solver for this exact BF16 NCDHW 3D conv descriptor.

## Follow-up: MIOpen Find Mode

Ran the improved standalone probe under Docker with explicit MIOpen settings.

Shared settings:

```bash
MIOPEN_USER_DB_PATH=/workspace/result/miopen_vae_conv_descriptor_db
```

### NORMAL + Persistent FindDb

Run 1:

- output: `result/miopen_vae_conv_descriptor_runs/finddb_run1.json`
- cold reference baseline: `47.329 sec`
- warm baseline min: `0.062 sec`

Run 2, same DB path:

- output: `result/miopen_vae_conv_descriptor_runs/finddb_run2.json`
- cold reference baseline: `46.505 sec`
- warm baseline min: `0.062 sec`

The persistent user DB file was created, but it did not remove the 46-47 sec cold cost for this standalone descriptor probe.

### FAST

Output:

`result/miopen_vae_conv_descriptor_runs/finddb_fast.json`

Result:

- cold reference baseline: `0.516 sec`
- warm baseline min: `0.058 sec`
- warmed variant baseline min: `0.059 sec`

`MIOPEN_FIND_MODE=FAST` removes the 40+ sec cold exhaustive Find cost for this descriptor.

### NONE

Attempted output:

`result/miopen_vae_conv_descriptor_runs/finddb_none.json`

Result:

- failed with `RuntimeError: miopenStatusUnknownError`
- stderr included `MIOpen Error: stoul`
- no JSON result was written because the first conv call failed

`MIOPEN_FIND_MODE=NONE` is not usable for this descriptor in the current stack.

## Follow-up: Full Policy FAST Validation

Ran full Policy video+action with:

```bash
MIOPEN_FIND_MODE=FAST
MIOPEN_USER_DB_PATH=/workspace/result/miopen_vae_conv_descriptor_db
```

Output:

`result/policy_v2_3_speedup/vae_miopen_fast_warm1`

Compared with v2.3 baseline:

`result/policy_v2_3_speedup/condition_cache_video_action_warm1`

| Case | generate_batch | generate_samples | decode | warmup generate_batch |
|---|---:|---:|---:|---:|
| v2.3 baseline | 147.757 sec | 51.465 sec | 96.081 sec | 1891.742 sec |
| MIOpen FAST | 188.871 sec | 84.917 sec | 103.714 sec | 281.724 sec |

FAST reduces warmup substantially, but measured runtime is slower:

- measured `generate_batch`: `+41.114 sec`
- measured `generate_samples`: `+33.451 sec`
- measured `decode`: `+7.633 sec`

Output equivalence:

- MP4 hash did not match.
- action `max_abs_diff`: `0.0423287451`
- action `mean_abs_diff`: `0.0064727642`
- nonzero action elements: `160 / 160`

## Follow-up: Action-only FAST Validation

Ran action-only Policy with the same FAST setting:

Output:

`result/policy_v2_3_speedup/action_only_miopen_fast_warm1`

Compared with:

`result/policy_v2_3_speedup/condition_cache_action_only_warm1`

| Case | generate_batch | generate_samples | decode | warmup generate_batch |
|---|---:|---:|---:|---:|
| v2.3 action-only baseline | 52.162 sec | 52.018 sec | ~0.0001 sec | 773.690 sec |
| MIOpen FAST action-only | 86.522 sec | 86.285 sec | ~0.0001 sec | 177.502 sec |

Action difference matched the full FAST run:

- action `max_abs_diff`: `0.0423287451`
- action `mean_abs_diff`: `0.0064727642`
- nonzero action elements: `160 / 160`

This proves the FAST-induced difference is upstream of VAE decode. The likely affected path is conditioning / VAE encode or other MIOpen-backed convolution before velocity/action generation.

## Updated Recommendation

`MIOPEN_FIND_MODE=FAST` is useful as a diagnostic or warmup-time reduction setting, but it is not acceptable for the quality-fixed/article-equivalent v2.3 comparison:

- It changes action output.
- It changes MP4 output.
- It slows measured Policy generation in this run.

For v2.3 quality-preserving comparison, keep `MIOPEN_FIND_MODE=NORMAL` or the previous default condition, and treat FAST only as a separate, non-equivalent speed/latency mode.

## Follow-up: Solver / Kernel Deep Dive

Implemented two probe improvements:

- `scripts/probe_wan_vae_conv_descriptor.py`
  - added `--save-baseline-output`
- `scripts/compare_tensor_outputs.py`
  - compares saved tensor outputs from different MIOpen modes

### Single Conv Output Equivalence

Saved output tensors for the same target descriptor under multiple MIOpen modes:

- `NORMAL`: `result/miopen_vae_conv_descriptor_runs/solver_normal_output.pt`
- `FAST`: `result/miopen_vae_conv_descriptor_runs/solver_fast_output.pt`
- `HYBRID`: `result/miopen_vae_conv_descriptor_runs/solver_hybrid_output.pt`
- `DYNAMIC_HYBRID`: `result/miopen_vae_conv_descriptor_runs/solver_dynamic_hybrid_output.pt`

Compared against `NORMAL`:

| Mode | Cold sec | Warm sec | max abs diff vs NORMAL | nonzero |
|---|---:|---:|---:|---:|
| NORMAL | 43.065 | 0.056 | reference | reference |
| FAST | 0.782 | 0.056 | 0.0 | 0 |
| HYBRID | 0.520 | 0.106 | 0.0 | 0 |
| DYNAMIC_HYBRID | 0.521 | 0.076 | 0.0 | 0 |

Artifacts:

- `result/miopen_vae_conv_descriptor_runs/solver_normal_vs_fast_output_diff.json`
- `result/miopen_vae_conv_descriptor_runs/solver_normal_vs_hybrid_output_diff.json`
- `result/miopen_vae_conv_descriptor_runs/solver_normal_vs_dynamic_hybrid_output_diff.json`

Conclusion: for this single VAE decoder conv descriptor, `FAST`, `HYBRID`, and `DYNAMIC_HYBRID` produce bit-identical output to `NORMAL`.

This means the full Policy action difference observed with `FAST` is not caused by this specific VAE decoder target conv. It is likely caused by another MIOpen-backed op upstream of action generation, such as VAE encode / conditioning.

### MIOpen Solver / Kernel Logs

Logs:

- `result/miopen_vae_conv_descriptor_runs/solver_fast_miopen.log`
- `result/miopen_vae_conv_descriptor_runs/solver_normal_miopen.log`
- `result/miopen_vae_conv_descriptor_runs/solver_hybrid_miopen.log`

The FindDb entry for the target descriptor is:

```text
512-3-242-322-3x3x3-256-1-240-320-1-0x0x0-1x1x1-1x1x1-0-NCDHW-BF16-F=GemmFwdRest:54.4888,2123366400,miopenConvolutionFwdAlgoGEMM;ConvDirectNaiveConvFwd:4966.66,0,miopenConvolutionFwdAlgoDirect
```

Key findings:

- `NORMAL` performs full Find.
- During full Find, MIOpen evaluates:
  - `ConvDirectNaiveConvFwd`
  - kernel: `naive_conv_ab_nonpacked_fwd_ncdhw_ushort_double_ushort`
  - measured around `4966 ms`
- `NORMAL` then evaluates and selects:
  - solver: `GemmFwdRest`
  - kernel path: `Im3d2Col` + `rocBLAS`
  - measured around `54 ms`
  - workspace: `2123366400` bytes
- `FAST` uses the DB/immediate path and chooses:
  - solver: `GemmFwdRest`
  - `Im3d2Col` + `rocBLAS`
- `HYBRID` with a DB hit also chooses:
  - solver: `GemmFwdRest`
  - `Im3d2Col` + `rocBLAS`
  - no `ConvDirectNaiveConvFwd` evaluation in the log

Important correction:

The previously observed `naive_conv_ab_nonpacked_fwd_ncdhw_ushort_double_ushort` bottleneck is likely a Find-time candidate benchmark, not the final steady-state kernel for this descriptor. The steady-state selected execution path is `GemmFwdRest` with `Im3d2Col` plus `rocBLAS`.

### HYBRID Policy Attempt

Attempted action-only Policy with:

```bash
MIOPEN_FIND_MODE=HYBRID
MIOPEN_USER_DB_PATH=/workspace/result/miopen_vae_conv_descriptor_db
```

Output directory:

`result/policy_v2_3_speedup/action_only_miopen_hybrid_warm1`

The run was stopped before sampling after several minutes. GPU was active at 100%, which is consistent with `HYBRID` falling back to full Find for descriptors that were not present in the small single-conv DB.

Interpretation:

- `HYBRID` is promising for a descriptor that already has a good DB entry.
- The current DB only covers the single decoder conv descriptor.
- Policy-wide use of `HYBRID` needs a DB warmup phase that covers all relevant VAE encode/decode/conditioning descriptors.

## Updated Solver-side Hypothesis

The correct hypothesis is now:

1. The target VAE decoder conv steady-state solver is already `GemmFwdRest` (`Im3d2Col` + `rocBLAS`), not naive conv.
2. The large `naive_conv` cost is paid during `NORMAL` full Find because MIOpen benchmarks `ConvDirectNaiveConvFwd` as a candidate.
3. `FAST` avoids that Find cost but can change other upstream Policy outputs, so it is not article-equivalent globally.
4. `HYBRID` can avoid the Find cost without changing this descriptor, but only if the DB contains all relevant descriptors.

Next best experiment:

Build a Policy-wide MIOpen user DB under `NORMAL` once, then rerun with `HYBRID` and compare:

- action equality
- MP4 hash
- warmup time
- measured time

This is the most direct route to a quality-preserving warmup reduction.
