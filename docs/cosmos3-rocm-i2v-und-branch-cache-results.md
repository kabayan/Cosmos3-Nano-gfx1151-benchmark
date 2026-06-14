# Cosmos3 I2V `und` Branch Cache Prototype Results

Date: 2026-06-04

## Summary

Implemented an opt-in `und` branch cache prototype for Cosmos3 I2V.

The prototype keeps quality settings unchanged and skips recomputation of the stable text-only `und` branch after the first transformer call.

Current best baseline:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_article_warm_full/summary.json
measured end-to-end:          77.160 sec
measured transformer_forward: 71.691 sec
```

New cache result:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_und_cache_warm_full/summary.json
measured end-to-end:          25.009 sec
measured transformer_forward: 19.665 sec
```

Improvement:

| Metric | Baseline | `und` cache | Speedup |
|---|---:|---:|---:|
| End-to-end | `77.160 sec` | `25.009 sec` | `3.09x` |
| Transformer | `71.691 sec` | `19.665 sec` | `3.65x` |

## Implementation

File:

```text
scripts/benchmark_classmethod_article_t2v_i2v_rocm.py
```

Added:

```text
--und-branch-cache
```

Runner case:

```text
i2v_article_und_cache_warm_full
```

File:

```text
scripts/run_rocm_speed_matrix.py
```

Dry-run command:

```text
COSMOS3_ROCM_IMAGE=cosmos3-rocm72-diffusers:local \
COSMOS3_DIFFUSERS_INSTALL=true \
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned \
  --case i2v_article_und_cache_warm_full
```

## How It Works

For I2V, earlier diagnostics showed these transformer inputs are stable across denoising steps:

```text
input_ids
text_indexes
position_ids
sequence_length
und_len = 1904
vision_sequence_indexes
vision_mse_loss_indexes
vision_noisy_frame_indexes
```

And these change:

```text
vision_tokens
vision_timesteps
```

Therefore, the `und` prefix is text-only and stable, while the `gen` branch changes with noisy vision latents.

The prototype patches each `Cosmos3VLTextMoTDecoderLayer`:

1. First transformer call:
   - compute full layer normally;
   - cache per layer:
     - `und_next`
     - rotary-applied `k_und`
     - `v_und`
2. Later transformer calls:
   - reuse cached `und_next`;
   - skip `und` self-attention;
   - skip `und` MLP;
   - compute only `gen` branch;
   - concatenate cached `k_und/v_und` with current `k_gen/v_gen` for generation attention.

Cache stats from full run:

```json
{
  "enabled": true,
  "transformer_calls": 70,
  "write_calls": 1,
  "read_calls": 69,
  "invalidations": 1,
  "cached_layers": 36,
  "cache_gib": 0.784
}
```

## Transformer-only Probes

2-step probe:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_und_cache_2step_probe/summary.json
```

| Case | Transformer time |
|---|---:|
| baseline 2-step input profile | `4.258 sec / 2 calls` |
| `und` cache 2-step | `2.832 sec / 2 calls` |

35-step transformer-only probe:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_und_cache_35step_probe/summary.json
```

| Case | Transformer time |
|---|---:|
| baseline transformer-only | `70.769 sec / 35 calls` |
| `und` cache transformer-only | `20.876 sec / 35 calls` |

## Full I2V Result

Command used:

```text
docker run ... cosmos3-rocm72-diffusers:local \
  python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py \
  --case i2v \
  --out-dir /workspace/result/rocm_speed_matrix/aotriton_tuned/i2v_und_cache_warm_full \
  --height 256 --width 448 \
  --frames 24 --fps 12 \
  --steps 35 --guidance 1.0 \
  --stage-profile \
  --vae-warmup --vae-warmup-shape 1,48,2,16,28 \
  --mode-warmup-runs 1 --measured-runs 1 \
  --inference-mode --disable-progress-bar \
  --und-branch-cache
```

Measured run:

| Stage | Time |
|---|---:|
| total | `25.009 sec` |
| transformer_forward | `19.665 sec` |
| vae_decode | `4.112 sec` |
| video_postprocess | `0.020 sec` |
| unattributed | `1.212 sec` |

Note:

The synthetic VAE warmup in this run took `368.829 sec`, which is the known cold Wan VAE path. It is outside the selected measured run and does not affect the measured result.

## Output Equivalence

Baseline output:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_article_warm_full/article_i2v_robot_arms_256p_24f_s35.mp4
```

Cache output:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_und_cache_warm_full/article_i2v_robot_arms_256p_24f_s35.mp4
```

Binary SHA256:

```text
04691d3b0e3450468ee41544e5727261ae40864217d5c0d71ee48e08e93c2dc1
```

The baseline and cache MP4 files are byte-identical. Decoded frame MD5s are also identical.

## Current Status

The prototype is successful:

- same input/prompt/settings;
- full I2V output succeeds;
- output is byte-identical to baseline;
- transformer time improves from `71.691 sec` to `19.665 sec`;
- end-to-end measured time improves from `77.160 sec` to `25.009 sec`.

## Next Hardening Steps

1. Move the prototype from benchmark monkeypatch to local Diffusers implementation.
2. Add strict cache invalidation:
   - prompt/input IDs;
   - `position_ids[:und_len]`;
   - `und_len`;
   - device/dtype;
   - layer count/config.
3. Support multi-sample batch only after single-sample path is stable.
4. Add a correctness test:
   - run with and without cache;
   - compare predicted latents or exported MP4 hash for fixed seed.
5. Add a runner variant/case for routine benchmark reporting.

