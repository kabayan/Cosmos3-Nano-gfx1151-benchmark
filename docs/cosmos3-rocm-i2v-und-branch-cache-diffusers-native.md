# Cosmos3 I2V und branch cache: Diffusers native implementation

## Scope

The previous `UndBranchCachePrototype` monkeypatch in `scripts/benchmark_classmethod_article_t2v_i2v_rocm.py` was moved into the Diffusers Cosmos3 transformer implementation:

- `third_party/diffusers/src/diffusers/models/transformers/transformer_cosmos3.py`
- Benchmark runner now prefers the native API when available:
  - `pipe.transformer.enable_und_branch_cache(True, reset=True)`
  - `pipe.transformer.get_und_branch_cache_stats()`
  - fallback remains the old prototype for older Diffusers builds

This optimization targets Cosmos3 I2V repeated transformer calls where the understanding branch is stable across denoising steps. It does not change sampling steps, resolution, frame count, prompt, seed, VAE, or generated-token computation.

## Implementation

`Cosmos3VLTextMoTDecoderLayer` now has a cache-aware path:

- write path:
  - computes the normal understanding branch attention and MLP
  - computes generation branch normally
  - caches per layer:
    - `und_next`
    - rotary-applied `k_und`
    - `v_und`
- read path:
  - reuses cached `und_next`, `k_und`, and `v_und`
  - recomputes generation branch Q/K/V, full attention, output projection, and generation MLP

`Cosmos3OmniTransformer` owns cache state and exposes:

- `enable_und_branch_cache(enabled=True, reset=True)`
- `disable_und_branch_cache()`
- `reset_und_branch_cache()`
- `get_und_branch_cache_stats()`

The cache is only active for inference:

- `self.training == False`
- `torch.is_grad_enabled() == False`

Training and gradient-enabled paths use the original layer forward.

## Cache invalidation

The cache is invalidated when any stable understanding-side signature changes, or when the cache is incomplete.

Signature fields:

- `input_ids`
- `text_indexes`
- `position_ids`
- `und_len`
- `sequence_length`

Tensor signatures include shape, dtype, device, sum, and absolute sum. This is intentionally conservative for the current benchmark work: it detects prompt/position/layout changes while avoiding `vision_tokens` and `vision_timesteps`, which change every denoising step and belong to the generation branch.

Observed full-run stats:

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

The single invalidation is the initial cache population. The 70 transformer calls are 35 warmup calls plus 35 measured calls.

## Tests

### Static/import checks

```bash
python3 -m py_compile \
  scripts/benchmark_classmethod_article_t2v_i2v_rocm.py \
  scripts/run_rocm_speed_matrix.py \
  third_party/diffusers/src/diffusers/models/transformers/transformer_cosmos3.py
```

Result: passed.

Docker image rebuilt:

```bash
scripts/build_cosmos3_rocm72_diffusers_image.sh cosmos3-rocm72-diffusers:local
```

Container API check:

```text
hasattr(Cosmos3OmniTransformer, enable_und_branch_cache): True
```

Layer equivalence check on a small CPU model:

```text
normal forward == cache write: True
cache write == cache read: True
max abs diff: 0.0
```

### I2V transformer-only smoke

Command shape:

```bash
python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py \
  --case i2v \
  --height 256 --width 448 --frames 24 --fps 12 \
  --steps 2 --guidance 1.0 \
  --stage-profile \
  --inference-mode --disable-progress-bar \
  --und-branch-cache \
  --abort-before-vae-decode --allow-pipeline-error
```

Artifact:

- `result/rocm_speed_matrix/aotriton_tuned/i2v_und_cache_native_2step_probe/summary.json`

Result:

```json
{
  "transformer_forward": {
    "seconds": 2.82,
    "calls": 2
  },
  "und_branch_cache": {
    "enabled": true,
    "transformer_calls": 2,
    "write_calls": 1,
    "read_calls": 1,
    "invalidations": 1,
    "cached_layers": 36,
    "cache_gib": 0.784
  }
}
```

### Full I2V article-equivalent run

Command:

```bash
COSMOS3_ROCM_IMAGE=cosmos3-rocm72-diffusers:local \
COSMOS3_DIFFUSERS_INSTALL=true \
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned \
  --case i2v_article_und_cache_warm_full \
  --execute
```

Artifact:

- `result/rocm_speed_matrix/aotriton_tuned/i2v_article_und_cache_warm_full/summary.json`
- `result/rocm_speed_matrix/aotriton_tuned/i2v_article_und_cache_warm_full/article_i2v_robot_arms_256p_24f_s35.mp4`

Measured result:

```json
{
  "seconds": 25.045,
  "stage_profile": {
    "transformer_forward": {
      "seconds": 19.68,
      "calls": 35
    },
    "vae_decode": {
      "seconds": 4.134,
      "calls": 1
    },
    "video_postprocess": {
      "seconds": 0.021,
      "calls": 1
    }
  }
}
```

The VAE warmup remained unchanged and took `370.074 sec`.

## Output equivalence

The native cache output matched the no-cache baseline MP4 exactly:

```text
04691d3b0e3450468ee41544e5727261ae40864217d5c0d71ee48e08e93c2dc1  i2v_article_und_cache_warm_full/article_i2v_robot_arms_256p_24f_s35.mp4
04691d3b0e3450468ee41544e5727261ae40864217d5c0d71ee48e08e93c2dc1  i2v_article_warm_full/article_i2v_robot_arms_256p_24f_s35.mp4
```

This confirms the Diffusers-native cache preserves the article-equivalent I2V output for the tested seed/settings.

## Comparison

Previous best no-cache warm full I2V:

- total: `77.160 sec`
- transformer: `71.691 sec`
- VAE decode: `4.195 sec`

Diffusers-native und branch cache:

- total: `25.045 sec`
- transformer: `19.680 sec`
- VAE decode: `4.134 sec`

Improvement:

- total: `3.08x faster`
- transformer: `3.64x faster`

The remaining bottleneck is still transformer forward, now mostly the generation branch GEMM/attention path plus fixed VAE decode.
