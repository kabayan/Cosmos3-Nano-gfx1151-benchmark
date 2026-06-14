# Cosmos3 T2I non-GEMM improvement investigation and plan

## Current baseline

T2I article-equivalent condition:

- `960x960`
- `35` steps
- `guidance_scale=1.0`
- seed `201`
- warmup excluded for representative speed

Current full measured best before cache validation:

| Variant | Total | Transformer | VAE decode | Notes |
|---|---:|---:|---:|---|
| Existing best `aotriton_tunable` | 88.344 sec | 85.995 sec | 1.769 sec | persisted/general TunableOp |
| Stream-K-aware T2I table | 87.420 sec | 85.078 sec | 1.763 sec | full output generated, not byte-identical |

The GEMM-only path has nearly saturated:

- GEMM after tuning: about `60.651 sec`
- Attention: about `12.349 sec`
- Elementwise/reduce/copy: about `11.095 sec`

## Key new finding: T2I also has a large cache opportunity

Although the initial expectation was that T2I would have less reuse than I2V, the native `und_branch_cache` works for T2I too.

Transformer-only T2I probe:

```bash
python3 scripts/benchmark_classmethod_article_t2i_rocm.py \
  --height 960 --width 960 --steps 35 --guidance 1.0 \
  --stage-profile \
  --measured-runs 1 \
  --abort-before-vae-decode \
  --allow-pipeline-error \
  --und-branch-cache
```

Artifact:

- `result/rocm_speed_matrix/aotriton_tuned/t2i_und_cache_transformer_only_probe/article_t2i_summary.json`

Result:

| Condition | Transformer |
|---|---:|
| Existing tuned transformer-only | `85.029 sec / 35 calls` |
| T2I native und branch cache | **`26.630 sec / 35 calls`** |

Cache stats:

```json
{
  "enabled": true,
  "transformer_calls": 35,
  "write_calls": 1,
  "read_calls": 34,
  "invalidations": 1,
  "cached_layers": 36,
  "cache_gib": 0.882
}
```

This is the highest-impact T2I improvement found so far. It is not a GEMM improvement; it avoids recomputing the stable understanding branch after the first denoising step.

Projected full measured time if VAE/postprocess/unattributed overhead remains similar:

- transformer: `~26.63 sec`
- VAE decode: `~1.76 sec`
- postprocess/unattributed: `<1 sec`
- projected full measured: about `28-30 sec`

This must be validated with a full output-producing run before promoting.

## 1. Attention backend improvement

### Evidence

Current selected-region T2I profile:

- `attn_fwd`: `12.349 sec`, `2520` calls, `14.68%` of transformer kernel total

Attention did not improve materially with GEMM/TunableOp work:

- AOTriton attention: `12.582 sec`
- Tuned attention: `12.349 sec`

### Interpretation

Attention backend work is still useful, but it is no longer the largest immediate opportunity if `und_branch_cache` is valid for full T2I.

After cache, the recomputed path should mostly be the generation branch. Attention cost should drop, but not disappear, because generation full attention still runs.

### Improvement plan

1. Run selected-region rocprof for T2I with `--und-branch-cache`.
   - Goal: identify remaining `attn_fwd` time after cache.
   - Expected: attention calls/time decrease materially from the no-cache profile.

2. Test attention backend variants only after cache profile.
   - Existing local backend switching was not helpful for I2V.
   - For T2I, repeat only if post-cache attention remains above about `5 sec`.

3. If attention remains high, test newer PyTorch ROCm/AOTriton stack.
   - This is likely a stack/kernel issue, not a local Diffusers Python change.

Success criteria:

- Reduce post-cache transformer by at least `2 sec`.
- Preserve image quality and avoid changing step/resolution.

## 2. RMSNorm/RoPE/MLP fusion

### Evidence

The no-cache selected-region profile shows scattered non-GEMM GPU work:

- pow: `1.479 sec`, `10150` calls
- mul elementwise: `1.342 sec`, `20230` calls
- reduce mean: `1.328 sec`, `10150` calls
- fp16 copy: `1.262 sec`, `10255` calls
- add: `1.195 sec`, `10080` calls
- dtype copy: `1.038 sec`, `10150` calls
- broadcast elementwise: `0.855 sec`
- SiLU: `0.694 sec`
- cat/copy kernels: about `0.544 sec`

These correspond to RMSNorm, RoPE, residual adds, SiLU/mul, cat/copy around attention, and dtype conversions.

### Interpretation

Fusion is a secondary target. In the no-cache path the total elementwise/reduce/copy cost is about `11.1 sec`. After cache, part of this cost should be skipped for the understanding branch.

### Improvement plan

1. Profile post-cache elementwise/reduce/copy first.
   - Do not implement fusion against the old no-cache profile.
   - The cache changes which operations are still important.

2. Prototype fused MLP activation only for remaining hot path:
   - Current MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`
   - Candidate: fuse `SiLU(gate) * up` into one Triton/torch.compile path.
   - Validate output tolerance.

3. Prototype RMSNorm replacement only if post-cache RMSNorm remains material.
   - Current RMSNorm likely maps to pow/reduce/rsqrt/mul/copy kernels.
   - Candidate: custom fused RMSNorm kernel or PyTorch compiled function.

4. RoPE fusion is lower priority.
   - RoPE is currently Python-level multiply/rotate/cat style.
   - It may be fused with Q/K projection output handling, but that is more invasive.

Success criteria:

- At least `1 sec` reduction in post-cache transformer.
- No image-level regression beyond acceptable fp variation.

## 3. Transformer kernel launch reduction

### Evidence

The transformer launches many small kernels:

- attention calls: `2520`
- pow/reduce/copy/add/mul kernels: around `10k-20k` calls each family
- cat/copy around attention: thousands of calls

The Python implementation also separates:

- Q/K/V projections
- Q/K RMSNorm
- RoPE
- two attention dispatches
- output projection
- post-attention RMSNorm
- MLP gate/up/down
- residual adds

### Interpretation

Launch reduction should be considered after cache, because cache removes many layer operations from 34 of 35 steps. Without cache, launch reduction helps but cannot beat the repeated full-branch compute.

### Improvement plan

1. Measure post-cache kernel call counts.
   - selected-region rocprof with `--und-branch-cache`
   - compare calls for GEMM, `attn_fwd`, pow/reduce/copy/add/mul

2. Try `torch.compile` only on small isolated functions first.
   - Good candidates: RMSNorm function, MLP activation `silu(gate) * up`, RoPE application.
   - Avoid compiling the whole pipeline initially; dynamic list inputs and Diffusers pipeline calls make it risky.

3. If isolated compile helps, integrate as optional transformer flag.
   - Keep default path unchanged.
   - Add output diff tests.

Success criteria:

- Reduce post-cache kernel count and transformer time.
- Avoid large compile overhead in the measured run.

## 4. T2I fixed computation cache

### Evidence

The native und branch cache is already valid at transformer-only level:

- first call writes cache
- next 34 calls read cache
- cached layers: `36`
- cache memory: `0.882 GiB`
- transformer: `85.029 -> 26.630 sec`

This is the best current improvement candidate.

### Required validation

1. Full T2I output run with cache:
   - VAE warmup
   - mode warmup run
   - measured run
   - image output generated

2. Output comparison:
   - SHA256 likely may differ because execution order changes.
   - Use image diff metrics and side-by-side visual check.

3. Reproducibility:
   - Repeat measured full run once.
   - Confirm speed is stable.

4. WebUI update:
   - Promote T2I value only after full output validation.

### Implementation notes

`scripts/benchmark_classmethod_article_t2i_rocm.py` now has:

- `--und-branch-cache`

The Diffusers native cache API is used:

- `pipe.transformer.enable_und_branch_cache(True, reset=True)`
- `pipe.transformer.get_und_branch_cache_stats()`

## Priority order

### P0: Full T2I und branch cache validation

This is the only candidate with a measured transformer-only improvement large enough to materially close the article gap.

Expected result:

- full measured T2I around `28-30 sec`
- article gap improves from `~4.0x slower` to about `1.3x slower`

### P1: Post-cache rocprof

Run selected-region rocprof with cache enabled.

Purpose:

- determine the new remaining bottleneck
- avoid optimizing old no-cache hotspots

### P2: Attention backend only if still large post-cache

If post-cache attention remains above about `5 sec`, test backend/stack changes.

### P3: Fusion and launch reduction

Only after post-cache profile:

- RMSNorm fusion
- MLP activation fusion
- RoPE fusion
- local `torch.compile` probes

### P4: GEMM table cleanup

The Stream-K-aware GEMM table gives about `1%` improvement and should not be the primary path anymore. Keep it as a secondary additive optimization after cache validation.

## Next concrete run

Run full T2I cache validation:

```bash
docker run ... \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e PYTORCH_TUNABLEOP_ENABLED=1 \
  -e PYTORCH_TUNABLEOP_TUNING=0 \
  -e PYTORCH_TUNABLEOP_FILENAME=/workspace/result/rocm_speed_matrix/tunableop_results%d.csv \
  cosmos3-rocm72-diffusers:local \
  bash -lc "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2i_rocm.py \
    --out-dir result/rocm_speed_matrix/aotriton_tuned/t2i_und_cache_warm_full \
    --height 960 --width 960 --steps 35 --guidance 1.0 \
    --stage-profile \
    --vae-warmup --vae-warmup-shape 1,48,1,60,60 \
    --mode-warmup-runs 1 \
    --measured-runs 1 \
    --und-branch-cache"
```

Then run:

- image diff against previous best
- selected-region rocprof with `--und-branch-cache`
