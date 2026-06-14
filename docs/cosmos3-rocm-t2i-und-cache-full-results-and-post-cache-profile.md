# Cosmos3 T2I und branch cache full result and post-cache profile

Date: 2026-06-04

## Scope

The requested four items were executed for the article-equivalent T2I case:

1. T2I fixed computation cache validation
2. Attention backend and post-cache rocprof
3. RMSNorm/RoPE/MLP fusion investigation
4. Transformer kernel launch reduction investigation

Condition:

- mode: text-to-image
- image: `960x960`
- steps: `35`
- guidance: `1.0`
- seed: article sample condition
- quality reduction: none
- cache: Diffusers-native `und_branch_cache`
- GEMM selection: persisted TunableOp table

## Executive result

| Variant | Total | Transformer | VAE decode | Article ratio |
|---|---:|---:|---:|---:|
| Previous best before T2I cache | `88.344 sec` | `85.995 sec` | `1.769 sec` | `4.0x slower` |
| T2I native und branch cache full output | **`27.136 sec`** | **`24.837 sec`** | `1.746 sec` | **`1.23x slower`** |

The article T2I generation time is `22 sec`. The new ROCm result is therefore:

- `27.136 / 22 = 1.23x slower`
- `88.344 / 27.136 = 3.26x faster` than the previous T2I best
- `974.611 / 27.136 = 35.9x faster` than cold v1

## 1. Fixed computation cache validation

Full output run:

- artifact: `result/rocm_speed_matrix/aotriton_tuned/t2i_und_cache_warm_full/article_t2i_summary.json`
- output image: `result/rocm_speed_matrix/aotriton_tuned/t2i_und_cache_warm_full/article_t2i_robotics_lab_960x960_s35_measured_r2.jpg`

Measured run:

| Stage | Time |
|---|---:|
| total | `27.136 sec` |
| transformer | `24.837 sec / 35 calls` |
| VAE decode | `1.746 sec` |
| video postprocess | `0.008 sec` |
| unattributed | `0.545 sec` |

Warmup was intentionally excluded from the measured value:

- VAE warmup: `659.752 sec`
- mode warmup total: `137.753 sec`
- mode warmup transformer: `26.742 sec`

Cache stats:

```json
{
  "enabled": true,
  "transformer_calls": 70,
  "write_calls": 1,
  "read_calls": 69,
  "invalidations": 1,
  "cached_layers": 36,
  "cache_gib": 0.882
}
```

Interpretation:

- The first denoising call writes the understanding-branch cache.
- The following calls reuse it.
- This avoids recomputing stable understanding-branch layer outputs while preserving the same denoising condition.

## Output validation

Compared against the previous best output:

- baseline: `result/rocm_speed_matrix/aotriton_tunable/t2i_article_warm_full/article_t2i_robotics_lab_960x960_s35_measured_r2.jpg`
- cache: `result/rocm_speed_matrix/aotriton_tuned/t2i_und_cache_warm_full/article_t2i_robotics_lab_960x960_s35_measured_r2.jpg`

SHA256:

| File | SHA256 |
|---|---|
| baseline | `9352d9cb724d510d6bb858a758233fee9ca97b61d58a6e66254f757c527a4bd5` |
| cache | `9352d9cb724d510d6bb858a758233fee9ca97b61d58a6e66254f757c527a4bd5` |

Image diff metrics:

- artifact: `result/t2i_und_cache_validation/image_diff_metrics.json`
- `size_equal: true`
- `mean_abs_rgb: [0.0, 0.0, 0.0]`
- `rms_rgb: [0.0, 0.0, 0.0]`
- `max_abs: 0`
- `nonzero_bbox: null`

Conclusion:

- The cached full output is byte-identical to the previous best image.
- The optimization does not lower image quality or change the result.

## 2. Attention backend and post-cache rocprof

Post-cache selected-region rocprof artifact:

- `result/rocm_speed_matrix/aotriton_tuned/t2i_und_cache_transformer_rocprof/article_t2i_summary.json`
- `result/rocm_speed_matrix/aotriton_tuned/t2i_und_cache_transformer_rocprof/rocprof/profile_kernel_stats.csv`

No-cache baseline:

- `result/rocm_speed_matrix/aotriton_tuned/t2i_transformer_only_rocprof/rocprof/profile_kernel_stats.csv`

Kernel category comparison:

| Category | No-cache time | No-cache calls | Cache time | Cache calls | Cache share |
|---|---:|---:|---:|---:|---:|
| GEMM | `60.651 sec` | `17,850` | **`19.559 sec`** | `9,282` | **`74.14%`** |
| attention | `12.349 sec` | `2,520` | `4.164 sec` | `1,296` | `15.78%` |
| elementwise/reduce/copy | `10.540 sec` | `113,715` | `2.348 sec` | `58,950` | `8.90%` |
| cat/copy | `0.555 sec` | `7,700` | `0.304 sec` | `5,252` | `1.15%` |
| other | `0.007 sec` | `179` | `0.007 sec` | `389` | `0.03%` |
| total kernels | `84.102 sec` | `141,964` | **`26.381 sec`** | **`75,169`** | `100%` |

Post-cache top kernels:

| Rank | Kernel family | Time | Calls |
|---:|---|---:|---:|
| 1 | GEMM | `8.822 sec` | `2,520` |
| 2 | GEMM | `4.834 sec` | `1,332` |
| 3 | `attn_fwd` | `4.164 sec` | `1,296` |
| 4 | GEMM | `3.344 sec` | `2,520` |
| 5 | GEMM | `0.861 sec` | `2,520` |
| 6 | time-embed-like GEMM | `0.688 sec` | `35` |

Conclusion:

- Attention dropped from `12.349 sec` to `4.164 sec` after cache.
- It remains meaningful, but it is no longer the dominant T2I bottleneck.
- A future attention backend improvement can target at most about `4 sec` of the measured transformer time unless it also changes GEMM-adjacent projection behavior.

## 3. RMSNorm/RoPE/MLP fusion investigation

Post-cache non-GEMM scalar work is much smaller than before:

- elementwise/reduce/copy: `10.540 -> 2.348 sec`
- cat/copy: `0.555 -> 0.304 sec`
- combined non-attention non-GEMM: about `2.65 sec`

Linear profile with cache:

- artifact: `result/rocm_speed_matrix/aotriton_tuned/t2i_und_cache_linear_probe/article_t2i_summary.json`
- transformer: `27.153 sec / 35 calls`
- profiled Linear total: `19.568 sec`
- profiled Linear modules: `508`

Top Linear work is generation-side MLP and projection:

| Module class | Signal |
|---|---|
| `mlp_moe_gen.down_proj` | repeated `~0.130 sec` per layer aggregate over 35 calls |
| `mlp_moe_gen.gate_proj` | repeated `~0.110-0.129 sec` per layer aggregate over 35 calls |
| `mlp_moe_gen.up_proj` | repeated `~0.106-0.123 sec` per layer aggregate over 35 calls |
| `self_attn.add_q_proj` | repeated `~0.052 sec` per layer aggregate over 35 calls |
| `time_embedder.linear_2` | `0.696 sec / 35 calls` |

Conclusion:

- MLP fusion should not be framed as a large standalone win unless it reduces GEMM count or fuses activation around the existing GEMM path.
- RMSNorm/RoPE fusion can only target the post-cache `~2.65 sec` non-GEMM bucket.
- The realistic local Python-level fusion target is about `1 sec`, not the original no-cache `~11 sec`.

## 4. Transformer kernel launch reduction investigation

Kernel launches reduced substantially after cache:

- no-cache: `141,964` kernel calls
- cache: `75,169` kernel calls
- reduction: `46.9%`

The remaining calls are still high, but the time distribution shows the launch-reduction target has narrowed:

- GEMM remains the main time consumer: `19.559 sec`
- attention remains second: `4.164 sec`
- elementwise/reduce/copy plus cat/copy: `2.652 sec`

Conclusion:

- The cache itself is already the largest launch-reduction change.
- Further launch reduction is worth investigating only after keeping the cache enabled.
- Whole-transformer `torch.compile` is risky because the pipeline has dynamic inputs and large graph boundaries.
- Isolated compile/fusion probes are more appropriate: RMSNorm, RoPE, and `silu(gate) * up`.

## Updated bottleneck model

The old bottleneck model was:

1. transformer forward dominates
2. within transformer, GEMM dominates
3. attention and elementwise are secondary

The updated post-cache bottleneck model is:

1. transformer forward still dominates total runtime: `24.837 / 27.136 sec`
2. within transformer, GEMM remains dominant: about `19.6 sec`
3. attention is a secondary `~4.2 sec` target
4. fusion/launch reduction is a smaller `~2.6 sec` target

To match the article `22 sec`, the remaining gap is about `5.1 sec` total. That requires reducing post-cache transformer from `24.837 sec` to about `19.7 sec` while keeping VAE/postprocess similar.

## Next improvement plan

Priority order:

1. Keep T2I `und_branch_cache` as the default optimized path for article-equivalent runs.
2. Deep-dive post-cache GEMM selection for the generation branch, because GEMM is still `~74%` of kernel time.
3. Test newer PyTorch ROCm/AOTriton attention stack only if it can improve `attn_fwd` without output changes.
4. Prototype isolated fusion for RMSNorm/RoPE/MLP activation, but treat it as a secondary optimization with a likely `<=1 sec` local win.
5. Avoid quality-changing changes such as lowering steps, resolution, dtype, or frame count for article comparison.

