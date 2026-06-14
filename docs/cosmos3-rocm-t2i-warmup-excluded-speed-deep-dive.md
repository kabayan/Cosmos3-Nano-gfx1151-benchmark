# Cosmos3 T2I speed deep dive, excluding warmup

## Scope

This note analyzes Text-to-image speed only for the measured generation window.

Excluded from the speed comparison:

- model load
- synthetic VAE warmup
- mode warmup run
- first VAE decode warmup effects

The article-equivalent T2I condition remains:

- `960x960`
- `35` diffusion steps
- `guidance_scale=1.0`
- seed `201`
- prompt from `inputs/omni/t2i.json`

## Measured speed history

| Variant | Measured total | Transformer | VAE decode | Image postprocess |
|---|---:|---:|---:|---:|
| `v1_0/t2i_article_warm_full` | 206.367 sec | 203.997 sec | 1.774 sec | 0.008 sec |
| `aotriton/t2i_article_warm_full` | 104.297 sec | 101.961 sec | 1.755 sec | 0.007 sec |
| `aotriton_tunable/t2i_article_warm_full` | 88.344 sec | 85.995 sec | 1.769 sec | 0.008 sec |
| `aotriton_tuned/t2i_transformer_only_rocprof` | transformer-only probe | 85.029 sec | aborted | aborted |

Warmup was disabled for the transformer-only probe and VAE decode was intentionally aborted after transformer completion. The full pipeline `seconds` in that abort probe includes exception/unattributed pipeline overhead and should not be used as the speed metric; the timed `transformer_forward` stage is the relevant value.

## Improvement attribution

Warmup-excluded measured speed:

- v1 measured total to current best: `206.367 -> 88.344 sec`, `2.34x faster`
- v1 transformer to current best transformer: `203.997 -> 85.995 sec`, `2.37x faster`
- AOTriton baseline to current best measured total: `104.297 -> 88.344 sec`, `1.18x faster`
- AOTriton transformer to current best transformer: `101.961 -> 85.995 sec`, `1.19x faster`

The improvement is transformer-side. VAE decode is already about `1.77 sec` in the measured window and is not the current T2I bottleneck.

## Kernel breakdown

Transformer-only rocprof, warmup excluded:

| Kernel category | AOTriton | TunableOp persisted/tuned | Change |
|---|---:|---:|---:|
| GEMM | 77.779 sec / 76.66% | 60.651 sec / 72.12% | `1.28x faster` |
| Attention | 12.582 sec / 12.40% | 12.349 sec / 14.68% | almost unchanged |
| Elementwise/reduce/copy | 11.100 sec / 10.94% | 11.095 sec / 13.19% | unchanged |
| Kernel total | 101.466 sec | 84.102 sec | `1.21x faster` |

Top AOTriton kernels were dominated by two `MT128x128x32` Tensile GEMM families:

- `52.643 sec`, `6300` calls
- `21.351 sec`, `6300` calls
- `attn_fwd`: `12.582 sec`, `2520` calls

After TunableOp, GEMM kernel selection changed:

- `MT64x192x32`: `20.000 sec`, `2520` calls
- `MT128x96x32`: `12.879 sec`, `1260` calls
- `MT96x128x32`: `12.645 sec`, `3780` calls
- `attn_fwd`: `12.349 sec`, `2520` calls

This confirms the same mechanism seen in T2V/I2V: TunableOp did not make the same GEMM kernel faster; it selected different GEMM kernels for the same transformer shapes.

## Linear profile signal

Artifact:

- `result/rocm_speed_matrix/aotriton_tuned/t2i_transformer_only_linear_probe/article_t2i_summary.json`

Result:

- transformer stage: `85.870 sec / 35 calls`
- linear profiler total: `61.638 sec`
- linear modules observed: `508`

The top linear records are MLP GEMMs over shape `(2141, 4096)` / `(2141, 12288)`.

Visible top-family totals from the saved top-80 records:

| Linear family | Time in saved top-80 | Calls in saved top-80 |
|---|---:|---:|
| `layers.*.mlp.down_proj` | 12.889 sec | 1260 |
| `layers.*.mlp.up_proj` | 10.074 sec | 1260 |
| `layers.*.mlp.gate_proj` | 1.922 sec | 245 |
| `time_embedder.linear_2` | 0.687 sec | 35 |

Because the saved profile stores only the top 80 records, this is not a complete per-family total. It is still enough to show that the biggest visible Linear cost is the transformer MLP GEMM path, matching the rocprof GEMM-dominant result.

## Current bottleneck

Current best T2I measured:

- total: `88.344 sec`
- transformer: `85.995 sec`
- transformer share: `97.3%`

Within the transformer:

- GEMM remains the main bottleneck: `60.651 sec`
- attention is second: `12.349 sec`
- elementwise/reduce/copy overhead is also non-trivial: `11.095 sec`

The current T2I gap vs article is:

- article: `22 sec`
- current best: `88.344 sec`
- difference: `4.0x slower`

## Implications for further improvement

Quality-preserving improvement paths:

1. Continue GEMM kernel selection/tuning work.
   - This is the proven path: GEMM dropped from `77.779` to `60.651 sec`.
   - Remaining target is the selected `MT64x192x32`, `MT128x96x32`, and `MT96x128x32` families.

2. Investigate attention backend improvements.
   - Attention is now a larger share after GEMM improved: `14.68%`.
   - It did not materially improve with TunableOp, so this requires SDPA/AOTriton/PyTorch ROCm attention work, not GEMM table persistence.

3. Reduce elementwise/reduce overhead through implementation fusion.
   - Elementwise/reduce/copy stayed at about `11.1 sec`.
   - This likely needs transformer implementation changes or compiler/kernel fusion, not only solver selection.

4. Avoid quality-changing optimizations for article comparison.
   - Step reduction, resolution reduction, frame/token reduction, or approximated attention would improve speed but invalidate article-equivalent comparison.

## Artifacts

- `result/rocm_speed_matrix/aotriton_tuned/t2i_transformer_only_rocprof/article_t2i_summary.json`
- `result/rocm_speed_matrix/aotriton_tuned/t2i_transformer_only_rocprof/rocprof/profile_kernel_stats.csv`
- `result/rocm_speed_matrix/aotriton_tuned/t2i_transformer_only_linear_probe/article_t2i_summary.json`
- `result/rocm_speed_matrix/aotriton/t2i_article_transformer_rocprof/rocprof/profile_kernel_stats.csv`
