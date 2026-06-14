# Cosmos3-Nano ROCm I2V speed deep dive

Date: 2026-06-04

## Scope

Deep-dive the current I2V performance after v2.1 TunableOp persistence, using the article-equivalent I2V case:

- Model: `nvidia/Cosmos3-Nano`
- Mode: image-to-video
- Input: Cosmos3 official I2V sample image and prompt
- Output: `448x256`, `24` frames, `12` fps
- Steps: `35`
- Guidance: `1.0`
- Seed: `203`

## Current result

| Version / variant | Total | Transformer | VAE decode | Postprocess | Notes |
|---|---:|---:|---:|---:|---|
| v1.0 | 167.219 sec | 161.762 sec | 4.168 sec | 0.019 sec | Initial article-equivalent run |
| v2 AOTriton | 94.515 sec | 89.021 sec | 4.175 sec | 0.020 sec | Transformer rocprof baseline |
| v2.1 AOTriton + TunableOp tuned | 77.160 sec | 71.691 sec | 4.195 sec | 0.021 sec | Persisted `tunableop_results0.csv`, no retuning |

Speedup:

- v1.0 -> v2.1: `2.17x`
- v2 AOTriton -> v2.1: `1.22x`
- v2 AOTriton transformer -> v2.1 transformer: `1.24x`

The article reports about `17 sec` for I2V on the DGX Spark / CUDA path. Current v2.1 is `77.160 sec`, or about `4.54x` slower for the same article-style case.

## v2.1 bottleneck

At v2.1, the measured total is still dominated by transformer forward:

| Stage | Time | Share of total |
|---|---:|---:|
| Transformer forward | 71.691 sec | 92.9% |
| VAE decode | 4.195 sec | 5.4% |
| Video postprocess | 0.021 sec | 0.0% |
| Other pipeline overhead | 1.253 sec | 1.6% |

VAE decode is no longer the target for I2V latency. Even a perfect VAE decode would only remove about `4.2 sec`.

## Transformer kernel split

Selected-region rocprof for measured-run `transformer.forward`:

| Variant | Kernel total | GEMM | Attention | Elementwise | Other/copy |
|---|---:|---:|---:|---:|---:|
| AOTriton | 88.418 sec | 69.619 sec / 78.74% | 9.702 sec / 10.97% | 7.603 sec / 8.60% | 1.494 sec / 1.69% |
| AOTriton + TunableOp | 71.048 sec | 51.852 sec / 72.98% | 9.926 sec / 13.97% | 7.743 sec / 10.90% | 1.526 sec / 2.15% |

TunableOp reduced the GEMM portion from `69.619 sec` to `51.852 sec` (`1.34x`). Attention and elementwise did not improve, so they became a larger percentage of the remaining transformer time.

Dominant GEMM changed as follows:

| Variant | Dominant GEMM family | Calls | Time | Avg |
|---|---|---:|---:|---:|
| AOTriton | `Cijk_Alik_Bljk_HHS_BH_MT128x128x32...WGM4` | 12600 | 66.471 sec | 5.275 ms |
| AOTriton + TunableOp | `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT96x128x32...` | 12600 | 49.107 sec | 3.897 ms |

This confirms that v2.1 is faster because TunableOp changes selected GEMM kernels, not because the same kernel became faster.

## Why I2V remains slower than T2V

Current v2.1 article-equivalent T2V transformer time is `26.794 sec`. I2V transformer time is `71.691 sec`, so I2V transformer is `2.68x` slower than T2V.

Observed reasons:

1. I2V has larger transformer GEMM shapes.
   - TunableOp table contains large sequence shapes such as `m=1904` and `m=2141`.
   - T2V-dominant shapes are smaller, such as `m=560`, `m=672`, and `m=900`.
   - The I2V image conditioning path increases token/sequence work.

2. I2V attention is much larger.
   - T2V tuned attention kernel time: about `1.256 sec`.
   - I2V tuned attention kernel time: about `9.926 sec`.
   - This is a direct sign that I2V carries heavier sequence/conditioning work beyond the shared video latent path.

3. I2V still has a large residual GEMM block after tuning.
   - Tuned I2V GEMM: `51.852 sec`.
   - Tuned T2V GEMM: `23.190 sec`.
   - The largest single I2V GEMM family still takes `49.107 sec`.

## TunableOp table observations

Persisted table:

```text
result/rocm_speed_matrix/tunableop_results0.csv
```

Validator rows lock the table to the current software/hardware stack:

- PyTorch: `2.9.1`
- HIP: `702`
- hipBLASLt: `100201-5b515cf1bc`
- rocBLAS: `5.2.0.5b515cf1bc`
- GCN arch: `gfx1151`

Shape summary from the parsed table:

| Sequence `m` | Entries | Sum of tuned op times |
|---:|---:|---:|
| 560 | 3 | 12.184 ms |
| 672 | 8 | 20.801 ms |
| 900 | 8 | 27.260 ms |
| 1904 | 4 | 17.845 ms |
| 2141 | 4 | 20.951 ms |
| 3600 | 4 | 13.289 ms |

The important I2V-like large shapes are `m=1904` and `m=2141`. Most high-time entries already select `Hipblaslt`, so the remaining improvement is less likely to come from simply enabling hipBLASLt and more likely to require better algorithms, deeper tuning, newer ROCm libraries, or reducing the amount of transformer work.

## Improvement options

### 1. Deeper I2V-specific TunableOp retune

Purpose: verify whether the current persisted table is already near-best for `m=1904` and `m=2141`.

Implementation direction:

- Use the PyTorch TunableOp Python API before the first transformer call:
  - `torch.cuda.tunable.set_max_tuning_duration(...)`
  - `torch.cuda.tunable.set_max_tuning_iterations(...)`
  - `torch.cuda.tunable.set_rotating_buffer_size(...)`
- Run an I2V-only tuning pass.
- Compare the newly selected solutions against `tunableop_results0.csv`.
- Re-run `i2v_article_warm_full` with tuning disabled and the new persisted table.

Expected impact: low to medium. The dominant GEMM already moved to a faster hipBLASLt/Tensile path, but deeper search may still improve large `m=1904/2141` shapes.

### 2. Step-count sweep

This is the highest-confidence latency lever, but it changes the generation setting and therefore is not like-for-like with the article benchmark.

Assuming transformer time scales linearly from the current `35` steps:

| Steps | Estimated total | Estimated speedup vs 35 steps |
|---:|---:|---:|
| 35 | 77.160 sec | 1.00x |
| 30 | 66.918 sec | 1.15x |
| 28 | 62.822 sec | 1.23x |
| 24 | 54.629 sec | 1.41x |
| 20 | 46.435 sec | 1.66x |
| 16 | 38.242 sec | 2.02x |
| 12 | 30.049 sec | 2.57x |
| 8 | 21.856 sec | 3.53x |

This should be measured, not only estimated, because different step counts may introduce new TunableOp shapes or change scheduler overhead.

### 3. Frame-count / latent-token sweep

Purpose: reduce sequence length rather than only reducing denoising iterations.

Candidate runs:

- `24` frames: current article-equivalent baseline
- `16` frames
- `12` frames
- `8` frames

Expected impact: medium to high. This can reduce transformer and attention work, but output duration changes and the TunableOp table must be refreshed for new shapes.

### 4. Attention backend verification

I2V tuned attention is `9.926 sec`, which is material but still smaller than GEMM.

Next checks:

- Confirm the selected SDPA path remains AOTriton-backed for I2V.
- Capture representative attention shapes.
- Test newer PyTorch/ROCm/AOTriton stack if available.

Expected impact: limited. Even removing all attention would cap improvement at about `14%` of transformer kernel time.

### 5. Elementwise fusion / compile probe

I2V tuned elementwise time is `7.743 sec`.

Next checks:

- Try transformer-only `torch.compile` smoke on I2V.
- Measure graph breaks and compile overhead.
- Only consider it for a persistent service where compile overhead can be amortized.

Expected impact: low to medium, with ROCm compatibility risk.

## Recommended next execution order

1. Implement an I2V-only step/frame sweep runner using the current `aotriton_tuned` environment.
2. Add an optional deeper TunableOp retune mode using the PyTorch TunableOp Python API.
3. Run:
   - `35` steps as control
   - `28`, `24`, `20` steps
   - `16` and `8` frames if output-duration changes are acceptable
4. For any new shape set, run one `aotriton_tunable` pass first, then re-run `aotriton_tuned` to measure no-retuning steady-state latency.
5. If step/frame sweep still leaves I2V too slow, profile attention and test `torch.compile` as secondary tracks.

## Conclusion

The I2V bottleneck is now clearly transformer-side. v2.1 already fixed the largest local issue by persisting TunableOp and selecting faster GEMM kernels, cutting I2V from `167.219 sec` to `77.160 sec`. The remaining I2V latency is mainly:

- large I2V GEMM shapes, especially `m=1904/2141`;
- larger I2V attention cost from image conditioning;
- repeated elementwise kernels;
- the unchanged `35` denoising steps.

The most practical next improvement is a measured I2V step/frame sweep plus shape-specific TunableOp refresh. Kernel/library-level improvement may still help, but current evidence says the largest user-visible speedup will come from reducing transformer work or improving GEMM algorithm selection for the exact large I2V shapes.

