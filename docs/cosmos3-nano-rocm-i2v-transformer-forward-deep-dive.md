# Cosmos3-Nano ROCm I2V transformer forward deep dive

Date: 2026-06-04

## Scope

Deep-dive the v2.1 I2V transformer-forward stage:

- Measured I2V total: `77.160 sec`
- Transformer forward: `71.691 sec`
- Transformer share: `92.9%`
- Steps: `35`
- Article-equivalent I2V: official sample image/prompt, `448x256`, `24` frames, `12` fps, guidance `1.0`

This analysis uses selected-region rocprof traces where each denoising step is wrapped by a `transformer_forward` ROCTx marker.

New analysis script:

```text
scripts/analyze_i2v_transformer_forward.py
```

New outputs:

```text
result/docs/i2v_transformer_forward_step_deep_dive_aotriton.json
result/docs/i2v_transformer_forward_step_deep_dive_aotriton_tunable.json
result/docs/t2v_transformer_forward_step_deep_dive_aotriton_tunable.json
```

## Step-level result

The tuned I2V transformer stage has `35` transformer-forward calls:

| Metric | Value |
|---|---:|
| ROCTx transformer wall time | 71.642863 sec |
| Wall time per step | 2.046939 sec |
| Kernel time | 71.047925 sec |
| Kernel time per step | 2.029941 sec |

Step-to-step variance is very small:

| Run | Per-step wall min | Avg | Max | Stddev |
|---|---:|---:|---:|---:|
| I2V AOTriton | 2.537519 sec | 2.543432 sec | 2.551716 sec | 0.003785 sec |
| I2V AOTriton + TunableOp | 2.038956 sec | 2.046939 sec | 2.054576 sec | 0.003083 sec |
| T2V AOTriton + TunableOp | 0.768155 sec | 0.772301 sec | 0.775508 sec | 0.001624 sec |

Interpretation:

- There is no single slow denoising step.
- The same expensive transformer workload is repeated evenly across all `35` steps.
- Step-count reduction should scale predictably because per-step time is stable.

## Tuned I2V per-step kernel split

| Category | Total | Per step | Calls | Calls/step | Share |
|---|---:|---:|---:|---:|---:|
| GEMM | 51.852 sec | 1.481 sec | 17850 | 510 | 72.98% |
| Attention | 9.926 sec | 0.284 sec | 2520 | 72 | 13.97% |
| Mul elementwise | 2.779 sec | 0.079 sec | 33110 | 946 | 3.91% |
| Copy/fill | 2.411 sec | 0.069 sec | 29659 | 847 | 3.39% |
| Pow | 1.298 sec | 0.037 sec | 10150 | 290 | 1.83% |
| Other elementwise | 1.142 sec | 0.033 sec | 35665 | 1019 | 1.61% |
| Reduce mean | 1.042 sec | 0.030 sec | 10150 | 290 | 1.47% |
| SiLU | 0.584 sec | 0.017 sec | 2555 | 73 | 0.82% |

The dominant single kernel family is:

| Kernel family | Calls | Total | Per step | Avg |
|---|---:|---:|---:|---:|
| `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT96x128x32...` | 12600 | 49.107 sec | 1.403 sec | 3.897 ms |

This one GEMM family alone is about `63.6%` of the full I2V wall time (`49.107 / 77.160`) and about `69.1%` of transformer kernel time.

## What TunableOp already fixed

I2V AOTriton baseline vs TunableOp:

| Category | AOTriton | Tuned | Change |
|---|---:|---:|---:|
| Kernel time / step | 2.526 sec | 2.030 sec | -0.496 sec/step |
| GEMM / step | 1.989 sec | 1.481 sec | -0.508 sec/step |
| Attention / step | 0.277 sec | 0.284 sec | +0.006 sec/step |
| Elementwise/copy groups | roughly unchanged | roughly unchanged | no material improvement |

Total savings:

- GEMM saved about `17.77 sec`.
- The full transformer stage improved from `89.021 sec` to `71.691 sec`.
- The improvement is almost entirely GEMM kernel selection.

This confirms the current v2.1 optimization is working, but it also shows why it does not solve the whole I2V problem: after tuning, there is still `51.852 sec` of GEMM and `9.926 sec` of attention.

## Why I2V transformer is much heavier than T2V

Tuned I2V vs tuned T2V:

| Category | I2V tuned | T2V tuned | I2V/T2V |
|---|---:|---:|---:|
| Kernel time / step | 2.030 sec | 0.759 sec | 2.68x |
| GEMM | 51.852 sec | 23.190 sec | 2.24x |
| Attention | 9.926 sec | 1.256 sec | 7.90x |
| Mul | 2.779 sec | 0.837 sec | 3.32x |
| Copy/fill | 2.411 sec | 0.477 sec | 5.06x |
| Pow | 1.298 sec | 0.177 sec | 7.32x |
| Reduce mean | 1.042 sec | 0.149 sec | 7.01x |

The number of attention calls is the same: `2520` total, or `72` calls/step. The difference is cost per call:

| Mode | Attention calls | Attention avg |
|---|---:|---:|
| T2V tuned | 2520 | 0.499 ms |
| I2V tuned | 2520 | 3.939 ms |

This is the clearest evidence that I2V is not slower because it has more denoising steps or more attention layers. It is slower because each layer is processing larger/heavier I2V sequence work, most likely from image conditioning tokens and the larger I2V transformer shapes.

The TunableOp table also shows large I2V-related sequence shapes:

- `m=1904`
- `m=2141`

These are larger than the T2V-dominant shapes such as:

- `m=560`
- `m=672`
- `m=900`

## Improvement ceiling from each component

Starting point: `77.160 sec` total, with `71.691 sec` transformer.

Approximate upper bounds:

| Hypothetical improvement | Saved time | New total |
|---|---:|---:|
| Dominant GEMM 1.2x faster | 8.18 sec | 68.98 sec |
| Dominant GEMM 1.5x faster | 16.37 sec | 60.79 sec |
| Dominant GEMM 2.0x faster | 24.55 sec | 52.61 sec |
| All GEMM 2.0x faster | 25.93 sec | 51.23 sec |
| Attention 2.0x faster | 4.96 sec | 72.20 sec |
| Elementwise/copy groups 2.0x faster | about 3.9 sec | about 73.3 sec |

Implications:

- Attention optimization alone cannot close the gap.
- Elementwise fusion alone cannot close the gap.
- Even a `2x` improvement to all remaining GEMM would still leave the run around `51 sec`.
- Reaching the article's `17 sec` with the same `35` steps would require a much larger transformer speedup than local kernel selection alone has shown so far.

## Best next actions

### 1. I2V step-count sweep

Because per-step time is stable, reducing steps is the most predictable latency control.

Estimated totals from current v2.1:

| Steps | Estimated total |
|---:|---:|
| 35 | 77.160 sec |
| 28 | 62.822 sec |
| 24 | 54.629 sec |
| 20 | 46.435 sec |
| 16 | 38.242 sec |
| 12 | 30.049 sec |
| 8 | 21.856 sec |

This is not an article-equivalent benchmark anymore, but it is the strongest user-facing latency lever.

### 2. I2V shape-specific deeper TunableOp retune

Target the large I2V GEMM shapes, especially `m=1904/2141`.

Use PyTorch TunableOp Python API before first transformer call:

```python
torch.cuda.tunable.set_max_tuning_duration(...)
torch.cuda.tunable.set_max_tuning_iterations(...)
torch.cuda.tunable.set_rotating_buffer_size(...)
```

Then compare the new tuning table against:

```text
result/rocm_speed_matrix/tunableop_results0.csv
```

Expected gain is uncertain. The current dominant GEMM is already much better than baseline, but this is the only remaining no-quality-change local optimization with plausible impact.

### 3. Frame/token reduction sweep

Run I2V with fewer frames, for example:

- `24` frames: current control
- `16` frames
- `12` frames
- `8` frames

This should reduce transformer sequence work, but it changes output duration and may require a fresh TunableOp table for new shapes.

### 4. Attention backend check

Attention is `9.926 sec`, and I2V attention is `7.9x` T2V attention despite the same call count. It is worth checking, but it is second-order after GEMM/work reduction.

Checks:

- Confirm the I2V path is always using the expected AOTriton `attn_fwd` kernel.
- Capture attention shapes if possible.
- Compare with a newer ROCm/PyTorch/AOTriton stack if available.

### 5. Elementwise fusion probe

Elementwise/copy work is spread across many small kernels. It may benefit from `torch.compile`, but ROCm graph breaks and compile overhead are risks.

This is only worth pursuing after:

- step/frame sweep is measured;
- deeper TunableOp does not produce a meaningful gain;
- the target is a persistent service where compile overhead can be amortized.

## Conclusion

The `71.691 sec` transformer-forward stage is not a warmup anomaly and not a single bad step. It is a stable `35 x ~2.047 sec` loop.

The current bottleneck inside transformer forward is:

1. `~1.48 sec/step` GEMM, dominated by one GEMM family at `~1.40 sec/step`.
2. `~0.28 sec/step` attention, much heavier than T2V because I2V has larger sequence/conditioning work.
3. `~0.26 sec/step` scattered elementwise/copy/reduce kernels.

The practical next move is to measure a step/frame sweep and run an I2V-specific deeper TunableOp retune. Kernel-only improvements can still help, but the current numbers show that article-level latency would require either substantially faster ROCm transformer kernels for `gfx1151` or less transformer work per generation.

