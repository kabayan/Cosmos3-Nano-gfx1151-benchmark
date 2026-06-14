# Cosmos3-Nano ROCm I2V implementation improvement review

Date: 2026-06-04

## Scope

Consider implementation-level changes that preserve the article-equivalent I2V quality settings:

- `448x256`
- `24` frames
- `35` steps
- guidance `1.0`
- same official sample image/prompt
- same seed `203`

Step/frame/resolution reduction is excluded because it changes quality and breaks comparison.

## Implemented changes

Added runtime options to:

```text
scripts/benchmark_classmethod_article_t2v_i2v_rocm.py
```

New options:

```text
--inference-mode
--disable-progress-bar
```

`--inference-mode` wraps the pipeline call in `torch.inference_mode()`. Diffusers generally already uses no-grad style execution, but this is a safe quality-preserving runtime option.

`--disable-progress-bar` removes tqdm update overhead.

Added a quality-fixed runtime benchmark case to:

```text
scripts/run_rocm_speed_matrix.py
```

New case:

```text
i2v_article_runtime_no_profile
```

This case also disables `--stage-profile`, avoiding the per-`transformer.forward` `torch.cuda.synchronize()` calls used for stage timing.

## Command

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned \
  --case i2v_article_runtime_no_profile \
  --execute
```

Runtime options:

```json
{
  "inference_mode": true,
  "disable_progress_bar": true,
  "stage_profile": false
}
```

## Result

| Run | Total | Notes |
|---|---:|---|
| v2.1 tuned with stage profile | 77.160 sec | Existing stage-profile benchmark |
| Runtime no-profile | 76.893 sec | `inference_mode`, progress off, no stage timing sync |

Improvement:

- Absolute: `0.267 sec`
- Speedup: `1.003x`
- Percent: about `0.35%`

This is technically faster, but not a meaningful improvement for the `71 sec` transformer-forward bottleneck.

## Interpretation

The result shows that Python/runtime overhead is not the main issue.

The removed overhead included:

- tqdm progress updates;
- stage-profile wrappers;
- `torch.cuda.synchronize()` before and after every `transformer.forward`;
- possible no-grad vs inference-mode overhead.

Even after removing those, the total only moved from `77.160 sec` to `76.893 sec`. Therefore the large remaining cost is still GPU kernel execution inside transformer forward, not benchmark instrumentation.

## Other implementation changes considered

### Keep using v2.1 TunableOp table

Already tested. The deeper I2V-specific table did not improve steady-state runtime:

| Run | Total | Transformer |
|---|---:|---:|
| v2.1 tuned | 77.160 sec | 71.691 sec |
| deep tuned | 77.225 sec | 71.743 sec |

Decision: keep `result/rocm_speed_matrix/tunableop_results0.csv`.

### `torch.compile` on the full transformer

Potential benefit:

- Could fuse some elementwise kernels.
- Might reduce the roughly `7-9 sec` elementwise/copy/reduce tail.

Risks:

- Full Cosmos3 transformer has dynamic multimodal inputs and large memory pressure.
- ROCm graph breaks are likely.
- Compile overhead may be very large.
- Even perfect elementwise fusion would not address the dominant `51.852 sec` GEMM block.

Decision: not the next main optimization. It is a probe candidate only, not an expected major fix.

### Attention implementation changes

Current I2V attention is `9.926 sec`, about `13.97%` of transformer kernel time.

Potential benefit:

- A better AOTriton/SDPA path could help.

Limit:

- Even a `2x` attention improvement saves only about `5 sec`.
- The dominant bottleneck remains GEMM.

Decision: worth checking with newer ROCm/PyTorch/AOTriton, but not likely to close the gap locally.

### Prompt/image preprocessing changes

Not relevant for the main bottleneck:

- The measured steady-state gap is in transformer/VAE GPU execution.
- Preprocessing/postprocessing is not a material part of the measured `77 sec`.

Decision: no meaningful quality-fixed speedup expected.

## Current best quality-fixed implementation

For runtime-style I2V execution on this stack:

```text
variant: aotriton_tuned
case: i2v_article_runtime_no_profile
table: result/rocm_speed_matrix/tunableop_results0.csv
```

This gives:

```text
76.893 sec
```

The stage-profile variant should still be used when internal timing is needed:

```text
variant: aotriton_tuned
case: i2v_article_warm_full
```

This gives:

```text
77.160 sec total
71.691 sec transformer
```

## Conclusion

Implementation cleanup can recover only about `0.3 sec` without changing quality. That confirms the remaining I2V latency is not caused by Python-side benchmark code, progress rendering, or timing synchronization.

Further meaningful quality-fixed improvement requires kernel/library/model implementation work, especially:

- better GEMM algorithms for the large I2V shapes;
- newer ROCm/hipBLASLt/rocBLAS/PyTorch kernel selection;
- custom transformer kernels;
- attention backend improvements as a secondary target.

