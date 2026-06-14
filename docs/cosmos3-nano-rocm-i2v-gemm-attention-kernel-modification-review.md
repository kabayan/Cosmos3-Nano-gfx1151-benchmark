# Cosmos3-Nano ROCm I2V GEMM/attention kernel modification review

Date: 2026-06-04

## Scope

Review whether the I2V transformer bottleneck can be improved by changing GPU-kernel-facing implementation while preserving the article-equivalent quality settings:

- `448x256`
- `24` frames
- `35` steps
- guidance `1.0`
- same official I2V image/prompt

The current v2.1 transformer bottleneck is:

| Component | Time |
|---|---:|
| Transformer forward | 71.691 sec |
| GEMM | 51.852 sec |
| Attention | 9.926 sec |
| Elementwise/copy/reduce tail | about 9 sec |

## Source-level findings

Diffusers `Cosmos3OmniTransformer` uses:

- `36` decoder layers
- hidden size `4096`
- intermediate size `12288`
- `32` query heads
- `8` key/value heads
- head dim `128`

Attention implementation:

```text
diffusers.models.transformers.transformer_cosmos3.Cosmos3AttnProcessor
```

The processor calls:

```text
diffusers.models.attention_dispatch.dispatch_attention_fn
```

This ultimately calls PyTorch SDPA:

```python
torch.nn.functional.scaled_dot_product_attention(...)
```

with `enable_gqa=True`.

GEMM implementation:

- `nn.Linear`
- `F.linear`
- routed by PyTorch to hipBLASLt / rocBLAS / Tensile
- selected through PyTorch TunableOp when enabled

This means the repo can change higher-level composition and backend selection, but it does not own the actual GEMM or attention kernels.

## Implemented hooks

Added attention backend variants to:

```text
scripts/run_rocm_speed_matrix.py
```

Variants:

```text
aotriton_tuned_attn_flash
aotriton_tuned_attn_efficient
aotriton_tuned_attn_math
```

These set:

```text
DIFFUSERS_ATTN_BACKEND=_native_flash
DIFFUSERS_ATTN_BACKEND=_native_efficient
DIFFUSERS_ATTN_BACKEND=_native_math
```

Dry-run succeeded, and Diffusers recognizes `_native_flash` as an active backend.

## Attention backend probe

Added:

```text
scripts/probe_cosmos3_i2v_attention_backends.py
```

Output:

```text
result/docs/cosmos3_i2v_attention_backend_probe.json
```

Representative I2V attention shapes:

| Case | Q | K/V | Causal | GQA |
|---|---|---|---|---|
| `causal_und` | `[1, 32, 237, 128]` | `[1, 8, 237, 128]` | true | true |
| `full_gen` | `[1, 32, 1904, 128]` | `[1, 8, 2141, 128]` | false | true |
| `full_gen_expanded_kv` | `[1, 32, 1904, 128]` | `[1, 32, 2141, 128]` | false | false |

Results:

| Case | Default | Flash | Efficient | Math |
|---|---:|---:|---:|---:|
| `causal_und` | 0.226 ms | 0.215 ms | unavailable | 0.688 ms |
| `full_gen` | 3.848 ms | 3.777 ms | unavailable | 42.526 ms |
| `full_gen_expanded_kv` | 3.818 ms | 3.821 ms | 3.841 ms | 42.243 ms |

Interpretation:

- Default and forced flash are already essentially the same.
- Efficient attention is unavailable for the native GQA shape.
- Expanding K/V heads makes efficient attention available, but it is not faster.
- Math backend is much slower and must not be used.

Decision: do not replace the attention backend locally. A meaningful attention improvement would require a better ROCm GQA flash/attention kernel, not a local backend switch.

## GEMM fusion probe

Added:

```text
scripts/probe_cosmos3_i2v_gemm_fusion.py
```

Output:

```text
result/docs/cosmos3_i2v_gemm_fusion_probe.json
```

Tested two quality-preserving implementation rewrites:

1. Fuse MLP `gate_proj` and `up_proj` into one larger linear, then split.
2. Fuse attention Q/K/V projections into one larger linear, then split.

Results:

| Shape `m` | MLP separate | MLP fused gate/up | QKV separate | QKV fused |
|---:|---:|---:|---:|---:|
| 237 | 6.923 ms | 6.552 ms | 1.273 ms | 1.184 ms |
| 1904 | 23.635 ms | 26.646 ms | 3.471 ms | 4.618 ms |
| 2141 | 27.056 ms | 30.753 ms | 4.351 ms | 5.365 ms |

Interpretation:

- Fusion helps the small `m=237` pathway.
- Fusion hurts the large I2V `m=1904/2141` pathways.
- I2V is dominated by the large pathways, so this would make the real workload slower.

Decision: do not implement MLP gate/up fusion or QKV fusion for I2V on this ROCm stack.

## Can we modify GEMM kernels in this repo?

Not meaningfully.

The dominant GEMMs are generated/selected by PyTorch's ROCm backend through hipBLASLt / rocBLAS / Tensile. This repo can:

- enable TunableOp;
- persist tuning tables;
- alter model composition;
- force attention backend selection;
- test `torch.compile`.

This repo cannot directly change:

- hipBLASLt algorithm implementation;
- rocBLAS/Tensile solution kernels;
- AOTriton `attn_fwd` implementation;
- PyTorch SDPA kernel selection internals.

We already tested the local levers:

- TunableOp: meaningful v2.1 gain already captured.
- Deeper TunableOp: no further gain.
- Runtime cleanup: only `0.35%`.
- Attention backend switch: no meaningful gain.
- K/V expansion for efficient attention: no gain.
- MLP/QKV fusion: slower for large I2V shapes.

## What would be required for real kernel improvement?

### GEMM

Required work:

- improve hipBLASLt / rocBLAS / Tensile solution selection or add better algorithms for shapes such as:
  - `tn_4096_1904_12288`
  - `tn_12288_1904_4096`
  - `tn_4096_2141_12288`
  - `tn_12288_2141_4096`
  - `tn_4096_1904_4096`
  - `tn_4096_2141_4096`
- or write a custom ROCm GEMM path / PyTorch extension for these exact shapes.

Expected difficulty: high. This is ROCm library or custom kernel work, not application-level code.

### Attention

Required work:

- a faster ROCm flash attention kernel for GQA shapes:
  - Q heads `32`
  - KV heads `8`
  - head dim `128`
  - Q length around `1904`
  - K/V length around `2141`
- or improve AOTriton/PyTorch SDPA backend for this shape.

Expected difficulty: high. Current AOTriton flash path is already what default uses, and local backend switching does not improve it.

## Recommendation

Do not implement local GEMM/attention rewrites in the current repo as a performance change. The tested quality-preserving rewrites are neutral or slower.

Keep the current best runtime path:

```text
aotriton_tuned + tunableop_results0.csv
```

For further quality-fixed improvement, the next credible path is an external kernel/library track:

1. Test newer ROCm / PyTorch / hipBLASLt / AOTriton.
2. If still slow, work at the hipBLASLt/Tensile/AOTriton level for the exact I2V shapes.
3. Use this repo's probes as regression tests for any kernel-library change.

