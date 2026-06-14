# Cosmos3 I2V Transformer GEMM Implementation Plan

Date: 2026-06-04

## Goal

Reduce I2V `transformer_forward` time without changing quality settings.

Fixed comparison settings:

```text
I2V official sample image/prompt
448x256
24 frames
35 steps
guidance 1.0
seed 203
```

Current best:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_article_runtime_no_profile/summary.json
end-to-end: 76.893 sec
```

Profiled:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_article_warm_full/summary.json
end-to-end:          77.160 sec
transformer_forward: 71.691 sec
vae_decode:           4.195 sec
```

Transformer kernel breakdown:

```text
GEMM:      51.852 sec / 72.98%
attention:  9.926 sec / 13.97%
tail:      about 9 sec
```

## Excluded Implementation Routes

Already tested and not selected:

- global Stream-K / Origami
  - safe variant: `78.192 sec / 35 calls`, slower than baseline
- Stream-K + existing TunableOp table
  - fails in transformer `F.linear`
- local gate/up fusion
  - synthetic fusion was slower
- local QKV fusion
  - synthetic fusion was slower
- deeper TunableOp table
  - no improvement
- attention backend switching
  - no useful speedup

## GEMM Implementation Targets

Measured Linear profile, 1-step diagnostic:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_linear_profile_1step_top600/summary.json
```

Top module families:

| Module family | 1-step diagnostic time | Calls |
|---|---:|---:|
| `und_mlp.down_proj 12288->4096 m=1904` | `0.3243 sec` | 36 |
| `und_mlp.up_proj 4096->12288 m=1904` | `0.2539 sec` | 36 |
| `und_mlp.gate_proj 4096->12288 m=1904` | `0.2483 sec` | 36 |
| `gen_mlp.down_proj 12288->4096 m~2141` | `0.1125 sec` | 36 |
| `und_attn.to_q 4096->4096 m=1904` | `0.1095 sec` | 36 |
| `gen_mlp.gate_proj 4096->12288 m~2141` | `0.1029 sec` | 36 |
| `gen_mlp.up_proj 4096->12288 m~2141` | `0.1013 sec` | 36 |
| `und_attn.to_out 4096->4096 m=1904` | `0.0956 sec` | 36 |

High-value GEMM shapes from TunableOp table:

```text
tn_4096_1904_12288
tn_12288_1904_4096
tn_4096_2141_12288
tn_12288_2141_4096
tn_4096_1904_4096
tn_4096_2141_4096
```

## Candidate 1: `und` Branch Cache

Type:

```text
model implementation rewrite
```

Rationale:

2-step transformer input profile showed:

Stable:

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

Changing:

```text
vision_tokens
vision_timesteps
```

This means the `und` prefix is text-only and stable across denoising steps, while `gen` vision tokens change.

Implementation sketch:

1. First transformer call computes normally.
2. Cache per layer:
   - post-layer `und_seq`
   - attention `k_und`
   - attention `v_und`
   - rotary-applied `k_und`
3. Later transformer calls:
   - skip `und` self-attention
   - skip `und` MLP
   - compute only `gen` branch
   - use cached `k_und/v_und` for gen full attention

Expected gain:

```text
10-25 sec transformer reduction if cache is valid
```

Risk:

- Must prove output equivalence.
- Cosmos3 layer updates `und_seq` layer by layer; cached state must be per-layer, not just input-level.
- Cache invalidation must be strict.

Validation:

1. transformer-only 2-step latent comparison.
2. transformer-only 35-step timing.
3. full I2V output comparison with fixed seed.

## Candidate 2: Split `und` and `gen` MLP GEMM Scheduling

Type:

```text
model implementation rewrite
```

Current code computes separate MLPs:

```python
mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))
```

The two MLPs are separate modules with separate weights, so naive concatenation is not valid.

Possible implementation:

- Keep weights separate.
- Reorder execution to group same-shape GEMMs across layers is not valid because each layer depends on the previous layer output.
- Within one layer, run independent `und` and `gen` GEMMs through batched/grouped GEMM only if hipBLASLt grouped GEMM is accessible from PyTorch or a custom extension.

Expected gain:

```text
low to medium
```

Reason:

- Python launch overhead is not dominant.
- GEMM kernels are already large enough.
- Custom grouped GEMM may reduce launch count but not necessarily improve total GEMM throughput.

Recommendation:

Do not implement before `und` cache and exact-shape tuning.

## Candidate 3: Custom Linear Wrapper for Exact GEMM Shapes

Type:

```text
model implementation + backend selection
```

Idea:

Replace selected `nn.Linear` calls with a wrapper that routes only exact high-time shapes to a custom backend:

```text
Cosmos3LinearGemmSelector
```

Target modules:

```text
layers.*.mlp.down_proj
layers.*.mlp.up_proj
layers.*.mlp.gate_proj
layers.*.mlp_moe_gen.down_proj
layers.*.mlp_moe_gen.up_proj
layers.*.mlp_moe_gen.gate_proj
layers.*.self_attn.to_q
layers.*.self_attn.to_out
layers.*.self_attn.add_q_proj
layers.*.self_attn.to_add_out
```

Routing options:

1. PyTorch `F.linear`
   - baseline
2. `torch.matmul` with pre-transposed cached weight
   - easy to test
   - may avoid some transpose/layout overhead
3. custom HIP extension calling hipBLASLt matmul with fixed algorithm
   - highest control
   - requires native extension and algorithm IDs
4. TensileLite tuned library
   - preferred for production if exact tuning succeeds

Expected gain:

```text
matmul/pretranspose wrapper: likely small, 0-3 sec
custom fixed hipBLASLt/Tensile: possible 5-10 sec if better algorithms exist
```

Risk:

- PyTorch `F.linear` already maps to GEMM efficiently.
- Replacing it can break TunableOp selection or produce slower kernels.
- Custom extension must preserve dtype/layout/accuracy.

Minimal test:

Add an opt-in wrapper for only:

```text
layers.*.mlp.down_proj
```

because this is the largest family in the module profile.

Compare:

```text
1-step linear profile
35-step transformer-only
full I2V
```

## Candidate 4: Weight Layout Prepacking

Type:

```text
model implementation rewrite
```

Idea:

Precompute layout-transformed weights for the target Linear modules so GEMM input uses a more favorable layout.

Example:

```text
store weight_t = weight.t().contiguous()
call input.matmul(weight_t)
```

Expected gain:

```text
small unless current F.linear path pays repeated layout conversion
```

Evidence:

The TunableOp shapes are already `TN`, consistent with transposed weight usage. This suggests PyTorch is not doing an obvious repeated transpose copy for the main GEMMs.

Recommendation:

Test only as a small probe before larger implementation.

## Candidate 5: Exact-Shape hipBLASLt/TensileLite Tuning

Type:

```text
backend/library implementation
```

This is still the main kernel-level route.

Plan:

1. Capture exact hipBLASLt descriptors for the target shape families.
2. Run TensileLite tuning for `gfx1151`.
3. Build tuned library/config.
4. Validate synthetic GEMM.
5. Validate transformer-only 35-step.
6. Validate full I2V.

Expected gain:

```text
GEMM 10% improvement -> about 5.2 sec
GEMM 20% improvement -> about 10.4 sec
```

Risk:

- The current TunableOp table already selects hipBLASLt for major shapes.
- Gains require better algorithms, not just enabling another backend.

## Recommended Execution Order

1. `und` branch cache prototype
   - highest code-level upside
   - supported by measured input stability
2. `mlp.down_proj` custom Linear wrapper probe
   - narrow, reversible, measures whether implementation-level GEMM routing can beat `F.linear`
3. exact-shape hipBLASLt/TensileLite tuning
   - high effort, strong kernel-level target
4. grouped GEMM / larger custom extension
   - only if the wrapper probe shows routing can help

## Acceptance Criteria

Any implementation must meet:

```text
same I2V input image/prompt
same resolution/frames/steps/guidance/seed
no quality-reducing shortcuts
transformer-only 35 calls succeeds
full I2V output succeeds
result faster than 76.893 sec end-to-end or 71.691 sec transformer
```

Minimum useful improvement:

```text
transformer_forward <= 65 sec
```

Strong improvement:

```text
transformer_forward <= 60 sec
```

