# Cosmos3 I2V Exact GEMM Tuning and Code Rewrite Deep Dive

Date: 2026-06-04

## Current Baseline

Best stable I2V article-equivalent runtime:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_article_runtime_no_profile/summary.json
76.893 sec
```

Profiled run:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_article_warm_full/summary.json
77.160 sec
transformer_forward: 71.691 sec / 35 calls
vae_decode:          4.195 sec
```

Transformer kernel breakdown:

```text
result/docs/i2v_transformer_forward_step_deep_dive_aotriton_tunable.json
```

| Category | Time | Share |
|---|---:|---:|
| GEMM | `51.852 sec` | `72.98%` |
| Attention | `9.926 sec` | `13.97%` |
| Elementwise/copy/reduce tail | about `9 sec` | about `13%` |

Therefore the remaining primary target is transformer GEMM.

## Already Excluded

The following have already been tried and should not be repeated as the next fix:

- global Stream-K / Origami selection
  - `aotriton_streamk_safe`: `78.192 sec / 35 transformer calls`
  - slower than `aotriton_tuned`: `70.769 sec / 35 calls`
- Stream-K + existing TunableOp table
  - fails in transformer `F.linear`
- attention backend switching
  - default/flash already near-best
  - math is much slower
- local gate/up or QKV fusion
  - synthetic fused variants were slower
- deeper TunableOp table
  - no improvement over persisted `tunableop_results0.csv`

## Exact GEMM Evidence

The persisted TunableOp table shows the high-time I2V GEMM shapes:

```text
result/rocm_speed_matrix/tunableop_results0.csv
result/docs/tunableop_results0_analysis.json
```

Top relevant shapes:

| Shape | Likely module family | Tuned time |
|---|---|---:|
| `tn_4096_2141_12288` | MLP down, gen/full sequence | `9.740 ms` |
| `tn_4096_1904_12288` | MLP down, und/text prefix | `8.235 ms` |
| `tn_12288_2141_4096` | MLP gate/up, gen/full sequence | `7.941 ms` |
| `tn_12288_1904_4096` | MLP gate/up, und/text prefix | `6.725 ms` |
| `tn_4096_2141_4096` | attention/output 4096 GEMM | `2.754 ms` |
| `tn_4096_1904_4096` | attention/output 4096 GEMM | `2.489 ms` |

These shapes already select hipBLASLt solutions in the current TunableOp table. That is why simply enabling hipBLASLt or Stream-K globally did not help.

## New Diagnostic Added

Added benchmark options:

```text
--linear-profile
--linear-profile-top
--transformer-input-profile
--transformer-input-profile-calls
```

File:

```text
scripts/benchmark_classmethod_article_t2v_i2v_rocm.py
```

Artifacts:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_linear_profile_1step_top600/summary.json
result/rocm_speed_matrix/aotriton_tuned/i2v_transformer_input_profile_2step/summary.json
```

## Linear Module Profile

I2V 1-step diagnostic:

```text
transformer_forward: 2.308 sec / 1 call
linear_profile_total: 1.537733 sec
linear modules: 508
```

Family aggregation:

| Module family | 1-step time | Calls |
|---|---:|---:|
| `und_mlp.down_proj 12288->4096 m=1904` | `0.3243 sec` | 36 |
| `und_mlp.up_proj 4096->12288 m=1904` | `0.2539 sec` | 36 |
| `und_mlp.gate_proj 4096->12288 m=1904` | `0.2483 sec` | 36 |
| `gen_mlp.down_proj 12288->4096 m~2141` | `0.1125 sec` | 36 |
| `und_attn.to_q 4096->4096 m=1904` | `0.1095 sec` | 36 |
| `gen_mlp.gate_proj 4096->12288 m~2141` | `0.1029 sec` | 36 |
| `gen_mlp.up_proj 4096->12288 m~2141` | `0.1013 sec` | 36 |
| `und_attn.to_out 4096->4096 m=1904` | `0.0956 sec` | 36 |
| `gen_attn.to_add_out 4096->4096 m~2141` | `0.0371 sec` | 36 |
| `gen_attn.add_q 4096->4096 m~2141` | `0.0365 sec` | 36 |
| `und_attn.to_k 4096->1024 m=1904` | `0.0346 sec` | 36 |
| `und_attn.to_v 4096->1024 m=1904` | `0.0186 sec` | 36 |

The important observation is that the `und` branch dominates module-level Linear time.

`und_mlp.*` alone is:

```text
0.3243 + 0.2539 + 0.2483 = 0.8265 sec / step diagnostic
```

`und_attn` projection/output Linear adds:

```text
0.1095 + 0.0956 + 0.0346 + 0.0186 = 0.2583 sec / step diagnostic
```

So `und` Linear work is about:

```text
1.0848 sec of 1.5377 sec linear-profile total
```

The absolute values include profiler synchronization overhead, but the module ranking is actionable.

## Transformer Input Stability

I2V 2-step input diagnostic:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_transformer_input_profile_2step/summary.json
```

Stable across first two transformer calls:

| Input | Stable |
|---|---|
| `input_ids` | yes |
| `text_indexes` | yes |
| `position_ids` | yes |
| `sequence_length` | yes, `2576` |
| `und_len` | yes, `1904` |
| `vision_sequence_indexes` | yes |
| `vision_mse_loss_indexes` | yes |
| `vision_noisy_frame_indexes` | yes |

Changing across steps:

| Input | Change |
|---|---|
| `vision_tokens` | changes |
| `vision_timesteps` | changes |

Interpretation:

- The `und` prefix is text-only and stable for I2V.
- The generated/noisy vision part changes per denoising step.
- This makes `und` branch caching a plausible quality-preserving code rewrite.

## Candidate A: hipBLASLt/TensileLite Offline Tuning

Goal:

Tune exact GEMM shapes rather than enabling Stream-K globally.

Targets:

```text
tn_4096_1904_12288
tn_12288_1904_4096
tn_4096_2141_12288
tn_12288_2141_4096
tn_4096_1904_4096
tn_4096_2141_4096
```

Expected upside:

- GEMM total is `51.852 sec`.
- A 10% GEMM improvement gives about `5.2 sec`.
- A 20% GEMM improvement gives about `10.4 sec`.

Required work:

1. Extract exact hipBLASLt descriptors from rocBLAS/hipBLASLt logs for the target shapes.
2. Run TensileLite tuning for `gfx1151`.
3. Build a tuned hipBLASLt/Tensile library or solution database.
4. Validate synthetic GEMM speed.
5. Validate I2V transformer-only `35` calls.
6. Validate full I2V output.

Risk:

- High build/tuning effort.
- Current TunableOp already selects hipBLASLt for most important shapes, so gains require better algorithms, not just enabling hipBLASLt.

## Candidate B: Code Rewrite - Cache Stable `und` Branch Across Denoising Steps

Goal:

Avoid recomputing text-only `und` branch for every denoising step.

Current per layer:

```text
und_norm = input_layernorm(und_seq)
gen_norm = input_layernorm_moe_gen(gen_seq)
und_attn_out, gen_attn_out = self.self_attn(und_norm, gen_norm, rotary_emb)
residual_und = und_seq + und_attn_out
residual_gen = gen_seq + gen_attn_out
mlp_out_und = self.mlp(post_attention_layernorm(residual_und))
mlp_out_gen = self.mlp_moe_gen(post_attention_layernorm_moe_gen(residual_gen))
```

Proposed cached execution:

1. On first denoising step, compute normally and cache per layer:
   - next `und_seq`
   - `k_und`
   - `v_und`
   - any required rotary-applied `k_und`
2. On later steps:
   - reuse cached `und_seq` progression
   - skip `und` self-attention and `und` MLP
   - compute only `gen` pathway
   - concatenate cached `k_und/v_und` with current `k_gen/v_gen` for full generation attention

Expected upside:

The diagnostic Linear profile suggests `und` Linear work is about 70% of Linear module time:

```text
und Linear diagnostic: 1.0848 sec
all Linear diagnostic: 1.5377 sec
```

If this maps proportionally to the `51.852 sec` GEMM total, the theoretical upper bound is large. A conservative target is:

```text
10-25 sec transformer reduction
```

This is higher upside than offline tuning, but also higher implementation risk.

Memory estimate:

- `und_seq`: `1904 * 4096 * fp16 ~= 15.6 MB` per layer state
- 36 layers plus initial/final states: roughly `0.6 GB`
- `k_und/v_und`: `1904 * 8 * 128 * 2 tensors * fp16 ~= 7.8 MB` per layer
- 36 layers: roughly `0.28 GB`
- Total cache budget: below about `1 GB`, acceptable on a 120 GB GPU.

Correctness requirements:

- Cache key must include:
  - `input_ids`
  - `text_indexes`
  - `position_ids[:und_len]`
  - `und_len`
  - model dtype/device
  - relevant layer weights identity/version
- Disable cache if any `und` input changes.
- Keep output settings unchanged.
- Verify generated latent/video output equivalence or near-equivalence for fixed seed.

Implementation options:

1. Prototype monkeypatch in benchmark runner.
   - Fastest to validate.
   - Risky but isolated.
2. Patch local Diffusers `transformer_cosmos3.py`.
   - Cleaner for repeated testing.
   - Requires rebuilding `cosmos3-rocm72-diffusers:local`.
3. Upstream-quality implementation.
   - Add explicit cache object to `Cosmos3OmniTransformer.forward`.
   - More work, but maintainable.

## Recommendation

Proceed in this order:

1. Implement an `und` branch cache prototype.
   - This is the largest code-level opportunity not yet tested.
   - It is supported by measured input stability and module-level Linear dominance.
2. In parallel or after that, prepare hipBLASLt/TensileLite exact-shape tuning.
   - This is the kernel-level route.
   - It has lower model-code risk but higher library build/tuning effort.

Decision point:

- If `und` cache reduces transformer-only I2V below `60 sec`, continue to full I2V validation.
- If it does not, move to exact-shape hipBLASLt/TensileLite tuning.

