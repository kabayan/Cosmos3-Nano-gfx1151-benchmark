# Cosmos3-Nano ROCm T2V/I2V transformer improvement plan

Date: 2026-06-03

## Current bottleneck

Latest custom MIOpen retest:

| Mode | Total | Transformer forward | VAE decode | Transformer share |
|---|---:|---:|---:|---:|
| T2V | 46.861 sec | 41.514 sec | 4.127 sec | 88.6% |
| I2V | 93.976 sec | 88.560 sec | 4.144 sec | 94.2% |

The custom MIOpen Conv3D change does not materially affect T2V/I2V because the measured steady-state path is dominated by diffusion transformer forward, not VAE decode.

## Existing evidence

Transformer-only rocprof for I2V showed:

| Category | Time | Share |
|---|---:|---:|
| GEMM / Tensile | 69.619 sec | 78.74% |
| Attention `attn_fwd` | 9.702 sec | 10.97% |
| Elementwise | 7.603 sec | 8.60% |
| Other | 1.038 sec | 1.17% |
| Copy/fill | 0.456 sec | 0.52% |

Therefore the improvement priority is:

1. GEMM backend and GEMM tuning.
2. Attention backend.
3. Elementwise fusion / compile.
4. Algorithmic reduction of steps, frames, or latent tokens.

## Improvement options

### 1. AOTriton + TunableOp as the near-term target

Previous `aotriton_tunable` result for I2V:

| Case | AOTriton baseline | AOTriton + TunableOp | Improvement |
|---|---:|---:|---:|
| I2V measured | 94.219 sec | 77.083 sec | 1.22x |
| I2V transformer | 88.753 sec | 71.621 sec | 1.24x |

This is the strongest local evidence for a T2V/I2V transformer improvement.

Problem: PyTorch did not persist `tunableop_results.csv` in this environment, so the exact tuned table was not captured. The result is usable as a performance signal, but not yet stable as a reproducible optimization artifact.

Next actions:

- Add explicit `t2v_article_warm_full` and `t2v_article_transformer_rocprof` runner cases.
- Re-run `aotriton_tunable` for T2V and I2V separately.
- Capture selected-region rocprof for `aotriton_tunable` to confirm GEMM kernel names or durations changed.
- Investigate why `PYTORCH_TUNABLEOP_FILENAME` is not persisted.

Expected impact:

- If T2V follows I2V behavior, T2V could drop from about `46.9 sec` to roughly `38-40 sec`.
- I2V has already shown a drop to about `77 sec`.

### 2. hipBLASLt / Tensile kernel selection

The top transformer kernels are Tensile GEMM kernels. Improvements require better kernel selection for the exact Cosmos3 GEMM shapes on `gfx1151`.

Next actions:

- Extract GEMM shapes from rocBLAS/hipBLASLt logging during selected transformer windows.
- Compare selected kernels between `aotriton` and `aotriton_tunable`.
- Check whether hipBLASLt is being used for the dominant GEMMs or whether rocBLAS/Tensile is selected.
- If hipBLASLt is not used for the dominant shapes, test enabling/disabling available backend flags and compare.

Expected impact:

- Medium to high if the current GEMM kernels are suboptimal for `gfx1151`.
- This is the most direct path because GEMM is about 79% of I2V transformer time.

### 3. Attention backend verification

Attention is secondary but still material:

- I2V attention: about `9.7 sec`
- T2I attention: about `12.6 sec`

AOTriton already enables fused SDPA paths that failed without it, but the current `attn_fwd` kernel still accounts for about 11% of I2V transformer time.

Next actions:

- Verify that the pipeline is always using the AOTriton-backed fused SDPA path during T2V/I2V.
- Compare `torch.backends.cuda.sdp_kernel` / `sdpa_kernel()` forced modes on a small representative attention shape.
- Capture attention shape and test whether a newer PyTorch/ROCm stack changes `attn_fwd` kernel time.

Expected impact:

- Limited compared with GEMM.
- Even eliminating attention entirely would only cap I2V improvement at about 11%, so this is second priority.

### 4. torch.compile / fusion probe

Elementwise kernels are about 8-9% of transformer time. There are many repeated small kernels: pow, copy, mul, mean/reduce, add, SiLU, cat.

Next actions:

- Test `torch.compile()` on the transformer module only.
- Use a smoke setting first because compile time and graph breaks may be large.
- If it runs, compare transformer-only stage time and kernel count.

Risks:

- ROCm graph breaks or unsupported ops may make this unusable.
- Compile overhead may be too high unless the process is long-lived.

Expected impact:

- Low to medium.
- Useful only after GEMM tuning, or in a persistent service where compile overhead is amortized.

### 5. Reduce transformer work

This is the most reliable way to reduce latency, but changes output conditions.

Levers:

- Steps: `35 -> 28`, `24`, `20`
- Frames: `24 -> 16`
- Resolution: keep article comparison at `448x256`; reduce only for non-article modes
- Guidance: already `1.0`, so no major gain there

Expected impact:

- Step reduction is roughly linear for transformer time.
- For T2V, `35 -> 24` could reduce transformer from about `41.5 sec` to about `28.5 sec`, with total near `34 sec`.
- For I2V, `35 -> 24` could reduce transformer from about `88.6 sec` to about `60.7 sec`, with total near `66 sec`.

This is not a like-for-like article benchmark improvement, but it is the highest-confidence user-facing latency control.

## Recommended execution order

1. Add missing T2V-only runner and transformer-only rocprof cases.
2. Re-run `aotriton_tunable` for both T2V and I2V.
3. Run selected-region rocprof for `aotriton_tunable` and compare GEMM kernels against `aotriton`.
4. Capture rocBLAS/hipBLASLt logging for transformer-only windows.
5. Test `torch.compile()` transformer-only smoke.
6. Run a steps/frame quality-speed sweep as a separate non-article optimization track.

## Decision

The next practical improvement should be `AOTriton + TunableOp` reproducibility and GEMM kernel selection analysis. This is the only path that has already shown a material I2V speedup on this machine without changing output settings.

MIOpen Conv3D work should continue for Policy Model, but it is not the main T2V/I2V path.
