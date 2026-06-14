# Cosmos3-Nano ROCm I2V kernel stack deep dive

Date: 2026-06-04

## Scope

Deep-dive the next level below application code:

- GEMM path: PyTorch ROCm -> hipBLASLt / rocBLAS / Tensile
- Attention path: PyTorch SDPA -> AOTriton `attn_fwd`

Quality settings remain fixed:

- I2V
- `448x256`
- `24` frames
- `35` steps
- guidance `1.0`
- official sample image/prompt

## Current measured bottleneck

v2.1 I2V:

| Component | Time |
|---|---:|
| Total | 77.160 sec |
| Transformer forward | 71.691 sec |
| GEMM | 51.852 sec |
| Attention | 9.926 sec |
| Elementwise/copy/reduce | about 9 sec |

TunableOp already changed the dominant GEMM kernel and cut I2V from v1.0 `167.219 sec` to v2.1 `77.160 sec`. Deeper TunableOp did not improve further.

## Official mechanisms relevant to this stack

### PyTorch TunableOp

PyTorch TunableOp supports ROCm GEMM tuning and validates saved tuning files against software/hardware versions. PyTorch documentation notes that changes in library, ROCm, PyTorch, or GPU versions can invalidate a tuning file. It also notes that for ROCm hipBLAS/hipBLASLt, a known solution index can be overridden by editing the tuning result value.

Source: https://docs.pytorch.org/docs/stable/cuda.tunable.html

Local status:

- `tunableop_results0.csv` is persisted and valid for:
  - PyTorch `2.9.1`
  - HIP `702`
  - hipBLASLt `100201-5b515cf1bc`
  - rocBLAS `5.2.0.5b515cf1bc`
  - arch `gfx1151`
- Deeper local TunableOp search did not find a better end-to-end solution.

### rocBLAS / hipBLASLt backend control

rocBLAS documents `ROCBLAS_USE_HIPBLASLT`:

- unset: backend selected automatically
- `0`: force Tensile
- `1`: prefer hipBLASLt, falling back to Tensile if needed

Source: https://rocmdocs.amd.com/projects/rocBLAS/en/latest/conceptual/rocblas-design-notes.html

Local implication:

- For Cosmos3 I2V GEMMs, forcing hipBLASLt can be tested.
- It is not guaranteed to improve the whole workload because PyTorch/TunableOp may already select hipBLASLt for many shapes.

### hipBLASLt Stream-K / Origami

hipBLASLt documents `TENSILE_SOLUTION_SELECTION_METHOD`:

- `0`: standard tuned libraries
- `2`: Origami with Stream-K selection

The docs say user-driven tuning can consider Stream-K kernels when this method is enabled, and Stream-K is particularly relevant for non-uniform GEMM dimensions.

Sources:

- https://rocmdocs.amd.com/projects/hipBLASLt/en/develop/how-to/how-to-use-streamk.html
- https://rocmdocs.amd.com/projects/hipBLASLt/en/develop/reference/env-variables.html

Local implication:

- Cosmos3 I2V has non-uniform large GEMMs such as `m=1904/2141`, `n=4096/12288`, `k=4096/12288`.
- Stream-K is a plausible candidate for the GEMM part.

### hipBLASLt offline / TensileLite tuning

AMD's hipBLASLt TensileLite tuning guide describes:

1. Extract/tune target GEMM shapes.
2. Generate tuning results under `3_LibraryLogic`.
3. Merge the new logic into the hipBLASLt source tree.
4. Rebuild hipBLASLt so the tuned solution is available automatically.

Source: https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html

Local implication:

- This is the credible path if PyTorch TunableOp cannot find a better solution from available kernels.
- It is external library work, not a small application patch.

### AOTriton / PyTorch SDPA

AOTriton is consumed by PyTorch through SDPA kernels. The AOTriton project documents a PyTorch compatibility matrix and notes that ROCm's PyTorch release branches can differ from upstream PyTorch and may support newer AOTriton versions.

Source: https://github.com/ROCm/aotriton

Local implication:

- Current attention uses the default SDPA/AOTriton `attn_fwd` path.
- Local backend forcing did not improve attention.
- A newer PyTorch ROCm/AOTriton stack is the plausible attention-side improvement path.

## New probes added

### GEMM environment selection probe

Added:

```text
scripts/probe_rocm_i2v_gemm_env_selection.py
```

Outputs:

```text
result/docs/rocm_i2v_gemm_env_default_probe.json
result/docs/rocm_i2v_gemm_env_streamk_probe.json
```

The probe measures representative I2V GEMM shapes with:

- default ROCm GEMM selection
- `TENSILE_SOLUTION_SELECTION_METHOD=2`
- `ROCBLAS_USE_HIPBLASLT=1`

Results:

| Shape | Default | Stream-K + hipBLASLt | Speedup |
|---|---:|---:|---:|
| `tn_4096_672_12288` | 4.619 ms | 3.908 ms | 1.18x |
| `tn_12288_672_4096` | 3.834 ms | 2.793 ms | 1.37x |
| `tn_4096_1904_12288` | 11.081 ms | 9.278 ms | 1.19x |
| `tn_12288_1904_4096` | 8.863 ms | 8.025 ms | 1.10x |
| `tn_4096_2141_12288` | 12.411 ms | 10.044 ms | 1.24x |
| `tn_12288_2141_4096` | 10.149 ms | 8.619 ms | 1.18x |

Interpretation:

- Stream-K/hipBLASLt is promising for standalone GEMM.
- This is the first quality-fixed kernel-stack lever that shows material speed on representative large GEMM shapes after TunableOp.

### Attention backend probe

Added earlier:

```text
scripts/probe_cosmos3_i2v_attention_backends.py
```

Output:

```text
result/docs/cosmos3_i2v_attention_backend_probe.json
```

Key result:

| Case | Default | Flash | Efficient | Math |
|---|---:|---:|---:|---:|
| `full_gen` | 3.848 ms | 3.777 ms | unavailable | 42.526 ms |
| `full_gen_expanded_kv` | 3.818 ms | 3.821 ms | 3.841 ms | 42.243 ms |

Interpretation:

- Default is already using the fast SDPA/AOTriton-like path.
- Forcing flash is not meaningfully faster.
- Expanding K/V heads to enable efficient attention does not help.
- Math is not usable for performance.

## Full I2V Stream-K attempt

Added runner variants:

```text
aotriton_streamk
aotriton_tuned_streamk
```

The full quality-fixed run attempted:

```bash
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned_streamk \
  --case i2v_article_runtime_no_profile \
  --execute
```

Environment:

```text
TENSILE_SOLUTION_SELECTION_METHOD=2
ROCBLAS_USE_HIPBLASLT=1
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
```

Result:

- Model load succeeded.
- VAE warmup failed inside Wan VAE attention:

```text
RuntimeError: Expected iter != ops_.end() to be true, but got false.
```

The error occurred in:

```text
diffusers/models/autoencoders/autoencoder_kl_wan.py
F.scaled_dot_product_attention(q, k, v)
```

A simple standalone SDPA probe with the same environment did not fail, so the failure is likely tied to a specific Wan VAE decode tensor shape/stride/backend selection path rather than all SDPA.

Decision:

- Do not enable Stream-K globally for the full pipeline yet.
- It is promising for transformer GEMM but unsafe for the current full I2V process because VAE decode can fail.

## Current hypothesis

### GEMM

The best current hypothesis is:

> Stream-K / Origami kernel selection can improve representative Cosmos3 I2V GEMM shapes, but global process-level activation currently conflicts with the full pipeline. The next useful work is to isolate Stream-K to transformer GEMMs or run a transformer-only benchmark without VAE decode.

Possible routes:

1. Transformer-only benchmark with VAE decode bypassed.
   - Purpose: verify if Stream-K reduces transformer time.
   - Risk: not an end-to-end quality output, but valid for kernel-stack diagnosis.

2. Tune a new TunableOp table while `TENSILE_SOLUTION_SELECTION_METHOD=2`.
   - Purpose: allow TunableOp to select Stream-K candidates where beneficial.
   - Risk: the current full process still fails in VAE decode unless the failure is bypassed or fixed.

3. hipBLASLt offline/TensileLite tuning for exact I2V shapes.
   - Purpose: permanently add better solutions without relying on global env behavior.
   - Risk: high effort; requires hipBLASLt source/build/test.

### Attention

The best current hypothesis is:

> Local attention backend selection is already near-best for the current PyTorch ROCm/AOTriton stack. Meaningful improvement requires a newer or modified AOTriton/PyTorch SDPA kernel for GQA shape `(q_heads=32, kv_heads=8, head_dim=128, q_len~1904, kv_len~2141)`.

Possible routes:

1. Test newer PyTorch ROCm / AOTriton.
2. Build PyTorch ROCm against newer AOTriton.
3. Modify AOTriton kernels for the exact GQA shapes.

## Recommended next technical work

1. Implement transformer-only Stream-K probe.
   - Bypass or stub VAE decode.
   - Keep `448x256`, `24 frames`, `35 steps`.
   - Compare transformer time only:
     - `aotriton_tuned`
     - `aotriton_tuned_streamk`

2. If transformer-only Stream-K improves:
   - Investigate how to avoid the VAE SDPA failure.
   - Try disabling VAE warmup only.
   - Try forcing VAE attention backend separately if possible.
   - If needed, isolate Stream-K to transformer process or perform two-process generation/decode only as a diagnostic path.

3. If Stream-K improves transformer but cannot be made stable:
   - Move to hipBLASLt offline/TensileLite tuning for exact GEMM shapes.

4. For attention:
   - Treat as upstream stack work.
   - Test newer ROCm/PyTorch/AOTriton before writing custom kernels.

## Conclusion

The deepest local finding is new and actionable:

- Attention backend switching is not promising.
- GEMM local fusion is not promising.
- Deeper TunableOp is not promising.
- Stream-K/hipBLASLt selection is promising for representative GEMMs, but unsafe as a global full-pipeline setting in the current stack.

Therefore, the next real improvement path is not another local model rewrite. It is a kernel-stack experiment:

```text
transformer-only Stream-K validation -> stable isolation or hipBLASLt offline tuning
```

