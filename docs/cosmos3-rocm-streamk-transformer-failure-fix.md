# Cosmos3 ROCm Stream-K Transformer Failure Fix

Date: 2026-06-04

## Problem

The Stream-K probe initially looked like an SDPA/attention failure:

```text
RuntimeError: Expected iter != ops_.end() to be true, but got false.
```

However, a no-allow-error stack trace showed the actual failure point:

```text
diffusers/models/transformers/transformer_cosmos3.py
k_und = attn.to_k(und_seq)
torch.nn.modules.linear.Linear.forward
F.linear(input, self.weight, self.bias)
```

So the failure is not transformer attention itself. It is a transformer Linear/GEMM failure.

## Cause

The failing variant was:

```text
aotriton_tuned_streamk
```

It combined:

```text
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_FILENAME=/workspace/result/rocm_speed_matrix/tunableop_results%d.csv
TENSILE_SOLUTION_SELECTION_METHOD=2
ROCBLAS_USE_HIPBLASLT=1
```

The existing TunableOp table was collected for the normal tuned path, not for the Stream-K/Origami selection mode.

When reused with Stream-K enabled, PyTorch/TunableOp selected a GEMM path that failed in `F.linear`.

## Implemented Fix

Added a safe Stream-K variant:

```text
aotriton_streamk_safe
```

Runner file:

```text
scripts/run_rocm_speed_matrix.py
```

Safe env:

```text
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PYTORCH_TUNABLEOP_ENABLED=0
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_RECORD_UNTUNED=0
TENSILE_SOLUTION_SELECTION_METHOD=2
ROCBLAS_USE_HIPBLASLT=1
```

This avoids reusing the incompatible TunableOp table with Stream-K.

## Transformer Attention Fallback

An additional diagnostic fallback was implemented:

```text
--cosmos3-transformer-attention-fallback
```

It wraps Cosmos3 `dispatch_attention_fn` and falls back to a local matmul/softmax implementation only if the known `ops_.end()` RuntimeError occurs in attention.

The stack trace showed the current failure is in `F.linear`, so this fallback is not the main fix for the Stream-K failure. It remains useful as a diagnostic guard for true attention-side failures.

## Results

Command:

```text
COSMOS3_ROCM_IMAGE=cosmos3-rocm72-diffusers:local \
COSMOS3_DIFFUSERS_INSTALL=true \
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_streamk_safe \
  --case i2v_article_transformer_streamk_probe \
  --execute
```

Artifact:

```text
result/rocm_speed_matrix/aotriton_streamk_safe/i2v_article_transformer_streamk_probe/summary.json
```

Comparison:

| Variant | Transformer calls | Transformer time | Status |
|---|---:|---:|---|
| `aotriton_tuned` | 35 | `70.769 sec` | OK |
| `aotriton_tuned_streamk` | 1 | `0.174 sec` | Fails in `F.linear` |
| `aotriton_streamk` | 35 | `78.071 sec` | OK |
| `aotriton_streamk_safe` | 35 | `78.192 sec` | OK |

## Conclusion

The Stream-K failure is fixed by not combining Stream-K with the existing TunableOp table.

But Stream-K is not a performance improvement on this workload:

```text
aotriton_tuned baseline:       70.769 sec
aotriton_streamk_safe:         78.192 sec
```

Stream-K safe is about `10.1%` slower than the current tuned baseline.

Decision:

- Do not use `aotriton_tuned_streamk`.
- Keep `aotriton_streamk_safe` only as a diagnostic variant.
- Current best remains `aotriton_tuned` with persisted TunableOp table.

