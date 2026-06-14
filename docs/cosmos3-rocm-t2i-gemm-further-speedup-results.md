# Cosmos3 T2I GEMM further speedup exploration

## Objective

Explore additional quality-preserving GEMM speedups for T2I after the existing `aotriton_tuned` / persisted TunableOp result.

The target is the warmup-excluded T2I transformer window:

- `960x960`
- `35` steps
- `guidance_scale=1.0`
- VAE decode intentionally excluded for transformer-only probes

Current reference:

- `aotriton_tuned/t2i_transformer_only_rocprof`
- transformer: `85.029 sec / 35 calls`
- GEMM kernel total: `60.651 sec`

## Probe implementation

Added:

- `scripts/probe_t2i_gemm_rocm.py`

It benchmarks representative T2I `F.linear` shapes:

| Probe name | M | K | N | Meaning |
|---|---:|---:|---:|---|
| `t2i_mlp_up_gate_2141` | 2141 | 4096 | 12288 | main sequence MLP up/gate |
| `t2i_mlp_down_2141` | 2141 | 12288 | 4096 | main sequence MLP down |
| `t2i_attn_hidden_2141` | 2141 | 4096 | 4096 | hidden-size projection |
| `t2i_attn_qkv_2141` | 2141 | 4096 | 1024 | smaller QKV-like projection |
| `t2i_time_mlp_up_gate_900` | 900 | 4096 | 12288 | timestep-conditioned token MLP up/gate |
| `t2i_time_mlp_down_900` | 900 | 12288 | 4096 | timestep-conditioned token MLP down |
| `t2i_time_hidden_900` | 900 | 4096 | 4096 | timestep hidden projection |
| `t2i_time_qkv_900` | 900 | 4096 | 1024 | smaller timestep projection |

Artifacts:

- `result/rocm_speed_matrix/t2i_gemm_probe/baseline.json`
- `result/rocm_speed_matrix/t2i_gemm_probe/tuned_table.json`
- `result/rocm_speed_matrix/t2i_gemm_probe/streamk_safe.json`
- `result/rocm_speed_matrix/t2i_gemm_probe/rocblas_only.json`
- `result/rocm_speed_matrix/t2i_gemm_probe/streamk_tunable.json`
- `result/rocm_speed_matrix/t2i_gemm_probe/streamk_tunable0.csv`

## Microbench results

Average milliseconds per `F.linear`, 40 repeats:

| Shape | Baseline | Existing TunableOp | Stream-K safe | rocBLAS-only | Stream-K + newly tuned table | Best |
|---|---:|---:|---:|---:|---:|---|
| `t2i_mlp_up_gate_2141` | 10.117 | 8.063 | 8.733 | 10.111 | **7.698** | Stream-K tuned |
| `t2i_mlp_down_2141` | 12.383 | 10.005 | 10.034 | 12.326 | **9.900** | Stream-K tuned |
| `t2i_attn_hidden_2141` | 3.564 | 2.863 | 3.038 | 3.569 | **2.782** | Stream-K tuned |
| `t2i_attn_qkv_2141` | 0.511 | 0.515 | 0.534 | **0.508** | 0.509 | rocBLAS-only |
| `t2i_time_mlp_up_gate_900` | 3.895 | 3.153 | 4.991 | 3.868 | **3.126** | Stream-K tuned |
| `t2i_time_mlp_down_900` | 4.615 | 3.411 | 5.639 | 4.841 | **3.384** | Stream-K tuned |
| `t2i_time_hidden_900` | 1.444 | 1.146 | 1.565 | 1.441 | **1.065** | Stream-K tuned |
| `t2i_time_qkv_900` | 0.247 | 0.210 | 0.210 | 0.245 | **0.209** | Stream-K tuned |

Findings:

- Existing TunableOp table already improves the important T2I GEMMs.
- Global Stream-K without a matching tuning table is not enough; it is worse for several shapes.
- `ROCBLAS_USE_HIPBLASLT=0` is not useful for the large shapes.
- A new Stream-K-aware TunableOp table improves the microbench over the existing table for most T2I shapes.

## New Stream-K-aware TunableOp table

Created with:

```bash
TENSILE_SOLUTION_SELECTION_METHOD=2
ROCBLAS_USE_HIPBLASLT=1
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=1
PYTORCH_TUNABLEOP_FILENAME=/workspace/result/rocm_speed_matrix/t2i_gemm_probe/streamk_tunable%d.csv
```

Generated table:

- `result/rocm_speed_matrix/t2i_gemm_probe/streamk_tunable0.csv`

Key selected entries:

```text
tn_4096_2141_12288 -> Gemm_Rocblas_6176
tn_12288_2141_4096 -> Gemm_Rocblas_6260
tn_4096_2141_4096 -> Gemm_Rocblas_6260
tn_4096_900_12288 -> Gemm_Hipblaslt_6166
tn_12288_900_4096 -> Gemm_Hipblaslt_6259
tn_4096_900_4096 -> Gemm_Rocblas_6260
```

Notably, the best Stream-K-aware table is not hipBLASLt-only; several high-value shapes choose `Gemm_Rocblas_*`.

## Transformer-only validation

Command shape:

```bash
TENSILE_SOLUTION_SELECTION_METHOD=2
ROCBLAS_USE_HIPBLASLT=1
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_FILENAME=/workspace/result/rocm_speed_matrix/t2i_gemm_probe/streamk_tunable%d.csv
python3 scripts/benchmark_classmethod_article_t2i_rocm.py \
  --height 960 --width 960 --steps 35 --guidance 1.0 \
  --stage-profile \
  --measured-runs 1 \
  --abort-before-vae-decode \
  --allow-pipeline-error
```

Artifact:

- `result/rocm_speed_matrix/aotriton_streamk_tuned/t2i_transformer_only_probe/article_t2i_summary.json`

Result:

| Condition | Transformer |
|---|---:|
| Existing `aotriton_tuned` table | `85.029 sec / 35 calls` |
| Stream-K-aware newly tuned table | **`84.143 sec / 35 calls`** |

Improvement:

- `0.886 sec` faster in the transformer window
- `1.01x faster`
- about `1.0%` transformer improvement

## Interpretation

The standalone GEMM improvement is real, but it does not translate strongly to the whole T2I transformer.

Likely reasons:

- The transformer still spends about `12.3 sec` in attention and about `11.1 sec` in elementwise/reduce/copy kernels, which are unaffected by GEMM tuning.
- Some microbench improvements are on lower-frequency or lower-impact shapes.
- Stream-K-aware tuning changes solver choices, but the existing table was already close for the dominant GEMMs.
- Full transformer execution includes scheduling, cat/copy, normalization, RoPE, and dispatch overhead around the GEMMs.

## Current status

This is a valid but small improvement candidate:

- Keep the Stream-K-aware T2I table as an experimental artifact.
- Do not promote it to the default v2.2 result yet because only transformer-only/VAE-abort validation was run.
- A full T2I article-equivalent run with VAE warmup and image output is needed before WebUI result replacement.

Projected full measured time if VAE/postprocess/unattributed overhead remains unchanged:

- previous current best total: `88.344 sec`
- transformer improvement: `85.995 -> 84.143 sec`
- projected total: about `86.5 sec`

This projection is not a substitute for a full output-producing benchmark.

## Next improvement targets

1. Run selected-region rocprof for the Stream-K-aware table.
   - Goal: confirm the new table reduces GEMM kernel total, not just wall-time variance.

2. Full T2I article-equivalent validation.
   - Use VAE warmup + one mode warmup + one measured run.
   - Confirm image output is generated and comparable.

3. Broaden tuning beyond the 8 manually selected shapes.
   - Capture all T2I transformer GEMM shapes from logs.
   - Generate a T2I-specific TunableOp table in a real transformer run, not only the microbench.

4. Move beyond GEMM.
   - GEMM-only tuning has only about `1%` more available in this stack.
   - Larger gains likely require attention backend improvements and elementwise/reduce fusion.
