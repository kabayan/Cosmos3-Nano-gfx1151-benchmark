# Cosmos3 T2I Stream-K-aware GEMM full result and next plan

## Full benchmark result

The Stream-K-aware T2I TunableOp table was validated with the full article-equivalent T2I path:

- VAE warmup: enabled
- mode warmup run: `1`
- measured run: `1`
- output image: generated
- resolution: `960x960`
- steps: `35`
- guidance: `1.0`
- seed: `201`

Command characteristics:

```bash
TENSILE_SOLUTION_SELECTION_METHOD=2
ROCBLAS_USE_HIPBLASLT=1
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_FILENAME=/workspace/result/rocm_speed_matrix/t2i_gemm_probe/streamk_tunable%d.csv
python3 scripts/benchmark_classmethod_article_t2i_rocm.py \
  --height 960 --width 960 --steps 35 --guidance 1.0 \
  --stage-profile \
  --vae-warmup --vae-warmup-shape 1,48,1,60,60 \
  --mode-warmup-runs 1 \
  --measured-runs 1
```

Artifact:

- `result/rocm_speed_matrix/aotriton_streamk_tuned/t2i_article_warm_full/article_t2i_summary.json`
- `result/rocm_speed_matrix/aotriton_streamk_tuned/t2i_article_warm_full/article_t2i_robotics_lab_960x960_s35_measured_r2.jpg`

## Speed comparison

| Condition | Total | Transformer | VAE decode | Postprocess |
|---|---:|---:|---:|---:|
| Previous best `aotriton_tunable` | 88.344 sec | 85.995 sec | 1.769 sec | 0.008 sec |
| Stream-K-aware T2I table | **87.420 sec** | **85.078 sec** | 1.763 sec | 0.007 sec |

Improvement:

- total: `0.924 sec` faster, `1.011x`
- transformer: `0.917 sec` faster, `1.011x`

The full run confirms the transformer-only probe result direction, but the improvement is small.

## Warmup

VAE warmup remains excluded from the representative speed value:

| Condition | VAE warmup |
|---|---:|
| Previous best `aotriton_tunable` | 669.992 sec |
| Stream-K-aware T2I table | 659.754 sec |

These warmup values are not part of the measured generation time.

## Output status

The Stream-K-aware run generated a valid JPEG output.

SHA256 differs from the previous best output:

```text
previous best: 9352d9cb724d510d6bb858a758233fee9ca97b61d58a6e66254f757c527a4bd5
streamk tuned: 91b7bf8708e510472bafbe70a2335ac78d3de20b1845e844bb951fffc8a1eb68
```

This means the result is not byte-identical. Since GEMM solver selection changed, small floating-point differences are plausible. Treat this as speed-valid but requiring visual/image-diff validation before replacing the WebUI headline value.

## Interpretation

The T2I-specific Stream-K-aware table improves full measured speed, but only by about 1%.

This is consistent with the deeper profile:

- GEMM improved in microbench and slightly in full transformer.
- Attention remains about `12.3 sec`.
- Elementwise/reduce/copy remains about `11.1 sec`.
- Existing TunableOp selection was already close to optimal for the dominant GEMMs.

Therefore, further GEMM-only work is unlikely to close the article gap alone.

Current gap:

- article T2I: `22 sec`
- current measured full result: `87.420 sec`
- difference: `3.97x slower`

## Next improvement plan

### Phase 1: validate whether to promote Stream-K-aware T2I table

1. Run an image comparison against previous best.
   - Confirm dimensions.
   - Compute simple image diff metrics such as mean absolute pixel difference and max difference.
   - Save side-by-side preview.

2. Re-run full measured once more.
   - Purpose: determine if `87.420 sec` is stable or run-to-run variance.
   - If the second run is still faster than `88.344 sec`, promote the value.

3. Update WebUI only after validation.
   - T2I current best can become `87.420 sec` if visual/output validation is acceptable.
   - Note that output is not byte-identical.

### Phase 2: collect full T2I GEMM logs

1. Capture rocBLAS/hipBLASLt logs for the T2I transformer window.
2. Extract all T2I GEMM descriptors, not only the eight manually probed shapes.
3. Generate a T2I-specific tuning table from an actual transformer run.
4. Compare against:
   - existing `tunableop_results0.csv`
   - microbench-generated `streamk_tunable0.csv`

Goal: determine if the 1% gain is the ceiling or if unprobed shapes still have tuning room.

### Phase 3: selected-region rocprof for Stream-K-aware full path

1. Run selected-region rocprof for the Stream-K-aware table.
2. Compare GEMM kernel total against the previous `60.651 sec`.
3. Confirm whether transformer wall-time improvement corresponds to lower GEMM time.

Success criteria:

- GEMM total decreases by more than 1 sec, or
- non-GEMM overhead decreases unexpectedly and can be explained.

### Phase 4: move beyond GEMM

If Phase 2/3 shows GEMM is near its practical limit:

1. Attention backend work.
   - Current attention cost is about `12.3 sec`.
   - Test newer PyTorch ROCm / AOTriton / SDPA stack.
   - Investigate forcing or improving the attention backend only for Cosmos3 transformer.

2. Elementwise/reduce/copy fusion.
   - Current non-GEMM, non-attention GPU cost is about `11.1 sec`.
   - Targets include RMSNorm, RoPE, cat/copy around attention, SiLU/mul, and dtype copy kernels.
   - Likely requires transformer implementation changes or compiled fusion.

3. Full transformer implementation optimization.
   - Fused MLP gate/up path could reduce launch overhead and intermediate memory traffic.
   - Fused RMSNorm + projection is another candidate.
   - These require code changes and output validation.

### Recommended next action

Proceed with Phase 1 first:

1. image diff / side-by-side validation
2. one repeat full measured run
3. promote or reject `87.420 sec` as the new T2I value

Then run Phase 3 selected-region rocprof. Phase 2 full GEMM logging is useful if we want to keep exploring GEMM despite the small observed improvement.
