# Cosmos3 Nano ROCm PyTorch/AOTriton Newer Stack and Kernel Work

Date: 2026-06-04

## Scope

This document records the execution of the four next actions from the I2V kernel stack investigation:

1. Implement a transformer-only Stream-K probe.
2. If Stream-K improves or reaches transformer execution, investigate the VAE SDPA failure path.
3. If Stream-K is useful but unstable, define the hipBLASLt/TensileLite offline tuning path.
4. For attention, plan the newer PyTorch ROCm/AOTriton stack or kernel modification path.

The quality/comparison settings remain fixed:

- Mode: I2V
- Input data: Cosmos3 official `example_i2v_input.jpg` and `example_i2v_prompt.json`
- Resolution: `448x256`
- Frames: `24`
- FPS: `12`
- Steps: `35`
- Guidance: `1.0`
- Seed: `203`

The transformer-only probe intentionally does not export a video. It is only a kernel-stack diagnostic.

## Implemented Changes

### Benchmark runner

File:

```text
scripts/benchmark_classmethod_article_t2v_i2v_rocm.py
```

Added options:

- `--allow-pipeline-error`
  - Allows the benchmark to write a summary when the pipeline fails after `transformer_forward` has been measured.
- `--abort-before-vae-decode`
  - Replaces `pipe.vae.decode()` with an intentional abort so that cold VAE decode does not pollute transformer-only timing.
- `--force-vae-encode-sdpa-math`
  - Wraps only `pipe._encode_video()` and temporarily replaces `torch.nn.functional.scaled_dot_product_attention` with a local matmul/softmax implementation.
  - Purpose: avoid the ROCm SDPA failure seen inside Wan VAE encode/decode while leaving transformer attention/GEMM paths unchanged.

### Speed matrix runner

File:

```text
scripts/run_rocm_speed_matrix.py
```

Added case:

```text
i2v_article_transformer_streamk_probe
```

This case runs I2V with:

```text
--stage-profile
--mode-warmup-runs 0
--measured-runs 1
--inference-mode
--disable-progress-bar
--allow-pipeline-error
--abort-before-vae-decode
--force-vae-encode-sdpa-math
```

## Dry Run

Dry-run passed for:

```text
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned \
  --variant aotriton_tuned_streamk \
  --case i2v_article_transformer_streamk_probe
```

The generated commands preserve the fixed I2V quality settings.

## Execution Results

### Baseline: `aotriton_tuned`

Command:

```text
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned \
  --case i2v_article_transformer_streamk_probe \
  --execute
```

Artifact:

```text
result/rocm_speed_matrix/aotriton_tuned/i2v_article_transformer_streamk_probe/summary.json
```

Result:

| Metric | Value |
|---|---:|
| `transformer_forward` | `70.769 sec` |
| Calls | `35` |
| Per step | `2.022 sec/step` |
| Pipeline wall before intentional abort | `128.600 sec` |
| Error | Intentional abort before VAE decode |

Interpretation:

- The probe works for the baseline.
- The measured transformer time is consistent with the v2.1 warmed full result (`71.691 sec` transformer).
- The large unattributed wall time is pre-transformer setup and first-run I2V conditioning; it is not the comparison target.

### Stream-K: `aotriton_tuned_streamk`

Command attempted:

```text
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned_streamk \
  --case i2v_article_transformer_streamk_probe \
  --execute
```

Observed failures:

1. Before the VAE encode workaround, Stream-K failed inside Wan VAE encode SDPA:

```text
RuntimeError: Expected iter != ops_.end() to be true, but got false.
```

Location:

```text
diffusers/models/autoencoders/autoencoder_kl_wan.py
F.scaled_dot_product_attention(q, k, v)
```

2. A `sdpa_kernel([SDPBackend.MATH])` context around `_encode_video()` did not avoid this failure.

3. A local naive SDPA wrapper was implemented for VAE encode only, but the follow-up Stream-K run could not complete because GitHub DNS repeatedly failed while installing diffusers:

```text
fatal: unable to access 'https://github.com/huggingface/diffusers.git/':
Could not resolve host: github.com
```

No valid Stream-K transformer timing was collected yet.

## Status of Actions 1-4

### 1. Transformer-only Stream-K probe

Status: partially complete.

Completed:

- Implemented the runner and benchmark options.
- Verified dry-run.
- Collected baseline transformer-only timing.

Blocked:

- Stream-K measurement still needs a successful diffusers install after network/DNS recovery.
- The VAE encode SDPA workaround is implemented but not yet validated because the retry stopped before model execution.

### 2. VAE SDPA failure investigation

Status: in progress.

Findings:

- Global Stream-K/hipBLASLt env can break Wan VAE SDPA before transformer execution.
- The failure occurs in VAE encode for I2V, not only in VAE decode.
- PyTorch `sdpa_kernel([SDPBackend.MATH])` was not sufficient in this environment.

Implemented mitigation:

- Temporarily monkeypatch VAE encode SDPA to a local matmul/softmax implementation during `_encode_video()` only.
- This preserves transformer attention and GEMM kernel selection for the diagnostic probe.

Next validation:

```text
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned_streamk \
  --case i2v_article_transformer_streamk_probe \
  --execute
```

Expected outcome:

- If the wrapper works, the run should reach `35` transformer calls and then intentionally abort before VAE decode.

### 3. hipBLASLt/TensileLite offline tuning path

Status: planned, not executed in this step.

Condition to proceed:

- Only proceed if transformer-only Stream-K shows a material transformer speedup.

Reason:

- Earlier standalone GEMM probes showed Stream-K can improve representative I2V GEMM shapes by about `1.10x` to `1.37x`.
- Full pipeline Stream-K is unstable because of Wan VAE SDPA.
- Offline tuning is the stable route if the transformer-only result confirms real benefit.

Required work:

1. Extract exact rocBLAS/hipBLASLt problem descriptors from the I2V transformer GEMM log.
2. Use hipBLASLt/TensileLite tuning for the high-time shapes.
3. Build and test a tuned hipBLASLt/Tensile library or configuration.
4. Re-run:
   - synthetic GEMM probe
   - transformer-only I2V probe
   - full I2V article-equivalent benchmark

Acceptance criteria:

- Same quality settings.
- No global VAE SDPA failure.
- I2V transformer time improves versus `70.769-71.691 sec`.
- Full I2V output still succeeds.

### 4. Newer PyTorch ROCm/AOTriton or attention kernel path

Status: planned.

Reasoning:

- Current local attention backend switching did not improve I2V.
- Current default already uses a fast SDPA/AOTriton-like path.
- Attention is about `9.926 sec` of tuned I2V transformer time, while GEMM is about `51.852 sec`.
- Attention kernel work is therefore secondary to GEMM unless a newer stack provides a free improvement.

Plan:

1. Test a newer prebuilt PyTorch ROCm image if available.
   - Compare `torch.__version__`, `torch.version.hip`, AOTriton behavior, and SDPA probe output.
2. Re-run the existing attention backend probe:

```text
python3 scripts/probe_cosmos3_i2v_attention_backends.py \
  --out result/docs/cosmos3_i2v_attention_backend_probe_newer_stack.json
```

3. Re-run transformer-only I2V ROCprof for kernel breakdown.
4. If newer stack improves attention, run full I2V.
5. If newer stack does not improve attention, custom AOTriton work should target:
   - GQA shape with `q_heads=32`, `kv_heads=8`, `head_dim=128`
   - `q_len` around `1904`
   - `kv_len` around `2141`

Acceptance criteria:

- No quality-setting changes.
- Same prompt/image/seed.
- Attention kernel time decreases in rocprof.
- Transformer total decreases, not just synthetic SDPA time.

## Current Decision

Do not enable Stream-K globally for full I2V yet.

The current evidence is:

- Standalone GEMM: Stream-K is promising.
- Full I2V: Stream-K is unstable due to Wan VAE SDPA.
- Baseline transformer-only probe: implemented and measured.
- Stream-K transformer-only probe: implementation ready, but final validation is pending GitHub/DNS recovery.

The next concrete command after network recovery is:

```text
python3 scripts/run_rocm_speed_matrix.py \
  --variant aotriton_tuned_streamk \
  --case i2v_article_transformer_streamk_probe \
  --execute
```

