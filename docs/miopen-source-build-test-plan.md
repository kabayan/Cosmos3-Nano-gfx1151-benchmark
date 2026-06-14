# ROCm/MIOpen fork/build/test plan and execution result

Date: 2026-06-03

## Scope

Goal: verify whether a MIOpen-side change can make the Cosmos VAE large BF16 3D convolution descriptors avoid the `ConvDirectNaiveConvFwd` fallback and use a GEMM-based path.

Target source:

- ROCm rocm-libraries tag: `rocm-7.2.0`
- Commit: `5b515cf1bca9959fb434c2414cf79b42fe25e93b`
- Local source: `third_party/rocm-libraries-rocm-7.2.0/projects/miopen`
- Current runtime MIOpen version: `3.5.1.5b515cf1bc`

Official references:

- MIOpen install documentation: <https://rocm.docs.amd.com/projects/MIOpen/en/latest/install/install.html>
- MIOpen build-from-source documentation: <https://rocm.docs.amd.com/projects/MIOpen/en/latest/install/build-source.html>

## Hypothesis

The previous VAE probe showed full 3D BF16 descriptors falling back to `ConvDirectNaiveConvFwd`.
Source inspection found that `GemmFwdRest::GetWorkspaceSize()` returns `0` when the im2col/GEMM workspace exceeds `gemm::MaxMemAllocSz()`. That makes `GemmFwdRest::IsApplicable()` false. The hardcoded limit is about `7,287,183,769` bytes, and only FP32 has an existing double-limit workaround.

The two problematic full descriptors need:

| Descriptor | GEMM workspace |
|---|---:|
| 160 full BF16 | `13,837,271,040` bytes |
| 512 full BF16 | `8,493,465,600` bytes |

Therefore, a focused MIOpen experiment is:

1. Add an opt-in environment flag.
2. For FP16/BF16 forward GEMM-rest workspace gating, double the same hardcoded allocation limit used for FP32.
3. Rebuild MIOpen and test whether the full descriptors become `GemmFwdRest` eligible.

This is not a production fix. It is a technical validation patch to prove that the workspace gating is the immediate reason for the naive fallback.

## Implemented artifacts

- Build image: `docker/miopen-rocm72-build.Dockerfile`
- Docker image build runner: `scripts/miopen/build_miopen_rocm72_image.sh`
- MIOpen source build runner: `scripts/miopen/build_miopen_rocm72.sh`
- Docker source build runner: `scripts/miopen/run_miopen_source_build_in_docker.sh`
- Custom-library PyTorch probe runner: `scripts/miopen/run_custom_miopen_probe_in_docker.sh`
- MIOpenDriver descriptor test runner: `scripts/miopen/test_large_bf16_conv3d_descriptors.sh`
- Experimental patch: `patches/miopen/allow-large-fp16-bf16-gemm-rest-workspace-experiment.patch`

The patch adds:

- `MIOPEN_EXPERIMENT_LARGE_FP16_BF16_GEMM_REST`
- BF16/FP16 limit doubling in `gemm::MaxMemAllocSz()` when the env flag is enabled.

Build configuration adjustments used for this repo-local validation:

- `MIOPEN_USE_MLIR=Off`
- `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=Off`
- `MIOPEN_ENABLE_AI_KERNEL_TUNING=Off`
- `BUILD_TESTING=Off`

These avoid unrelated dependencies (`rocMLIR`, `frugally-deep`, `GTest`) and keep the build focused on libMIOpen and MIOpenDriver.

## Executed steps

1. Built the Docker build image:

```bash
scripts/miopen/build_miopen_rocm72_image.sh
```

Result: success.

2. Verified patch applicability inside Docker:

```bash
docker run --rm \
  -v /home/kabayan/workspace/cosmos3:/workspace \
  -w /workspace \
  cosmos3-miopen-rocm72-build:latest \
  bash -lc 'git -C third_party/rocm-libraries-rocm-7.2.0 apply --check /workspace/patches/miopen/allow-large-fp16-bf16-gemm-rest-workspace-experiment.patch'
```

Result: success.

3. Built and installed MIOpen with the experimental patch:

```bash
BUILD_JOBS=4 scripts/miopen/run_miopen_source_build_in_docker.sh
```

Result: success.

Installed artifacts:

- `result/miopen-build/rocm-7.2.0/install/lib/libMIOpen.so`
- `result/miopen-build/rocm-7.2.0/install/bin/MIOpenDriver`

Notable build findings:

- Base `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1` lacked `cmake`.
- Additional apt dependencies were needed: `libbz2-dev`, `nlohmann-json3-dev`.
- MIOpen default configure wanted `rocMLIR`, `frugally-deep`, and `GTest`; these were avoided with the focused options above.

4. Ran full descriptor smoke tests with the custom MIOpen:

```bash
MIOPEN_DRIVER=/workspace/result/miopen-build/rocm-7.2.0/install/bin/MIOpenDriver \
MIOPEN_LIB_DIR=/workspace/result/miopen-build/rocm-7.2.0/install/lib \
MIOPEN_EXPERIMENT_LARGE_FP16_BF16_GEMM_REST=1 \
timeout 300 scripts/miopen/test_large_bf16_conv3d_descriptors.sh \
  /workspace/result/miopen-large-bf16-conv3d-test-custom
```

The combined run timed out during the second descriptor after the first descriptor completed, so the 512 descriptor was rerun alone with a fresh 300 second timeout.

## Test results

Summary log:

- `result/docs/miopen-large-bf16-conv3d-test-summary.log`
- `result/docs/custom-miopen-pytorch-probe-summary.md`

| Descriptor | Workspace | Solver selected | Find time estimate | GPU kernel time | Validation |
|---|---:|---|---:|---:|---|
| 160 full BF16 | `13,837,271,040` bytes | `GemmFwdRest` | `182.983 ms` | `183.939636 ms` | Failed: `0.149259 > 0.0656` |
| 512 full BF16 | `8,493,465,600` bytes | `GemmFwdRest` | `250.206 ms` | `249.747253 ms` | OK: `6.22853e-05 < 0.0656` |

Key result:

- The experiment changed both full descriptors from the previous naive fallback condition to `GemmFwdRest`.
- The immediate workspace gating hypothesis is validated.
- The 160 descriptor needs correctness follow-up because MIOpenDriver's validation exceeded tolerance.

## PyTorch custom libMIOpen probe

After confirming MIOpenDriver behavior, the custom `libMIOpen.so` was loaded through the PyTorch ROCm container with:

- `LD_LIBRARY_PATH=/workspace/result/miopen-build/rocm-7.2.0/install/lib:...`
- `MIOPEN_EXPERIMENT_LARGE_FP16_BF16_GEMM_REST=1`
- `MIOPEN_FIND_MODE=NORMAL`
- `MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE`

The probe runner was updated so the output directory and descriptor can be selected through environment variables. This avoids writing new logs under root-owned `result/miopen-build/...` paths.

Single-run results:

| Descriptor | PyTorch wall time | Workspace | Solver |
|---|---:|---:|---|
| 160 full BF16 | `163.897s` | `13,837,271,040` bytes | `GemmFwdRest` |
| 512 full BF16 | `179.217s` | `8,493,465,600` bytes | `GemmFwdRest` |

Same-process `repeats=2` results:

| Descriptor | First run | Second run | Max allocated | Max reserved |
|---|---:|---:|---:|---:|
| 160 full BF16 | `164.119s` | `0.744s` | `15.45GB` | `28.80GB` |
| 512 full BF16 | `180.290s` | `0.593s` | `9.29GB` | `17.65GB` |

Interpretation:

- PyTorch also loads and uses the custom MIOpen path successfully.
- The first call is still dominated by MIOpen Find because it evaluates the naive solver before selecting `GemmFwdRest`.
- Once warmed in-process, the same descriptor runs below one second in this isolated conv probe.
- The reserved memory is high, especially for the 160 descriptor, because the 13.84GB workspace remains part of the allocator behavior.

## Interpretation

This confirms that repo-local source changes can alter MIOpen solver eligibility for the large BF16 3D conv descriptors.

The current experiment is still not sufficient as an upstream-ready kernel improvement because:

- It increases permitted workspace size rather than adding a new optimized 3D BF16 direct/implicit solver.
- It can allocate 8.49 to 13.84 GB of workspace per conv, which is risky inside end-to-end Cosmos VAE decode.
- It does not explain or fix the 160 descriptor validation failure.
- It may improve single-kernel compute time but still increase memory pressure and end-to-end latency if the model repeatedly allocates huge workspaces.

## Next test plan

1. Correctness isolation for the 160 descriptor:
   - rerun with smaller `-t 1` and logging reduced;
   - compare BF16 vs FP16;
   - run with CPU/GPU reference modes if MIOpenDriver supports selecting them;
   - check whether the validation failure is BF16 tolerance/reference related or an actual GEMM-rest correctness issue.

2. End-to-end custom lib probe:
   - run `scripts/miopen/run_custom_miopen_probe_in_docker.sh`;
   - verify PyTorch actually loads `result/miopen-build/rocm-7.2.0/install/lib/libMIOpen.so`;
   - confirm solver logs inside the existing VAE conv probe.

3. Memory pressure test:
   - run the 160 and 512 full descriptors back-to-back with ROCm memory telemetry;
   - confirm whether 13.84 GB workspace is sustainable in the full model memory budget.

4. Production-grade alternatives:
   - add or enable a true optimized 3D BF16 solver that does not require full im2col workspace;
   - avoid the descriptor at the model/framework level with decode tiling;
   - implement model-side tiling that balances workspace and kernel efficiency.
