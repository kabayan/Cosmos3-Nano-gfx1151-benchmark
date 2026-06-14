import argparse
import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path


IMAGE = os.environ.get("COSMOS3_ROCM_IMAGE", "rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1")
WORKDIR = Path("/home/kabayan/workspace/cosmos3")
HF_CACHE = Path("/home/kabayan/.cache/huggingface")
COSMOS_FRAMEWORK = Path("/tmp/cosmos-framework")


VARIANTS = {
    "v1_0": {},
    "aotriton": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
    },
    "tunable_collect": {
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "0",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "1",
        "PYTORCH_TUNABLEOP_FILENAME": "/workspace/result/rocm_speed_matrix/tunableop_untuned.csv",
    },
    "tunable_online": {
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "1",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0",
        "PYTORCH_TUNABLEOP_FILENAME": "/workspace/result/rocm_speed_matrix/tunableop_results.csv",
    },
    "allocator": {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    "aotriton_tunable": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "1",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0",
        "PYTORCH_TUNABLEOP_FILENAME": "/workspace/result/rocm_speed_matrix/tunableop_results%d.csv",
    },
    "aotriton_tuned": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "0",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0",
        "PYTORCH_TUNABLEOP_FILENAME": "/workspace/result/rocm_speed_matrix/tunableop_results%d.csv",
    },
    "aotriton_tuned_attn_flash": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "0",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0",
        "PYTORCH_TUNABLEOP_FILENAME": "/workspace/result/rocm_speed_matrix/tunableop_results%d.csv",
        "DIFFUSERS_ATTN_BACKEND": "_native_flash",
    },
    "aotriton_tuned_attn_efficient": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "0",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0",
        "PYTORCH_TUNABLEOP_FILENAME": "/workspace/result/rocm_speed_matrix/tunableop_results%d.csv",
        "DIFFUSERS_ATTN_BACKEND": "_native_efficient",
    },
    "aotriton_tuned_attn_math": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "0",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0",
        "PYTORCH_TUNABLEOP_FILENAME": "/workspace/result/rocm_speed_matrix/tunableop_results%d.csv",
        "DIFFUSERS_ATTN_BACKEND": "_native_math",
    },
    "aotriton_streamk": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "TENSILE_SOLUTION_SELECTION_METHOD": "2",
        "ROCBLAS_USE_HIPBLASLT": "1",
    },
    "aotriton_streamk_safe": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "0",
        "PYTORCH_TUNABLEOP_TUNING": "0",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0",
        "TENSILE_SOLUTION_SELECTION_METHOD": "2",
        "ROCBLAS_USE_HIPBLASLT": "1",
    },
    "aotriton_tuned_streamk": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "0",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0",
        "PYTORCH_TUNABLEOP_FILENAME": "/workspace/result/rocm_speed_matrix/tunableop_results%d.csv",
        "TENSILE_SOLUTION_SELECTION_METHOD": "2",
        "ROCBLAS_USE_HIPBLASLT": "1",
    },
    "aotriton_i2v_deep_tunable": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "1",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0",
        "PYTORCH_TUNABLEOP_FILENAME": "/workspace/result/rocm_speed_matrix/tunableop_i2v_deep%d.csv",
    },
    "aotriton_i2v_deep_tuned": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "0",
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0",
        "PYTORCH_TUNABLEOP_FILENAME": "/workspace/result/rocm_speed_matrix/tunableop_i2v_deep%d.csv",
    },
    "aotriton_miopen_find": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "MIOPEN_ENABLE_LOGGING": "1",
        "MIOPEN_ENABLE_LOGGING_CMD": "1",
        "MIOPEN_LOG_LEVEL": "5",
        "MIOPEN_FIND_MODE": "NORMAL",
        "MIOPEN_FIND_ENFORCE": "SEARCH_DB_UPDATE",
        "MIOPEN_USER_DB_PATH": "/workspace/result/rocm_speed_matrix/miopen_user_db",
    },
    "aotriton_miopen_warm": {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "MIOPEN_FIND_MODE": "NORMAL",
        "MIOPEN_USER_DB_PATH": "/workspace/result/rocm_speed_matrix/miopen_user_db",
    },
}


DIFFUSERS_INSTALL = os.environ.get(
    "COSMOS3_DIFFUSERS_INSTALL",
    (
        'python -m pip install --quiet --no-cache-dir '
        '"diffusers @ git+https://github.com/huggingface/diffusers.git" '
        "accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg scipy "
        "transformers safetensors pillow"
    ),
)

FRAMEWORK_INSTALL = os.environ.get(
    "COSMOS3_FRAMEWORK_INSTALL",
    (
        "python -m pip install --quiet --no-cache-dir --no-deps -e . && "
        "python -m pip install --quiet --no-cache-dir diffusers accelerate av huggingface_hub "
        "imageio imageio-ffmpeg scipy transformers==4.57.1 safetensors pillow einops "
        "omegaconf hydra-core loguru pydantic cattrs msgpack nvidia-ml-py obstore "
        "termcolor tyro uv websockets requests iopath boto3 qwen-vl-utils"
    ),
)


def docker_base(extra_env: dict[str, str], framework: bool = False) -> list[str]:
    env = {
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_DISABLE_XET": "1",
        **extra_env,
    }
    if framework:
        env["COSMOS_TRAINING"] = "0"

    cmd = [
        "docker",
        "run",
        "--rm",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add",
        "44",
        "--group-add",
        "993",
        "--cap-add=SYS_PTRACE",
        "--security-opt",
        "seccomp=unconfined",
        "--ipc=host",
    ]
    for key, value in env.items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.extend(
        [
            "-v",
            f"{HF_CACHE}:/root/.cache/huggingface",
            "-v",
            f"{WORKDIR}:/workspace",
            "-v",
            f"{COSMOS_FRAMEWORK}:/workspace/tmp/cosmos-framework",
            "-w",
            "/workspace/tmp/cosmos-framework" if framework else "/workspace",
            IMAGE,
            "bash",
            "-lc",
        ]
    )
    return cmd


def rocblas_log_env(out_dir: str) -> dict[str, str]:
    log_dir = f"{out_dir}/gemm_logs"
    return {
        "ROCBLAS_LAYER": "14",
        "ROCBLAS_LOG_BENCH_PATH": f"{log_dir}/rocblas_bench.log",
        "ROCBLAS_LOG_PROFILE_PATH": f"{log_dir}/rocblas_profile.yaml",
        "ROCBLAS_LOG_TRACE_PATH": f"{log_dir}/rocblas_trace.log",
        "ROCBLAS_VERBOSE_TENSILE_ERROR": "1",
        "ROCBLAS_VERBOSE_HIPBLASLT_ERROR": "1",
        "HIPBLASLT_LOG_LEVEL": "5",
        "HIPBLASLT_LOG_MASK": "242",
        "HIPBLASLT_LOG_FILE": f"{log_dir}/hipblaslt_%i.log",
        "HIPBLASLT_ENABLE_MARKER": "1",
    }


def rocprof_wrap(inner: str, out_dir: str, selected_regions: bool = False) -> str:
    profile_dir = f"{out_dir}/rocprof"
    selected = " --selected-regions" if selected_regions else ""
    marker = " --marker-trace" if selected_regions else ""
    return (
        f"mkdir -p {shlex.quote(profile_dir)} && "
        f"rocprofv3 --kernel-trace --memory-copy-trace{marker} --stats{selected} "
        f"--summary --summary-output-file {shlex.quote(profile_dir + '/summary.txt')} "
        f"--output-directory {shlex.quote(profile_dir)} --output-file profile -f csv -- "
        f"bash -lc {shlex.quote(inner)}"
    )


def case_command(case: str, variant: str) -> tuple[list[str], str]:
    out_dir = f"/workspace/result/rocm_speed_matrix/{variant}/{case}"
    extra_env = VARIANTS[variant]
    if case == "t2i_smoke":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2i_rocm.py "
            f"--out-dir {out_dir} --height 480 --width 480 --steps 8 --guidance 1.0"
        )
        return docker_base(extra_env), inner
    if case == "tech_validate":
        inner = (
            "PYTHONPATH=/workspace:/workspace/scripts "
            "python3 /workspace/scripts/validate_rocm_optimization_primitives.py "
            f"--out {out_dir}/tech_validate.json"
        )
        return docker_base(extra_env), inner
    if case == "t2i_article":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2i_rocm.py "
            f"--out-dir {out_dir} --height 960 --width 960 --steps 35 --guidance 1.0"
        )
        return docker_base(extra_env), inner
    if case == "t2i_article_warm_full":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2i_rocm.py "
            f"--out-dir {out_dir} --height 960 --width 960 --steps 35 --guidance 1.0 "
            "--stage-profile --vae-warmup --vae-warmup-shape 1,48,1,60,60 "
            "--mode-warmup-runs 1 --measured-runs 1"
        )
        return docker_base(extra_env), inner
    if case == "t2i_article_warm_full_rocprof":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2i_rocm.py "
            f"--out-dir {out_dir} --height 960 --width 960 --steps 35 --guidance 1.0 "
            "--stage-profile --vae-warmup --vae-warmup-shape 1,48,1,60,60 "
            "--mode-warmup-runs 1 --measured-runs 1"
        )
        return docker_base(extra_env), rocprof_wrap(inner, out_dir)
    if case == "t2i_article_transformer_rocprof":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2i_rocm.py "
            f"--out-dir {out_dir} --height 960 --width 960 --steps 35 --guidance 1.0 "
            "--stage-profile --vae-warmup --vae-warmup-shape 1,48,1,60,60 "
            "--mode-warmup-runs 1 --measured-runs 1 --rocprof-transformer-only"
        )
        return docker_base(extra_env), rocprof_wrap(inner, out_dir, selected_regions=True)
    if case == "t2v_i2v_smoke":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case both --out-dir {out_dir} --height 256 --width 448 "
            "--frames 8 --fps 8 --steps 8 --guidance 1.0"
        )
        return docker_base(extra_env), inner
    if case == "t2v_i2v_stage_smoke":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case both --out-dir {out_dir} --height 256 --width 448 "
            "--frames 8 --fps 8 --steps 8 --guidance 1.0 --stage-profile"
        )
        return docker_base(extra_env), inner
    if case == "vae_decode_probe":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/probe_cosmos3_vae_decode_rocm.py "
            f"--out-dir {out_dir} --height 256 --width 448 --frames 8 --fps 8 "
            "--steps 8 --guidance 1.0 --standalone-runs 2 --abort-after-capture"
        )
        return docker_base(extra_env), inner
    if case in {"vae_warmup_default", "vae_warmup_tiling", "vae_warmup_slicing"}:
        mode = {
            "vae_warmup_default": "default",
            "vae_warmup_tiling": "tiling",
            "vae_warmup_slicing": "slicing",
        }[case]
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/probe_cosmos3_vae_warmup_rocm.py "
            f"--out-dir {out_dir} --height 256 --width 448 --frames 8 --fps 8 "
            f"--steps 8 --guidance 1.0 --vae-mode {mode}"
        )
        return docker_base(extra_env), inner
    if case == "vae_warmup_t2v_twice":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/probe_cosmos3_vae_warmup_rocm.py "
            f"--out-dir {out_dir} --height 256 --width 448 --frames 8 --fps 8 "
            "--steps 8 --guidance 1.0 --vae-mode default --t2v-runs 2"
        )
        return docker_base(extra_env), inner
    if case == "t2v_i2v_article":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case both --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0"
        )
        return docker_base(extra_env), inner
    if case == "t2v_i2v_article_warm_full":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case both --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 --stage-profile "
            "--vae-warmup --vae-warmup-shape 1,48,2,16,28 "
            "--mode-warmup-runs 1 --measured-runs 1"
        )
        return docker_base(extra_env), inner
    if case == "t2v_article_warm_full":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case t2v --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 --stage-profile "
            "--vae-warmup --vae-warmup-shape 1,48,2,16,28 "
            "--mode-warmup-runs 1 --measured-runs 1"
        )
        return docker_base(extra_env), inner
    if case == "i2v_article_warm_full":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case i2v --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 --stage-profile "
            "--vae-warmup --vae-warmup-shape 1,48,2,16,28 "
            "--mode-warmup-runs 1 --measured-runs 1"
        )
        return docker_base(extra_env), inner
    if case == "i2v_article_und_cache_warm_full":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case i2v --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 --stage-profile "
            "--vae-warmup --vae-warmup-shape 1,48,2,16,28 "
            "--mode-warmup-runs 1 --measured-runs 1 "
            "--inference-mode --disable-progress-bar --und-branch-cache"
        )
        return docker_base(extra_env), inner
    if case == "i2v_article_runtime_no_profile":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case i2v --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 "
            "--vae-warmup --vae-warmup-shape 1,48,2,16,28 "
            "--mode-warmup-runs 1 --measured-runs 1 "
            "--inference-mode --disable-progress-bar"
        )
        return docker_base(extra_env), inner
    if case == "i2v_article_transformer_streamk_probe":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case i2v --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 "
            "--stage-profile --mode-warmup-runs 0 --measured-runs 1 "
            "--inference-mode --disable-progress-bar --allow-pipeline-error "
            "--abort-before-vae-decode --force-vae-encode-sdpa-math "
            "--cosmos3-transformer-attention-fallback"
        )
        return docker_base(extra_env), inner
    if case == "i2v_article_warm_full_deep_tune":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case i2v --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 --stage-profile "
            "--vae-warmup --vae-warmup-shape 1,48,2,16,28 "
            "--mode-warmup-runs 1 --measured-runs 1 "
            "--tunable-max-tuning-duration 100 "
            "--tunable-max-tuning-iterations 200 "
            "--tunable-rotating-buffer-size 1024"
        )
        return docker_base(extra_env), inner
    if case == "t2v_article_transformer_rocprof":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case t2v --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 --stage-profile "
            "--vae-warmup --vae-warmup-shape 1,48,2,16,28 "
            "--mode-warmup-runs 1 --measured-runs 1 --rocprof-transformer-only"
        )
        return docker_base(extra_env), rocprof_wrap(inner, out_dir, selected_regions=True)
    if case == "i2v_article_transformer_rocprof":
        inner = (
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case i2v --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 --stage-profile "
            "--vae-warmup --vae-warmup-shape 1,48,2,16,28 "
            "--mode-warmup-runs 1 --measured-runs 1 --rocprof-transformer-only"
        )
        return docker_base(extra_env), rocprof_wrap(inner, out_dir, selected_regions=True)
    if case == "t2v_article_gemm_log":
        inner = (
            "set -o pipefail && "
            f"mkdir -p {out_dir}/gemm_logs && "
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case t2v --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 --stage-profile "
            "--vae-warmup --vae-warmup-shape 1,48,2,16,28 "
            "--mode-warmup-runs 1 --measured-runs 1 "
            f"> {out_dir}/gemm_logs/run.log 2>&1"
        )
        return docker_base({**extra_env, **rocblas_log_env(out_dir)}), inner
    if case == "i2v_article_gemm_log":
        inner = (
            "set -o pipefail && "
            f"mkdir -p {out_dir}/gemm_logs && "
            f"{DIFFUSERS_INSTALL} && "
            "HF_HUB_DISABLE_XET=1 python3 scripts/benchmark_classmethod_article_t2v_i2v_rocm.py "
            f"--case i2v --out-dir {out_dir} --height 256 --width 448 "
            "--frames 24 --fps 12 --steps 35 --guidance 1.0 --stage-profile "
            "--vae-warmup --vae-warmup-shape 1,48,2,16,28 "
            "--mode-warmup-runs 1 --measured-runs 1 "
            f"> {out_dir}/gemm_logs/run.log 2>&1"
        )
        return docker_base({**extra_env, **rocblas_log_env(out_dir)}), inner
    if case == "policy_article":
        inner = (
            f"{FRAMEWORK_INSTALL} && "
            "PYTHONPATH=/workspace:/workspace/tmp/cosmos-framework HF_HUB_DISABLE_XET=1 "
            f"python /workspace/scripts/run_cosmos_framework_policy_rocm.py --out-dir {out_dir}"
        )
        return docker_base(extra_env, framework=True), inner
    if case == "policy_article_rocprof":
        inner = (
            f"{FRAMEWORK_INSTALL} && "
            "PYTHONPATH=/workspace:/workspace/tmp/cosmos-framework HF_HUB_DISABLE_XET=1 "
            f"python /workspace/scripts/run_cosmos_framework_policy_rocm.py --out-dir {out_dir}"
        )
        return docker_base(extra_env, framework=True), rocprof_wrap(inner, out_dir)
    if case == "policy_article_skip_vision_decode":
        inner = (
            f"{FRAMEWORK_INSTALL} && "
            "PYTHONPATH=/workspace:/workspace/tmp/cosmos-framework HF_HUB_DISABLE_XET=1 "
            f"python /workspace/scripts/run_cosmos_framework_policy_rocm.py --out-dir {out_dir} "
            "--skip-vision-decode"
        )
        return docker_base(extra_env, framework=True), inner
    if case == "policy_article_action_only":
        inner = (
            f"{FRAMEWORK_INSTALL} && "
            "PYTHONPATH=/workspace:/workspace/tmp/cosmos-framework HF_HUB_DISABLE_XET=1 "
            f"python /workspace/scripts/run_cosmos_framework_policy_rocm.py --out-dir {out_dir} "
            "--action-only"
        )
        return docker_base(extra_env, framework=True), inner
    if case == "policy_article_miopen_find":
        inner = (
            "set -o pipefail && "
            f"mkdir -p {out_dir}/miopen /workspace/result/rocm_speed_matrix/miopen_user_db && "
            f"({FRAMEWORK_INSTALL} && "
            "PYTHONPATH=/workspace:/workspace/tmp/cosmos-framework HF_HUB_DISABLE_XET=1 "
            f"python /workspace/scripts/run_cosmos_framework_policy_rocm.py --out-dir {out_dir}) "
            f"> {out_dir}/miopen/miopen_run.log 2>&1"
        )
        return docker_base(extra_env, framework=True), inner
    if case == "large_conv3d_probe":
        inner = (
            "set -o pipefail && "
            f"mkdir -p {out_dir}/miopen && "
            "PYTHONPATH=/workspace "
            f"python /workspace/scripts/probe_miopen_large_conv3d_descriptors.py --out-dir {out_dir} "
            "--descriptor policy_160_18_274_370_to_160 --dtype bf16 "
            "--tile 1x1 --tile 2x1 --tile 2x2 --repeats 1 "
            f"> {out_dir}/miopen/large_conv3d_probe.log 2>&1"
        )
        return docker_base(extra_env), inner
    if case == "large_conv3d_probe_512_bf16":
        inner = (
            "set -o pipefail && "
            f"mkdir -p {out_dir}/miopen && "
            "PYTHONPATH=/workspace "
            f"python /workspace/scripts/probe_miopen_large_conv3d_descriptors.py --out-dir {out_dir} "
            "--descriptor policy_512_6_242_322_to_256 --dtype bf16 "
            "--tile 2x1 --tile 2x2 --repeats 1 --skip-full "
            f"> {out_dir}/miopen/large_conv3d_probe.log 2>&1"
        )
        return docker_base(extra_env), inner
    raise ValueError(f"unknown case: {case}")


def _sample_rocm_smi() -> dict:
    completed = subprocess.run(
        ["rocm-smi", "--showuse", "--showmemuse", "--showclocks", "--showpower", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    data = {
        "timestamp": time.time(),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }
    try:
        data["rocm_smi"] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        data["stdout"] = completed.stdout.strip()
    return data


def run_with_rocm_smi(cmd: list[str], log_path: Path, interval: float) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def monitor() -> None:
        with log_path.open("w") as handle:
            while not stop.is_set():
                handle.write(json.dumps(_sample_rocm_smi(), ensure_ascii=False) + "\n")
                handle.flush()
                stop.wait(interval)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        subprocess.run(cmd, check=True)
    finally:
        stop.set()
        thread.join(timeout=interval + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), action="append", default=[])
    parser.add_argument(
        "--case",
        choices=[
            "tech_validate",
            "t2i_smoke",
            "t2i_article",
            "t2i_article_warm_full",
            "t2i_article_warm_full_rocprof",
            "t2i_article_transformer_rocprof",
            "t2v_i2v_smoke",
            "t2v_i2v_stage_smoke",
            "vae_decode_probe",
            "vae_warmup_default",
            "vae_warmup_tiling",
            "vae_warmup_slicing",
            "vae_warmup_t2v_twice",
            "t2v_i2v_article",
            "t2v_i2v_article_warm_full",
            "t2v_article_warm_full",
            "i2v_article_warm_full",
            "i2v_article_und_cache_warm_full",
            "i2v_article_runtime_no_profile",
            "i2v_article_transformer_streamk_probe",
            "i2v_article_warm_full_deep_tune",
            "t2v_article_transformer_rocprof",
            "i2v_article_transformer_rocprof",
            "t2v_article_gemm_log",
            "i2v_article_gemm_log",
            "policy_article",
            "policy_article_rocprof",
            "policy_article_skip_vision_decode",
            "policy_article_action_only",
            "policy_article_miopen_find",
            "large_conv3d_probe",
            "large_conv3d_probe_512_bf16",
        ],
        action="append",
        default=[],
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rocm-smi-log-dir", default="")
    parser.add_argument("--rocm-smi-interval", type=float, default=5.0)
    args = parser.parse_args()

    variants = args.variant or ["v1_0", "aotriton", "tunable_collect"]
    cases = args.case or ["tech_validate", "t2i_smoke", "t2v_i2v_smoke"]
    for variant in variants:
        for case in cases:
            base, inner = case_command(case, variant)
            cmd = [*base, inner]
            printable = " ".join(shlex.quote(part) for part in cmd)
            print(f"\n# variant={variant} case={case}\n{printable}", flush=True)
            if args.execute:
                if args.rocm_smi_log_dir:
                    log_path = Path(args.rocm_smi_log_dir) / variant / case / "rocm_smi.jsonl"
                    run_with_rocm_smi(cmd, log_path, args.rocm_smi_interval)
                else:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
