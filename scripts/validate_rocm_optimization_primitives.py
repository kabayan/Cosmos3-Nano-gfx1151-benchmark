import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return value, round(time.perf_counter() - started, 6)


def status(name: str, ok: bool, **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": ok, **extra}


def env_probe() -> list[dict[str, Any]]:
    rows = [
        status("torch_import", True, version=torch.__version__, hip=torch.version.hip),
        status("cuda_available", torch.cuda.is_available()),
    ]
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        rows.append(
            status(
                "cuda_device",
                True,
                device=torch.cuda.get_device_name(0),
                free_gib=round(free / 1024**3, 3),
                total_gib=round(total / 1024**3, 3),
            )
        )
    rows.append(
        status(
            "env_flags",
            True,
            TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
            PYTORCH_TUNABLEOP_ENABLED=os.environ.get("PYTORCH_TUNABLEOP_ENABLED"),
            PYTORCH_TUNABLEOP_TUNING=os.environ.get("PYTORCH_TUNABLEOP_TUNING"),
            PYTORCH_TUNABLEOP_RECORD_UNTUNED=os.environ.get("PYTORCH_TUNABLEOP_RECORD_UNTUNED"),
            PYTORCH_TUNABLEOP_FILENAME=os.environ.get("PYTORCH_TUNABLEOP_FILENAME"),
            PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        )
    )
    return rows


def sdpa_probe(device: torch.device) -> list[dict[str, Any]]:
    rows = []
    q = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)
    k = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)
    v = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)

    def run_default():
        return F.scaled_dot_product_attention(q, k, v)

    try:
        out, seconds = timed(run_default)
        rows.append(status("sdpa_default", bool(torch.isfinite(out).all()), seconds=seconds, shape=list(out.shape)))
    except Exception as exc:
        rows.append(status("sdpa_default", False, error=repr(exc)))

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend_names = ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"]
        for backend_name in backend_names:
            backend = getattr(SDPBackend, backend_name, None)
            if backend is None:
                rows.append(status(f"sdpa_backend_{backend_name.lower()}", False, error="backend enum missing"))
                continue

            def run_backend():
                with sdpa_kernel(backends=[backend]):
                    return F.scaled_dot_product_attention(q, k, v)

            try:
                out, seconds = timed(run_backend)
                rows.append(
                    status(
                        f"sdpa_backend_{backend_name.lower()}",
                        bool(torch.isfinite(out).all()),
                        seconds=seconds,
                        shape=list(out.shape),
                    )
                )
            except Exception as exc:
                rows.append(status(f"sdpa_backend_{backend_name.lower()}", False, error=repr(exc)))
    except Exception as exc:
        rows.append(status("sdpa_backend_import", False, error=repr(exc)))

    q_gqa = torch.randn(1, 32, 64, 64, device=device, dtype=torch.float16)
    k_gqa = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)
    v_gqa = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)
    try:
        out, seconds = timed(lambda: F.scaled_dot_product_attention(q_gqa, k_gqa, v_gqa, enable_gqa=True))
        rows.append(status("sdpa_gqa", bool(torch.isfinite(out).all()), seconds=seconds, shape=list(out.shape)))
    except Exception as exc:
        rows.append(status("sdpa_gqa", False, error=repr(exc)))
    return rows


def tunableop_probe(device: torch.device) -> list[dict[str, Any]]:
    rows = []
    try:
        import torch.cuda.tunable as tunable

        rows.append(status("tunable_import", True, module=str(tunable)))
    except Exception as exc:
        rows.append(status("tunable_import", False, error=repr(exc)))
        return rows

    a = torch.randn(512, 1024, device=device, dtype=torch.float16)
    w = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    try:
        out, seconds = timed(lambda: F.linear(a, w))
        rows.append(status("tunable_linear_probe", bool(torch.isfinite(out).all()), seconds=seconds, shape=list(out.shape)))
    except Exception as exc:
        rows.append(status("tunable_linear_probe", False, error=repr(exc)))
    return rows


def compile_probe(device: torch.device) -> list[dict[str, Any]]:
    rows = []
    try:
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.SiLU(),
            torch.nn.Linear(1024, 512),
        ).to(device=device, dtype=torch.float16)
        compiled = torch.compile(model, mode="reduce-overhead")
        x = torch.randn(32, 512, device=device, dtype=torch.float16)
        _, first_seconds = timed(lambda: compiled(x))
        out, second_seconds = timed(lambda: compiled(x))
        rows.append(
            status(
                "torch_compile_probe",
                bool(torch.isfinite(out).all()),
                first_seconds=first_seconds,
                second_seconds=second_seconds,
                shape=list(out.shape),
            )
        )
    except Exception as exc:
        rows.append(status("torch_compile_probe", False, error=repr(exc)))
    return rows


def policy_fallback_probe(device: torch.device) -> list[dict[str, Any]]:
    rows = []
    try:
        from run_cosmos_framework_policy_rocm import _sdpa_varlen_fallback
    except Exception as exc:
        return [status("policy_fallback_import", False, error=repr(exc))]

    q = torch.randn(1, 96, 32, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 96, 8, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 96, 8, 64, device=device, dtype=torch.float16)
    offsets = torch.tensor([0, 32, 96], device=device, dtype=torch.int32)
    try:
        out, seconds = timed(
            lambda: _sdpa_varlen_fallback(
                q,
                k,
                v,
                is_causal=True,
                cumulative_seqlen_Q=offsets,
                cumulative_seqlen_KV=offsets,
                max_seqlen_Q=64,
                max_seqlen_KV=64,
            )
        )
        rows.append(
            status(
                "policy_fallback_varlen_gqa",
                bool(torch.isfinite(out).all()),
                seconds=seconds,
                shape=list(out.shape),
            )
        )
    except Exception as exc:
        rows.append(status("policy_fallback_varlen_gqa", False, error=repr(exc)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/workspace/result/rocm_speed_matrix/tech_validate.json")
    parser.add_argument("--include-compile", action="store_true")
    args = parser.parse_args()

    rows = []
    rows.extend(env_probe())
    if torch.cuda.is_available():
        device = torch.device("cuda")
        rows.extend(sdpa_probe(device))
        rows.extend(tunableop_probe(device))
        rows.extend(policy_fallback_probe(device))
        if args.include_compile:
            rows.extend(compile_probe(device))

    report = {
        "ok": all(row["ok"] for row in rows if row["name"] not in {"sdpa_backend_flash_attention", "sdpa_backend_efficient_attention"}),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
