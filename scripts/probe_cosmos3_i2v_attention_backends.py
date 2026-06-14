import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


BACKENDS = {
    "default": None,
    "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    "math": torch.nn.attention.SDPBackend.MATH,
}


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeats: int) -> dict:
    values = []
    out = None
    for _ in range(repeats):
        sync()
        started = time.perf_counter()
        out = fn()
        sync()
        values.append(time.perf_counter() - started)
    assert out is not None
    return {
        "seconds": round(sum(values), 6),
        "average_ms": round(sum(values) / len(values) * 1000, 6),
        "min_ms": round(min(values) * 1000, 6),
        "max_ms": round(max(values) * 1000, 6),
        "finite": bool(torch.isfinite(out).all().item()),
        "shape": list(out.shape),
    }


def run_attention(query, key, value, *, is_causal: bool, enable_gqa: bool, backend):
    def call():
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

    if backend is None:
        return call()
    with torch.nn.attention.sdpa_kernel(backend):
        return call()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--und-len", type=int, default=237)
    parser.add_argument("--gen-len", type=int, default=1904)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=203)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16
    all_len = args.und_len + args.gen_len
    shapes = {
        "causal_und": {
            "q": [1, args.heads, args.und_len, args.head_dim],
            "k": [1, args.kv_heads, args.und_len, args.head_dim],
            "v": [1, args.kv_heads, args.und_len, args.head_dim],
            "is_causal": True,
            "enable_gqa": True,
        },
        "full_gen": {
            "q": [1, args.heads, args.gen_len, args.head_dim],
            "k": [1, args.kv_heads, all_len, args.head_dim],
            "v": [1, args.kv_heads, all_len, args.head_dim],
            "is_causal": False,
            "enable_gqa": True,
        },
        "full_gen_expanded_kv": {
            "q": [1, args.heads, args.gen_len, args.head_dim],
            "k": [1, args.heads, all_len, args.head_dim],
            "v": [1, args.heads, all_len, args.head_dim],
            "is_causal": False,
            "enable_gqa": False,
        },
    }

    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "shapes": shapes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": {},
    }

    for case_name, spec in shapes.items():
        q = torch.randn(spec["q"], device=device, dtype=dtype)
        if case_name == "full_gen_expanded_kv":
            base_k = torch.randn([1, args.kv_heads, all_len, args.head_dim], device=device, dtype=dtype)
            base_v = torch.randn([1, args.kv_heads, all_len, args.head_dim], device=device, dtype=dtype)
            repeat = args.heads // args.kv_heads
            k = base_k.repeat_interleave(repeat, dim=1).contiguous()
            v = base_v.repeat_interleave(repeat, dim=1).contiguous()
        else:
            k = torch.randn(spec["k"], device=device, dtype=dtype)
            v = torch.randn(spec["v"], device=device, dtype=dtype)
        report["results"][case_name] = {}
        for backend_name, backend in BACKENDS.items():
            try:
                for _ in range(args.warmup):
                    run_attention(q, k, v, is_causal=spec["is_causal"], enable_gqa=spec["enable_gqa"], backend=backend)
                result = timed(
                    lambda: run_attention(
                        q, k, v, is_causal=spec["is_causal"], enable_gqa=spec["enable_gqa"], backend=backend
                    ),
                    args.repeats,
                )
                report["results"][case_name][backend_name] = result
            except Exception as exc:
                report["results"][case_name][backend_name] = {"error": repr(exc)}
        del q, k, v

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
