import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    values = []
    out = None
    for _ in range(repeats):
        sync()
        started = time.perf_counter()
        out = fn()
        sync()
        values.append(time.perf_counter() - started)
    return {
        "average_ms": round(sum(values) / len(values) * 1000, 6),
        "min_ms": round(min(values) * 1000, 6),
        "max_ms": round(max(values) * 1000, 6),
        "shape": list(out.shape) if hasattr(out, "shape") else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--m", type=int, nargs="+", default=[237, 1904, 2141])
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16
    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "results": {},
    }

    torch.manual_seed(203)
    for m in args.m:
        result = {}
        x = torch.randn(m, 4096, device=device, dtype=dtype)

        gate = torch.randn(12288, 4096, device=device, dtype=dtype) * 0.01
        up = torch.randn(12288, 4096, device=device, dtype=dtype) * 0.01
        down = torch.randn(4096, 12288, device=device, dtype=dtype) * 0.01
        gate_up = torch.cat([gate, up], dim=0).contiguous()

        def mlp_separate():
            return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

        def mlp_fused_gate_up():
            combined = F.linear(x, gate_up)
            gate_out, up_out = combined.chunk(2, dim=-1)
            return F.linear(F.silu(gate_out) * up_out, down)

        result["mlp_separate_gate_up"] = bench(mlp_separate, args.repeats, args.warmup)
        result["mlp_fused_gate_up"] = bench(mlp_fused_gate_up, args.repeats, args.warmup)
        del gate, up, down, gate_up

        q = torch.randn(4096, 4096, device=device, dtype=dtype) * 0.01
        k = torch.randn(1024, 4096, device=device, dtype=dtype) * 0.01
        v = torch.randn(1024, 4096, device=device, dtype=dtype) * 0.01
        qkv = torch.cat([q, k, v], dim=0).contiguous()

        def qkv_separate():
            q_out = F.linear(x, q)
            F.linear(x, k)
            F.linear(x, v)
            return q_out

        def qkv_fused():
            qkv_out = F.linear(x, qkv)
            q_out, _, _ = torch.split(qkv_out, [4096, 1024, 1024], dim=-1)
            return q_out

        result["qkv_separate"] = bench(qkv_separate, args.repeats, args.warmup)
        result["qkv_fused"] = bench(qkv_fused, args.repeats, args.warmup)
        report["results"][str(m)] = result
        del x, q, k, v, qkv

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
