import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def sync() -> None:
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
        "finite": bool(torch.isfinite(out).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--m", type=int, nargs="+", default=[672, 1904, 2141])
    args = parser.parse_args()

    torch.manual_seed(7)
    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "env": {
            key: os.environ.get(key)
            for key in [
                "TENSILE_SOLUTION_SELECTION_METHOD",
                "ROCBLAS_USE_HIPBLASLT",
                "PYTORCH_TUNABLEOP_ENABLED",
                "PYTORCH_TUNABLEOP_TUNING",
                "PYTORCH_TUNABLEOP_FILENAME",
            ]
        },
        "repeats": args.repeats,
        "warmup": args.warmup,
        "shapes": {},
    }

    for m in args.m:
        x = torch.randn(m, 12288, device="cuda", dtype=torch.float16)
        w = torch.randn(4096, 12288, device="cuda", dtype=torch.float16) * 0.01
        report["shapes"][f"tn_4096_{m}_12288"] = bench(lambda: F.linear(x, w), args.repeats, args.warmup)
        del x, w

        x = torch.randn(m, 4096, device="cuda", dtype=torch.float16)
        w = torch.randn(12288, 4096, device="cuda", dtype=torch.float16) * 0.01
        report["shapes"][f"tn_12288_{m}_4096"] = bench(lambda: F.linear(x, w), args.repeats, args.warmup)
        del x, w

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
