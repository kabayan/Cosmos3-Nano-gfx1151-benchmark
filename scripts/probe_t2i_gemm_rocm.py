import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


SHAPES = [
    ("t2i_mlp_up_gate_2141", 2141, 4096, 12288),
    ("t2i_mlp_down_2141", 2141, 12288, 4096),
    ("t2i_attn_hidden_2141", 2141, 4096, 4096),
    ("t2i_attn_qkv_2141", 2141, 4096, 1024),
    ("t2i_time_mlp_up_gate_900", 900, 4096, 12288),
    ("t2i_time_mlp_down_900", 900, 12288, 4096),
    ("t2i_time_hidden_900", 900, 4096, 4096),
    ("t2i_time_qkv_900", 900, 4096, 1024),
]


def sync() -> None:
    torch.cuda.synchronize()


def bench_shape(name: str, m: int, k: int, n: int, warmup: int, repeats: int) -> dict:
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    w = torch.randn((n, k), device="cuda", dtype=torch.float16)
    for _ in range(warmup):
        y = F.linear(x, w)
    sync()
    started = time.perf_counter()
    for _ in range(repeats):
        y = F.linear(x, w)
    sync()
    seconds = time.perf_counter() - started
    checksum = float(y.float().sum().item())
    del x, w, y
    torch.cuda.empty_cache()
    return {
        "name": name,
        "m": m,
        "k": k,
        "n": n,
        "repeats": repeats,
        "seconds": round(seconds, 6),
        "average_ms": round(seconds * 1000 / repeats, 6),
        "checksum": round(checksum, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()

    data = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": [bench_shape(*shape, args.warmup, args.repeats) for shape in SHAPES],
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(data, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
