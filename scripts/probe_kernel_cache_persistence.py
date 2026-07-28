"""カーネルキャッシュ永続化の実効確認プローブ。

`scripts/run_cosmos_framework_policy_docker.sh` が MIOpen / inductor / triton の
キャッシュをホストへ逃がしている。その効果は「2 回目の起動が速いか」でしか
確認できないため、Policy 本体（1 run 25〜40 分）ではなく小さい conv3d で測る。

同一プロセス内の 2 回目ではなく、**コンテナを分けた 2 回目**が速くなることが
確認したい性質なので、このスクリプトは 1 回分だけを測って JSON に出す。
呼び出し側で 2 回実行して比較する。

使い方（ラッパー経由ではなく直接 docker run で呼ぶ）:
    python3 scripts/probe_kernel_cache_persistence.py --out /workspace/result/kcache/run1.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU が見えない")

    # Policy の VAE encode に近い形状の conv3d。MIOpen のアルゴリズム探索を誘発する。
    shapes = [
        (1, 3, 17, 544, 736),
        (1, 96, 5, 136, 184),
    ]

    torch.manual_seed(0)
    results = []
    total_start = time.perf_counter()
    for in_shape in shapes:
        c_in = in_shape[1]
        x = torch.randn(*in_shape, dtype=torch.bfloat16, device="cuda")
        conv = torch.nn.Conv3d(c_in, 96, kernel_size=3, padding=1).to("cuda", torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            y = conv(x)
        torch.cuda.synchronize()
        first = time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):
                y = conv(x)
        torch.cuda.synchronize()
        steady = (time.perf_counter() - t1) / 3

        results.append(
            {
                "input_shape": list(in_shape),
                "first_call_seconds": round(first, 4),
                "steady_call_seconds": round(steady, 4),
                "out_sum": float(y.float().sum().item()),
            }
        )

    payload = {
        "total_seconds": round(time.perf_counter() - total_start, 4),
        "device": torch.cuda.get_device_name(0),
        "miopen_cache_dir_env": os.environ.get("MIOPEN_USER_DB_PATH", "(default)"),
        "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR", "(default)"),
        "miopen_cache_files": sorted(
            str(p.relative_to(Path("/root/.cache/miopen")))
            for p in Path("/root/.cache/miopen").rglob("*")
            if p.is_file()
        )
        if Path("/root/.cache/miopen").is_dir()
        else [],
        "shapes": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
