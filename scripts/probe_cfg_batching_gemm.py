"""CFG バッチ化の利得を GEMM マイクロベンチで先行確認する（DLS-017 PoC）。

問い: CFG の cond / uncond を 1 回の forward にバッチ化すると、重み読み出しが
償却されて「2 倍未満」のコストで済むか。済むなら CFG 条件での per-call 短縮に
構造的な余地があり、済まないなら計算量律速で余地は無い。

形状は result/rocm_speed_matrix/tunableop_results0.csv の実測形状から採る
（Cosmos3 transformer の Linear: hidden 4096、MoT/FFN 12288、qkv 1024/192）。
トークン数 N は 261 / 672 / 900 / 1904 / 2141 が実際に現れる。

TunableOp は N と 2N で調律状態が非対称になるため本プローブでは無効化して測る
（素の hipBLASLt/rocBLAS 選択で比較の土台を揃える）。

使い方:
    python scripts/probe_cfg_batching_gemm.py --out result/cfg_batch_probe/gemm.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch


# (name, K_in, M_out) — y[N, M] = x[N, K] @ W[K, M]
LAYERS = [
    ("qkv_proj_1024", 4096, 1024),
    ("attn_out_4096", 4096, 4096),
    ("ffn_up_12288", 4096, 12288),
    ("ffn_down_4096", 12288, 4096),
]
TOKENS = [261, 672, 900, 1904, 2141]


def bench(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="result/cfg_batch_probe/gemm.json")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    records = []
    for name, k_in, m_out in LAYERS:
        w = torch.randn(k_in, m_out, device=dev, dtype=dtype)
        w_bytes = w.numel() * w.element_size()
        for n in TOKENS:
            row = {"layer": name, "K": k_in, "M": m_out, "N": n, "weight_bytes": w_bytes}
            for label, tokens in (("n", n), ("2n", 2 * n)):
                x = torch.randn(tokens, k_in, device=dev, dtype=dtype)
                sec = bench(lambda: torch.mm(x, w))
                flops = 2 * tokens * k_in * m_out
                act_bytes = (x.numel() * x.element_size()) + (tokens * m_out * w.element_size())
                row[f"{label}_seconds"] = sec
                row[f"{label}_tflops"] = flops / sec / 1e12
                row[f"{label}_gbps"] = (w_bytes + act_bytes) / sec / 1e9
                del x
            row["ratio_2n_over_n"] = row["2n_seconds"] / row["n_seconds"]
            records.append(row)
            print(
                f"{name:16s} N={n:5d} "
                f"n={row['n_seconds']*1e3:7.3f}ms ({row['n_tflops']:5.2f} TF, {row['n_gbps']:6.1f} GB/s)  "
                f"2n={row['2n_seconds']*1e3:7.3f}ms ({row['2n_tflops']:5.2f} TF, {row['2n_gbps']:6.1f} GB/s)  "
                f"ratio={row['ratio_2n_over_n']:.3f}",
                flush=True,
            )
        del w
        torch.cuda.empty_cache()

    ratios = [r["ratio_2n_over_n"] for r in records]
    summary = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "tunableop_enabled": os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "unset"),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "ratio_mean": sum(ratios) / len(ratios),
        "interpretation": (
            "ratio<<2.0 means doubling the token count is cheaper than two separate calls, "
            "so batching CFG cond+uncond amortizes weight reads. ratio~=2.0 means compute-bound "
            "with no headroom from batching."
        ),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nratio mean={summary['ratio_mean']:.3f} min={summary['ratio_min']:.3f} max={summary['ratio_max']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
