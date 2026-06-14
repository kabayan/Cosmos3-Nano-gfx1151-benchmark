import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def timed_repeated(fn, repeats: int):
    out = None
    seconds = []
    for _ in range(repeats):
        out, elapsed = timed(fn)
        seconds.append(elapsed)
    return out, seconds


def pad_input(x: torch.Tensor, causal_padding: tuple[int, ...], cache_x: torch.Tensor | None = None) -> torch.Tensor:
    padding = list(causal_padding)
    if cache_x is not None:
        cache_x = cache_x.to(x.device)
        if padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] = max(0, padding[4] - cache_x.shape[2])
    return F.pad(x, tuple(padding))


def conv_baseline(x, weight, bias, causal_padding, stride, dilation, groups, cache_x=None):
    x_pad = pad_input(x, causal_padding, cache_x)
    return F.conv3d(x_pad, weight, bias, stride=stride, padding=0, dilation=dilation, groups=groups)


def conv_channels_last_3d(x, weight, bias, causal_padding, stride, dilation, groups, cache_x=None):
    x_pad = pad_input(x, causal_padding, cache_x).contiguous(memory_format=torch.channels_last_3d)
    return F.conv3d(x_pad, weight, bias, stride=stride, padding=0, dilation=dilation, groups=groups)


def conv_chunk_h(x, weight, bias, causal_padding, stride, dilation, groups, chunks: int, cache_x=None):
    if stride != (1, 1, 1) or dilation != (1, 1, 1) or weight.shape[2:] != (3, 3, 3):
        raise ValueError("chunk_h currently supports stride=1, dilation=1, kernel=3 only")
    x_pad = pad_input(x, causal_padding, cache_x)
    out_h = x.shape[3]
    pieces = []
    for idx in range(chunks):
        h0 = out_h * idx // chunks
        h1 = out_h * (idx + 1) // chunks
        x_part = x_pad[:, :, :, h0 : h1 + 2, :]
        pieces.append(F.conv3d(x_part, weight, bias, stride=stride, padding=0, dilation=dilation, groups=groups))
    return torch.cat(pieces, dim=3)


def conv_chunk_w(x, weight, bias, causal_padding, stride, dilation, groups, chunks: int, cache_x=None):
    if stride != (1, 1, 1) or dilation != (1, 1, 1) or weight.shape[2:] != (3, 3, 3):
        raise ValueError("chunk_w currently supports stride=1, dilation=1, kernel=3 only")
    x_pad = pad_input(x, causal_padding, cache_x)
    out_w = x.shape[4]
    pieces = []
    for idx in range(chunks):
        w0 = out_w * idx // chunks
        w1 = out_w * (idx + 1) // chunks
        x_part = x_pad[:, :, :, :, w0 : w1 + 2]
        pieces.append(F.conv3d(x_part, weight, bias, stride=stride, padding=0, dilation=dilation, groups=groups))
    return torch.cat(pieces, dim=4)


def conv_chunk_hw(x, weight, bias, causal_padding, stride, dilation, groups, h_chunks: int, w_chunks: int, cache_x=None):
    if stride != (1, 1, 1) or dilation != (1, 1, 1) or weight.shape[2:] != (3, 3, 3):
        raise ValueError("chunk_hw currently supports stride=1, dilation=1, kernel=3 only")
    x_pad = pad_input(x, causal_padding, cache_x)
    out_h, out_w = x.shape[3], x.shape[4]
    h_rows = []
    for hi in range(h_chunks):
        h0 = out_h * hi // h_chunks
        h1 = out_h * (hi + 1) // h_chunks
        w_cols = []
        for wi in range(w_chunks):
            w0 = out_w * wi // w_chunks
            w1 = out_w * (wi + 1) // w_chunks
            x_part = x_pad[:, :, :, h0 : h1 + 2, w0 : w1 + 2]
            w_cols.append(F.conv3d(x_part, weight, bias, stride=stride, padding=0, dilation=dilation, groups=groups))
        h_rows.append(torch.cat(w_cols, dim=4))
    return torch.cat(h_rows, dim=3)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--save-baseline-output", default=None)
    parser.add_argument("--variants", nargs="*", default=["baseline", "channels_last_3d", "chunk_h2", "chunk_h4", "chunk_w2", "chunk_hw2x2"])
    args = parser.parse_args()
    repeats = max(1, args.repeats)

    payload = torch.load(args.probe, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    x = payload["input"].to(device)
    weight = payload["weight"].to(device)
    bias = payload["bias"].to(device) if payload["bias"] is not None else None
    cache_x = payload.get("cache_input")
    cache_x = cache_x.to(device) if cache_x is not None else None
    causal_padding = tuple(payload["causal_padding"])
    stride = tuple(payload["stride"])
    dilation = tuple(payload["dilation"])
    groups = int(payload["groups"])

    results = {
        "probe": str(args.probe),
        "name": payload.get("name"),
        "input_shape": list(x.shape),
        "cache_shape": list(cache_x.shape) if cache_x is not None else None,
        "cache_index": payload.get("cache_index"),
        "weight_shape": list(weight.shape),
        "dtype": str(x.dtype),
        "causal_padding": list(causal_padding),
        "stride": list(stride),
        "dilation": list(dilation),
        "groups": groups,
        "repeats": repeats,
        "variants": [],
    }

    with torch.inference_mode():
        baseline, cold_baseline_sec = timed(lambda: conv_baseline(x, weight, bias, causal_padding, stride, dilation, groups, cache_x))
        _, warm_baseline_times = timed_repeated(
            lambda: conv_baseline(x, weight, bias, causal_padding, stride, dilation, groups, cache_x), repeats
        )
        warm_baseline_sec = min(warm_baseline_times)
        results["baseline_seconds"] = cold_baseline_sec
        results["cold_reference_baseline_seconds"] = cold_baseline_sec
        results["warm_baseline_seconds"] = warm_baseline_sec
        results["warm_baseline_seconds_all"] = warm_baseline_times
        results["warm_baseline_seconds_median"] = statistics.median(warm_baseline_times)
        results["output_shape"] = list(baseline.shape)
        if args.save_baseline_output:
            output_path = Path(args.save_baseline_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "output": baseline.detach().cpu(),
                    "probe": str(args.probe),
                    "name": payload.get("name"),
                    "cold_reference_baseline_seconds": cold_baseline_sec,
                    "warm_baseline_seconds": warm_baseline_sec,
                },
                output_path,
            )
            results["saved_baseline_output"] = str(output_path)

        fns = {
            "baseline": lambda: conv_baseline(x, weight, bias, causal_padding, stride, dilation, groups, cache_x),
            "channels_last_3d": lambda: conv_channels_last_3d(x, weight, bias, causal_padding, stride, dilation, groups, cache_x),
            "chunk_h2": lambda: conv_chunk_h(x, weight, bias, causal_padding, stride, dilation, groups, 2, cache_x),
            "chunk_h4": lambda: conv_chunk_h(x, weight, bias, causal_padding, stride, dilation, groups, 4, cache_x),
            "chunk_h8": lambda: conv_chunk_h(x, weight, bias, causal_padding, stride, dilation, groups, 8, cache_x),
            "chunk_w2": lambda: conv_chunk_w(x, weight, bias, causal_padding, stride, dilation, groups, 2, cache_x),
            "chunk_w4": lambda: conv_chunk_w(x, weight, bias, causal_padding, stride, dilation, groups, 4, cache_x),
            "chunk_hw2x2": lambda: conv_chunk_hw(x, weight, bias, causal_padding, stride, dilation, groups, 2, 2, cache_x),
            "chunk_hw4x2": lambda: conv_chunk_hw(x, weight, bias, causal_padding, stride, dilation, groups, 4, 2, cache_x),
        }
        for name in args.variants:
            entry = {"name": name}
            try:
                out, times = timed_repeated(fns[name], repeats)
                seconds = min(times)
                diff = max_abs_diff(baseline, out)
                entry.update(
                    {
                        "seconds": seconds,
                        "seconds_all": times,
                        "seconds_median": statistics.median(times),
                        "speedup_vs_baseline": warm_baseline_sec / seconds if seconds > 0 else None,
                        "speedup_vs_warm_baseline": warm_baseline_sec / seconds if seconds > 0 else None,
                        "speedup_vs_cold_reference": cold_baseline_sec / seconds if seconds > 0 else None,
                        "max_abs_diff": diff,
                        "exact_match": diff == 0.0,
                        "shape": list(out.shape),
                    }
                )
            except Exception as exc:
                entry.update({"error": repr(exc)})
            results["variants"].append(entry)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
