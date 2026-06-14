import argparse
import csv
import json
from bisect import bisect_right
from pathlib import Path


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if name.startswith("Cijk_") or "tensile" in lowered or "hipblaslt" in lowered:
        return "gemm"
    if "attn_fwd" in lowered:
        return "attention"
    if "copy" in lowered or "copybuffer" in lowered or "fill" in lowered:
        return "copy_fill"
    if "reduce_kernel" in lowered or "meanops" in lowered:
        return "reduce_mean"
    if "pow_tensor_scalar" in lowered:
        return "pow"
    if "binary_internal::mulfunctor" in lowered:
        return "mul"
    if "silu" in lowered:
        return "silu"
    if "catarray" in lowered or "cat" in lowered:
        return "cat"
    if "gather" in lowered:
        return "gather"
    if "index" in lowered:
        return "index"
    if "elementwise" in lowered or "kernel_impl" in lowered or "vectorized" in lowered:
        return "elementwise_other"
    return "other"


def short_kernel_name(name: str) -> str:
    if name.startswith("Cijk_"):
        if "_SN_" in name:
            return name.split("_SN_", 1)[0] + "..."
        return name[:140] + "..."
    if name == "attn_fwd":
        return name
    for needle in [
        "pow_tensor_scalar",
        "float16_copy_kernel",
        "direct_copy_kernel",
        "BinaryFunctor",
        "MeanOps",
        "FillFunctor",
        "silu",
        "vectorized_gather_kernel",
        "index_elementwise_kernel",
    ]:
        if needle in name:
            return needle
    return name[:160]


def read_markers(path: Path) -> list[dict]:
    markers = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Function"] != "transformer_forward":
                continue
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            markers.append(
                {
                    "step": len(markers) + 1,
                    "start": start,
                    "end": end,
                    "wall_seconds": (end - start) / 1_000_000_000,
                    "categories": {},
                    "kernels": {},
                    "kernel_count": 0,
                    "kernel_seconds": 0.0,
                }
            )
    markers.sort(key=lambda item: item["start"])
    return markers


def add_duration(bucket: dict, key: str, seconds: float) -> None:
    entry = bucket.setdefault(key, {"seconds": 0.0, "calls": 0})
    entry["seconds"] += seconds
    entry["calls"] += 1


def assign_kernels(markers: list[dict], trace_path: Path) -> None:
    starts = [m["start"] for m in markers]
    with trace_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            idx = bisect_right(starts, start) - 1
            if idx < 0:
                continue
            marker = markers[idx]
            if end > marker["end"]:
                continue
            name = row["Kernel_Name"]
            seconds = (end - start) / 1_000_000_000
            category = classify_kernel(name)
            short_name = short_kernel_name(name)
            add_duration(marker["categories"], category, seconds)
            add_duration(marker["kernels"], short_name, seconds)
            marker["kernel_count"] += 1
            marker["kernel_seconds"] += seconds


def summarize(markers: list[dict], top: int) -> dict:
    category_totals: dict[str, dict] = {}
    kernel_totals: dict[str, dict] = {}
    wall_seconds = 0.0
    kernel_seconds = 0.0
    for marker in markers:
        wall_seconds += marker["wall_seconds"]
        kernel_seconds += marker["kernel_seconds"]
        for category, value in marker["categories"].items():
            entry = category_totals.setdefault(category, {"seconds": 0.0, "calls": 0})
            entry["seconds"] += value["seconds"]
            entry["calls"] += value["calls"]
        for name, value in marker["kernels"].items():
            entry = kernel_totals.setdefault(name, {"seconds": 0.0, "calls": 0})
            entry["seconds"] += value["seconds"]
            entry["calls"] += value["calls"]

    def finish_totals(values: dict[str, dict]) -> list[dict]:
        rows = []
        for name, value in values.items():
            seconds = value["seconds"]
            calls = value["calls"]
            rows.append(
                {
                    "name": name,
                    "seconds": round(seconds, 6),
                    "calls": calls,
                    "average_ms": round(seconds / calls * 1000, 6) if calls else 0.0,
                    "share_of_kernel_seconds": round(seconds / kernel_seconds * 100, 3)
                    if kernel_seconds
                    else 0.0,
                    "seconds_per_step": round(seconds / len(markers), 6) if markers else 0.0,
                    "calls_per_step": round(calls / len(markers), 3) if markers else 0.0,
                }
            )
        rows.sort(key=lambda item: item["seconds"], reverse=True)
        return rows

    step_rows = []
    for marker in markers:
        categories = {
            name: {
                "seconds": round(value["seconds"], 6),
                "calls": value["calls"],
                "share_of_step_kernel_seconds": round(
                    value["seconds"] / marker["kernel_seconds"] * 100, 3
                )
                if marker["kernel_seconds"]
                else 0.0,
            }
            for name, value in sorted(
                marker["categories"].items(), key=lambda item: item[1]["seconds"], reverse=True
            )
        }
        top_kernels = []
        for name, value in sorted(
            marker["kernels"].items(), key=lambda item: item[1]["seconds"], reverse=True
        )[:top]:
            top_kernels.append(
                {
                    "name": name,
                    "seconds": round(value["seconds"], 6),
                    "calls": value["calls"],
                }
            )
        step_rows.append(
            {
                "step": marker["step"],
                "wall_seconds": round(marker["wall_seconds"], 6),
                "kernel_seconds": round(marker["kernel_seconds"], 6),
                "kernel_count": marker["kernel_count"],
                "categories": categories,
                "top_kernels": top_kernels,
            }
        )

    category_rows = finish_totals(category_totals)
    kernel_rows = finish_totals(kernel_totals)
    return {
        "steps": len(markers),
        "wall_seconds": round(wall_seconds, 6),
        "wall_seconds_per_step": round(wall_seconds / len(markers), 6) if markers else 0.0,
        "kernel_seconds": round(kernel_seconds, 6),
        "kernel_seconds_per_step": round(kernel_seconds / len(markers), 6) if markers else 0.0,
        "category_totals": category_rows,
        "top_kernels": kernel_rows[:top],
        "per_step": step_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-trace", type=Path, required=True)
    parser.add_argument("--kernel-trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    markers = read_markers(args.marker_trace)
    assign_kernels(markers, args.kernel_trace)
    report = summarize(markers, args.top)
    report["marker_trace"] = str(args.marker_trace)
    report["kernel_trace"] = str(args.kernel_trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ["steps", "wall_seconds", "wall_seconds_per_step", "kernel_seconds", "kernel_seconds_per_step"]}, indent=2))
    print("Top categories:")
    for row in report["category_totals"][:8]:
        print(row)
    print("Top kernels:")
    for row in report["top_kernels"][:8]:
        print(row)


if __name__ == "__main__":
    main()
