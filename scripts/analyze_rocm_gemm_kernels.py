import argparse
import csv
import json
from pathlib import Path


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if name.startswith("Cijk_") or "tensile" in lowered:
        return "gemm_tensile"
    if "hipblaslt" in lowered:
        return "gemm_hipblaslt"
    if "attn_fwd" in lowered:
        return "attention"
    if "elementwise" in lowered or "kernel_impl" in lowered or "vectorized" in lowered:
        return "elementwise"
    if "copy" in lowered or "fill" in lowered or "catarray" in lowered:
        return "copy_fill"
    return "other"


def read_kernel_stats(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            duration_ns = int(row["TotalDurationNs"])
            rows.append(
                {
                    "name": row["Name"],
                    "category": classify_kernel(row["Name"]),
                    "calls": int(row["Calls"]),
                    "total_seconds": duration_ns / 1_000_000_000,
                    "average_ms": float(row["AverageNs"]) / 1_000_000,
                    "percentage": float(row["Percentage"]),
                }
            )
    rows.sort(key=lambda item: item["total_seconds"], reverse=True)
    return rows


def summarize(path: Path, top: int) -> dict:
    rows = read_kernel_stats(path)
    categories: dict[str, dict[str, float | int]] = {}
    for row in rows:
        category = categories.setdefault(row["category"], {"seconds": 0.0, "calls": 0, "kernels": 0})
        category["seconds"] = float(category["seconds"]) + row["total_seconds"]
        category["calls"] = int(category["calls"]) + row["calls"]
        category["kernels"] = int(category["kernels"]) + 1

    total_seconds = sum(float(value["seconds"]) for value in categories.values())
    for value in categories.values():
        value["seconds"] = round(float(value["seconds"]), 3)
        value["share"] = round(float(value["seconds"]) / total_seconds * 100, 2) if total_seconds else 0.0

    gemm_rows = [row for row in rows if row["category"].startswith("gemm")]
    top_gemm = [
        {
            "name": row["name"],
            "calls": row["calls"],
            "seconds": round(row["total_seconds"], 3),
            "average_ms": round(row["average_ms"], 3),
            "share": round(row["percentage"], 2),
        }
        for row in gemm_rows[:top]
    ]
    return {
        "source": str(path),
        "total_kernel_seconds": round(total_seconds, 3),
        "categories": dict(sorted(categories.items(), key=lambda item: float(item[1]["seconds"]), reverse=True)),
        "top_gemm": top_gemm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = {"files": [summarize(path, args.top) for path in args.stats]}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
