import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_shape(params: str) -> dict[str, int | str]:
    parts = params.split("_")
    if len(parts) < 8:
        return {"raw": params}
    layout = parts[0]
    n = int(parts[1])
    m = int(parts[2])
    k = int(parts[3])
    return {"layout": layout, "n": n, "m": m, "k": k}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    validators = []
    rows = []
    with args.csv_file.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0] == "Validator":
                validators.append(row)
                continue
            if len(row) < 4:
                continue
            shape = parse_shape(row[1])
            rows.append(
                {
                    "op": row[0],
                    "params": row[1],
                    "solution": row[2],
                    "time_ms": float(row[3]),
                    **shape,
                }
            )

    by_m = defaultdict(lambda: {"count": 0, "entries": [], "time_ms_sum": 0.0})
    by_solution = defaultdict(lambda: {"count": 0, "time_ms_sum": 0.0})
    for row in rows:
        m = row.get("m", "unknown")
        by_m[str(m)]["count"] += 1
        by_m[str(m)]["time_ms_sum"] += float(row["time_ms"])
        by_m[str(m)]["entries"].append(row)

        solution_family = str(row["solution"]).split("_")[1] if "_" in str(row["solution"]) else str(row["solution"])
        by_solution[solution_family]["count"] += 1
        by_solution[solution_family]["time_ms_sum"] += float(row["time_ms"])

    result = {
        "source": str(args.csv_file),
        "validators": validators,
        "entry_count": len(rows),
        "by_sequence_m": {
            key: {
                "count": value["count"],
                "time_ms_sum": round(value["time_ms_sum"], 3),
                "entries": sorted(value["entries"], key=lambda item: float(item["time_ms"]), reverse=True),
            }
            for key, value in sorted(by_m.items(), key=lambda item: int(item[0]) if item[0].isdigit() else -1)
        },
        "by_solution_family": {
            key: {"count": value["count"], "time_ms_sum": round(value["time_ms_sum"], 3)}
            for key, value in sorted(by_solution.items(), key=lambda item: item[0])
        },
        "top_by_time": sorted(rows, key=lambda item: float(item["time_ms"]), reverse=True)[:20],
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
