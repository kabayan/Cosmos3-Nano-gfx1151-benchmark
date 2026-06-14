import argparse
import csv
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video


MODEL_ID = "nvidia/Cosmos3-Nano"
DEFAULT_PROMPT = "A mobile robot in a clean warehouse aisle."
DEFAULT_VIDEO_PROMPT = "A mobile robot slowly moves through a clean warehouse aisle."
NEGATIVE_PROMPT = "blurry, distorted, low quality"


CASES = {
    "load_fp16": {
        "kind": "load",
        "dtype": "float16",
        "suite": "core",
        "description": "Pipeline load only, FP16",
    },
    "load_bf16": {
        "kind": "load",
        "dtype": "bfloat16",
        "suite": "extended",
        "description": "Pipeline load only, BF16",
    },
    "t2i_256_fp16_s4_g1": {
        "kind": "generate",
        "media": "image",
        "dtype": "float16",
        "height": 256,
        "width": 448,
        "num_frames": 1,
        "steps": 4,
        "guidance": 1.0,
        "suite": "core",
        "description": "Text-to-image, 256p class, FP16, 4 steps",
    },
    "t2i_480_fp16_s4_g1": {
        "kind": "generate",
        "media": "image",
        "dtype": "float16",
        "height": 480,
        "width": 832,
        "num_frames": 1,
        "steps": 4,
        "guidance": 1.0,
        "suite": "core",
        "description": "Text-to-image baseline, 480p, FP16, 4 steps",
    },
    "t2i_480_bf16_s4_g1": {
        "kind": "generate",
        "media": "image",
        "dtype": "bfloat16",
        "height": 480,
        "width": 832,
        "num_frames": 1,
        "steps": 4,
        "guidance": 1.0,
        "suite": "core",
        "description": "Text-to-image baseline, 480p, BF16, 4 steps",
    },
    "t2i_480_fp16_s8_g1": {
        "kind": "generate",
        "media": "image",
        "dtype": "float16",
        "height": 480,
        "width": 832,
        "num_frames": 1,
        "steps": 8,
        "guidance": 1.0,
        "suite": "core",
        "description": "Text-to-image, 480p, FP16, 8 steps",
    },
    "t2i_480_fp16_s4_g4": {
        "kind": "generate",
        "media": "image",
        "dtype": "float16",
        "height": 480,
        "width": 832,
        "num_frames": 1,
        "steps": 4,
        "guidance": 4.0,
        "suite": "extended",
        "description": "Text-to-image, 480p, FP16, guidance 4",
    },
    "t2v5_256_fp16_s4_g1": {
        "kind": "generate",
        "media": "video",
        "dtype": "float16",
        "height": 256,
        "width": 448,
        "num_frames": 5,
        "fps": 5,
        "steps": 4,
        "guidance": 1.0,
        "suite": "core",
        "description": "Text-to-video, 5 frames, 256p class, FP16",
    },
    "t2v5_480_fp16_s4_g1": {
        "kind": "generate",
        "media": "video",
        "dtype": "float16",
        "height": 480,
        "width": 832,
        "num_frames": 5,
        "fps": 5,
        "steps": 4,
        "guidance": 1.0,
        "suite": "extended",
        "description": "Text-to-video baseline, 5 frames, 480p, FP16",
    },
}


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def mem_gib() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"free_gib": None, "total_gib": None}
    free, total = torch.cuda.mem_get_info()
    return {
        "free_gib": round(free / 1024**3, 3),
        "total_gib": round(total / 1024**3, 3),
    }


def load_pipeline(dtype_name: str) -> Cosmos3OmniPipeline:
    return Cosmos3OmniPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype(dtype_name),
        device_map="cuda",
        enable_safety_checker=False,
    )


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_load_case(case_name: str, case: dict, repeats: int, out_dir: Path, jsonl_path: Path) -> list[dict]:
    rows = []
    for run_idx in range(1, repeats + 1):
        before = mem_gib()
        started = time.perf_counter()
        status = "passed"
        error = ""
        pipe = None
        try:
            pipe = load_pipeline(case["dtype"])
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = repr(exc)
        finally:
            del pipe
            cleanup()
        seconds = round(time.perf_counter() - started, 3)
        after = mem_gib()
        row = {
            "case": case_name,
            "run": run_idx,
            "suite": case["suite"],
            "kind": case["kind"],
            "dtype": case["dtype"],
            "status": status,
            "seconds": seconds,
            "error": error,
            "output": "",
            "before_mem": before,
            "after_mem": after,
        }
        rows.append(row)
        write_jsonl(jsonl_path, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if status != "passed":
            break
    return rows


def run_generate_case(case_name: str, case: dict, repeats: int, out_dir: Path, jsonl_path: Path) -> list[dict]:
    rows = []
    load_started = time.perf_counter()
    pipe = None
    try:
        pipe = load_pipeline(case["dtype"])
        torch.cuda.synchronize()
        load_seconds = round(time.perf_counter() - load_started, 3)

        for run_idx in range(1, repeats + 1):
            seed = run_idx - 1
            before = mem_gib()
            started = time.perf_counter()
            status = "passed"
            error = ""
            output_path = ""
            result = None
            try:
                prompt = DEFAULT_VIDEO_PROMPT if case["media"] == "video" else DEFAULT_PROMPT
                call_kwargs = {
                    "prompt": prompt,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "num_frames": case["num_frames"],
                    "height": case["height"],
                    "width": case["width"],
                    "num_inference_steps": case["steps"],
                    "guidance_scale": case["guidance"],
                    "generator": torch.Generator(device="cuda").manual_seed(seed),
                }
                if case["media"] == "video":
                    call_kwargs["fps"] = case.get("fps", 5)
                result = pipe(**call_kwargs)
                torch.cuda.synchronize()
                stem = f"{case_name}_run{run_idx:02d}"
                if case["media"] == "video":
                    output_path = str(out_dir / f"{stem}.mp4")
                    export_to_video(result.video, output_path, fps=case.get("fps", 5), macro_block_size=1)
                else:
                    output_path = str(out_dir / f"{stem}.jpg")
                    result.video[0].save(output_path, format="JPEG", quality=85)
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                error = repr(exc)
            finally:
                del result
                cleanup()
            seconds = round(time.perf_counter() - started, 3)
            after = mem_gib()
            row = {
                "case": case_name,
                "run": run_idx,
                "suite": case["suite"],
                "kind": case["kind"],
                "media": case["media"],
                "dtype": case["dtype"],
                "height": case["height"],
                "width": case["width"],
                "num_frames": case["num_frames"],
                "steps": case["steps"],
                "guidance": case["guidance"],
                "status": status,
                "load_seconds_once": load_seconds,
                "seconds": seconds,
                "error": error,
                "output": output_path,
                "before_mem": before,
                "after_mem": after,
            }
            rows.append(row)
            write_jsonl(jsonl_path, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if status != "passed":
                break
    finally:
        del pipe
        cleanup()
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)

    for case_name, case_rows in sorted(by_case.items()):
        passed = [row["seconds"] for row in case_rows if row["status"] == "passed"]
        failed = [row for row in case_rows if row["status"] != "passed"]
        summary = {
            "case": case_name,
            "runs": len(case_rows),
            "passed": len(passed),
            "failed": len(failed),
            "mean_seconds": round(statistics.mean(passed), 3) if passed else math.nan,
            "min_seconds": round(min(passed), 3) if passed else math.nan,
            "max_seconds": round(max(passed), 3) if passed else math.nan,
            "stdev_seconds": round(statistics.stdev(passed), 3) if len(passed) > 1 else 0.0,
            "cv_percent": round((statistics.stdev(passed) / statistics.mean(passed)) * 100, 2)
            if len(passed) > 1 and statistics.mean(passed) > 0
            else 0.0,
        }
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos3-Nano ROCm benchmark runner")
    parser.add_argument("--suite", choices=["core", "extended", "all"], default="core")
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="Run specific case. Repeatable.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", default="/workspace/result/benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    csv_path = out_dir / "runs.csv"
    summary_json_path = out_dir / "summary.json"
    summary_csv_path = out_dir / "summary.csv"

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "device": torch.cuda.get_device_name(0),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "mem": mem_gib(),
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    selected = args.case or [
        name
        for name, case in CASES.items()
        if args.suite == "all" or case["suite"] == args.suite or (args.suite == "extended" and case["suite"] == "core")
    ]

    all_rows = []
    for case_name in selected:
        case = CASES[case_name]
        print(f"running {case_name}: {case['description']}", flush=True)
        if case["kind"] == "load":
            rows = run_load_case(case_name, case, args.repeats, out_dir, jsonl_path)
        else:
            rows = run_generate_case(case_name, case, args.repeats, out_dir, jsonl_path)
        all_rows.extend(rows)

    summaries = summarize(all_rows)
    write_csv(csv_path, all_rows)
    summary_json_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(summary_csv_path, summaries)
    print(json.dumps({"summary": summaries}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
