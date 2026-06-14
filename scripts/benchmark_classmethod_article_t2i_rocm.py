import argparse
import ctypes
import ctypes.util
import gc
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from diffusers import Cosmos3OmniPipeline


MODEL_ID = "nvidia/Cosmos3-Nano"


class StageProfiler:
    def __init__(self, enabled: bool, rocprof_transformer_only: bool = False):
        self.enabled = enabled
        self.rocprof_transformer_only = rocprof_transformer_only
        self.rocprof_active = False
        self._roctx = None
        self.records: dict[str, dict[str, float | int]] = {}
        self._patched: list[tuple[Any, str, Any]] = []
        if rocprof_transformer_only:
            self._roctx = self._load_roctx()

    def _load_roctx(self) -> Any:
        lib_name = ctypes.util.find_library("rocprofiler-sdk-roctx") or ctypes.util.find_library("roctx64")
        if not lib_name:
            raise RuntimeError("ROCTx library not found for selected-region profiling")
        lib = ctypes.CDLL(lib_name)
        lib.roctxProfilerResume.argtypes = [ctypes.c_int]
        lib.roctxProfilerResume.restype = None
        lib.roctxProfilerPause.argtypes = [ctypes.c_int]
        lib.roctxProfilerPause.restype = None
        lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        lib.roctxRangePushA.restype = ctypes.c_int
        lib.roctxRangePop.argtypes = []
        lib.roctxRangePop.restype = ctypes.c_int
        return lib

    def set_rocprof_active(self, active: bool) -> None:
        self.rocprof_active = active

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _add(self, name: str, seconds: float) -> None:
        record = self.records.setdefault(name, {"seconds": 0.0, "calls": 0})
        record["seconds"] = float(record["seconds"]) + seconds
        record["calls"] = int(record["calls"]) + 1

    def timed(self, name: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if not self.enabled:
            return fn(*args, **kwargs)
        self._sync()
        use_roctx = bool(self._roctx and self.rocprof_active and name == "transformer_forward")
        if use_roctx:
            self._roctx.roctxProfilerResume(0)
            self._roctx.roctxRangePushA(name.encode("utf-8"))
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self._sync()
            if use_roctx:
                self._roctx.roctxRangePop()
                self._roctx.roctxProfilerPause(0)
            self._add(name, time.perf_counter() - started)

    def patch_method(self, obj: Any, method_name: str, label: str) -> None:
        if not self.enabled or obj is None or not hasattr(obj, method_name):
            return
        original = getattr(obj, method_name)
        if not callable(original):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self.timed(label, original, *args, **kwargs)

        setattr(obj, method_name, wrapped)
        self._patched.append((obj, method_name, original))

    def install(self, pipe: Cosmos3OmniPipeline) -> None:
        self.patch_method(getattr(pipe, "transformer", None), "forward", "transformer_forward")
        self.patch_method(getattr(pipe, "vae", None), "decode", "vae_decode")
        self.patch_method(getattr(pipe, "image_processor", None), "postprocess", "image_postprocess")
        self.patch_method(getattr(pipe, "video_processor", None), "postprocess_video", "video_postprocess")

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            key: {"seconds": round(float(value["seconds"]), 3), "calls": int(value["calls"])}
            for key, value in sorted(self.records.items())
        }

    def reset(self) -> None:
        self.records.clear()

    def restore(self) -> None:
        for obj, method_name, original in reversed(self._patched):
            setattr(obj, method_name, original)
        self._patched.clear()


class LinearProfiler:
    def __init__(self, enabled: bool, top: int = 40):
        self.enabled = enabled
        self.top = top
        self.records: dict[str, dict[str, Any]] = {}
        self._patched: list[tuple[Any, str, Any]] = []

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def install(self, pipe: Cosmos3OmniPipeline) -> None:
        if not self.enabled:
            return
        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            return
        for name, module in transformer.named_modules():
            if isinstance(module, nn.Linear):
                self._patch_linear(name, module)

    def _patch_linear(self, name: str, module: nn.Linear) -> None:
        original = module.forward

        def wrapped(input: torch.Tensor) -> torch.Tensor:
            self._sync()
            started = time.perf_counter()
            try:
                return original(input)
            finally:
                self._sync()
                elapsed = time.perf_counter() - started
                record = self.records.setdefault(
                    name,
                    {
                        "seconds": 0.0,
                        "calls": 0,
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "bias": module.bias is not None,
                        "input_shapes": {},
                    },
                )
                record["seconds"] += elapsed
                record["calls"] += 1
                shape = tuple(int(dim) for dim in input.shape)
                record["input_shapes"][str(shape)] = record["input_shapes"].get(str(shape), 0) + 1

        module.forward = wrapped
        self._patched.append((module, "forward", original))

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        rows = []
        for name, record in self.records.items():
            seconds = float(record["seconds"])
            calls = int(record["calls"])
            rows.append(
                {
                    "name": name,
                    "seconds": round(seconds, 6),
                    "calls": calls,
                    "average_ms": round(seconds * 1000 / calls, 6) if calls else 0.0,
                    "in_features": int(record["in_features"]),
                    "out_features": int(record["out_features"]),
                    "bias": bool(record["bias"]),
                    "input_shapes": dict(record["input_shapes"]),
                }
            )
        rows.sort(key=lambda item: item["seconds"], reverse=True)
        return {
            "enabled": True,
            "top": self.top,
            "linear_count": len(self.records),
            "total_seconds": round(sum(float(row["seconds"]) for row in rows), 6),
            "records": rows[: self.top],
        }

    def restore(self) -> None:
        for obj, method_name, original in reversed(self._patched):
            setattr(obj, method_name, original)
        self._patched.clear()


def mem(label: str) -> dict:
    free, total = torch.cuda.mem_get_info()
    data = {
        "label": label,
        "free_gib": round(free / 1024**3, 3),
        "total_gib": round(total / 1024**3, 3),
    }
    print(json.dumps(data), flush=True)
    return data


def install_vae_decode_abort(pipe: Cosmos3OmniPipeline) -> Callable[[], None]:
    original = pipe.vae.decode

    def aborting_decode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Intentional abort before VAE decode for transformer-only benchmark")

    pipe.vae.decode = aborting_decode

    def restore() -> None:
        pipe.vae.decode = original

    return restore


def load_prompt() -> tuple[str, str]:
    source = Path("/workspace/tmp/cosmos-framework/inputs/omni/t2i.json")
    if source.exists():
        return json.loads(source.read_text())["prompt"], str(source)
    fallback = Path("/tmp/cosmos-framework/inputs/omni/t2i.json")
    return json.loads(fallback.read_text())["prompt"], str(fallback)


def synthetic_vae_warmup(pipe: Cosmos3OmniPipeline, shape_text: str, profiler: StageProfiler) -> dict:
    shape = tuple(int(item) for item in shape_text.split(","))
    latent = torch.randn(shape, device="cuda", dtype=torch.float16)
    profiler.reset()
    started = time.perf_counter()
    with torch.inference_mode():
        output = pipe.vae.decode(latent)
    torch.cuda.synchronize()
    seconds = round(time.perf_counter() - started, 3)
    del output
    del latent
    data = {
        "enabled": True,
        "seconds": seconds,
        "latent_shape": list(shape),
        "latent_dtype": "torch.float16",
        "stage_profile": profiler.snapshot(),
    }
    print(json.dumps({"vae_warmup": data}, ensure_ascii=False), flush=True)
    return data


def run_t2i(
    pipe: Cosmos3OmniPipeline,
    prompt: str,
    prompt_source: str,
    args: argparse.Namespace,
    out_dir: Path,
    profiler: StageProfiler,
    run_index: int,
    role: str,
) -> dict:
    profiler.set_rocprof_active(args.rocprof_transformer_only and role == "measured")
    profiler.reset()
    try:
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt="blurry, distorted, low quality",
                    num_frames=1,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=torch.Generator(device="cuda").manual_seed(args.seed),
                    enable_safety_check=False,
                )
        except Exception as exc:
            if not args.allow_pipeline_error:
                raise
            torch.cuda.synchronize()
            seconds = round(time.perf_counter() - started, 3)
            stages = profiler.snapshot()
            stage_sum = round(sum(float(value["seconds"]) for value in stages.values()), 3)
            data = {
                "case": "article_t2i_robotics_lab",
                "run": run_index,
                "measurement_role": role,
                "seconds": seconds,
                "output": None,
                "prompt_source": prompt_source,
                "pipeline_error": {
                    "allowed": True,
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "stage_profile": {
                    "enabled": args.stage_profile,
                    "records": stages,
                    "timed_stage_sum_seconds": stage_sum,
                    "unattributed_pipe_seconds": round(seconds - stage_sum, 3),
                },
                "settings": {
                    "height": args.height,
                    "width": args.width,
                    "frames": 1,
                    "steps": args.steps,
                    "guidance": args.guidance,
                    "seed": args.seed,
                },
            }
            print(json.dumps(data, ensure_ascii=False), flush=True)
            return data
        torch.cuda.synchronize()
        seconds = round(time.perf_counter() - started, 3)
    finally:
        profiler.set_rocprof_active(False)
    stages = profiler.snapshot()
    stage_sum = round(sum(float(value["seconds"]) for value in stages.values()), 3)
    output = out_dir / f"article_t2i_robotics_lab_{args.width}x{args.height}_s{args.steps}_{role}_r{run_index}.jpg"
    result.video[0].save(output, format="JPEG", quality=90)
    data = {
        "case": "article_t2i_robotics_lab",
        "run": run_index,
        "measurement_role": role,
        "seconds": seconds,
        "output": str(output),
        "prompt_source": prompt_source,
        "stage_profile": {
            "enabled": args.stage_profile,
            "records": stages,
            "timed_stage_sum_seconds": stage_sum,
            "unattributed_pipe_seconds": round(seconds - stage_sum, 3),
        },
        "settings": {
            "height": args.height,
            "width": args.width,
            "frames": 1,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
        },
    }
    print(json.dumps(data, ensure_ascii=False), flush=True)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/workspace/result/classmethod_article_benchmark")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--stage-profile", action="store_true")
    parser.add_argument("--vae-warmup", action="store_true")
    parser.add_argument("--vae-warmup-shape", default="1,48,1,60,60")
    parser.add_argument("--mode-warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--rocprof-transformer-only", action="store_true")
    parser.add_argument("--allow-pipeline-error", action="store_true")
    parser.add_argument("--abort-before-vae-decode", action="store_true")
    parser.add_argument("--linear-profile", action="store_true")
    parser.add_argument("--linear-profile-top", type=int, default=40)
    parser.add_argument("--und-branch-cache", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, flush=True)
    print("hip", torch.version.hip, flush=True)
    print("device", torch.cuda.get_device_name(0), flush=True)
    mem("before_load")

    started = time.perf_counter()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        enable_safety_checker=False,
    )
    torch.cuda.synchronize()
    load_seconds = round(time.perf_counter() - started, 3)
    print("load_seconds", load_seconds, flush=True)
    mem("after_load")

    profiler = StageProfiler(enabled=args.stage_profile, rocprof_transformer_only=args.rocprof_transformer_only)
    profiler.install(pipe)
    linear_profiler = LinearProfiler(enabled=args.linear_profile, top=args.linear_profile_top)
    linear_profiler.install(pipe)
    native_und_cache = False
    if args.und_branch_cache:
        if not hasattr(pipe.transformer, "enable_und_branch_cache"):
            raise RuntimeError("Transformer does not expose enable_und_branch_cache")
        pipe.transformer.enable_und_branch_cache(True, reset=True)
        native_und_cache = True
    restore_vae_decode_abort = install_vae_decode_abort(pipe) if args.abort_before_vae_decode else None
    warmup = synthetic_vae_warmup(pipe, args.vae_warmup_shape, profiler) if args.vae_warmup else {"enabled": False}
    prompt, prompt_source = load_prompt()
    runs = []
    for index in range(args.mode_warmup_runs + args.measured_runs):
        role = "warmup" if index < args.mode_warmup_runs else "measured"
        runs.append(run_t2i(pipe, prompt, prompt_source, args, out_dir, profiler, index + 1, role))
    measured = [item for item in runs if item["measurement_role"] == "measured"]
    summary = {
        "model": MODEL_ID,
        "dtype": "float16",
        "load_seconds": load_seconds,
        "case": "article_t2i_robotics_lab",
        "seconds": measured[-1]["seconds"] if measured else None,
        "output": measured[-1]["output"] if measured else None,
        "prompt_source": prompt_source,
        "warmup": warmup,
        "mode_warmup_runs": args.mode_warmup_runs,
        "measured_runs": args.measured_runs,
        "runs": runs,
        "comparison_target": {
            "source": "Classmethod DGX Spark article",
            "t2i": "960x960, 35 steps, model resident after about 22 sec",
        },
        "settings": {
            "height": args.height,
            "width": args.width,
            "frames": 1,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
        },
        "runtime_options": {
            "stage_profile": args.stage_profile,
            "rocprof_transformer_only": args.rocprof_transformer_only,
            "allow_pipeline_error": args.allow_pipeline_error,
            "abort_before_vae_decode": args.abort_before_vae_decode,
            "linear_profile": args.linear_profile,
            "und_branch_cache": args.und_branch_cache,
        },
        "linear_profile": linear_profiler.snapshot(),
        "und_branch_cache": (
            pipe.transformer.get_und_branch_cache_stats() if native_und_cache else {"enabled": False}
        ),
    }
    (out_dir / "article_t2i_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)

    if restore_vae_decode_abort is not None:
        restore_vae_decode_abort()
    if native_und_cache:
        pipe.transformer.disable_und_branch_cache()
    linear_profiler.restore()
    profiler.restore()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    mem("after_cleanup")


if __name__ == "__main__":
    main()
