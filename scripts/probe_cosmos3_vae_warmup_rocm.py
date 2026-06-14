import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from diffusers import Cosmos3OmniPipeline


MODEL_ID = "nvidia/Cosmos3-Nano"

PROMPT = json.dumps(
    {
        "temporal_caption": "A robotic gripper descends toward a red cube and lifts it.",
        "subjects": [
            {
                "description": "A two-finger robotic gripper and a red cube on a lab bench.",
                "action": "The gripper grasps and lifts the cube.",
                "state_changes": "The cube is stationary, grasped, then lifted.",
            }
        ],
        "background_setting": "A clean robotics laboratory workbench.",
        "duration": "1s",
        "fps": 8,
    },
    ensure_ascii=False,
)


class StageProfiler:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, float | int]] = {}
        self._patched: list[tuple[Any, str, Any]] = []

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _add(self, name: str, seconds: float) -> None:
        record = self.records.setdefault(name, {"seconds": 0.0, "calls": 0})
        record["seconds"] = float(record["seconds"]) + seconds
        record["calls"] = int(record["calls"]) + 1

    def timed(self, name: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        self._sync()
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self._sync()
            self._add(name, time.perf_counter() - started)

    def patch_method(self, obj: Any, method_name: str, label: str) -> None:
        if obj is None or not hasattr(obj, method_name):
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


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    sync()
    started = time.perf_counter()
    value = fn()
    sync()
    return value, round(time.perf_counter() - started, 3)


def parse_shape(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def apply_vae_mode(vae: Any, mode: str) -> list[str]:
    applied = []
    if mode in {"tiling", "tiling_slicing"}:
        vae.enable_tiling()
        applied.append("enable_tiling")
    if mode in {"slicing", "tiling_slicing"}:
        vae.enable_slicing()
        applied.append("enable_slicing")
    return applied


def run_t2v(pipe: Cosmos3OmniPipeline, args: argparse.Namespace, profiler: StageProfiler) -> dict[str, Any]:
    profiler.reset()
    _, seconds = timed(
        lambda: pipe(
            prompt=PROMPT,
            negative_prompt="blurry, distorted, low quality",
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            fps=float(args.fps),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            enable_safety_check=False,
        )
    )
    stages = profiler.snapshot()
    stage_sum = round(sum(float(value["seconds"]) for value in stages.values()), 3)
    return {
        "seconds": seconds,
        "stage_profile": {
            "records": stages,
            "timed_stage_sum_seconds": stage_sum,
            "unattributed_pipe_seconds": round(seconds - stage_sum, 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/workspace/result/rocm_speed_matrix/vae_warmup_probe")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=204)
    parser.add_argument("--latent-shape", default="1,48,2,16,28")
    parser.add_argument("--vae-mode", choices=["default", "tiling", "slicing", "tiling_slicing"], default="default")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--t2v-runs", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("torch", torch.__version__, flush=True)
    print("hip", torch.version.hip, flush=True)
    print("device", torch.cuda.get_device_name(0), flush=True)

    pipe, load_seconds = timed(
        lambda: Cosmos3OmniPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
            enable_safety_checker=False,
        )
    )

    vae = pipe.vae
    applied_methods = apply_vae_mode(vae, args.vae_mode)
    profiler = StageProfiler()
    profiler.install(pipe)

    warmup = None
    latent_shape = parse_shape(args.latent_shape)
    if not args.skip_warmup:
        latent = torch.randn(latent_shape, device="cuda", dtype=torch.float16)
        profiler.reset()
        with torch.inference_mode():
            output, warmup_seconds = timed(lambda: vae.decode(latent))
        del output
        del latent
        warmup = {
            "seconds": warmup_seconds,
            "stage_profile": profiler.snapshot(),
            "latent_shape": list(latent_shape),
            "latent_dtype": "torch.float16",
        }

    t2v_runs = []
    with torch.inference_mode():
        for index in range(args.t2v_runs):
            result = run_t2v(pipe, args, profiler)
            result["run"] = index + 1
            t2v_runs.append(result)

    report = {
        "model": MODEL_ID,
        "load_seconds": load_seconds,
        "settings": {
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "fps": args.fps,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "latent_shape": list(latent_shape),
            "vae_mode": args.vae_mode,
            "skip_warmup": args.skip_warmup,
            "t2v_runs": args.t2v_runs,
        },
        "vae": {
            "class": type(vae).__name__,
            "dtype": str(next(vae.parameters()).dtype),
            "applied_methods": applied_methods,
        },
        "warmup_decode": warmup,
        "t2v_runs": t2v_runs,
    }
    (out_dir / "vae_warmup_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    profiler.restore()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
