import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

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


class CaptureComplete(RuntimeError):
    pass


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    started = time.perf_counter()
    value = fn()
    sync()
    return value, round(time.perf_counter() - started, 3)


def tensor_info(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": value.numel(),
        }
    if isinstance(value, (list, tuple)):
        return [tensor_info(item) for item in value]
    if isinstance(value, dict):
        return {key: tensor_info(item) for key, item in value.items()}
    return repr(value)


def clone_tensor_args(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_tensor_args(item) for item in value)
    if isinstance(value, list):
        return [clone_tensor_args(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tensor_args(item) for key, item in value.items()}
    return value


def selected_config(obj: Any) -> dict[str, Any]:
    config = getattr(obj, "config", None)
    if config is None:
        return {}
    data = dict(config) if isinstance(config, dict) else dict(getattr(config, "__dict__", {}))
    keys = [
        key
        for key in sorted(data)
        if any(token in key.lower() for token in ["tile", "chunk", "slice", "block", "attention", "dtype"])
    ]
    return {key: data[key] for key in keys}


def available_methods(obj: Any) -> list[str]:
    candidates = [
        "enable_tiling",
        "disable_tiling",
        "enable_slicing",
        "disable_slicing",
        "enable_gradient_checkpointing",
        "set_attn_processor",
    ]
    return [name for name in candidates if hasattr(obj, name)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/workspace/result/rocm_speed_matrix/vae_decode_probe")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=204)
    parser.add_argument("--standalone-runs", type=int, default=1)
    parser.add_argument("--abort-after-capture", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("torch", torch.__version__, flush=True)
    print("hip", torch.version.hip, flush=True)
    print("device", torch.cuda.get_device_name(0), flush=True)
    print(
        json.dumps(
            {
                "env": {
                    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": __import__("os").environ.get(
                        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
                    ),
                    "PYTORCH_TUNABLEOP_ENABLED": __import__("os").environ.get("PYTORCH_TUNABLEOP_ENABLED"),
                }
            }
        ),
        flush=True,
    )

    pipe, load_seconds = timed(
        lambda: Cosmos3OmniPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
            enable_safety_checker=False,
        )
    )

    vae = pipe.vae
    original_decode = vae.decode
    captured: dict[str, Any] = {}

    def capture_decode(*decode_args: Any, **decode_kwargs: Any) -> Any:
        if not captured:
            sync()
            captured["args"] = clone_tensor_args(decode_args)
            captured["kwargs"] = clone_tensor_args(decode_kwargs)
            captured["args_info"] = tensor_info(decode_args)
            captured["kwargs_info"] = tensor_info(decode_kwargs)
            if args.abort_after_capture:
                raise CaptureComplete("captured vae.decode inputs")
        return original_decode(*decode_args, **decode_kwargs)

    vae.decode = capture_decode
    pipe_status = "success"
    try:
        _, pipe_seconds = timed(
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
    except CaptureComplete:
        sync()
        pipe_status = "aborted_after_capture"
        pipe_seconds = None
    finally:
        vae.decode = original_decode

    standalone = []
    for index in range(args.standalone_runs):
        _, seconds = timed(lambda: original_decode(*captured["args"], **captured["kwargs"]))
        standalone.append({"run": index + 1, "seconds": seconds})

    report = {
        "model": MODEL_ID,
        "load_seconds": load_seconds,
        "pipe_seconds": pipe_seconds,
        "pipe_status": pipe_status,
        "settings": {
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "fps": args.fps,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
        },
        "vae": {
            "class": type(vae).__name__,
            "module": type(vae).__module__,
            "dtype": str(next(vae.parameters()).dtype),
            "config_subset": selected_config(vae),
            "available_methods": available_methods(vae),
        },
        "decode_input": {
            "args": captured.get("args_info"),
            "kwargs": captured.get("kwargs_info"),
        },
        "standalone_decode": standalone,
    }
    (out_dir / "vae_decode_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
