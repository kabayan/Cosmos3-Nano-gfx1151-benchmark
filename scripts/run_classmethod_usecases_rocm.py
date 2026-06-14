import argparse
import gc
import json
import time
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video
from PIL import Image


MODEL_ID = "nvidia/Cosmos3-Nano"
ASSET_DIR = Path("Cosmos3-Nano-assets/assets")
OUT_DIR = Path("/workspace/result/classmethod")


ARTICLE_T2V_PROMPT = json.dumps(
    {
        "temporal_caption": (
            "A robotic gripper descends toward a red cube, makes contact, grasps it, "
            "and slowly lifts it upward in a physically plausible sequence."
        ),
        "subjects": [
            {
                "description": "An industrial robotic arm with a two-finger gripper.",
                "action": "The gripper lowers, grasps a red cube, and slowly lifts it.",
                "state_changes": "Open gripper, contact with cube, closed grip, cube lifted.",
            },
            {
                "description": "A small red cube on a clean work surface.",
                "action": "The cube is picked up by the gripper and lifted.",
                "state_changes": "Stationary on table, grasped, then raised above the table.",
            },
        ],
        "background_setting": "A clean robotics lab workbench with neutral lighting.",
        "cinematography": {
            "camera_motion": "Static",
            "framing": "Medium shot",
            "camera_angle": "Eye-level",
        },
        "duration": "1s",
        "fps": 5,
    },
    ensure_ascii=False,
)


def mem(label: str) -> None:
    free, total = torch.cuda.mem_get_info()
    print(
        json.dumps(
            {
                "label": label,
                "free_gib": round(free / 1024**3, 3),
                "total_gib": round(total / 1024**3, 3),
            }
        ),
        flush=True,
    )


def load_json(path: Path):
    return json.loads(path.read_text())


def load_pipeline() -> Cosmos3OmniPipeline:
    started = time.perf_counter()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        enable_safety_checker=False,
    )
    torch.cuda.synchronize()
    print("load_seconds", round(time.perf_counter() - started, 3), flush=True)
    mem("after_load")
    return pipe


def run_t2i(pipe: Cosmos3OmniPipeline, out_dir: Path) -> dict:
    source = Path("/workspace/tmp/cosmos-framework/inputs/omni/t2i.json")
    if source.exists():
        prompt = load_json(source)["prompt"]
        data_source = str(source)
    else:
        prompt = load_json(Path("/tmp/cosmos-framework/inputs/omni/t2i.json"))["prompt"]
        data_source = "/tmp/cosmos-framework/inputs/omni/t2i.json"

    started = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality",
        num_frames=1,
        height=480,
        width=480,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(101),
        enable_safety_check=False,
    )
    torch.cuda.synchronize()
    seconds = round(time.perf_counter() - started, 3)
    output = out_dir / "classmethod_t2i_robotics_lab_480.jpg"
    result.video[0].save(output, format="JPEG", quality=85)
    print(json.dumps({"case": "t2i_article_robotics_lab", "seconds": seconds, "output": str(output)}), flush=True)
    return {
        "case": "t2i_article_robotics_lab",
        "source": data_source,
        "seconds": seconds,
        "output": str(output),
        "settings": {"height": 480, "width": 480, "steps": 4, "seed": 101},
    }


def run_t2v(pipe: Cosmos3OmniPipeline, out_dir: Path) -> dict:
    started = time.perf_counter()
    result = pipe(
        prompt=ARTICLE_T2V_PROMPT,
        negative_prompt="blurry, distorted, low quality",
        num_frames=5,
        height=256,
        width=448,
        fps=5.0,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(102),
        enable_safety_check=False,
    )
    torch.cuda.synchronize()
    seconds = round(time.perf_counter() - started, 3)
    output = out_dir / "classmethod_t2v_red_cube_grasp_256_5f.mp4"
    export_to_video(result.video, str(output), fps=5, macro_block_size=1)
    print(json.dumps({"case": "t2v_article_red_cube_grasp", "seconds": seconds, "output": str(output)}), flush=True)
    return {
        "case": "t2v_article_red_cube_grasp",
        "source": "article prompt text",
        "seconds": seconds,
        "output": str(output),
        "settings": {"height": 256, "width": 448, "frames": 5, "fps": 5, "steps": 4, "seed": 102},
    }


def run_i2v(pipe: Cosmos3OmniPipeline, out_dir: Path) -> dict:
    prompt = json.dumps(load_json(ASSET_DIR / "example_i2v_prompt.json"), ensure_ascii=False)
    negative = json.dumps(load_json(ASSET_DIR / "negative_prompt.json"), ensure_ascii=False)
    image = Image.open(ASSET_DIR / "example_i2v_input.jpg").convert("RGB")
    started = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=image,
        num_frames=5,
        height=256,
        width=448,
        fps=5.0,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(103),
        enable_safety_check=False,
    )
    torch.cuda.synchronize()
    seconds = round(time.perf_counter() - started, 3)
    output = out_dir / "classmethod_i2v_robot_arms_256_5f.mp4"
    export_to_video(result.video, str(output), fps=5, macro_block_size=1)
    print(json.dumps({"case": "i2v_article_robot_arms", "seconds": seconds, "output": str(output)}), flush=True)
    return {
        "case": "i2v_article_robot_arms",
        "source": "Cosmos3-Nano-assets/assets/example_i2v_input.jpg + example_i2v_prompt.json",
        "seconds": seconds,
        "output": str(output),
        "settings": {"height": 256, "width": 448, "frames": 5, "fps": 5, "steps": 4, "seed": 103},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["t2i", "t2v", "i2v", "all"], default="all")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, flush=True)
    print("hip", torch.version.hip, flush=True)
    print("device", torch.cuda.get_device_name(0), flush=True)
    mem("before_load")

    pipe = load_pipeline()
    results = []
    if args.case in {"t2i", "all"}:
        results.append(run_t2i(pipe, out_dir))
    if args.case in {"t2v", "all"}:
        results.append(run_t2v(pipe, out_dir))
    if args.case in {"i2v", "all"}:
        results.append(run_i2v(pipe, out_dir))

    summary = {
        "model": MODEL_ID,
        "dtype": "float16",
        "results": results,
    }
    (out_dir / "classmethod_usecases_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    mem("after_cleanup")


if __name__ == "__main__":
    main()
