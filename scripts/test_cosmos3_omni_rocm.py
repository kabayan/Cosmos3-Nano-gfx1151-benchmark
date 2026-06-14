import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video
from PIL import Image
from scipy.io import wavfile


MODEL_ID = "nvidia/Cosmos3-Nano"
ASSET_DIR = Path("Cosmos3-Nano-assets/assets")


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


def load_json_prompt(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=False)


def load_pipeline(dtype: torch.dtype) -> Cosmos3OmniPipeline:
    started = time.perf_counter()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="cuda",
        enable_safety_checker=False,
    )
    torch.cuda.synchronize()
    print("load_seconds", round(time.perf_counter() - started, 3), flush=True)
    mem("after_load")
    return pipe


def save_sound(path: Path, sound: torch.Tensor | None) -> None:
    if sound is None:
        print("sound_none", flush=True)
        return
    audio = sound.detach().float().cpu().numpy()
    if audio.ndim == 2:
        audio = audio.T
    peak = np.max(np.abs(audio)) if audio.size else 0
    if peak > 0:
        audio = audio / max(peak, 1.0)
    wavfile.write(path, 48000, audio.astype(np.float32))
    print("saved_sound", str(path), "shape", list(sound.shape), flush=True)


def run_i2v(pipe: Cosmos3OmniPipeline, out_dir: Path) -> None:
    prompt = load_json_prompt(ASSET_DIR / "example_i2v_prompt.json")
    negative = load_json_prompt(ASSET_DIR / "negative_prompt.json")
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
        generator=torch.Generator(device="cuda").manual_seed(11),
        enable_safety_check=False,
    )
    torch.cuda.synchronize()
    seconds = round(time.perf_counter() - started, 3)
    output = out_dir / "omni_i2v_256_5f.mp4"
    export_to_video(result.video, str(output), fps=5, macro_block_size=1)
    print(json.dumps({"case": "i2v_256_5f", "seconds": seconds, "output": str(output)}), flush=True)
    mem("after_i2v")


def run_t2vs(pipe: Cosmos3OmniPipeline, out_dir: Path) -> None:
    prompt = load_json_prompt(ASSET_DIR / "example_t2vs_prompt.json")
    negative = load_json_prompt(ASSET_DIR / "negative_prompt.json")
    started = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_frames=5,
        height=256,
        width=448,
        fps=5.0,
        num_inference_steps=2,
        guidance_scale=1.0,
        enable_sound=True,
        generator=torch.Generator(device="cuda").manual_seed(12),
        enable_safety_check=False,
    )
    torch.cuda.synchronize()
    seconds = round(time.perf_counter() - started, 3)
    video_path = out_dir / "omni_t2vs_256_5f.mp4"
    wav_path = out_dir / "omni_t2vs_256_5f.wav"
    export_to_video(result.video, str(video_path), fps=5, macro_block_size=1)
    save_sound(wav_path, result.sound)
    print(
        json.dumps(
            {
                "case": "t2v_sound_256_5f",
                "seconds": seconds,
                "video": str(video_path),
                "wav": str(wav_path),
                "sound_present": result.sound is not None,
            }
        ),
        flush=True,
    )
    mem("after_t2vs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["i2v", "t2vs", "all"], default="all")
    parser.add_argument("--out-dir", default="/workspace/result/omni")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, flush=True)
    print("hip", torch.version.hip, flush=True)
    print("device", torch.cuda.get_device_name(0), flush=True)
    mem("before_load")
    pipe = load_pipeline(torch.float16)
    if args.case in {"i2v", "all"}:
        run_i2v(pipe, out_dir)
    if args.case in {"t2vs", "all"}:
        run_t2vs(pipe, out_dir)


if __name__ == "__main__":
    main()
