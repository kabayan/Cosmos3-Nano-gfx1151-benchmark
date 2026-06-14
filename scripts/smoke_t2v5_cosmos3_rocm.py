import time

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video


pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.float16,
    device_map="cuda",
    enable_safety_checker=False,
)

started = time.time()
result = pipe(
    prompt="A mobile robot slowly moves through a clean warehouse aisle.",
    negative_prompt="blurry, distorted, low quality",
    num_frames=5,
    height=480,
    width=832,
    fps=5.0,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(1),
)
print("generate_seconds", round(time.time() - started, 1), flush=True)
export_to_video(
    result.video,
    "/workspace/cosmos3_rocm_t2v5_smoke.mp4",
    fps=5,
    macro_block_size=1,
)
print("saved /workspace/cosmos3_rocm_t2v5_smoke.mp4", flush=True)
