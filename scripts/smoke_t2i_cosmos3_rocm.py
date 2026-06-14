import gc
import json
import time

import torch
from diffusers import Cosmos3OmniPipeline


def mem(label: str) -> None:
    free, total = torch.cuda.mem_get_info()
    print(
        json.dumps(
            {
                "label": label,
                "free_gib": round(free / 1024**3, 2),
                "total_gib": round(total / 1024**3, 2),
            }
        ),
        flush=True,
    )


torch.manual_seed(0)
print("torch", torch.__version__, flush=True)
print("hip", torch.version.hip, flush=True)
print("device", torch.cuda.get_device_name(0), flush=True)
mem("before_load")

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.float16,
    device_map="cuda",
    enable_safety_checker=False,
)
mem("after_load")

started = time.time()
result = pipe(
    prompt="A mobile robot in a clean warehouse aisle.",
    negative_prompt="blurry, distorted, low quality",
    num_frames=1,
    height=480,
    width=832,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(0),
)
print("generate_seconds", round(time.time() - started, 1), flush=True)
mem("after_generate")

out = "/workspace/cosmos3_rocm_t2i_smoke.jpg"
result.video[0].save(out, format="JPEG", quality=85)
print("saved", out, flush=True)

del pipe, result
gc.collect()
torch.cuda.empty_cache()
mem("after_cleanup")
