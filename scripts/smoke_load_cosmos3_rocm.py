import gc
import json
import time

import torch
from diffusers import Cosmos3OmniPipeline


def mem(label: str) -> None:
    if torch.cuda.is_available():
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


print("torch", torch.__version__, flush=True)
print("hip", torch.version.hip, flush=True)
print("device", torch.cuda.get_device_name(0), flush=True)
print("bf16", torch.cuda.is_bf16_supported(), flush=True)
mem("before_load")

started = time.time()
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.float16,
    device_map="cuda",
    enable_safety_checker=False,
)
print("loaded_seconds", round(time.time() - started, 1), flush=True)
mem("after_load")

del pipe
gc.collect()
torch.cuda.empty_cache()
mem("after_cleanup")
