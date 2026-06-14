import argparse
import ctypes
import ctypes.util
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video
from PIL import Image


MODEL_ID = "nvidia/Cosmos3-Nano"
ASSET_DIR = Path("Cosmos3-Nano-assets/assets")


ARTICLE_T2V_PROMPT = json.dumps(
    {
        "temporal_caption": (
            "A robotic gripper descends toward a red cube, makes contact, grasps it, "
            "and slowly lifts it upward in a physically plausible sequence."
        ),
        "subjects": [
            {
                "description": "A robotic arm with a two-finger gripper in a clean robotics lab.",
                "action": "The gripper descends, closes around a red cube, and slowly lifts it.",
                "state_changes": "The red cube starts on the table, is grasped, and rises above the table.",
            },
            {
                "description": "A small red cube on a workbench.",
                "action": "The cube is picked up by the gripper.",
                "state_changes": "Stationary, grasped, lifted.",
            },
        ],
        "background_setting": "A clean robotics laboratory workbench with neutral lighting.",
        "cinematography": {
            "camera_motion": "Static",
            "framing": "Medium shot",
            "camera_angle": "Eye-level",
        },
        "duration": "2s",
        "fps": 12,
    },
    ensure_ascii=False,
)


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
        if not self.enabled:
            return
        self.patch_method(getattr(pipe, "transformer", None), "forward", "transformer_forward")
        self.patch_method(getattr(pipe, "vae", None), "decode", "vae_decode")
        self.patch_method(getattr(pipe, "video_processor", None), "postprocess_video", "video_postprocess")
        self.patch_method(getattr(pipe, "image_processor", None), "postprocess", "image_postprocess")

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
            if isinstance(module, torch.nn.Linear):
                self._patch_linear(name, module)

    def _patch_linear(self, name: str, module: torch.nn.Linear) -> None:
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


class TransformerInputProfiler:
    def __init__(self, enabled: bool, max_calls: int = 3):
        self.enabled = enabled
        self.max_calls = max_calls
        self.records: list[dict[str, Any]] = []
        self._patched: list[tuple[Any, str, Any]] = []

    def install(self, pipe: Cosmos3OmniPipeline) -> None:
        if not self.enabled or getattr(pipe, "transformer", None) is None:
            return
        transformer = pipe.transformer
        original = transformer.forward

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if len(self.records) < self.max_calls:
                self.records.append(self._summarize_call(args, kwargs))
            return original(*args, **kwargs)

        transformer.forward = wrapped
        self._patched.append((transformer, "forward", original))

    def _tensor_summary(self, tensor: torch.Tensor) -> dict[str, Any]:
        data = {
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        if tensor.numel() == 0:
            data.update({"sum": 0.0, "abs_sum": 0.0, "mean": 0.0})
            return data
        sample = tensor.detach()
        if sample.is_floating_point() or sample.is_complex():
            sample = sample.float()
            data.update(
                {
                    "sum": round(float(sample.sum().item()), 6),
                    "abs_sum": round(float(sample.abs().sum().item()), 6),
                    "mean": round(float(sample.mean().item()), 9),
                }
            )
        else:
            sample64 = sample.to(torch.int64)
            data.update(
                {
                    "sum": int(sample64.sum().item()),
                    "abs_sum": int(sample64.abs().sum().item()),
                    "mean": round(float(sample64.float().mean().item()), 9),
                }
            )
        return data

    def _summarize_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return self._tensor_summary(value)
        if isinstance(value, list):
            return [self._summarize_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._summarize_value(item) for item in value]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return repr(value)

    def _summarize_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "call": len(self.records) + 1,
            "args": [self._summarize_value(arg) for arg in args],
            "kwargs": {key: self._summarize_value(value) for key, value in sorted(kwargs.items())},
        }

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {"enabled": True, "max_calls": self.max_calls, "records": self.records}

    def restore(self) -> None:
        for obj, method_name, original in reversed(self._patched):
            setattr(obj, method_name, original)
        self._patched.clear()


class UndBranchCachePrototype:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._patched: list[tuple[Any, str, Any]] = []
        self.cache: dict[int, dict[str, torch.Tensor]] = {}
        self.signature: tuple[Any, ...] | None = None
        self.mode: str = "disabled"
        self.current_layer = 0
        self.transformer_calls = 0
        self.write_calls = 0
        self.read_calls = 0
        self.invalidations = 0

    def install(self, pipe: Cosmos3OmniPipeline) -> None:
        if not self.enabled or getattr(pipe, "transformer", None) is None:
            return
        transformer = pipe.transformer
        original_forward = transformer.forward

        def wrapped_transformer_forward(*args: Any, **kwargs: Any) -> Any:
            signature = self._signature(args, kwargs)
            if self.signature != signature or len(self.cache) != len(getattr(transformer, "layers", [])):
                self.cache.clear()
                self.signature = signature
                self.mode = "write"
                self.invalidations += 1
            else:
                self.mode = "read"
            self.current_layer = 0
            self.transformer_calls += 1
            if self.mode == "write":
                self.write_calls += 1
            else:
                self.read_calls += 1
            try:
                return original_forward(*args, **kwargs)
            finally:
                self.mode = "disabled"

        transformer.forward = wrapped_transformer_forward
        self._patched.append((transformer, "forward", original_forward))

        for index, layer in enumerate(transformer.layers):
            original_layer_forward = layer.forward

            def wrapped_layer_forward(
                und_seq: torch.Tensor,
                gen_seq: torch.Tensor,
                rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                *,
                _layer=layer,
                _index=index,
                _original=original_layer_forward,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if self.mode == "read" and _index in self.cache:
                    self.current_layer += 1
                    return self._forward_read(_layer, _index, gen_seq, rotary_emb)
                if self.mode == "write":
                    self.current_layer += 1
                    return self._forward_write(_layer, _index, und_seq, gen_seq, rotary_emb)
                self.current_layer += 1
                return _original(und_seq, gen_seq, rotary_emb)

            layer.forward = wrapped_layer_forward
            self._patched.append((layer, "forward", original_layer_forward))

    def _tensor_signature(self, tensor: torch.Tensor) -> tuple[Any, ...]:
        data = tensor.detach()
        if data.numel() == 0:
            return (tuple(data.shape), str(data.dtype), str(data.device), 0)
        if data.is_floating_point():
            sample = data.float()
            return (
                tuple(int(dim) for dim in data.shape),
                str(data.dtype),
                str(data.device),
                round(float(sample.sum().item()), 5),
                round(float(sample.abs().sum().item()), 5),
            )
        sample_i = data.to(torch.int64)
        return (
            tuple(int(dim) for dim in data.shape),
            str(data.dtype),
            str(data.device),
            int(sample_i.sum().item()),
            int(sample_i.abs().sum().item()),
        )

    def _signature(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        stable_keys = ("input_ids", "text_indexes", "position_ids", "und_len", "sequence_length")
        items: list[Any] = []
        for key in stable_keys:
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                items.append((key, self._tensor_signature(value)))
            else:
                items.append((key, value))
        return tuple(items)

    def _attention_backend_args(self, layer: Any) -> dict[str, Any]:
        processor = getattr(layer.self_attn, "processor", None)
        return {
            "backend": getattr(processor, "_attention_backend", None),
            "parallel_config": getattr(processor, "_parallel_config", None),
        }

    def _project_und(
        self,
        layer: Any,
        und_norm: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.models.transformers import transformer_cosmos3

        attn = layer.self_attn
        q_und = attn.to_q(und_norm).view(-1, attn.num_attention_heads, attn.head_dim)
        k_und = attn.to_k(und_norm).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.to_v(und_norm).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        cos_und, sin_und, _, _ = rotary_emb
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + transformer_cosmos3._rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + transformer_cosmos3._rotate_half(k_und) * sin_und
        return q_und, k_und, v_und

    def _project_gen(
        self,
        layer: Any,
        gen_norm: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.models.transformers import transformer_cosmos3

        attn = layer.self_attn
        q_gen = attn.add_q_proj(gen_norm).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_norm).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_norm).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)
        _, _, cos_gen, sin_gen = rotary_emb
        cos_gen = cos_gen.unsqueeze(1)
        sin_gen = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + transformer_cosmos3._rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + transformer_cosmos3._rotate_half(k_gen) * sin_gen
        return q_gen, k_gen, v_gen

    def _full_attention(
        self,
        layer: Any,
        q_gen: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        k_gen: torch.Tensor,
        v_gen: torch.Tensor,
    ) -> torch.Tensor:
        from diffusers.models.transformers import transformer_cosmos3

        all_k = torch.cat([k_und, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        full_out = transformer_cosmos3.dispatch_attention_fn(
            q_gen.unsqueeze(0),
            all_k.unsqueeze(0),
            all_v.unsqueeze(0),
            is_causal=False,
            enable_gqa=True,
            **self._attention_backend_args(layer),
        )
        return full_out.squeeze(0).flatten(-2, -1)

    def _forward_write(
        self,
        layer: Any,
        index: int,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from diffusers.models.transformers import transformer_cosmos3

        und_norm = layer.input_layernorm(und_seq)
        gen_norm = layer.input_layernorm_moe_gen(gen_seq)
        q_und, k_und, v_und = self._project_und(layer, und_norm, rotary_emb)
        q_gen, k_gen, v_gen = self._project_gen(layer, gen_norm, rotary_emb)
        causal_out = transformer_cosmos3.dispatch_attention_fn(
            q_und.unsqueeze(0),
            k_und.unsqueeze(0),
            v_und.unsqueeze(0),
            is_causal=True,
            enable_gqa=True,
            **self._attention_backend_args(layer),
        )
        causal_out = causal_out.squeeze(0).flatten(-2, -1)
        full_out = self._full_attention(layer, q_gen, k_und, v_und, k_gen, v_gen)
        und_attn_out = layer.self_attn.to_out(causal_out)
        gen_attn_out = layer.self_attn.to_add_out(full_out)
        residual_und = und_seq + und_attn_out
        residual_gen = gen_seq + gen_attn_out
        mlp_out_und = layer.mlp(layer.post_attention_layernorm(residual_und))
        mlp_out_gen = layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(residual_gen))
        und_next = residual_und + mlp_out_und
        gen_next = residual_gen + mlp_out_gen
        self.cache[index] = {
            "und_next": und_next.detach(),
            "k_und": k_und.detach(),
            "v_und": v_und.detach(),
        }
        return und_next, gen_next

    def _forward_read(
        self,
        layer: Any,
        index: int,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self.cache[index]
        gen_norm = layer.input_layernorm_moe_gen(gen_seq)
        q_gen, k_gen, v_gen = self._project_gen(layer, gen_norm, rotary_emb)
        full_out = self._full_attention(layer, q_gen, cached["k_und"], cached["v_und"], k_gen, v_gen)
        gen_attn_out = layer.self_attn.to_add_out(full_out)
        residual_gen = gen_seq + gen_attn_out
        mlp_out_gen = layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(residual_gen))
        gen_next = residual_gen + mlp_out_gen
        return cached["und_next"], gen_next

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        bytes_total = 0
        for layer_cache in self.cache.values():
            for tensor in layer_cache.values():
                bytes_total += tensor.numel() * tensor.element_size()
        return {
            "enabled": True,
            "transformer_calls": self.transformer_calls,
            "write_calls": self.write_calls,
            "read_calls": self.read_calls,
            "invalidations": self.invalidations,
            "cached_layers": len(self.cache),
            "cache_gib": round(bytes_total / 1024**3, 3),
        }

    def restore(self) -> None:
        for obj, method_name, original in reversed(self._patched):
            setattr(obj, method_name, original)
        self._patched.clear()
        self.cache.clear()


def mem(label: str) -> dict:
    free, total = torch.cuda.mem_get_info()
    data = {
        "label": label,
        "free_gib": round(free / 1024**3, 3),
        "total_gib": round(total / 1024**3, 3),
    }
    print(json.dumps(data), flush=True)
    return data


def read_prompt_json(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=False)


def load_pipeline() -> tuple[Cosmos3OmniPipeline, float]:
    started = time.perf_counter()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        enable_safety_checker=False,
    )
    torch.cuda.synchronize()
    seconds = round(time.perf_counter() - started, 3)
    print("load_seconds", seconds, flush=True)
    mem("after_load")
    return pipe, seconds


def configure_tunableop(args: argparse.Namespace) -> dict[str, Any]:
    requested = {
        "max_tuning_duration": args.tunable_max_tuning_duration,
        "max_tuning_iterations": args.tunable_max_tuning_iterations,
        "rotating_buffer_size": args.tunable_rotating_buffer_size,
    }
    if not any(value is not None for value in requested.values()):
        return {"enabled": False, "requested": requested, "applied": {}}

    try:
        import torch.cuda.tunable as tunable
    except Exception as exc:
        data = {"enabled": False, "requested": requested, "applied": {}, "error": repr(exc)}
        print(json.dumps({"tunableop_config": data}, ensure_ascii=False), flush=True)
        return data

    setters = {
        "max_tuning_duration": "set_max_tuning_duration",
        "max_tuning_iterations": "set_max_tuning_iterations",
        "rotating_buffer_size": "set_rotating_buffer_size",
    }
    applied = {}
    errors = {}
    for key, setter_name in setters.items():
        value = requested[key]
        if value is None:
            continue
        setter = getattr(tunable, setter_name, None)
        if not callable(setter):
            errors[key] = f"{setter_name} is not available"
            continue
        try:
            setter(int(value))
            applied[key] = int(value)
        except Exception as exc:
            errors[key] = repr(exc)

    data = {"enabled": bool(applied), "requested": requested, "applied": applied}
    if errors:
        data["errors"] = errors
    print(json.dumps({"tunableop_config": data}, ensure_ascii=False), flush=True)
    return data


def call_pipe(args: argparse.Namespace, fn: Callable[[], Any]) -> Any:
    if args.inference_mode:
        with torch.inference_mode():
            return fn()
    return fn()


def install_vae_decode_abort(pipe: Cosmos3OmniPipeline) -> Callable[[], None]:
    original = pipe.vae.decode

    def aborting_decode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Intentional abort before VAE decode for transformer-only benchmark")

    pipe.vae.decode = aborting_decode

    def restore() -> None:
        pipe.vae.decode = original

    return restore


def install_vae_encode_sdpa_math(pipe: Cosmos3OmniPipeline) -> Callable[[], None]:
    original = pipe._encode_video
    original_sdpa = F.scaled_dot_product_attention

    def naive_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if enable_gqa and query.size(-3) != key.size(-3):
            repeat = query.size(-3) // key.size(-3)
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)
        scale_factor = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        if is_causal:
            causal_mask = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        if dropout_p:
            probs = torch.dropout(probs, dropout_p, train=True)
        return torch.matmul(probs, value)

    def wrapped_encode_video(*args: Any, **kwargs: Any) -> Any:
        F.scaled_dot_product_attention = naive_sdpa
        try:
            return original(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = original_sdpa

    pipe._encode_video = wrapped_encode_video

    def restore() -> None:
        pipe._encode_video = original
        F.scaled_dot_product_attention = original_sdpa

    return restore


def naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    if enable_gqa and query.size(-2) != key.size(-2):
        repeat = query.size(-2) // key.size(-2)
        key = key.repeat_interleave(repeat, dim=-2)
        value = value.repeat_interleave(repeat, dim=-2)
    scale_factor = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))
    query_t, key_t, value_t = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    scores = torch.matmul(query_t.float(), key_t.float().transpose(-2, -1)) * scale_factor
    if is_causal:
        causal_mask = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if attn_mask is not None:
        mask = attn_mask
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask
    probs = torch.softmax(scores, dim=-1).to(value_t.dtype)
    if dropout_p:
        probs = torch.dropout(probs, dropout_p, train=True)
    out = torch.matmul(probs, value_t)
    return out.permute(0, 2, 1, 3)


def install_cosmos3_transformer_attention_fallback() -> Callable[[], None]:
    import diffusers.models.transformers.transformer_cosmos3 as transformer_cosmos3

    original = transformer_cosmos3.dispatch_attention_fn

    def wrapped_dispatch_attention_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
        try:
            return original(*args, **kwargs)
        except RuntimeError as exc:
            if "Expected iter != ops_.end()" not in str(exc):
                raise
            return naive_attention(*args, **kwargs)

    transformer_cosmos3.dispatch_attention_fn = wrapped_dispatch_attention_fn

    def restore() -> None:
        transformer_cosmos3.dispatch_attention_fn = original

    return restore


def run_t2v(pipe: Cosmos3OmniPipeline, args: argparse.Namespace, out_dir: Path, profiler: StageProfiler) -> dict:
    profiler.reset()
    started = time.perf_counter()
    try:
        result = call_pipe(
            args,
            lambda: pipe(
                prompt=ARTICLE_T2V_PROMPT,
                negative_prompt="blurry, distorted, low quality, physically implausible motion",
                num_frames=args.frames,
                height=args.height,
                width=args.width,
                fps=float(args.fps),
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(device="cuda").manual_seed(args.t2v_seed),
                enable_safety_check=False,
            ),
        )
        torch.cuda.synchronize()
    except Exception as exc:
        seconds = round(time.perf_counter() - started, 3)
        stages = profiler.snapshot()
        stage_sum = round(sum(float(value["seconds"]) for value in stages.values()), 3)
        if not args.allow_pipeline_error or "transformer_forward" not in stages:
            raise
        data = {
            "case": "article_t2v_red_cube_grasp",
            "seconds": seconds,
            "export_seconds": None,
            "stage_profile": {
                "enabled": args.stage_profile,
                "records": stages,
                "timed_stage_sum_seconds": stage_sum,
                "unattributed_pipe_seconds": round(seconds - stage_sum, 3),
            },
            "output": None,
            "pipeline_error": {
                "allowed": True,
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "prompt_source": "Classmethod article text: gripper grasps and slowly lifts a red cube",
            "settings": {
                "height": args.height,
                "width": args.width,
                "frames": args.frames,
                "fps": args.fps,
                "steps": args.steps,
                "guidance": args.guidance,
                "seed": args.t2v_seed,
            },
        }
        print(json.dumps(data, ensure_ascii=False), flush=True)
        return data
    seconds = round(time.perf_counter() - started, 3)
    output = out_dir / f"article_t2v_red_cube_{args.height}p_{args.frames}f_s{args.steps}.mp4"
    export_started = time.perf_counter()
    export_to_video(result.video, str(output), fps=args.fps, macro_block_size=1)
    export_seconds = round(time.perf_counter() - export_started, 3)
    stages = profiler.snapshot()
    stage_sum = round(sum(float(value["seconds"]) for value in stages.values()), 3)
    data = {
        "case": "article_t2v_red_cube_grasp",
        "seconds": seconds,
        "export_seconds": export_seconds,
        "stage_profile": {
            "enabled": args.stage_profile,
            "records": stages,
            "timed_stage_sum_seconds": stage_sum,
            "unattributed_pipe_seconds": round(seconds - stage_sum, 3),
        },
        "output": str(output),
        "prompt_source": "Classmethod article text: gripper grasps and slowly lifts a red cube",
        "settings": {
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "fps": args.fps,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.t2v_seed,
        },
    }
    print(json.dumps(data, ensure_ascii=False), flush=True)
    return data


def run_i2v(pipe: Cosmos3OmniPipeline, args: argparse.Namespace, out_dir: Path, profiler: StageProfiler) -> dict:
    profiler.reset()
    prompt = read_prompt_json(ASSET_DIR / "example_i2v_prompt.json")
    negative = read_prompt_json(ASSET_DIR / "negative_prompt.json")
    image = Image.open(ASSET_DIR / "example_i2v_input.jpg").convert("RGB")
    started = time.perf_counter()
    try:
        result = call_pipe(
            args,
            lambda: pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=image,
                num_frames=args.frames,
                height=args.height,
                width=args.width,
                fps=float(args.fps),
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(device="cuda").manual_seed(args.i2v_seed),
                enable_safety_check=False,
            ),
        )
        torch.cuda.synchronize()
    except Exception as exc:
        seconds = round(time.perf_counter() - started, 3)
        stages = profiler.snapshot()
        stage_sum = round(sum(float(value["seconds"]) for value in stages.values()), 3)
        if not args.allow_pipeline_error or "transformer_forward" not in stages:
            raise
        data = {
            "case": "article_i2v_robot_arms",
            "seconds": seconds,
            "export_seconds": None,
            "stage_profile": {
                "enabled": args.stage_profile,
                "records": stages,
                "timed_stage_sum_seconds": stage_sum,
                "unattributed_pipe_seconds": round(seconds - stage_sum, 3),
            },
            "output": None,
            "pipeline_error": {
                "allowed": True,
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "prompt_source": "Cosmos3 official sample: example_i2v_input.jpg + example_i2v_prompt.json",
            "settings": {
                "height": args.height,
                "width": args.width,
                "frames": args.frames,
                "fps": args.fps,
                "steps": args.steps,
                "guidance": args.guidance,
                "seed": args.i2v_seed,
            },
        }
        print(json.dumps(data, ensure_ascii=False), flush=True)
        return data
    seconds = round(time.perf_counter() - started, 3)
    output = out_dir / f"article_i2v_robot_arms_{args.height}p_{args.frames}f_s{args.steps}.mp4"
    export_started = time.perf_counter()
    export_to_video(result.video, str(output), fps=args.fps, macro_block_size=1)
    export_seconds = round(time.perf_counter() - export_started, 3)
    stages = profiler.snapshot()
    stage_sum = round(sum(float(value["seconds"]) for value in stages.values()), 3)
    data = {
        "case": "article_i2v_robot_arms",
        "seconds": seconds,
        "export_seconds": export_seconds,
        "stage_profile": {
            "enabled": args.stage_profile,
            "records": stages,
            "timed_stage_sum_seconds": stage_sum,
            "unattributed_pipe_seconds": round(seconds - stage_sum, 3),
        },
        "output": str(output),
        "prompt_source": "Cosmos3 official sample: example_i2v_input.jpg + example_i2v_prompt.json",
        "settings": {
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "fps": args.fps,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.i2v_seed,
        },
    }
    print(json.dumps(data, ensure_ascii=False), flush=True)
    return data


def synthetic_vae_warmup(pipe: Cosmos3OmniPipeline, args: argparse.Namespace, profiler: StageProfiler) -> dict:
    shape = tuple(int(item) for item in args.vae_warmup_shape.split(","))
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


def measured_runs(
    label: str,
    run_fn: Callable[[Cosmos3OmniPipeline, argparse.Namespace, Path, StageProfiler], dict],
    pipe: Cosmos3OmniPipeline,
    args: argparse.Namespace,
    out_dir: Path,
    profiler: StageProfiler,
) -> dict:
    runs = []
    for index in range(args.mode_warmup_runs + args.measured_runs):
        role = "warmup" if index < args.mode_warmup_runs else "measured"
        profiler.set_rocprof_active(args.rocprof_transformer_only and role == "measured")
        try:
            result = run_fn(pipe, args, out_dir, profiler)
        finally:
            profiler.set_rocprof_active(False)
        result["run"] = index + 1
        result["measurement_role"] = role
        runs.append(result)
    measured = [item for item in runs if item["measurement_role"] == "measured"]
    summary = {
        "case": label,
        "mode_warmup_runs": args.mode_warmup_runs,
        "measured_runs": args.measured_runs,
        "runs": runs,
        "selected_seconds": measured[-1]["seconds"] if measured else None,
    }
    print(json.dumps({"measured_case": summary}, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["t2v", "i2v", "both"], default="both")
    parser.add_argument("--out-dir", default="/workspace/result/classmethod_article_benchmark")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--t2v-seed", type=int, default=202)
    parser.add_argument("--i2v-seed", type=int, default=203)
    parser.add_argument("--stage-profile", action="store_true")
    parser.add_argument("--vae-warmup", action="store_true")
    parser.add_argument("--vae-warmup-shape", default="1,48,2,16,28")
    parser.add_argument("--mode-warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--rocprof-transformer-only", action="store_true")
    parser.add_argument("--inference-mode", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--allow-pipeline-error", action="store_true")
    parser.add_argument("--abort-before-vae-decode", action="store_true")
    parser.add_argument("--force-vae-encode-sdpa-math", action="store_true")
    parser.add_argument("--cosmos3-transformer-attention-fallback", action="store_true")
    parser.add_argument("--linear-profile", action="store_true")
    parser.add_argument("--linear-profile-top", type=int, default=40)
    parser.add_argument("--transformer-input-profile", action="store_true")
    parser.add_argument("--transformer-input-profile-calls", type=int, default=3)
    parser.add_argument("--und-branch-cache", action="store_true")
    parser.add_argument("--tunable-max-tuning-duration", type=int)
    parser.add_argument("--tunable-max-tuning-iterations", type=int)
    parser.add_argument("--tunable-rotating-buffer-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, flush=True)
    print("hip", torch.version.hip, flush=True)
    print("device", torch.cuda.get_device_name(0), flush=True)
    tunableop_config = configure_tunableop(args)
    mem("before_load")
    pipe, load_seconds = load_pipeline()
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    input_profiler = TransformerInputProfiler(
        enabled=args.transformer_input_profile, max_calls=args.transformer_input_profile_calls
    )
    input_profiler.install(pipe)
    native_und_cache = False
    und_cache = UndBranchCachePrototype(enabled=False)
    if args.und_branch_cache and hasattr(pipe.transformer, "enable_und_branch_cache"):
        pipe.transformer.enable_und_branch_cache(True, reset=True)
        native_und_cache = True
    else:
        und_cache = UndBranchCachePrototype(enabled=args.und_branch_cache)
        und_cache.install(pipe)
    profiler = StageProfiler(enabled=args.stage_profile, rocprof_transformer_only=args.rocprof_transformer_only)
    profiler.install(pipe)
    linear_profiler = LinearProfiler(enabled=args.linear_profile, top=args.linear_profile_top)
    linear_profiler.install(pipe)
    restore_vae_encode_sdpa_math = install_vae_encode_sdpa_math(pipe) if args.force_vae_encode_sdpa_math else None
    restore_vae_decode_abort = install_vae_decode_abort(pipe) if args.abort_before_vae_decode else None
    restore_transformer_attention_fallback = (
        install_cosmos3_transformer_attention_fallback() if args.cosmos3_transformer_attention_fallback else None
    )
    warmup = synthetic_vae_warmup(pipe, args, profiler) if args.vae_warmup else {"enabled": False}
    results = []
    if args.case in {"t2v", "both"}:
        results.append(measured_runs("article_t2v_red_cube_grasp", run_t2v, pipe, args, out_dir, profiler))
    if args.case in {"i2v", "both"}:
        results.append(measured_runs("article_i2v_robot_arms", run_i2v, pipe, args, out_dir, profiler))
    summary = {
        "model": MODEL_ID,
        "dtype": "float16",
        "load_seconds": load_seconds,
        "comparison_target": {
            "source": "Classmethod DGX Spark article",
            "t2v": "256p, 24 frames, 12 fps, model resident after about 22 sec",
            "i2v": "official sample image/prompt, article reported about 17 sec",
        },
        "warmup": warmup,
        "tunableop_config": tunableop_config,
        "runtime_options": {
            "inference_mode": args.inference_mode,
            "disable_progress_bar": args.disable_progress_bar,
            "stage_profile": args.stage_profile,
            "allow_pipeline_error": args.allow_pipeline_error,
            "abort_before_vae_decode": args.abort_before_vae_decode,
            "force_vae_encode_sdpa_math": args.force_vae_encode_sdpa_math,
            "cosmos3_transformer_attention_fallback": args.cosmos3_transformer_attention_fallback,
            "linear_profile": args.linear_profile,
            "transformer_input_profile": args.transformer_input_profile,
            "und_branch_cache": args.und_branch_cache,
        },
        "results": results,
        "linear_profile": linear_profiler.snapshot(),
        "transformer_input_profile": input_profiler.snapshot(),
        "und_branch_cache": (
            pipe.transformer.get_und_branch_cache_stats() if native_und_cache else und_cache.snapshot()
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if restore_transformer_attention_fallback is not None:
        restore_transformer_attention_fallback()
    if restore_vae_decode_abort is not None:
        restore_vae_decode_abort()
    if restore_vae_encode_sdpa_math is not None:
        restore_vae_encode_sdpa_math()
    linear_profiler.restore()
    profiler.restore()
    if native_und_cache:
        pipe.transformer.disable_und_branch_cache()
    else:
        und_cache.restore()
    input_profiler.restore()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    mem("after_cleanup")


if __name__ == "__main__":
    main()
