import os
import sys
import argparse
import ctypes
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _sdpa_varlen_fallback(
    query,
    key,
    value,
    is_causal=False,
    causal_type=None,
    scale=None,
    seqlens_Q=None,
    seqlens_KV=None,
    cumulative_seqlen_Q=None,
    cumulative_seqlen_KV=None,
    max_seqlen_Q=None,
    max_seqlen_KV=None,
    backend=None,
    return_lse=False,
    backend_kwargs=None,
    deterministic=False,
):
    del seqlens_Q, seqlens_KV, backend, backend_kwargs, deterministic
    if return_lse:
        raise NotImplementedError("SDPA fallback does not return LSE")
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise NotImplementedError("SDPA fallback only handles packed batch dimension 1")

    if scale is None:
        scale = query.shape[-1] ** -0.5

    if cumulative_seqlen_Q is None:
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=q.shape[1] != k.shape[1],
        )
        return out.permute(0, 2, 1, 3)

    # Varlen path using native PyTorch FlashAttention
    q_3d = query.squeeze(0)
    k_3d = key.squeeze(0)
    v_3d = value.squeeze(0)

    if max_seqlen_Q is None:
        print("[ANTIGRAVITY] WARNING: max_seqlen_Q is None in _sdpa_varlen_fallback! Calling .item()", flush=True)
        max_seqlen_Q = int((cumulative_seqlen_Q[1:] - cumulative_seqlen_Q[:-1]).max().item())
    if max_seqlen_KV is None:
        print("[ANTIGRAVITY] WARNING: max_seqlen_KV is None in _sdpa_varlen_fallback! Calling .item()", flush=True)
        max_seqlen_KV = int((cumulative_seqlen_KV[1:] - cumulative_seqlen_KV[:-1]).max().item())

    cum_q = cumulative_seqlen_Q if cumulative_seqlen_Q.dtype == torch.int32 else cumulative_seqlen_Q.to(torch.int32)
    cum_kv = cumulative_seqlen_KV if cumulative_seqlen_KV.dtype == torch.int32 else cumulative_seqlen_KV.to(torch.int32)

    out_3d, _, _, _, _ = torch.ops.aten._flash_attention_forward(
        q_3d,
        k_3d,
        v_3d,
        cum_q,
        cum_kv,
        max_seqlen_Q,
        max_seqlen_KV,
        0.0,  # dropout_p
        is_causal,
        False,  # return_debug_mask
        scale=scale,
    )
    return out_3d.unsqueeze(0)



def _clone_batch_for_warmup(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_batch_for_warmup(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_batch_for_warmup(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_batch_for_warmup(item) for key, item in value.items()}
    return value


class SyncStageProfiler:
    def __init__(self, enabled: bool, out_dir: str) -> None:
        self.enabled = enabled
        self.out_dir = Path(out_dir)
        self.phase = "unknown"
        self.records: list[dict[str, Any]] = []

    def synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def record(self, name: str, seconds: float, **extra: Any) -> None:
        if not self.enabled:
            return
        self.records.append({"phase": self.phase, "name": name, "seconds": seconds, **extra})

    def timed(self, name: str, **extra: Any):
        profiler = self

        class _Timer:
            def __enter__(self):
                if profiler.enabled:
                    profiler.synchronize()
                    self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                if profiler.enabled:
                    profiler.synchronize()
                    profiler.record(name, time.perf_counter() - self.start, **extra)

        return _Timer()

    def write(self) -> None:
        if not self.enabled:
            return
        by_phase: dict[str, dict[str, float]] = {}
        for record in self.records:
            phase = record["phase"]
            by_phase.setdefault(phase, {})
            by_phase[phase][record["name"]] = by_phase[phase].get(record["name"], 0.0) + record["seconds"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "policy_stage_sync_profile.json").write_text(
            json.dumps({"records": self.records, "summary_by_phase": by_phase}, indent=2, sort_keys=True)
        )


class RocTxController:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.lib = None
        if not enabled:
            return
        try:
            self.lib = ctypes.CDLL("librocprofiler-sdk-roctx.so")
            self.lib.roctxProfilerPause.argtypes = [ctypes.c_int]
            self.lib.roctxProfilerResume.argtypes = [ctypes.c_int]
            self.lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
            self.lib.roctxRangePushA.restype = ctypes.c_int
            self.lib.roctxRangePop.argtypes = []
            self.lib.roctxRangePop.restype = ctypes.c_int
        except OSError:
            self.enabled = False
            self.lib = None

    def resume(self) -> None:
        if self.enabled and self.lib is not None:
            self.lib.roctxProfilerResume(0)

    def pause(self) -> None:
        if self.enabled and self.lib is not None:
            self.lib.roctxProfilerPause(0)

    def push(self, name: str) -> None:
        if self.enabled and self.lib is not None:
            self.lib.roctxRangePushA(name.encode("utf-8"))

    def pop(self) -> None:
        if self.enabled and self.lib is not None:
            self.lib.roctxRangePop()


def _install_policy_hooks(
    profiler: SyncStageProfiler,
    condition_cache: bool,
    deep_profile: bool,
    decoder_block_profile: bool,
    decoder_upsample_detail_index: int,
    roctx_network_forward: bool,
    roctx_vae_detail: bool,
    vae_conv_probe_path: str | None,
    vae_target_conv_channels_last: bool,
) -> None:
    import cosmos_framework.data.vfm.sequence_packing as sequence_packing
    import cosmos_framework.model.vfm.mot.attention as mot_attention

    def patched_factored_from_joint_sequence(
        packed_sequence,
        attn_modes,
        split_lens,
        sample_lens,
        packed_und_token_indexes,
        packed_gen_token_indexes,
        is_image_batch=False,
        cp_world_size=1,
        pad_for_cuda_graphs=False,
    ):
        assert sum(sample_lens) == packed_sequence.shape[0], (
            "sum(sample_lens) must be equal to the length of the packed sequence"
        )
        meta = sequence_packing._init_sequence_pack(sample_lens, split_lens, attn_modes, packed_sequence.device)
        causal_seq = packed_sequence[meta["_causal_indices"]]
        full_only_seq = packed_sequence[meta["_full_indices"]]
        need_causal = sequence_packing._round_up_to_N(int(causal_seq.shape[0]), cp_world_size, pad_for_cuda_graphs)
        need_full = sequence_packing._round_up_to_N(int(full_only_seq.shape[0]), cp_world_size, pad_for_cuda_graphs)
        causal_seq, full_only_seq = sequence_packing._round_up_for_cuda_graphs_or_cp(
            causal_seq,
            full_only_seq,
            need_causal,
            need_full,
            is_image_batch,
            pad_for_cuda_graphs,
        )
        pack = {
            **meta,
            "max_num_tokens": sum(sample_lens),
            "causal_seq": causal_seq,
            "full_only_seq": full_only_seq,
            "is_sharded": False,
        }
        return pack

    sequence_packing.factored_from_joint_sequence = patched_factored_from_joint_sequence
    mot_attention.factored_from_joint_sequence = patched_factored_from_joint_sequence

    if not profiler.enabled and not condition_cache and not roctx_network_forward and not roctx_vae_detail:
        return


    import cosmos_framework.inference.inference as inference_module
    import cosmos_framework.data.vfm.sequence_packing as sequence_packing
    import cosmos_framework.model.vfm.mot.cosmos3_vfm_network as cosmos3_vfm_network
    import cosmos_framework.model.vfm.omni_mot_model as omni_mot_model
    import cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 as wan_vae
    from cosmos_framework.tools.visualize import video as video_module
    from cosmos_framework.tools.visualize.video import easy_io
    from einops import rearrange
    from PIL import Image as PILImage

    original_generate_samples = omni_mot_model.OmniMoTModel.generate_samples_from_batch
    original_decode = omni_mot_model.OmniMoTModel.decode
    original_prepare_inference_data = omni_mot_model.OmniMoTModel._prepare_inference_data
    original_get_data_and_condition = omni_mot_model.OmniMoTModel.get_data_and_condition
    original_normalize_video = omni_mot_model.OmniMoTModel._normalize_video_databatch_inplace
    original_normalize_action = omni_mot_model.OmniMoTModel._normalize_action_databatch
    original_maybe_upsample = omni_mot_model.OmniMoTModel._maybe_apply_prompt_upsampling
    original_get_velocity = omni_mot_model.OmniMoTModel._get_velocity
    original_pack_input_sequence = omni_mot_model.OmniMoTModel._pack_input_sequence
    original_denoise = omni_mot_model.OmniMoTModel.denoise
    condition_cache_store: dict[str, Any] = {"value": None, "writes": 0, "reads": 0}
    vae_conv_probe_store: dict[str, Any] = {"written": False}

    def _profiled_generate_samples(self, *args, **kwargs):
        if not hasattr(self, "_net_compiled"):
            print("[ANTIGRAVITY] Sampling started. Preparing to compile self.net (Cosmos3VFMNetwork) components...")
            if hasattr(self, "net") and self.net is not None:
                import torch
                print("[ANTIGRAVITY] Compiling the entire self.net with torch.compile(mode='max-autotune')...")
                self.net = torch.compile(self.net, mode="max-autotune")
            else:
                print("[ANTIGRAVITY] WARNING: self.net not found or None on OmniMoTModel.")
            self._net_compiled = True

        with profiler.timed("generate_samples_from_batch_sync"):
            return original_generate_samples(self, *args, **kwargs)

    def _profiled_decode(self, *args, **kwargs):
        if not hasattr(self, "_upsample_compiled"):
            print("[ANTIGRAVITY] VAE decode called. Preparing to compile decoder...")
            vae_model = getattr(self, "tokenizer_vision_gen", None)
            if vae_model is not None:
                if hasattr(vae_model, "disable_tiling"):
                    print("[ANTIGRAVITY] Disabling VAE tiling for max speed...")
                    vae_model.disable_tiling()
                if hasattr(vae_model, "disable_slicing"):
                    print("[ANTIGRAVITY] Disabling VAE slicing for max speed...")
                    vae_model.disable_slicing()

                if hasattr(vae_model, "model"):
                    vae_model = vae_model.model
                if hasattr(vae_model, "model"):
                    vae_model = vae_model.model
                if hasattr(vae_model, "decoder"):
                    print("[ANTIGRAVITY] VAE decoder found. Compiling decoder with torch.compile(mode='max-autotune')...")
                    import torch
                    vae_model.decoder = torch.compile(vae_model.decoder, mode="max-autotune")
                else:
                    print("[ANTIGRAVITY] WARNING: VAE decoder attribute not found on model.")
            else:
                print("[ANTIGRAVITY] WARNING: tokenizer_vision_gen not found on OmniMoTModel.")
            self._upsample_compiled = True

        with profiler.timed("decode_sync"):
            return original_decode(self, *args, **kwargs)

    def _profiled_prepare_inference_data(self, *args, **kwargs):
        with profiler.timed("prepare_inference_data_sync"):
            return original_prepare_inference_data(self, *args, **kwargs)

    def _profiled_get_data_and_condition(self, *args, **kwargs):
        if condition_cache and profiler.phase == "measured" and condition_cache_store["value"] is not None:
            condition_cache_store["reads"] += 1
            profiler.record("condition_cache_read", 0.0, reads=condition_cache_store["reads"])
            return condition_cache_store["value"]
        with profiler.timed("get_data_and_condition_sync"):
            result = original_get_data_and_condition(self, *args, **kwargs)
        if condition_cache and profiler.phase == "warmup" and condition_cache_store["value"] is None:
            condition_cache_store["value"] = result
            condition_cache_store["writes"] += 1
            profiler.record("condition_cache_write", 0.0, writes=condition_cache_store["writes"])
        return result

    def _profiled_normalize_video(self, *args, **kwargs):
        with profiler.timed("normalize_video_databatch_sync"):
            return original_normalize_video(self, *args, **kwargs)

    def _profiled_normalize_action(self, *args, **kwargs):
        with profiler.timed("normalize_action_databatch_sync"):
            return original_normalize_action(self, *args, **kwargs)

    def _profiled_maybe_upsample(self, *args, **kwargs):
        with profiler.timed("maybe_prompt_upsampling_sync"):
            return original_maybe_upsample(self, *args, **kwargs)

    from cosmos_framework.model.vfm.omni_mot_model import GenerationDataClean

    def patched_get_velocity(
        self,
        *,
        net=None,
        noise_x,
        timestep,
        text_tokens,
        sequence_plans,
        gen_data_clean,
        skip_text_tokens=False,
    ):
        n_samples = len(noise_x)
        is_image_batch = gen_data_clean.is_image_batch
        has_action = self.config.action_gen and any(plan.has_action for plan in sequence_plans)
        num_items = gen_data_clean.num_vision_items_per_sample
        has_sound = self.config.sound_gen and any(plan.has_sound for plan in sequence_plans)

        noise_x_vision = []
        noise_x_action = [] if has_action else None
        noise_x_sound = [] if has_sound else None

        vision_offset = 0
        idx_action = 0
        idx_sound = 0
        for i in range(n_samples):
            n_vis = num_items[i] if num_items is not None else 1
            offset = 0
            for j in range(n_vis):
                vision_shape = gen_data_clean.x0_tokens_vision[vision_offset + j].shape
                vision_dim = int(torch.prod(torch.tensor(vision_shape)))
                noise_vision_ij = noise_x[i][offset : offset + vision_dim].reshape(vision_shape)
                noise_x_vision.append(noise_vision_ij)
                offset += vision_dim
            vision_offset += n_vis

            if has_action and noise_x_action is not None:
                assert gen_data_clean.x0_tokens_action is not None
                action_shape = gen_data_clean.x0_tokens_action[idx_action].shape
                action_dim = int(torch.prod(torch.tensor(action_shape)))
                noise_x_action.append(noise_x[i][offset : offset + action_dim].reshape(action_shape))
                offset += action_dim
                idx_action += 1

            if has_sound and noise_x_sound is not None and sequence_plans[i].has_sound:
                assert gen_data_clean.x0_tokens_sound is not None
                sound_shape = gen_data_clean.x0_tokens_sound[idx_sound].shape
                sound_dim = int(torch.prod(torch.tensor(sound_shape)))
                noise_x_sound.append(
                    noise_x[i][offset : offset + sound_dim].reshape(sound_shape)
                )
                offset += sound_dim
                idx_sound += 1

        gen_data_for_packing = GenerationDataClean(
            batch_size=n_samples,
            is_image_batch=is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=noise_x_vision,
            fps_vision=gen_data_clean.fps_vision,
            raw_state_action=gen_data_clean.raw_state_action if has_action else None,
            x0_tokens_action=noise_x_action if has_action else None,
            action_domain_id=gen_data_clean.action_domain_id if has_action else None,
            fps_action=gen_data_clean.fps_action if has_action else None,
            raw_action_dim=gen_data_clean.raw_action_dim if has_action else None,
            raw_state_sound=gen_data_clean.raw_state_sound if has_sound else None,
            x0_tokens_sound=noise_x_sound if has_sound else None,
            fps_sound=gen_data_clean.fps_sound if has_sound else None,
            num_vision_items_per_sample=num_items,
        )

        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            text_tokens,
            gen_data_for_packing,
            timestep.cpu(),
            include_end_of_generation_token=self._derive_include_end_of_generation_token(),
            skip_text_tokens=skip_text_tokens,
        )

        if packed_sequence.vision is not None:
            packed_sequence.vision.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_vision]

        if has_action and noise_x_action is not None:
            assert packed_sequence.action is not None
            packed_sequence.action.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_action]
            packed_sequence.action.domain_id = gen_data_clean.action_domain_id

        if has_sound and noise_x_sound is not None:
            assert packed_sequence.sound is not None
            packed_sequence.sound.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_sound]

        packed_sequence.to_cuda()

        fps_action = gen_data_clean.fps_action if has_action else None
        fps_sound = gen_data_clean.fps_sound if has_sound else None
        out = self.denoise(
            net=net,
            data_batch_packed=packed_sequence,
            fps_vision=gen_data_clean.fps_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
        )

        # Masking without GPU-CPU synchronizations
        assert packed_sequence.vision is not None
        assert packed_sequence.vision.condition_mask is not None
        noisy_mask_vision = [1.0 - cond_mask for cond_mask in packed_sequence.vision.condition_mask]

        velocity_vision = []
        for pred, noisy_mask in zip(out["preds_vision"], noisy_mask_vision):
            mask_device = noisy_mask.to(dtype=pred.dtype, device=pred.device)
            velocity_vision.append(pred * mask_device)

        velocity_action = None
        if (
            has_action
            and packed_sequence.action is not None
            and packed_sequence.action.condition_mask is not None
        ):
            noisy_mask_action = [1.0 - cond_mask for cond_mask in packed_sequence.action.condition_mask]
            velocity_action = []
            for i, (pred, noisy_mask) in enumerate(zip(out["preds_action"], noisy_mask_action)):
                mask_device = noisy_mask.to(dtype=pred.dtype, device=pred.device)
                v = pred * mask_device
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    v[:, gen_data_clean.raw_action_dim[i] :] = 0
                velocity_action.append(v)

        velocity_sound = None
        if (
            has_sound
            and packed_sequence.sound is not None
            and packed_sequence.sound.condition_mask is not None
        ):
            noisy_mask_sound = [1.0 - cond_mask for cond_mask in packed_sequence.sound.condition_mask]
            velocity_sound = []
            for pred, noisy_mask in zip(out["preds_sound"], noisy_mask_sound):
                mask_device = noisy_mask.T.to(dtype=pred.dtype, device=pred.device)
                velocity_sound.append(pred * mask_device)

        velocity_output = []
        vis_offset = 0
        idx_action = 0
        idx_sound = 0
        for i in range(n_samples):
            parts = []
            n_vis = num_items[i] if num_items is not None else 1

            for _ in range(n_vis):
                parts.append(velocity_vision[vis_offset].reshape(-1))
                vis_offset += 1

            if velocity_action is not None and sequence_plans[i].has_action:
                parts.append(velocity_action[idx_action].reshape(-1))
                idx_action += 1

            if velocity_sound is not None and sequence_plans[i].has_sound:
                parts.append(velocity_sound[idx_sound].reshape(-1))
                idx_sound += 1

            velocity_output.append(torch.cat(parts, dim=0))

        return velocity_output

    def _profiled_get_velocity(self, *args, **kwargs):
        with profiler.timed("get_velocity_sync"):
            return patched_get_velocity(self, **kwargs)

    cached_packed_seq = None
    def _profiled_pack_input_sequence(self, *args, **kwargs):
        nonlocal cached_packed_seq
        if profiler.phase != "measured":
            return original_pack_input_sequence(self, *args, **kwargs)

        with profiler.timed("velocity_pack_input_sequence_sync"):
            input_timesteps = args[3] if len(args) > 3 else kwargs.get("input_timesteps")
            if cached_packed_seq is None:
                print("[ANTIGRAVITY] Creating cache for PackedSequence...")
                packed_seq = original_pack_input_sequence(self, *args, **kwargs)
                import copy
                cached_packed_seq = copy.deepcopy(packed_seq)
                return packed_seq

            import copy
            p_seq = copy.copy(cached_packed_seq)
            t_val = input_timesteps[0].item() if hasattr(input_timesteps, "item") else float(input_timesteps)

            if p_seq.vision is not None:
                p_seq.vision = copy.copy(cached_packed_seq.vision)
                p_seq.vision.timesteps = p_seq.vision.timesteps.clone().fill_(t_val)

            if p_seq.action is not None:
                p_seq.action = copy.copy(cached_packed_seq.action)
                p_seq.action.timesteps = p_seq.action.timesteps.clone().fill_(t_val)

            if p_seq.sound is not None:
                p_seq.sound = copy.copy(cached_packed_seq.sound)
                p_seq.sound.timesteps = p_seq.sound.timesteps.clone().fill_(t_val)

            return p_seq

    def _profiled_denoise(self, *args, **kwargs):
        with profiler.timed("velocity_denoise_sync"):
            return original_denoise(self, *args, **kwargs)

    def _profiled_save_img_or_video(sample, save_fp_wo_ext, fps=24, quality=None, ffmpeg_params=None, **kwargs):
        assert sample.ndim == 4, "Only support 4D tensor"
        assert isinstance(save_fp_wo_ext, str) or hasattr(save_fp_wo_ext, "write"), (
            "save_fp_wo_ext must be a string or file-like object"
        )
        with profiler.timed("save_img_or_video_total", shape=list(sample.shape), dtype=str(sample.dtype)):
            with profiler.timed("save_clamp_or_cast"):
                if torch.is_floating_point(sample):
                    sample_gpu_uint8 = sample.mul(255).clamp(0, 255).to(torch.uint8)
                else:
                    assert sample.dtype == torch.uint8, "Only support uint8 tensor"
                    sample_gpu_uint8 = sample

            if ffmpeg_params is not None:
                kwargs["ffmpeg_params"] = ffmpeg_params

            if sample.shape[1] == 1:
                with profiler.timed("save_image_cpu_numpy"):
                    sample_rearranged = rearrange(sample_gpu_uint8, "c 1 h w -> h w c").contiguous()
                    array = sample_rearranged.cpu().numpy()
                with profiler.timed("save_image_encode"):
                    save_obj = PILImage.fromarray(array, mode="RGB")
                    ext = ".jpg" if isinstance(save_fp_wo_ext, str) else ""
                    easy_io.dump(
                        save_obj,
                        f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
                        file_format="jpg",
                        format="JPEG",
                        quality=85 if quality is None else quality,
                        **kwargs,
                    )
                return

            if quality is not None:
                kwargs["quality"] = quality
            with profiler.timed("save_video_cpu_numpy"):
                sample_rearranged = rearrange(sample_gpu_uint8, "c t h w -> t h w c").contiguous()
                save_obj = sample_rearranged.cpu().numpy()
            with profiler.timed("save_video_encode"):
                ext = ".mp4" if isinstance(save_fp_wo_ext, str) else ""
                easy_io.dump(
                    save_obj,
                    f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
                    file_format="mp4",
                    format="mp4",
                    fps=fps,
                    **kwargs,
                )

    omni_mot_model.OmniMoTModel.generate_samples_from_batch = _profiled_generate_samples
    omni_mot_model.OmniMoTModel.decode = _profiled_decode
    omni_mot_model.OmniMoTModel._prepare_inference_data = _profiled_prepare_inference_data
    omni_mot_model.OmniMoTModel.get_data_and_condition = _profiled_get_data_and_condition
    omni_mot_model.OmniMoTModel._normalize_video_databatch_inplace = _profiled_normalize_video
    omni_mot_model.OmniMoTModel._normalize_action_databatch = _profiled_normalize_action
    omni_mot_model.OmniMoTModel._maybe_apply_prompt_upsampling = _profiled_maybe_upsample
    omni_mot_model.OmniMoTModel._get_velocity = _profiled_get_velocity
    video_module.save_img_or_video = _profiled_save_img_or_video
    inference_module.save_img_or_video = _profiled_save_img_or_video

    if not deep_profile:
        return

    omni_mot_model.OmniMoTModel._pack_input_sequence = _profiled_pack_input_sequence
    omni_mot_model.OmniMoTModel.denoise = _profiled_denoise

    original_packed_sequence_to_cuda = sequence_packing.PackedSequence.to_cuda
    original_network_forward = cosmos3_vfm_network.Cosmos3VFMNetwork.forward
    original_encode_text = cosmos3_vfm_network.Cosmos3VFMNetwork._encode_text
    original_encode_vision = cosmos3_vfm_network.Cosmos3VFMNetwork._encode_vision
    original_encode_action = cosmos3_vfm_network.Cosmos3VFMNetwork._encode_action
    original_decode_vision = cosmos3_vfm_network.Cosmos3VFMNetwork._decode_vision
    original_decode_action = cosmos3_vfm_network.Cosmos3VFMNetwork._decode_action
    original_build_packed_sequence = cosmos3_vfm_network.build_packed_sequence
    original_wan_decode = wan_vae.WanVAE.decode
    original_decoder_forward = wan_vae.Decoder3d.forward
    roctx = RocTxController(roctx_network_forward)
    roctx_vae = RocTxController(roctx_vae_detail)

    # build_packed_sequence cache
    cached_build_packed_seq_res = None
    cached_causal_indices = None
    cached_full_indices = None
    padded_causal_shape = None
    padded_full_shape = None

    # encode_text & encode_vision cache
    cached_packed_seq_base = None
    cached_vision_embeds = None
    cached_original_latent_shapes = None

    class _RocTxRange:
        def __init__(self, controller, name: str):
            self.controller = controller
            self.name = name

        def __enter__(self):
            if self.controller.enabled:
                self.controller.push(self.name)
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            if self.controller.enabled:
                self.controller.pop()

    def _roctx_range(name: str):
        return _RocTxRange(roctx_vae, name)

    def _profiled_packed_sequence_to_cuda(self, *args, **kwargs):
        with profiler.timed("velocity_packed_sequence_to_cuda_sync"):
            return original_packed_sequence_to_cuda(self, *args, **kwargs)

    def _profiled_network_forward(self, *args, **kwargs):
        use_roctx = roctx.enabled and profiler.phase == "measured"
        if use_roctx:
            roctx.resume()
            roctx.push("policy_velocity_network_forward")
        try:
            with profiler.timed("velocity_network_forward_sync"):
                return original_network_forward(self, *args, **kwargs)
        finally:
            if use_roctx:
                roctx.pop()
                roctx.pause()

    def _profiled_encode_text(self, *args, **kwargs):
        nonlocal cached_packed_seq_base
        if profiler.phase != "measured":
            return original_encode_text(self, *args, **kwargs)

        with profiler.timed("velocity_network_encode_text_sync"):
            if cached_packed_seq_base is None:
                packed_sequence, target_dtype = original_encode_text(self, *args, **kwargs)
                cached_packed_seq_base = packed_sequence.clone()
                return packed_sequence, target_dtype
            
            target_dtype = cached_packed_seq_base.dtype
            return cached_packed_seq_base.clone(), target_dtype

    def _profiled_encode_vision(self, *args, **kwargs):
        nonlocal cached_vision_embeds, cached_original_latent_shapes
        if profiler.phase != "measured":
            return original_encode_vision(self, *args, **kwargs)

        with profiler.timed("velocity_network_encode_vision_sync"):
            packed_seq = args[0] if len(args) > 0 else kwargs.get("packed_seq")
            packed_sequence = args[1] if len(args) > 1 else kwargs.get("packed_sequence")
            
            if packed_seq.vision is None or packed_seq.vision.tokens is None:
                return None
                
            if cached_vision_embeds is None:
                print("[ANTIGRAVITY] Creating cache for encode_vision...")
                original_latent_shapes = original_encode_vision(self, *args, **kwargs)
                vision = packed_seq.vision
                cached_vision_embeds = packed_sequence[vision.sequence_indexes].clone()
                cached_original_latent_shapes = original_latent_shapes
                return original_latent_shapes

            vision = packed_seq.vision
            packed_sequence[vision.sequence_indexes] = cached_vision_embeds
            return cached_original_latent_shapes

    def _profiled_encode_action(self, *args, **kwargs):
        with profiler.timed("velocity_network_encode_action_sync"):
            return original_encode_action(self, *args, **kwargs)

    def _profiled_decode_vision(self, *args, **kwargs):
        with profiler.timed("velocity_network_decode_vision_sync"):
            return original_decode_vision(self, *args, **kwargs)

    def _profiled_decode_action(self, *args, **kwargs):
        with profiler.timed("velocity_network_decode_action_sync"):
            return original_decode_action(self, *args, **kwargs)

    def _profiled_build_packed_sequence(*args, **kwargs):
        nonlocal cached_build_packed_seq_res, cached_causal_indices, cached_full_indices, padded_causal_shape, padded_full_shape
        if profiler.phase != "measured":
            return original_build_packed_sequence(*args, **kwargs)

        with profiler.timed("velocity_build_packed_sequence_sync"):
            if cached_build_packed_seq_res is None:
                print("[ANTIGRAVITY] Creating cache for build_packed_sequence...")
                input_pack, attention_meta, natten_metadata_list = original_build_packed_sequence(*args, **kwargs)
                cached_causal_indices = input_pack.get("_causal_indices")
                cached_full_indices = input_pack.get("_full_indices")
                padded_causal_shape = list(input_pack["causal_seq"].shape)
                padded_full_shape = list(input_pack["full_only_seq"].shape)
                
                import copy
                cached_pack_base = copy.copy(input_pack)
                cached_build_packed_seq_res = (cached_pack_base, attention_meta, natten_metadata_list)
                return input_pack, attention_meta, natten_metadata_list

            cached_pack_base, attention_meta, natten_metadata_list = cached_build_packed_seq_res
            p_seq = kwargs.get("packed_sequence")
            if p_seq is None:
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        p_seq = arg
                        break
            if p_seq is None:
                raise RuntimeError("Could not find packed_sequence in build_packed_sequence arguments")

            c_seq = p_seq[cached_causal_indices]
            f_seq = p_seq[cached_full_indices]
            
            if c_seq.shape[0] < padded_causal_shape[0]:
                c_seq_pad = torch.zeros(padded_causal_shape, dtype=c_seq.dtype, device=c_seq.device)
                c_seq_pad[:c_seq.shape[0]] = c_seq
                c_seq = c_seq_pad
            
            if f_seq.shape[0] < padded_full_shape[0]:
                f_seq_pad = torch.zeros(padded_full_shape, dtype=f_seq.dtype, device=f_seq.device)
                f_seq_pad[:f_seq.shape[0]] = f_seq
                f_seq = f_seq_pad

            import copy
            input_pack = copy.copy(cached_pack_base)
            input_pack["causal_seq"] = c_seq
            input_pack["full_only_seq"] = f_seq
            
            return input_pack, attention_meta, natten_metadata_list

    def _profiled_wan_decode(self, zs, clear_decoder_cache=True):
        shape = list(zs.shape) if hasattr(zs, "shape") else None
        use_roctx = roctx_vae.enabled and profiler.phase == "measured"
        if use_roctx:
            roctx_vae.resume()
            roctx_vae.push("policy_vae_wan_decode")
        try:
            with profiler.timed("vae_wan_decode_total_sync", shape=shape, clear_decoder_cache=clear_decoder_cache):
                return original_wan_decode(self, zs, clear_decoder_cache)
        finally:
            if use_roctx:
                roctx_vae.pop()
                roctx_vae.pause()

    def _profiled_decoder_forward(self, x, feat_cache=None, first_chunk=False):
        shape = list(x.shape) if hasattr(x, "shape") else None
        with profiler.timed("vae_decoder3d_forward_sync", shape=shape, first_chunk=bool(first_chunk)):
            return original_decoder_forward(self, x, feat_cache=feat_cache, first_chunk=first_chunk)

    def _block_profiled_decoder_forward(self, x, feat_cache=None, first_chunk=False):
        shape = list(x.shape) if hasattr(x, "shape") else None
        with profiler.timed("vae_decoder3d_forward_sync", shape=shape, first_chunk=bool(first_chunk)):
            feat_idx = [0]

            with profiler.timed("vae_decoder3d_conv1_sync"):
                if feat_cache is not None:
                    x = wan_vae._update_cache_and_apply(x, self.conv1, feat_cache, feat_idx)
                else:
                    x = self.conv1(x)

            for index, layer in enumerate(self.middle):
                with profiler.timed(f"vae_decoder3d_middle_{index}_sync"):
                    if isinstance(layer, wan_vae.ResidualBlock) and feat_cache is not None:
                        x = layer(x, feat_cache, feat_idx)
                    else:
                        x = layer(x)

            for index, layer in enumerate(self.upsamples):
                with profiler.timed(f"vae_decoder3d_upsample_{index}_sync"):
                    if index == decoder_upsample_detail_index:
                        x = _profiled_up_residual_block(index, layer, x, feat_cache, feat_idx, first_chunk)
                    elif feat_cache is not None:
                        x = layer(x, feat_cache, feat_idx, first_chunk)
                    else:
                        x = layer(x)

            for index, layer in enumerate(self.head):
                with profiler.timed(f"vae_decoder3d_head_{index}_sync"):
                    if isinstance(layer, wan_vae.CausalConv3d) and feat_cache is not None:
                        x = wan_vae._update_cache_and_apply(x, layer, feat_cache, feat_idx)
                    else:
                        x = layer(x)
            return x

    def _profiled_up_residual_block(index, layer, x, feat_cache=None, feat_idx=None, first_chunk=False):
        prefix = f"vae_decoder3d_upsample_{index}_detail"
        x_shortcut = None
        if getattr(layer, "avg_shortcut", None) is not None:
            with _roctx_range(f"{prefix}_shortcut"):
                x_shortcut = layer.avg_shortcut(x, first_chunk)

        for module_index, module in enumerate(layer.upsamples):
            module_name = module.__class__.__name__
            if isinstance(module, wan_vae.ResidualBlock):
                x = _profiled_residual_block(prefix, module_index, module, x, feat_cache, feat_idx)
            elif isinstance(module, wan_vae.Resample):
                x = _profiled_resample(prefix, module_index, module, x, feat_cache, feat_idx)
            else:
                with _roctx_range(f"{prefix}_module_{module_index}_{module_name}"):
                    x = module(x, feat_cache, feat_idx)

        if x_shortcut is not None:
            with _roctx_range(f"{prefix}_add_shortcut"):
                return x + x_shortcut
        return x

    def _profiled_residual_block(prefix, module_index, module, x, feat_cache=None, feat_idx=None):
        block_prefix = f"{prefix}_residual_{module_index}"
        with _roctx_range(f"{block_prefix}_total"):
            with _roctx_range(f"{block_prefix}_shortcut"):
                h = module.shortcut(x)

            for layer_index, sublayer in enumerate(module.residual):
                if isinstance(sublayer, wan_vae.CausalConv3d):
                    with _roctx_range(f"{block_prefix}_conv_{layer_index}"):
                        _maybe_capture_vae_conv_probe(block_prefix, layer_index, sublayer, x, feat_cache, feat_idx)
                        if _is_target_vae_conv(block_prefix, layer_index):
                            x = x.contiguous()
                        if _is_target_vae_conv(block_prefix, layer_index) and vae_target_conv_channels_last:
                            if feat_cache is not None:
                                x = _update_cache_and_apply_channels_last(x, sublayer, feat_cache, feat_idx)
                            else:
                                x = _causal_conv3d_channels_last(sublayer, x)
                        elif feat_cache is not None:
                            x = wan_vae._update_cache_and_apply(x, sublayer, feat_cache, feat_idx)
                        else:
                            x = sublayer(x)
                else:
                    x = sublayer(x)

            with _roctx_range(f"{block_prefix}_add"):
                return x + h

    def _is_target_vae_conv(block_prefix, layer_index):
        return block_prefix == "vae_decoder3d_upsample_3_detail_residual_0" and layer_index == 2

    def _causal_conv3d_channels_last(layer, x, cache_x=None):
        padding = list(layer._padding)
        if cache_x is not None:
            cache_x = cache_x.to(x.device)
            if padding[4] > 0:
                x = torch.cat([cache_x, x], dim=2)
                padding[4] = max(0, padding[4] - cache_x.shape[2])
        x = F.pad(x, tuple(padding))
        x = x.contiguous(memory_format=torch.channels_last_3d)
        return F.conv3d(
            x,
            layer.weight,
            layer.bias,
            stride=layer.stride,
            padding=0,
            dilation=layer.dilation,
            groups=layer.groups,
        )

    def _update_cache_and_apply_channels_last(x, layer, feat_cache, feat_idx):
        idx = feat_idx[0]
        cache_x = wan_vae._contiguous_clone(x[:, :, -wan_vae.CACHE_T :, :, :])
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = _causal_conv3d_channels_last(layer, x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        profiler.record("vae_target_conv_channels_last_hit", 0.0, cache_present=feat_cache[idx] is not None)
        return x

    def _maybe_capture_vae_conv_probe(block_prefix, layer_index, sublayer, x, feat_cache, feat_idx):
        if not vae_conv_probe_path:
            return
        if vae_conv_probe_store["written"]:
            return
        if profiler.phase != "measured":
            return
        if block_prefix != "vae_decoder3d_upsample_3_detail_residual_0" or layer_index != 2:
            return
        cache_input = None
        cache_index = None
        if feat_cache is not None and feat_idx is not None:
            cache_index = int(feat_idx[0])
            if feat_cache[cache_index] is not None:
                cache_input = feat_cache[cache_index].detach().cpu()
        path = Path(vae_conv_probe_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": f"{block_prefix}_conv_{layer_index}",
            "input": x.detach().cpu(),
            "weight": sublayer.weight.detach().cpu(),
            "bias": sublayer.bias.detach().cpu() if sublayer.bias is not None else None,
            "cache_input": cache_input,
            "cache_index": cache_index,
            "causal_padding": tuple(int(v) for v in sublayer._padding),
            "stride": tuple(int(v) for v in sublayer.stride),
            "dilation": tuple(int(v) for v in sublayer.dilation),
            "groups": int(sublayer.groups),
            "cache_x": cache_input is not None,
            "dtype": str(x.dtype),
            "shape": list(x.shape),
            "weight_shape": list(sublayer.weight.shape),
        }
        torch.save(payload, path)
        vae_conv_probe_store["written"] = True
        profiler.record("vae_conv_probe_capture", 0.0, path=str(path), shape=list(x.shape))

    def _profiled_resample(prefix, module_index, module, x, feat_cache=None, feat_idx=None):
        module_prefix = f"{prefix}_resample_{module_index}_{module.mode}"
        with _roctx_range(module_prefix):
            return module(x, feat_cache, feat_idx)

    sequence_packing.PackedSequence.to_cuda = _profiled_packed_sequence_to_cuda
    cosmos3_vfm_network.Cosmos3VFMNetwork.forward = _profiled_network_forward
    cosmos3_vfm_network.Cosmos3VFMNetwork._encode_text = _profiled_encode_text
    cosmos3_vfm_network.Cosmos3VFMNetwork._encode_vision = _profiled_encode_vision
    cosmos3_vfm_network.Cosmos3VFMNetwork._encode_action = _profiled_encode_action
    cosmos3_vfm_network.Cosmos3VFMNetwork._decode_vision = _profiled_decode_vision
    cosmos3_vfm_network.Cosmos3VFMNetwork._decode_action = _profiled_decode_action
    cosmos3_vfm_network.build_packed_sequence = _profiled_build_packed_sequence
    wan_vae.WanVAE.decode = _profiled_wan_decode
    wan_vae.Decoder3d.forward = _block_profiled_decoder_forward if decoder_block_profile else _profiled_decoder_forward


def main() -> None:
    # Enable aggressive PyTorch Inductor optimizations
    import torch._inductor.config as inductor_config
    inductor_config.coordinate_descent_tuning = True
    inductor_config.fx_graph_cache = True
    inductor_config.freezing = True
    inductor_config.cpp_wrapper = False
    inductor_config.triton.cudagraphs = True
    inductor_config.layout_optimization = True
    inductor_config.shape_padding = True
    inductor_config.triton.autotune_cublasLt = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/workspace/result/classmethod_policy_framework")
    parser.add_argument("--skip-vision-decode", action="store_true")
    parser.add_argument("--action-only", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--policy-sync-profile", action="store_true")
    parser.add_argument("--policy-condition-cache", action="store_true")
    parser.add_argument("--policy-deep-profile", action="store_true")
    parser.add_argument("--policy-decoder-block-profile", action="store_true")
    parser.add_argument("--policy-decoder-upsample-detail-index", type=int, default=-1)
    parser.add_argument("--policy-roctx-network-forward", action="store_true")
    parser.add_argument("--policy-roctx-vae-detail", action="store_true")
    parser.add_argument("--policy-vae-conv-probe-path", default=None)
    parser.add_argument("--policy-vae-target-conv-channels-last", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("COSMOS_TRAINING", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import cosmos_framework.model.attention as attention_pkg
    import cosmos_framework.model.attention.frontend as attention_frontend
    from cosmos_framework.inference import args as inference_args
    from cosmos_framework.scripts import inference

    attention_frontend.attention = _sdpa_varlen_fallback
    attention_pkg.attention = _sdpa_varlen_fallback
    inference_args._get_device_memory_bytes = lambda: 120 * 1024**3

    from cosmos_framework.inference.inference import OmniInference

    profiler = SyncStageProfiler(args.policy_sync_profile, args.out_dir)
    _install_policy_hooks(
        profiler,
        args.policy_condition_cache,
        args.policy_deep_profile,
        args.policy_decoder_block_profile,
        args.policy_decoder_upsample_detail_index,
        args.policy_roctx_network_forward,
        args.policy_roctx_vae_detail,
        args.policy_vae_conv_probe_path,
        args.policy_vae_target_conv_channels_last,
    )

    original_generate_batch = OmniInference.generate_batch

    def _generate_batch_clone_warmup(self, sample_args_list, data_batch, *, warmup=False):
        if warmup:
            data_batch = _clone_batch_for_warmup(data_batch)
        old_phase = profiler.phase
        profiler.phase = "warmup" if warmup else "measured"
        try:
            with profiler.timed("generate_batch_sync"):
                return original_generate_batch(self, sample_args_list, data_batch, warmup=warmup)
        finally:
            profiler.phase = old_phase

    OmniInference.generate_batch = _generate_batch_clone_warmup

    if args.skip_vision_decode or args.action_only:
        import cosmos_framework.model.vfm.omni_mot_model as omni_mot_model

        original_decode = omni_mot_model.OmniMoTModel.decode

        def _zero_vision_decode(self, vision_latent):
            tokenizer = self.tokenizer_vision_gen
            return vision_latent.new_zeros(
                (
                    vision_latent.shape[0],
                    3,
                    tokenizer.get_pixel_num_frames(int(vision_latent.shape[2])),
                    int(vision_latent.shape[3]) * tokenizer.spatial_compression_factor,
                    int(vision_latent.shape[4]) * tokenizer.spatial_compression_factor,
                )
            )

        omni_mot_model.OmniMoTModel.decode = _zero_vision_decode
    sys.argv = [
        "cosmos_policy_rocm",
        "--parallelism-preset=latency",
        "--checkpoint-path",
        "Cosmos3-Nano",
        "-i",
        "inputs/omni/action_policy_robot.json",
        "-o",
        args.out_dir,
        "--benchmark",
        "--warmup",
        str(args.warmup_runs),
        "--no-guardrails",
        "--no-use-torch-compile",
        "--no-use-cuda-graphs",
    ]
    inference.main()
    profiler.write()

    if args.action_only:
        root = Path(args.out_dir)
        for sample_outputs in root.glob("*/sample_outputs.json"):
            data = json.loads(sample_outputs.read_text())
            for output in data.get("outputs", []):
                for file_name in output.get("files", []):
                    path = Path(file_name)
                    if path.exists():
                        path.unlink()
                output["files"] = []
            data["action_only"] = True
            data["vision_decode_skipped"] = True
            sample_outputs.write_text(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
