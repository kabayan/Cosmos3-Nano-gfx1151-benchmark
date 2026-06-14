# Cosmos3 ROCm v2.3 Policy VAE decode and velocity speedup plan

Date: 2026-06-05

## Current measured state

v2.3 Policy condition cache result:

| Stage | Time | Share |
|---|---:|---:|
| Full video + action measured run | 147.757 sec | 100% |
| VAE decode | 96.080 sec | 65.0% |
| `_get_velocity` / denoise sampling | 51.429 sec | 34.8% |
| save/postprocess | 0.204 sec | 0.1% |

The output was validated as identical to the no-cache Policy output:

- MP4 SHA256: `05cb83e594b65b34675e2ffbcec3d7807790d12d833d4d82a891c01688356cb4`
- action/output JSON max abs diff: `0.0`

Quality-reducing changes are out of scope. Therefore the plan does not reduce `num_steps`, output frames, resolution, dtype, or output modalities.

## Source check

### VAE decode path

Files:

- `/tmp/cosmos-framework/cosmos_framework/model/vfm/omni_mot_model.py`
- `/tmp/cosmos-framework/cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py`
- `/tmp/cosmos-framework/cosmos_framework/inference/inference.py`

Relevant code:

- `OmniMoTModel.decode()` only delegates to `self.tokenizer_vision_gen.decode(latent)`.
- `Wan2pt2VAEInterface.decode()` calls `WanVAE.decode(latent, clear_decoder_cache=...)`.
- `WanVAE.decode()` denormalizes latent, applies `conv2`, then loops over latent temporal dimension:

```python
parts = []
for i in range(x.shape[2]):
    first_chunk = (i == 0) and all(c is None for c in self._dec_cache)
    parts.append(
        self.decoder(
            x[:, :, i : i + 1],
            feat_cache=self._dec_cache,
            first_chunk=first_chunk,
        )
    )
decoded = unpatchify(torch.cat(parts, dim=2), patch_size=2)
```

The decoder uses `CausalConv3d`, `ResidualBlock`, `Up_ResidualBlock`, `Resample`, and `AttentionBlock`. `CausalConv3d.forward()` performs temporal padding and then calls `nn.Conv3d`.

Implication:

- Current decode is streaming-style, one latent frame at a time.
- Kernel launch count is likely high.
- ROCm/MIOpen conv solver selection remains a likely bottleneck.
- There is an existing decoder cache mechanism: `keep_decoder_cache` / `use_cached_decoder()`.

### `_get_velocity` path

Files:

- `/tmp/cosmos-framework/cosmos_framework/model/vfm/omni_mot_model.py`
- `/tmp/cosmos-framework/cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py`
- `/tmp/cosmos-framework/cosmos_framework/data/vfm/sequence_packing.py`

Relevant code:

- `_get_velocity()` is called once per sampler step.
- Policy defaults use `num_steps=30`, `guidance=1.0`, so CFG does not run a second unconditional forward.
- Each `_get_velocity()` call:
  1. splits flattened `noise_x` into vision/action tensors,
  2. builds a fresh `GenerationDataClean`,
  3. calls `_pack_input_sequence(...)`,
  4. moves the packed sequence to CUDA,
  5. calls `self.denoise(...)`,
  6. applies velocity masks and concatenates flattened outputs.
- `denoise()` delegates to `Cosmos3VFMNetwork.forward()`.
- `Cosmos3VFMNetwork.forward()` runs:
  - `_encode_text`
  - `_encode_vision`
  - `_encode_action`
  - `_encode_sound` if enabled
  - `build_packed_sequence`
  - `language_model(...)`
  - `_decode_vision`
  - `_decode_action`

Implication:

- The 51.429 sec is not CFG duplication.
- It is 30 single denoise steps, roughly 1.71 sec/step.
- There may be removable overhead in repeated packing and metadata construction, but the main cost is probably transformer forward unless measured otherwise.

## Improvement plan

## Phase 0: measurement split before code changes

Goal: prove which sub-paths actually dominate before optimizing.

Add profiling hooks for:

### VAE decode

- `WanVAE.decode`
- `conv2`
- each decoder temporal iteration
- `Decoder3d.conv1`
- `Decoder3d.middle`
- each `Up_ResidualBlock`
- `Decoder3d.head`
- `unpatchify`

Expected output:

- per-block seconds
- per-block call count
- latent shape and output shape

Decision criteria:

- If most time is inside `CausalConv3d` / `nn.Conv3d`, prioritize MIOpen solver/shape changes.
- If time is spread across many small ops, prioritize compile/fusion or temporal batching.
- If first few temporal chunks are much slower, prioritize decoder cache/warmup/FindDb.

### `_get_velocity`

Add profiling hooks for:

- split/reshape noise
- `_pack_input_sequence`
- `PackedSequence.to_cuda`
- `denoise`
- `Cosmos3VFMNetwork._encode_text`
- `_encode_vision`
- `_encode_action`
- `build_packed_sequence`
- `language_model`
- `_decode_vision`
- `_decode_action`
- mask/concat output

Decision criteria:

- If `language_model` dominates, use rocprof and kernel/backend work.
- If `_pack_input_sequence` or `build_packed_sequence` is significant, implement static packed metadata cache.
- If `_encode_vision` / `_encode_action` is significant, cache fixed clean-token projections and update only noisy token/time embeddings.

## Phase 1: VAE decode low-risk probes

### Probe A: decoder cache behavior

Test:

- default `clear_decoder_cache=True`
- `tokenizer_vision_gen.use_cached_decoder()` around warmup + measured decode
- `keep_decoder_cache=True` config override

Validate:

- MP4 SHA256 identical
- action JSON unchanged
- measured decode time improves

Risk:

- Low for same-shape sequential decode, but cache leakage across unrelated samples must be prevented.

Expected benefit:

- Unknown. It may primarily affect cold/warm behavior, not single measured decode.

### Probe B: one-shot non-streaming decode

Prototype:

- Add an optional path to call `self.decoder(x, feat_cache=None)` once for full latent T instead of looping over `T`.

Validate:

- Compare decoded tensor max abs diff before MP4 save.
- Compare MP4 SHA256.
- If not bit-identical, do not promote for article comparison.

Risk:

- Medium. Causal padding semantics may differ from streaming cached decode.

Expected benefit:

- Potentially large if launch overhead dominates.
- Potentially zero or invalid if cache semantics are required for exact output.

### Probe C: temporal micro-batching

Prototype:

- Decode chunks of latent T larger than 1 while preserving left cache.
- Try chunk sizes `1`, `2`, `3`, `full`.

Validate:

- Tensor max abs diff and MP4 SHA.
- Per-chunk time.

Risk:

- Medium. `Resample` has special first-chunk logic and cache shape assumptions.

Expected benefit:

- Could reduce kernel launch overhead while preserving causal semantics.

### Probe D: decode `torch.compile`

Prototype:

- Compile stable submodules or a wrapper around the measured decode shape.
- Start with `Decoder3d.forward` or full `WanVAE.decode`.

Validate:

- compilation time excluded from measured run
- output equality
- measured decode speed

Risk:

- Medium/high on ROCm due graph breaks, dynamic cache lists, and Conv3d solver behavior.

Expected benefit:

- Better if overhead is Python/launch/pointwise; limited if MIOpen conv dominates.

## Phase 2: VAE decode ROCm/MIOpen path

If Phase 0 shows Conv3d dominates:

1. Run measured-only rocprof for VAE decode.
2. Extract conv descriptors:
   - input shape
   - weight shape
   - dtype
   - stride/padding/dilation
   - selected solver/kernel name
   - FindDb entry
3. Identify descriptors falling to `naive_conv`.
4. Try shape-preserving code changes:
   - contiguous layout normalization before conv
   - chunk size changes
   - avoiding pathological `T=1` Conv3d descriptors where possible
   - replacing `(3,1,1)` temporal Conv3d with equivalent Conv1d/2D reshape only if output is identical
5. If repo-side changes cannot change solver choice, move to MIOpen full descriptor support or tuned solver enablement.

Promotion criteria:

- exact output match
- decode speed improvement
- no regression in warmup-excluded full Policy run

## Phase 3: `_get_velocity` metadata/cache optimization

### Candidate A: static packed metadata cache

Observation:

- `sequence_plans`, text tokens, condition masks, position IDs, split lengths, attention modes, token shapes, and many sequence indexes are invariant across sampler steps.
- Only noisy vision/action tensors and timestep embeddings change.

Prototype:

- Build packed sequence template once after `_prepare_inference_data`.
- Per step, clone or shallow-copy only mutable token fields.
- Reuse:
  - `sample_lens`
  - `split_lens`
  - `attn_modes`
  - `position_ids`
  - sequence indexes
  - condition masks
  - token shapes
  - domain IDs

Validation:

- action/output JSON max abs diff `0.0`
- MP4 SHA identical
- measured `_get_velocity` time improves

Risk:

- Medium. The packed object contains tensors and lists; mutation by downstream code must be audited.

Expected benefit:

- Good only if pack/build overhead is non-trivial. Phase 0 decides.

### Candidate B: fixed text embedding cache

Observation:

- `_encode_text()` embeds the same text tokens every step.

Prototype:

- Cache text embedding and text scatter destination for the measured sample.
- Reuse the prefilled text region in `packed_sequence`.

Validation:

- exact output match
- per-step `_encode_text` timing reduction

Risk:

- Low/medium. Must preserve dtype/device and avoid mutation.

Expected benefit:

- Small unless `_encode_text` is significant.

### Candidate C: fixed position/attention metadata cache

Observation:

- `build_packed_sequence(...)` likely rebuilds attention metadata every step from invariant lengths and token shapes.

Prototype:

- Cache `attention_meta`, `natten_metadata_list`, and context-parallel sharding metadata when memory is `None` and shapes are fixed.
- Reuse for every timestep.

Validation:

- exact output match
- profile shows `build_packed_sequence` reduced

Risk:

- Medium/high. Attention metadata can be backend-specific and may include device tensors.

Expected benefit:

- Potentially meaningful if metadata construction appears in Phase 0.

### Candidate D: timestep embedding update only

Observation:

- For each step, the model must update noisy token values and timestep embeddings, but clean conditioned token embeddings are invariant.

Prototype:

- Precompute clean/condition token embeddings.
- Per step, compute only noisy token projections and timestep embeddings, then scatter into template.

Validation:

- exact output match
- `_encode_vision` / `_encode_action` reduction

Risk:

- Medium. Need careful separation of clean vs noisy token indexes.

Expected benefit:

- Potentially useful if `_encode_vision` or `_encode_action` contributes significantly.

## Phase 4: `_get_velocity` transformer kernel/backend optimization

If `language_model` dominates:

1. Run measured-only rocprof on cached Policy `_get_velocity`.
2. Split kernels into:
   - GEMM / hipBLASLt / rocBLAS / Tensile
   - SDPA/AOTriton attention
   - RMSNorm
   - RoPE/mRoPE
   - MLP/activation
   - scatter/gather/indexing
3. Compare backend toggles:
   - AOTriton enabled/disabled
   - TunableOp table on/off
   - Stream-K on/off only if SDPA failure remains fixed
4. Test `torch.compile` on limited blocks:
   - RMSNorm + RoPE + projection wrappers
   - avoid compiling the full model first
5. Only promote if output is identical.

Expected benefit:

- Medium. T2I/I2V results suggest ROCm transformer forward is already a key gap versus CUDA/Blackwell.

## Recommended execution order

1. Implement Phase 0 profiling hooks.
2. Run one condition-cache video + action measured benchmark with profiling.
3. Run VAE decode probes A/B/C on the captured latent, not the full benchmark.
4. Implement the lowest-risk validated VAE change.
5. Profile `_get_velocity` sub-stages.
6. Implement static packed metadata cache only if measured overhead justifies it.
7. Run full Policy condition-cache benchmark and validate exact output.
8. Update WebUI and docs.

## Expected outcome range

Conservative target:

- VAE decode: `96.080 sec` -> `70-80 sec`
- `_get_velocity`: `51.429 sec` -> `45-50 sec`
- Full Policy: `147.757 sec` -> `115-130 sec`

Aggressive target:

- VAE decode: `96.080 sec` -> `40-60 sec`
- `_get_velocity`: `51.429 sec` -> `35-45 sec`
- Full Policy: `147.757 sec` -> `75-105 sec`

The aggressive target likely requires either a successful decode chunking/compile path or ROCm/MIOpen Conv3d solver improvement.
