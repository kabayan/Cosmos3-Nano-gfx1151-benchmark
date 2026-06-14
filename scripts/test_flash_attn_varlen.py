import torch
import torch.nn.functional as F

def _sdpa_varlen_fallback(
    query,
    key,
    value,
    is_causal=False,
    scale=None,
    cumulative_seqlen_Q=None,
    cumulative_seqlen_KV=None,
):
    if cumulative_seqlen_Q is None:
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=scale, enable_gqa=q.shape[1] != k.shape[1]
        )
        return out.permute(0, 2, 1, 3)

    q_offsets = cumulative_seqlen_Q.detach().cpu().tolist()
    k_offsets = cumulative_seqlen_KV.detach().cpu().tolist()
    chunks = []
    for i in range(len(q_offsets) - 1):
        qs, qe = q_offsets[i], q_offsets[i + 1]
        ks, ke = k_offsets[i], k_offsets[i + 1]
        q = query[:, qs:qe].permute(0, 2, 1, 3)
        k = key[:, ks:ke].permute(0, 2, 1, 3)
        v = value[:, ks:ke].permute(0, 2, 1, 3)
        use_causal = bool(is_causal and q.shape[-2] == k.shape[-2])
        attn_mask = None
        if is_causal and not use_causal:
            q_len, k_len = q.shape[-2], k.shape[-2]
            attn_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=use_causal,
            scale=scale,
            enable_gqa=q.shape[1] != k.shape[1],
        )
        chunks.append(out.permute(0, 2, 1, 3))
    return torch.cat(chunks, dim=1)

def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # Simulate batch of variable length sequences packed into 1 batch dimension.
    # Say we have 3 sequences of lengths: Q: [100, 200, 150], KV: [150, 250, 200]
    seqlens_q = [100, 200, 150]
    seqlens_kv = [150, 250, 200]
    
    cum_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens_q), 0)), dtype=torch.int32, device=device)
    cum_kv = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens_kv), 0)), dtype=torch.int32, device=device)
    
    total_q = sum(seqlens_q)
    total_kv = sum(seqlens_kv)
    
    heads_q = 32
    heads_kv = 8
    head_dim = 128
    
    # Query, Key, Value shapes in cosmos3 are:
    # [1, total_len, num_heads, head_dim]
    q = torch.randn((1, total_q, heads_q, head_dim), dtype=dtype, device=device)
    k = torch.randn((1, total_kv, heads_kv, head_dim), dtype=dtype, device=device)
    v = torch.randn((1, total_kv, heads_kv, head_dim), dtype=dtype, device=device)
    
    ref = _sdpa_varlen_fallback(
        q, k, v, is_causal=False, scale=1.0,
        cumulative_seqlen_Q=cum_q, cumulative_seqlen_KV=cum_kv
    )
    
    print("Reference output shape:", ref.shape)
    
    # Try using torch.ops.aten._flash_attention_forward
    # Let's see what input shapes it expects.
    # Usually it expects 3D tensors: [total_len, heads, head_dim]
    # Let's try 3D tensors.
    q_3d = q.squeeze(0)
    k_3d = k.squeeze(0)
    v_3d = v.squeeze(0)
    
    max_q = max(seqlens_q)
    max_kv = max(seqlens_kv)
    
    # API: _flash_attention_forward(Tensor query, Tensor key, Tensor value, Tensor? cum_seq_q, Tensor? cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, bool return_debug_mask, *, float? scale=None, ...)
    # returns: (output, softmax_logsumexp, rng_state, unused, debug_attn_mask)
    try:
        out_3d, _, _, _, _ = torch.ops.aten._flash_attention_forward(
            q_3d, k_3d, v_3d,
            cum_q, cum_kv,
            max_q, max_kv,
            0.0, # dropout_p
            False, # is_causal
            False, # return_debug_mask
            scale=1.0
        )
        print("3D flash attention varlen output shape:", out_3d.shape)
        # Compare
        diff = (ref.squeeze(0) - out_3d).abs().max().item()
        print("Diff with 3D:", diff)
    except Exception as e:
        print("Error with 3D:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
