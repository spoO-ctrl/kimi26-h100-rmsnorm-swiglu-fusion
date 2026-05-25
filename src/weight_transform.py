"""
Pre-compute fused LayerNorm+Linear weights.

For LayerNorm(gamma, beta) followed by Linear(W, b):
    output = Linear(LayerNorm(x))
           = x @ (I - E/h) @ diag(gamma) @ W.T / std(x) + beta @ W.T + b

where std(x) = sqrt(v(x)^2/h + eps), v(x) = ||x - mean(x)||_2.

The forward pass computes: x @ W_new.T / std(x) + b_new
where std(x) is derived from v(x) returned by the CUDA kernel.

Notation: E = 11^T is the h x h all-ones matrix (outer product of the all-ones
vector 1 with itself). Thus (I - E/h) is the centering matrix that subtracts
the mean from each column.

Assumptions: LayerNorm has both weight (gamma) and bias (beta);
normalized_shape is 1-dimensional.

So:
    M = (W * gamma).T            # element-wise (W * gamma), then .T = diag(gamma) @ W.T, shape [h, out]
    F_new = M - M.mean(dim=0)    # = (I - E/h) @ M, center each column, shape [h, out]
    W_new = F_new.T              # shape [out, h]
    b_new = beta @ W.T + b       # matrix multiply beta with W.T + bias, shape [out]
"""

import torch
import torch.nn as nn


def compute_fused_weights(
    ln: nn.LayerNorm,
    linear: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """
    Compute fused weights W_new, b_new for LayerNorm + Linear fusion.

    Args:
        ln: LayerNorm module with weight (gamma) and bias (beta)
        linear: Linear module with weight W [out, h] and optional bias b [out]

    Returns:
        W_new: [out, h] fused weight
        b_new: [out] fused bias
        h: hidden dimension (for std computation)
        eps: LayerNorm epsilon
    """
    gamma = ln.weight.data       # [h]
    beta = ln.bias.data          # [h]
    W = linear.weight.data       # [out, h]
    b = linear.bias.data if linear.bias is not None else torch.zeros(
        W.size(0), device=W.device, dtype=W.dtype
    )
    h = ln.normalized_shape[0]
    eps = ln.eps

    # M = diag(gamma) @ W.T = (W * gamma).T, shape [h, out]
    M = (W * gamma).T

    # (I - E/h) @ M: center each column (subtract column mean)
    F_new = M - M.mean(dim=0, keepdim=True)

    W_new = F_new.T              # [out, h]
    b_new = beta @ W.T + b      # [out]: beta @ W.T is like linear(beta)

    return W_new, b_new, h, eps


def compute_fused_weights_rmsnorm(
    rms_norm,
    linear: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """
    Compute fused weights W_new, b_new for RMSNorm + Linear fusion.

    RMSNorm: output = x / rms(x) * gamma, where rms(x) = sqrt(mean(x^2) + eps).
    Unlike LayerNorm, there is no mean subtraction and typically no bias.

    The fused forward computes: x @ W_new.T / rms(x) + b_new
    where:
        W_new = W * gamma   (element-wise, no centering needed)
        b_new = b           (no beta term since RMSNorm has no bias)

    Args:
        rms_norm: RMSNorm module with weight (gamma), no bias
        linear: Linear module with weight W [out, h] and optional bias b [out]

    Returns:
        W_new: [out, h] fused weight
        b_new: [out] fused bias
        h: hidden dimension
        eps: RMSNorm epsilon
    """
    gamma = rms_norm.weight.data    # [h]
    W = linear.weight.data          # [out, h]
    b = linear.bias.data if linear.bias is not None else torch.zeros(
        W.size(0), device=W.device, dtype=W.dtype
    )
    h = gamma.shape[0]
    eps = rms_norm.eps if hasattr(rms_norm, 'eps') else rms_norm.variance_epsilon

    # W_new = W * gamma (element-wise broadcast), shape [out, h]
    # No centering needed since RMSNorm doesn't subtract mean
    W_new = W * gamma

    # b_new = b (no beta in RMSNorm)
    b_new = b

    return W_new, b_new, h, eps


def transform_opt_layer(decoder_layer) -> dict:
    """
    Compute fused weights for all LayerNorm+Linear pairs in an OPT decoder layer.

    Pairs:
        1. self_attn_layer_norm -> q_proj, k_proj, v_proj
        2. final_layer_norm -> fc1

    Returns:
        dict mapping projection name to (W_new, b_new)
    """
    attn = decoder_layer.self_attn
    ln1 = decoder_layer.self_attn_layer_norm
    ln2 = decoder_layer.final_layer_norm

    fused = {}

    # Attention projections share the same layer norm
    for name in ["q_proj", "k_proj", "v_proj"]:
        proj = getattr(attn, name)
        W_new, b_new, h, eps = compute_fused_weights(ln1, proj)
        fused[f"attn_{name}"] = (W_new, b_new, h, eps)

    # FFN fc1
    W_new, b_new, h, eps = compute_fused_weights(ln2, decoder_layer.fc1)
    fused["fc1"] = (W_new, b_new, h, eps)

    return fused


def compute_fused_weights_rmsnorm_combined(
    rms_norm,
    linears: list[nn.Linear],
) -> tuple[torch.Tensor, torch.Tensor, list[int], int, float]:
    """
    Compute fused weights for RMSNorm + multiple Linear layers that share the same norm.

    Concatenates the weight matrices along dim=0 so a single matmul replaces
    multiple separate projections (e.g. Q/K/V or gate/up).

    Args:
        rms_norm: RMSNorm module with weight (gamma)
        linears: list of Linear modules sharing this norm

    Returns:
        W_combined: [sum(out_dims), h] fused weight
        b_combined: [sum(out_dims)] fused bias
        split_sizes: list of output dims per linear, for torch.split
        h: hidden dimension
        eps: RMSNorm epsilon
    """
    gamma = rms_norm.weight.data    # [h]
    h = gamma.shape[0]
    eps = rms_norm.eps if hasattr(rms_norm, 'eps') else rms_norm.variance_epsilon

    W_parts = []
    b_parts = []
    split_sizes = []

    for linear in linears:
        W = linear.weight.data          # [out_i, h]
        b = linear.bias.data if linear.bias is not None else torch.zeros(
            W.size(0), device=W.device, dtype=W.dtype
        )
        W_parts.append(W * gamma)       # element-wise, no centering
        b_parts.append(b)
        split_sizes.append(W.size(0))

    W_combined = torch.cat(W_parts, dim=0)  # [sum(out_dims), h]
    b_combined = torch.cat(b_parts, dim=0)  # [sum(out_dims)]

    return W_combined, b_combined, split_sizes, h, eps


def transform_llama_layer_combined(decoder_layer) -> dict:
    """
    Compute fused combined weights for a Llama decoder layer.

    Groups projections that share the same RMSNorm:
        1. input_layernorm -> [q_proj, k_proj, v_proj]  (attention QKV)
        2. post_attention_layernorm -> [gate_proj, up_proj]  (MLP gate+up)

    Returns:
        dict with keys "attn_qkv" and "mlp_gate_up", each mapping to
        (W_combined, b_combined, split_sizes, h, eps)
    """
    attn = decoder_layer.self_attn
    ln1 = decoder_layer.input_layernorm
    ln2 = decoder_layer.post_attention_layernorm

    fused = {}

    # QKV combined
    qkv_linears = [attn.q_proj, attn.k_proj, attn.v_proj]
    fused["attn_qkv"] = compute_fused_weights_rmsnorm_combined(ln1, qkv_linears)

    # Gate + Up combined
    gate_up_linears = [decoder_layer.mlp.gate_proj, decoder_layer.mlp.up_proj]
    fused["mlp_gate_up"] = compute_fused_weights_rmsnorm_combined(ln2, gate_up_linears)

    return fused


def transform_gpt_oss_layer(decoder_layer) -> dict:
    """
    Compute fused combined weights for a GPT-OSS decoder layer.

    Only fuses attention QKV (input_layernorm -> [q_proj, k_proj, v_proj]).
    The MoE MLP is NOT fused (router dispatches between norm and experts).

    Returns:
        dict with key "attn_qkv" mapping to
        (W_combined, b_combined, split_sizes, h, eps)
    """
    attn = decoder_layer.self_attn
    norm = decoder_layer.input_layernorm

    qkv_linears = [attn.q_proj, attn.k_proj, attn.v_proj]
    return {"attn_qkv": compute_fused_weights_rmsnorm_combined(norm, qkv_linears)}


def transform_llama_layer(decoder_layer) -> dict:
    """
    Compute fused weights for all RMSNorm+Linear pairs in a Llama decoder layer.

    Pairs:
        1. input_layernorm -> q_proj, k_proj, v_proj (attention)
        2. post_attention_layernorm -> gate_proj, up_proj (MLP)

    Note: down_proj is NOT fused (it follows an activation, not a norm).

    Returns:
        dict mapping projection name to (W_new, b_new, h, eps)
    """
    attn = decoder_layer.self_attn
    ln1 = decoder_layer.input_layernorm
    ln2 = decoder_layer.post_attention_layernorm

    fused = {}

    # Attention projections share input_layernorm
    for name in ["q_proj", "k_proj", "v_proj"]:
        proj = getattr(attn, name)
        W_new, b_new, h, eps = compute_fused_weights_rmsnorm(ln1, proj)
        fused[f"attn_{name}"] = (W_new, b_new, h, eps)

    # MLP gate_proj and up_proj share post_attention_layernorm
    for name in ["gate_proj", "up_proj"]:
        proj = getattr(decoder_layer.mlp, name)
        W_new, b_new, h, eps = compute_fused_weights_rmsnorm(ln2, proj)
        fused[name] = (W_new, b_new, h, eps)

    return fused
