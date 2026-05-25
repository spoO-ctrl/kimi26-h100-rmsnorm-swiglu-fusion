"""
Monkey-patch GPT-OSS MoE layers to use optimized grouped GEMM forward pass.

Replaces the naive Python expert loop with:
  - Token sorting by expert assignment
  - Triton grouped GEMM for gate_up and down projections
  - Fused Triton gating kernel
  - Weighted scatter-add combine

Can be applied on top of the existing QKV fusion from patch_gpt_oss.py.
"""

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

from src.moe_grouped_gemm import optimized_moe_forward


def patch_gpt_oss_moe(model, use_triton_gemm=True):
    """Patch all MoE layers in a GPT-OSS model to use grouped GEMM.

    Args:
        model: HuggingFace GptOssForCausalLM model
        use_triton_gemm: if True, use Triton grouped GEMM; else padded batched

    Returns:
        The patched model (modified in-place)
    """
    for layer_idx, layer in enumerate(model.model.layers):
        _patch_moe_layer(layer, use_triton_gemm)
    return model


def _patch_moe_layer(layer: GptOssDecoderLayer, use_triton_gemm: bool):
    """Patch a single GPT-OSS decoder layer's MoE MLP."""
    experts = layer.mlp.experts
    alpha = experts.alpha
    limit = experts.limit

    gate_up_proj = experts.gate_up_proj       # [num_experts, hidden, 2*intermediate]
    gate_up_proj_bias = experts.gate_up_proj_bias  # [num_experts, 2*intermediate]
    down_proj = experts.down_proj             # [num_experts, intermediate, hidden]
    down_proj_bias = experts.down_proj_bias   # [num_experts, hidden]

    def patched_experts_forward(self, hidden_states, router_indices=None, routing_weights=None):
        return optimized_moe_forward(
            hidden_states, router_indices, routing_weights,
            gate_up_proj, gate_up_proj_bias,
            down_proj, down_proj_bias,
            alpha=alpha, limit=limit,
            use_triton_gemm=use_triton_gemm,
        )

    # Bind the patched forward
    import types
    experts.forward = types.MethodType(patched_experts_forward, experts)
