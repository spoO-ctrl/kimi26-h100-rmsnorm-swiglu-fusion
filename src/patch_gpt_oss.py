"""
Monkey-patch a GPT-OSS model to use fused RMSNorm+Linear modules.

Replaces the forward pass of each GPT-OSS decoder layer so that:
  - input_layernorm + q/k/v_proj -> fused_qkv (combined)

Only QKV attention is fused. The MoE MLP is left untouched because
the router dispatches tokens between norm output and expert execution,
making norm-into-expert fusion impractical.

Note: GPT-OSS uses chunk-based RoPE (not interleaved like Llama),
attention has sinks and sliding_window, and projections have bias.
"""

import torch
import torch.nn as nn
from typing import Callable
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssDecoderLayer, GptOssAttention,
    ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
    apply_rotary_pos_emb,
)

from src.weight_transform import transform_gpt_oss_layer
from src.fused_forward import (
    FusedRMSNormCombinedLinearV1, FusedRMSNormCombinedLinearV3,
)
from src.gqa_attention_forward import GQADecodeAttentionV3

_COMBINED_VARIANT_CLASSES = {
    "V1": FusedRMSNormCombinedLinearV1,
    "V3": FusedRMSNormCombinedLinearV3,
}


def patch_gpt_oss_model(model, device=None, variant="V1", gqa_decode=False):
    """
    Patch all decoder layers in a GPT-OSS model to use fused RMSNorm+Linear.

    Only QKV attention is fused (combined mode). MoE MLP is untouched.

    Args:
        model: HuggingFace GptOssForCausalLM model
        device: target device (defaults to model's device)
        variant: kernel variant -- "V1" (256 threads) or "V3" (512 threads)
        gqa_decode: if True, use GQA-optimized V3 kernel for single-token decode steps

    Returns:
        The patched model (modified in-place)
    """
    if variant not in _COMBINED_VARIANT_CLASSES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(_COMBINED_VARIANT_CLASSES)}")

    if device is None:
        device = next(model.parameters()).device

    config = model.config
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_q_heads

    for layer_idx, layer in enumerate(model.model.layers):
        _patch_decoder_layer(layer, device, variant)

        if gqa_decode and num_kv_heads < num_q_heads:
            layer.self_attn.gqa_decode_module = GQADecodeAttentionV3(
                num_q_heads, num_kv_heads, head_dim
            )

    return model


def _patch_decoder_layer(layer: GptOssDecoderLayer, device, variant="V1"):
    """Patch a single GPT-OSS decoder layer with fused QKV."""
    fused_weights = transform_gpt_oss_layer(layer)
    cls = _COMBINED_VARIANT_CLASSES[variant]

    # Combined QKV
    W_comb, b_comb, split_sizes, h, eps = fused_weights["attn_qkv"]
    layer.self_attn.fused_qkv = cls(
        W_comb.to(device), b_comb.to(device), split_sizes, h, eps
    )

    # Patch forwards
    _patch_attention_forward(layer.self_attn)
    _patch_layer_forward(layer)


def _patch_attention_forward(attn: GptOssAttention):
    """Replace attention forward to use fused QKV projection (skip RMSNorm)."""

    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        # Single fused call for Q, K, V (RMSNorm baked in)
        q_raw, k_raw, v_raw = attn.fused_qkv(hidden_states)

        # Use .reshape (not .view) since split outputs were made contiguous
        query_states = q_raw.reshape(hidden_shape).transpose(1, 2)
        key_states = k_raw.reshape(hidden_shape).transpose(1, 2)
        value_states = v_raw.reshape(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # GPT-OSS cache update: no sin/cos, only cache_position
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )

        # GQA decode path: single-token decode with custom kernel
        if query_states.size(2) == 1 and hasattr(attn, 'gqa_decode_module'):
            q_dec = query_states.squeeze(2)
            k_dec = key_states.permute(0, 2, 1, 3).contiguous()
            v_dec = value_states.permute(0, 2, 1, 3).contiguous()
            attn_output = attn.gqa_decode_module(q_dec, k_dec, v_dec)
            attn_output = attn_output.unsqueeze(2)
            attn_weights = None
        else:
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                attn.config._attn_implementation, eager_attention_forward
            )

            attn_output, attn_weights = attention_interface(
                attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not attn.training else attn.attention_dropout,
                scaling=attn.scaling,
                sliding_window=attn.sliding_window,
                s_aux=attn.sinks,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights

    attn.forward = patched_forward


def _patch_layer_forward(layer: GptOssDecoderLayer):
    """Replace decoder layer forward to skip input_layernorm (fused into QKV)."""

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        # Skip input_layernorm: fused QKV projection handles RMSNorm internally
        hidden_states, _ = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP path unchanged: post_attention_layernorm -> MoE MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states, _ = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    layer.forward = patched_forward
