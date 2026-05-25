"""
Monkey-patch a Llama model to use fused RMSNorm+Linear modules.

Replaces the forward pass of each Llama decoder layer so that:
  - input_layernorm + q/k/v_proj -> fused_q/k/v_proj
  - post_attention_layernorm + gate_proj, up_proj -> fused_gate_proj, fused_up_proj

The RMSNorm layers are skipped in the forward pass; their effect (gamma
scaling and 1/rms normalization) is baked into the fused weight matrices.

Note: down_proj is NOT fused (it follows an activation, not a norm).
"""

import torch
import torch.nn as nn
from typing import Callable
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaAttention,
    ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
    apply_rotary_pos_emb,
)

from src.weight_transform import transform_llama_layer, transform_llama_layer_combined
from src.fused_forward import (
    FusedRMSNormLinearV1, FusedRMSNormLinearV3,
    FusedRMSNormCombinedLinearV1, FusedRMSNormCombinedLinearV3,
    FusedRMSNormSwiGLUV1, FusedRMSNormSwiGLUV3,
)
from src.gqa_attention_forward import GQADecodeAttentionV3

_VARIANT_CLASSES = {
    "V1": FusedRMSNormLinearV1,
    "V3": FusedRMSNormLinearV3,
}

_COMBINED_VARIANT_CLASSES = {
    "V1": FusedRMSNormCombinedLinearV1,
    "V3": FusedRMSNormCombinedLinearV3,
}

_SWIGLU_VARIANT_CLASSES = {
    "V1": FusedRMSNormSwiGLUV1,
    "V3": FusedRMSNormSwiGLUV3,
}


def patch_llama_model(model, device=None, variant="V1", combined=False, swiglu=False,
                      gqa_decode=False):
    """
    Patch all decoder layers in a Llama model to use fused RMSNorm+Linear.

    Args:
        model: HuggingFace LlamaForCausalLM model
        device: target device (defaults to model's device)
        variant: kernel variant -- "V1" (256 threads) or "V3" (512 threads)
        combined: if True, combine QKV into one matmul and gate+up into one matmul
        swiglu: if True (implies combined), fuse SiLU+multiply into the MLP normalize kernel
        gqa_decode: if True, use GQA-optimized V3 kernel for single-token decode steps

    Returns:
        The patched model (modified in-place)
    """
    if variant not in _VARIANT_CLASSES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(_VARIANT_CLASSES)}")

    if swiglu:
        combined = True

    if device is None:
        device = next(model.parameters()).device

    # Extract GQA parameters from model config
    config = model.config
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_q_heads

    for layer_idx, layer in enumerate(model.model.layers):
        if swiglu:
            _patch_decoder_layer_swiglu(layer, device, variant)
        elif combined:
            _patch_decoder_layer_combined(layer, device, variant)
        else:
            _patch_decoder_layer(layer, device, variant)

        if gqa_decode and num_kv_heads < num_q_heads:
            layer.self_attn.gqa_decode_module = GQADecodeAttentionV3(
                num_q_heads, num_kv_heads, head_dim
            )

    return model


def _patch_decoder_layer(layer: LlamaDecoderLayer, device, variant="V1"):
    """Patch a single Llama decoder layer."""
    fused_weights = transform_llama_layer(layer)
    cls = _VARIANT_CLASSES[variant]

    # Fused attention projections (q/k/v share input_layernorm)
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        W_new, b_new, h, eps = fused_weights[f"attn_{proj_name}"]
        fused_mod = cls(W_new.to(device), b_new.to(device), h, eps)
        setattr(layer.self_attn, f"fused_{proj_name}", fused_mod)

    # Fused MLP projections (gate_proj, up_proj share post_attention_layernorm)
    for proj_name in ["gate_proj", "up_proj"]:
        W_new, b_new, h, eps = fused_weights[proj_name]
        fused_mod = cls(W_new.to(device), b_new.to(device), h, eps)
        setattr(layer.mlp, f"fused_{proj_name}", fused_mod)

    # Patch forwards
    _patch_attention_forward(layer.self_attn)
    _patch_layer_forward(layer)


def _patch_attention_forward(attn: LlamaAttention):
    """Replace attention forward to use fused q/k/v projections (skip RMSNorm)."""

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

        # Fused projections: RMSNorm is baked in, input is raw hidden_states
        query_states = attn.fused_q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn.fused_k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn.fused_v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )

        # GQA decode path: single-token decode with custom kernel
        if query_states.size(2) == 1 and hasattr(attn, 'gqa_decode_module'):
            q_dec = query_states.squeeze(2)                         # [batch, q_heads, head_dim]
            k_dec = key_states.permute(0, 2, 1, 3).contiguous()    # [batch, ctx, kv_heads, hd]
            v_dec = value_states.permute(0, 2, 1, 3).contiguous()
            attn_output = attn.gqa_decode_module(q_dec, k_dec, v_dec)
            attn_output = attn_output.unsqueeze(2)                  # [batch, q_heads, 1, hd]
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
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights

    attn.forward = patched_forward


def _patch_layer_forward(layer: LlamaDecoderLayer):
    """Replace decoder layer forward to skip standalone RMSNorm calls."""

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

        # Skip input_layernorm: fused q/k/v projections handle RMSNorm internally
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

        # MLP with fused gate_proj and up_proj (skip post_attention_layernorm)
        residual = hidden_states
        # Replicate LlamaMLP.forward but using fused projections
        gate = layer.mlp.fused_gate_proj(hidden_states)
        up = layer.mlp.fused_up_proj(hidden_states)
        hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
        hidden_states = residual + hidden_states

        return hidden_states

    layer.forward = patched_forward


# ---------------------------------------------------------------------------
# Combined QKV / gate+up patching
# ---------------------------------------------------------------------------

def _patch_decoder_layer_combined(layer: LlamaDecoderLayer, device, variant="V1"):
    """Patch a single Llama decoder layer using combined QKV and gate+up projections."""
    fused_weights = transform_llama_layer_combined(layer)
    cls = _COMBINED_VARIANT_CLASSES[variant]

    # Combined QKV
    W_comb, b_comb, split_sizes, h, eps = fused_weights["attn_qkv"]
    layer.self_attn.fused_qkv = cls(
        W_comb.to(device), b_comb.to(device), split_sizes, h, eps
    )

    # Combined gate+up
    W_comb, b_comb, split_sizes, h, eps = fused_weights["mlp_gate_up"]
    layer.mlp.fused_gate_up = cls(
        W_comb.to(device), b_comb.to(device), split_sizes, h, eps
    )

    _patch_attention_forward_combined(layer.self_attn)
    _patch_layer_forward_combined(layer)


def _patch_attention_forward_combined(attn: LlamaAttention):
    """Replace attention forward to use a single fused QKV projection."""

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

        # Single fused call for Q, K, V
        q_raw, k_raw, v_raw = attn.fused_qkv(hidden_states)

        # Use .reshape (not .view) since split outputs were made contiguous
        query_states = q_raw.reshape(hidden_shape).transpose(1, 2)
        key_states = k_raw.reshape(hidden_shape).transpose(1, 2)
        value_states = v_raw.reshape(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
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
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights

    attn.forward = patched_forward


def _patch_layer_forward_combined(layer: LlamaDecoderLayer):
    """Replace decoder layer forward using combined gate+up projection."""

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

        # Attention with fused combined QKV (skip input_layernorm)
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

        # MLP with single fused gate+up call (skip post_attention_layernorm)
        residual = hidden_states
        gate, up = layer.mlp.fused_gate_up(hidden_states)
        hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
        hidden_states = residual + hidden_states

        return hidden_states

    layer.forward = patched_forward


# ---------------------------------------------------------------------------
# SwiGLU-fused patching (combined QKV + fused SiLU+multiply in MLP)
# ---------------------------------------------------------------------------

def _patch_decoder_layer_swiglu(layer: LlamaDecoderLayer, device, variant="V1"):
    """Patch a single Llama decoder layer using combined QKV and fused SwiGLU MLP."""
    fused_weights = transform_llama_layer_combined(layer)
    combined_cls = _COMBINED_VARIANT_CLASSES[variant]
    swiglu_cls = _SWIGLU_VARIANT_CLASSES[variant]

    # Combined QKV (same as combined mode)
    W_comb, b_comb, split_sizes, h, eps = fused_weights["attn_qkv"]
    layer.self_attn.fused_qkv = combined_cls(
        W_comb.to(device), b_comb.to(device), split_sizes, h, eps
    )

    # SwiGLU MLP: use FusedRMSNormSwiGLU instead of FusedRMSNormCombinedLinear
    W_comb, b_comb, split_sizes, h, eps = fused_weights["mlp_gate_up"]
    intermediate = split_sizes[0]  # gate and up have same size
    layer.mlp.fused_gate_up_swiglu = swiglu_cls(
        W_comb.to(device), b_comb.to(device), intermediate, h, eps
    )

    _patch_attention_forward_combined(layer.self_attn)
    _patch_layer_forward_swiglu(layer)


def _patch_layer_forward_swiglu(layer: LlamaDecoderLayer):
    """Replace decoder layer forward using fused SwiGLU MLP."""

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

        # Attention with fused combined QKV (skip input_layernorm)
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

        # MLP: single fused call replaces norm + gate + up + SiLU + mul
        residual = hidden_states
        activated = layer.mlp.fused_gate_up_swiglu(hidden_states)
        hidden_states = layer.mlp.down_proj(activated)
        hidden_states = residual + hidden_states

        return hidden_states

    layer.forward = patched_forward
