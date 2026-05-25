"""
Monkey-patch a Llama model to use Hopper-optimized cuDNN attention (FA-3 equivalent).

Replaces the attention forward pass to force the cuDNN SDPA backend for prefill,
which uses Hopper-specific WGMMA and TMA instructions on H100 GPUs.

Composable with existing fused RMSNorm+Linear patches from patch_llama.py:
  - patch_llama_fa3_only: only replaces attention backend
  - patch_llama_fa3_with_fused: combines FA-3 attention + SwiGLU-V3 fused kernels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaAttention,
    apply_rotary_pos_emb,
)

from src.fa3_attention import SDPA_BACKENDS


def _should_use_gqa_sdpa(attention_mask, key):
    """Check if we can use enable_gqa=True in SDPA (no mask or simple 2D mask)."""
    if attention_mask is None:
        return True
    if attention_mask.dim() == 2:
        return True
    return False


def patch_llama_fa3_only(model, backend="cudnn"):
    """
    Patch all decoder layers to force a specific SDPA backend for attention.

    This only changes the attention backend; all other computations (norms,
    projections, MLP) remain unchanged.

    Args:
        model: HuggingFace LlamaForCausalLM model
        backend: SDPA backend name ("cudnn", "flash", "efficient", "math")

    Returns:
        The patched model (modified in-place)
    """
    if backend not in SDPA_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {list(SDPA_BACKENDS)}")

    sdp_backend = SDPA_BACKENDS[backend]
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    use_gqa = num_kv_heads < num_q_heads
    num_key_value_groups = num_q_heads // num_kv_heads

    for layer in model.model.layers:
        _patch_attention_fa3(layer.self_attn, sdp_backend, use_gqa, num_key_value_groups)

    return model


def patch_llama_fa3_with_fused(model, device=None, backend="cudnn"):
    """
    Patch all decoder layers to use:
      - FA-3/cuDNN attention for prefill
      - SwiGLU-V3 fused RMSNorm+Linear for norms and projections
      - GQA-V3 decode kernel for single-token decode

    This is the "full stack" configuration combining all optimizations.

    Args:
        model: HuggingFace LlamaForCausalLM model
        device: target device (defaults to model's device)
        backend: SDPA backend name for prefill attention

    Returns:
        The patched model (modified in-place)
    """
    from src.patch_llama import patch_llama_model

    if backend not in SDPA_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {list(SDPA_BACKENDS)}")

    # First apply fused RMSNorm+Linear + SwiGLU + GQA decode patches
    patch_llama_model(model, device=device, variant="V3", swiglu=True, gqa_decode=True)

    # Then override the attention to use the specified SDPA backend for prefill
    sdp_backend = SDPA_BACKENDS[backend]
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    use_gqa = num_kv_heads < num_q_heads
    num_key_value_groups = num_q_heads // num_kv_heads

    for layer in model.model.layers:
        _patch_attention_fa3_fused(layer.self_attn, sdp_backend, use_gqa, num_key_value_groups)

    return model


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query heads (for non-GQA SDPA path)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _patch_attention_fa3(attn: LlamaAttention, sdp_backend, use_gqa: bool,
                         num_key_value_groups: int):
    """Patch attention forward to force a specific SDPA backend.

    This version works with unmodified (non-fused) Q/K/V projections.
    Replicates HuggingFace's sdpa_attention_forward logic but wraps with
    the specified backend context manager.
    """

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

        # hidden_states are already normed by the decoder layer's input_layernorm
        query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )

        # Replicate HF SDPA logic for GQA and causal handling
        sdpa_kwargs = {}
        if use_gqa:
            if _should_use_gqa_sdpa(attention_mask, key_states):
                sdpa_kwargs["enable_gqa"] = True
            else:
                key_states = _repeat_kv(key_states, num_key_value_groups)
                value_states = _repeat_kv(value_states, num_key_value_groups)

        is_causal = query_states.shape[2] > 1 and attention_mask is None

        # Force the specified SDPA backend
        with torch.nn.attention.sdpa_kernel(sdp_backend):
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=attn.scaling,
                is_causal=is_causal,
                **sdpa_kwargs,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = attn.o_proj(attn_output)
        return attn_output, None

    attn.forward = patched_forward


def _patch_attention_fa3_fused(attn: LlamaAttention, sdp_backend, use_gqa: bool,
                               num_key_value_groups: int):
    """Patch attention forward to force a specific SDPA backend.

    This version works with fused Q/K/V projections from patch_llama.py (SwiGLU mode).
    It overrides the attention mechanism while keeping fused projections and GQA decode.
    """

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

        # Use fused QKV projection (RMSNorm baked in)
        q_raw, k_raw, v_raw = attn.fused_qkv(hidden_states)
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
        else:
            # Prefill: force the specified SDPA backend (FA-3/cuDNN)
            sdpa_kwargs = {}
            if use_gqa:
                if _should_use_gqa_sdpa(attention_mask, key_states):
                    sdpa_kwargs["enable_gqa"] = True
                else:
                    key_states = _repeat_kv(key_states, num_key_value_groups)
                    value_states = _repeat_kv(value_states, num_key_value_groups)

            is_causal = query_states.shape[2] > 1 and attention_mask is None

            with torch.nn.attention.sdpa_kernel(sdp_backend):
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    scale=attn.scaling,
                    is_causal=is_causal,
                    **sdpa_kwargs,
                )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = attn.o_proj(attn_output)
        return attn_output, None

    attn.forward = patched_forward
