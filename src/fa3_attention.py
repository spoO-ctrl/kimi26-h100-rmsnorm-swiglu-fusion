"""
FlashAttention-3 / Hopper-optimized attention via PyTorch cuDNN SDPA backend.

On H100 (sm_90), the cuDNN SDPA backend uses Hopper-specific WGMMA and TMA
instructions for fused attention, achieving similar performance characteristics
to Dao-AILab's FlashAttention-3.

This module provides:
  - FA3Attention: wrapper around torch SDPA with explicit backend selection
  - Supports MHA and GQA (different Q vs KV head counts)
  - Causal masking
  - BF16/FP16 precision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Available SDPA backends for benchmarking
SDPA_BACKENDS = {
    "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
    "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    "math": torch.nn.attention.SDPBackend.MATH,
}


class FA3Attention(nn.Module):
    """Attention module that forces a specific SDPA backend.

    When backend="cudnn", this uses the Hopper-optimized cuDNN fused attention
    path (FA-3 equivalent) on H100 GPUs.
    """

    def __init__(self, backend="cudnn"):
        super().__init__()
        if backend not in SDPA_BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; choose from {list(SDPA_BACKENDS)}")
        self.backend = backend
        self._sdp_backend = SDPA_BACKENDS[backend]

    def forward(self, query, key, value, is_causal=True, enable_gqa=False, scale=None):
        """
        Args:
            query:  [batch, num_q_heads, seq_len, head_dim]
            key:    [batch, num_kv_heads, seq_len, head_dim]
            value:  [batch, num_kv_heads, seq_len, head_dim]
            is_causal: apply causal mask
            enable_gqa: if True, allow different num_q_heads vs num_kv_heads
            scale: optional softmax scale (default: 1/sqrt(head_dim))
        Returns:
            output: [batch, num_q_heads, seq_len, head_dim]
        """
        with torch.nn.attention.sdpa_kernel(self._sdp_backend):
            return F.scaled_dot_product_attention(
                query, key, value,
                is_causal=is_causal,
                enable_gqa=enable_gqa,
                scale=scale,
            )


def sdpa_with_backend(query, key, value, backend="cudnn", is_causal=True,
                      enable_gqa=False, scale=None):
    """Functional interface for SDPA with explicit backend selection."""
    sdp_backend = SDPA_BACKENDS[backend]
    with torch.nn.attention.sdpa_kernel(sdp_backend):
        return F.scaled_dot_product_attention(
            query, key, value,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
            scale=scale,
        )


def sdpa_default(query, key, value, is_causal=True, enable_gqa=False, scale=None):
    """SDPA with PyTorch auto-selected backend (no forcing)."""
    return F.scaled_dot_product_attention(
        query, key, value,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
        scale=scale,
    )
