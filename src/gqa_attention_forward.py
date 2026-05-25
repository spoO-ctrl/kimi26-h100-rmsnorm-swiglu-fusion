"""
Python modules for GQA-optimized decode attention.

V2 (baseline): Per-query-head — each block handles one query head,
    loads K,V from global memory (redundant across grouped heads).

V3 (optimized): Per-KV-head — each block loads K,V into shared memory
    once and computes attention for all group_size query heads.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.load_gqa_attention import gqa_attention_cuda


class GQADecodeAttentionV2(nn.Module):
    """V2 baseline: per-query-head decode attention."""

    def __init__(self, num_q_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, q, k_cache, v_cache):
        """
        Args:
            q:       [batch, num_q_heads, head_dim]
            k_cache: [batch, context_len, num_kv_heads, head_dim]
            v_cache: [batch, context_len, num_kv_heads, head_dim]
        Returns:
            output:  [batch, num_q_heads, head_dim]
        """
        output = torch.empty_like(q)
        gqa_attention_cuda.gqa_decode_v2(q, k_cache, v_cache, output, self.scale)
        return output


class GQADecodeAttentionV3(nn.Module):
    """V3 optimized: per-KV-head with shared K,V in shared memory."""

    def __init__(self, num_q_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_q_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, q, k_cache, v_cache):
        """
        Args:
            q:       [batch, num_q_heads, head_dim]
            k_cache: [batch, context_len, num_kv_heads, head_dim]
            v_cache: [batch, context_len, num_kv_heads, head_dim]
        Returns:
            output:  [batch, num_q_heads, head_dim]
        """
        output = torch.empty_like(q)
        gqa_attention_cuda.gqa_decode_v3(
            q, k_cache, v_cache, output, self.group_size, self.scale
        )
        return output


def pytorch_gqa_decode_attention(q, k_cache, v_cache, num_kv_heads):
    """
    Reference PyTorch implementation for correctness testing.

    Args:
        q:       [batch, num_q_heads, head_dim]
        k_cache: [batch, context_len, num_kv_heads, head_dim]
        v_cache: [batch, context_len, num_kv_heads, head_dim]
        num_kv_heads: number of KV heads
    Returns:
        output:  [batch, num_q_heads, head_dim]
    """
    batch, num_q_heads, head_dim = q.shape
    ctx_len = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Expand KV heads to match query heads: repeat_interleave along head dim
    # k_cache: [batch, ctx_len, num_kv_heads, head_dim] -> [batch, ctx_len, num_q_heads, head_dim]
    k_expanded = k_cache.repeat_interleave(group_size, dim=2)
    v_expanded = v_cache.repeat_interleave(group_size, dim=2)

    # q: [batch, num_q_heads, 1, head_dim]
    # k: [batch, num_q_heads, ctx_len, head_dim]  (after transpose)
    q_4d = q.unsqueeze(2)  # [batch, num_q_heads, 1, head_dim]
    k_4d = k_expanded.permute(0, 2, 1, 3)  # [batch, num_q_heads, ctx_len, head_dim]
    v_4d = v_expanded.permute(0, 2, 1, 3)  # [batch, num_q_heads, ctx_len, head_dim]

    # Use SDPA for reference (numerically stable)
    out = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=scale)
    return out.squeeze(2)  # [batch, num_q_heads, head_dim]
