"""
Optimized MoE (Mixture of Experts) forward pass using Triton grouped GEMM.

Replaces the naive Python loop over experts with:
  1. Token sorting by expert assignment for contiguous memory access
  2. Triton grouped GEMM kernel for all expert gate_up projections
  3. Fused Triton gating kernel (clamped sigmoid GLU, alpha=1.702)
  4. Triton grouped GEMM kernel for all expert down projections
  5. Weighted scatter-add combine

Targets GPT-OSS-20B architecture:
  - 128 experts, top-4 routing
  - hidden=2880, intermediate=2880
  - Interleaved gate/up layout: gate_up[..., ::2] = gate, gate_up[..., 1::2] = up
  - Custom gating: glu = gate * sigmoid(gate * alpha), out = (up + 1) * glu
  - Weights are transposed: [num_experts, hidden, out_dim]
  - Experts have bias
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton fused gating kernel for GPT-OSS custom activation
# ---------------------------------------------------------------------------

@triton.jit
def _gptoss_gate_kernel(
    gate_up_ptr,    # [total_tokens, 2*intermediate] - interleaved gate/up
    output_ptr,     # [total_tokens, intermediate] - gated output
    alpha,          # 1.702
    limit,          # 7.0
    total_tokens,   # number of tokens
    intermediate,   # intermediate dim
    stride_gu_row,  # stride of gate_up along row dim
    stride_out_row, # stride of output along row dim
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GPT-OSS gating: gate=clamp(gate,max=L), up=clamp(up,-L,L),
    glu = gate * sigmoid(gate * alpha), out = (up + 1) * glu.

    gate_up has interleaved layout: even indices are gate, odd are up.
    """
    pid = tl.program_id(0)
    row = pid // tl.cdiv(intermediate, BLOCK_SIZE)
    col_block = pid % tl.cdiv(intermediate, BLOCK_SIZE)

    if row >= total_tokens:
        return

    col_offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offs < intermediate

    # Interleaved: gate is at even indices (2*col), up is at odd indices (2*col+1)
    gate_offs = row * stride_gu_row + col_offs * 2
    up_offs = row * stride_gu_row + col_offs * 2 + 1

    gate = tl.load(gate_up_ptr + gate_offs, mask=mask, other=0.0)
    up = tl.load(gate_up_ptr + up_offs, mask=mask, other=0.0)

    # Clamp
    gate = tl.minimum(gate, limit)
    up = tl.maximum(tl.minimum(up, limit), -limit)

    # GLU: gate * sigmoid(gate * alpha)
    glu = gate * tl.sigmoid(gate * alpha)

    # Output: (up + 1) * glu
    out = (up + 1.0) * glu

    out_offs = row * stride_out_row + col_offs
    tl.store(output_ptr + out_offs, out, mask=mask)


def fused_gptoss_gate(gate_up: torch.Tensor, intermediate: int,
                      alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """Apply GPT-OSS custom gating activation.

    Args:
        gate_up: [total_tokens, 2*intermediate] with interleaved gate/up
        intermediate: size of the intermediate dimension
        alpha: sigmoid scaling factor (1.702)
        limit: clamp limit (7.0)

    Returns:
        output: [total_tokens, intermediate]
    """
    total_tokens = gate_up.shape[0]
    output = torch.empty(total_tokens, intermediate,
                         device=gate_up.device, dtype=gate_up.dtype)

    BLOCK_SIZE = 256
    num_col_blocks = triton.cdiv(intermediate, BLOCK_SIZE)
    grid = (total_tokens * num_col_blocks,)

    _gptoss_gate_kernel[grid](
        gate_up, output,
        alpha, limit,
        total_tokens, intermediate,
        gate_up.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ---------------------------------------------------------------------------
# Triton grouped GEMM kernel (persistent, tile-map based)
# ---------------------------------------------------------------------------
#
# Strategy: precompute a flat array of (expert_id, row_start, num_rows) per
# tile on the host, then the kernel simply indexes into it. This avoids
# unsupported control flow (continue/break) inside Triton.
#

@triton.jit
def _grouped_gemm_kernel(
    # Pointers
    A_ptr,            # [total_tokens, K]
    B_ptr,            # [num_experts, K, N]
    C_ptr,            # [total_tokens, N]
    bias_ptr,         # [num_experts, N]
    # Tile map: each entry is (expert_id, tile_row_start_in_A, expert_num_rows)
    tile_expert_ids,  # [num_tiles] int32
    tile_row_starts,  # [num_tiles] int32
    tile_n_ids,       # [num_tiles] int32 - which N-tile
    tile_end_rows,    # [num_tiles] int32 - end row for bounds checking
    num_tiles,
    # Dimensions
    K, N,
    # Strides for A
    stride_a_row, stride_a_col,
    # Strides for B
    stride_b_expert, stride_b_k, stride_b_n,
    # Strides for C
    stride_c_row, stride_c_col,
    # Strides for bias
    stride_bias_expert, stride_bias_n,
    # Flags
    HAS_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent grouped GEMM with precomputed tile mapping.

    Each program picks tiles from the tile map in a strided fashion.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # Persistent loop: each SM processes tiles in stride
    for tile_idx in tl.range(pid, num_tiles, num_pids):
        # Read tile assignment
        expert_id = tl.load(tile_expert_ids + tile_idx)
        row_start = tl.load(tile_row_starts + tile_idx)
        n_tile = tl.load(tile_n_ids + tile_idx)
        end_row = tl.load(tile_end_rows + tile_idx)

        col_start = n_tile * BLOCK_N

        # Row and column offsets for this tile
        rm = row_start + tl.arange(0, BLOCK_M)
        rn = col_start + tl.arange(0, BLOCK_N)

        # Masks
        mask_m = rm < end_row
        mask_n = rn < N

        # Accumulator in FP32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # GEMM loop over K dimension
        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < K

            # Load A tile: [BLOCK_M, BLOCK_K]
            a_offs = rm[:, None] * stride_a_row + rk[None, :] * stride_a_col
            a = tl.load(A_ptr + a_offs,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0)

            # Load B tile: [BLOCK_K, BLOCK_N] from expert weights
            b_offs = (expert_id * stride_b_expert +
                      rk[:, None] * stride_b_k +
                      rn[None, :] * stride_b_n)
            b = tl.load(B_ptr + b_offs,
                        mask=mask_k[:, None] & mask_n[None, :],
                        other=0.0)

            acc += tl.dot(a, b)

        # Add bias
        if HAS_BIAS:
            bias_offs = expert_id * stride_bias_expert + rn * stride_bias_n
            bias_val = tl.load(bias_ptr + bias_offs, mask=mask_n, other=0.0)
            acc += bias_val[None, :]

        # Store result
        c_offs = rm[:, None] * stride_c_row + rn[None, :] * stride_c_col
        tl.store(C_ptr + c_offs, acc.to(C_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


def _build_tile_map(offsets, N, BLOCK_M, BLOCK_N, device):
    """Build flat tile map on CPU, transfer to GPU.

    Returns tile_expert_ids, tile_row_starts, tile_n_ids, tile_end_rows as int32 tensors.
    """
    offsets_cpu = offsets.cpu().tolist()
    num_experts = len(offsets_cpu) - 1
    num_n_tiles = triton.cdiv(N, BLOCK_N)

    expert_ids = []
    row_starts = []
    n_ids = []
    end_rows = []

    for e in range(num_experts):
        start = offsets_cpu[e]
        end = offsets_cpu[e + 1]
        num_rows = end - start
        if num_rows <= 0:
            pass  # skip empty experts
        else:
            num_m_tiles = triton.cdiv(num_rows, BLOCK_M)
            for m in range(num_m_tiles):
                for n in range(num_n_tiles):
                    expert_ids.append(e)
                    row_starts.append(start + m * BLOCK_M)
                    n_ids.append(n)
                    end_rows.append(end)

    if not expert_ids:
        return (torch.zeros(1, dtype=torch.int32, device=device),) * 4 + (0,)

    return (
        torch.tensor(expert_ids, dtype=torch.int32, device=device),
        torch.tensor(row_starts, dtype=torch.int32, device=device),
        torch.tensor(n_ids, dtype=torch.int32, device=device),
        torch.tensor(end_rows, dtype=torch.int32, device=device),
        len(expert_ids),
    )


def triton_grouped_gemm(
    A_sorted: torch.Tensor,        # [total_tokens, K]
    B_experts: torch.Tensor,        # [num_experts, K, N]
    offsets: torch.Tensor,          # [num_experts + 1] int32
    bias: torch.Tensor | None = None,  # [num_experts, N] or None
) -> torch.Tensor:
    """Execute grouped GEMM across all experts using Triton persistent kernel.

    Args:
        A_sorted: activations sorted by expert, [total_tokens, K]
        B_experts: expert weight matrices, [num_experts, K, N]
        offsets: cumulative token counts, [num_experts + 1]
        bias: optional per-expert bias, [num_experts, N]

    Returns:
        C: [total_tokens, N] output
    """
    total_tokens, K = A_sorted.shape
    num_experts, _, N = B_experts.shape

    C = torch.empty(total_tokens, N, device=A_sorted.device, dtype=A_sorted.dtype)

    if total_tokens == 0:
        return C

    # Pick block sizes based on problem dimensions
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32
    if K >= 2048 and N >= 2048:
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

    # Build tile map
    tile_expert_ids, tile_row_starts, tile_n_ids, tile_end_rows, num_tiles = \
        _build_tile_map(offsets, N, BLOCK_M, BLOCK_N, A_sorted.device)

    if num_tiles == 0:
        return C

    NUM_SMS = 132  # H100
    grid = (min(NUM_SMS, num_tiles),)

    has_bias = bias is not None
    if not has_bias:
        bias = torch.empty(1, 1, device=A_sorted.device, dtype=A_sorted.dtype)

    _grouped_gemm_kernel[grid](
        A_sorted, B_experts, C, bias,
        tile_expert_ids, tile_row_starts, tile_n_ids, tile_end_rows,
        num_tiles,
        K, N,
        A_sorted.stride(0), A_sorted.stride(1),
        B_experts.stride(0), B_experts.stride(1), B_experts.stride(2),
        C.stride(0), C.stride(1),
        bias.stride(0) if has_bias else 0,
        bias.stride(1) if has_bias and bias.ndim == 2 else (bias.stride(0) if has_bias else 0),
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


# ---------------------------------------------------------------------------
# Padded batched GEMM approach (uses torch.bmm)
# ---------------------------------------------------------------------------

def padded_batched_gemm(
    A_sorted: torch.Tensor,        # [total_tokens, K]
    B_experts: torch.Tensor,        # [num_experts, K, N]
    offsets: torch.Tensor,          # [num_experts + 1] int32
    bias: torch.Tensor | None = None,  # [num_experts, N] or None
) -> torch.Tensor:
    """Execute grouped GEMM by padding to max tokens and using batched matmul."""
    total_tokens, K = A_sorted.shape
    num_experts, _, N = B_experts.shape

    C = torch.zeros(total_tokens, N, device=A_sorted.device, dtype=A_sorted.dtype)

    if total_tokens == 0:
        return C

    offsets_cpu = offsets.cpu()
    counts = offsets_cpu[1:] - offsets_cpu[:-1]
    max_tokens = int(counts.max().item())

    if max_tokens == 0:
        return C

    # Find active experts
    active_mask = counts > 0
    active_indices = active_mask.nonzero().squeeze(-1)
    num_active = len(active_indices)

    if num_active == 0:
        return C

    # Pad activations into [num_active, max_tokens, K]
    A_padded = torch.zeros(num_active, max_tokens, K,
                           device=A_sorted.device, dtype=A_sorted.dtype)
    for i, expert_idx in enumerate(active_indices):
        start = offsets_cpu[expert_idx].item()
        end = offsets_cpu[expert_idx + 1].item()
        count = end - start
        if count > 0:
            A_padded[i, :count] = A_sorted[start:end]

    # Batched matmul
    B_active = B_experts[active_indices.to(B_experts.device)]  # [num_active, K, N]
    C_padded = torch.bmm(A_padded, B_active)  # [num_active, max_tokens, N]

    # Add bias
    if bias is not None:
        bias_active = bias[active_indices.to(bias.device)]  # [num_active, N]
        C_padded = C_padded + bias_active[:, None, :]

    # Scatter back
    for i, expert_idx in enumerate(active_indices):
        start = offsets_cpu[expert_idx].item()
        end = offsets_cpu[expert_idx + 1].item()
        count = end - start
        if count > 0:
            C[start:end] = C_padded[i, :count]

    return C


# ---------------------------------------------------------------------------
# Complete optimized MoE forward pass
# ---------------------------------------------------------------------------

def sort_tokens_by_expert(
    hidden_states: torch.Tensor,    # [num_tokens, hidden]
    router_indices: torch.Tensor,   # [num_tokens, top_k]
    router_scores: torch.Tensor,    # [num_tokens, top_k]
    num_experts: int,
):
    """Sort tokens by expert assignment for contiguous memory access.

    Returns:
        sorted_states: [total_active, hidden] - tokens sorted by expert
        sorted_scores: [total_active] - corresponding routing weights
        sorted_token_ids: [total_active] - original token indices
        offsets: [num_experts + 1] - cumulative token count per expert
        sorted_top_k_pos: [total_active] - which top-k slot each entry came from
    """
    num_tokens, top_k = router_indices.shape
    device = hidden_states.device

    # Flatten: each (token, k) pair -> (expert_id, token_id, score, k_pos)
    flat_experts = router_indices.reshape(-1)        # [num_tokens * top_k]
    flat_token_ids = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    flat_scores = router_scores.reshape(-1)          # [num_tokens * top_k]
    flat_k_pos = torch.arange(top_k, device=device).unsqueeze(0).expand(num_tokens, -1).reshape(-1)

    # Sort by expert
    sorted_expert_ids, sort_perm = flat_experts.sort(stable=True)
    sorted_token_ids = flat_token_ids[sort_perm]
    sorted_scores = flat_scores[sort_perm]
    sorted_top_k_pos = flat_k_pos[sort_perm]

    # Gather sorted hidden states
    sorted_states = hidden_states[sorted_token_ids]   # [total_active, hidden]

    # Compute offsets using bincount
    counts = torch.bincount(sorted_expert_ids.int(), minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = counts.cumsum(0).int()

    return sorted_states, sorted_scores, sorted_token_ids, offsets, sorted_top_k_pos


def optimized_moe_forward(
    hidden_states: torch.Tensor,    # [num_tokens, hidden]
    router_indices: torch.Tensor,   # [num_tokens, top_k]
    router_scores: torch.Tensor,    # [num_tokens, top_k]
    gate_up_proj: torch.Tensor,     # [num_experts, hidden, 2*intermediate]
    gate_up_proj_bias: torch.Tensor,  # [num_experts, 2*intermediate]
    down_proj: torch.Tensor,        # [num_experts, intermediate, hidden]
    down_proj_bias: torch.Tensor,   # [num_experts, hidden]
    alpha: float = 1.702,
    limit: float = 7.0,
    use_triton_gemm: bool = True,
) -> torch.Tensor:
    """Optimized MoE forward pass with grouped GEMM and fused gating.

    Args:
        hidden_states: input activations [num_tokens, hidden]
        router_indices: expert assignments [num_tokens, top_k]
        router_scores: routing weights [num_tokens, top_k]
        gate_up_proj: expert gate+up weights [num_experts, hidden, 2*intermediate]
        gate_up_proj_bias: expert gate+up biases [num_experts, 2*intermediate]
        down_proj: expert down weights [num_experts, intermediate, hidden]
        down_proj_bias: expert down biases [num_experts, hidden]
        alpha: gating sigmoid scale
        limit: gating clamp limit
        use_triton_gemm: if True, use Triton grouped GEMM; else use padded batched

    Returns:
        output: [num_tokens, hidden]
    """
    num_tokens, hidden = hidden_states.shape
    num_experts = gate_up_proj.shape[0]
    intermediate = down_proj.shape[1]

    # Step 1: Sort tokens by expert assignment
    sorted_states, sorted_scores, sorted_token_ids, offsets, _ = \
        sort_tokens_by_expert(hidden_states, router_indices, router_scores, num_experts)

    total_active = sorted_states.shape[0]

    if total_active == 0:
        return torch.zeros_like(hidden_states)

    gemm_fn = triton_grouped_gemm if use_triton_gemm else padded_batched_gemm

    # Step 2: Grouped GEMM for gate_up projection
    gate_up_out = gemm_fn(sorted_states, gate_up_proj, offsets, gate_up_proj_bias)

    # Step 3: Fused gating activation
    gated_out = fused_gptoss_gate(gate_up_out, intermediate, alpha, limit)

    # Step 4: Grouped GEMM for down projection
    down_out = gemm_fn(gated_out, down_proj, offsets, down_proj_bias)

    # Step 5: Apply routing weights and scatter-add
    weighted_out = down_out * sorted_scores.unsqueeze(-1)

    # Scatter back to original token positions
    output = torch.zeros(num_tokens, hidden, device=hidden_states.device,
                         dtype=hidden_states.dtype)
    output.index_add_(0, sorted_token_ids, weighted_out)

    return output


# ---------------------------------------------------------------------------
# Baseline MoE forward (loop-based, matching GptOssExperts.forward)
# ---------------------------------------------------------------------------

def baseline_moe_forward(
    hidden_states: torch.Tensor,    # [num_tokens, hidden]
    router_indices: torch.Tensor,   # [num_tokens, top_k]
    router_scores: torch.Tensor,    # [num_tokens, top_k]
    gate_up_proj: torch.Tensor,     # [num_experts, hidden, 2*intermediate]
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,        # [num_experts, intermediate, hidden]
    down_proj_bias: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Baseline MoE forward matching GptOssExperts loop implementation."""
    num_tokens = hidden_states.shape[0]
    num_experts = gate_up_proj.shape[0]

    next_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            router_indices, num_classes=num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        # gate_up projection
        gate_up = current_state @ gate_up_proj[expert_idx] + gate_up_proj_bias[expert_idx]

        # Custom gating
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        gated_output = (up + 1) * glu

        # down projection
        out = gated_output @ down_proj[expert_idx] + down_proj_bias[expert_idx]

        weighted_output = out * router_scores[token_idx, top_k_pos, None]
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

    return next_states
