"""
Synthetic Kimi K2.6 CUDA benchmarks for the H100 RMSNorm fusion kernels.

These benchmarks target Kimi's direct LoRA RMSNorm -> Linear pairs:
  - q_a_layernorm -> q_b_proj
  - kv_a_layernorm -> kv_b_proj

They do not require downloading full Kimi K2.6 weights.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from src.fused_forward import FusedRMSNormLinearV1, FusedRMSNormLinearV3
from src.weight_transform import compute_fused_weights_rmsnorm


class SimpleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.variance_epsilon).to(x.dtype) * self.weight.to(x.dtype)


CONFIGS = {
    # q_lora_rank -> num_heads * (qk_nope_head_dim + qk_rope_head_dim)
    "kimi_q_b": {"tokens": 512, "hidden": 1536, "out": 12288},
    # kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
    "kimi_kv_b": {"tokens": 512, "hidden": 512, "out": 16384},
    "kimi_q_b_long": {"tokens": 4096, "hidden": 1536, "out": 12288},
    "kimi_kv_b_long": {"tokens": 4096, "hidden": 512, "out": 16384},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Kimi K2.6 RMSNorm fusion CUDA kernels.")
    parser.add_argument("--config", choices=CONFIGS.keys(), default="kimi_q_b")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--out", type=int)
    parser.add_argument("--variant", choices=["V1", "V3"], default="V3")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=300)
    return parser.parse_args()


def sync() -> None:
    torch.cuda.synchronize()


def bench(fn, x, warmup: int, iters: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        sync()
        start = time.perf_counter()
        for _ in range(iters):
            fn(x)
        sync()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    args = parse_args()
    cfg = CONFIGS[args.config]
    tokens = args.tokens or cfg["tokens"]
    hidden = args.hidden or cfg["hidden"]
    out = args.out or cfg["out"]
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")

    torch.manual_seed(1234)
    norm = SimpleRMSNorm(hidden, eps=1e-5).to(device=device, dtype=dtype)
    linear = nn.Linear(hidden, out, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)

    W_new, b_new, h, eps = compute_fused_weights_rmsnorm(norm, linear)
    fused_cls = FusedRMSNormLinearV3 if args.variant == "V3" else FusedRMSNormLinearV1
    fused = fused_cls(W_new.to(device), b_new.to(device), h, eps)

    baseline = lambda t: linear(norm(t))

    with torch.no_grad():
        diff = (baseline(x).float() - fused(x).float()).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())

    baseline_ms = bench(baseline, x, args.warmup, args.iters)
    fused_ms = bench(fused, x, args.warmup, args.iters)

    print(f"config={args.config} variant={args.variant}")
    print(f"gpu={torch.cuda.get_device_name(0)} dtype={dtype}")
    print(f"shape=({tokens}, {hidden}) -> {out}")
    print(f"max_abs_diff={max_diff:.6e}")
    print(f"mean_abs_diff={mean_diff:.6e}")
    print(f"baseline_ms={baseline_ms:.4f}")
    print(f"fused_ms={fused_ms:.4f}")
    print(f"speedup={baseline_ms / fused_ms:.3f}x")


if __name__ == "__main__":
    main()
