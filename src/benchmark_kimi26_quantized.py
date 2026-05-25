"""
Quantization-aware Kimi K2.6 RMSNorm fusion benchmark.

This is a synthetic weight-only int8 experiment for Kimi's direct pairs:
  - q_a_layernorm -> q_b_proj
  - kv_a_layernorm -> kv_b_proj

It answers a narrower question than full quantized LLM serving:
  "Does the algebraic RMSNorm weight folding still work when weights are
   quantized after folding?"

It does NOT claim production INT8/INT4 speed. The benchmark dequantizes weights
to BF16 before matmul because the custom CUDA fusion kernels currently expect
BF16 weights. For real quantized LLM speed, the dequantization step must be
fused into the GEMM kernel or handled by a serving engine such as vLLM/AWQ/GPTQ.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.benchmark_kimi26 import CONFIGS, SimpleRMSNorm


def quantize_int8_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantization for Linear weights."""
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(1)


def dequantize_int8(qweight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (qweight.float() * scale.float().unsqueeze(1)).to(dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark quantization-aware Kimi RMSNorm fusion.")
    parser.add_argument("--config", choices=CONFIGS.keys(), default="kimi_q_b")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--out", type=int)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=300)
    return parser.parse_args()


def sync() -> None:
    torch.cuda.synchronize()


def bench(fn, warmup: int, iters: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

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

    # Baseline quantized path: quantize original W.
    q_w, q_scale = quantize_int8_per_output_channel(linear.weight.detach())
    q_w = q_w.to(device)
    q_scale = q_scale.to(device)

    # Fused quantized path: fold gamma first, then quantize W_fused.
    fused_weight = linear.weight.detach() * norm.weight.detach().to(dtype).unsqueeze(0)
    q_fw, q_fscale = quantize_int8_per_output_channel(fused_weight)
    q_fw = q_fw.to(device)
    q_fscale = q_fscale.to(device)

    def baseline_quantized():
        w = dequantize_int8(q_w, q_scale, dtype)
        return F.linear(norm(x), w)

    def fused_quantized():
        w = dequantize_int8(q_fw, q_fscale, dtype)
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + norm.variance_epsilon).to(dtype)
        return F.linear(x * inv_rms, w)

    with torch.no_grad():
        baseline = baseline_quantized().float()
        fused = fused_quantized().float()
        diff = (baseline - fused).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())

    baseline_ms = bench(baseline_quantized, args.warmup, args.iters)
    fused_ms = bench(fused_quantized, args.warmup, args.iters)

    print(f"config={args.config} quantization=int8_per_output_channel")
    print(f"gpu={torch.cuda.get_device_name(0)} dtype={dtype}")
    print(f"shape=({tokens}, {hidden}) -> {out}")
    print("note=weights are dequantized to BF16 before matmul; this is not production int8 GEMM")
    print(f"max_abs_diff={max_diff:.6e}")
    print(f"mean_abs_diff={mean_diff:.6e}")
    print(f"baseline_quantized_ms={baseline_ms:.4f}")
    print(f"fused_quantized_ms={fused_ms:.4f}")
    print(f"speedup={baseline_ms / fused_ms:.3f}x")


if __name__ == "__main__":
    main()
