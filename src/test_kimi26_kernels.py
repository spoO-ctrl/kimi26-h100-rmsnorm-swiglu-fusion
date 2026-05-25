"""
Kimi K2.6-specific CUDA correctness tests.

This intentionally avoids inherited OPT/Llama/GPT-OSS integration tests. Those
tests are useful upstream references, but this repo's active target is Kimi's
direct DeepseekV3RMSNorm -> Linear LoRA projection pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.benchmark_kimi26 import CONFIGS, SimpleRMSNorm
from src.fused_forward import FusedRMSNormLinearV1, FusedRMSNormLinearV3
from src.weight_transform import compute_fused_weights_rmsnorm


def _check_pair(config_name: str, variant: str, dtype: torch.dtype) -> None:
    cfg = CONFIGS[config_name]
    tokens = min(cfg["tokens"], 512)
    hidden = cfg["hidden"]
    out = cfg["out"]
    device = torch.device("cuda")

    torch.manual_seed(1234)
    norm = SimpleRMSNorm(hidden, eps=1e-5).to(device=device, dtype=dtype)
    linear = nn.Linear(hidden, out, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)

    W_new, b_new, h, eps = compute_fused_weights_rmsnorm(norm, linear)
    fused_cls = FusedRMSNormLinearV3 if variant == "V3" else FusedRMSNormLinearV1
    fused = fused_cls(W_new.to(device), b_new.to(device), h, eps)

    with torch.no_grad():
        baseline = linear(norm(x)).float()
        actual = fused(x).float()
        diff = (baseline - actual).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())

    # BF16 matmul paths may differ slightly; these thresholds are tighter than
    # the inherited BF16 tests and match the Kimi synthetic path.
    threshold = 2e-2 if dtype is torch.bfloat16 else 1e-4
    status = "PASS" if max_diff <= threshold else "FAIL"
    print(
        f"{status}: {config_name} {variant} dtype={dtype} "
        f"shape=({tokens}, {hidden})->{out} "
        f"max_diff={max_diff:.6e} mean_diff={mean_diff:.6e}"
    )
    if status != "PASS":
        raise AssertionError(f"{config_name} {variant} max_diff {max_diff} > {threshold}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}")

    for config_name in ("kimi_q_b", "kimi_kv_b"):
        for variant in ("V1", "V3"):
            _check_pair(config_name, variant, torch.bfloat16)

    print("All Kimi K2.6 RMSNorm fusion kernel tests passed.")


if __name__ == "__main__":
    main()
