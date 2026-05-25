"""
Patch Kimi K2.6 attention LoRA projection norms with fused RMSNorm+Linear.

Kimi K2.6 uses DeepseekV3RMSNorm even though several module names contain
"layernorm". The attention code has direct pairs like:

    q = q_b_proj(q_a_layernorm(q_a_proj(hidden_states)))
    kv = kv_b_proj(kv_a_layernorm(kv_a_proj(...)))

This patch keeps the original Kimi forward code intact by replacing:

    q_a_layernorm -> Identity
    q_b_proj      -> FusedRMSNormLinearV3(q_a_layernorm, q_b_proj)

and the same for kv. It is intentionally conservative: it does not patch the
main input_layernorm or post_attention_layernorm paths, because those feed
attention branches, RoPE/cache logic, and MoE routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.fused_forward import FusedRMSNormLinearV1, FusedRMSNormLinearV3
from src.weight_transform import compute_fused_weights_rmsnorm


_VARIANT_CLASSES = {
    "V1": FusedRMSNormLinearV1,
    "V3": FusedRMSNormLinearV3,
}


class _AlreadyPatchedIdentity(nn.Identity):
    """Marker identity so repeated patch calls can skip already-patched pairs."""


def _resolve_text_layers(model) -> list[nn.Module]:
    """Find decoder layers across common HF remote-code nesting patterns."""
    candidates = [
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("text_model", "layers"),
        ("model", "model", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, (list, nn.ModuleList)):
            return list(obj)
    raise AttributeError(
        "Could not find Kimi decoder layers. Inspect model.named_modules() and "
        "update _resolve_text_layers for this model wrapper."
    )


def _patch_norm_linear_pair(
    owner: nn.Module,
    norm_name: str,
    linear_name: str,
    *,
    device: torch.device,
    variant: str,
) -> bool:
    norm = getattr(owner, norm_name, None)
    linear = getattr(owner, linear_name, None)
    if norm is None or linear is None:
        return False
    if isinstance(norm, _AlreadyPatchedIdentity):
        return False
    if not hasattr(norm, "weight") or not isinstance(linear, nn.Linear):
        return False

    W_new, b_new, h, eps = compute_fused_weights_rmsnorm(norm, linear)
    fused_cls = _VARIANT_CLASSES[variant]
    fused = fused_cls(W_new.to(device), b_new.to(device), h, eps)

    setattr(owner, linear_name, fused)
    setattr(owner, norm_name, _AlreadyPatchedIdentity())
    return True


def patch_kimi_q_lora_norms(model, device=None, variant: str = "V3") -> dict[str, int]:
    """Patch Kimi K2.6 q_a/k_a LoRA RMSNorm -> projection pairs in-place.

    Args:
        model: Kimi K2.6 HF model or text model.
        device: target device; defaults to the first model parameter's device.
        variant: "V1" or "V3" CUDA denominator kernel.

    Returns:
        Counts of patched q and kv pairs.
    """
    if variant not in _VARIANT_CLASSES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(_VARIANT_CLASSES)}")
    if device is None:
        device = next(model.parameters()).device

    counts = {"q_a_layernorm_to_q_b_proj": 0, "kv_a_layernorm_to_kv_b_proj": 0}
    layers = _resolve_text_layers(model)

    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue

        if _patch_norm_linear_pair(
            attn,
            "q_a_layernorm",
            "q_b_proj",
            device=device,
            variant=variant,
        ):
            counts["q_a_layernorm_to_q_b_proj"] += 1

        if _patch_norm_linear_pair(
            attn,
            "kv_a_layernorm",
            "kv_b_proj",
            device=device,
            variant=variant,
        ):
            counts["kv_a_layernorm_to_kv_b_proj"] += 1

    return counts


def print_kimi_patchable_pairs(model) -> None:
    """Print layers containing the direct Kimi RMSNorm -> Linear pairs."""
    layers = _resolve_text_layers(model)
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        pairs = []
        if hasattr(attn, "q_a_layernorm") and hasattr(attn, "q_b_proj"):
            pairs.append("q_a_layernorm -> q_b_proj")
        if hasattr(attn, "kv_a_layernorm") and hasattr(attn, "kv_b_proj"):
            pairs.append("kv_a_layernorm -> kv_b_proj")
        if pairs:
            print(f"layer {idx}: " + ", ".join(pairs))
