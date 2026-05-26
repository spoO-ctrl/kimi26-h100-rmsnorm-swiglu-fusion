# Kimi K2.6 CUDA RMSNorm Fusion

This repo contains the Kimi K2.6-specific CUDA fusion implementation work derived from the original H100 RMSNorm/SwiGLU fusion repo.

The focus here is no longer generic LayerNorm fusion for LLaMA or GPT-style models. The target is specifically:

```text
moonshotai/Kimi-K2.6
```

The first fusion targets are Kimi’s direct RMSNorm-to-projection pairs:

```text
q_a_layernorm -> q_b_proj
kv_a_layernorm -> kv_b_proj
```

These are safe initial fusion points because the Kimi attention code already uses direct calls:

```text
q = q_b_proj(q_a_layernorm(q_a_proj(hidden_states)))
kv = kv_b_proj(kv_a_layernorm(compressed_kv))
```

The patch preserves Kimi’s original attention forward path by:
- replacing the RMSNorm layer with `Identity`
- replacing the following projection with a fused CUDA RMSNorm+Linear module

---

## Why RMSNorm Instead of LayerNorm

Kimi K2.6 uses `DeepseekV3RMSNorm`. Some module names contain `layernorm`, but the implementation is RMSNorm, not classic LayerNorm.

Verified from `moonshotai/Kimi-K2.6`:

| Field | Value |
| --- | --- |
| Architecture | `KimiK25ForConditionalGeneration` |
| Outer model type | `kimi_k25` |
| Text model type | `kimi_k2` |
| Norm class | `DeepseekV3RMSNorm` |
| `rms_norm_eps` | `1e-5` |
| Hidden size | `7168` |
| q LoRA rank | `1536` |
| kv LoRA rank | `512` |
| Attention heads | `64` |
| qk no-PE head dim | `128` |
| qk RoPE head dim | `64` |
| MoE intermediate size | `2048` |
| Routed experts | `384` |
| Experts per token | `8` |

Kimi RMSNorm math:

```text
RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
```

Fusion:

```text
W_fused = W * gamma
out = Linear(x * inv_rms, W_fused)
```

Do not apply standard LLaMA LayerNorm fusion logic with mean subtraction. Kimi K2.6 uses RMSNorm only.

---

## Fusion Strategy

### Phase 1 — Direct q/kv LoRA Fusion

Patch:

| Original Pair | Fused Replacement |
| --- | --- |
| `q_a_layernorm -> q_b_proj` | `q_a_layernorm = Identity`, `q_b_proj = FusedRMSNormLinear(...)` |
| `kv_a_layernorm -> kv_b_proj` | `kv_a_layernorm = Identity`, `kv_b_proj = FusedRMSNormLinear(...)` |

Current H100 synthetic benchmarks show fusion should be applied conditionally:
- strong gains for shorter q/kv projections
- nearly neutral for long q
- slower for long kv

Expected full-model patch count:

```text
q_a_layernorm_to_q_b_proj: 61
kv_a_layernorm_to_kv_b_proj: 61
```

---

## H100 Benchmark Results

| Config | Variant | Shape | Baseline | Fused | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `kimi_q_b` | `V1` | `(512, 1536) -> 12288` | `0.0628 ms` | `0.0378 ms` | `1.660x` |
| `kimi_kv_b` | `V1` | `(512, 512) -> 16384` | `0.0530 ms` | `0.0269 ms` | `1.971x` |
| `kimi_q_b_long` | `V1` | `(4096, 1536) -> 12288` | `0.2992 ms` | `0.2858 ms` | `1.047x` |
| `kimi_kv_b_long` | `V1` | `(4096, 512) -> 16384` | `0.1546 ms` | `0.2057 ms` | `0.752x` |
| `kimi_q_b` | `V3` | `(512, 1536) -> 12288` | `0.0631 ms` | `0.0383 ms` | `1.648x` |
| `kimi_kv_b` | `V3` | `(512, 512) -> 16384` | `0.0666 ms` | `0.0274 ms` | `2.429x` |
| `kimi_q_b_long` | `V3` | `(4096, 1536) -> 12288` | `0.3000 ms` | `0.2900 ms` | `1.035x` |
| `kimi_kv_b_long` | `V3` | `(4096, 512) -> 16384` | `0.1540 ms` | `0.2050 ms` | `0.751x` |

### Interpretation

| Scenario | Recommendation |
| --- | --- |
| 512-token `q_b_proj` | Fuse; both V1 and V3 are ~`1.65x` faster |
| 512-token `kv_b_proj` | Fuse; V3 reached `2.429x` |
| 4096-token `q_b_proj` | Optional; speedup is small |
| 4096-token `kv_b_proj` | Do not fuse with current kernel |

Current policy:

```text
Use fused q_b_proj for short and medium token counts.
Use fused kv_b_proj for short and medium token counts.
Do not patch long kv_b paths until a better kernel or heuristic is added.
```

---

## End-to-End Status & Findings

End-to-end benchmarking on the full Kimi K2.6 model was not completed because of three main blockers:

1. The released checkpoint uses FP8/INT4 compressed weights. Weight absorption (`W_new = W × gamma`) requires dense BF16 weights and cannot be directly applied to compressed integer formats without dequantization.

2. Kimi K2.6 uses MLA instead of standard Q/K/V projections. The correct fusion targets are:
   - `q_a_layernorm -> q_b_proj`
   - `kv_a_layernorm -> kv_b_proj`

3. MoE routing blocks expert fusion. The router consumes the norm output before expert execution, preventing direct norm-to-expert fusion.

---

## Files Summary

| File | Purpose |
| --- | --- |
| `src/patch_kimi.py` | Patches q_a_layernorm→q_b_proj and kv_a_layernorm→kv_b_proj |
| `src/benchmark_kimi26.py` | Kimi-shaped synthetic CUDA benchmarks |
| `src/benchmark_kimi26_quantized.py` | Quantization-aware INT8 benchmark |
| `src/test_kimi26_kernels.py` | Kimi-specific correctness tests |
| `scripts/run_kimi26_cuda_benchmarks.sh` | Runs benchmarks and saves results |
