# Kimi K2.6 H100 RMSNorm Fusion Plan

This repo is derived from `h100-kernel-fusion-rmsnorm-swiglu-main` and adapts
its CUDA RMSNorm fusion path to actual Kimi K2.6.

## Why This Repo

The earlier `kimi26-rmsnorm-fusion` repo proves the math in plain PyTorch.
This repo is the next step: it uses the H100 CUDA kernels for fused RMSNorm
normalization epilogues and model patching.

Actual Kimi K2.6 uses `DeepseekV3RMSNorm`, even though module names contain
`layernorm`. Verified from `moonshotai/Kimi-K2.6`:

| Field | Value |
| --- | --- |
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

## First Safe Kimi Targets

Kimi attention has direct pairs:

```text
q_b_proj(q_a_layernorm(q_a_proj(hidden_states)))
kv_b_proj(kv_a_layernorm(compressed_kv))
```

The conservative patch in `src/patch_kimi.py` replaces:

```text
q_a_layernorm -> Identity
q_b_proj      -> FusedRMSNormLinearV3(q_a_layernorm, q_b_proj)

kv_a_layernorm -> Identity
kv_b_proj      -> FusedRMSNormLinearV3(kv_a_layernorm, kv_b_proj)
```

This keeps Kimi's remote-code attention forward unchanged. Do not start by
patching `input_layernorm` or `post_attention_layernorm`; those paths feed
attention branches, RoPE/cache logic, and MoE routing.

## RunPod Setup

Use an H100 if possible because the upstream repo was tuned for H100. A 4090 can
still validate correctness and compile the CUDA extension.

```bash
cd /workspace/kimi26-h100-rmsnorm-swiglu-fusion

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
PY
```

Install dependencies if needed:

```bash
pip install -U torch transformers accelerate ninja packaging
export CUDA_HOME=/usr/local/cuda-12.8
```

## CUDA Correctness

```bash
python3 -m src.test_correctness
```

The first run JIT-builds the CUDA extension.

## Kimi Synthetic Benchmarks

Run all Kimi-shaped direct-pair benchmarks:

```bash
bash scripts/run_kimi26_cuda_benchmarks.sh
```

Or run one at a time:

```bash
python3 -m src.benchmark_kimi26 --config kimi_q_b --variant V3
python3 -m src.benchmark_kimi26 --config kimi_kv_b --variant V3
python3 -m src.benchmark_kimi26 --config kimi_q_b_long --variant V3
python3 -m src.benchmark_kimi26 --config kimi_kv_b_long --variant V3
```

The synthetic configs map to Kimi's direct LoRA norm-projection pairs:

| Config | Meaning | Shape |
| --- | --- | --- |
| `kimi_q_b` | `q_a_layernorm -> q_b_proj` | `(512, 1536) -> 12288` |
| `kimi_kv_b` | `kv_a_layernorm -> kv_b_proj` | `(512, 512) -> 16384` |
| `kimi_q_b_long` | Longer q benchmark | `(4096, 1536) -> 12288` |
| `kimi_kv_b_long` | Longer kv benchmark | `(4096, 512) -> 16384` |

## Patch Actual Kimi K2.6

Full Kimi K2.6 requires a large multi-GPU setup. Once loaded, patch only the
direct q/kv LoRA RMSNorm pairs:

```python
from src.patch_kimi import patch_kimi_q_lora_norms

model.eval()
counts = patch_kimi_q_lora_norms(model, device="cuda", variant="V3")
print(counts)
```

Expected counts on the full text stack:

```text
q_a_layernorm_to_q_b_proj: 61
kv_a_layernorm_to_kv_b_proj: 61
```

Then compare generation logits against the unpatched model on the same prompt.

## Why Not Fuse Kimi MoE First

Kimi's MoE path has routing between the layer norm output and expert execution.
Fusing `post_attention_layernorm` directly into expert weights would change the
router input unless the router is also handled. The safer path is:

1. q/kv LoRA norm-projection fusion.
2. Measure end-to-end effect.
3. Then investigate MoE grouped GEMM/router-aware fusion.

## Recommendation

Use this repo, not the earlier PyTorch-only prototype, for GPU speedup work.
Keep the earlier repo as a correctness/math reference.
