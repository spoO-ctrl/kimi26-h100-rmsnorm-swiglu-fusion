# Kimi K2.6 CUDA RMSNorm Fusion

This repo is the **Kimi K2.6-specific CUDA fusion implementation track**.

It is derived from the H100 RMSNorm/SwiGLU kernel repo you provided, but the
active target here is no longer Llama, GPT-OSS, or generic LayerNorm fusion.
The target is:

```text
moonshotai/Kimi-K2.6
```

Specifically, this repo focuses first on Kimi's direct RMSNorm-to-projection
pairs:

```text
q_a_layernorm -> q_b_proj
kv_a_layernorm -> kv_b_proj
```

These are the safest first places to fuse because the Kimi model code already
has direct calls:

```text
q = q_b_proj(q_a_layernorm(q_a_proj(hidden_states)))
kv = kv_b_proj(kv_a_layernorm(compressed_kv))
```

The patch keeps Kimi's original attention forward logic intact by replacing the
norm with identity and replacing the following projection with a fused CUDA
RMSNorm+Linear module.

## Why RMSNorm, Not Classic LayerNorm

Actual Kimi K2.6 uses `DeepseekV3RMSNorm`. Some module names contain
`layernorm`, but the implementation is RMSNorm.

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

Do not apply the LLaMA LayerNorm formula with mean subtraction to Kimi K2.6.
That would be wrong for this model.

## Kimi-Specific Files

| File | Purpose |
| --- | --- |
| `src/patch_kimi.py` | Patches actual Kimi q/kv LoRA RMSNorm projection pairs |
| `src/benchmark_kimi26.py` | Kimi-shaped CUDA benchmarks for q/kv direct pairs |
| `scripts/run_kimi26_cuda_benchmarks.sh` | Runs all Kimi synthetic CUDA benchmarks |
| `README.md` | This Kimi-specific guide |

The inherited files in `csrc/` and `src/fused_forward.py` provide the CUDA
extension and fused RMSNorm modules. Llama/GPT-OSS patchers are retained only as
reference implementations for the kernel infrastructure.

## Kimi Fusion Strategy

### Phase 1: Direct q/kv LoRA Fusion

Patch these:

| Original Kimi pair | Fused replacement |
| --- | --- |
| `q_a_layernorm -> q_b_proj` | `q_a_layernorm = Identity`, `q_b_proj = FusedRMSNormLinearV3(...)` |
| `kv_a_layernorm -> kv_b_proj` | `kv_a_layernorm = Identity`, `kv_b_proj = FusedRMSNormLinearV3(...)` |

Expected full-model patch count:

```text
q_a_layernorm_to_q_b_proj: 61
kv_a_layernorm_to_kv_b_proj: 61
```

### Phase 2: Measure End-To-End

Once full Kimi K2.6 is loaded on a large multi-GPU setup, compare:

```text
baseline logits/generation
patched logits/generation
tokens/sec
prefill latency
decode latency
GPU memory
```

### Phase 3: MoE Investigation

Do not start by fusing `post_attention_layernorm` into experts. Kimi's MoE path
has routing between the norm output and expert execution. Router-aware fusion or
grouped GEMM may help later, but q/kv LoRA fusion is the safer first target.

## RunPod Setup

Use this repo on RunPod:

```text
kimi26-h100-rmsnorm-swiglu-fusion
```

Recommended GPUs:

| Goal | GPU |
| --- | --- |
| Compile and synthetic benchmarks | RTX 4090, L40S, A100, H100 |
| Match upstream kernel tuning | H100 80GB |
| Full Kimi K2.6 experiments | Multi-GPU, large RAM, model-serving stack |

On RunPod:

```bash
cd /workspace
git clone YOUR_GITHUB_REPO_URL
cd kimi26-h100-rmsnorm-swiglu-fusion
```

Verify CUDA:

```bash
nvidia-smi

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("cuda version:", torch.version.cuda)
PY
```

Install any missing dependencies:

```bash
pip install -U transformers accelerate ninja packaging
```

Set CUDA path:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
```

If that path does not exist:

```bash
ls /usr/local
export CUDA_HOME=/usr/local/cuda
```

## Correctness

Run the inherited CUDA correctness tests:

```bash
python3 -m src.test_correctness
```

The first run JIT-builds the CUDA extension.

## Kimi Synthetic Benchmarks

Run all Kimi-shaped benchmarks:

```bash
bash scripts/run_kimi26_cuda_benchmarks.sh
```

Or run individually:

```bash
python3 -m src.benchmark_kimi26 --config kimi_q_b --variant V3
python3 -m src.benchmark_kimi26 --config kimi_kv_b --variant V3
python3 -m src.benchmark_kimi26 --config kimi_q_b_long --variant V3
python3 -m src.benchmark_kimi26 --config kimi_kv_b_long --variant V3
```

Benchmark configs:

| Config | Kimi pair | Shape |
| --- | --- | --- |
| `kimi_q_b` | `q_a_layernorm -> q_b_proj` | `(512, 1536) -> 12288` |
| `kimi_kv_b` | `kv_a_layernorm -> kv_b_proj` | `(512, 512) -> 16384` |
| `kimi_q_b_long` | Longer q benchmark | `(4096, 1536) -> 12288` |
| `kimi_kv_b_long` | Longer kv benchmark | `(4096, 512) -> 16384` |

Save environment:

```bash
mkdir -p results/kimi26
nvidia-smi | tee results/kimi26/nvidia_smi.txt

python3 - <<'PY' | tee results/kimi26/env.txt
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("cuda version:", torch.version.cuda)
PY
```

## Verify Actual Kimi Code

Download code/config only, not full weights:

```bash
mkdir -p /workspace/kimi26_code

hf download moonshotai/Kimi-K2.6 \
  --local-dir /workspace/kimi26_code \
  --include config.json \
  --include "*.py"
```

Inspect the normalization/projection sites:

```bash
grep -n "class .*Norm\|RMSNorm\|q_a_layernorm\|kv_a_layernorm\|q_b_proj\|kv_b_proj" \
  /workspace/kimi26_code/modeling_deepseek.py
```

Expected evidence includes:

```text
class DeepseekV3RMSNorm(nn.Module)
self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
q = self.q_b_proj(self.q_a_layernorm(...))
kv = self.kv_b_proj(self.kv_a_layernorm(...))
```

## Patch Actual Kimi K2.6

Only do this after you can load Kimi K2.6 in your serving/inference stack.

```python
from src.patch_kimi import patch_kimi_q_lora_norms

model.eval()
counts = patch_kimi_q_lora_norms(model, device="cuda", variant="V3")
print(counts)
```

Expected:

```text
{'q_a_layernorm_to_q_b_proj': 61, 'kv_a_layernorm_to_kv_b_proj': 61}
```

Then verify against the unpatched model:

```text
same prompt
same dtype
same seed/settings
compare logits max/mean diff
compare generated tokens
compare tokens/sec
```

## What To Run First

On a fresh RunPod:

```bash
cd /workspace/kimi26-h100-rmsnorm-swiglu-fusion
export CUDA_HOME=/usr/local/cuda-12.8
python3 -m src.test_correctness
bash scripts/run_kimi26_cuda_benchmarks.sh
```

Paste the outputs from:

```text
results/kimi26/correctness.txt
results/kimi26/kimi_q_b_v3.txt
results/kimi26/kimi_kv_b_v3.txt
results/kimi26/kimi_q_b_long_v3.txt
results/kimi26/kimi_kv_b_long_v3.txt
```

## Current Status

Implemented:

- Kimi-specific README
- Kimi q/kv LoRA RMSNorm patcher
- Kimi-shaped CUDA synthetic benchmarks
- RunPod benchmark script

Next:

- Run CUDA benchmarks on H100 or 4090
- Validate actual Kimi patch counts
- Integrate into a full Kimi serving stack only after synthetic results look good
