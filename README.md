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
| `q_a_layernorm -> q_b_proj` | `q_a_layernorm = Identity`, `q_b_proj = FusedRMSNormLinear(...)` |
| `kv_a_layernorm -> kv_b_proj` | `kv_a_layernorm = Identity`, `kv_b_proj = FusedRMSNormLinear(...)` |

Current H100 synthetic results show that fusion should be applied
conditionally. It is strong for the 512-token q/kv projection benchmarks, close
to neutral for long q, and slower for long kv.

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
export TORCH_CUDA_ARCH_LIST=9.0
```

If that path does not exist:

```bash
ls /usr/local
export CUDA_HOME=/usr/local/cuda
```

## Correctness

Run the Kimi-specific CUDA correctness tests:

```bash
python3 -m src.test_kimi26_kernels
```

The first run JIT-builds the CUDA extension. The inherited upstream
`src.test_correctness` also runs old OPT/Llama/GPT-OSS integration tests and is
not required for this Kimi workflow.

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

For H100, set:

```bash
export TORCH_CUDA_ARCH_LIST=9.0
```

This avoids compiling kernels for every visible architecture.

Benchmark configs:

| Config | Kimi pair | Shape |
| --- | --- | --- |
| `kimi_q_b` | `q_a_layernorm -> q_b_proj` | `(512, 1536) -> 12288` |
| `kimi_kv_b` | `kv_a_layernorm -> kv_b_proj` | `(512, 512) -> 16384` |
| `kimi_q_b_long` | Longer q benchmark | `(4096, 1536) -> 12288` |
| `kimi_kv_b_long` | Longer kv benchmark | `(4096, 512) -> 16384` |

## H100 Benchmark Results

Environment:

| Item | Value |
| --- | --- |
| GPU | `NVIDIA H100 80GB HBM3` |
| Driver CUDA | `13.0` from `nvidia-smi` |
| PyTorch | `2.8.0+cu128` |
| PyTorch CUDA | `12.8` |
| dtype | `bfloat16` |
| `CUDA_HOME` | `/usr/local/cuda-12.8` |
| `TORCH_CUDA_ARCH_LIST` | `9.0` |

Kimi-specific correctness:

| Config | Variant | Shape | Max Diff | Mean Diff | Status |
| --- | --- | --- | ---: | ---: | --- |
| `kimi_q_b` | `V1` | `(512, 1536) -> 12288` | `3.125000e-02` | `1.097941e-03` | PASS |
| `kimi_q_b` | `V3` | `(512, 1536) -> 12288` | `3.125000e-02` | `1.097941e-03` | PASS |
| `kimi_kv_b` | `V1` | `(512, 512) -> 16384` | `1.562500e-02` | `1.127254e-03` | PASS |
| `kimi_kv_b` | `V3` | `(512, 512) -> 16384` | `1.562500e-02` | `1.127254e-03` | PASS |

Synthetic benchmark results:

| Config | Variant | Shape | Baseline | Fused | Speedup | Max Diff | Mean Diff |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `kimi_q_b` | `V1` | `(512, 1536) -> 12288` | `0.0628 ms` | `0.0378 ms` | `1.660x` | `3.125000e-02` | `1.097941e-03` |
| `kimi_kv_b` | `V1` | `(512, 512) -> 16384` | `0.0530 ms` | `0.0269 ms` | `1.971x` | `1.562500e-02` | `1.127254e-03` |
| `kimi_q_b_long` | `V1` | `(4096, 1536) -> 12288` | `0.2992 ms` | `0.2858 ms` | `1.047x` | `3.125000e-02` | `1.107869e-03` |
| `kimi_kv_b_long` | `V1` | `(4096, 512) -> 16384` | `0.1546 ms` | `0.2057 ms` | `0.752x` | `3.125000e-02` | `1.147462e-03` |
| `kimi_q_b` | `V3` | `(512, 1536) -> 12288` | `0.0631 ms` | `0.0383 ms` | `1.648x` | `3.125000e-02` | `1.097941e-03` |
| `kimi_kv_b` | `V3` | `(512, 512) -> 16384` | `0.0666 ms` | `0.0274 ms` | `2.429x` | `1.562500e-02` | `1.127254e-03` |
| `kimi_q_b_long` | `V3` | `(4096, 1536) -> 12288` | `0.3000 ms` | `0.2900 ms` | `1.035x` | `3.125000e-02` | `1.107869e-03` |
| `kimi_kv_b_long` | `V3` | `(4096, 512) -> 16384` | `0.1540 ms` | `0.2050 ms` | `0.751x` | `3.125000e-02` | `1.147462e-03` |

Interpretation:

| Scenario | Recommendation |
| --- | --- |
| 512-token `q_b_proj` | Fuse; both V1 and V3 are about `1.65x` faster |
| 512-token `kv_b_proj` | Fuse; V3 reached `2.429x`, V1 reached `1.971x` |
| 4096-token `q_b_proj` | Optional; speedup is small at `1.035x-1.047x` |
| 4096-token `kv_b_proj` | Do not fuse with current kernel; it is about `25%` slower |

Current best policy:

```text
Use fused q_b_proj for short and medium token counts.
Use fused kv_b_proj for short and medium token counts.
Do not blindly patch long kv_b paths until a better kernel/heuristic is added.
```

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
export TORCH_CUDA_ARCH_LIST=9.0
python3 -m src.test_kimi26_kernels
bash scripts/run_kimi26_cuda_benchmarks.sh
```

Paste the outputs from:

```text
results/kimi26/correctness.txt
results/kimi26/kimi_q_b_v1.txt
results/kimi26/kimi_kv_b_v1.txt
results/kimi26/kimi_q_b_long_v1.txt
results/kimi26/kimi_kv_b_long_v1.txt
results/kimi26/kimi_q_b_v3.txt
results/kimi26/kimi_kv_b_v3.txt
results/kimi26/kimi_q_b_long_v3.txt
results/kimi26/kimi_kv_b_long_v3.txt
```

