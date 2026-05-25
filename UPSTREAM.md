# Upstream Kernel Base

This repo was derived from `h100-kernel-fusion-rmsnorm-swiglu-main`.

The upstream code supplied the CUDA extension, fused RMSNorm epilogue kernels,
SwiGLU kernels, grouped GEMM experiments, and reference patchers for Llama and
GPT-OSS.

The Kimi-specific work in this repo is:

- `README.md`
- `src/patch_kimi.py`
- `src/benchmark_kimi26.py`
- `scripts/run_kimi26_cuda_benchmarks.sh`

Non-Kimi files are retained as kernel infrastructure and implementation
references, not as the target workload.
