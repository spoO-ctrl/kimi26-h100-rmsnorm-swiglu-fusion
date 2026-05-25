#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/kimi26
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

python3 -m src.test_kimi26_kernels | tee results/kimi26/correctness.txt

for variant in V1 V3; do
  python3 -m src.benchmark_kimi26 --config kimi_q_b --variant "${variant}" | tee "results/kimi26/kimi_q_b_${variant,,}.txt"
  python3 -m src.benchmark_kimi26 --config kimi_kv_b --variant "${variant}" | tee "results/kimi26/kimi_kv_b_${variant,,}.txt"
  python3 -m src.benchmark_kimi26 --config kimi_q_b_long --variant "${variant}" | tee "results/kimi26/kimi_q_b_long_${variant,,}.txt"
  python3 -m src.benchmark_kimi26 --config kimi_kv_b_long --variant "${variant}" | tee "results/kimi26/kimi_kv_b_long_${variant,,}.txt"
done

python3 -m src.benchmark_kimi26_quantized --config kimi_q_b | tee results/kimi26/kimi_q_b_int8.txt
python3 -m src.benchmark_kimi26_quantized --config kimi_kv_b | tee results/kimi26/kimi_kv_b_int8.txt
