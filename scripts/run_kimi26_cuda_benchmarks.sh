#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/kimi26

python3 -m src.test_correctness | tee results/kimi26/correctness.txt
python3 -m src.benchmark_kimi26 --config kimi_q_b --variant V3 | tee results/kimi26/kimi_q_b_v3.txt
python3 -m src.benchmark_kimi26 --config kimi_kv_b --variant V3 | tee results/kimi26/kimi_kv_b_v3.txt
python3 -m src.benchmark_kimi26 --config kimi_q_b_long --variant V3 | tee results/kimi26/kimi_q_b_long_v3.txt
python3 -m src.benchmark_kimi26 --config kimi_kv_b_long --variant V3 | tee results/kimi26/kimi_kv_b_long_v3.txt
