"""
Benchmarks for FlashAttention-3 / Hopper-optimized attention (cuDNN SDPA backend).

1. Single-op: Compare SDPA backends (default, flash, cudnn, efficient, math)
   for prefill attention across model configs, sequence lengths, and batch sizes.

2. E2E: Measure prefill throughput (TTFT) on Llama models with different
   attention backends and composability with fused RMSNorm+Linear patches.
"""

import gc
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse helpers from the main benchmark module
from src.benchmark import (
    _get_hardware_info,
    _get_software_info,
    _measure_per_iter,
    _save_json_results,
)

from src.fa3_attention import SDPA_BACKENDS, sdpa_with_backend, sdpa_default


# ---------------------------------------------------------------------------
# Single-op benchmark: compare SDPA backends for prefill attention
# ---------------------------------------------------------------------------

def benchmark_fa3_single_op():
    """Benchmark SDPA backends for prefill attention on H100.

    Compares: default (auto-select), flash (FA-2), cudnn (FA-3/Hopper),
    efficient (xformers-style), math (PyTorch native).
    """
    print("=" * 140)
    print("FA-3 SINGLE-OP BENCHMARK: SDPA Backend Comparison for Prefill Attention")
    print("=" * 140)

    configs = [
        # (label, num_q_heads, num_kv_heads, head_dim, description)
        ("TinyLlama",    32,  4,  64,  "h=2048, GQA group=8"),
        ("Llama-3.2-3B", 24,  8,  128, "h=3072, GQA group=3"),
        ("Llama-3.1-8B", 32,  8,  128, "h=4096, GQA group=4"),
    ]

    batch_sizes = [1, 4, 8, 16, 32]
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    warmup_iters = 100
    measure_iters = 1000
    dtype = torch.bfloat16

    # Backends to compare: "default" uses no forcing
    backends_to_test = ["default", "flash", "cudnn", "efficient"]

    header = f"{'Config':<16} {'q_h':>4} {'kv_h':>4} {'hd':>4} {'batch':>5} {'seq':>5} | "
    header += "  ".join(f"{b:>12}" for b in backends_to_test)
    header += "  cudnn_speedup"
    print(f"\n{header}")
    print("-" * (len(header) + 10))

    json_entries = []

    for label, num_q, num_kv, hd, desc in configs:
        use_gqa = num_kv < num_q

        for batch in batch_sizes:
            for seq_len in seq_lengths:
                # Check memory: rough estimate for q+k+v
                mem_per_tensor = batch * max(num_q, num_kv) * seq_len * hd * 2  # BF16 = 2 bytes
                total_mem_mb = 3 * mem_per_tensor / (1024 * 1024)
                if total_mem_mb > 20000:  # Skip if > 20GB per set of tensors
                    continue

                try:
                    q = torch.randn(batch, num_q, seq_len, hd, device="cuda", dtype=dtype)
                    k = torch.randn(batch, num_kv, seq_len, hd, device="cuda", dtype=dtype)
                    v = torch.randn(batch, num_kv, seq_len, hd, device="cuda", dtype=dtype)
                except torch.cuda.OutOfMemoryError:
                    print(f"{label:<16} {num_q:>4} {num_kv:>4} {hd:>4} {batch:>5} {seq_len:>5} | OOM")
                    torch.cuda.empty_cache()
                    continue

                results = {}

                for backend_name in backends_to_test:
                    try:
                        if backend_name == "default":
                            fn = lambda: sdpa_default(q, k, v, is_causal=True, enable_gqa=use_gqa)
                        else:
                            # Capture backend_name in closure
                            _bn = backend_name
                            fn = lambda _bn=_bn: sdpa_with_backend(
                                q, k, v, backend=_bn, is_causal=True, enable_gqa=use_gqa
                            )

                        mean_ms, std_ms, _ = _measure_per_iter(fn, warmup_iters, measure_iters)
                        results[backend_name] = (mean_ms, std_ms)
                    except Exception as e:
                        results[backend_name] = None

                # Clean up tensors
                del q, k, v
                torch.cuda.empty_cache()

                # Flash as baseline for speedup calculation
                flash_mean = results.get("flash", (None, None))[0] if results.get("flash") else None
                cudnn_mean = results.get("cudnn", (None, None))[0] if results.get("cudnn") else None

                # Format output line
                parts = []
                for bn in backends_to_test:
                    if results.get(bn):
                        mean, std = results[bn]
                        parts.append(f"{mean:>7.4f}ms")
                    else:
                        parts.append(f"{'N/A':>12}")

                cudnn_speedup_str = ""
                if flash_mean and cudnn_mean and cudnn_mean > 0:
                    cudnn_speedup = flash_mean / cudnn_mean
                    cudnn_speedup_str = f"{cudnn_speedup:>7.2f}x"

                print(f"{label:<16} {num_q:>4} {num_kv:>4} {hd:>4} {batch:>5} {seq_len:>5} | "
                      f"{'  '.join(parts)}  {cudnn_speedup_str}")

                # JSON entries for each backend
                for backend_name in backends_to_test:
                    if not results.get(backend_name):
                        continue
                    mean_ms, std_ms = results[backend_name]

                    speedup_info = {}
                    if backend_name != "flash" and flash_mean and flash_mean > 0:
                        speedup_info["vs_flash"] = round(flash_mean / mean_ms, 4)
                        speedup_info["flash_mean_ms"] = round(flash_mean, 6)
                    if backend_name != "default":
                        default_result = results.get("default")
                        if default_result:
                            speedup_info["vs_default"] = round(default_result[0] / mean_ms, 4)
                            speedup_info["default_mean_ms"] = round(default_result[0], 6)

                    json_entries.append({
                        "config": {
                            "benchmark_id": (
                                f"fa3_attn_{label.replace(' ', '_').replace('.', '')}"
                                f"_{backend_name}_bf16_b{batch}_s{seq_len}"
                            ),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "hardware": _get_hardware_info(),
                            "software": _get_software_info(f"SDPA-{backend_name}"),
                            "model": {
                                "name": label,
                                "description": desc,
                                "precision": "BF16",
                            },
                            "workload": {
                                "batch_size": batch,
                                "sequence_length": seq_len,
                                "dimensions": {
                                    "num_q_heads": num_q,
                                    "num_kv_heads": num_kv,
                                    "head_dim": hd,
                                    "group_size": num_q // num_kv,
                                },
                            },
                        },
                        "metrics": {
                            "single_op": {
                                "mean_ms": round(mean_ms, 6),
                                "stddev_ms": round(std_ms, 6),
                                "variant": f"SDPA-{backend_name}",
                            },
                            "speedup": speedup_info,
                        },
                    })

    print()
    _save_json_results(json_entries, prefix="fa3_single_op_bf16")


# ---------------------------------------------------------------------------
# E2E benchmark: prefill throughput (TTFT) with FA-3 attention
# ---------------------------------------------------------------------------

def benchmark_fa3_e2e():
    """End-to-end prefill benchmark: TTFT and throughput on Llama models.

    Compares:
      - Original: unpatched model (PyTorch auto-selects backend)
      - FA3-cuDNN: force cuDNN backend for attention
      - FA3-cuDNN+SwiGLU-V3: cuDNN attention + fused RMSNorm+Linear + GQA decode
    """
    print("=" * 140)
    print("FA-3 END-TO-END BENCHMARK: Prefill Throughput (TTFT)")
    print("=" * 140)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama_fa3 import patch_llama_fa3_only, patch_llama_fa3_with_fused

    models_to_test = [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama", torch.bfloat16),
        ("unsloth/Llama-3.2-3B", "Llama-3.2-3B", torch.bfloat16),
        ("NousResearch/Meta-Llama-3.1-8B", "Llama-3.1-8B", torch.bfloat16),
    ]

    # Varying-length prompts for prefill benchmarking (longer = more prefill work)
    base_prompts = [
        "The future of artificial intelligence is",
        "In a shocking finding, scientists discovered that",
        "The economic impact of climate change will",
        "Recent advances in quantum computing have shown",
        "The most important lesson from history is",
        "Space exploration in the next decade will focus on",
        "The relationship between technology and society has",
        "New research in neuroscience suggests that the brain",
    ]

    # Test configs: (display_name, patch_fn_name)
    test_configs = [
        ("Original",              "none"),
        ("FA3-cuDNN",             "fa3_only"),
        ("FA3-cuDNN+SwiGLU-V3",  "fa3_fused"),
    ]

    # Prefill-focused: vary input lengths and batch sizes
    # Use longer input sequences to stress prefill
    input_lengths = [128, 256, 512, 1024, 2048]
    batch_sizes = [1, 4, 8, 16]
    num_runs = 5
    max_new_tokens = 1  # Only care about prefill (TTFT), generate 1 token

    json_entries = []

    for model_name, label, model_dtype in models_to_test:
        precision_str = "BF16"
        print(f"\n{'='*140}")
        print(f"  {label} ({model_name}) -- {precision_str}")
        print(f"{'='*140}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        all_results = {}

        for display_name, patch_mode in test_configs:
            print(f"\nLoading {label} for {display_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=model_dtype
            ).cuda().eval()

            if patch_mode == "fa3_only":
                print(f"  Patching: force cuDNN SDPA backend...")
                patch_llama_fa3_only(model, backend="cudnn")
            elif patch_mode == "fa3_fused":
                print(f"  Patching: cuDNN attention + SwiGLU-V3 + GQA decode...")
                patch_llama_fa3_with_fused(model, backend="cudnn")

            variant_results = {}

            for input_len in input_lengths:
                for bs in batch_sizes:
                    # Build input of approximately input_len tokens
                    # Repeat base prompt to reach desired length
                    single_prompt = " ".join(base_prompts) * ((input_len // 50) + 1)
                    batch_prompts = [single_prompt] * bs

                    try:
                        inputs = tokenizer(
                            batch_prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=input_len,
                        ).to("cuda")
                    except Exception:
                        continue

                    actual_len = inputs["input_ids"].shape[1]

                    # Check rough memory estimate
                    if actual_len * bs > 100000 and label == "Llama-3.1-8B":
                        del inputs
                        continue

                    gen_kwargs = dict(
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )

                    # Warmup
                    try:
                        with torch.no_grad():
                            _ = model.generate(**inputs, **gen_kwargs)
                        torch.cuda.synchronize()
                    except torch.cuda.OutOfMemoryError:
                        print(f"  OOM: {display_name} batch={bs}, input_len={actual_len}")
                        torch.cuda.empty_cache()
                        del inputs
                        continue

                    # Measure TTFT (time to first token = prefill time)
                    times = []
                    for _ in range(num_runs):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            _ = model.generate(**inputs, **gen_kwargs)
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        times.append(t1 - t0)

                    mean_time = sum(times) / len(times)
                    stddev_time = math.sqrt(
                        sum((t - mean_time) ** 2 for t in times) / len(times)
                    )

                    total_prefill_tokens = bs * actual_len
                    prefill_tps = total_prefill_tokens / mean_time

                    key = (actual_len, bs)
                    variant_results[key] = {
                        "mean_time": mean_time,
                        "stddev_time": stddev_time,
                        "actual_input_len": actual_len,
                        "batch_size": bs,
                        "total_prefill_tokens": total_prefill_tokens,
                        "prefill_tps": prefill_tps,
                        "ttft_ms": mean_time * 1000,
                    }

                    del inputs

            all_results[display_name] = variant_results

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # --- Report ---
        print(f"\n{'─'*140}")
        print(f"  Prefill Results for {label} ({precision_str})")
        print(f"{'─'*140}")

        header = f"  {'InLen':>5} {'Batch':>5} | {'Original':>18} {'TPS':>10}"
        for dn, _ in test_configs[1:]:
            header += f" | {dn:>22} {'TPS':>10} {'Speedup':>8}"
        print(header)
        print(f"  {'-'*130}")

        orig = all_results.get("Original", {})
        for key in sorted(orig.keys()):
            o = orig[key]
            actual_len, bs = key
            line = (f"  {actual_len:>5} {bs:>5} | "
                    f"{o['ttft_ms']:>8.2f}±{o['stddev_time']*1000:>5.2f}ms "
                    f"{o['prefill_tps']:>9.0f}")

            for dn, _ in test_configs[1:]:
                vr = all_results.get(dn, {}).get(key)
                if vr:
                    speedup = vr["prefill_tps"] / o["prefill_tps"] if o["prefill_tps"] > 0 else 0
                    line += (f" | {vr['ttft_ms']:>8.2f}±{vr['stddev_time']*1000:>5.2f}ms "
                             f"{vr['prefill_tps']:>9.0f} {speedup:>7.3f}x")
                else:
                    line += f" | {'N/A':>22} {'':>10} {'':>8}"
            print(line)

        # JSON entries
        for dn, patch_mode in test_configs[1:]:
            for key in sorted(all_results.get(dn, {}).keys()):
                vr = all_results[dn][key]
                actual_len, bs = key
                o = orig.get(key)
                speedup_val = (vr["prefill_tps"] / o["prefill_tps"]
                               if o and o["prefill_tps"] > 0 else 0)

                json_entries.append({
                    "config": {
                        "benchmark_id": (
                            f"fa3_e2e_{label.replace(' ', '_').replace('.', '')}"
                            f"_{dn.replace('+', '_').replace('-', '_')}"
                            f"_bf16_b{bs}_s{actual_len}"
                        ),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(dn),
                        "model": {
                            "name": model_name,
                            "label": label,
                            "precision": precision_str,
                        },
                        "workload": {
                            "batch_size": bs,
                            "input_tokens": actual_len,
                            "output_tokens": max_new_tokens,
                            "sampling": {"temperature": 0, "top_p": 1.0},
                        },
                    },
                    "metrics": {
                        "throughput": {
                            "tokens_per_second": round(vr["prefill_tps"], 1),
                            "prefill_tokens_per_second": round(vr["prefill_tps"], 1),
                        },
                        "latency": {
                            "mean_ms": round(vr["ttft_ms"], 2),
                            "stddev_ms": round(vr["stddev_time"] * 1000, 2),
                            "ttft_ms": round(vr["ttft_ms"], 2),
                        },
                        "speedup": {
                            "vs_baseline": round(speedup_val, 4),
                            "baseline_description": f"Unpatched {label} ({precision_str})",
                        },
                    },
                })

    _save_json_results(json_entries, prefix="fa3_e2e_bf16")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FA-3 / Hopper attention benchmarks")
    parser.add_argument("--single-op", action="store_true",
                        help="Run single-op SDPA backend comparison")
    parser.add_argument("--e2e", action="store_true",
                        help="Run E2E prefill throughput benchmark")
    parser.add_argument("--all", action="store_true",
                        help="Run all FA-3 benchmarks")
    args = parser.parse_args()

    if not any([args.single_op, args.e2e, args.all]):
        args.all = True

    if args.single_op or args.all:
        benchmark_fa3_single_op()

    if args.e2e or args.all:
        benchmark_fa3_e2e()
