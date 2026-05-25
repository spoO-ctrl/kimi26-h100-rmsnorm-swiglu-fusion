"""
Performance benchmarks for fused LayerNorm+Linear.

1. Single-operation benchmark: isolated LN+Linear vs fused for various configs
2. End-to-end benchmark: token generation throughput on OPT models
"""

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

from src.load_cuda import denominator_cuda
from src.weight_transform import compute_fused_weights, compute_fused_weights_rmsnorm, compute_fused_weights_rmsnorm_combined
from src.fused_forward import (
    fused_ln_linear_forward,
    fused_ln_linear_forward_v1,
    fused_ln_linear_forward_v3,
    fused_rmsnorm_linear_forward_v1,
    fused_rmsnorm_linear_forward_v3,
    FusedRMSNormLinearV1,
    FusedRMSNormLinearV3,
    FusedRMSNormCombinedLinearV1,
    FusedRMSNormCombinedLinearV3,
    FusedRMSNormSwiGLUV1,
    FusedRMSNormSwiGLUV3,
)

# Transformer Engine is an optional dependency for baseline comparison
_HAS_TE = False
try:
    import ctypes
    import glob as _glob
    import site as _site
    # TE needs cuDNN/NCCL at runtime; pre-load all .so from pip-installed nvidia packages
    _sp = _site.getsitepackages()[0] if _site.getsitepackages() else ""
    for _pkg_dir in ["nvidia/cudnn/lib", "nvidia/nccl/lib"]:
        _lib_dir = os.path.join(_sp, _pkg_dir)
        if os.path.isdir(_lib_dir):
            for _so in sorted(_glob.glob(os.path.join(_lib_dir, "*.so.*"))):
                try:
                    ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    import transformer_engine.pytorch as te
    _HAS_TE = True
except (ImportError, OSError):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_hardware_info():
    """Gather hardware metadata for JSON output."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
    try:
        import subprocess
        cpu_info = subprocess.check_output("lscpu | head -20", shell=True, text=True)
        cpu_model = [l.split(":")[1].strip() for l in cpu_info.splitlines() if "Model name" in l]
        cpu_model = cpu_model[0] if cpu_model else platform.processor()
    except Exception:
        cpu_model = platform.processor()
    try:
        import subprocess
        mem_total = subprocess.check_output("free -g | awk '/Mem:/{print $2}'", shell=True, text=True).strip()
        host_ram_gb = int(mem_total)
    except Exception:
        host_ram_gb = 0
    return {
        "gpu_vendor": "NVIDIA",
        "gpu_model": gpu_name,
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_gb": gpu_mem_gb,
        "host_cpu": cpu_model,
        "host_ram_gb": host_ram_gb,
    }


def _get_software_info(variant="N/A"):
    """Gather software metadata for JSON output."""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "driver": f"CUDA {torch.version.cuda}" if torch.version.cuda else "N/A",
        "framework": "PyTorch",
        "framework_version": torch.__version__,
        "runtime": "custom CUDA kernel",
        "runtime_version": variant,
    }


def _measure_per_iter(fn, warmup_iters=100, measure_iters=1000):
    """
    Measure per-iteration GPU time using pre-allocated CUDA events.

    Returns (mean_ms, stddev_ms, times_ms_list).
    """
    # Warmup
    for _ in range(warmup_iters):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()

    # Pre-allocate events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]

    for i in range(measure_iters):
        start_events[i].record()
        with torch.no_grad():
            fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(measure_iters)]

    mean_ms = sum(times) / len(times)
    variance = sum((t - mean_ms) ** 2 for t in times) / len(times)
    stddev_ms = math.sqrt(variance)
    return mean_ms, stddev_ms, times


def _infer_benchmark_type(entry):
    """Infer 'single_op' or 'e2e' from metrics keys."""
    metrics = entry.get("metrics", {})
    if "single_op" in metrics:
        return "single_op"
    if "throughput" in metrics:
        return "e2e"
    return "unknown"


def _infer_benchmark_category(entry):
    """Infer category from benchmark_id prefix and precision."""
    bid = entry["config"]["benchmark_id"]
    precision = entry["config"]["model"].get("precision", "").upper()
    if bid.startswith("fused_ln_linear"):
        return "opt_ln"
    if bid.startswith("llama_"):
        return "llama_rmsnorm_bf16" if precision == "BF16" else "llama_rmsnorm"
    if bid.startswith("gpt_oss_"):
        return "gpt_oss"
    if bid.startswith("swiglu_"):
        return "swiglu"
    if bid.startswith("gqa_decode") or bid.startswith("gqa_e2e"):
        return "gqa_decode"
    if bid.startswith("fp8_"):
        return "fp8"
    if bid.startswith("fa3_"):
        return "fa3_attention"
    if bid.startswith("moe_"):
        return "moe"
    return "unknown"


def _append_to_consolidated(results_dir, new_entries):
    """Append entries to all_benchmarks.json, deduplicating by benchmark_id."""
    consolidated_path = os.path.join(results_dir, "all_benchmarks.json")

    # Load existing consolidated data
    existing = []
    if os.path.exists(consolidated_path):
        with open(consolidated_path) as f:
            existing = json.load(f)

    # Index existing by benchmark_id
    by_id = {e["config"]["benchmark_id"]: e for e in existing}

    # Merge new entries (overwrite if same benchmark_id, since new is later)
    for entry in new_entries:
        enriched = dict(entry)
        enriched["benchmark_type"] = _infer_benchmark_type(entry)
        enriched["benchmark_category"] = _infer_benchmark_category(entry)
        by_id[entry["config"]["benchmark_id"]] = enriched

    merged = sorted(by_id.values(),
                    key=lambda e: (e.get("benchmark_category", ""), e["config"]["benchmark_id"]))

    with open(consolidated_path, "w") as f:
        json.dump(merged, f, indent=2)


def _save_json_results(results_data, prefix="benchmark"):
    """Write structured JSON results to results/ directory."""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{prefix}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {filepath}")

    # Also append to consolidated file
    entries = results_data if isinstance(results_data, list) else [results_data]
    _append_to_consolidated(results_dir, entries)
    print(f"Updated {os.path.join(results_dir, 'all_benchmarks.json')}")

    return filepath


def load_results(path):
    """Load benchmark results from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single-operation benchmark
# ---------------------------------------------------------------------------

def benchmark_single_op():
    """Benchmark individual LN+Linear vs all fused variants for various dimension configs."""
    print("=" * 120)
    print("SINGLE-OPERATION BENCHMARK: LayerNorm+Linear vs Fused Variants")
    print("=" * 120)

    configs = [
        # (label, h, out_dim, batch_sizes)
        ("OPT-125m attn",  768,   768,  [1, 32, 128, 512]),
        ("OPT-125m FFN",   768,  3072,  [1, 32, 128, 512]),
        ("OPT-1.3b attn", 2048,  2048,  [1, 32, 128, 512]),
        ("OPT-1.3b FFN",  2048,  8192,  [1, 32, 128, 512]),
        ("OPT-6.7b attn", 4096,  4096,  [1, 32, 128, 512]),
        ("OPT-6.7b FFN",  4096, 16384,  [1, 32, 128, 512]),
    ]

    warmup_iters = 100
    measure_iters = 1000
    denom_stream = torch.cuda.Stream()
    # Pre-allocate events for V0
    input_ready_evt = torch.cuda.Event(enable_timing=False)
    denom_done_evt = torch.cuda.Event(enable_timing=False)

    te_header = f" {'TE-Baseline':>14}" if _HAS_TE else ""
    print(f"\n{'Config':<18} {'h':>5} {'out':>6} {'batch':>6} | "
          f"{'Original':>14} {'V0-Stream':>14} {'V1-Fused':>14} {'V2-Welford':>14} {'V3-Combined':>14}{te_header}")
    print("-" * (120 + (15 if _HAS_TE else 0)))

    results = []
    json_entries = []

    for label, h, out_dim, batch_sizes in configs:
        ln = nn.LayerNorm(h).cuda()
        linear = nn.Linear(h, out_dim).cuda()
        nn.init.normal_(ln.weight, mean=1.0, std=0.1)
        nn.init.normal_(ln.bias, mean=0.0, std=0.01)

        W_new, b_new, h_dim, eps = compute_fused_weights(ln, linear)

        for batch in batch_sizes:
            x = torch.randn(batch, h, device="cuda")

            # --- Measure original ---
            orig_mean, orig_std, _ = _measure_per_iter(
                lambda: linear(ln(x)), warmup_iters, measure_iters)

            # --- Measure V0 (stream-based) ---
            v0_mean, v0_std, _ = _measure_per_iter(
                lambda: fused_ln_linear_forward(x, W_new, b_new, denom_stream, h_dim, eps, input_ready_evt, denom_done_evt),
                warmup_iters, measure_iters)

            # --- Measure V1 (fused normalize, no streams) ---
            v1_mean, v1_std, _ = _measure_per_iter(
                lambda: fused_ln_linear_forward_v1(x, W_new, b_new, h_dim, eps),
                warmup_iters, measure_iters)

            # --- Measure V2 (Welford denom, stream-based forward) ---
            def fused_v2():
                x_2d = x.reshape(-1, x.size(-1))
                default_stream = torch.cuda.current_stream()
                input_ready_evt.record(default_stream)
                denom_stream.wait_event(input_ready_evt)
                with torch.cuda.stream(denom_stream):
                    v = denominator_cuda.compute_denominator_welford(x_2d)
                raw_output = F.linear(x_2d, W_new)
                denom_done_evt.record(denom_stream)
                default_stream.wait_event(denom_done_evt)
                std = torch.sqrt(v * v / h_dim + eps)
                return raw_output / std.unsqueeze(-1) + b_new

            v2_mean, v2_std, _ = _measure_per_iter(fused_v2, warmup_iters, measure_iters)

            # --- Measure V3 (Welford + fused normalize + 512 threads) ---
            v3_mean, v3_std, _ = _measure_per_iter(
                lambda: fused_ln_linear_forward_v3(x, W_new, b_new, h_dim, eps),
                warmup_iters, measure_iters)

            # --- Measure TE (Transformer Engine LayerNormLinear baseline) ---
            te_mean, te_std, s_te = None, None, None
            if _HAS_TE:
                te_lnl = te.LayerNormLinear(h, out_dim, eps=eps).cuda()
                # Copy weights from our LN + Linear to match
                with torch.no_grad():
                    te_lnl.layer_norm_weight.copy_(ln.weight)
                    te_lnl.layer_norm_bias.copy_(ln.bias)
                    te_lnl.weight.copy_(linear.weight)
                    te_lnl.bias.copy_(linear.bias)
                # Verify correctness before timing
                with torch.no_grad():
                    te_out = te_lnl(x)
                    ref_out = linear(ln(x))
                    te_diff = (te_out - ref_out).abs().max().item()
                    if te_diff > 0.01:
                        print(f"  WARNING: TE output differs from reference by {te_diff:.2e}")
                te_mean, te_std, _ = _measure_per_iter(
                    lambda: te_lnl(x), warmup_iters, measure_iters)
                s_te = orig_mean / te_mean if te_mean > 0 else float("inf")
                del te_lnl

            s0 = orig_mean / v0_mean if v0_mean > 0 else float("inf")
            s1 = orig_mean / v1_mean if v1_mean > 0 else float("inf")
            s2 = orig_mean / v2_mean if v2_mean > 0 else float("inf")
            s3 = orig_mean / v3_mean if v3_mean > 0 else float("inf")

            result_entry = {
                "label": label, "h": h, "out_dim": out_dim, "batch": batch,
                "orig_ms": orig_mean, "orig_std": orig_std,
                "v0_ms": v0_mean, "v0_std": v0_std, "v0_speedup": s0,
                "v1_ms": v1_mean, "v1_std": v1_std, "v1_speedup": s1,
                "v2_ms": v2_mean, "v2_std": v2_std, "v2_speedup": s2,
                "v3_ms": v3_mean, "v3_std": v3_std, "v3_speedup": s3,
            }
            if _HAS_TE:
                result_entry.update({
                    "te_ms": te_mean, "te_std": te_std, "te_speedup": s_te,
                })
            results.append(result_entry)

            # JSON entry for each variant
            variant_list = [
                ("V0", v0_mean, v0_std, s0), ("V1", v1_mean, v1_std, s1),
                ("V2", v2_mean, v2_std, s2), ("V3", v3_mean, v3_std, s3),
            ]
            if _HAS_TE and te_mean is not None:
                variant_list.append(("TE", te_mean, te_std, s_te))
            for vname, vmean, vstd, vspeed in variant_list:
                json_entries.append({
                    "config": {
                        "benchmark_id": f"fused_ln_linear_single_op_{label.replace(' ', '_')}_{vname}_fp32_{batch}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(vname),
                        "model": {"name": label, "precision": "FP32"},
                        "workload": {
                            "batch_size": batch,
                            "dimensions": {"h": h, "out_dim": out_dim},
                        },
                    },
                    "metrics": {
                        "single_op": {
                            "mean_ms": round(vmean, 6),
                            "stddev_ms": round(vstd, 6),
                            "variant": vname,
                        },
                        "speedup": {
                            "vs_baseline": round(vspeed, 4),
                            "baseline_description": "PyTorch nn.LayerNorm + nn.Linear",
                            "baseline_mean_ms": round(orig_mean, 6),
                            "baseline_stddev_ms": round(orig_std, 6),
                        },
                    },
                })

            def _fmt(mean, std, speedup):
                return f"{mean:>7.4f}±{std:>5.4f}({speedup:.2f}x)"

            te_col = f" {_fmt(te_mean, te_std, s_te)}" if _HAS_TE and te_mean is not None else ""
            print(f"{label:<18} {h:>5} {out_dim:>6} {batch:>6} | "
                  f"{orig_mean:>7.4f}±{orig_std:<5.4f}ms "
                  f"{_fmt(v0_mean, v0_std, s0)} "
                  f"{_fmt(v1_mean, v1_std, s1)} "
                  f"{_fmt(v2_mean, v2_std, s2)} "
                  f"{_fmt(v3_mean, v3_std, s3)}{te_col}")

    print()

    # Save JSON
    _save_json_results(json_entries, prefix="single_op")

    return results


# ---------------------------------------------------------------------------
# End-to-end benchmark
# ---------------------------------------------------------------------------

def benchmark_end_to_end():
    """End-to-end token generation throughput on OPT models with multi-batch support."""
    print("=" * 110)
    print("END-TO-END BENCHMARK: OPT Model Token Generation")
    print("=" * 110)

    from transformers import AutoTokenizer, OPTForCausalLM
    from src.patch_model import patch_opt_model

    models_to_test = [
        ("facebook/opt-1.3b", "OPT-1.3b"),
        ("facebook/opt-6.7b", "OPT-6.7b"),
    ]

    prompts = [
        "The future of artificial intelligence is",
        "In a shocking finding, scientists discovered that",
        "The economic impact of climate change will",
        "Recent advances in quantum computing have shown",
        "The most important lesson from history is",
        "Space exploration in the next decade will focus on",
        "The relationship between technology and society has",
        "New research in neuroscience suggests that the brain",
        "The key to understanding complex systems lies in",
        "Advances in renewable energy technology have made",
        "The intersection of biology and computing creates",
        "Modern cryptography relies fundamentally on the",
        "The philosophical implications of consciousness are",
        "Deep ocean exploration has revealed unexpected",
        "The evolution of programming languages shows that",
        "Quantum entanglement challenges our understanding of",
        "The future of transportation will be shaped by",
        "Breakthroughs in materials science have enabled",
        "The role of microbiomes in human health is",
        "Artificial general intelligence remains a topic of",
        "Climate modeling has become increasingly accurate",
        "The history of mathematics reveals patterns in",
        "Neural network architectures continue to evolve",
        "The economics of space mining could transform",
        "Genetic engineering tools like CRISPR have opened",
        "The social impact of automation extends beyond",
        "Advances in battery technology are crucial for",
        "The study of ancient civilizations reveals that",
        "Protein folding prediction has been revolutionized",
        "The future of education will be transformed by",
        "Sustainable agriculture requires innovative approaches",
        "The development of fusion energy has reached",
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32]
    variants = ["V0", "V1", "V3"]

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )
    num_runs = 5
    json_entries = []

    for model_name, label in models_to_test:
        print(f"\n{'='*110}")
        print(f"  {label} ({model_name})")
        print(f"{'='*110}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # --- Benchmark original model ---
        print(f"\nLoading original {label}...")
        model_orig = OPTForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).cuda().eval()

        orig_results = {}
        for bs in batch_sizes:
            batch_prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to("cuda")

            print(f"  Warming up original model (batch_size={bs})...")
            with torch.no_grad():
                _ = model_orig.generate(**inputs, **gen_kwargs)
            torch.cuda.synchronize()

            print(f"  Benchmarking original model (batch_size={bs})...")
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model_orig.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
            total_tokens = bs * num_new_tokens
            mean_time = sum(times) / len(times)
            stddev_time = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

            # Save output text for batch_size=1 comparison
            orig_text = None
            if bs == 1:
                with torch.no_grad():
                    out_check = model_orig.generate(**inputs, **gen_kwargs)
                orig_text = tokenizer.decode(out_check[0], skip_special_tokens=True)

            orig_results[bs] = {
                "mean_time": mean_time,
                "stddev_time": stddev_time,
                "num_new_tokens": num_new_tokens,
                "total_tokens": total_tokens,
                "tps": total_tokens / mean_time,
                "text": orig_text,
            }

        # Free original model before loading fused variants
        del model_orig
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # --- Benchmark each fused variant ---
        all_variant_results = {}
        for variant in variants:
            print(f"\nLoading fused {label} (variant={variant})...")
            model_fused = OPTForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32
            ).cuda().eval()
            print(f"  Patching model with variant={variant}...")
            patch_opt_model(model_fused, variant=variant)

            variant_results = {}
            for bs in batch_sizes:
                batch_prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
                inputs = tokenizer(
                    batch_prompts, return_tensors="pt", padding=True
                ).to("cuda")

                print(f"  Warming up {variant} model (batch_size={bs})...")
                with torch.no_grad():
                    _ = model_fused.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()

                print(f"  Benchmarking {variant} model (batch_size={bs})...")
                times = []
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model_fused.generate(**inputs, **gen_kwargs)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
                total_tokens = bs * num_new_tokens
                mean_time = sum(times) / len(times)
                stddev_time = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

                fused_text = None
                if bs == 1:
                    with torch.no_grad():
                        out_check = model_fused.generate(**inputs, **gen_kwargs)
                    fused_text = tokenizer.decode(out_check[0], skip_special_tokens=True)

                variant_results[bs] = {
                    "mean_time": mean_time,
                    "stddev_time": stddev_time,
                    "num_new_tokens": num_new_tokens,
                    "total_tokens": total_tokens,
                    "tps": total_tokens / mean_time,
                    "text": fused_text,
                }

                # JSON entry
                o = orig_results[bs]
                json_entries.append({
                    "config": {
                        "benchmark_id": f"fused_ln_linear_{label.replace(' ', '_')}_{variant}_fp32_{bs}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(variant),
                        "model": {
                            "name": model_name,
                            "precision": "FP32",
                            "max_context": 2048,
                        },
                        "workload": {
                            "batch_size": bs,
                            "input_tokens": inputs["input_ids"].shape[1],
                            "output_tokens": num_new_tokens,
                            "sampling": {"temperature": 0, "top_p": 1.0},
                        },
                    },
                    "metrics": {
                        "throughput": {
                            "tokens_per_second": round(variant_results[bs]["tps"], 1),
                        },
                        "latency": {
                            "mean_ms": round(mean_time * 1000, 1),
                            "stddev_ms": round(stddev_time * 1000, 1),
                        },
                        "speedup": {
                            "vs_baseline": round(variant_results[bs]["tps"] / o["tps"], 4),
                            "baseline_description": "PyTorch nn.LayerNorm + nn.Linear",
                        },
                    },
                })

            all_variant_results[variant] = variant_results
            del model_fused
            gc.collect()
            torch.cuda.empty_cache()

        # --- Report ---
        print(f"\n{'─'*120}")
        print(f"  Results for {label}")
        print(f"{'─'*120}")

        # Header
        header = f"  {'Batch':>5} {'Tokens':>7} | {'Original':>16} {'tok/s':>8}"
        for v in variants:
            header += f" | {v:>16} {'tok/s':>8} {'Speedup':>7}"
        print(header)
        print(f"  {'-'*115}")

        for bs in batch_sizes:
            o = orig_results[bs]
            line = (f"  {bs:>5} {o['total_tokens']:>7} | "
                    f"{o['mean_time']*1000:>8.1f}±{o['stddev_time']*1000:>5.1f}ms {o['tps']:>7.1f}")
            for v in variants:
                vr = all_variant_results[v][bs]
                speedup = vr["tps"] / o["tps"]
                line += (f" | {vr['mean_time']*1000:>8.1f}±{vr['stddev_time']*1000:>4.1f}ms "
                         f"{vr['tps']:>7.1f} {speedup:>6.3f}x")
            print(line)

        # Output comparison for batch_size=1
        o_text = orig_results[1]["text"]
        for v in variants:
            f_text = all_variant_results[v][1].get("text")
            if o_text and f_text:
                match = "YES" if o_text == f_text else "NO (expected - floating point differences)"
                print(f"\n  {v} output match (batch_size=1): {match}")

    # Save JSON
    _save_json_results(json_entries, prefix="e2e")


# ---------------------------------------------------------------------------
# Llama single-operation benchmark (separate vs combined)
# ---------------------------------------------------------------------------

def benchmark_llama_single_op():
    """Benchmark RMSNorm+Linear: baseline vs separate fused vs combined fused for Llama configs."""
    print("=" * 130)
    print("LLAMA SINGLE-OPERATION BENCHMARK: RMSNorm+Linear (Separate vs Combined)")
    print("=" * 130)

    configs = [
        # (label, h, out_dims, description)
        ("TinyLlama attn", 2048, [2048, 256, 256], "GQA QKV"),
        ("TinyLlama MLP",  2048, [5632, 5632],     "gate+up"),
        ("Llama-3-8B attn", 4096, [4096, 1024, 1024], "GQA QKV"),
        ("Llama-3-8B MLP",  4096, [14336, 14336],   "gate+up"),
    ]

    batch_sizes = [1, 8, 32, 128, 512]
    warmup_iters = 100
    measure_iters = 1000

    print(f"\n{'Config':<18} {'h':>5} {'out_dims':<20} {'batch':>6} | "
          f"{'Baseline':>14} {'Sep-V1':>14} {'Comb-V1':>14} {'Sep-V3':>14} {'Comb-V3':>14}")
    print("-" * 130)

    json_entries = []

    for label, h, out_dims, desc in configs:
        torch.manual_seed(42)
        rms_norm = torch.nn.RMSNorm(h, eps=1e-6).cuda()
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

        linears = []
        for od in out_dims:
            lin = nn.Linear(h, od, bias=False).cuda()
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            linears.append(lin)

        # Separate fused weights
        sep_weights = []
        for lin in linears:
            W_new, b_new, h_dim, eps = compute_fused_weights_rmsnorm(rms_norm, lin)
            sep_weights.append((W_new, b_new, h_dim, eps))

        # Combined fused weights
        W_comb, b_comb, split_sizes, h_dim, eps = compute_fused_weights_rmsnorm_combined(
            rms_norm, linears
        )

        # Build module instances
        sep_v1_mods = [FusedRMSNormLinearV1(w, b, hd, e) for w, b, hd, e in sep_weights]
        sep_v3_mods = [FusedRMSNormLinearV3(w, b, hd, e) for w, b, hd, e in sep_weights]
        comb_v1 = FusedRMSNormCombinedLinearV1(W_comb, b_comb, split_sizes, h_dim, eps)
        comb_v3 = FusedRMSNormCombinedLinearV3(W_comb, b_comb, split_sizes, h_dim, eps)

        out_dims_str = "+".join(str(d) for d in out_dims)

        for batch in batch_sizes:
            x = torch.randn(batch, h, device="cuda")

            # Baseline: nn.RMSNorm + nn.Linear (separate calls)
            def baseline():
                normed = rms_norm(x)
                return [lin(normed) for lin in linears]
            base_mean, base_std, _ = _measure_per_iter(baseline, warmup_iters, measure_iters)

            # Separate fused V1
            def sep_v1_fn():
                return [m(x) for m in sep_v1_mods]
            sv1_mean, sv1_std, _ = _measure_per_iter(sep_v1_fn, warmup_iters, measure_iters)

            # Combined fused V1
            def comb_v1_fn():
                return comb_v1(x)
            cv1_mean, cv1_std, _ = _measure_per_iter(comb_v1_fn, warmup_iters, measure_iters)

            # Separate fused V3
            def sep_v3_fn():
                return [m(x) for m in sep_v3_mods]
            sv3_mean, sv3_std, _ = _measure_per_iter(sep_v3_fn, warmup_iters, measure_iters)

            # Combined fused V3
            def comb_v3_fn():
                return comb_v3(x)
            cv3_mean, cv3_std, _ = _measure_per_iter(comb_v3_fn, warmup_iters, measure_iters)

            s_sv1 = base_mean / sv1_mean if sv1_mean > 0 else float("inf")
            s_cv1 = base_mean / cv1_mean if cv1_mean > 0 else float("inf")
            s_sv3 = base_mean / sv3_mean if sv3_mean > 0 else float("inf")
            s_cv3 = base_mean / cv3_mean if cv3_mean > 0 else float("inf")

            def _fmt(mean, std, speedup):
                return f"{mean:>7.4f}±{std:>5.4f}({speedup:.2f}x)"

            print(f"{label:<18} {h:>5} {out_dims_str:<20} {batch:>6} | "
                  f"{base_mean:>7.4f}±{base_std:<5.4f}ms "
                  f"{_fmt(sv1_mean, sv1_std, s_sv1)} "
                  f"{_fmt(cv1_mean, cv1_std, s_cv1)} "
                  f"{_fmt(sv3_mean, sv3_std, s_sv3)} "
                  f"{_fmt(cv3_mean, cv3_std, s_cv3)}")

            # JSON entries
            for vname, vmean, vstd, vspeed in [
                ("Sep-V1", sv1_mean, sv1_std, s_sv1),
                ("Comb-V1", cv1_mean, cv1_std, s_cv1),
                ("Sep-V3", sv3_mean, sv3_std, s_sv3),
                ("Comb-V3", cv3_mean, cv3_std, s_cv3),
            ]:
                json_entries.append({
                    "config": {
                        "benchmark_id": f"llama_single_op_{label.replace(' ', '_')}_{vname}_fp32_{batch}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(vname),
                        "model": {"name": label, "description": desc, "precision": "FP32"},
                        "workload": {
                            "batch_size": batch,
                            "dimensions": {"h": h, "out_dims": out_dims},
                        },
                    },
                    "metrics": {
                        "single_op": {
                            "mean_ms": round(vmean, 6),
                            "stddev_ms": round(vstd, 6),
                            "variant": vname,
                        },
                        "speedup": {
                            "vs_baseline": round(vspeed, 4),
                            "baseline_description": "PyTorch nn.RMSNorm + nn.Linear (separate calls)",
                            "baseline_mean_ms": round(base_mean, 6),
                            "baseline_stddev_ms": round(base_std, 6),
                        },
                    },
                })

    print()
    _save_json_results(json_entries, prefix="llama_single_op")


# ---------------------------------------------------------------------------
# Llama end-to-end benchmark
# ---------------------------------------------------------------------------

def benchmark_llama_e2e():
    """End-to-end TinyLlama token generation: original vs separate fused vs combined fused."""
    print("=" * 130)
    print("END-TO-END BENCHMARK: TinyLlama Token Generation (Separate vs Combined)")
    print("=" * 130)

    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama import patch_llama_model

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    prompts = [
        "The future of artificial intelligence is",
        "In a shocking finding, scientists discovered that",
        "The economic impact of climate change will",
        "Recent advances in quantum computing have shown",
        "The most important lesson from history is",
        "Space exploration in the next decade will focus on",
        "The relationship between technology and society has",
        "New research in neuroscience suggests that the brain",
        "The key to understanding complex systems lies in",
        "Advances in renewable energy technology have made",
        "The intersection of biology and computing creates",
        "Modern cryptography relies fundamentally on the",
        "The philosophical implications of consciousness are",
        "Deep ocean exploration has revealed unexpected",
        "The evolution of programming languages shows that",
        "Quantum entanglement challenges our understanding of",
        "The future of transportation will be shaped by",
        "Breakthroughs in materials science have enabled",
        "The role of microbiomes in human health is",
        "Artificial general intelligence remains a topic of",
        "Climate modeling has become increasingly accurate",
        "The history of mathematics reveals patterns in",
        "Neural network architectures continue to evolve",
        "The economics of space mining could transform",
        "Genetic engineering tools like CRISPR have opened",
        "The social impact of automation extends beyond",
        "Advances in battery technology are crucial for",
        "The study of ancient civilizations reveals that",
        "Protein folding prediction has been revolutionized",
        "The future of education will be transformed by",
        "Sustainable agriculture requires innovative approaches",
        "The development of fusion energy has reached",
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # (display_name, variant, combined)
    test_configs = [
        ("Original", None, None),
        ("Sep-V1", "V1", False),
        ("Comb-V1", "V1", True),
        ("Sep-V3", "V3", False),
        ("Comb-V3", "V3", True),
    ]

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )
    num_runs = 5
    json_entries = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}  # display_name -> {bs: {...}}

    for display_name, variant, combined in test_configs:
        print(f"\nLoading TinyLlama for {display_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).cuda().eval()

        if variant is not None:
            print(f"  Patching model (variant={variant}, combined={combined})...")
            patch_llama_model(model, variant=variant, combined=combined)

        variant_results = {}
        for bs in batch_sizes:
            batch_prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to("cuda")

            print(f"  Warming up {display_name} (batch_size={bs})...")
            with torch.no_grad():
                _ = model.generate(**inputs, **gen_kwargs)
            torch.cuda.synchronize()

            print(f"  Benchmarking {display_name} (batch_size={bs})...")
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
            total_tokens = bs * num_new_tokens
            mean_time = sum(times) / len(times)
            stddev_time = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

            variant_results[bs] = {
                "mean_time": mean_time,
                "stddev_time": stddev_time,
                "num_new_tokens": num_new_tokens,
                "total_tokens": total_tokens,
                "tps": total_tokens / mean_time,
            }

        all_results[display_name] = variant_results

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # --- Report ---
    print(f"\n{'─'*130}")
    print(f"  Results for TinyLlama")
    print(f"{'─'*130}")

    header = f"  {'Batch':>5} {'Tokens':>7} | {'Original':>16} {'tok/s':>8}"
    for dn, v, c in test_configs[1:]:
        header += f" | {dn:>16} {'tok/s':>8} {'Speedup':>7}"
    print(header)
    print(f"  {'-'*125}")

    orig = all_results["Original"]
    for bs in batch_sizes:
        o = orig[bs]
        line = (f"  {bs:>5} {o['total_tokens']:>7} | "
                f"{o['mean_time']*1000:>8.1f}±{o['stddev_time']*1000:>5.1f}ms {o['tps']:>7.1f}")
        for dn, v, c in test_configs[1:]:
            vr = all_results[dn][bs]
            speedup = vr["tps"] / o["tps"]
            line += (f" | {vr['mean_time']*1000:>8.1f}±{vr['stddev_time']*1000:>4.1f}ms "
                     f"{vr['tps']:>7.1f} {speedup:>6.3f}x")
        print(line)

    # JSON entries
    for dn, v, c in test_configs[1:]:
        for bs in batch_sizes:
            vr = all_results[dn][bs]
            o = orig[bs]
            json_entries.append({
                "config": {
                    "benchmark_id": f"llama_e2e_TinyLlama_{dn}_fp32_{bs}",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "hardware": _get_hardware_info(),
                    "software": _get_software_info(dn),
                    "model": {
                        "name": model_name,
                        "precision": "FP32",
                    },
                    "workload": {
                        "batch_size": bs,
                        "input_tokens": "variable",
                        "output_tokens": vr["num_new_tokens"],
                        "sampling": {"temperature": 0, "top_p": 1.0},
                    },
                },
                "metrics": {
                    "throughput": {
                        "tokens_per_second": round(vr["tps"], 1),
                    },
                    "latency": {
                        "mean_ms": round(vr["mean_time"] * 1000, 1),
                        "stddev_ms": round(vr["stddev_time"] * 1000, 1),
                    },
                    "speedup": {
                        "vs_baseline": round(vr["tps"] / o["tps"], 4),
                        "baseline_description": "Unpatched TinyLlama",
                    },
                },
            })

    _save_json_results(json_entries, prefix="llama_e2e")


# ---------------------------------------------------------------------------
# Llama BF16 single-operation benchmark
# ---------------------------------------------------------------------------

def benchmark_llama_single_op_bf16():
    """Benchmark RMSNorm+Linear in BF16: baseline vs separate vs combined for Llama-3 configs."""
    print("=" * 130)
    print("LLAMA BF16 SINGLE-OPERATION BENCHMARK: RMSNorm+Linear (Separate vs Combined)")
    print("=" * 130)

    configs = [
        # (label, h, out_dims, description)
        ("Llama-3.2-3B attn", 3072, [3072, 1024, 1024], "GQA QKV"),
        ("Llama-3.2-3B MLP",  3072, [8192, 8192],       "gate+up"),
        ("Llama-3.1-8B attn", 4096, [4096, 1024, 1024], "GQA QKV"),
        ("Llama-3.1-8B MLP",  4096, [14336, 14336],     "gate+up"),
    ]

    batch_sizes = [1, 8, 32, 128, 512]
    warmup_iters = 100
    measure_iters = 1000
    dtype = torch.bfloat16

    print(f"\n{'Config':<20} {'h':>5} {'out_dims':<20} {'batch':>6} | "
          f"{'Baseline':>14} {'Sep-V1':>14} {'Comb-V1':>14} {'Sep-V3':>14} {'Comb-V3':>14}")
    print("-" * 132)

    json_entries = []

    for label, h, out_dims, desc in configs:
        torch.manual_seed(42)
        rms_norm = torch.nn.RMSNorm(h, eps=1e-5).to(device="cuda", dtype=dtype)
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

        linears = []
        for od in out_dims:
            lin = nn.Linear(h, od, bias=False, device="cuda", dtype=dtype)
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            linears.append(lin)

        # Compute fused weights in FP32 for numerical stability, then cast to BF16
        rms_norm_f32 = torch.nn.RMSNorm(h, eps=1e-5).cuda()
        rms_norm_f32.weight.data.copy_(rms_norm.weight.data.float())

        linears_f32 = []
        for lin in linears:
            lin_f32 = nn.Linear(lin.in_features, lin.out_features, bias=False).cuda()
            lin_f32.weight.data.copy_(lin.weight.data.float())
            linears_f32.append(lin_f32)

        sep_weights = []
        for lin_f32 in linears_f32:
            W_new, b_new, h_dim, eps = compute_fused_weights_rmsnorm(rms_norm_f32, lin_f32)
            sep_weights.append((W_new.to(dtype), b_new.to(dtype), h_dim, eps))

        W_comb, b_comb, split_sizes, h_dim, eps = compute_fused_weights_rmsnorm_combined(
            rms_norm_f32, linears_f32
        )
        W_comb = W_comb.to(dtype)
        b_comb = b_comb.to(dtype)

        del rms_norm_f32, linears_f32

        # Build module instances
        sep_v1_mods = [FusedRMSNormLinearV1(w, b, hd, e) for w, b, hd, e in sep_weights]
        sep_v3_mods = [FusedRMSNormLinearV3(w, b, hd, e) for w, b, hd, e in sep_weights]
        comb_v1 = FusedRMSNormCombinedLinearV1(W_comb, b_comb, split_sizes, h_dim, eps)
        comb_v3 = FusedRMSNormCombinedLinearV3(W_comb, b_comb, split_sizes, h_dim, eps)

        out_dims_str = "+".join(str(d) for d in out_dims)

        for batch in batch_sizes:
            x = torch.randn(batch, h, device="cuda", dtype=dtype)

            # Baseline: nn.RMSNorm + nn.Linear (separate calls)
            def baseline():
                normed = rms_norm(x)
                return [lin(normed) for lin in linears]
            base_mean, base_std, _ = _measure_per_iter(baseline, warmup_iters, measure_iters)

            # Separate fused V1
            def sep_v1_fn():
                return [m(x) for m in sep_v1_mods]
            sv1_mean, sv1_std, _ = _measure_per_iter(sep_v1_fn, warmup_iters, measure_iters)

            # Combined fused V1
            def comb_v1_fn():
                return comb_v1(x)
            cv1_mean, cv1_std, _ = _measure_per_iter(comb_v1_fn, warmup_iters, measure_iters)

            # Separate fused V3
            def sep_v3_fn():
                return [m(x) for m in sep_v3_mods]
            sv3_mean, sv3_std, _ = _measure_per_iter(sep_v3_fn, warmup_iters, measure_iters)

            # Combined fused V3
            def comb_v3_fn():
                return comb_v3(x)
            cv3_mean, cv3_std, _ = _measure_per_iter(comb_v3_fn, warmup_iters, measure_iters)

            s_sv1 = base_mean / sv1_mean if sv1_mean > 0 else float("inf")
            s_cv1 = base_mean / cv1_mean if cv1_mean > 0 else float("inf")
            s_sv3 = base_mean / sv3_mean if sv3_mean > 0 else float("inf")
            s_cv3 = base_mean / cv3_mean if cv3_mean > 0 else float("inf")

            def _fmt(mean, std, speedup):
                return f"{mean:>7.4f}±{std:>5.4f}({speedup:.2f}x)"

            print(f"{label:<20} {h:>5} {out_dims_str:<20} {batch:>6} | "
                  f"{base_mean:>7.4f}±{base_std:<5.4f}ms "
                  f"{_fmt(sv1_mean, sv1_std, s_sv1)} "
                  f"{_fmt(cv1_mean, cv1_std, s_cv1)} "
                  f"{_fmt(sv3_mean, sv3_std, s_sv3)} "
                  f"{_fmt(cv3_mean, cv3_std, s_cv3)}")

            # JSON entries
            for vname, vmean, vstd, vspeed in [
                ("Sep-V1", sv1_mean, sv1_std, s_sv1),
                ("Comb-V1", cv1_mean, cv1_std, s_cv1),
                ("Sep-V3", sv3_mean, sv3_std, s_sv3),
                ("Comb-V3", cv3_mean, cv3_std, s_cv3),
            ]:
                json_entries.append({
                    "config": {
                        "benchmark_id": f"llama_single_op_{label.replace(' ', '_')}_{vname}_bf16_{batch}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(vname),
                        "model": {"name": label, "description": desc, "precision": "BF16"},
                        "workload": {
                            "batch_size": batch,
                            "dimensions": {"h": h, "out_dims": out_dims},
                        },
                    },
                    "metrics": {
                        "single_op": {
                            "mean_ms": round(vmean, 6),
                            "stddev_ms": round(vstd, 6),
                            "variant": vname,
                        },
                        "speedup": {
                            "vs_baseline": round(vspeed, 4),
                            "baseline_description": "PyTorch nn.RMSNorm + nn.Linear (BF16, separate calls)",
                            "baseline_mean_ms": round(base_mean, 6),
                            "baseline_stddev_ms": round(base_std, 6),
                        },
                    },
                })

    print()
    _save_json_results(json_entries, prefix="llama_single_op_bf16")


# ---------------------------------------------------------------------------
# Llama BF16 end-to-end benchmark
# ---------------------------------------------------------------------------

def benchmark_llama_e2e_bf16():
    """End-to-end Llama-3.1-8B and Llama-3.2-3B token generation in BF16."""
    print("=" * 130)
    print("END-TO-END BF16 BENCHMARK: Llama Token Generation (Separate vs Combined)")
    print("=" * 130)

    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama import patch_llama_model

    models_to_test = [
        ("unsloth/Llama-3.2-3B", "Llama-3.2-3B"),
        ("NousResearch/Meta-Llama-3.1-8B", "Llama-3.1-8B"),
    ]

    prompts = [
        "The future of artificial intelligence is",
        "In a shocking finding, scientists discovered that",
        "The economic impact of climate change will",
        "Recent advances in quantum computing have shown",
        "The most important lesson from history is",
        "Space exploration in the next decade will focus on",
        "The relationship between technology and society has",
        "New research in neuroscience suggests that the brain",
        "The key to understanding complex systems lies in",
        "Advances in renewable energy technology have made",
        "The intersection of biology and computing creates",
        "Modern cryptography relies fundamentally on the",
        "The philosophical implications of consciousness are",
        "Deep ocean exploration has revealed unexpected",
        "The evolution of programming languages shows that",
        "Quantum entanglement challenges our understanding of",
        "The future of transportation will be shaped by",
        "Breakthroughs in materials science have enabled",
        "The role of microbiomes in human health is",
        "Artificial general intelligence remains a topic of",
        "Climate modeling has become increasingly accurate",
        "The history of mathematics reveals patterns in",
        "Neural network architectures continue to evolve",
        "The economics of space mining could transform",
        "Genetic engineering tools like CRISPR have opened",
        "The social impact of automation extends beyond",
        "Advances in battery technology are crucial for",
        "The study of ancient civilizations reveals that",
        "Protein folding prediction has been revolutionized",
        "The future of education will be transformed by",
        "Sustainable agriculture requires innovative approaches",
        "The development of fusion energy has reached",
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # (display_name, variant, combined)
    test_configs = [
        ("Original", None, None),
        ("Sep-V1", "V1", False),
        ("Comb-V1", "V1", True),
        ("Sep-V3", "V3", False),
        ("Comb-V3", "V3", True),
    ]

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )
    num_runs = 5
    json_entries = []

    for model_name, label in models_to_test:
        print(f"\n{'='*130}")
        print(f"  {label} ({model_name}) — BF16")
        print(f"{'='*130}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        all_results = {}

        for display_name, variant, combined in test_configs:
            print(f"\nLoading {label} for {display_name} (BF16)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16
            ).cuda().eval()

            if variant is not None:
                print(f"  Patching model (variant={variant}, combined={combined})...")
                patch_llama_model(model, variant=variant, combined=combined)

            variant_results = {}
            for bs in batch_sizes:
                batch_prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
                inputs = tokenizer(
                    batch_prompts, return_tensors="pt", padding=True
                ).to("cuda")

                print(f"  Warming up {display_name} (batch_size={bs})...")
                with torch.no_grad():
                    _ = model.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()

                print(f"  Benchmarking {display_name} (batch_size={bs})...")
                times = []
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model.generate(**inputs, **gen_kwargs)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
                total_tokens = bs * num_new_tokens
                mean_time = sum(times) / len(times)
                stddev_time = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

                variant_results[bs] = {
                    "mean_time": mean_time,
                    "stddev_time": stddev_time,
                    "num_new_tokens": num_new_tokens,
                    "total_tokens": total_tokens,
                    "tps": total_tokens / mean_time,
                }

            all_results[display_name] = variant_results

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # --- Report ---
        print(f"\n{'─'*130}")
        print(f"  Results for {label} (BF16)")
        print(f"{'─'*130}")

        header = f"  {'Batch':>5} {'Tokens':>7} | {'Original':>16} {'tok/s':>8}"
        for dn, v, c in test_configs[1:]:
            header += f" | {dn:>16} {'tok/s':>8} {'Speedup':>7}"
        print(header)
        print(f"  {'-'*125}")

        orig = all_results["Original"]
        for bs in batch_sizes:
            o = orig[bs]
            line = (f"  {bs:>5} {o['total_tokens']:>7} | "
                    f"{o['mean_time']*1000:>8.1f}±{o['stddev_time']*1000:>5.1f}ms {o['tps']:>7.1f}")
            for dn, v, c in test_configs[1:]:
                vr = all_results[dn][bs]
                speedup = vr["tps"] / o["tps"]
                line += (f" | {vr['mean_time']*1000:>8.1f}±{vr['stddev_time']*1000:>4.1f}ms "
                         f"{vr['tps']:>7.1f} {speedup:>6.3f}x")
            print(line)

        # JSON entries
        for dn, v, c in test_configs[1:]:
            for bs in batch_sizes:
                vr = all_results[dn][bs]
                o = orig[bs]
                json_entries.append({
                    "config": {
                        "benchmark_id": f"llama_e2e_{label.replace(' ', '_').replace('.', '')}_{dn}_bf16_{bs}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(dn),
                        "model": {
                            "name": model_name,
                            "label": label,
                            "precision": "BF16",
                        },
                        "workload": {
                            "batch_size": bs,
                            "input_tokens": "variable",
                            "output_tokens": vr["num_new_tokens"],
                            "sampling": {"temperature": 0, "top_p": 1.0},
                        },
                    },
                    "metrics": {
                        "throughput": {
                            "tokens_per_second": round(vr["tps"], 1),
                        },
                        "latency": {
                            "mean_ms": round(vr["mean_time"] * 1000, 1),
                            "stddev_ms": round(vr["stddev_time"] * 1000, 1),
                        },
                        "speedup": {
                            "vs_baseline": round(vr["tps"] / o["tps"], 4),
                            "baseline_description": f"Unpatched {label} (BF16)",
                        },
                    },
                })

    _save_json_results(json_entries, prefix="llama_e2e_bf16")


# ---------------------------------------------------------------------------
# GPT-OSS single-operation benchmark (QKV only)
# ---------------------------------------------------------------------------

def benchmark_gpt_oss_single_op():
    """Benchmark RMSNorm+Linear combined for GPT-OSS QKV dimensions (BF16)."""
    print("=" * 130)
    print("GPT-OSS SINGLE-OPERATION BENCHMARK: RMSNorm+Linear Combined QKV (BF16)")
    print("=" * 130)

    configs = [
        # (label, h, out_dims, description)
        ("GPT-OSS-20B attn", 2880, [4096, 512, 512], "GQA QKV (with bias)"),
    ]

    batch_sizes = [1, 8, 32, 128, 512]
    warmup_iters = 100
    measure_iters = 1000
    dtype = torch.bfloat16

    print(f"\n{'Config':<20} {'h':>5} {'out_dims':<20} {'batch':>6} | "
          f"{'Baseline':>14} {'Comb-V1':>14} {'Comb-V3':>14}")
    print("-" * 110)

    json_entries = []

    for label, h, out_dims, desc in configs:
        torch.manual_seed(42)
        rms_norm = torch.nn.RMSNorm(h, eps=1e-6).to(device="cuda", dtype=dtype)
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

        linears = []
        for od in out_dims:
            lin = nn.Linear(h, od, bias=True, device="cuda", dtype=dtype)
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            nn.init.normal_(lin.bias, mean=0.0, std=0.01)
            linears.append(lin)

        # Compute fused weights in FP32 for stability, then cast to BF16
        rms_norm_f32 = torch.nn.RMSNorm(h, eps=1e-6).cuda()
        rms_norm_f32.weight.data.copy_(rms_norm.weight.data.float())

        linears_f32 = []
        for lin in linears:
            lin_f32 = nn.Linear(lin.in_features, lin.out_features, bias=True).cuda()
            lin_f32.weight.data.copy_(lin.weight.data.float())
            lin_f32.bias.data.copy_(lin.bias.data.float())
            linears_f32.append(lin_f32)

        W_comb, b_comb, split_sizes, h_dim, eps = compute_fused_weights_rmsnorm_combined(
            rms_norm_f32, linears_f32
        )
        W_comb = W_comb.to(dtype)
        b_comb = b_comb.to(dtype)

        del rms_norm_f32, linears_f32

        comb_v1 = FusedRMSNormCombinedLinearV1(W_comb, b_comb, split_sizes, h_dim, eps)
        comb_v3 = FusedRMSNormCombinedLinearV3(W_comb, b_comb, split_sizes, h_dim, eps)

        out_dims_str = "+".join(str(d) for d in out_dims)

        for batch in batch_sizes:
            x = torch.randn(batch, h, device="cuda", dtype=dtype)

            # Baseline: nn.RMSNorm + nn.Linear (separate calls)
            def baseline():
                normed = rms_norm(x)
                return [lin(normed) for lin in linears]
            base_mean, base_std, _ = _measure_per_iter(baseline, warmup_iters, measure_iters)

            # Combined fused V1
            def comb_v1_fn():
                return comb_v1(x)
            cv1_mean, cv1_std, _ = _measure_per_iter(comb_v1_fn, warmup_iters, measure_iters)

            # Combined fused V3
            def comb_v3_fn():
                return comb_v3(x)
            cv3_mean, cv3_std, _ = _measure_per_iter(comb_v3_fn, warmup_iters, measure_iters)

            s_cv1 = base_mean / cv1_mean if cv1_mean > 0 else float("inf")
            s_cv3 = base_mean / cv3_mean if cv3_mean > 0 else float("inf")

            def _fmt(mean, std, speedup):
                return f"{mean:>7.4f}±{std:>5.4f}({speedup:.2f}x)"

            print(f"{label:<20} {h:>5} {out_dims_str:<20} {batch:>6} | "
                  f"{base_mean:>7.4f}±{base_std:<5.4f}ms "
                  f"{_fmt(cv1_mean, cv1_std, s_cv1)} "
                  f"{_fmt(cv3_mean, cv3_std, s_cv3)}")

            for vname, vmean, vstd, vspeed in [
                ("Comb-V1", cv1_mean, cv1_std, s_cv1),
                ("Comb-V3", cv3_mean, cv3_std, s_cv3),
            ]:
                json_entries.append({
                    "config": {
                        "benchmark_id": f"gpt_oss_single_op_{label.replace(' ', '_')}_{vname}_bf16_{batch}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(vname),
                        "model": {"name": label, "description": desc, "precision": "BF16"},
                        "workload": {
                            "batch_size": batch,
                            "dimensions": {"h": h, "out_dims": out_dims},
                        },
                    },
                    "metrics": {
                        "single_op": {
                            "mean_ms": round(vmean, 6),
                            "stddev_ms": round(vstd, 6),
                            "variant": vname,
                        },
                        "speedup": {
                            "vs_baseline": round(vspeed, 4),
                            "baseline_description": "PyTorch nn.RMSNorm + nn.Linear (BF16, separate calls)",
                            "baseline_mean_ms": round(base_mean, 6),
                            "baseline_stddev_ms": round(base_std, 6),
                        },
                    },
                })

    print()
    _save_json_results(json_entries, prefix="gpt_oss_single_op_bf16")


# ---------------------------------------------------------------------------
# GPT-OSS end-to-end benchmark
# ---------------------------------------------------------------------------

def benchmark_gpt_oss_e2e():
    """End-to-end GPT-OSS-20B token generation in BF16 (QKV fusion only)."""
    print("=" * 130)
    print("END-TO-END BF16 BENCHMARK: GPT-OSS-20B Token Generation (QKV Fusion)")
    print("=" * 130)

    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_gpt_oss import patch_gpt_oss_model

    model_name = "openai/gpt-oss-20b"

    prompts = [
        "The future of artificial intelligence is",
        "In a shocking finding, scientists discovered that",
        "The economic impact of climate change will",
        "Recent advances in quantum computing have shown",
        "The most important lesson from history is",
        "Space exploration in the next decade will focus on",
        "The relationship between technology and society has",
        "New research in neuroscience suggests that the brain",
        "The key to understanding complex systems lies in",
        "Advances in renewable energy technology have made",
        "The intersection of biology and computing creates",
        "Modern cryptography relies fundamentally on the",
        "The philosophical implications of consciousness are",
        "Deep ocean exploration has revealed unexpected",
        "The evolution of programming languages shows that",
        "Quantum entanglement challenges our understanding of",
    ]
    batch_sizes = [1, 2, 4, 8, 16]

    # (display_name, variant)
    test_configs = [
        ("Original", None),
        ("Fused-V1", "V1"),
        ("Fused-V3", "V3"),
    ]

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )
    num_runs = 5
    json_entries = []

    print(f"\n  Model: {model_name} (BF16)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    for display_name, variant in test_configs:
        print(f"\nLoading GPT-OSS for {display_name} (BF16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).cuda().eval()

        if variant is not None:
            print(f"  Patching model (variant={variant})...")
            patch_gpt_oss_model(model, variant=variant)

        variant_results = {}
        for bs in batch_sizes:
            batch_prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to("cuda")

            print(f"  Warming up {display_name} (batch_size={bs})...")
            with torch.no_grad():
                _ = model.generate(**inputs, **gen_kwargs)
            torch.cuda.synchronize()

            print(f"  Benchmarking {display_name} (batch_size={bs})...")
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
            total_tokens = bs * num_new_tokens
            mean_time = sum(times) / len(times)
            stddev_time = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

            variant_results[bs] = {
                "mean_time": mean_time,
                "stddev_time": stddev_time,
                "num_new_tokens": num_new_tokens,
                "total_tokens": total_tokens,
                "tps": total_tokens / mean_time,
            }

        all_results[display_name] = variant_results

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # --- Report ---
    print(f"\n{'─'*130}")
    print(f"  Results for GPT-OSS-20B (BF16, QKV fusion only)")
    print(f"{'─'*130}")

    header = f"  {'Batch':>5} {'Tokens':>7} | {'Original':>16} {'tok/s':>8}"
    for dn, v in test_configs[1:]:
        header += f" | {dn:>16} {'tok/s':>8} {'Speedup':>7}"
    print(header)
    print(f"  {'-'*100}")

    orig = all_results["Original"]
    for bs in batch_sizes:
        o = orig[bs]
        line = (f"  {bs:>5} {o['total_tokens']:>7} | "
                f"{o['mean_time']*1000:>8.1f}±{o['stddev_time']*1000:>5.1f}ms {o['tps']:>7.1f}")
        for dn, v in test_configs[1:]:
            vr = all_results[dn][bs]
            speedup = vr["tps"] / o["tps"]
            line += (f" | {vr['mean_time']*1000:>8.1f}±{vr['stddev_time']*1000:>4.1f}ms "
                     f"{vr['tps']:>7.1f} {speedup:>6.3f}x")
        print(line)

    # JSON entries
    for dn, v in test_configs[1:]:
        for bs in batch_sizes:
            vr = all_results[dn][bs]
            o = orig[bs]
            json_entries.append({
                "config": {
                    "benchmark_id": f"gpt_oss_e2e_{dn}_bf16_{bs}",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "hardware": _get_hardware_info(),
                    "software": _get_software_info(dn),
                    "model": {
                        "name": model_name,
                        "label": "GPT-OSS-20B",
                        "precision": "BF16",
                    },
                    "workload": {
                        "batch_size": bs,
                        "input_tokens": "variable",
                        "output_tokens": vr["num_new_tokens"],
                        "sampling": {"temperature": 0, "top_p": 1.0},
                    },
                },
                "metrics": {
                    "throughput": {
                        "tokens_per_second": round(vr["tps"], 1),
                    },
                    "latency": {
                        "mean_ms": round(vr["mean_time"] * 1000, 1),
                        "stddev_ms": round(vr["stddev_time"] * 1000, 1),
                    },
                    "speedup": {
                        "vs_baseline": round(vr["tps"] / o["tps"], 4),
                        "baseline_description": "Unpatched GPT-OSS-20B (BF16)",
                    },
                },
            })

    _save_json_results(json_entries, prefix="gpt_oss_e2e_bf16")


# ---------------------------------------------------------------------------
# SwiGLU single-operation benchmark
# ---------------------------------------------------------------------------

def benchmark_swiglu_single_op():
    """Benchmark MLP gate+up+activation: baseline vs combined vs SwiGLU-fused for Llama configs."""
    print("=" * 140)
    print("SWIGLU SINGLE-OPERATION BENCHMARK: RMSNorm + gate+up + SiLU*mul (BF16)")
    print("=" * 140)

    configs = [
        # (label, h, intermediate, description)
        ("TinyLlama MLP",    2048,  5632,  "h=2048, inter=5632"),
        ("Llama-3.2-3B MLP", 3072,  8192,  "h=3072, inter=8192"),
        ("Llama-3.1-8B MLP", 4096, 14336,  "h=4096, inter=14336"),
    ]

    batch_sizes = [1, 8, 32, 128, 512]
    warmup_iters = 100
    measure_iters = 1000
    dtype = torch.bfloat16

    print(f"\n{'Config':<20} {'h':>5} {'inter':>6} {'batch':>6} | "
          f"{'Baseline':>14} {'Combined-V3':>14} {'SwiGLU-V1':>14} {'SwiGLU-V3':>14}")
    print("-" * 120)

    json_entries = []

    for label, h, intermediate, desc in configs:
        torch.manual_seed(42)
        out_dims = [intermediate, intermediate]

        rms_norm = torch.nn.RMSNorm(h, eps=1e-5).to(device="cuda", dtype=dtype)
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

        gate_proj = nn.Linear(h, intermediate, bias=False, device="cuda", dtype=dtype)
        up_proj = nn.Linear(h, intermediate, bias=False, device="cuda", dtype=dtype)
        nn.init.normal_(gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(up_proj.weight, mean=0.0, std=0.02)
        linears = [gate_proj, up_proj]

        # Compute fused weights in FP32 then cast
        rms_norm_f32 = torch.nn.RMSNorm(h, eps=1e-5).cuda()
        rms_norm_f32.weight.data.copy_(rms_norm.weight.data.float())

        linears_f32 = []
        for lin in linears:
            lin_f32 = nn.Linear(lin.in_features, lin.out_features, bias=False).cuda()
            lin_f32.weight.data.copy_(lin.weight.data.float())
            linears_f32.append(lin_f32)

        W_comb, b_comb, split_sizes, h_dim, eps = compute_fused_weights_rmsnorm_combined(
            rms_norm_f32, linears_f32
        )
        W_comb_bf16 = W_comb.to(dtype)
        b_comb_bf16 = b_comb.to(dtype)

        del rms_norm_f32, linears_f32

        # Build modules
        comb_v3 = FusedRMSNormCombinedLinearV3(W_comb_bf16, b_comb_bf16, split_sizes, h_dim, eps)
        swiglu_v1 = FusedRMSNormSwiGLUV1(W_comb_bf16, b_comb_bf16, intermediate, h_dim, eps)
        swiglu_v3 = FusedRMSNormSwiGLUV3(W_comb_bf16, b_comb_bf16, intermediate, h_dim, eps)

        for batch in batch_sizes:
            x = torch.randn(batch, h, device="cuda", dtype=dtype)

            # Baseline: nn.RMSNorm + gate + up + SiLU + mul
            def baseline():
                normed = rms_norm(x)
                g = gate_proj(normed)
                u = up_proj(normed)
                return F.silu(g) * u
            base_mean, base_std, _ = _measure_per_iter(baseline, warmup_iters, measure_iters)

            # Combined-V3: single matmul + normalize + split + SiLU + mul
            def comb_v3_fn():
                parts = comb_v3(x)
                return F.silu(parts[0]) * parts[1]
            cv3_mean, cv3_std, _ = _measure_per_iter(comb_v3_fn, warmup_iters, measure_iters)

            # SwiGLU-V1: single matmul + fused normalize+SiLU+mul
            def swiglu_v1_fn():
                return swiglu_v1(x)
            sg1_mean, sg1_std, _ = _measure_per_iter(swiglu_v1_fn, warmup_iters, measure_iters)

            # SwiGLU-V3: 512 threads
            def swiglu_v3_fn():
                return swiglu_v3(x)
            sg3_mean, sg3_std, _ = _measure_per_iter(swiglu_v3_fn, warmup_iters, measure_iters)

            s_cv3 = base_mean / cv3_mean if cv3_mean > 0 else float("inf")
            s_sg1 = base_mean / sg1_mean if sg1_mean > 0 else float("inf")
            s_sg3 = base_mean / sg3_mean if sg3_mean > 0 else float("inf")

            def _fmt(mean, std, speedup):
                return f"{mean:>7.4f}±{std:>5.4f}({speedup:.2f}x)"

            print(f"{label:<20} {h:>5} {intermediate:>6} {batch:>6} | "
                  f"{base_mean:>7.4f}±{base_std:<5.4f}ms "
                  f"{_fmt(cv3_mean, cv3_std, s_cv3)} "
                  f"{_fmt(sg1_mean, sg1_std, s_sg1)} "
                  f"{_fmt(sg3_mean, sg3_std, s_sg3)}")

            for vname, vmean, vstd, vspeed in [
                ("Comb-V3", cv3_mean, cv3_std, s_cv3),
                ("SwiGLU-V1", sg1_mean, sg1_std, s_sg1),
                ("SwiGLU-V3", sg3_mean, sg3_std, s_sg3),
            ]:
                json_entries.append({
                    "config": {
                        "benchmark_id": f"swiglu_single_op_{label.replace(' ', '_')}_{vname}_bf16_{batch}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(vname),
                        "model": {"name": label, "description": desc, "precision": "BF16"},
                        "workload": {
                            "batch_size": batch,
                            "dimensions": {"h": h, "intermediate": intermediate},
                        },
                    },
                    "metrics": {
                        "single_op": {
                            "mean_ms": round(vmean, 6),
                            "stddev_ms": round(vstd, 6),
                            "variant": vname,
                        },
                        "speedup": {
                            "vs_baseline": round(vspeed, 4),
                            "baseline_description": "PyTorch nn.RMSNorm + 2x nn.Linear + SiLU + mul (BF16)",
                            "baseline_mean_ms": round(base_mean, 6),
                            "baseline_stddev_ms": round(base_std, 6),
                        },
                    },
                })

    print()
    _save_json_results(json_entries, prefix="swiglu_single_op_bf16")


# ---------------------------------------------------------------------------
# SwiGLU end-to-end benchmark
# ---------------------------------------------------------------------------

def benchmark_swiglu_e2e():
    """End-to-end Llama token generation comparing Original vs Combined-V3 vs SwiGLU-V3."""
    print("=" * 140)
    print("END-TO-END SWIGLU BENCHMARK: Llama Token Generation (Original vs Combined vs SwiGLU)")
    print("=" * 140)

    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama import patch_llama_model

    models_to_test = [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama", torch.float32),
        ("unsloth/Llama-3.2-3B", "Llama-3.2-3B", torch.bfloat16),
        ("NousResearch/Meta-Llama-3.1-8B", "Llama-3.1-8B", torch.bfloat16),
    ]

    prompts = [
        "The future of artificial intelligence is",
        "In a shocking finding, scientists discovered that",
        "The economic impact of climate change will",
        "Recent advances in quantum computing have shown",
        "The most important lesson from history is",
        "Space exploration in the next decade will focus on",
        "The relationship between technology and society has",
        "New research in neuroscience suggests that the brain",
        "The key to understanding complex systems lies in",
        "Advances in renewable energy technology have made",
        "The intersection of biology and computing creates",
        "Modern cryptography relies fundamentally on the",
        "The philosophical implications of consciousness are",
        "Deep ocean exploration has revealed unexpected",
        "The evolution of programming languages shows that",
        "Quantum entanglement challenges our understanding of",
        "The future of transportation will be shaped by",
        "Breakthroughs in materials science have enabled",
        "The role of microbiomes in human health is",
        "Artificial general intelligence remains a topic of",
        "Climate modeling has become increasingly accurate",
        "The history of mathematics reveals patterns in",
        "Neural network architectures continue to evolve",
        "The economics of space mining could transform",
        "Genetic engineering tools like CRISPR have opened",
        "The social impact of automation extends beyond",
        "Advances in battery technology are crucial for",
        "The study of ancient civilizations reveals that",
        "Protein folding prediction has been revolutionized",
        "The future of education will be transformed by",
        "Sustainable agriculture requires innovative approaches",
        "The development of fusion energy has reached",
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # (display_name, variant, combined, swiglu)
    test_configs = [
        ("Original",    None, None,  None),
        ("Comb-V3",     "V3", True,  False),
        ("SwiGLU-V3",   "V3", None,  True),
    ]

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )
    num_runs = 5
    json_entries = []

    for model_name, label, model_dtype in models_to_test:
        precision_str = "FP32" if model_dtype == torch.float32 else "BF16"
        print(f"\n{'='*140}")
        print(f"  {label} ({model_name}) — {precision_str}")
        print(f"{'='*140}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        all_results = {}

        for display_name, variant, combined, swiglu in test_configs:
            print(f"\nLoading {label} for {display_name} ({precision_str})...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=model_dtype
            ).cuda().eval()

            if variant is not None:
                if swiglu:
                    print(f"  Patching model (variant={variant}, swiglu=True)...")
                    patch_llama_model(model, variant=variant, swiglu=True)
                else:
                    print(f"  Patching model (variant={variant}, combined={combined})...")
                    patch_llama_model(model, variant=variant, combined=combined)

            variant_results = {}
            for bs in batch_sizes:
                batch_prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
                inputs = tokenizer(
                    batch_prompts, return_tensors="pt", padding=True
                ).to("cuda")

                print(f"  Warming up {display_name} (batch_size={bs})...")
                with torch.no_grad():
                    _ = model.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()

                print(f"  Benchmarking {display_name} (batch_size={bs})...")
                times = []
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model.generate(**inputs, **gen_kwargs)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
                total_tokens = bs * num_new_tokens
                mean_time = sum(times) / len(times)
                stddev_time = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

                variant_results[bs] = {
                    "mean_time": mean_time,
                    "stddev_time": stddev_time,
                    "num_new_tokens": num_new_tokens,
                    "total_tokens": total_tokens,
                    "tps": total_tokens / mean_time,
                }

            all_results[display_name] = variant_results

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # --- Report ---
        print(f"\n{'─'*140}")
        print(f"  Results for {label} ({precision_str})")
        print(f"{'─'*140}")

        header = f"  {'Batch':>5} {'Tokens':>7} | {'Original':>16} {'tok/s':>8}"
        for dn, v, c, s in test_configs[1:]:
            header += f" | {dn:>16} {'tok/s':>8} {'Speedup':>7}"
        print(header)
        print(f"  {'-'*130}")

        orig = all_results["Original"]
        for bs in batch_sizes:
            o = orig[bs]
            line = (f"  {bs:>5} {o['total_tokens']:>7} | "
                    f"{o['mean_time']*1000:>8.1f}±{o['stddev_time']*1000:>5.1f}ms {o['tps']:>7.1f}")
            for dn, v, c, s in test_configs[1:]:
                vr = all_results[dn][bs]
                speedup = vr["tps"] / o["tps"]
                line += (f" | {vr['mean_time']*1000:>8.1f}±{vr['stddev_time']*1000:>4.1f}ms "
                         f"{vr['tps']:>7.1f} {speedup:>6.3f}x")
            print(line)

        # JSON entries
        for dn, v, c, s in test_configs[1:]:
            for bs in batch_sizes:
                vr = all_results[dn][bs]
                o = orig[bs]
                json_entries.append({
                    "config": {
                        "benchmark_id": f"swiglu_e2e_{label.replace(' ', '_').replace('.', '')}_{dn}_{precision_str.lower()}_{bs}",
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
                            "input_tokens": "variable",
                            "output_tokens": vr["num_new_tokens"],
                            "sampling": {"temperature": 0, "top_p": 1.0},
                        },
                    },
                    "metrics": {
                        "throughput": {
                            "tokens_per_second": round(vr["tps"], 1),
                        },
                        "latency": {
                            "mean_ms": round(vr["mean_time"] * 1000, 1),
                            "stddev_ms": round(vr["stddev_time"] * 1000, 1),
                        },
                        "speedup": {
                            "vs_baseline": round(vr["tps"] / o["tps"], 4),
                            "baseline_description": f"Unpatched {label} ({precision_str})",
                        },
                    },
                })

    _save_json_results(json_entries, prefix="swiglu_e2e")


# ---------------------------------------------------------------------------
# GQA decode single-op benchmark
# ---------------------------------------------------------------------------

def benchmark_gqa_decode_single_op():
    """Benchmark GQA decode attention: PyTorch SDPA vs V2 vs V3 for all model configs."""
    print("=" * 140)
    print("GQA DECODE SINGLE-OP BENCHMARK: SDPA vs V2 (per-query-head) vs V3 (per-KV-head, shared)")
    print("=" * 140)

    from src.gqa_attention_forward import (
        GQADecodeAttentionV2, GQADecodeAttentionV3, pytorch_gqa_decode_attention,
    )

    configs = [
        # (label, num_q_heads, num_kv_heads, head_dim, description)
        ("TinyLlama",    32,  4, 64,  "h=2048, group=8"),
        ("Llama-3.2-3B", 24,  8, 128, "h=3072, group=3"),
        ("Llama-3.1-8B", 32,  8, 128, "h=4096, group=4"),
        ("GPT-OSS-20B",  32,  4, 128, "h=2880, group=8"),
    ]

    batch_sizes = [1, 4, 8, 16, 32]
    ctx_lengths = [128, 512, 1024, 2048, 4096]
    warmup_iters = 100
    measure_iters = 1000
    dtype = torch.bfloat16

    print(f"\n{'Config':<16} {'q_h':>4} {'kv_h':>4} {'hd':>4} {'batch':>5} {'ctx':>5} | "
          f"{'SDPA':>14} {'V2-ours':>14} {'V3-ours':>14}")
    print("-" * 110)

    json_entries = []

    for label, num_q, num_kv, hd, desc in configs:
        v2 = GQADecodeAttentionV2(num_q, num_kv, hd)
        v3 = GQADecodeAttentionV3(num_q, num_kv, hd)
        group_size = num_q // num_kv

        for batch in batch_sizes:
            for ctx_len in ctx_lengths:
                q = torch.randn(batch, num_q, hd, device="cuda", dtype=dtype)
                k = torch.randn(batch, ctx_len, num_kv, hd, device="cuda", dtype=dtype)
                v = torch.randn(batch, ctx_len, num_kv, hd, device="cuda", dtype=dtype)

                # Pre-expand for SDPA baseline
                k_expanded = k.repeat_interleave(group_size, dim=2)
                v_expanded = v.repeat_interleave(group_size, dim=2)
                q_4d = q.unsqueeze(2)
                k_4d = k_expanded.permute(0, 2, 1, 3)
                v_4d = v_expanded.permute(0, 2, 1, 3)
                scale = 1.0 / math.sqrt(hd)

                def sdpa_fn():
                    return F.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=scale)
                sdpa_mean, sdpa_std, _ = _measure_per_iter(sdpa_fn, warmup_iters, measure_iters)

                def v2_fn():
                    return v2(q, k, v)
                v2_mean, v2_std, _ = _measure_per_iter(v2_fn, warmup_iters, measure_iters)

                def v3_fn():
                    return v3(q, k, v)
                v3_mean, v3_std, _ = _measure_per_iter(v3_fn, warmup_iters, measure_iters)

                s_v2 = sdpa_mean / v2_mean if v2_mean > 0 else float("inf")
                s_v3 = sdpa_mean / v3_mean if v3_mean > 0 else float("inf")

                def _fmt(mean, std, speedup):
                    return f"{mean:>7.4f}±{std:>5.4f}({speedup:.2f}x)"

                print(f"{label:<16} {num_q:>4} {num_kv:>4} {hd:>4} {batch:>5} {ctx_len:>5} | "
                      f"{sdpa_mean:>7.4f}±{sdpa_std:<5.4f}ms "
                      f"{_fmt(v2_mean, v2_std, s_v2)} "
                      f"{_fmt(v3_mean, v3_std, s_v3)}")

                for vname, vmean, vstd, vspeed in [
                    ("V2", v2_mean, v2_std, s_v2),
                    ("V3", v3_mean, v3_std, s_v3),
                ]:
                    json_entries.append({
                        "config": {
                            "benchmark_id": f"gqa_decode_{label.replace(' ', '_')}_{vname}_bf16_b{batch}_c{ctx_len}",
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "hardware": _get_hardware_info(),
                            "software": _get_software_info(vname),
                            "model": {"name": label, "description": desc, "precision": "BF16"},
                            "workload": {
                                "batch_size": batch,
                                "context_length": ctx_len,
                                "dimensions": {
                                    "num_q_heads": num_q,
                                    "num_kv_heads": num_kv,
                                    "head_dim": hd,
                                    "group_size": group_size,
                                },
                            },
                        },
                        "metrics": {
                            "single_op": {
                                "mean_ms": round(vmean, 6),
                                "stddev_ms": round(vstd, 6),
                                "variant": vname,
                            },
                            "speedup": {
                                "vs_sdpa": round(vspeed, 4),
                                "sdpa_mean_ms": round(sdpa_mean, 6),
                                "sdpa_stddev_ms": round(sdpa_std, 6),
                            },
                        },
                    })

    print()
    _save_json_results(json_entries, prefix="gqa_decode_single_op_bf16")


# ---------------------------------------------------------------------------
# GQA decode end-to-end benchmark
# ---------------------------------------------------------------------------

def benchmark_gqa_decode_e2e():
    """End-to-end Llama token generation comparing Original vs SwiGLU-V3 vs SwiGLU-V3+GQA-V3."""
    print("=" * 140)
    print("END-TO-END GQA DECODE BENCHMARK: Token Generation (Original vs SwiGLU-V3 vs SwiGLU-V3+GQA-V3)")
    print("=" * 140)

    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama import patch_llama_model

    models_to_test = [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama", torch.float32),
        ("unsloth/Llama-3.2-3B", "Llama-3.2-3B", torch.bfloat16),
        ("NousResearch/Meta-Llama-3.1-8B", "Llama-3.1-8B", torch.bfloat16),
    ]

    prompts = [
        "The future of artificial intelligence is",
        "In a shocking finding, scientists discovered that",
        "The economic impact of climate change will",
        "Recent advances in quantum computing have shown",
        "The most important lesson from history is",
        "Space exploration in the next decade will focus on",
        "The relationship between technology and society has",
        "New research in neuroscience suggests that the brain",
        "The key to understanding complex systems lies in",
        "Advances in renewable energy technology have made",
        "The intersection of biology and computing creates",
        "Modern cryptography relies fundamentally on the",
        "The philosophical implications of consciousness are",
        "Deep ocean exploration has revealed unexpected",
        "The evolution of programming languages shows that",
        "Quantum entanglement challenges our understanding of",
        "The future of transportation will be shaped by",
        "Breakthroughs in materials science have enabled",
        "The role of microbiomes in human health is",
        "Artificial general intelligence remains a topic of",
        "Climate modeling has become increasingly accurate",
        "The history of mathematics reveals patterns in",
        "Neural network architectures continue to evolve",
        "The economics of space mining could transform",
        "Genetic engineering tools like CRISPR have opened",
        "The social impact of automation extends beyond",
        "Advances in battery technology are crucial for",
        "The study of ancient civilizations reveals that",
        "Protein folding prediction has been revolutionized",
        "The future of education will be transformed by",
        "Sustainable agriculture requires innovative approaches",
        "The development of fusion energy has reached",
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # (display_name, swiglu, gqa_decode)
    test_configs = [
        ("Original",          False, False),
        ("SwiGLU-V3",         True,  False),
        ("SwiGLU-V3+GQA-V3",  True,  True),
    ]

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )
    num_runs = 5
    json_entries = []

    for model_name, label, model_dtype in models_to_test:
        precision_str = "FP32" if model_dtype == torch.float32 else "BF16"
        print(f"\n{'='*140}")
        print(f"  {label} ({model_name}) — {precision_str}")
        print(f"{'='*140}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        all_results = {}

        for display_name, use_swiglu, use_gqa in test_configs:
            print(f"\nLoading {label} for {display_name} ({precision_str})...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=model_dtype
            ).cuda().eval()

            if use_swiglu or use_gqa:
                print(f"  Patching model (swiglu={use_swiglu}, gqa_decode={use_gqa})...")
                patch_llama_model(model, variant="V3", swiglu=use_swiglu, gqa_decode=use_gqa)

            variant_results = {}
            for bs in batch_sizes:
                batch_prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
                inputs = tokenizer(
                    batch_prompts, return_tensors="pt", padding=True
                ).to("cuda")

                print(f"  Warming up {display_name} (batch_size={bs})...")
                with torch.no_grad():
                    _ = model.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()

                print(f"  Benchmarking {display_name} (batch_size={bs})...")
                times = []
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model.generate(**inputs, **gen_kwargs)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
                total_tokens = bs * num_new_tokens
                mean_time = sum(times) / len(times)
                stddev_time = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

                variant_results[bs] = {
                    "mean_time": mean_time,
                    "stddev_time": stddev_time,
                    "num_new_tokens": num_new_tokens,
                    "total_tokens": total_tokens,
                    "tps": total_tokens / mean_time,
                }

            all_results[display_name] = variant_results

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # --- Report ---
        print(f"\n{'─'*140}")
        print(f"  Results for {label} ({precision_str})")
        print(f"{'─'*140}")

        header = f"  {'Batch':>5} {'Tokens':>7} | {'Original':>16} {'tok/s':>8}"
        for dn, sw, gq in test_configs[1:]:
            header += f" | {dn:>20} {'tok/s':>8} {'Speedup':>7}"
        print(header)
        print(f"  {'-'*130}")

        orig = all_results["Original"]
        for bs in batch_sizes:
            o = orig[bs]
            line = (f"  {bs:>5} {o['total_tokens']:>7} | "
                    f"{o['mean_time']*1000:>8.1f}±{o['stddev_time']*1000:>5.1f}ms {o['tps']:>7.1f}")
            for dn, sw, gq in test_configs[1:]:
                vr = all_results[dn][bs]
                speedup = vr["tps"] / o["tps"]
                line += (f" | {vr['mean_time']*1000:>8.1f}±{vr['stddev_time']*1000:>4.1f}ms "
                         f"{vr['tps']:>7.1f} {speedup:>6.3f}x")
            print(line)

        # JSON entries
        for dn, sw, gq in test_configs[1:]:
            for bs in batch_sizes:
                vr = all_results[dn][bs]
                o = orig[bs]
                json_entries.append({
                    "config": {
                        "benchmark_id": f"gqa_e2e_{label.replace(' ', '_').replace('.', '')}_{dn.replace('+', '_')}_{precision_str.lower()}_{bs}",
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
                            "input_tokens": "variable",
                            "output_tokens": vr["num_new_tokens"],
                            "sampling": {"temperature": 0, "top_p": 1.0},
                        },
                    },
                    "metrics": {
                        "throughput": {
                            "tokens_per_second": round(vr["tps"], 1),
                        },
                        "latency": {
                            "mean_ms": round(vr["mean_time"] * 1000, 1),
                            "stddev_ms": round(vr["stddev_time"] * 1000, 1),
                        },
                        "speedup": {
                            "vs_baseline": round(vr["tps"] / o["tps"], 4),
                            "baseline_description": f"Unpatched {label} ({precision_str})",
                        },
                    },
                })

    _save_json_results(json_entries, prefix="gqa_decode_e2e")


# ---------------------------------------------------------------------------
# MoE single-operation benchmark
# ---------------------------------------------------------------------------

def benchmark_moe_single_op():
    """Benchmark MoE forward: baseline loop vs Triton grouped GEMM vs padded batched (BF16)."""
    print("=" * 130)
    print("MOE SINGLE-OPERATION BENCHMARK: Baseline Loop vs Grouped GEMM (BF16)")
    print("=" * 130)

    from src.moe_grouped_gemm import (
        baseline_moe_forward, optimized_moe_forward, fused_gptoss_gate,
    )

    configs = [
        # (label, hidden, intermediate, num_experts, top_k, description)
        ("GPT-OSS-20B MoE", 2880, 2880, 128, 4, "128 experts, top-4"),
        ("GPT-OSS-20B MoE-s", 2880, 2880, 32, 4, "32 experts, top-4"),
        ("MoE-small", 1024, 1024, 16, 2, "16 experts, top-2"),
    ]

    batch_sizes = [64, 128, 256, 512, 1024]
    warmup_iters = 20
    measure_iters = 100
    dtype = torch.bfloat16

    print(f"\n{'Config':<22} {'experts':>7} {'top_k':>5} {'batch':>6} | "
          f"{'Baseline':>14} {'Triton-GGEMM':>14} {'Padded-BMM':>14}")
    print("-" * 110)

    json_entries = []

    for label, hidden, intermediate, num_experts, top_k, desc in configs:
        torch.manual_seed(42)

        # Create synthetic MoE weights
        gate_up_proj = torch.randn(num_experts, hidden, 2 * intermediate,
                                   device="cuda", dtype=dtype) * 0.02
        gate_up_proj_bias = torch.randn(num_experts, 2 * intermediate,
                                        device="cuda", dtype=dtype) * 0.01
        down_proj = torch.randn(num_experts, intermediate, hidden,
                                device="cuda", dtype=dtype) * 0.02
        down_proj_bias = torch.randn(num_experts, hidden,
                                     device="cuda", dtype=dtype) * 0.01

        for batch in batch_sizes:
            x = torch.randn(batch, hidden, device="cuda", dtype=dtype)

            # Create random routing
            router_logits = torch.randn(batch, num_experts, device="cuda", dtype=dtype)
            router_values, router_indices = torch.topk(router_logits, top_k, dim=-1)
            router_scores = torch.softmax(router_values, dim=-1)

            # --- Baseline loop ---
            def baseline_fn():
                return baseline_moe_forward(
                    x, router_indices, router_scores,
                    gate_up_proj, gate_up_proj_bias,
                    down_proj, down_proj_bias,
                )
            base_mean, base_std, _ = _measure_per_iter(baseline_fn, warmup_iters, measure_iters)

            # --- Triton grouped GEMM ---
            def triton_fn():
                return optimized_moe_forward(
                    x, router_indices, router_scores,
                    gate_up_proj, gate_up_proj_bias,
                    down_proj, down_proj_bias,
                    use_triton_gemm=True,
                )
            triton_mean, triton_std, _ = _measure_per_iter(triton_fn, warmup_iters, measure_iters)

            # --- Padded batched ---
            def padded_fn():
                return optimized_moe_forward(
                    x, router_indices, router_scores,
                    gate_up_proj, gate_up_proj_bias,
                    down_proj, down_proj_bias,
                    use_triton_gemm=False,
                )
            padded_mean, padded_std, _ = _measure_per_iter(padded_fn, warmup_iters, measure_iters)

            s_triton = base_mean / triton_mean if triton_mean > 0 else float("inf")
            s_padded = base_mean / padded_mean if padded_mean > 0 else float("inf")

            def _fmt(mean, std, speedup):
                return f"{mean:>7.3f}±{std:>5.3f}({speedup:.2f}x)"

            print(f"{label:<22} {num_experts:>7} {top_k:>5} {batch:>6} | "
                  f"{base_mean:>7.3f}±{base_std:<5.3f}ms "
                  f"{_fmt(triton_mean, triton_std, s_triton)} "
                  f"{_fmt(padded_mean, padded_std, s_padded)}")

            for vname, vmean, vstd, vspeed in [
                ("Triton-GGEMM", triton_mean, triton_std, s_triton),
                ("Padded-BMM", padded_mean, padded_std, s_padded),
            ]:
                json_entries.append({
                    "config": {
                        "benchmark_id": f"moe_single_op_{label.replace(' ', '_')}_{vname}_bf16_{batch}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(vname),
                        "model": {
                            "name": label,
                            "description": desc,
                            "precision": "BF16",
                        },
                        "workload": {
                            "batch_size": batch,
                            "dimensions": {
                                "hidden": hidden,
                                "intermediate": intermediate,
                                "num_experts": num_experts,
                                "top_k": top_k,
                            },
                        },
                    },
                    "metrics": {
                        "single_op": {
                            "mean_ms": round(vmean, 6),
                            "stddev_ms": round(vstd, 6),
                            "variant": vname,
                        },
                        "speedup": {
                            "vs_baseline": round(vspeed, 4),
                            "baseline_description": "PyTorch expert loop (BF16)",
                            "baseline_mean_ms": round(base_mean, 6),
                            "baseline_stddev_ms": round(base_std, 6),
                        },
                    },
                })

    print()
    _save_json_results(json_entries, prefix="moe_single_op_bf16")


# ---------------------------------------------------------------------------
# MoE end-to-end benchmark
# ---------------------------------------------------------------------------

def benchmark_moe_e2e():
    """End-to-end GPT-OSS MoE token generation with MoE optimization (BF16).

    Uses a scaled-down GPT-OSS model (4 layers, 64 experts) with the same
    architecture as GPT-OSS-20B to fit in H100 80GB. The full 20B model has
    128 experts x 36 layers = ~230GB BF16, far exceeding single-GPU memory.
    Single-op benchmarks are the primary metric; E2E demonstrates integration.
    """
    print("=" * 130)
    print("END-TO-END BF16 BENCHMARK: GPT-OSS MoE Token Generation (QKV + MoE Fusion)")
    print("  NOTE: Uses scaled-down GPT-OSS (4 layers, 64 experts) to fit in 80GB.")
    print("  Single-op benchmarks are the primary MoE kernel performance metric.")
    print("=" * 130)

    import gc
    from transformers import AutoTokenizer
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssConfig, GptOssForCausalLM,
    )
    from src.patch_gpt_oss import patch_gpt_oss_model
    from src.patch_gpt_oss_moe import patch_gpt_oss_moe

    # Scaled-down config: same hidden/intermediate/heads as GPT-OSS-20B,
    # but only 4 layers and 64 experts (vs 36 layers, 128 experts)
    model_config = GptOssConfig(
        num_hidden_layers=4,
        num_local_experts=64,
        vocab_size=201088,
        hidden_size=2880,
        intermediate_size=2880,
        head_dim=64,
        num_attention_heads=64,
        num_key_value_heads=8,
        num_experts_per_tok=4,
        sliding_window=128,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    model_label = "GPT-OSS-MoE-4L64E"

    # Use a generic tokenizer (GPT-OSS-20B tokenizer might not be available)
    tokenizer_name = "openai/gpt-oss-20b"

    prompts = [
        "The future of artificial intelligence is",
        "In a shocking finding, scientists discovered that",
        "The economic impact of climate change will",
        "Recent advances in quantum computing have shown",
        "The most important lesson from history is",
        "Space exploration in the next decade will focus on",
        "The relationship between technology and society has",
        "New research in neuroscience suggests that the brain",
    ]
    batch_sizes = [1, 2, 4, 8]

    # (display_name, qkv_variant, moe_patch)
    test_configs = [
        ("Original", None, False),
        ("QKV-V1", "V1", False),
        ("QKV-V1+MoE", "V1", True),
    ]

    gen_kwargs = dict(
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
    )
    num_runs = 5
    json_entries = []

    print(f"\n  Model: {model_label} (BF16, {model_config.num_hidden_layers} layers, "
          f"{model_config.num_local_experts} experts, top-{model_config.num_experts_per_tok})")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    for display_name, qkv_variant, moe_patch in test_configs:
        print(f"\nCreating {model_label} for {display_name} (BF16)...")
        model = GptOssForCausalLM(model_config).to(dtype=torch.bfloat16).cuda().eval()
        # Initialize weights to reasonable values
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() >= 2:
                    torch.nn.init.normal_(p, std=0.02)
                else:
                    torch.nn.init.zeros_(p)

        param_count = sum(p.numel() for p in model.parameters())
        mem_gb = param_count * 2 / 1e9
        print(f"  {param_count/1e6:.0f}M params, {mem_gb:.1f} GB BF16")

        if qkv_variant is not None:
            print(f"  Patching QKV (variant={qkv_variant})...")
            patch_gpt_oss_model(model, variant=qkv_variant)

        if moe_patch:
            print(f"  Patching MoE (Triton grouped GEMM)...")
            patch_gpt_oss_moe(model, use_triton_gemm=True)

        variant_results = {}
        for bs in batch_sizes:
            batch_prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to("cuda")

            print(f"  Warming up {display_name} (batch_size={bs})...")
            with torch.no_grad():
                _ = model.generate(**inputs, **gen_kwargs)
            torch.cuda.synchronize()

            print(f"  Benchmarking {display_name} (batch_size={bs})...")
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
            total_tokens = bs * num_new_tokens
            mean_time = sum(times) / len(times)
            stddev_time = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

            variant_results[bs] = {
                "mean_time": mean_time,
                "stddev_time": stddev_time,
                "num_new_tokens": num_new_tokens,
                "total_tokens": total_tokens,
                "tps": total_tokens / mean_time,
            }

        all_results[display_name] = variant_results

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # --- Report ---
    print(f"\n{'─'*130}")
    print(f"  Results for {model_label} (BF16, QKV + MoE Fusion)")
    print(f"  NOTE: Scaled-down model (4 layers, 64 experts). See single-op for kernel speedups.")
    print(f"{'─'*130}")

    header = f"  {'Batch':>5} {'Tokens':>7} | {'Original':>16} {'tok/s':>8}"
    for dn, _, _ in test_configs[1:]:
        header += f" | {dn:>16} {'tok/s':>8} {'Speedup':>7}"
    print(header)
    print(f"  {'-'*110}")

    orig = all_results["Original"]
    for bs in batch_sizes:
        o = orig[bs]
        line = (f"  {bs:>5} {o['total_tokens']:>7} | "
                f"{o['mean_time']*1000:>8.1f}±{o['stddev_time']*1000:>5.1f}ms {o['tps']:>7.1f}")
        for dn, _, _ in test_configs[1:]:
            vr = all_results[dn][bs]
            speedup = vr["tps"] / o["tps"]
            line += (f" | {vr['mean_time']*1000:>8.1f}±{vr['stddev_time']*1000:>4.1f}ms "
                     f"{vr['tps']:>7.1f} {speedup:>6.3f}x")
        print(line)

    # JSON entries
    for dn, _, _ in test_configs[1:]:
        for bs in batch_sizes:
            vr = all_results[dn][bs]
            o = orig[bs]
            json_entries.append({
                "config": {
                    "benchmark_id": f"moe_e2e_{dn}_bf16_{bs}",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "hardware": _get_hardware_info(),
                    "software": _get_software_info(dn),
                    "model": {
                        "name": model_label,
                        "label": f"{model_label} (scaled-down)",
                        "precision": "BF16",
                        "description": (
                            f"Scaled-down GPT-OSS: {model_config.num_hidden_layers} layers, "
                            f"{model_config.num_local_experts} experts, "
                            f"h={model_config.hidden_size}, inter={model_config.intermediate_size}. "
                            f"Full GPT-OSS-20B (36L, 128E) exceeds single-GPU memory."
                        ),
                    },
                    "workload": {
                        "batch_size": bs,
                        "input_tokens": "variable",
                        "output_tokens": vr["num_new_tokens"],
                        "sampling": {"temperature": 0, "top_p": 1.0},
                    },
                },
                "metrics": {
                    "throughput": {
                        "tokens_per_second": round(vr["tps"], 1),
                    },
                    "latency": {
                        "mean_ms": round(vr["mean_time"] * 1000, 1),
                        "stddev_ms": round(vr["stddev_time"] * 1000, 1),
                    },
                    "speedup": {
                        "vs_baseline": round(vr["tps"] / o["tps"], 4),
                        "baseline_description": f"Unpatched {model_label} (BF16)",
                    },
                },
            })

    _save_json_results(json_entries, prefix="moe_e2e_bf16")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-op", action="store_true", help="Run single-operation benchmark")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end OPT benchmark")
    parser.add_argument("--llama-single-op", action="store_true", help="Run Llama single-op benchmark (FP32)")
    parser.add_argument("--llama-e2e", action="store_true", help="Run Llama end-to-end benchmark (FP32)")
    parser.add_argument("--llama-single-op-bf16", action="store_true", help="Run Llama single-op benchmark (BF16)")
    parser.add_argument("--llama-e2e-bf16", action="store_true", help="Run Llama-3.1-8B/3.2-3B E2E benchmark (BF16)")
    parser.add_argument("--gpt-oss-single-op", action="store_true", help="Run GPT-OSS single-op benchmark (BF16)")
    parser.add_argument("--gpt-oss-e2e", action="store_true", help="Run GPT-OSS-20B E2E benchmark (BF16)")
    parser.add_argument("--swiglu-single-op", action="store_true", help="Run SwiGLU single-op benchmark (BF16)")
    parser.add_argument("--swiglu-e2e", action="store_true", help="Run SwiGLU E2E benchmark (all Llama models)")
    parser.add_argument("--gqa-single-op", action="store_true", help="Run GQA decode single-op benchmark (BF16)")
    parser.add_argument("--gqa-e2e", action="store_true", help="Run GQA decode E2E benchmark (Llama models)")
    parser.add_argument("--fp8-single-op", action="store_true", help="Run FP8 single-op benchmark (BF16 vs FP8)")
    parser.add_argument("--fp8-e2e", action="store_true", help="Run FP8 E2E benchmark (Llama BF16 vs FP8)")
    parser.add_argument("--fa3-single-op", action="store_true", help="Run FA-3 SDPA backend single-op benchmark")
    parser.add_argument("--fa3-e2e", action="store_true", help="Run FA-3 prefill E2E benchmark (Llama models)")
    parser.add_argument("--moe-single-op", action="store_true", help="Run MoE single-op benchmark (BF16)")
    parser.add_argument("--moe-e2e", action="store_true", help="Run MoE E2E benchmark (GPT-OSS-20B BF16)")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if not any([args.single_op, args.e2e, args.llama_single_op, args.llama_e2e,
                args.llama_single_op_bf16, args.llama_e2e_bf16,
                args.gpt_oss_single_op, args.gpt_oss_e2e,
                args.swiglu_single_op, args.swiglu_e2e,
                args.gqa_single_op, args.gqa_e2e,
                args.fp8_single_op, args.fp8_e2e,
                args.fa3_single_op, args.fa3_e2e,
                args.moe_single_op, args.moe_e2e, args.all]):
        args.all = True

    if args.single_op or args.all:
        benchmark_single_op()

    if args.e2e or args.all:
        benchmark_end_to_end()

    if args.llama_single_op or args.all:
        benchmark_llama_single_op()

    if args.llama_e2e or args.all:
        benchmark_llama_e2e()

    if args.llama_single_op_bf16 or args.all:
        benchmark_llama_single_op_bf16()

    if args.llama_e2e_bf16 or args.all:
        benchmark_llama_e2e_bf16()

    if args.gpt_oss_single_op or args.all:
        benchmark_gpt_oss_single_op()

    if args.gpt_oss_e2e or args.all:
        benchmark_gpt_oss_e2e()

    if args.swiglu_single_op or args.all:
        benchmark_swiglu_single_op()

    if args.swiglu_e2e or args.all:
        benchmark_swiglu_e2e()

    if args.gqa_single_op or args.all:
        benchmark_gqa_decode_single_op()

    if args.gqa_e2e or args.all:
        benchmark_gqa_decode_e2e()

    if args.fp8_single_op or args.all:
        from src.benchmark_fp8 import benchmark_fp8_single_op
        benchmark_fp8_single_op()

    if args.fp8_e2e or args.all:
        from src.benchmark_fp8 import benchmark_fp8_e2e
        benchmark_fp8_e2e()

    if args.fa3_single_op or args.all:
        from src.benchmark_fa3 import benchmark_fa3_single_op
        benchmark_fa3_single_op()

    if args.fa3_e2e or args.all:
        from src.benchmark_fa3 import benchmark_fa3_e2e
        benchmark_fa3_e2e()

    if args.moe_single_op or args.all:
        benchmark_moe_single_op()

    if args.moe_e2e or args.all:
        benchmark_moe_e2e()
