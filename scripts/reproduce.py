#!/usr/bin/env python3
"""Generate step-by-step reproduction instructions for a benchmark result.

Usage:
    # By benchmark_id (exact or substring match):
    python3 scripts/reproduce.py fa3_attn_Llama-31-8B_cudnn_bf16_b16_s1024

    # By index in all_benchmarks.json:
    python3 scripts/reproduce.py --index 0

    # List all benchmark IDs matching a pattern:
    python3 scripts/reproduce.py --list fa3

    # Show the raw JSON entry:
    python3 scripts/reproduce.py --json fa3_attn_Llama-31-8B_cudnn_bf16_b16_s1024
"""

import argparse
import json
import os
import sys
import textwrap

RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "all_benchmarks.json",
)

# Maps (benchmark_category, benchmark_type) -> CLI command to reproduce
CATEGORY_COMMANDS = {
    # Main benchmark.py categories
    ("opt_ln", "single_op"): "python3 -m src.benchmark --single-op",
    ("opt_ln", "e2e"): "python3 -m src.benchmark --e2e",
    ("llama_rmsnorm", "single_op"): "python3 -m src.benchmark --llama-single-op",
    ("llama_rmsnorm", "e2e"): "python3 -m src.benchmark --llama-e2e",
    ("llama_rmsnorm_bf16", "single_op"): "python3 -m src.benchmark --llama-single-op-bf16",
    ("llama_rmsnorm_bf16", "e2e"): "python3 -m src.benchmark --llama-e2e-bf16",
    ("gpt_oss", "single_op"): "python3 -m src.benchmark --gpt-oss-single-op",
    ("gpt_oss", "e2e"): "python3 -m src.benchmark --gpt-oss-e2e",
    ("swiglu", "single_op"): "python3 -m src.benchmark --swiglu-single-op",
    ("swiglu", "e2e"): "python3 -m src.benchmark --swiglu-e2e",
    ("gqa_decode", "single_op"): "python3 -m src.benchmark --gqa-single-op",
    ("gqa_decode", "e2e"): "python3 -m src.benchmark --gqa-e2e",
    ("moe", "single_op"): "python3 -m src.benchmark --moe-single-op",
    ("moe", "e2e"): "python3 -m src.benchmark --moe-e2e",
    # Separate benchmark scripts
    ("fa3_attention", "single_op"): "python3 -m src.benchmark_fa3 --single-op",
    ("fa3_attention", "e2e"): "python3 -m src.benchmark_fa3 --e2e",
}

# Maps benchmark_category -> benchmark script file (for source reading)
CATEGORY_SCRIPT = {
    "opt_ln": "src/benchmark.py",
    "llama_rmsnorm": "src/benchmark.py",
    "llama_rmsnorm_bf16": "src/benchmark.py",
    "gpt_oss": "src/benchmark.py",
    "swiglu": "src/benchmark.py",
    "gqa_decode": "src/benchmark.py",
    "moe": "src/benchmark.py",
    "fa3_attention": "src/benchmark_fa3.py",
}


def load_benchmarks():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def find_entry(data, query):
    """Find a benchmark entry by exact id, substring, or index."""
    # Exact match first
    for entry in data:
        if entry["config"]["benchmark_id"] == query:
            return entry
    # Substring match
    matches = [e for e in data if query in e["config"]["benchmark_id"]]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous query '{query}' matched {len(matches)} benchmarks:")
        for m in matches[:20]:
            print(f"  - {m['config']['benchmark_id']}")
        if len(matches) > 20:
            print(f"  ... and {len(matches) - 20} more")
        print("\nPlease use a more specific benchmark_id.")
        sys.exit(1)
    print(f"No benchmark found matching '{query}'.")
    sys.exit(1)


def format_hardware(hw):
    lines = []
    lines.append(f"  GPU:  {hw['gpu_model']} x{hw['gpu_count']} ({hw['gpu_memory_gb']:.0f} GB)")
    lines.append(f"  CPU:  {hw['host_cpu']}")
    lines.append(f"  RAM:  {hw['host_ram_gb']} GB")
    return "\n".join(lines)


def format_software(sw):
    lines = []
    lines.append(f"  OS:        {sw['os']}")
    lines.append(f"  CUDA:      {sw['driver']}")
    lines.append(f"  PyTorch:   {sw['framework_version']}")
    lines.append(f"  Variant:   {sw['runtime_version']}")
    return "\n".join(lines)


def format_workload(wl, btype):
    lines = []
    if "batch_size" in wl:
        lines.append(f"  Batch size:       {wl['batch_size']}")
    if "sequence_length" in wl:
        lines.append(f"  Sequence length:  {wl['sequence_length']}")
    if "context_length" in wl:
        lines.append(f"  Context length:   {wl['context_length']}")
    if "input_tokens" in wl:
        lines.append(f"  Input tokens:     {wl['input_tokens']}")
    if "output_tokens" in wl:
        lines.append(f"  Output tokens:    {wl['output_tokens']}")
    if "dimensions" in wl:
        dims = wl["dimensions"]
        parts = [f"{k}={v}" for k, v in dims.items()]
        lines.append(f"  Dimensions:       {', '.join(parts)}")
    if "sampling" in wl:
        s = wl["sampling"]
        lines.append(f"  Sampling:         temp={s.get('temperature', 'N/A')}, top_p={s.get('top_p', 'N/A')}")
    return "\n".join(lines)


def format_metrics(metrics, btype):
    lines = []
    if "single_op" in metrics:
        so = metrics["single_op"]
        lines.append(f"  Kernel latency:   {so['mean_ms']:.4f} ms +/- {so['stddev_ms']:.4f} ms")
        lines.append(f"  Variant:          {so['variant']}")
    if "throughput" in metrics:
        tp = metrics["throughput"]
        lines.append(f"  Throughput:       {tp.get('tokens_per_sec', 'N/A')} tok/s (mean of {tp.get('num_runs', '?')} runs)")
        if "stddev_tokens_per_sec" in tp:
            lines.append(f"  Throughput std:   {tp['stddev_tokens_per_sec']:.2f} tok/s")
    if "latency" in metrics:
        lat = metrics["latency"]
        if "mean_ms" in lat:
            lines.append(f"  Latency:          {lat['mean_ms']:.2f} ms")
    if "speedup" in metrics:
        sp = metrics["speedup"]
        for k, v in sp.items():
            if k.startswith("vs_"):
                baseline_name = k[3:]
                lines.append(f"  Speedup vs {baseline_name}: {v:.4f}x")
            elif k.endswith("_mean_ms"):
                baseline_name = k.replace("_mean_ms", "")
                lines.append(f"  {baseline_name} baseline: {v:.4f} ms")
    if "accuracy" in metrics:
        acc = metrics["accuracy"]
        if isinstance(acc, dict):
            for k, v in acc.items():
                lines.append(f"  Accuracy {k}: {v}")
    return "\n".join(lines)


def format_baselines(metrics):
    """Extract baseline info from speedup dict for reproduction instructions."""
    sp = metrics.get("speedup", {})
    lines = []
    if "baseline_description" in sp:
        lines.append(f"  - {sp['baseline_description']}")
        if "baseline_mean_ms" in sp:
            lines.append(f"    (original baseline: {sp['baseline_mean_ms']:.4f} ms)")
    if "vs_sdpa" in sp:
        lines.append(f"  - PyTorch SDPA (scaled dot-product attention)")
        if "sdpa_mean_ms" in sp:
            lines.append(f"    (original baseline: {sp['sdpa_mean_ms']:.4f} ms)")
    if "vs_flash" in sp:
        lines.append(f"  - Flash Attention v2")
        if "flash_mean_ms" in sp:
            lines.append(f"    (original baseline: {sp['flash_mean_ms']:.4f} ms)")
    if "vs_default" in sp:
        lines.append(f"  - PyTorch default attention backend")
        if "default_mean_ms" in sp:
            lines.append(f"    (original baseline: {sp['default_mean_ms']:.4f} ms)")
    return "\n".join(lines) if lines else "  - (no baseline info recorded)"


def generate_instructions(entry):
    config = entry["config"]
    bid = config["benchmark_id"]
    cat = entry["benchmark_category"]
    btype = entry["benchmark_type"]
    hw = config["hardware"]
    sw = config["software"]
    model = config["model"]
    wl = config["workload"]
    metrics = entry["metrics"]
    timestamp = config.get("timestamp_utc", "N/A")

    cmd = CATEGORY_COMMANDS.get((cat, btype), "# Unknown category/type combination")
    script = CATEGORY_SCRIPT.get(cat, "src/benchmark.py")

    model_name = model.get("label", model.get("name", "N/A"))
    precision = model.get("precision", "N/A")

    sep = "=" * 72
    thin_sep = "-" * 72

    output = f"""\
{sep}
BENCHMARK REPRODUCTION GUIDE
{sep}

Benchmark ID:  {bid}
Category:      {cat}
Type:          {btype}
Timestamp:     {timestamp}

{thin_sep}
STEP 1: Hardware Requirements
{thin_sep}

Ensure you have matching (or equivalent) hardware:

{format_hardware(hw)}

  NOTE: This benchmark was run on an NVIDIA H100 80GB HBM3.
  Results will differ on other GPU architectures.

{thin_sep}
STEP 2: Software Environment
{thin_sep}

Verify your software stack matches:

{format_software(sw)}

  Install dependencies if needed:
    bash scripts/install_deps.sh

{thin_sep}
STEP 3: Lock GPU Clocks (Recommended)
{thin_sep}

  For reproducible results, lock GPU clock frequencies:

    sudo bash scripts/lock_clocks.sh

  This prevents thermal throttling and frequency scaling from
  affecting measurements.

{thin_sep}
STEP 4: Understand the Workload
{thin_sep}

  Model:     {model_name}
  Precision: {precision}

{format_workload(wl, btype)}

  The benchmark script ({script}) has these parameters
  hardcoded. The command below runs all configurations for this
  category, including this specific workload.

{thin_sep}
STEP 5: Run the Benchmark
{thin_sep}

  From the project root directory:

    cd {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    {cmd}

  This will run ALL {cat}/{btype} benchmarks, which includes the
  specific configuration for "{bid}".

  NOTE: Baselines are measured automatically by the same command.
  No separate step is needed. Speedup is computed inline.

  Baseline(s) for this benchmark:
{format_baselines(metrics)}

  Results are saved to:
    results/  (timestamped JSON files)

  To consolidate into all_benchmarks.json:
    python3 scripts/consolidate_results.py

{thin_sep}
STEP 6: Verify Results
{thin_sep}

  Original recorded metrics:

{format_metrics(metrics, btype)}

  Compare your new results against these values. Variation of
  up to ~5% is normal due to system noise. Larger differences
  may indicate:
    - Different GPU clocks (run scripts/lock_clocks.sh)
    - Thermal throttling
    - Different driver/PyTorch version
    - Background processes consuming GPU resources

  Both the optimized kernel and baseline timings should be close
  to the values above. If only one changes, the difference is in
  that specific implementation. If both shift proportionally, it's
  likely a system-level factor (clocks, thermals, driver).

{thin_sep}
STEP 7: (Optional) Profile with Nsight Systems
{thin_sep}

  For detailed kernel-level profiling:

    FUSED_LN_NVTX=1 bash scripts/profile_nsys.sh {btype.replace('_', '-')}

  This generates an .nsys-rep file you can open in Nsight Systems UI.

{thin_sep}
STEP 8: (Optional) Run Correctness Tests
{thin_sep}

  Verify kernel correctness against PyTorch reference:

    python3 -m src.test_correctness

{sep}
"""
    return output


def list_benchmarks(data, pattern=None):
    ids = [e["config"]["benchmark_id"] for e in data]
    if pattern:
        ids = [bid for bid in ids if pattern in bid]
    if not ids:
        print(f"No benchmarks matching '{pattern}'.")
        return
    print(f"Found {len(ids)} benchmark(s):")
    for bid in ids:
        print(f"  {bid}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate step-by-step reproduction instructions for a benchmark result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s fa3_attn_Llama-31-8B_cudnn_bf16_b16_s1024
              %(prog)s --index 42
              %(prog)s --list fa3
              %(prog)s --json moe_e2e_QKV-V1+MoE_bf16_1
        """),
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Benchmark ID (exact or substring match)",
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        help="Lookup by index in all_benchmarks.json (0-based)",
    )
    parser.add_argument(
        "--list", "-l",
        nargs="?",
        const="",
        metavar="PATTERN",
        help="List all benchmark IDs, optionally filtered by PATTERN",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Print the raw JSON entry instead of instructions",
    )

    args = parser.parse_args()

    if args.list is not None and args.query is None and args.index is None:
        data = load_benchmarks()
        list_benchmarks(data, args.list if args.list else None)
        return

    if args.query is None and args.index is None:
        parser.print_help()
        sys.exit(1)

    data = load_benchmarks()

    if args.index is not None:
        if args.index < 0 or args.index >= len(data):
            print(f"Index {args.index} out of range (0-{len(data) - 1}).")
            sys.exit(1)
        entry = data[args.index]
    else:
        entry = find_entry(data, args.query)

    if args.json:
        print(json.dumps(entry, indent=2))
    else:
        print(generate_instructions(entry))


if __name__ == "__main__":
    main()
