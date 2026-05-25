#!/usr/bin/env python3
"""Consolidate all per-run benchmark JSON files into a single all_benchmarks.json.

Usage:
    python3 scripts/consolidate_results.py

Merges all results/*.json files (excluding all_benchmarks.json itself) into
results/all_benchmarks.json. Deduplicates by benchmark_id, keeping the entry
with the latest timestamp. Adds benchmark_type and benchmark_category fields.
"""

import glob
import json
import os
import sys


def infer_benchmark_type(entry):
    """Infer 'single_op' or 'e2e' from metrics keys."""
    metrics = entry.get("metrics", {})
    if "single_op" in metrics:
        return "single_op"
    if "throughput" in metrics:
        return "e2e"
    return "unknown"


def infer_benchmark_category(entry):
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


def consolidate(results_dir):
    """Load all per-run JSON files, merge, deduplicate, and write consolidated file."""
    consolidated_path = os.path.join(results_dir, "all_benchmarks.json")
    json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    # Exclude the consolidated file itself
    json_files = [f for f in json_files if os.path.basename(f) != "all_benchmarks.json"]

    if not json_files:
        print("No JSON result files found in", results_dir)
        return

    # Load all entries
    all_entries = []
    for filepath in json_files:
        with open(filepath) as f:
            data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                entry["_source_file"] = os.path.basename(filepath)
                all_entries.append(entry)
        else:
            data["_source_file"] = os.path.basename(filepath)
            all_entries.append(data)

    print(f"Loaded {len(all_entries)} entries from {len(json_files)} files")

    # Deduplicate by benchmark_id — keep entry with latest timestamp
    seen = {}
    for entry in all_entries:
        bid = entry["config"]["benchmark_id"]
        ts = entry["config"]["timestamp_utc"]
        if bid not in seen or ts > seen[bid]["config"]["timestamp_utc"]:
            seen[bid] = entry

    deduped = list(seen.values())
    removed = len(all_entries) - len(deduped)
    if removed:
        print(f"Deduplicated: removed {removed} duplicate entries")

    # Add benchmark_type and benchmark_category, remove _source_file
    for entry in deduped:
        entry["benchmark_type"] = infer_benchmark_type(entry)
        entry["benchmark_category"] = infer_benchmark_category(entry)
        entry.pop("_source_file", None)

    # Sort by category then benchmark_id for stable output
    deduped.sort(key=lambda e: (e["benchmark_category"], e["config"]["benchmark_id"]))

    with open(consolidated_path, "w") as f:
        json.dump(deduped, f, indent=2)

    print(f"Wrote {len(deduped)} entries to {consolidated_path}")

    # Summary by category
    cats = {}
    for entry in deduped:
        cat = entry["benchmark_category"]
        btype = entry["benchmark_type"]
        key = f"{cat}/{btype}"
        cats[key] = cats.get(key, 0) + 1
    print("\nBreakdown by category/type:")
    for key in sorted(cats):
        print(f"  {key}: {cats[key]}")

    return consolidated_path


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    consolidate(results_dir)
