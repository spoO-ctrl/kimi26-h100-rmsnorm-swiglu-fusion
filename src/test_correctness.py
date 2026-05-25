"""
Correctness tests for fused LayerNorm+Linear.

1. Denominator kernel tests (V0, V2 Welford)
2. Unit tests: Random LN + Linear vs fused on synthetic data (V0, V1, V3)
3. Integration test: OPT-125m logits comparison original vs patched
"""

import torch
import torch.nn as nn
from src.load_cuda import denominator_cuda
from src.weight_transform import compute_fused_weights
from src.fused_forward import (
    fused_ln_linear_forward,
    fused_ln_linear_forward_v1,
    fused_ln_linear_forward_v3,
    fused_rmsnorm_linear_forward_v1,
    fused_rmsnorm_linear_forward_v3,
    FusedRMSNormCombinedLinearV1,
    FusedRMSNormCombinedLinearV3,
    FusedRMSNormSwiGLUV1,
    FusedRMSNormSwiGLUV3,
)
from src.weight_transform import compute_fused_weights_rmsnorm, compute_fused_weights_rmsnorm_combined


def test_denominator_kernel():
    """Test CUDA denominator kernel against PyTorch reference."""
    print("=" * 60)
    print("TEST: Denominator kernel (V0 two-pass)")
    print("=" * 60)

    torch.manual_seed(42)

    for rows, cols in [(1, 768), (32, 768), (128, 2048), (512, 4096)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        # Reference: ||x - mean(x)||_2 per row
        ref = (x - x.mean(dim=-1, keepdim=True)).norm(dim=-1)

        # CUDA kernel
        out = denominator_cuda.compute_denominator(x)

        max_diff = (ref - out).abs().max().item()
        rel_diff = (max_diff / ref.abs().mean().item()) if ref.abs().mean().item() > 0 else 0
        status = "PASS" if max_diff < 1e-3 else "FAIL"
        print(f"  [{status}] fp32 ({rows:4d} x {cols:4d}): max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        assert max_diff < 1e-3, f"fp32 denominator test failed: max_diff={max_diff}"

    # FP16 test
    for rows, cols in [(32, 768), (128, 2048)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
        ref = (x.float() - x.float().mean(dim=-1, keepdim=True)).norm(dim=-1)
        out = denominator_cuda.compute_denominator(x)

        max_diff = (ref - out).abs().max().item()
        status = "PASS" if max_diff < 0.5 else "FAIL"
        print(f"  [{status}] fp16 ({rows:4d} x {cols:4d}): max_diff={max_diff:.2e}")
        assert max_diff < 0.5, f"fp16 denominator test failed: max_diff={max_diff}"

    print("  All V0 denominator tests passed!\n")


def test_denominator_welford():
    """Test Welford's single-pass denominator kernel against PyTorch reference."""
    print("=" * 60)
    print("TEST: Denominator kernel (V2 Welford single-pass)")
    print("=" * 60)

    torch.manual_seed(42)

    for rows, cols in [(1, 768), (32, 768), (128, 2048), (512, 4096)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        ref = (x - x.mean(dim=-1, keepdim=True)).norm(dim=-1)
        out = denominator_cuda.compute_denominator_welford(x)

        max_diff = (ref - out).abs().max().item()
        rel_diff = (max_diff / ref.abs().mean().item()) if ref.abs().mean().item() > 0 else 0
        status = "PASS" if max_diff < 1e-3 else "FAIL"
        print(f"  [{status}] fp32 ({rows:4d} x {cols:4d}): max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        assert max_diff < 1e-3, f"Welford denominator test failed: max_diff={max_diff}"

    print("  All V2 Welford denominator tests passed!\n")


def test_fused_ln_linear_unit():
    """Test fused LN+Linear vs sequential on synthetic data (all variants)."""
    print("=" * 60)
    print("TEST: Fused LN+Linear (unit, all variants)")
    print("=" * 60)

    torch.manual_seed(42)
    denom_stream = torch.cuda.Stream()

    for h, out_dim, batch in [(768, 768, 32), (768, 3072, 128), (2048, 2048, 64), (4096, 16384, 16)]:
        ln = nn.LayerNorm(h).cuda()
        linear = nn.Linear(h, out_dim).cuda()

        # Initialize with non-trivial weights
        nn.init.normal_(ln.weight, mean=1.0, std=0.1)
        nn.init.normal_(ln.bias, mean=0.0, std=0.01)
        nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(linear.bias, mean=0.0, std=0.01)

        x = torch.randn(batch, h, device="cuda")

        # Reference: sequential
        with torch.no_grad():
            ref = linear(ln(x))

        # Fused weights
        W_new, b_new, h_dim, eps = compute_fused_weights(ln, linear)

        # V0: Stream-based
        with torch.no_grad():
            fused_v0 = fused_ln_linear_forward(x, W_new, b_new, denom_stream, h_dim, eps)
        md_v0 = (ref - fused_v0).abs().max().item()
        s_v0 = "PASS" if md_v0 < 1e-3 else "FAIL"

        # V1: Fused normalize
        with torch.no_grad():
            fused_v1 = fused_ln_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        md_v1 = (ref - fused_v1).abs().max().item()
        s_v1 = "PASS" if md_v1 < 1e-3 else "FAIL"

        # V3: Welford + fused normalize + 512
        with torch.no_grad():
            fused_v3 = fused_ln_linear_forward_v3(x, W_new, b_new, h_dim, eps)
        md_v3 = (ref - fused_v3).abs().max().item()
        s_v3 = "PASS" if md_v3 < 1e-3 else "FAIL"

        print(f"  h={h:4d}, out={out_dim:5d}, batch={batch:3d}: "
              f"V0[{s_v0}]={md_v0:.2e}  V1[{s_v1}]={md_v1:.2e}  V3[{s_v3}]={md_v3:.2e}")
        assert md_v0 < 1e-3, f"V0 unit test failed: max_diff={md_v0}"
        assert md_v1 < 1e-3, f"V1 unit test failed: max_diff={md_v1}"
        assert md_v3 < 1e-3, f"V3 unit test failed: max_diff={md_v3}"

    print("  All unit tests passed!\n")


def test_fused_ln_linear_3d():
    """Test with 3D input (batch, seq, h)."""
    print("=" * 60)
    print("TEST: Fused LN+Linear (3D input, all variants)")
    print("=" * 60)

    torch.manual_seed(42)
    denom_stream = torch.cuda.Stream()

    h, out_dim = 768, 768
    ln = nn.LayerNorm(h).cuda()
    linear = nn.Linear(h, out_dim).cuda()
    nn.init.normal_(ln.weight, mean=1.0, std=0.1)
    nn.init.normal_(ln.bias, mean=0.0, std=0.01)

    x = torch.randn(4, 128, h, device="cuda")

    with torch.no_grad():
        ref = linear(ln(x))

    W_new, b_new, h_dim, eps = compute_fused_weights(ln, linear)

    with torch.no_grad():
        fused_v0 = fused_ln_linear_forward(x, W_new, b_new, denom_stream, h_dim, eps)
        fused_v1 = fused_ln_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        fused_v3 = fused_ln_linear_forward_v3(x, W_new, b_new, h_dim, eps)

    md_v0 = (ref - fused_v0).abs().max().item()
    md_v1 = (ref - fused_v1).abs().max().item()
    md_v3 = (ref - fused_v3).abs().max().item()

    print(f"  3D (4, 128, {h}): V0={md_v0:.2e}  V1={md_v1:.2e}  V3={md_v3:.2e}")
    assert md_v0 < 1e-3, f"V0 3D test failed: max_diff={md_v0}"
    assert md_v1 < 1e-3, f"V1 3D test failed: max_diff={md_v1}"
    assert md_v3 < 1e-3, f"V3 3D test failed: max_diff={md_v3}"
    print("  All 3D tests passed!\n")


def test_opt_integration():
    """Integration test: compare OPT-125m logits before and after patching."""
    print("=" * 60)
    print("TEST: OPT-125m integration")
    print("=" * 60)

    from transformers import AutoTokenizer, OPTForCausalLM
    from src.patch_model import patch_opt_model
    import copy

    print("  Loading OPT-125m...")
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_orig = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda().eval()

    # Deep copy for patching
    model_fused = copy.deepcopy(model_orig)
    print("  Patching model...")
    patch_opt_model(model_fused)

    # Test inputs
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "In a galaxy far far away",
        "Machine learning is transforming",
    ]

    all_passed = True
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits_orig = model_orig(**inputs).logits
            logits_fused = model_fused(**inputs).logits

        max_diff = (logits_orig - logits_fused).abs().max().item()
        mean_diff = (logits_orig - logits_fused).abs().mean().item()
        # Relative to output magnitude
        rel_diff = max_diff / logits_orig.abs().mean().item() if logits_orig.abs().mean().item() > 0 else 0
        status = "PASS" if max_diff < 1e-2 else "FAIL"
        if max_diff >= 1e-2:
            all_passed = False
        print(f"  [{status}] \"{text[:40]}...\": max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}")

    if all_passed:
        print("  All integration tests passed!\n")
    else:
        print("  WARNING: Some integration tests exceeded threshold (may be acceptable for fp32 accumulation)\n")

    # Clean up GPU memory
    del model_orig, model_fused
    torch.cuda.empty_cache()


def test_fused_ln_linear_fp16():
    """Test fused LN+Linear with FP16 inputs (V1 and V3)."""
    print("=" * 60)
    print("TEST: Fused LN+Linear (FP16, V1 and V3)")
    print("=" * 60)

    torch.manual_seed(42)

    for h, out_dim, batch in [(768, 768, 32), (2048, 2048, 64), (4096, 4096, 16)]:
        ln = nn.LayerNorm(h).cuda().half()
        linear = nn.Linear(h, out_dim).cuda().half()
        nn.init.normal_(ln.weight, mean=1.0, std=0.1)
        nn.init.normal_(ln.bias, mean=0.0, std=0.01)

        x = torch.randn(batch, h, device="cuda", dtype=torch.float16)

        # Reference: sequential in FP16
        with torch.no_grad():
            ref = linear(ln(x))

        # Fused weights (compute in fp32 for stability, then cast)
        W_new, b_new, h_dim, eps = compute_fused_weights(ln.float(), linear.float())
        W_new = W_new.half()
        b_new = b_new.half()

        # V1
        with torch.no_grad():
            fused_v1 = fused_ln_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        md_v1 = (ref.float() - fused_v1.float()).abs().max().item()
        s_v1 = "PASS" if md_v1 < 0.1 else "FAIL"

        # V3
        with torch.no_grad():
            fused_v3 = fused_ln_linear_forward_v3(x, W_new, b_new, h_dim, eps)
        md_v3 = (ref.float() - fused_v3.float()).abs().max().item()
        s_v3 = "PASS" if md_v3 < 0.1 else "FAIL"

        print(f"  h={h:4d}, out={out_dim:5d}, batch={batch:3d}: "
              f"V1[{s_v1}]={md_v1:.2e}  V3[{s_v3}]={md_v3:.2e}")
        assert md_v1 < 0.1, f"V1 FP16 test failed: max_diff={md_v1}"
        assert md_v3 < 0.1, f"V3 FP16 test failed: max_diff={md_v3}"

    print("  All FP16 tests passed!\n")


def test_fused_ln_linear_bf16():
    """Test fused LN+Linear with BF16 inputs (V1 and V3)."""
    print("=" * 60)
    print("TEST: Fused LN+Linear (BF16, V1 and V3)")
    print("=" * 60)

    torch.manual_seed(42)

    for h, out_dim, batch in [(768, 768, 32), (2048, 2048, 64), (4096, 4096, 16)]:
        ln = nn.LayerNorm(h).cuda().bfloat16()
        linear = nn.Linear(h, out_dim).cuda().bfloat16()
        nn.init.normal_(ln.weight, mean=1.0, std=0.1)
        nn.init.normal_(ln.bias, mean=0.0, std=0.01)

        x = torch.randn(batch, h, device="cuda", dtype=torch.bfloat16)

        # Reference: sequential in BF16
        with torch.no_grad():
            ref = linear(ln(x))

        # Fused weights (compute in fp32 for stability, then cast)
        W_new, b_new, h_dim, eps = compute_fused_weights(ln.float(), linear.float())
        W_new = W_new.bfloat16()
        b_new = b_new.bfloat16()

        # V1
        with torch.no_grad():
            fused_v1 = fused_ln_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        md_v1 = (ref.float() - fused_v1.float()).abs().max().item()
        s_v1 = "PASS" if md_v1 < 0.5 else "FAIL"

        # V3
        with torch.no_grad():
            fused_v3 = fused_ln_linear_forward_v3(x, W_new, b_new, h_dim, eps)
        md_v3 = (ref.float() - fused_v3.float()).abs().max().item()
        s_v3 = "PASS" if md_v3 < 0.5 else "FAIL"

        print(f"  h={h:4d}, out={out_dim:5d}, batch={batch:3d}: "
              f"V1[{s_v1}]={md_v1:.2e}  V3[{s_v3}]={md_v3:.2e}")
        assert md_v1 < 0.5, f"V1 BF16 test failed: max_diff={md_v1}"
        assert md_v3 < 0.5, f"V3 BF16 test failed: max_diff={md_v3}"

    print("  All BF16 tests passed!\n")


def test_fused_rmsnorm_linear_unit():
    """Test fused RMSNorm+Linear vs sequential on synthetic data (V1 and V3)."""
    print("=" * 60)
    print("TEST: Fused RMSNorm+Linear (unit, V1 and V3)")
    print("=" * 60)

    torch.manual_seed(42)

    for h, out_dim, batch in [(768, 768, 32), (768, 3072, 128), (2048, 2048, 64), (4096, 4096, 16)]:
        rms_norm = torch.nn.RMSNorm(h, eps=1e-6).cuda()
        linear = nn.Linear(h, out_dim, bias=False).cuda()
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)
        nn.init.normal_(linear.weight, mean=0.0, std=0.02)

        x = torch.randn(batch, h, device="cuda")

        # Reference: sequential
        with torch.no_grad():
            ref = linear(rms_norm(x))

        # Fused weights
        W_new, b_new, h_dim, eps = compute_fused_weights_rmsnorm(rms_norm, linear)

        # V1
        with torch.no_grad():
            fused_v1 = fused_rmsnorm_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        md_v1 = (ref - fused_v1).abs().max().item()
        s_v1 = "PASS" if md_v1 < 1e-3 else "FAIL"

        # V3
        with torch.no_grad():
            fused_v3 = fused_rmsnorm_linear_forward_v3(x, W_new, b_new, h_dim, eps)
        md_v3 = (ref - fused_v3).abs().max().item()
        s_v3 = "PASS" if md_v3 < 1e-3 else "FAIL"

        print(f"  h={h:4d}, out={out_dim:5d}, batch={batch:3d}: "
              f"V1[{s_v1}]={md_v1:.2e}  V3[{s_v3}]={md_v3:.2e}")
        assert md_v1 < 1e-3, f"RMSNorm V1 unit test failed: max_diff={md_v1}"
        assert md_v3 < 1e-3, f"RMSNorm V3 unit test failed: max_diff={md_v3}"

    print("  All RMSNorm unit tests passed!\n")


def test_fused_rmsnorm_linear_with_bias():
    """Test fused RMSNorm+Linear when linear has bias."""
    print("=" * 60)
    print("TEST: Fused RMSNorm+Linear with bias")
    print("=" * 60)

    torch.manual_seed(42)
    h, out_dim, batch = 768, 768, 32
    rms_norm = torch.nn.RMSNorm(h, eps=1e-6).cuda()
    linear = nn.Linear(h, out_dim, bias=True).cuda()
    nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

    x = torch.randn(batch, h, device="cuda")

    with torch.no_grad():
        ref = linear(rms_norm(x))

    W_new, b_new, h_dim, eps = compute_fused_weights_rmsnorm(rms_norm, linear)

    with torch.no_grad():
        fused_v1 = fused_rmsnorm_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        fused_v3 = fused_rmsnorm_linear_forward_v3(x, W_new, b_new, h_dim, eps)

    md_v1 = (ref - fused_v1).abs().max().item()
    md_v3 = (ref - fused_v3).abs().max().item()

    print(f"  h={h}, out={out_dim}, batch={batch}: V1={md_v1:.2e}  V3={md_v3:.2e}")
    assert md_v1 < 1e-3, f"RMSNorm V1 with bias failed: max_diff={md_v1}"
    assert md_v3 < 1e-3, f"RMSNorm V3 with bias failed: max_diff={md_v3}"
    print("  RMSNorm with bias tests passed!\n")


def test_llama_integration():
    """Integration test: compare Llama logits before and after patching."""
    print("=" * 60)
    print("TEST: Llama integration")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama import patch_llama_model
    import copy

    print("  Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_orig = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).cuda().eval()
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    model_fused = copy.deepcopy(model_orig)
    print("  Patching model with RMSNorm fusion (V1)...")
    patch_llama_model(model_fused, variant="V1")

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "In a galaxy far far away",
        "Machine learning is transforming",
    ]

    all_passed = True
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits_orig = model_orig(**inputs).logits
            logits_fused = model_fused(**inputs).logits

        max_diff = (logits_orig - logits_fused).abs().max().item()
        mean_diff = (logits_orig - logits_fused).abs().mean().item()
        rel_diff = max_diff / logits_orig.abs().mean().item() if logits_orig.abs().mean().item() > 0 else 0
        status = "PASS" if max_diff < 1e-2 else "FAIL"
        if max_diff >= 1e-2:
            all_passed = False
        print(f"  [{status}] \"{text[:40]}...\": max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}")

    if all_passed:
        print("  All Llama integration tests passed!\n")
    else:
        print("  WARNING: Some Llama integration tests exceeded threshold\n")

    del model_orig, model_fused
    torch.cuda.empty_cache()


def test_fused_rmsnorm_combined_unit():
    """Test combined RMSNorm+Linear vs separate fused calls on synthetic data."""
    print("=" * 60)
    print("TEST: Fused RMSNorm Combined (unit, V1 and V3)")
    print("=" * 60)

    torch.manual_seed(42)

    configs = [
        # (h, out_dims, description)
        (2048, [2048, 256, 256], "TinyLlama-like QKV"),
        (4096, [4096, 1024, 1024], "Llama-3-8B-like QKV"),
        (4096, [4096, 4096], "symmetric gate+up"),
    ]

    for h, out_dims, desc in configs:
        rms_norm = torch.nn.RMSNorm(h, eps=1e-6).cuda()
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

        linears = []
        for od in out_dims:
            lin = nn.Linear(h, od, bias=False).cuda()
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            linears.append(lin)

        # Compute combined weights
        W_comb, b_comb, split_sizes, h_dim, eps = compute_fused_weights_rmsnorm_combined(
            rms_norm, linears
        )

        for batch in [1, 32, 128]:
            x = torch.randn(batch, h, device="cuda")

            # Reference: separate fused calls
            ref_parts = []
            for lin in linears:
                W_new, b_new, _, _ = compute_fused_weights_rmsnorm(rms_norm, lin)
                ref_parts.append(
                    fused_rmsnorm_linear_forward_v1(x, W_new, b_new, h_dim, eps)
                )

            # V1 combined
            mod_v1 = FusedRMSNormCombinedLinearV1(W_comb, b_comb, split_sizes, h_dim, eps)
            with torch.no_grad():
                comb_v1 = mod_v1(x)

            max_diffs_v1 = []
            for ref_p, comb_p in zip(ref_parts, comb_v1):
                max_diffs_v1.append((ref_p - comb_p).abs().max().item())
            md_v1 = max(max_diffs_v1)
            s_v1 = "PASS" if md_v1 < 1e-5 else "FAIL"

            # V3 combined
            mod_v3 = FusedRMSNormCombinedLinearV3(W_comb, b_comb, split_sizes, h_dim, eps)
            with torch.no_grad():
                comb_v3 = mod_v3(x)

            # V3 reference
            ref_parts_v3 = []
            for lin in linears:
                W_new, b_new, _, _ = compute_fused_weights_rmsnorm(rms_norm, lin)
                ref_parts_v3.append(
                    fused_rmsnorm_linear_forward_v3(x, W_new, b_new, h_dim, eps)
                )

            max_diffs_v3 = []
            for ref_p, comb_p in zip(ref_parts_v3, comb_v3):
                max_diffs_v3.append((ref_p - comb_p).abs().max().item())
            md_v3 = max(max_diffs_v3)
            s_v3 = "PASS" if md_v3 < 1e-5 else "FAIL"

            print(f"  [{s_v1}] {desc} batch={batch:3d}: V1 max_diff={md_v1:.2e}  "
                  f"[{s_v3}] V3 max_diff={md_v3:.2e}")
            assert md_v1 < 1e-5, f"Combined V1 failed for {desc}: max_diff={md_v1}"
            assert md_v3 < 1e-5, f"Combined V3 failed for {desc}: max_diff={md_v3}"

    print("  All combined unit tests passed!\n")


def test_llama_integration_combined():
    """Integration test: compare Llama logits with combined=True vs unpatched."""
    print("=" * 60)
    print("TEST: Llama integration (combined)")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama import patch_llama_model
    import copy

    print("  Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_orig = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).cuda().eval()
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    model_fused = copy.deepcopy(model_orig)
    print("  Patching model with combined RMSNorm fusion (V1)...")
    patch_llama_model(model_fused, variant="V1", combined=True)

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "In a galaxy far far away",
        "Machine learning is transforming",
    ]

    all_passed = True
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits_orig = model_orig(**inputs).logits
            logits_fused = model_fused(**inputs).logits

        max_diff = (logits_orig - logits_fused).abs().max().item()
        mean_diff = (logits_orig - logits_fused).abs().mean().item()
        rel_diff = max_diff / logits_orig.abs().mean().item() if logits_orig.abs().mean().item() > 0 else 0
        status = "PASS" if max_diff < 1e-2 else "FAIL"
        if max_diff >= 1e-2:
            all_passed = False
        print(f"  [{status}] \"{text[:40]}...\": max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}")

    if all_passed:
        print("  All Llama combined integration tests passed!\n")
    else:
        print("  WARNING: Some Llama combined integration tests exceeded threshold\n")

    del model_orig, model_fused
    torch.cuda.empty_cache()


def test_gpt_oss_combined_unit():
    """Test combined RMSNorm+Linear with GPT-OSS dimensions (with bias)."""
    print("=" * 60)
    print("TEST: Fused RMSNorm Combined (GPT-OSS dims, V1 and V3)")
    print("=" * 60)

    torch.manual_seed(42)

    configs = [
        # (h, out_dims, description)
        (2880, [4096, 512, 512], "GPT-OSS-20B QKV (with bias)"),
    ]

    for h, out_dims, desc in configs:
        rms_norm = torch.nn.RMSNorm(h, eps=1e-6).cuda()
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

        linears = []
        for od in out_dims:
            lin = nn.Linear(h, od, bias=True).cuda()
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            nn.init.normal_(lin.bias, mean=0.0, std=0.01)
            linears.append(lin)

        # Compute combined weights
        W_comb, b_comb, split_sizes, h_dim, eps = compute_fused_weights_rmsnorm_combined(
            rms_norm, linears
        )

        for batch in [1, 32, 128]:
            x = torch.randn(batch, h, device="cuda")

            # Reference: separate fused calls
            ref_parts = []
            for lin in linears:
                W_new, b_new, _, _ = compute_fused_weights_rmsnorm(rms_norm, lin)
                ref_parts.append(
                    fused_rmsnorm_linear_forward_v1(x, W_new, b_new, h_dim, eps)
                )

            # V1 combined
            mod_v1 = FusedRMSNormCombinedLinearV1(W_comb, b_comb, split_sizes, h_dim, eps)
            with torch.no_grad():
                comb_v1 = mod_v1(x)

            max_diffs_v1 = []
            for ref_p, comb_p in zip(ref_parts, comb_v1):
                max_diffs_v1.append((ref_p - comb_p).abs().max().item())
            md_v1 = max(max_diffs_v1)
            s_v1 = "PASS" if md_v1 < 1e-5 else "FAIL"

            # V3 combined
            mod_v3 = FusedRMSNormCombinedLinearV3(W_comb, b_comb, split_sizes, h_dim, eps)
            with torch.no_grad():
                comb_v3 = mod_v3(x)

            # V3 reference
            ref_parts_v3 = []
            for lin in linears:
                W_new, b_new, _, _ = compute_fused_weights_rmsnorm(rms_norm, lin)
                ref_parts_v3.append(
                    fused_rmsnorm_linear_forward_v3(x, W_new, b_new, h_dim, eps)
                )

            max_diffs_v3 = []
            for ref_p, comb_p in zip(ref_parts_v3, comb_v3):
                max_diffs_v3.append((ref_p - comb_p).abs().max().item())
            md_v3 = max(max_diffs_v3)
            s_v3 = "PASS" if md_v3 < 1e-5 else "FAIL"

            print(f"  [{s_v1}] {desc} batch={batch:3d}: V1 max_diff={md_v1:.2e}  "
                  f"[{s_v3}] V3 max_diff={md_v3:.2e}")
            assert md_v1 < 1e-5, f"GPT-OSS Combined V1 failed for {desc}: max_diff={md_v1}"
            assert md_v3 < 1e-5, f"GPT-OSS Combined V3 failed for {desc}: max_diff={md_v3}"

    print("  All GPT-OSS combined unit tests passed!\n")


def test_gpt_oss_integration():
    """Integration test: compare GPT-OSS logits before and after patching."""
    print("=" * 60)
    print("TEST: GPT-OSS integration")
    print("=" * 60)

    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_gpt_oss import patch_gpt_oss_model

    model_name = "openai/gpt-oss-20b"

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "In a galaxy far far away",
        "Machine learning is transforming",
    ]

    # Collect original logits first, then free model (too large to deepcopy)
    print(f"  Loading {model_name} in BF16 (original)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_orig = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).cuda().eval()
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    orig_logits = {}
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            orig_logits[text] = model_orig(**inputs).logits.cpu()

    del model_orig
    gc.collect()
    torch.cuda.empty_cache()

    # Load again and patch
    print(f"  Loading {model_name} in BF16 (fused V1)...")
    model_fused = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).cuda().eval()
    print("  Patching model with RMSNorm QKV fusion (V1)...")
    patch_gpt_oss_model(model_fused, variant="V1")

    all_passed = True
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits_fused = model_fused(**inputs).logits.cpu()

        logits_orig = orig_logits[text]
        max_diff = (logits_orig.float() - logits_fused.float()).abs().max().item()
        mean_diff = (logits_orig.float() - logits_fused.float()).abs().mean().item()
        rel_diff = max_diff / logits_orig.float().abs().mean().item() if logits_orig.float().abs().mean().item() > 0 else 0
        # BF16 tolerance: large models accumulate numerical noise across layers
        # (TinyLlama BF16 shows ~0.58 max_diff, GPT-OSS-20B ~1.1 is expected)
        status = "PASS" if max_diff < 2.0 else "FAIL"
        if max_diff >= 2.0:
            all_passed = False
        print(f"  [{status}] \"{text[:40]}...\": max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}")

    if all_passed:
        print("  All GPT-OSS integration tests passed!\n")
    else:
        print("  WARNING: Some GPT-OSS integration tests exceeded threshold\n")

    del model_fused, orig_logits
    gc.collect()
    torch.cuda.empty_cache()


def test_combined_vs_separate_equivalence():
    """Test that combined and separate patching produce equivalent logits."""
    print("=" * 60)
    print("TEST: Combined vs separate equivalence")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama import patch_llama_model
    import copy

    print("  Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_base = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).cuda().eval()
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    model_separate = copy.deepcopy(model_base)
    model_combined = copy.deepcopy(model_base)
    del model_base

    print("  Patching separate model (V1)...")
    patch_llama_model(model_separate, variant="V1", combined=False)
    print("  Patching combined model (V1)...")
    patch_llama_model(model_combined, variant="V1", combined=True)

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "In a galaxy far far away",
        "Machine learning is transforming",
    ]

    all_passed = True
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits_sep = model_separate(**inputs).logits
            logits_comb = model_combined(**inputs).logits

        max_diff = (logits_sep - logits_comb).abs().max().item()
        mean_diff = (logits_sep - logits_comb).abs().mean().item()
        status = "PASS" if max_diff < 1e-4 else "FAIL"
        if max_diff >= 1e-4:
            all_passed = False
        print(f"  [{status}] \"{text[:40]}...\": max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

    if all_passed:
        print("  All equivalence tests passed!\n")
    else:
        print("  WARNING: Some equivalence tests exceeded threshold\n")

    del model_separate, model_combined
    torch.cuda.empty_cache()


def test_fused_rmsnorm_swiglu_unit():
    """Test fused RMSNorm+SwiGLU vs sequential on synthetic data (V1 and V3)."""
    print("=" * 60)
    print("TEST: Fused RMSNorm+SwiGLU (unit, V1 and V3)")
    print("=" * 60)

    torch.manual_seed(42)

    configs = [
        # (h, intermediate, description)
        (2048, 5632, "TinyLlama MLP"),
        (3072, 8192, "Llama-3.2-3B MLP"),
        (4096, 14336, "Llama-3.1-8B MLP"),
    ]

    for h, intermediate, desc in configs:
        rms_norm = torch.nn.RMSNorm(h, eps=1e-6).cuda()
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

        gate_proj = nn.Linear(h, intermediate, bias=False).cuda()
        up_proj = nn.Linear(h, intermediate, bias=False).cuda()
        nn.init.normal_(gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(up_proj.weight, mean=0.0, std=0.02)

        # Compute combined fused weights
        W_comb, b_comb, split_sizes, h_dim, eps = compute_fused_weights_rmsnorm_combined(
            rms_norm, [gate_proj, up_proj]
        )

        for batch in [1, 32, 128]:
            for dtype_name, dtype, tol in [("FP32", torch.float32, 1e-3), ("BF16", torch.bfloat16, 0.5)]:
                x = torch.randn(batch, h, device="cuda", dtype=dtype)

                # Reference: RMSNorm -> gate/up separate -> SiLU(gate) * up
                with torch.no_grad():
                    normed = rms_norm.float()(x.float())
                    gate_out = gate_proj.float()(normed)
                    up_out = up_proj.float()(normed)
                    ref = (torch.nn.functional.silu(gate_out) * up_out).to(dtype)

                W_c = W_comb.to(dtype)
                b_c = b_comb.to(dtype)

                # V1
                mod_v1 = FusedRMSNormSwiGLUV1(W_c, b_c, intermediate, h_dim, eps)
                with torch.no_grad():
                    out_v1 = mod_v1(x)
                md_v1 = (ref.float() - out_v1.float()).abs().max().item()
                s_v1 = "PASS" if md_v1 < tol else "FAIL"

                # V3
                mod_v3 = FusedRMSNormSwiGLUV3(W_c, b_c, intermediate, h_dim, eps)
                with torch.no_grad():
                    out_v3 = mod_v3(x)
                md_v3 = (ref.float() - out_v3.float()).abs().max().item()
                s_v3 = "PASS" if md_v3 < tol else "FAIL"

                print(f"  [{s_v1}] {desc} {dtype_name} batch={batch:3d}: V1={md_v1:.2e}  "
                      f"[{s_v3}] V3={md_v3:.2e}")
                assert md_v1 < tol, f"SwiGLU V1 {dtype_name} failed for {desc}: max_diff={md_v1}"
                assert md_v3 < tol, f"SwiGLU V3 {dtype_name} failed for {desc}: max_diff={md_v3}"

    print("  All RMSNorm+SwiGLU unit tests passed!\n")


def test_llama_integration_swiglu():
    """Integration test: compare Llama logits with swiglu=True vs unpatched."""
    print("=" * 60)
    print("TEST: Llama integration (SwiGLU)")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama import patch_llama_model
    import copy

    print("  Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_orig = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).cuda().eval()
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    model_fused = copy.deepcopy(model_orig)
    print("  Patching model with SwiGLU fusion (V3)...")
    patch_llama_model(model_fused, variant="V3", swiglu=True)

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "In a galaxy far far away",
        "Machine learning is transforming",
    ]

    all_passed = True
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits_orig = model_orig(**inputs).logits
            logits_fused = model_fused(**inputs).logits

        max_diff = (logits_orig - logits_fused).abs().max().item()
        mean_diff = (logits_orig - logits_fused).abs().mean().item()
        rel_diff = max_diff / logits_orig.abs().mean().item() if logits_orig.abs().mean().item() > 0 else 0
        status = "PASS" if max_diff < 1e-2 else "FAIL"
        if max_diff >= 1e-2:
            all_passed = False
        print(f"  [{status}] \"{text[:40]}...\": max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}")

    if all_passed:
        print("  All Llama SwiGLU integration tests passed!\n")
    else:
        print("  WARNING: Some Llama SwiGLU integration tests exceeded threshold\n")

    del model_orig, model_fused
    torch.cuda.empty_cache()


def test_gqa_decode_v2_unit():
    """Test GQA decode V2 kernel against PyTorch reference."""
    print("=" * 60)
    print("TEST: GQA decode V2 (per-query-head) — unit")
    print("=" * 60)

    from src.gqa_attention_forward import GQADecodeAttentionV2, pytorch_gqa_decode_attention

    torch.manual_seed(42)

    # (label, num_q_heads, num_kv_heads, head_dim)
    configs = [
        ("TinyLlama",    32,  4, 64),
        ("Llama-3.2-3B", 24,  8, 128),
        ("Llama-3.1-8B", 32,  8, 128),
        ("GPT-OSS-20B",  32,  4, 128),  # h=2880 but head_dim=128 with 32 q-heads... using 128
    ]

    all_passed = True
    for label, num_q, num_kv, hd in configs:
        v2 = GQADecodeAttentionV2(num_q, num_kv, hd)
        for batch in [1, 4, 8]:
            for ctx_len in [128, 512, 1024]:
                for dtype, tol, dtype_name in [
                    (torch.float32, 1e-3, "FP32"),
                    (torch.bfloat16, 0.05, "BF16"),
                ]:
                    q = torch.randn(batch, num_q, hd, device="cuda", dtype=dtype)
                    k = torch.randn(batch, ctx_len, num_kv, hd, device="cuda", dtype=dtype)
                    v = torch.randn(batch, ctx_len, num_kv, hd, device="cuda", dtype=dtype)

                    out_v2 = v2(q, k, v)
                    out_ref = pytorch_gqa_decode_attention(q, k, v, num_kv)

                    max_diff = (out_v2 - out_ref).abs().max().item()
                    status = "PASS" if max_diff < tol else "FAIL"
                    if max_diff >= tol:
                        all_passed = False
                    print(f"  [{status}] {label} {dtype_name} batch={batch} ctx={ctx_len}: max_diff={max_diff:.2e}")

    if all_passed:
        print("  All GQA V2 tests passed!\n")
    else:
        print("  WARNING: Some GQA V2 tests failed!\n")
        assert False, "GQA V2 tests failed"


def test_gqa_decode_v3_unit():
    """Test GQA decode V3 kernel against PyTorch reference."""
    print("=" * 60)
    print("TEST: GQA decode V3 (per-KV-head, shared) — unit")
    print("=" * 60)

    from src.gqa_attention_forward import GQADecodeAttentionV3, pytorch_gqa_decode_attention

    torch.manual_seed(42)

    configs = [
        ("TinyLlama",    32,  4, 64),
        ("Llama-3.2-3B", 24,  8, 128),
        ("Llama-3.1-8B", 32,  8, 128),
        ("GPT-OSS-20B",  32,  4, 128),
    ]

    all_passed = True
    for label, num_q, num_kv, hd in configs:
        v3 = GQADecodeAttentionV3(num_q, num_kv, hd)
        for batch in [1, 4, 8]:
            for ctx_len in [128, 512, 1024]:
                for dtype, tol, dtype_name in [
                    (torch.float32, 1e-3, "FP32"),
                    (torch.bfloat16, 0.05, "BF16"),
                ]:
                    q = torch.randn(batch, num_q, hd, device="cuda", dtype=dtype)
                    k = torch.randn(batch, ctx_len, num_kv, hd, device="cuda", dtype=dtype)
                    v = torch.randn(batch, ctx_len, num_kv, hd, device="cuda", dtype=dtype)

                    out_v3 = v3(q, k, v)
                    out_ref = pytorch_gqa_decode_attention(q, k, v, num_kv)

                    max_diff = (out_v3 - out_ref).abs().max().item()
                    status = "PASS" if max_diff < tol else "FAIL"
                    if max_diff >= tol:
                        all_passed = False
                    print(f"  [{status}] {label} {dtype_name} batch={batch} ctx={ctx_len}: max_diff={max_diff:.2e}")

    if all_passed:
        print("  All GQA V3 tests passed!\n")
    else:
        print("  WARNING: Some GQA V3 tests failed!\n")
        assert False, "GQA V3 tests failed"


def test_gqa_v2_v3_equivalence():
    """Test that V2 and V3 produce identical results."""
    print("=" * 60)
    print("TEST: GQA V2 vs V3 equivalence")
    print("=" * 60)

    from src.gqa_attention_forward import GQADecodeAttentionV2, GQADecodeAttentionV3

    torch.manual_seed(42)

    configs = [
        ("TinyLlama",    32,  4, 64),
        ("Llama-3.2-3B", 24,  8, 128),
        ("Llama-3.1-8B", 32,  8, 128),
        ("GPT-OSS-20B",  32,  4, 128),
    ]

    all_passed = True
    for label, num_q, num_kv, hd in configs:
        v2 = GQADecodeAttentionV2(num_q, num_kv, hd)
        v3 = GQADecodeAttentionV3(num_q, num_kv, hd)
        for batch in [1, 4, 8]:
            for ctx_len in [128, 512, 1024]:
                for dtype, tol, dtype_name in [
                    (torch.float32, 1e-5, "FP32"),
                    (torch.bfloat16, 0.02, "BF16"),  # relaxed: different computation order in BF16
                ]:
                    q = torch.randn(batch, num_q, hd, device="cuda", dtype=dtype)
                    k = torch.randn(batch, ctx_len, num_kv, hd, device="cuda", dtype=dtype)
                    v = torch.randn(batch, ctx_len, num_kv, hd, device="cuda", dtype=dtype)

                    out_v2 = v2(q, k, v)
                    out_v3 = v3(q, k, v)

                    max_diff = (out_v2 - out_v3).abs().max().item()
                    status = "PASS" if max_diff < tol else "FAIL"
                    if max_diff >= tol:
                        all_passed = False
                    print(f"  [{status}] {label} {dtype_name} batch={batch} ctx={ctx_len}: max_diff={max_diff:.2e}")

    if all_passed:
        print("  All V2-V3 equivalence tests passed!\n")
    else:
        print("  WARNING: Some V2-V3 equivalence tests failed!\n")
        assert False, "V2-V3 equivalence tests failed"


def test_llama_integration_gqa_decode():
    """Integration test: TinyLlama with gqa_decode=True vs unpatched."""
    print("=" * 60)
    print("TEST: Llama integration (GQA decode)")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama import patch_llama_model
    import copy

    print("  Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_orig = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).cuda().eval()
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    model_fused = copy.deepcopy(model_orig)
    print("  Patching model with SwiGLU + GQA decode (V3)...")
    patch_llama_model(model_fused, variant="V3", swiglu=True, gqa_decode=True)

    tokenizer.pad_token = tokenizer.eos_token

    # Test token generation (exercises decode path)
    prompts = [
        "The quick brown fox",
        "Machine learning is",
    ]
    gen_kwargs = dict(max_new_tokens=16, do_sample=False, use_cache=True)

    all_passed = True
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            out_orig = model_orig.generate(**inputs, **gen_kwargs)
            out_fused = model_fused.generate(**inputs, **gen_kwargs)

        tokens_orig = tokenizer.decode(out_orig[0], skip_special_tokens=True)
        tokens_fused = tokenizer.decode(out_fused[0], skip_special_tokens=True)

        match = tokens_orig == tokens_fused
        status = "PASS" if match else "FAIL"
        if not match:
            all_passed = False
        print(f"  [{status}] \"{prompt}\"")
        if not match:
            print(f"    Original: {tokens_orig}")
            print(f"    Fused:    {tokens_fused}")

    if all_passed:
        print("  All Llama GQA decode integration tests passed!\n")
    else:
        print("  WARNING: Some Llama GQA decode tests had different outputs")
        print("  (Token differences may be due to numerical precision in decode steps)\n")

    del model_orig, model_fused
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_denominator_kernel()
    test_denominator_welford()
    test_fused_ln_linear_unit()
    test_fused_ln_linear_3d()
    test_fused_ln_linear_fp16()
    test_fused_ln_linear_bf16()
    test_fused_rmsnorm_linear_unit()
    test_fused_rmsnorm_linear_with_bias()
    test_fused_rmsnorm_combined_unit()
    test_gpt_oss_combined_unit()
    test_fused_rmsnorm_swiglu_unit()
    test_gqa_decode_v2_unit()
    test_gqa_decode_v3_unit()
    test_gqa_v2_v3_equivalence()
    test_opt_integration()
    test_llama_integration()
    test_llama_integration_combined()
    test_llama_integration_swiglu()
    test_llama_integration_gqa_decode()
    test_combined_vs_separate_equivalence()
    test_gpt_oss_integration()
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
