# tests/test_phase2_analytic_moment_spectral_signatures.py

import json
import pytest
import re
import subprocess

from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip

from src.phase2.analytic_moment_spectral_signatures import (
    ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_CONTRACT_VERSION,
    ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_KIND,
    ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_MODULE_NAME,
    P45_DEFAULT_FREQ_COUNT,
    P45_DEFAULT_GARCH_LAG_COUNT,
    P45_DEFAULT_EPS,
    FC_VAE_SIGNATURES_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_SIGNATURES_STATUS_AVAILABLE,
    FC_VAE_SIGNATURES_STATUS_CONTRACT_MISMATCH,
    SUPPORTED_FC_VAE_SIGNATURES_STATUSES,
    load_torch_for_p45_signatures,
    build_frequency_grid,
    ar_spectral_signature,
    garch_moment_persistence_signature,
    combined_ar_garch_signature,
    run_analytic_moment_spectral_signatures_probe,
    analytic_moment_spectral_signatures_probe_to_json_dict,
    compact_analytic_moment_spectral_signatures_json,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
)
from src.phase2.tensor_native_constraint_primitives import (
    pacf_to_stable_ar_coefficients,
    garch_mass_allocation_from_logits,
    default_garch_allocation_bias,
)


def test_p45_01_constants():
    assert ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_CONTRACT_VERSION == "phase2_p45_analytic_moment_spectral_signatures_contract_v1"
    assert ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_KIND == "analytic_moment_spectral_signatures_no_loss_no_model_no_training"
    assert ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_MODULE_NAME == "src.phase2.analytic_moment_spectral_signatures"
    assert P45_DEFAULT_FREQ_COUNT == 16
    assert P45_DEFAULT_GARCH_LAG_COUNT == 8
    assert P45_DEFAULT_EPS == 1.0e-6


def test_p45_02_statuses():
    assert FC_VAE_SIGNATURES_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_SIGNATURES_STATUS_AVAILABLE == "analytic_moment_spectral_signatures_available_no_loss_no_model_no_training"
    assert FC_VAE_SIGNATURES_STATUS_CONTRACT_MISMATCH == "blocked_by_contract_mismatch"
    
    assert len(SUPPORTED_FC_VAE_SIGNATURES_STATUSES) == 3
    assert FC_VAE_SIGNATURES_STATUS_AVAILABLE in SUPPORTED_FC_VAE_SIGNATURES_STATUSES


def test_p45_03_static_checks():
    filepath = "src/phase2/analytic_moment_spectral_signatures.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # Verify no top-level import torch or from torch
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"

    # Verify no forbidden libraries used in core
    forbidden_libs = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden_libs:
        assert lib not in code, f"Forbidden library '{lib}' referenced in {filepath}"

    # Check for no detach inside primitive math
    assert ".detach()" not in code, f"Forbidden .detach() call found in {filepath}"
    
    # Check for no .item() inside primitive math
    assert ".item()" not in code, f"Forbidden .item() call found in {filepath}"
    
    # Check for no CPU transfer
    assert ".cpu()" not in code, f"Forbidden .cpu() call found in {filepath}"
    assert ".to('cpu')" not in code, f"Forbidden CPU transfer found in {filepath}"
    
    # Check for no root solving
    assert "root_solving" not in code.lower()
    
    # Check for no time-domain recursion/loops over T
    assert not re.search(r"\btime\b", code.lower()), f"Forbidden word 'time' found in {filepath}"
    assert not re.search(r"\bepoch\b", code.lower()), f"Forbidden word 'epoch' found in {filepath}"


def test_p45_04_torch_loader():
    torch = load_torch_for_p45_signatures()
    assert torch is not None
    assert hasattr(torch, "is_tensor")


def test_p45_05_frequency_grid():
    torch = load_torch_for_p45_signatures()
    grid = build_frequency_grid(freq_count=10)
    assert grid.shape == (10,)
    assert grid[0].item() == 0.0
    assert abs(grid[-1].item() - 3.141592653589793) < 1.0e-5
    # Check monotonicity
    assert torch.all(grid[1:] > grid[:-1])


def test_p45_06_ar_spectrum_properties():
    torch = load_torch_for_p45_signatures()
    raw_kappa = torch.tensor(
        [[0.1, -0.05, 0.02, -0.01, 0.005],
         [-0.1, 0.05, -0.02, 0.01, -0.005]],
        dtype=torch.float32
    )
    ar_res = pacf_to_stable_ar_coefficients(raw_kappa)
    ar_sig = ar_spectral_signature(ar_res["ar_coefficients"])
    
    assert ar_sig["ar_spectrum"].shape == (2, 16)
    assert torch.all(ar_sig["ar_spectrum"] > 0)
    assert torch.all(torch.isfinite(ar_sig["ar_spectrum"]))
    assert ar_sig["ar_order"] == 5
    assert ar_sig["freq_count"] == 16


def test_p45_07_ar_spectrum_autograd():
    torch = load_torch_for_p45_signatures()
    raw_kappa = torch.tensor(
        [[0.1, -0.05, 0.02, -0.01, 0.005],
         [-0.1, 0.05, -0.02, 0.01, -0.005]],
        dtype=torch.float32,
        requires_grad=True
    )
    ar_res = pacf_to_stable_ar_coefficients(raw_kappa)
    ar_sig = ar_spectral_signature(ar_res["ar_coefficients"])
    
    loss = ar_sig["ar_spectrum"].sum()
    loss.backward()
    
    assert raw_kappa.grad is not None
    assert torch.any(raw_kappa.grad != 0.0)


def test_p45_08_garch_signature_properties():
    torch = load_torch_for_p45_signatures()
    raw_omega = torch.tensor([[-0.5], [0.5]], dtype=torch.float32)
    raw_total_mass = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    raw_allocation_logits = torch.zeros((2, 2), dtype=torch.float32)
    bias = default_garch_allocation_bias(1, 1, dtype=torch.float32)
    
    garch_res = garch_mass_allocation_from_logits(
        raw_omega=raw_omega,
        raw_total_mass=raw_total_mass,
        raw_allocation_logits=raw_allocation_logits,
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    
    garch_sig = garch_moment_persistence_signature(garch_res)
    
    assert garch_sig["unconditional_variance"].shape == (2, 1)
    assert torch.all(garch_sig["unconditional_variance"] > 0)
    assert torch.all(garch_sig["persistence"] < 1.0)
    assert garch_sig["persistence_decay"].shape == (2, 8)
    assert garch_sig["lag_count"] == 8


def test_p45_09_garch_signature_autograd():
    torch = load_torch_for_p45_signatures()
    raw_omega = torch.tensor([[-0.5], [0.5]], dtype=torch.float32, requires_grad=True)
    raw_total_mass = torch.tensor([[1.0], [-1.0]], dtype=torch.float32, requires_grad=True)
    raw_allocation_logits = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    bias = default_garch_allocation_bias(1, 1, dtype=torch.float32)
    
    garch_res = garch_mass_allocation_from_logits(
        raw_omega=raw_omega,
        raw_total_mass=raw_total_mass,
        raw_allocation_logits=raw_allocation_logits,
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    
    garch_sig = garch_moment_persistence_signature(garch_res)
    
    # Asymmetric coefficients to prevent derivative cancellations
    # beta_share depends on raw_allocation_logits
    loss = (
        garch_sig["unconditional_variance"].sum() +
        2.5 * garch_sig["persistence"].sum() +
        1.5 * garch_sig["persistence_decay"].sum() +
        2.0 * garch_sig["beta_share"].sum()
    )
    loss.backward()
    
    assert raw_omega.grad is not None
    assert raw_total_mass.grad is not None
    assert raw_allocation_logits.grad is not None
    assert torch.any(raw_omega.grad != 0.0)
    assert torch.any(raw_total_mass.grad != 0.0)
    assert torch.any(raw_allocation_logits.grad != 0.0)


def test_p45_10_combined_signature():
    torch = load_torch_for_p45_signatures()
    raw_kappa = torch.tensor(
        [[0.1, -0.05, 0.02, -0.01, 0.005],
         [-0.1, 0.05, -0.02, 0.01, -0.005]],
        dtype=torch.float32
    )
    ar_res = pacf_to_stable_ar_coefficients(raw_kappa)
    ar_sig = ar_spectral_signature(ar_res["ar_coefficients"])
    
    raw_omega = torch.tensor([[-0.5], [0.5]], dtype=torch.float32)
    raw_total_mass = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    raw_allocation_logits = torch.zeros((2, 2), dtype=torch.float32)
    bias = default_garch_allocation_bias(1, 1, dtype=torch.float32)
    
    garch_res = garch_mass_allocation_from_logits(
        raw_omega=raw_omega,
        raw_total_mass=raw_total_mass,
        raw_allocation_logits=raw_allocation_logits,
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    garch_sig = garch_moment_persistence_signature(garch_res)
    
    comb_sig = combined_ar_garch_signature(ar_sig, garch_sig)
    
    assert comb_sig["signature_kind"] == "combined_ar_garch_analytic_signature_no_loss"
    assert "ar_spectrum" in comb_sig
    assert "ar_spectrum_log" in comb_sig
    assert "ar_spectrum_mean" in comb_sig
    assert "ar_spectrum_std" in comb_sig
    assert "ar_low_high_ratio" in comb_sig
    assert "garch_unconditional_variance" in comb_sig
    assert "garch_persistence" in comb_sig
    assert "garch_stationarity_margin" in comb_sig
    assert "garch_alpha_share" in comb_sig
    assert "garch_beta_share" in comb_sig
    assert "garch_persistence_decay" in comb_sig


def test_p45_11_path_leakage_exception():
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("C:\\Users\\wekab\\project")
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("/home/user/project")


def test_p45_12_claim_leakage_exception():
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("state of the art results")
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("we solved time series prediction")


def test_p45_13_serialization():
    probe_res = run_analytic_moment_spectral_signatures_probe()
    d = analytic_moment_spectral_signatures_probe_to_json_dict(probe_res)
    
    assert d["contract_version"] == ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_CONTRACT_VERSION
    assert d["status"] == FC_VAE_SIGNATURES_STATUS_AVAILABLE
    assert d["torch_available"] is True
    assert "ar_spectrum" not in d
    
    compact = compact_analytic_moment_spectral_signatures_json(probe_res)
    assert isinstance(compact, str)
    assert "\n" not in compact


def test_p45_14_scope_gate():
    allowed = {
        "src/phase2/analytic_moment_spectral_signatures.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_analytic_moment_spectral_signatures.py",
        "tools/phase2/run_p45_analytic_moment_spectral_signatures_smoke.py",
        "tests/test_phase2_p45_analytic_moment_spectral_signatures_smoke.py",
        "reports/PHASE_2_P45_ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_NO_LOSS_NO_MODEL_NO_TRAINING_REPORT.md",
    }
    # Phase-local scope gate: only enforces on the P45 branch; skips on later cumulative branches.
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p45-analytic-moment-spectral-signatures-no-loss-no-model-no-training",
        base_commit="a64297edf97eaffe3fb0d84ef2bcedc58d543276",
        allowed_files=allowed,
        phase_label="P45",
    )
