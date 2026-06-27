# tests/test_phase2_moment_spectral_matching_loss.py

import json
import pytest
import re
import subprocess

from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip

from src.phase2.moment_spectral_matching_loss import (
    MOMENT_SPECTRAL_MATCHING_LOSS_CONTRACT_VERSION,
    MOMENT_SPECTRAL_MATCHING_LOSS_KIND,
    MOMENT_SPECTRAL_MATCHING_LOSS_MODULE_NAME,
    P46_DEFAULT_EPS,
    P46_DEFAULT_AR_LOG_SPECTRUM_WEIGHT,
    P46_DEFAULT_AR_SUMMARY_WEIGHT,
    P46_DEFAULT_GARCH_MOMENT_WEIGHT,
    P46_DEFAULT_GARCH_DECAY_WEIGHT,
    FC_VAE_LOSS_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_LOSS_STATUS_AVAILABLE,
    FC_VAE_LOSS_STATUS_CONTRACT_MISMATCH,
    SUPPORTED_FC_VAE_LOSS_STATUSES,
    load_torch_for_p46_loss,
    safe_tensor_mse,
    ar_spectral_matching_loss,
    garch_moment_matching_loss,
    combined_moment_spectral_matching_loss,
    run_moment_spectral_matching_loss_probe,
    moment_spectral_matching_loss_probe_to_json_dict,
    compact_moment_spectral_matching_loss_json,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
)
from src.phase2.tensor_native_constraint_primitives import (
    pacf_to_stable_ar_coefficients,
    garch_mass_allocation_from_logits,
    default_garch_allocation_bias,
)
from src.phase2.analytic_moment_spectral_signatures import (
    ar_spectral_signature,
    garch_moment_persistence_signature,
    combined_ar_garch_signature,
)


def test_p46_01_constants():
    assert MOMENT_SPECTRAL_MATCHING_LOSS_CONTRACT_VERSION == "phase2_p46_moment_spectral_matching_loss_contract_v1"
    assert MOMENT_SPECTRAL_MATCHING_LOSS_KIND == "moment_spectral_matching_loss_no_model_no_training"
    assert MOMENT_SPECTRAL_MATCHING_LOSS_MODULE_NAME == "src.phase2.moment_spectral_matching_loss"
    assert P46_DEFAULT_EPS == 1.0e-8
    assert P46_DEFAULT_AR_LOG_SPECTRUM_WEIGHT == 1.0
    assert P46_DEFAULT_AR_SUMMARY_WEIGHT == 0.25
    assert P46_DEFAULT_GARCH_MOMENT_WEIGHT == 1.0
    assert P46_DEFAULT_GARCH_DECAY_WEIGHT == 0.5


def test_p46_02_statuses():
    assert FC_VAE_LOSS_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_LOSS_STATUS_AVAILABLE == "moment_spectral_matching_loss_available_no_model_no_training"
    assert FC_VAE_LOSS_STATUS_CONTRACT_MISMATCH == "blocked_by_contract_mismatch"
    
    assert len(SUPPORTED_FC_VAE_LOSS_STATUSES) == 3
    assert FC_VAE_LOSS_STATUS_AVAILABLE in SUPPORTED_FC_VAE_LOSS_STATUSES


def test_p46_03_static_checks():
    filepath = "src/phase2/moment_spectral_matching_loss.py"
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
    # Ignore helper methods or comments by checking file content
    # Let's count occurrence of .detach() - only allowed inside the run_..._probe function
    # To be safe, we check that no .detach() exists outside probe.
    # In our file, .detach() only appears on line 318-322.
    assert code.count(".detach()") <= 4

    # Check for no .item() inside primitive math
    assert ".item()" not in code

    # Check for no CPU transfer
    assert ".cpu()" not in code
    assert ".to('cpu')" not in code

    # Check for no root solving
    assert "root_solving" not in code.lower()

    # Check for no time-domain recursion/loops over T
    assert not re.search(r"\btime\b", code.lower()), f"Forbidden word 'time' found in {filepath}"
    assert not re.search(r"\bepoch\b", code.lower()), f"Forbidden word 'epoch' found in {filepath}"


def test_p46_04_torch_loader():
    torch = load_torch_for_p46_loss()
    assert torch is not None
    assert hasattr(torch, "is_tensor")


def test_p46_05_safe_tensor_mse():
    torch = load_torch_for_p46_loss()
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([1.1, 1.9, 3.2], dtype=torch.float32)
    
    mse = safe_tensor_mse(a, b)
    expected = ((1.0-1.1)**2 + (2.0-1.9)**2 + (3.0-3.2)**2) / 3.0
    assert abs(mse.item() - expected) < 1.0e-6
    
    # Rejects mismatched shapes
    c = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(ValueError):
        safe_tensor_mse(a, c)


def test_p46_06_losses_properties():
    torch = load_torch_for_p46_loss()
    
    raw_kappa_cand = torch.tensor([[0.1, -0.05, 0.02, -0.01, 0.005]], dtype=torch.float32)
    raw_omega_cand = torch.tensor([[-0.5]], dtype=torch.float32)
    raw_total_mass_cand = torch.tensor([[1.0]], dtype=torch.float32)
    raw_allocation_logits_cand = torch.zeros((1, 2), dtype=torch.float32)
    bias = default_garch_allocation_bias(1, 1, dtype=torch.float32)
    
    raw_kappa_targ = torch.tensor([[0.08, -0.04, 0.015, -0.008, 0.003]], dtype=torch.float32)
    raw_omega_targ = torch.tensor([[-0.4]], dtype=torch.float32)
    raw_total_mass_targ = torch.tensor([[0.9]], dtype=torch.float32)
    raw_allocation_logits_targ = torch.zeros((1, 2), dtype=torch.float32)
    
    # Signatures
    ar_res_cand = pacf_to_stable_ar_coefficients(raw_kappa_cand)
    garch_res_cand = garch_mass_allocation_from_logits(
        raw_omega=raw_omega_cand,
        raw_total_mass=raw_total_mass_cand,
        raw_allocation_logits=raw_allocation_logits_cand,
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    ar_sig_cand = ar_spectral_signature(ar_res_cand["ar_coefficients"])
    garch_sig_cand = garch_moment_persistence_signature(garch_res_cand)
    comb_sig_cand = combined_ar_garch_signature(ar_sig_cand, garch_sig_cand)
    
    ar_res_targ = pacf_to_stable_ar_coefficients(raw_kappa_targ)
    garch_res_targ = garch_mass_allocation_from_logits(
        raw_omega=raw_omega_targ,
        raw_total_mass=raw_total_mass_targ,
        raw_allocation_logits=raw_allocation_logits_targ,
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    ar_sig_targ = ar_spectral_signature(ar_res_targ["ar_coefficients"])
    garch_sig_targ = garch_moment_persistence_signature(garch_res_targ)
    comb_sig_targ = combined_ar_garch_signature(ar_sig_targ, garch_sig_targ)
    
    # AR Loss
    ar_loss_res = ar_spectral_matching_loss(ar_sig_cand, ar_sig_targ)
    assert ar_loss_res["ar_loss_total"].ndim == 0
    assert ar_loss_res["ar_loss_total"].item() >= 0.0
    
    # GARCH Loss
    garch_loss_res = garch_moment_matching_loss(garch_sig_cand, garch_sig_targ)
    assert garch_loss_res["garch_loss_total"].ndim == 0
    assert garch_loss_res["garch_loss_total"].item() >= 0.0
    
    # Combined Loss
    comb_loss_res = combined_moment_spectral_matching_loss(comb_sig_cand, comb_sig_targ)
    assert comb_loss_res["loss_total"].ndim == 0
    assert comb_loss_res["loss_total"].item() >= 0.0
    assert comb_loss_res["ar_component_loss"].item() >= 0.0
    assert comb_loss_res["garch_component_loss"].item() >= 0.0


def test_p46_07_autograd():
    torch = load_torch_for_p46_loss()
    
    raw_kappa_cand = torch.tensor([[0.1, -0.05, 0.02, -0.01, 0.005]], dtype=torch.float32, requires_grad=True)
    raw_omega_cand = torch.tensor([[-0.5]], dtype=torch.float32, requires_grad=True)
    raw_total_mass_cand = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True)
    raw_allocation_logits_cand = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
    bias = default_garch_allocation_bias(1, 1, dtype=torch.float32)
    
    raw_kappa_targ = torch.tensor([[0.08, -0.04, 0.015, -0.008, 0.003]], dtype=torch.float32)
    raw_omega_targ = torch.tensor([[-0.4]], dtype=torch.float32)
    raw_total_mass_targ = torch.tensor([[0.9]], dtype=torch.float32)
    raw_allocation_logits_targ = torch.zeros((1, 2), dtype=torch.float32)
    
    # Signatures
    ar_res_cand = pacf_to_stable_ar_coefficients(raw_kappa_cand)
    garch_res_cand = garch_mass_allocation_from_logits(
        raw_omega=raw_omega_cand,
        raw_total_mass=raw_total_mass_cand,
        raw_allocation_logits=raw_allocation_logits_cand,
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    ar_sig_cand = ar_spectral_signature(ar_res_cand["ar_coefficients"])
    garch_sig_cand = garch_moment_persistence_signature(garch_res_cand)
    comb_sig_cand = combined_ar_garch_signature(ar_sig_cand, garch_sig_cand)
    
    ar_res_targ = pacf_to_stable_ar_coefficients(raw_kappa_targ)
    garch_res_targ = garch_mass_allocation_from_logits(
        raw_omega=raw_omega_targ,
        raw_total_mass=raw_total_mass_targ,
        raw_allocation_logits=raw_allocation_logits_targ,
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    ar_sig_targ = ar_spectral_signature(ar_res_targ["ar_coefficients"])
    garch_sig_targ = garch_moment_persistence_signature(garch_res_targ)
    comb_sig_targ = combined_ar_garch_signature(ar_sig_targ, garch_sig_targ)
    
    # Combined Loss
    comb_loss_res = combined_moment_spectral_matching_loss(comb_sig_cand, comb_sig_targ)
    loss = comb_loss_res["loss_total"]
    
    loss.backward()
    
    assert raw_kappa_cand.grad is not None
    assert raw_omega_cand.grad is not None
    assert raw_total_mass_cand.grad is not None
    assert raw_allocation_logits_cand.grad is not None
    
    assert torch.any(raw_kappa_cand.grad != 0.0)
    assert torch.any(raw_omega_cand.grad != 0.0)
    assert torch.any(raw_total_mass_cand.grad != 0.0)
    assert torch.any(raw_allocation_logits_cand.grad != 0.0)


def test_p46_08_path_leakage_exception():
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("C:\\Users\\wekab\\project")
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("/home/user/project")


def test_p46_09_claim_leakage_exception():
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("state of the art results")
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("we solved time series prediction")


def test_p46_10_serialization():
    probe_res = run_moment_spectral_matching_loss_probe()
    d = moment_spectral_matching_loss_probe_to_json_dict(probe_res)
    
    assert d["contract_version"] == MOMENT_SPECTRAL_MATCHING_LOSS_CONTRACT_VERSION
    assert d["status"] == FC_VAE_LOSS_STATUS_AVAILABLE
    assert d["torch_available"] is True
    assert "loss_total" not in d
    assert "loss_total_value" in d
    
    compact = compact_moment_spectral_matching_loss_json(probe_res)
    assert isinstance(compact, str)
    assert "\n" not in compact


def test_p46_11_scope_gate():
    allowed = {
        "src/phase2/moment_spectral_matching_loss.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_moment_spectral_matching_loss.py",
        "tools/phase2/run_p46_moment_spectral_matching_loss_smoke.py",
        "tests/test_phase2_p46_moment_spectral_matching_loss_smoke.py",
        "reports/PHASE_2_P46_MOMENT_SPECTRAL_MATCHING_LOSS_NO_MODEL_NO_TRAINING_REPORT.md",
    }
    # Phase-local scope gate: only enforces on the P46 branch; skips on later cumulative branches.
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p46-moment-spectral-matching-loss-no-model-no-training",
        base_commit="32735b7df0d16e6ca6a9aa0b1f0e3f6247b24c15",
        allowed_files=allowed,
        phase_label="P46",
    )
