# tests/test_phase2_tensor_native_constraint_primitives.py

import json
import pytest
import re
import subprocess

from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip

from src.phase2.tensor_native_constraint_primitives import (
    TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_CONTRACT_VERSION,
    TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_KIND,
    TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_MODULE_NAME,
    P44_MAX_AR_ORDER,
    P44_DEFAULT_EPS,
    FC_VAE_PRIMITIVES_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_PRIMITIVES_STATUS_AVAILABLE,
    FC_VAE_PRIMITIVES_STATUS_CONTRACT_MISMATCH,
    SUPPORTED_FC_VAE_PRIMITIVES_STATUSES,
    load_torch_for_p44_tensor_primitives,
    pacf_to_stable_ar_coefficients,
    garch_mass_allocation_from_logits,
    default_garch_allocation_bias,
    run_tensor_native_constraint_primitives_probe,
    tensor_native_constraint_primitives_probe_to_json_dict,
    compact_tensor_native_constraint_primitives_json,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
)


def test_p44_01_constants():
    assert TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_CONTRACT_VERSION == "phase2_p44_tensor_native_constraint_primitives_contract_v1"
    assert TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_KIND == "tensor_native_constraint_primitives_no_model_no_loss_no_training"
    assert TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_MODULE_NAME == "src.phase2.tensor_native_constraint_primitives"
    assert P44_MAX_AR_ORDER == 5
    assert P44_DEFAULT_EPS == 1.0e-4


def test_p44_02_statuses():
    assert FC_VAE_PRIMITIVES_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_PRIMITIVES_STATUS_AVAILABLE == "tensor_native_constraint_primitives_available_no_model_no_loss_no_training"
    assert FC_VAE_PRIMITIVES_STATUS_CONTRACT_MISMATCH == "blocked_by_contract_mismatch"
    
    assert len(SUPPORTED_FC_VAE_PRIMITIVES_STATUSES) == 3
    assert FC_VAE_PRIMITIVES_STATUS_AVAILABLE in SUPPORTED_FC_VAE_PRIMITIVES_STATUSES


def test_p44_03_static_checks():
    filepath = "src/phase2/tensor_native_constraint_primitives.py"
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
    
    # Check for no root solving
    assert "root_solving" not in code.lower()
    
    # Check for no time-domain recursion/loops over T
    # Use word boundary check to avoid matching substrings like "RuntimeError"
    assert not re.search(r"\btime\b", code.lower()), f"Forbidden word 'time' found in {filepath}"
    assert not re.search(r"\bepoch\b", code.lower()), f"Forbidden word 'epoch' found in {filepath}"


def test_p44_04_torch_loader():
    torch = load_torch_for_p44_tensor_primitives()
    assert torch is not None
    assert hasattr(torch, "is_tensor")


def test_p44_05_pacf_to_ar_shapes():
    torch = load_torch_for_p44_tensor_primitives()
    raw_kappa = torch.tensor(
        [[0.1, -0.05, 0.02, -0.01, 0.005],
         [-0.1, 0.05, -0.02, 0.01, -0.005]],
        dtype=torch.float32
    )
    ar_res = pacf_to_stable_ar_coefficients(raw_kappa)
    assert ar_res["ar_coefficients"].shape == (2, 5)
    assert ar_res["pacf_values"].shape == (2, 5)
    assert ar_res["order"] == 5
    assert ar_res["stationarity_parameterization"] == "pacf_reflection_coefficients"


def test_p44_06_pacf_bounds():
    torch = load_torch_for_p44_tensor_primitives()
    # Large inputs to tanh to test eps limit
    raw_kappa = torch.tensor([[100.0, -100.0, 50.0, -50.0, 20.0]], dtype=torch.float32)
    ar_res = pacf_to_stable_ar_coefficients(raw_kappa, eps=0.01)
    pacf = ar_res["pacf_values"]
    assert torch.all(pacf < 0.991)
    assert torch.all(pacf > -0.991)


def test_p44_07_pacf_autograd():
    torch = load_torch_for_p44_tensor_primitives()
    raw_kappa = torch.tensor(
        [[0.1, -0.05, 0.02, -0.01, 0.005],
         [-0.1, 0.05, -0.02, 0.01, -0.005]],
        dtype=torch.float32,
        requires_grad=True
    )
    ar_res = pacf_to_stable_ar_coefficients(raw_kappa)
    loss = ar_res["ar_coefficients"].sum()
    loss.backward()
    
    assert raw_kappa.grad is not None
    assert torch.any(raw_kappa.grad != 0.0)


def test_p44_08_garch_properties():
    torch = load_torch_for_p44_tensor_primitives()
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
    
    # omega > 0
    assert torch.all(garch_res["omega"] > 0)
    
    # alpha >= 0, beta >= 0
    assert torch.all(garch_res["alpha"] >= 0)
    assert torch.all(garch_res["beta"] >= 0)
    
    # alpha + beta sum < 1 - eps
    eps = garch_res["eps_margin"]
    assert torch.all(garch_res["alpha_beta_sum"] < 1.0 - eps)
    assert torch.all(garch_res["stationarity_margin"] > 0)


def test_p44_09_garch_autograd():
    torch = load_torch_for_p44_tensor_primitives()
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
    
    # Use asymmetric coefficients to prevent derivative cancellation for raw_total_mass
    loss = (
        garch_res["omega"].sum() +
        1.5 * garch_res["alpha"].sum() +
        2.5 * garch_res["beta"].sum() +
        0.5 * garch_res["stationarity_margin"].sum()
    )
    loss.backward()
    
    assert raw_omega.grad is not None
    assert raw_total_mass.grad is not None
    assert raw_allocation_logits.grad is not None
    assert torch.any(raw_omega.grad != 0.0)
    assert torch.any(raw_total_mass.grad != 0.0)
    assert torch.any(raw_allocation_logits.grad != 0.0)


def test_p44_10_default_bias():
    torch = load_torch_for_p44_tensor_primitives()
    bias = default_garch_allocation_bias(2, 3, beta_bias=3.0, alpha_bias=-1.5)
    assert bias.shape == (5,)
    assert bias[0].item() == -1.5
    assert bias[1].item() == -1.5
    assert bias[2].item() == 3.0
    assert bias[3].item() == 3.0
    assert bias[4].item() == 3.0


def test_p44_11_path_leakage_exception():
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("C:\\Users\\wekab\\project")
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("/home/user/project")


def test_p44_12_claim_leakage_exception():
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("state of the art results")
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("we solved time series prediction")


def test_p44_13_serialization():
    probe_res = run_tensor_native_constraint_primitives_probe()
    d = tensor_native_constraint_primitives_probe_to_json_dict(probe_res)
    
    assert d["contract_version"] == TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_CONTRACT_VERSION
    assert d["status"] == FC_VAE_PRIMITIVES_STATUS_AVAILABLE
    assert d["torch_available"] is True
    assert "raw_kappa" not in d
    assert "omega" not in d
    
    compact = compact_tensor_native_constraint_primitives_json(probe_res)
    assert isinstance(compact, str)
    assert "\n" not in compact


def test_p44_14_scope_gate():
    allowed = {
        "src/phase2/tensor_native_constraint_primitives.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_tensor_native_constraint_primitives.py",
        "tools/phase2/run_p44_tensor_native_constraint_primitives_smoke.py",
        "tests/test_phase2_p44_tensor_native_constraint_primitives_smoke.py",
        "reports/PHASE_2_P44_TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_NO_MODEL_NO_LOSS_NO_TRAINING_REPORT.md",
        "reports/PHASE_2_P43_OWN_FORWARD_BOUNDARY_STUB_DECLARED_NO_FORWARD_EXECUTION_NO_OUTPUT_NO_TRAINING_REPORT.md",
    }
    # Phase-local scope gate: only enforces on the P44 branch; skips on later cumulative branches.
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p44-tensor-native-constraint-primitives-no-model-no-loss-no-training",
        base_commit="fa57af8ac38398a64347391afa2ab6cd8f5c8932",
        allowed_files=allowed,
        phase_label="P44",
    )
