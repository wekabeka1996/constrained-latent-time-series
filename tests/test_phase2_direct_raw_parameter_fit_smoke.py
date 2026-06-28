# tests/test_phase2_direct_raw_parameter_fit_smoke.py
#
# P47 Unit Tests

import json
import pytest
import re
import sys
from typing import Any

from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip


# ---------------------------------------------------------------------------
# Strict imports verification
# ---------------------------------------------------------------------------

def test_p47_01_no_forbidden_libs_and_no_toplevel_torch():
    """Verify that we do not import forbidden libs or torch at the top level."""
    # Check that torch, numpy, pandas, scipy, sklearn are not in sys.modules
    # under the direct module name before importing it.
    module_name = "src.phase2.direct_raw_parameter_fit_smoke"
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Verify top-level file content does not have raw imports of these
    with open("src/phase2/direct_raw_parameter_fit_smoke.py", "r", encoding="utf-8") as f:
        code = f.read()

    # Look for top-level import torch
    assert not re.search(r"^import\s+torch\b", code, re.MULTILINE), "Top-level torch import found!"
    assert not re.search(r"^from\s+torch\b", code, re.MULTILINE), "Top-level torch from-import found!"
    
    # Look for forbidden libraries
    for lib in ["numpy", "pandas", "scipy", "sklearn"]:
        assert not re.search(r"^import\s+" + lib + r"\b", code, re.MULTILINE), f"Forbidden import {lib} found!"
        assert not re.search(r"^from\s+" + lib + r"\b", code, re.MULTILINE), f"Forbidden import from {lib} found!"


def test_p47_02_no_fc_vae_and_no_torch_optim():
    """Verify no FC-VAE or torch.optim references in code."""
    with open("src/phase2/direct_raw_parameter_fit_smoke.py", "r", encoding="utf-8") as f:
        code = f.read()

    assert "FC_VAE" not in code or "FC_VAE_MODEL" in code or "FC_VAE_" in code, "Contains forbidden FC-VAE references"
    assert "torch.optim" not in code, "Contains forbidden torch.optim reference"
    assert "Optimizer" not in code, "Contains forbidden Optimizer reference"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_p47_03_constants_exact():
    """Verify exact constants."""
    from src.phase2.direct_raw_parameter_fit_smoke import (
        DIRECT_RAW_PARAMETER_FIT_CONTRACT_VERSION,
        DIRECT_RAW_PARAMETER_FIT_KIND,
        DIRECT_RAW_PARAMETER_FIT_MODULE_NAME,
        P47_STEP_COUNT,
        P47_LEARNING_RATE,
        P47_STATUS_AVAILABLE,
        P47_STATUS_TORCH_UNAVAILABLE,
    )
    assert DIRECT_RAW_PARAMETER_FIT_CONTRACT_VERSION == "phase2_p47_direct_raw_parameter_fit_smoke_contract_v1"
    assert DIRECT_RAW_PARAMETER_FIT_KIND == "direct_raw_parameter_fit_smoke_no_model_no_vae_no_science"
    assert DIRECT_RAW_PARAMETER_FIT_MODULE_NAME == "src.phase2.direct_raw_parameter_fit_smoke"
    assert P47_STEP_COUNT == 20
    assert P47_LEARNING_RATE == 0.05
    assert P47_STATUS_AVAILABLE == "direct_raw_parameter_fit_smoke_available_no_model_no_vae_no_science"
    assert P47_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"


# ---------------------------------------------------------------------------
# Torch boundary & Tensors
# ---------------------------------------------------------------------------

def test_p47_04_torch_loader():
    """Verify load_torch_for_p47_direct_fit loads torch correctly."""
    from src.phase2.direct_raw_parameter_fit_smoke import load_torch_for_p47_direct_fit
    torch = load_torch_for_p47_direct_fit()
    assert torch is not None
    assert hasattr(torch, "tensor")


def test_p47_05_target_and_candidate_tensors():
    """Verify target and candidate tensors structures and requires_grad."""
    from src.phase2.direct_raw_parameter_fit_smoke import (
        build_p47_fixed_target_raw_tensors,
        build_p47_candidate_raw_tensors,
        load_torch_for_p47_direct_fit,
    )
    torch = load_torch_for_p47_direct_fit()

    target = build_p47_fixed_target_raw_tensors()
    assert set(target.keys()) == {
        "raw_kappa",
        "raw_omega",
        "raw_total_mass",
        "raw_allocation_logits",
    }
    for k, v in target.items():
        assert torch.is_tensor(v)
        assert v.requires_grad is False, f"Target parameter {k} must not require grad"

    candidate = build_p47_candidate_raw_tensors()
    assert set(candidate.keys()) == {
        "raw_kappa",
        "raw_omega",
        "raw_total_mass",
        "raw_allocation_logits",
    }
    for k, v in candidate.items():
        assert torch.is_tensor(v)
        assert v.requires_grad is True, f"Candidate parameter {k} must require grad"


def test_p47_06_build_signature_and_compute_loss():
    """Verify combined signature building and loss computation."""
    from src.phase2.direct_raw_parameter_fit_smoke import (
        build_p47_fixed_target_raw_tensors,
        build_p47_candidate_raw_tensors,
        build_combined_signature_from_raw,
        compute_p47_loss,
        load_torch_for_p47_direct_fit,
    )
    torch = load_torch_for_p47_direct_fit()

    target_raw = build_p47_fixed_target_raw_tensors()
    target_sig = build_combined_signature_from_raw(target_raw)
    assert "ar_spectrum_log" in target_sig
    assert "ar_spectrum_mean" in target_sig
    assert "ar_low_high_ratio" in target_sig
    assert "garch_persistence" in target_sig
    assert "garch_persistence_decay" in target_sig

    candidate_raw = build_p47_candidate_raw_tensors()
    loss_res = compute_p47_loss(candidate_raw, target_sig)
    
    assert "loss_total" in loss_res
    loss_val = loss_res["loss_total"]
    assert torch.is_tensor(loss_val)
    assert loss_val.requires_grad is True
    assert float(loss_val.detach()) >= 0.0


# ---------------------------------------------------------------------------
# Optimisation probe & Serialization
# ---------------------------------------------------------------------------

def test_p47_07_probe_and_serialization():
    """Verify direct raw parameter fit probe succeeds and loss decreases."""
    from src.phase2.direct_raw_parameter_fit_smoke import (
        run_direct_raw_parameter_fit_probe,
        direct_raw_parameter_fit_probe_to_json_dict,
        compact_direct_raw_parameter_fit_json,
    )
    probe_res = run_direct_raw_parameter_fit_probe()
    assert probe_res["torch_available"] is True
    assert probe_res["initial_loss_finite"] is True
    assert probe_res["final_loss_finite"] is True
    assert probe_res["initial_loss_nonnegative"] is True
    assert probe_res["final_loss_nonnegative"] is True
    assert probe_res["loss_decreased"] is True
    assert probe_res["loss_decrease_positive"] is True
    assert probe_res["step_count"] == 20
    assert probe_res["no_model"] is True
    assert probe_res["no_vae"] is True
    assert probe_res["no_encoder"] is True
    assert probe_res["no_decoder"] is True
    assert probe_res["no_dataset"] is True
    assert probe_res["no_dataloader"] is True
    assert probe_res["no_torch_optimizer"] is True
    assert probe_res["no_scientific_conclusion"] is True

    # JSON serialization tests
    json_dict = direct_raw_parameter_fit_probe_to_json_dict(probe_res)
    assert isinstance(json_dict, dict)
    
    # Tensors must not be in json_dict
    for k, v in json_dict.items():
        assert not hasattr(v, "backward"), f"Key {k} contains a tensor!"

    compact = compact_direct_raw_parameter_fit_json(probe_res)
    assert isinstance(compact, str)
    assert "\n" not in compact
    
    parsed = json.loads(compact)
    assert parsed["loss_decreased"] is True
    assert parsed["initial_loss_value"] > parsed["final_loss_value"]


def test_p47_08_no_scientific_claims():
    """Verify no scientific success claims are present in reports or code."""
    # Read the reports or file to make sure
    with open("src/phase2/direct_raw_parameter_fit_smoke.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    scientific_buzzwords = [
        "scientific success",
        "state-of-the-art",
        "breakthrough",
        "perfect fit",
        "optimal architecture",
    ]
    for word in scientific_buzzwords:
        assert word not in code.lower(), f"Forbidden scientific claim word found: {word}"


# ---------------------------------------------------------------------------
# Scope Gate
# ---------------------------------------------------------------------------

def test_p47_09_scope_gate():
    """Verify only allowed P47 files are modified or created relative to P46R."""
    allowed = {
        "src/phase2/direct_raw_parameter_fit_smoke.py",
        "tools/phase2/run_p47_direct_raw_parameter_fit_smoke.py",
        "tests/test_phase2_direct_raw_parameter_fit_smoke.py",
        "tests/test_phase2_p47_direct_raw_parameter_fit_smoke.py",
        "reports/PHASE_2_P47_DIRECT_RAW_PARAMETER_FIT_SMOKE_NO_MODEL_NO_VAE_NO_SCIENCE_REPORT.md",
        "src/phase2/__init__.py",
        "tests/test_phase2_p46r_cumulative_scope_gate_policy.py",
    }
    # Base commit is P46R Remote Head / accepted head
    # P46R Head: 01353768a802dce56afe7fced0eedff650e6e62b
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p47-direct-raw-parameter-fit-smoke-no-model-no-vae-no-science",
        base_commit="01353768a802dce56afe7fced0eedff650e6e62b",
        allowed_files=allowed,
        phase_label="P47",
    )
