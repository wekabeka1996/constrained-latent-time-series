# tests/test_phase2_direct_raw_parameter_fit_robustness_smoke.py
#
# P48 Unit Tests

import json
import re
import sys

import pytest

from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip


# ---------------------------------------------------------------------------
# Source code checks
# ---------------------------------------------------------------------------

def test_p48_01_no_toplevel_torch_and_no_forbidden_libs():
    """No top-level torch import and no forbidden library imports."""
    with open("src/phase2/direct_raw_parameter_fit_robustness_smoke.py", "r", encoding="utf-8") as f:
        code = f.read()

    assert not re.search(r"^import\s+torch\b", code, re.MULTILINE), "Top-level torch import found!"
    assert not re.search(r"^from\s+torch\b", code, re.MULTILINE), "Top-level torch from-import found!"

    for lib in ["numpy", "pandas", "scipy", "sklearn"]:
        assert not re.search(r"^import\s+" + lib + r"\b", code, re.MULTILINE), f"Forbidden import {lib} found!"
        assert not re.search(r"^from\s+" + lib + r"\b", code, re.MULTILINE), f"Forbidden from-import {lib} found!"


def test_p48_02_no_fc_vae_and_no_torch_optim():
    """No FC-VAE imports, no torch optimizer usage."""
    with open("src/phase2/direct_raw_parameter_fit_robustness_smoke.py", "r", encoding="utf-8") as f:
        code = f.read()

    assert "torch.optim" not in code, "Forbidden torch.optim reference found!"
    assert "Optimizer" not in code, "Forbidden Optimizer reference found!"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_p48_03_constants_exact():
    """Verify exact constants."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
        DIRECT_FIT_ROBUSTNESS_CONTRACT_VERSION,
        DIRECT_FIT_ROBUSTNESS_KIND,
        DIRECT_FIT_ROBUSTNESS_MODULE_NAME,
        P48_DEFAULT_STEP_COUNT,
        P48_DEFAULT_LEARNING_RATE,
        P48_SCENARIO_COUNT,
        P48_STATUS_AVAILABLE,
        P48_STATUS_TORCH_UNAVAILABLE,
    )
    assert DIRECT_FIT_ROBUSTNESS_CONTRACT_VERSION == (
        "phase2_p48_deterministic_direct_fit_robustness_smoke_contract_v1"
    )
    assert DIRECT_FIT_ROBUSTNESS_KIND == (
        "deterministic_direct_fit_robustness_smoke_no_model_no_vae_no_science"
    )
    assert DIRECT_FIT_ROBUSTNESS_MODULE_NAME == (
        "src.phase2.direct_raw_parameter_fit_robustness_smoke"
    )
    assert P48_DEFAULT_STEP_COUNT == 20
    assert P48_DEFAULT_LEARNING_RATE == 0.05
    assert P48_SCENARIO_COUNT == 4
    assert P48_STATUS_AVAILABLE == "direct_fit_robustness_smoke_available_no_model_no_vae_no_science"
    assert P48_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"


# ---------------------------------------------------------------------------
# Torch loader
# ---------------------------------------------------------------------------

def test_p48_04_torch_loader():
    """Verify load_torch_for_p48_direct_fit_robustness loads torch."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
        load_torch_for_p48_direct_fit_robustness,
    )
    torch = load_torch_for_p48_direct_fit_robustness()
    assert torch is not None
    assert hasattr(torch, "tensor")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def test_p48_05_scenario_count():
    """Verify exactly 4 scenarios returned."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
        build_p48_scenarios,
        P48_SCENARIO_COUNT,
    )
    scenarios = build_p48_scenarios()
    assert len(scenarios) == P48_SCENARIO_COUNT


def test_p48_06_scenario_required_keys():
    """Verify each scenario has required keys."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import build_p48_scenarios
    required_keys = {"scenario_id", "target_raw", "candidate_raw", "step_count", "learning_rate"}
    for s in build_p48_scenarios():
        assert required_keys.issubset(s.keys()), f"Missing keys in scenario {s.get('scenario_id')}"


def test_p48_07_target_tensors_no_grad():
    """Verify all target tensors have requires_grad=False."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
        build_p48_scenarios,
        load_torch_for_p48_direct_fit_robustness,
    )
    torch = load_torch_for_p48_direct_fit_robustness()
    param_keys = ["raw_kappa", "raw_omega", "raw_total_mass", "raw_allocation_logits"]
    for s in build_p48_scenarios():
        for k in param_keys:
            t = s["target_raw"][k]
            assert torch.is_tensor(t)
            assert t.requires_grad is False, f"Target {k} must not require grad in scenario {s['scenario_id']}"


def test_p48_08_candidate_tensors_require_grad():
    """Verify all candidate tensors have requires_grad=True."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
        build_p48_scenarios,
        load_torch_for_p48_direct_fit_robustness,
    )
    torch = load_torch_for_p48_direct_fit_robustness()
    param_keys = ["raw_kappa", "raw_omega", "raw_total_mass", "raw_allocation_logits"]
    for s in build_p48_scenarios():
        for k in param_keys:
            t = s["candidate_raw"][k]
            assert torch.is_tensor(t)
            assert t.requires_grad is True, f"Candidate {k} must require grad in scenario {s['scenario_id']}"


def test_p48_09_signature_builds_from_raw():
    """Verify combined signature can be built from raw tensors in each scenario."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
        build_p48_scenarios,
        build_combined_signature_from_raw,
    )
    for s in build_p48_scenarios():
        sig = build_combined_signature_from_raw(s["target_raw"])
        assert isinstance(sig, dict)
        assert "ar_spectrum_log" in sig
        assert "garch_persistence" in sig


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def test_p48_10_single_scenario_loss_finite():
    """Verify initial/final losses are finite for each scenario."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
        build_p48_scenarios,
        run_single_p48_scenario,
    )
    for s in build_p48_scenarios():
        result = run_single_p48_scenario(s)
        assert result["initial_loss_finite"] is True, f"Initial loss not finite in {s['scenario_id']}"
        assert result["final_loss_finite"] is True, f"Final loss not finite in {s['scenario_id']}"


def test_p48_11_single_scenario_loss_decreases():
    """Verify loss decreases for each scenario."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
        build_p48_scenarios,
        run_single_p48_scenario,
    )
    for s in build_p48_scenarios():
        result = run_single_p48_scenario(s)
        assert result["loss_decreased"] is True, (
            f"Loss did not decrease in scenario {s['scenario_id']}: "
            f"init={result['initial_loss_value']:.4f}, final={result['final_loss_value']:.4f}"
        )


# ---------------------------------------------------------------------------
# Aggregate probe
# ---------------------------------------------------------------------------

def test_p48_12_aggregate_probe():
    """Verify aggregate probe returns 4/4 passed with all losses finite."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
        run_direct_fit_robustness_probe,
        P48_SCENARIO_COUNT,
    )
    result = run_direct_fit_robustness_probe()
    assert result["torch_available"] is True
    assert result["scenario_count"] == P48_SCENARIO_COUNT
    assert result["scenarios_passed"] == P48_SCENARIO_COUNT
    assert result["all_scenarios_loss_decreased"] is True
    assert result["all_losses_finite"] is True
    assert result["no_model"] is True
    assert result["no_vae"] is True
    assert result["no_encoder"] is True
    assert result["no_decoder"] is True
    assert result["no_dataset"] is True
    assert result["no_dataloader"] is True
    assert result["no_torch_optimizer"] is True
    assert result["no_scientific_conclusion"] is True


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_p48_13_serialization_excludes_raw_tensors():
    """Verify JSON serialization excludes raw tensors and parameter arrays."""
    from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
        run_direct_fit_robustness_probe,
        direct_fit_robustness_probe_to_json_dict,
        compact_direct_fit_robustness_json,
    )
    probe_res = run_direct_fit_robustness_probe()
    json_dict = direct_fit_robustness_probe_to_json_dict(probe_res)

    # No raw tensors anywhere in dict
    def check_no_tensors(obj, path=""):
        if hasattr(obj, "backward"):
            raise AssertionError(f"Tensor found at path '{path}'!")
        if isinstance(obj, dict):
            for k, v in obj.items():
                check_no_tensors(v, path=f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                check_no_tensors(v, path=f"{path}[{i}]")

    check_no_tensors(json_dict)

    compact = compact_direct_fit_robustness_json(probe_res)
    assert isinstance(compact, str)
    parsed = json.loads(compact)
    assert parsed["all_scenarios_loss_decreased"] is True
    assert parsed["scenarios_passed"] == 4


def test_p48_14_no_scientific_claims():
    """Verify no scientific success claims in source code."""
    with open("src/phase2/direct_raw_parameter_fit_robustness_smoke.py", "r", encoding="utf-8") as f:
        code = f.read()
    for phrase in ["scientific success", "state-of-the-art", "breakthrough", "convergence proved"]:
        assert phrase not in code.lower(), f"Forbidden scientific claim found: {phrase}"


# ---------------------------------------------------------------------------
# Scope Gate
# ---------------------------------------------------------------------------

def test_p48_15_scope_gate():
    """Verify only allowed P48 files are modified or created relative to P47R base."""
    allowed = {
        "src/phase2/direct_raw_parameter_fit_robustness_smoke.py",
        "tools/phase2/run_p48_direct_raw_parameter_fit_robustness_smoke.py",
        "tests/test_phase2_direct_raw_parameter_fit_robustness_smoke.py",
        "tests/test_phase2_p48_direct_raw_parameter_fit_robustness_smoke.py",
        "reports/PHASE_2_P48_DETERMINISTIC_DIRECT_FIT_ROBUSTNESS_SMOKE_NO_MODEL_NO_VAE_NO_SCIENCE_REPORT.md",
        "src/phase2/__init__.py",
        "tests/test_phase2_p46r_cumulative_scope_gate_policy.py",
    }
    # P47R Head: 7c65cfb181cf0e2272ccd4a272d8eabfd7ed3365
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p48-deterministic-direct-fit-robustness-smoke-no-model-no-vae-no-science",
        base_commit="7c65cfb181cf0e2272ccd4a272d8eabfd7ed3365",
        allowed_files=allowed,
        phase_label="P48",
    )
