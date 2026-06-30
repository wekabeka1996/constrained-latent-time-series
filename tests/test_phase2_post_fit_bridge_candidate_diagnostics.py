# tests/test_phase2_post_fit_bridge_candidate_diagnostics.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.
# We import torch inside the tests if needed, or rely on loaders.

from src.phase2.post_fit_bridge_candidate_diagnostics import (
    POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_CONTRACT_VERSION,
    POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_KIND,
    POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_MODULE_NAME,
    P51_TARGET_COUNT,
    load_torch_for_p51_post_fit_diagnostics,
    run_single_p51_fit_with_final_raw,
    build_post_fit_constrained_diagnostics,
    build_post_fit_signature_diagnostics,
    run_single_p51_post_fit_diagnostics,
    run_post_fit_bridge_candidate_diagnostics_probe,
    post_fit_bridge_candidate_diagnostics_probe_to_json_dict,
    compact_post_fit_bridge_candidate_diagnostics_json,
)

# P50 allowed imports to check
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    P50_STEP_COUNT,
    P50_LEARNING_RATE,
    build_p50_bridge_targets,
)


def test_p51_01_constants_exact():
    assert POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_CONTRACT_VERSION == "phase2_p51_post_fit_bridge_candidate_diagnostics_contract_v1"
    assert POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_KIND == "post_fit_bridge_candidate_diagnostics_no_model_no_vae_no_gsb_no_science"
    assert POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_MODULE_NAME == "src.phase2.post_fit_bridge_candidate_diagnostics"
    assert P51_TARGET_COUNT == 3


def test_p51_02_no_top_level_torch_import():
    filepath = "src/phase2/post_fit_bridge_candidate_diagnostics.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p51_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/post_fit_bridge_candidate_diagnostics.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p51_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/post_fit_bridge_candidate_diagnostics.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p51_05_no_fc_vae_imports():
    p = pathlib.Path("src/phase2/post_fit_bridge_candidate_diagnostics.py").read_text(encoding="utf-8")
    assert "fc_vae_model" not in p
    assert "fc_vae_torch_shell" not in p
    assert "fc_vae_torch_module_stub" not in p


def test_p51_06_no_torch_optim():
    p = pathlib.Path("src/phase2/post_fit_bridge_candidate_diagnostics.py").read_text(encoding="utf-8")
    assert "torch.optim" not in p
    assert "optimizer =" not in p
    assert ".step()" not in p
    assert not re.search(r"\bOptimizer\b", p)


def test_p51_07_p50_imports_present_and_allowed():
    p = pathlib.Path("src/phase2/post_fit_bridge_candidate_diagnostics.py").read_text(encoding="utf-8")
    assert "P50_STEP_COUNT" in p
    assert "P50_LEARNING_RATE" in p
    assert "build_p50_bridge_targets" in p
    assert "build_p50_candidate_raw_tensors" in p


def test_p51_08_target_count_exactly_3():
    targets = build_p50_bridge_targets()
    assert len(targets) == 3


def test_p51_09_lambda_values_exact():
    targets = build_p50_bridge_targets()
    lambdas = [t["lambda_value"] for t in targets]
    assert lambdas == [0.25, 0.50, 0.75]


def test_p51_10_single_target_diagnostic_loss_finite():
    targets = build_p50_bridge_targets()
    for t in targets:
        res = run_single_p51_post_fit_diagnostics(t)
        assert res["initial_loss_finite"] is True
        assert res["final_loss_finite"] is True


def test_p51_11_single_target_loss_decreases():
    targets = build_p50_bridge_targets()
    for t in targets:
        res = run_single_p51_post_fit_diagnostics(t)
        assert res["loss_decreased"] is True
        assert res["final_loss_value"] < res["initial_loss_value"]


def test_p51_12_ar_coefficient_diagnostics_finite():
    targets = build_p50_bridge_targets()
    for t in targets:
        res = run_single_p51_post_fit_diagnostics(t)
        assert res["ar_coefficients_finite"] is True
        assert isinstance(res["ar_abs_max_value"], float)
        assert isinstance(res["ar_abs_mean_value"], float)


def test_p51_13_ar_shape_diagnostic_json_safe():
    targets = build_p50_bridge_targets()
    for t in targets:
        res = run_single_p51_post_fit_diagnostics(t)
        shape = res["ar_coefficients_tensor_shape"]
        assert isinstance(shape, list)
        assert len(shape) == 2
        assert all(isinstance(x, int) for x in shape)


def test_p51_14_garch_tensor_diagnostics_finite():
    targets = build_p50_bridge_targets()
    for t in targets:
        res = run_single_p51_post_fit_diagnostics(t)
        assert res["garch_tensor_fields_finite"] is True
        assert res["garch_persistence_available"] is True
        assert res["garch_persistence_finite"] is True
        assert isinstance(res["garch_persistence_max_value"], float)
        assert isinstance(res["garch_persistence_min_value"], float)


def test_p51_15_signature_tensor_diagnostics_finite():
    targets = build_p50_bridge_targets()
    for t in targets:
        res = run_single_p51_post_fit_diagnostics(t)
        assert res["signature_all_tensor_fields_finite"] is True
        assert res["signature_tensor_field_count"] == 11
        assert res["signature_has_ar_spectrum"] is True
        assert res["signature_has_ar_spectrum_log"] is True
        assert res["signature_has_garch_persistence"] is True
        assert res["signature_has_garch_stationarity_margin"] is True


def test_p51_16_aggregate_targets_passed():
    probe_res = run_post_fit_bridge_candidate_diagnostics_probe()
    assert probe_res["targets_passed_loss_decrease"] == 3


def test_p51_17_aggregate_all_finite_flags_true():
    probe_res = run_post_fit_bridge_candidate_diagnostics_probe()
    assert probe_res["all_targets_loss_decreased"] is True
    assert probe_res["all_losses_finite"] is True
    assert probe_res["all_ar_coefficients_finite"] is True
    assert probe_res["all_garch_tensor_fields_finite"] is True
    assert probe_res["all_signature_tensor_fields_finite"] is True


def test_p51_18_serialization_excludes_raw_and_signatures():
    probe_res = run_post_fit_bridge_candidate_diagnostics_probe()
    json_dict = post_fit_bridge_candidate_diagnostics_probe_to_json_dict(probe_res)
    
    # Verify no raw parameters or full signature arrays are present in the serialized output
    for r in json_dict["target_diagnostics"]:
        for k, v in r.items():
            if isinstance(v, list):
                if k == "ar_coefficients_tensor_shape":
                    assert len(v) == 2
                else:
                    assert False, f"Unexpected list field leaked in diagnostics: {k}"


def test_p51_19_no_forbidden_claims_in_code_or_reason():
    probe_res = run_post_fit_bridge_candidate_diagnostics_probe()
    assert probe_res["no_gsb_claim"] is True
    assert probe_res["no_generation_claim"] is True
    assert probe_res["no_scientific_conclusion"] is True


def test_p51_20_no_scientific_success_claims():
    p1 = pathlib.Path("src/phase2/post_fit_bridge_candidate_diagnostics.py").read_text(encoding="utf-8")
    assert "scientific success" not in p1.lower()
    assert "gsb solved" not in p1.lower()
    assert "optimal transport" not in p1.lower()
    
    p2 = pathlib.Path("reports/PHASE_2_P51_POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_NO_MODEL_NO_VAE_NO_GSB_NO_SCIENCE_REPORT.md")
    if p2.exists():
        text = p2.read_text(encoding="utf-8")
        assert "scientific success" not in text.lower()
        assert "optimal transport" not in text.lower()


def test_p51_21_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/post_fit_bridge_candidate_diagnostics.py",
        "tools/phase2/run_p51_post_fit_bridge_candidate_diagnostics_smoke.py",
        "tests/test_phase2_post_fit_bridge_candidate_diagnostics.py",
        "tests/test_phase2_p51_post_fit_bridge_candidate_diagnostics_smoke.py",
        "reports/PHASE_2_P51_POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_NO_MODEL_NO_VAE_NO_GSB_NO_SCIENCE_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p51-post-fit-bridge-candidate-diagnostics-no-model-no-vae-no-gsb-no-science",
        base_commit="e661e399525338ce0aa4404f3f5dab044220801f",
        allowed_files=allowed,
        phase_label="P51",
    )
