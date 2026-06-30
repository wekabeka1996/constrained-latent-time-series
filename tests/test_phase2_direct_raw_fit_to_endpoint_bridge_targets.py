# tests/test_phase2_direct_raw_fit_to_endpoint_bridge_targets.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.
# We import torch inside the tests if needed, or rely on load_torch_for_p50_direct_bridge_fit.

from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_CONTRACT_VERSION,
    DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_KIND,
    DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_MODULE_NAME,
    P50_STEP_COUNT,
    P50_LEARNING_RATE,
    P50_TARGET_COUNT,
    load_torch_for_p50_direct_bridge_fit,
    build_p50_bridge_targets,
    freeze_bridge_target_signature,
    build_p50_candidate_raw_tensors,
    build_combined_signature_from_raw,
    compute_p50_bridge_target_loss,
    run_single_p50_bridge_fit,
    run_direct_raw_fit_to_bridge_targets_probe,
    direct_raw_fit_to_bridge_targets_probe_to_json_dict,
    compact_direct_raw_fit_to_bridge_targets_json,
)


def test_p50_01_constants_exact():
    assert DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_CONTRACT_VERSION == "phase2_p50_direct_raw_fit_to_endpoint_bridge_targets_contract_v1"
    assert DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_KIND == "direct_raw_fit_to_endpoint_bridge_targets_no_model_no_vae_no_gsb_no_science"
    assert DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_MODULE_NAME == "src.phase2.direct_raw_fit_to_endpoint_bridge_targets"
    assert P50_STEP_COUNT == 30
    assert P50_LEARNING_RATE == 0.05
    assert P50_TARGET_COUNT == 3


def test_p50_02_no_top_level_torch_import():
    filepath = "src/phase2/direct_raw_fit_to_endpoint_bridge_targets.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p50_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/direct_raw_fit_to_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p50_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/direct_raw_fit_to_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p50_05_no_fc_vae_imports():
    p = pathlib.Path("src/phase2/direct_raw_fit_to_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    assert "fc_vae_model" not in p
    assert "fc_vae_torch_shell" not in p
    assert "fc_vae_torch_module_stub" not in p


def test_p50_06_no_torch_optim():
    p = pathlib.Path("src/phase2/direct_raw_fit_to_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    assert "torch.optim" not in p
    assert "optimizer =" not in p
    assert ".step()" not in p
    assert not re.search(r"\bOptimizer\b", p)


def test_p50_07_p46_loss_import_allowed():
    p = pathlib.Path("src/phase2/direct_raw_fit_to_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    assert "combined_moment_spectral_matching_loss" in p
    
    # Check that they do not run actual training/loss loops over datasets
    assert not re.search(r"\bfit\s*\(", p)
    assert not re.search(r"\btrain\b", p)


def test_p50_08_bridge_target_count_exactly_3():
    targets = build_p50_bridge_targets()
    assert len(targets) == 3


def test_p50_09_lambda_values_exact():
    targets = build_p50_bridge_targets()
    lambdas = [t["lambda_value"] for t in targets]
    assert lambdas == [0.25, 0.50, 0.75]


def test_p50_10_candidate_raw_tensor_keys_and_shapes():
    torch = load_torch_for_p50_direct_bridge_fit()
    for l_val in [0.25, 0.50, 0.75]:
        cand = build_p50_candidate_raw_tensors(l_val)
        assert "raw_kappa" in cand
        assert "raw_omega" in cand
        assert "raw_total_mass" in cand
        assert "raw_allocation_logits" in cand
        
        assert cand["raw_kappa"].shape == (2, 5)
        assert cand["raw_omega"].shape == (2, 1)
        assert cand["raw_total_mass"].shape == (2, 1)
        assert cand["raw_allocation_logits"].shape == (2, 2)


def test_p50_11_candidate_tensors_require_grad():
    for l_val in [0.25, 0.50, 0.75]:
        cand = build_p50_candidate_raw_tensors(l_val)
        for k, v in cand.items():
            assert v.requires_grad


def test_p50_12_frozen_target_does_not_require_grad():
    targets = build_p50_bridge_targets()
    for t in targets:
        frozen = freeze_bridge_target_signature(t)
        for k, v in frozen.items():
            if hasattr(v, "requires_grad"):
                assert not v.requires_grad


def test_p50_13_single_target_loss_finite_before_after():
    targets = build_p50_bridge_targets()
    for t in targets:
        res = run_single_p50_bridge_fit(t)
        assert res["initial_loss_finite"] is True
        assert res["final_loss_finite"] is True


def test_p50_14_single_target_loss_decreases():
    targets = build_p50_bridge_targets()
    for t in targets:
        res = run_single_p50_bridge_fit(t)
        assert res["loss_decreased"] is True
        assert res["final_loss_value"] < res["initial_loss_value"]


def test_p50_15_aggregate_targets_passed():
    probe_res = run_direct_raw_fit_to_bridge_targets_probe()
    assert probe_res["targets_passed"] == 3


def test_p50_16_aggregate_all_targets_loss_decreased():
    probe_res = run_direct_raw_fit_to_bridge_targets_probe()
    assert probe_res["all_targets_loss_decreased"] is True


def test_p50_17_serialization_excludes_raw_and_signatures():
    probe_res = run_direct_raw_fit_to_bridge_targets_probe()
    json_dict = direct_raw_fit_to_bridge_targets_probe_to_json_dict(probe_res)
    
    for key in json_dict:
        assert "raw_kappa" not in key
        assert "ar_spectrum" not in key
        
    compact_json = compact_direct_raw_fit_to_bridge_targets_json(probe_res)
    assert "raw_kappa" not in compact_json
    assert "ar_spectrum" not in compact_json


def test_p50_18_no_forbidden_claims_in_code_or_reason():
    probe_res = run_direct_raw_fit_to_bridge_targets_probe()
    assert probe_res["no_gsb_claim"] is True
    assert probe_res["no_generation_claim"] is True
    assert probe_res["no_scientific_conclusion"] is True


def test_p50_19_no_scientific_success_claims():
    p1 = pathlib.Path("src/phase2/direct_raw_fit_to_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    assert "scientific success" not in p1.lower()
    assert "gsb solved" not in p1.lower()
    assert "optimal transport" not in p1.lower()
    
    p2 = pathlib.Path("reports/PHASE_2_P50_DIRECT_RAW_FIT_TO_ENDPOINT_BRIDGE_TARGETS_NO_MODEL_NO_VAE_NO_GSB_NO_SCIENCE_REPORT.md")
    if p2.exists():
        text = p2.read_text(encoding="utf-8")
        assert "scientific success" not in text.lower()
        assert "optimal transport" not in text.lower()


def test_p50_20_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/direct_raw_fit_to_endpoint_bridge_targets.py",
        "tools/phase2/run_p50_direct_raw_fit_to_endpoint_bridge_targets_smoke.py",
        "tests/test_phase2_direct_raw_fit_to_endpoint_bridge_targets.py",
        "tests/test_phase2_p50_direct_raw_fit_to_endpoint_bridge_targets_smoke.py",
        "reports/PHASE_2_P50_DIRECT_RAW_FIT_TO_ENDPOINT_BRIDGE_TARGETS_NO_MODEL_NO_VAE_NO_GSB_NO_SCIENCE_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p50-direct-raw-fit-to-endpoint-bridge-targets-no-model-no-vae-no-gsb-no-science",
        base_commit="d82ad563996d834e2b3b67efc1a5c70f931abfed",
        allowed_files=allowed,
        phase_label="P50",
    )
