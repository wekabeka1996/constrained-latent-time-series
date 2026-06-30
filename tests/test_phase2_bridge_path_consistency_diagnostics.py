# tests/test_phase2_bridge_path_consistency_diagnostics.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.bridge_path_consistency_diagnostics import (
    BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_CONTRACT_VERSION,
    BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_KIND,
    BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_MODULE_NAME,
    P52_TARGET_COUNT,
    P52_LAMBDA_VALUES,
    load_torch_for_p52_path_consistency,
    signature_tensor_field_names,
    flatten_signature_tensor_fields,
    build_endpoint_signature_vectors,
    build_p52_post_fit_candidate_signature,
    compute_l2_distance,
    compute_projection_position,
    run_single_p52_path_point_diagnostics,
    run_bridge_path_consistency_diagnostics_probe,
    bridge_path_consistency_diagnostics_probe_to_json_dict,
    compact_bridge_path_consistency_diagnostics_json,
)

# P49/P50/P51 allowed imports
from src.phase2.deterministic_endpoint_bridge_targets import (
    build_p49_endpoint_raw_tensors,
    build_endpoint_signatures,
)
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    P50_STEP_COUNT,
    P50_LEARNING_RATE,
    build_p50_bridge_targets,
)
from src.phase2.post_fit_bridge_candidate_diagnostics import (
    run_single_p51_fit_with_final_raw,
)


def test_p52_01_constants_exact():
    assert BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_CONTRACT_VERSION == "phase2_p52_bridge_path_consistency_diagnostics_contract_v1"
    assert BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_KIND == "bridge_path_consistency_diagnostics_no_model_no_vae_no_gsb_no_science"
    assert BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_MODULE_NAME == "src.phase2.bridge_path_consistency_diagnostics"
    assert P52_TARGET_COUNT == 3
    assert P52_LAMBDA_VALUES == (0.25, 0.50, 0.75)


def test_p52_02_no_top_level_torch_import():
    filepath = "src/phase2/bridge_path_consistency_diagnostics.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p52_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/bridge_path_consistency_diagnostics.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p52_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/bridge_path_consistency_diagnostics.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p52_05_no_fc_vae_imports():
    p = pathlib.Path("src/phase2/bridge_path_consistency_diagnostics.py").read_text(encoding="utf-8")
    assert "fc_vae_model" not in p
    assert "fc_vae_torch_shell" not in p
    assert "fc_vae_torch_module_stub" not in p


def test_p52_06_no_torch_optim():
    p = pathlib.Path("src/phase2/bridge_path_consistency_diagnostics.py").read_text(encoding="utf-8")
    assert "torch.optim" not in p
    assert "optimizer =" not in p
    assert ".step()" not in p
    assert not re.search(r"\bOptimizer\b", p)


def test_p52_07_allowed_imports_present():
    p = pathlib.Path("src/phase2/bridge_path_consistency_diagnostics.py").read_text(encoding="utf-8")
    assert "build_p49_endpoint_raw_tensors" in p
    assert "build_endpoint_signatures" in p
    assert "build_p50_bridge_targets" in p
    assert "run_single_p51_fit_with_final_raw" in p


def test_p52_08_endpoint_signature_vector_length_positive():
    vectors = build_endpoint_signature_vectors()
    assert vectors["vector_length"] > 0
    assert vectors["vector_length"] == 96


def test_p52_09_endpoint_tensor_field_names_deterministic_and_non_empty():
    vectors = build_endpoint_signature_vectors()
    fields = vectors["field_names"]
    assert len(fields) > 0
    assert "ar_spectrum" in fields
    assert "ar_spectrum_log" in fields
    assert "garch_persistence" in fields
    # Should be alphabetically sorted
    assert list(fields) == sorted(list(fields))


def test_p52_10_target_count_exactly_3():
    targets = build_p50_bridge_targets()
    assert len(targets) == 3


def test_p52_11_lambda_values_exact():
    targets = build_p50_bridge_targets()
    lambdas = [t["lambda_value"] for t in targets]
    assert lambdas == [0.25, 0.50, 0.75]


def test_p52_12_single_path_point_distances_finite():
    vectors = build_endpoint_signature_vectors()
    targets = build_p50_bridge_targets()
    for t in targets:
        res = run_single_p52_path_point_diagnostics(t, vectors)
        assert res["distances_finite"] is True
        assert isinstance(res["distance_to_endpoint_A_value"], float)
        assert isinstance(res["distance_to_endpoint_B_value"], float)
        assert isinstance(res["distance_to_own_target_value"], float)


def test_p52_13_single_path_point_projection_finite():
    vectors = build_endpoint_signature_vectors()
    targets = build_p50_bridge_targets()
    for t in targets:
        res = run_single_p52_path_point_diagnostics(t, vectors)
        assert res["projection_position_finite"] is True
        assert isinstance(res["projection_position_value"], float)


def test_p52_14_projection_positions_strictly_increasing():
    probe_res = run_bridge_path_consistency_diagnostics_probe()
    assert probe_res["projection_positions_strictly_increasing"] is True
    positions = probe_res["projection_positions"]
    assert positions[0] < positions[1] < positions[2]


def test_p52_15_own_target_distances_finite():
    probe_res = run_bridge_path_consistency_diagnostics_probe()
    assert probe_res["own_target_distances_finite"] is True
    for d in probe_res["own_target_distances"]:
        assert isinstance(d, float)
        assert d > 0.0


def test_p52_16_all_targets_loss_decreased():
    probe_res = run_bridge_path_consistency_diagnostics_probe()
    assert probe_res["all_targets_loss_decreased"] is True


def test_p52_17_aggregate_path_consistency_passed():
    probe_res = run_bridge_path_consistency_diagnostics_probe()
    assert probe_res["path_consistency_passed"] is True


def test_p52_18_serialization_excludes_raw_and_vectors():
    probe_res = run_bridge_path_consistency_diagnostics_probe()
    json_dict = bridge_path_consistency_diagnostics_probe_to_json_dict(probe_res)
    
    # Verify no raw parameters or full signature arrays are present in the serialized output
    for key in json_dict:
        assert "endpoint_A_vector" not in key
        assert "endpoint_B_vector" not in key
        
    for p in json_dict["path_points"]:
        for k, v in p.items():
            if isinstance(v, list):
                assert False, f"Unexpected list leaked in path point: {k}"


def test_p52_19_no_forbidden_claims_in_code_or_reason():
    probe_res = run_bridge_path_consistency_diagnostics_probe()
    assert probe_res["no_gsb_claim"] is True
    assert probe_res["no_generation_claim"] is True
    assert probe_res["no_scientific_conclusion"] is True


def test_p52_20_no_scientific_success_claims():
    p1 = pathlib.Path("src/phase2/bridge_path_consistency_diagnostics.py").read_text(encoding="utf-8")
    assert "scientific success" not in p1.lower()
    assert "gsb solved" not in p1.lower()
    assert "optimal transport" not in p1.lower()
    
    p2 = pathlib.Path("reports/PHASE_2_P52_BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_NO_MODEL_NO_VAE_NO_GSB_NO_SCIENCE_REPORT.md")
    if p2.exists():
        text = p2.read_text(encoding="utf-8")
        assert "scientific success" not in text.lower()
        assert "optimal transport" not in text.lower()


def test_p52_21_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/bridge_path_consistency_diagnostics.py",
        "tools/phase2/run_p52_bridge_path_consistency_diagnostics_smoke.py",
        "tests/test_phase2_bridge_path_consistency_diagnostics.py",
        "tests/test_phase2_p52_bridge_path_consistency_diagnostics_smoke.py",
        "reports/PHASE_2_P52_BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_NO_MODEL_NO_VAE_NO_GSB_NO_SCIENCE_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p52-bridge-path-consistency-diagnostics-no-model-no-vae-no-gsb-no-science",
        base_commit="cca2fd7c6320660f1d3ace3e62e0b82bef942006",
        allowed_files=allowed,
        phase_label="P52",
    )
