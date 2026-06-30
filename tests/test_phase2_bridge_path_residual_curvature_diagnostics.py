# tests/test_phase2_bridge_path_residual_curvature_diagnostics.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.bridge_path_residual_curvature_diagnostics import (
    BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_CONTRACT_VERSION,
    BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_KIND,
    BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_MODULE_NAME,
    P53_TARGET_COUNT,
    P53_LAMBDA_VALUES,
    load_torch_for_p53_residual_curvature,
    build_p53_path_vectors,
    compute_line_closest_point,
    run_single_p53_residual_point_diagnostics,
    run_p53_curvature_smoothness_diagnostics,
    run_bridge_path_residual_curvature_diagnostics_probe,
    bridge_path_residual_curvature_diagnostics_probe_to_json_dict,
    compact_bridge_path_residual_curvature_diagnostics_json,
)

# P52 / P50 allowed imports to check
from src.phase2.bridge_path_consistency_diagnostics import (
    build_endpoint_signature_vectors,
    compute_l2_distance,
)
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    build_p50_bridge_targets,
)


def test_p53_01_constants_exact():
    assert BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_CONTRACT_VERSION == "phase2_p53_bridge_path_residual_curvature_diagnostics_contract_v1"
    assert BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_KIND == "bridge_path_residual_curvature_diagnostics_no_model_no_vae_no_gsb_no_science"
    assert BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_MODULE_NAME == "src.phase2.bridge_path_residual_curvature_diagnostics"
    assert P53_TARGET_COUNT == 3
    assert P53_LAMBDA_VALUES == (0.25, 0.50, 0.75)


def test_p53_02_no_top_level_torch_import():
    filepath = "src/phase2/bridge_path_residual_curvature_diagnostics.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p53_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/bridge_path_residual_curvature_diagnostics.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p53_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/bridge_path_residual_curvature_diagnostics.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p53_05_no_fc_vae_imports():
    p = pathlib.Path("src/phase2/bridge_path_residual_curvature_diagnostics.py").read_text(encoding="utf-8")
    assert "fc_vae_model" not in p
    assert "fc_vae_torch_shell" not in p
    assert "fc_vae_torch_module_stub" not in p


def test_p53_06_no_torch_optim():
    p = pathlib.Path("src/phase2/bridge_path_residual_curvature_diagnostics.py").read_text(encoding="utf-8")
    assert "torch.optim" not in p
    assert "optimizer =" not in p
    assert ".step()" not in p
    assert not re.search(r"\bOptimizer\b", p)


def test_p53_07_allowed_p52_p50_imports_present():
    p = pathlib.Path("src/phase2/bridge_path_residual_curvature_diagnostics.py").read_text(encoding="utf-8")
    assert "build_endpoint_signature_vectors" in p
    assert "build_p52_post_fit_candidate_signature" in p
    assert "flatten_signature_tensor_fields" in p
    assert "compute_l2_distance" in p
    assert "compute_projection_position" in p
    assert "build_p50_bridge_targets" in p


def test_p53_08_path_vectors_internal_build():
    res = build_p53_path_vectors()
    assert res["signature_vector_length"] == 96
    assert len(res["path_points_internal"]) == 3
    assert res["endpoint_distance_value"] > 0.0


def test_p53_09_no_raw_vector_serialization():
    probe_res = run_bridge_path_residual_curvature_diagnostics_probe()
    json_dict = bridge_path_residual_curvature_diagnostics_probe_to_json_dict(probe_res)
    
    for key, val in json_dict.items():
        if "vector" in key.lower():
            assert key == "signature_vector_length"
            assert isinstance(val, int)
        assert "raw" not in key.lower()
        
    for pt in json_dict["path_points"]:
        for k, v in pt.items():
            if isinstance(v, list):
                assert False, f"Unexpected list leaked in path point: {k}"


def test_p53_10_per_point_projection_finite():
    path_vectors = build_p53_path_vectors()
    for pt in path_vectors["path_points_internal"]:
        res = run_single_p53_residual_point_diagnostics(pt, path_vectors)
        assert res["projection_position_finite"] is True
        assert isinstance(res["projection_position_value"], float)


def test_p53_11_per_point_line_residual_finite():
    path_vectors = build_p53_path_vectors()
    for pt in path_vectors["path_points_internal"]:
        res = run_single_p53_residual_point_diagnostics(pt, path_vectors)
        assert res["line_residual_finite"] is True
        assert isinstance(res["line_residual_value"], float)


def test_p53_12_per_point_normalized_line_residual_finite():
    path_vectors = build_p53_path_vectors()
    for pt in path_vectors["path_points_internal"]:
        res = run_single_p53_residual_point_diagnostics(pt, path_vectors)
        assert res["normalized_line_residual_value"] is not None
        assert isinstance(res["normalized_line_residual_value"], float)


def test_p53_13_own_target_distance_finite():
    path_vectors = build_p53_path_vectors()
    for pt in path_vectors["path_points_internal"]:
        res = run_single_p53_residual_point_diagnostics(pt, path_vectors)
        assert res["distance_to_own_target_finite"] is True
        assert isinstance(res["distance_to_own_target_value"], float)


def test_p53_14_projection_positions_strictly_increasing():
    probe_res = run_bridge_path_residual_curvature_diagnostics_probe()
    assert probe_res["projection_positions_strictly_increasing"] is True


def test_p53_15_curvature_smoothness_values_finite():
    probe_res = run_bridge_path_residual_curvature_diagnostics_probe()
    assert probe_res["curvature_values_finite"] is True
    assert isinstance(probe_res["bend_ratio_value"], float)
    assert isinstance(probe_res["curvature_proxy_value"], float)


def test_p53_16_segment_length_ratio_finite_and_positive():
    probe_res = run_bridge_path_residual_curvature_diagnostics_probe()
    assert probe_res["segment_lengths_finite"] is True
    ratio = probe_res["segment_length_ratio_value"]
    assert ratio > 0.0
    assert isinstance(ratio, float)


def test_p53_17_bend_ratio_finite_and_bounds():
    probe_res = run_bridge_path_residual_curvature_diagnostics_probe()
    bend = probe_res["bend_ratio_value"]
    assert bend >= 1.0
    assert isinstance(bend, float)


def test_p53_18_midpoint_deviation_finite():
    probe_res = run_bridge_path_residual_curvature_diagnostics_probe()
    assert probe_res["midpoint_deviation_finite"] is True
    assert isinstance(probe_res["midpoint_deviation_value"], float)
    assert isinstance(probe_res["normalized_midpoint_deviation_value"], float)


def test_p53_19_aggregate_residual_curvature_diagnostics_passed():
    probe_res = run_bridge_path_residual_curvature_diagnostics_probe()
    assert probe_res["residual_curvature_diagnostics_passed"] is True


def test_p53_20_aggregate_path_shape_diagnostics_passed():
    probe_res = run_bridge_path_residual_curvature_diagnostics_probe()
    assert probe_res["path_shape_diagnostics_passed"] is True


def test_p53_21_no_forbidden_claims_in_code_or_reason():
    probe_res = run_bridge_path_residual_curvature_diagnostics_probe()
    assert probe_res["no_gsb_claim"] is True
    assert probe_res["no_generation_claim"] is True
    assert probe_res["no_scientific_conclusion"] is True


def test_p53_22_no_scientific_success_claims():
    p1 = pathlib.Path("src/phase2/bridge_path_residual_curvature_diagnostics.py").read_text(encoding="utf-8")
    assert "scientific success" not in p1.lower()
    assert "gsb solved" not in p1.lower()
    assert "optimal transport" not in p1.lower()
    
    p2 = pathlib.Path("reports/PHASE_2_P53_BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_NO_MODEL_NO_VAE_NO_GSB_NO_SCIENCE_REPORT.md")
    if p2.exists():
        text = p2.read_text(encoding="utf-8")
        assert "scientific success" not in text.lower()
        assert "optimal transport" not in text.lower()


def test_p53_23_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/bridge_path_residual_curvature_diagnostics.py",
        "tools/phase2/run_p53_bridge_path_residual_curvature_diagnostics_smoke.py",
        "tests/test_phase2_bridge_path_residual_curvature_diagnostics.py",
        "tests/test_phase2_p53_bridge_path_residual_curvature_diagnostics_smoke.py",
        "reports/PHASE_2_P53_BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_NO_MODEL_NO_VAE_NO_GSB_NO_SCIENCE_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p53-bridge-path-residual-curvature-diagnostics-no-model-no-vae-no-gsb-no-science",
        base_commit="a6ca2465a9111b42efc255a819ad480121c86105",
        allowed_files=allowed,
        phase_label="P53",
    )
