# tests/test_phase4_support_delta_metric_hard_generalization.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase4.support_delta_metric_hard_generalization import (
    run_p89_support_delta_metric_hard_generalization_probe,
    ALL_CLASSES,
)


def test_p89_01_probe_constants():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    assert res["phase"] == "P89"
    assert res["phase_group"] == "PHASE_4"
    assert res["phase_name"] == "Support-Delta Metric Hard Generalization and External Context"
    assert res["contract_version"] == "phase4_p89_support_delta_metric_hard_generalization_v1"
    
    assert res["source_nonlearned_metric_phase"] == "P88"
    assert res["source_few_shot_selector_phase"] == "P87"
    assert res["source_materializer_phase"] == "P86"
    assert res["source_externalized_context_phase"] == "P85"
    assert res["source_enrichment_phase"] == "P84"
    assert res["source_identifiability_phase"] == "P83"
    assert res["source_selector_pilot_phase"] == "P82"
    assert res["source_dataset_phase"] == "P81"
    
    assert res["verdict"] == "P89_READY_FOR_REVIEW"


def test_p89_02_no_torch_import():
    core_path = pathlib.Path("src/phase4/support_delta_metric_hard_generalization.py")
    src = core_path.read_text(encoding="utf-8")
    
    assert "import torch" not in src
    assert "torch." not in src


def test_p89_03_no_training_performed():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    assert res["model_training_performed"] is False
    assert res["torch_training_performed"] is False
    assert res["new_model_implemented"] is False
    assert res["optimizer_created"] is False
    assert res["checkpoint_written"] is False


def test_p89_04_source_contracts_and_reproduction():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    assert res["source_contracts_validated"] is True
    
    repro = res["p88_reproduction_result"]
    assert repro["effective_accuracy"] >= 0.60
    assert repro["mode"] == "delta_only"
    assert repro["distance_family"] == "l2"


def test_p89_05_hard_generalization_results():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    assert "hard_generalization_results" in res
    
    hgr = res["hard_generalization_results"]
    
    # base state generalization
    base_state = hgr["train_to_heldout_base_state"]
    assert "effective_accuracy" in base_state
    assert "coverage" in base_state
    
    # magnitude generalization
    magnitude = hgr["train_to_heldout_magnitude"]
    assert "effective_accuracy" in magnitude
    assert "coverage" in magnitude
    
    # Assert at least one hard generalization accuracy is >= 0.50
    best_hard = max(base_state["effective_accuracy"], magnitude["effective_accuracy"])
    assert best_hard >= 0.50
    assert res["hard_generalization_supported"] is True

    # Assert new hard policy control flags and dictionaries
    assert res["best_raw_hard_policy"] in ["train_to_heldout_base_state", "train_to_heldout_magnitude"]
    assert res["best_evidence_passing_hard_policy"] in ["train_to_heldout_base_state", "train_to_heldout_magnitude"]
    assert res["hard_policy_controls_aligned"] is True
    assert res["hard_policy_fallback_controls_used"] is False

    assert "hard_policy_control_results" in res
    hpcr = res["hard_policy_control_results"]
    assert "train_to_heldout_base_state" in hpcr
    assert "train_to_heldout_magnitude" in hpcr

    base_ctrls = hpcr["train_to_heldout_base_state"]
    assert "query_source_only_negative_control" in base_ctrls
    assert "shuffled_support_delta_negative_control" in base_ctrls
    assert "zero_delta_negative_control" in base_ctrls

    mag_ctrls = hpcr["train_to_heldout_magnitude"]
    assert "query_source_only_negative_control" in mag_ctrls
    assert "shuffled_support_delta_negative_control" in mag_ctrls
    assert "zero_delta_negative_control" in mag_ctrls


def test_p89_06_leave_relation_family_out():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    hgr = res["hard_generalization_results"]
    
    family = hgr["leave_relation_family_out"]
    assert "effective_accuracy" in family
    assert "coverage" in family
    assert family["coverage"] < 1.0
    assert family["unsupported_sample_count"] > 0
    
    # Verify target predictions for unsupported classes are flagged correctly
    predictions = family["predictions"]
    for pred in predictions:
        if pred["true_label"] in family["held_out_classes"]:
            assert pred["unsupported"] is True
            assert pred["is_correct"] is False


def test_p89_07_leave_domain_out():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    hgr = res["hard_generalization_results"]
    
    domain_out = hgr["leave_domain_out"]
    assert domain_out["train_p70a_test_p70b"]["coverage"] == 0.0
    assert domain_out["train_p70b_test_p70a"]["coverage"] == 0.0


def test_p89_08_leakage_and_audits():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    assert res["metric_input_leakage_audit"]["diagnostic_pass"] is True
    assert "support_policy_flow_audits" in res


def test_p89_09_shuffled_control_assertions():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    ctrl = res["negative_control_results"]
    
    assert ctrl["shuffled_support_delta_control_implemented"] is True
    assert ctrl["shuffled_control_target_labels_preserved"] is True
    assert ctrl["shuffled_control_support_deltas_permuted"] is True
    assert ctrl["target_labels_shifted"] is False
    assert ctrl["true_labels_preserved"] is True
    
    # Assert shuffled control degrades performance significantly
    best_hard = max(
        res["hard_generalization_results"]["train_to_heldout_base_state"]["effective_accuracy"],
        res["hard_generalization_results"]["train_to_heldout_magnitude"]["effective_accuracy"]
    )
    assert best_hard - ctrl["shuffled_support_delta_negative_control"] >= 0.10


def test_p89_10_domain_and_relation_breakdowns():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    assert "domain_results" in res
    assert "p70a_vector_world" in res["domain_results"]
    assert "p70b_time_series_parameter_world" in res["domain_results"]
    
    assert "relation_results" in res
    for cls in ALL_CLASSES:
        if cls in res["relation_results"]:
            assert "accuracy" in res["relation_results"][cls]


def test_p89_11_nonlinear_x_from_y_stress():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    assert "nonlinear_x_from_y_stress_result" in res
    stress = res["nonlinear_x_from_y_stress_result"]
    assert "nonlinear_x_from_y_total" in stress
    assert "nonlinear_x_from_y_correct" in stress
    assert "nonlinear_x_from_y_predicted_as_translate_x" in stress


def test_p89_12_external_context_contract():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    assert "external_context_contract" in res
    contract = res["external_context_contract"]
    assert contract["external_context_contract_defined"] is True
    assert contract["requires_real_external_support_demonstrations"] is True
    assert contract["requires_query_observation_source"] is True


def test_p89_13_bridge_ready_remains_false():
    res = run_p89_support_delta_metric_hard_generalization_probe()
    assert res["learned_selector_evidence_present"] is False
    assert res["learned_metric_evidence_present"] is False
    assert res["semantic_metric_ready"] is False
    assert res["bridge_implementation_allowed"] is False
    assert res["bridge_ready"] is False
    assert res["generation_claims_allowed"] is False
    assert res["semantic_geometry_claims_allowed"] is False


def test_p89_14_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase4/support_delta_metric_hard_generalization.py")
    src = core_path.read_text(encoding="utf-8")
    
    forbidden = ["import numpy", "import pandas", "import sklearn", "import random", "np."]
    for item in forbidden:
        assert item not in src


def test_p89_15_scope_gate():
    cmd = ["git", "diff", "--name-only", "311e3887dd73a1999401dc604fa0036c294ae75d"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase4/support_delta_metric_hard_generalization.py",
        "tools/phase4/run_p89_support_delta_metric_hard_generalization_smoke.py",
        "tests/test_phase4_support_delta_metric_hard_generalization.py",
        "tests/test_phase4_p89_support_delta_metric_hard_generalization_smoke.py",
        "reports/PHASE_4_P89_SUPPORT_DELTA_METRIC_HARD_GENERALIZATION_AND_EXTERNAL_CONTEXT_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P89."
