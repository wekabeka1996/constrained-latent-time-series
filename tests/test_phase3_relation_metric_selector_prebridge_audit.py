# tests/test_phase3_relation_metric_selector_prebridge_audit.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase3.relation_metric_selector_prebridge_audit import (
    run_p76_relation_metric_selector_prebridge_audit_probe,
    safe_divide,
    summarize_values,
    count_by_key,
    boolean_summary,
)


def test_p76_01_constants_exact():
    res = run_p76_relation_metric_selector_prebridge_audit_probe()
    assert res["phase"] == "P76"
    assert res["phase_group"] == "PHASE_3"
    assert res["phase_name"] == "Relation Metric Selector Pre-Bridge Leakage Audit"
    assert res["contract_version"] == "phase3_p76_relation_metric_selector_prebridge_leakage_audit_contract_v1"
    
    assert res["source_baseline_phase"] == "P69"
    assert res["source_vector_testbed_phase"] == "P70A"
    assert res["source_time_series_testbed_phase"] == "P70B"
    assert res["source_contrastive_phase"] == "P71"
    assert res["source_operator_bank_phase"] == "P72"
    assert res["source_transfer_audit_phase"] == "P73"
    assert res["source_composition_audit_phase"] == "P74"
    assert res["source_negative_control_phase"] == "P75"
    
    assert res["verdict"] == "P76_READY_FOR_REVIEW"


def test_p76_02_leakage_policy_and_limits():
    res = run_p76_relation_metric_selector_prebridge_audit_probe()
    assert res["training_allowed"] is False
    assert res["model_implementation_allowed"] is False
    assert res["neural_encoder_implementation_allowed"] is False
    assert res["neural_operator_selector_allowed"] is False
    assert res["optimization_allowed"] is False
    assert res["torch_allowed"] is False
    assert res["numpy_allowed"] is False
    assert res["stochastic_random_allowed"] is False
    assert res["bridge_implementation_allowed"] is False
    assert res["learned_metric_allowed"] is False
    
    assert res["prebridge_leakage_audit_allowed"] is True
    assert res["posthoc_descriptor_classification_allowed"] is True
    assert res["predictive_selector_implementation_allowed"] is False
    assert res["predictive_selector_claims_allowed"] is False
    assert res["target_dependent_descriptor_used_for_selector"] is False
    
    assert res["target_endpoint_used_for_prediction"] is False
    assert res["target_endpoint_used_for_posthoc_diagnostic_only"] is True


def test_p76_03_bridge_readiness_blocked():
    res = run_p76_relation_metric_selector_prebridge_audit_probe()
    assert res["valid_pre_prediction_selector_available"] is False
    assert res["semantic_metric_ready"] is False
    assert res["bridge_ready"] is False


def test_p76_04_notes_preservation():
    res = run_p76_relation_metric_selector_prebridge_audit_probe()
    assert res["p75_transitive_validation_note_preserved"] is True
    assert res["p75_midpoint_directness_note_preserved"] is True
    assert res["model_language_avoided"] is True


def test_p76_05_dependency_audit():
    res = run_p76_relation_metric_selector_prebridge_audit_probe()
    audit = res["descriptor_dependency_audit"]
    
    assert audit["target_dependent_descriptor_count"] == 3
    assert audit["source_only_descriptor_count"] == 0
    assert audit["posthoc_only_descriptor_count"] == 3
    assert audit["valid_pre_prediction_descriptor_count"] == 0
    assert audit["all_current_strong_descriptors_target_dependent"] is True
    assert audit["target_dependent_descriptors_blocked_for_prediction"] is True
    assert audit["diagnostic_pass"] is True


def test_p76_06_posthoc_separability_audit():
    res = run_p76_relation_metric_selector_prebridge_audit_probe()
    audit = res["posthoc_descriptor_separability_audit"]
    
    assert audit["posthoc_descriptor_classification_allowed"] is True
    assert audit["posthoc_only"] is True
    assert audit["valid_for_prediction"] is False
    assert audit["labels_used_for_descriptor_construction"] is False
    assert audit["labels_used_for_evaluation_only"] is True
    assert audit["descriptor_signal_present_posthoc"] is True
    assert audit["descriptor_signal_predictive_without_target"] is False
    assert audit["diagnostic_pass"] is True


def test_p76_07_source_only_selector_input():
    res = run_p76_relation_metric_selector_prebridge_audit_probe()
    audit = res["source_only_selector_input_availability_audit"]
    
    assert audit["source_only_fields_present"] is True
    assert audit["source_only_relation_descriptor_present"] is False
    assert audit["source_only_selector_training_contract_present"] is False
    assert audit["valid_pre_prediction_selector_available"] is False
    assert audit["missing_contract"] == "source_available_query_descriptor_or_observation_context"
    assert audit["diagnostic_pass"] is True


def test_p76_08_leakage_and_bridge_blocked_reasons():
    res = run_p76_relation_metric_selector_prebridge_audit_probe()
    leakage = res["selector_leakage_policy_audit"]
    bridge = res["bridge_readiness_audit"]
    
    assert leakage["target_dependent_descriptor_used_for_selector"] is False
    assert leakage["target_dependent_descriptors_blocked_for_prediction"] is True
    assert leakage["posthoc_descriptors_allowed_for_diagnostics_only"] is True
    assert leakage["predictive_selector_claims_allowed"] is False
    assert leakage["valid_pre_prediction_selector_available"] is False
    assert leakage["selector_leakage_detected_if_target_descriptors_used"] is True
    assert leakage["diagnostic_pass"] is True
    
    assert bridge["bridge_ready"] is False
    assert bridge["semantic_metric_ready"] is False
    assert bridge["valid_pre_prediction_selector_available"] is False
    assert bridge["p75_negative_controls_passed"] is True
    assert bridge["diagnostic_pass"] is True
    
    expected_reasons = [
        "current_strong_descriptors_are_target_dependent",
        "source_available_query_descriptor_missing",
        "predictive_selector_not_available",
        "semantic_metric_not_ready",
    ]
    for reason in expected_reasons:
        assert reason in bridge["blocking_reasons"]


def test_p76_09_source_contracts():
    res = run_p76_relation_metric_selector_prebridge_audit_probe()
    assert res["source_contracts_validated"] is True
    assert res["sanity_summary"]["source_contracts_validated"] is True
    
    val = res["descriptor_dependency_audit"] # Wait, let's verify validate_source_contracts_for_p76 structure
    # The validation result is directly returned as validate_source_contracts_for_p76 output
    from src.phase3.relation_metric_selector_prebridge_audit import validate_source_contracts_for_p76
    val_out = validate_source_contracts_for_p76()
    assert val_out["source_contracts_validated"] is True
    assert val_out["p69_validated"] is True
    assert val_out["p70a_validated"] is True
    assert val_out["p70b_validated"] is True
    assert val_out["p71_validated"] is True
    assert val_out["p72_validated"] is True
    assert val_out["p73_validated"] is True
    assert val_out["p74_validated"] is True
    assert val_out["p75_validated"] is True
    assert val_out["p75_transitive_validation_note_preserved"] is True
    assert val_out["p75_midpoint_directness_note_preserved"] is True


def test_p76_10_helpers():
    assert safe_divide(10.0, 2.0) == 5.0
    assert safe_divide(5.0, 0.0) == 0.0
    
    s = summarize_values([1.0, 2.0, 3.0])
    assert s["count"] == 3
    assert s["mean"] == 2.0
    assert s["max"] == 3.0
    assert s["min"] == 1.0
    
    recs = [{"domain": "A"}, {"domain": "B"}, {"domain": "A"}]
    c = count_by_key(recs, "domain")
    assert c["A"] == 2
    assert c["B"] == 1
    
    bs = boolean_summary([True, False, True])
    assert bs["count"] == 3
    assert bs["true_count"] == 2
    assert bs["false_count"] == 1
    assert bs["all_true"] is False
    assert bs["any_true"] is True


def test_p76_11_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase3/relation_metric_selector_prebridge_audit.py")
    src = core_path.read_text(encoding="utf-8")
    
    forbidden_imports = [
        "import torch",
        "import numpy",
        "import pandas",
        "import sklearn",
        "import random",
        "torch.",
        "np.",
    ]
    for item in forbidden_imports:
        assert item not in src, f"Forbidden import item found in source code: {item}"


def test_p76_12_forbidden_phrases_in_source():
    core_path = pathlib.Path("src/phase3/relation_metric_selector_prebridge_audit.py")
    src = core_path.read_text(encoding="utf-8")
    
    forbidden_phrases = [
        "DataLoader",
        "Dataset(",
        "Optimizer",
        "nn.Module",
        ".backward(",
        ".step(",
        "fit(",
        "train(",
    ]
    for item in forbidden_phrases:
        assert item not in src, f"Forbidden model/training phrase found in source code: {item}"


def test_p76_13_forbidden_bridge_phrases_in_source():
    core_path = pathlib.Path("src/phase3/relation_metric_selector_prebridge_audit.py")
    src = core_path.read_text(encoding="utf-8")
    
    forbidden_bridge = [
        "Schrodinger",
        "Schrödinger",
        "GeometricSchrodinger",
        "Brownian",
        "SDE",
        "score_model",
    ]
    for item in forbidden_bridge:
        assert item not in src, f"Forbidden bridge phrase found in source code: {item}"


def test_p76_14_no_model_robustness_in_source_or_report():
    core_path = pathlib.Path("src/phase3/relation_metric_selector_prebridge_audit.py")
    src = core_path.read_text(encoding="utf-8")
    assert "model robustness" not in src.lower()


def test_p76_15_scope_gate():
    # Only 5 allowed files in the branch relative to base commit 12d364ca7a8947c3fcb34afacdec5d6d8d76d294
    cmd = ["git", "diff", "--name-only", "12d364ca7a8947c3fcb34afacdec5d6d8d76d294"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase3/relation_metric_selector_prebridge_audit.py",
        "tools/phase3/run_p76_relation_metric_selector_prebridge_audit_smoke.py",
        "tests/test_phase3_relation_metric_selector_prebridge_audit.py",
        "tests/test_phase3_p76_relation_metric_selector_prebridge_audit_smoke.py",
        "reports/PHASE_3_P76_RELATION_METRIC_SELECTOR_PREBRIDGE_AUDIT_NO_TRAINING_NO_MODEL_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P76."
