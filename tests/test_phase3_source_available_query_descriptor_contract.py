# tests/test_phase3_source_available_query_descriptor_contract.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase3.source_available_query_descriptor_contract import (
    run_p77_source_available_query_descriptor_contract_probe,
    build_p70a_source_available_query_descriptor_records,
    build_p70b_source_available_query_descriptor_records,
    safe_divide,
    boolean_summary,
)
from src.phase3.pure_numeric_relation_testbed import run_p70a_pure_numeric_relation_testbed_probe
from src.phase3.synthetic_time_series_relation_testbed import run_p70b_synthetic_time_series_relation_testbed_probe


def test_p77_01_constants_exact():
    res = run_p77_source_available_query_descriptor_contract_probe()
    assert res["phase"] == "P77"
    assert res["phase_group"] == "PHASE_3"
    assert res["phase_name"] == "Source-Available Query Descriptor Contract"
    assert res["contract_version"] == "phase3_p77_source_available_query_descriptor_contract_v1"
    
    assert res["source_baseline_phase"] == "P69"
    assert res["source_vector_testbed_phase"] == "P70A"
    assert res["source_time_series_testbed_phase"] == "P70B"
    assert res["source_contrastive_phase"] == "P71"
    assert res["source_operator_bank_phase"] == "P72"
    assert res["source_transfer_audit_phase"] == "P73"
    assert res["source_composition_audit_phase"] == "P74"
    assert res["source_negative_control_phase"] == "P75"
    assert res["source_prebridge_leakage_phase"] == "P76"
    
    assert res["verdict"] == "P77_READY_FOR_REVIEW"


def test_p77_02_leakage_and_claims_allowed():
    res = run_p77_source_available_query_descriptor_contract_probe()
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
    
    assert res["source_available_query_contract_allowed"] is True
    assert res["predictive_selector_implementation_allowed"] is False
    assert res["predictive_selector_claims_allowed"] is False
    assert res["learned_selector_evidence_present"] is False
    
    assert res["target_endpoint_used_for_descriptor"] is False
    assert res["target_delta_used_for_descriptor"] is False
    assert res["exact_relation_label_used_for_selector"] is False
    assert res["exact_operator_id_used_for_selector"] is False


def test_p77_03_bridge_status():
    res = run_p77_source_available_query_descriptor_contract_probe()
    assert res["valid_for_future_selector_experiment"] is True
    assert res["semantic_metric_ready"] is False
    assert res["bridge_ready"] is False


def test_p77_04_notes_preservation():
    res = run_p77_source_available_query_descriptor_contract_probe()
    assert res["p76_predictive_selector_block_preserved"] is True
    assert res["p76_bridge_not_ready_preserved"] is True
    assert res["p76_direct_plus_transitive_validation_note_preserved"] is True


def test_p77_05_source_contract_validation():
    res = run_p77_source_available_query_descriptor_contract_probe()
    assert res["source_contracts_validated"] is True
    assert res["sanity_summary"]["source_contracts_validated"] is True


def test_p77_06_records_built_and_pre_endpoint_status():
    res = run_p77_source_available_query_descriptor_contract_probe()
    audit = res["source_available_query_contract_audit"]
    
    assert res["sanity_summary"]["source_available_query_contract_present"] is True
    assert res["sanity_summary"]["descriptor_records_built"] is True
    assert res["sanity_summary"]["all_descriptor_records_available_before_target_endpoint"] is True
    
    assert audit["source_available_query_contract_present"] is True
    assert audit["endpoint_leakage_detected"] is False
    assert audit["target_delta_leakage_detected"] is False
    assert audit["exact_label_pass_through_detected"] is False
    assert audit["family_uniqueness_risk_reported"] is True
    assert audit["valid_for_future_selector_experiment"] is True
    assert audit["diagnostic_pass"] is True


def test_p77_07_future_selector_readiness():
    res = run_p77_source_available_query_descriptor_contract_probe()
    audit = res["future_selector_experiment_readiness_audit"]
    
    assert audit["valid_for_future_selector_experiment"] is True
    assert audit["learned_selector_evidence_present"] is False
    assert audit["predictive_selector_implementation_present"] is False
    assert audit["predictive_selector_claims_allowed"] is False
    assert audit["p76_predictive_selector_block_preserved"] is True
    assert audit["diagnostic_pass"] is True
    
    expected_changed = [
        "source_available_query_descriptor_contract_defined",
        "endpoint_leakage_blocked_by_schema",
        "exact_relation_label_pass_through_blocked_by_schema",
    ]
    for c in expected_changed:
        assert c in audit["what_changed_since_p76"]
        
    expected_not_changed = [
        "no_learned_selector_evidence",
        "semantic_metric_not_ready",
        "bridge_not_ready",
    ]
    for nc in expected_not_changed:
        assert nc in audit["what_did_not_change_since_p76"]


def test_p77_08_bridge_boundary():
    res = run_p77_source_available_query_descriptor_contract_probe()
    audit = res["bridge_boundary_after_contract_audit"]
    
    assert audit["bridge_ready"] is False
    assert res["sanity_summary"]["bridge_not_ready_preserved"] is True
    assert audit["semantic_metric_ready"] is False
    assert audit["valid_source_available_contract_present"] is True
    assert audit["valid_pre_prediction_selector_available"] is False
    assert audit["learned_selector_evidence_present"] is False
    assert audit["diagnostic_pass"] is True
    
    expected_reasons = [
        "contract_exists_but_selector_not_learned_or_validated",
        "semantic_metric_not_ready",
        "bridge_not_ready_without_selector_evidence",
    ]
    for r in expected_reasons:
        assert r in audit["blocking_reasons"]
    assert audit["next_required_phase"] == "learned_or_rule_based_selector_candidate_under_contract"


def test_p77_09_descriptor_records_forbidden_keys():
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    recs_a = build_p70a_source_available_query_descriptor_records(p70a)
    recs_b = build_p70b_source_available_query_descriptor_records(p70b)
    
    forbidden_keys = [
        "relation_type",
        "operator_id",
        "z_b",
        "params_b",
        "series_b",
        "z_b_minus_z_a",
        "params_b_minus_params_a",
        "series_summary_delta",
    ]
    
    for r in recs_a + recs_b:
        # Recursively check no forbidden key exists at any level (excluding _audit_metadata)
        def check_dict(d: dict):
            for k, v in d.items():
                if k == "_audit_metadata":
                    continue
                assert k not in forbidden_keys, f"Forbidden key '{k}' found in descriptor record."
                if isinstance(v, dict):
                    check_dict(v)
        check_dict(r)


def test_p77_10_helpers():
    assert safe_divide(10.0, 2.0) == 5.0
    assert safe_divide(5.0, 0.0) == 0.0
    
    bs = boolean_summary([True, False, True])
    assert bs["count"] == 3
    assert bs["true_count"] == 2
    assert bs["false_count"] == 1
    assert bs["all_true"] is False
    assert bs["any_true"] is True


def test_p77_11_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase3/source_available_query_descriptor_contract.py")
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


def test_p77_12_forbidden_phrases_in_source():
    core_path = pathlib.Path("src/phase3/source_available_query_descriptor_contract.py")
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


def test_p77_13_forbidden_bridge_phrases_in_source():
    core_path = pathlib.Path("src/phase3/source_available_query_descriptor_contract.py")
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


def test_p77_14_scope_gate():
    # Only 5 allowed files in the branch relative to base commit 4e213400f771d994254293973babeb43374ff4fd
    cmd = ["git", "diff", "--name-only", "4e213400f771d994254293973babeb43374ff4fd"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase3/source_available_query_descriptor_contract.py",
        "tools/phase3/run_p77_source_available_query_descriptor_contract_smoke.py",
        "tests/test_phase3_source_available_query_descriptor_contract.py",
        "tests/test_phase3_p77_source_available_query_descriptor_contract_smoke.py",
        "reports/PHASE_3_P77_SOURCE_AVAILABLE_QUERY_DESCRIPTOR_CONTRACT_NO_TRAINING_NO_MODEL_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P77."
