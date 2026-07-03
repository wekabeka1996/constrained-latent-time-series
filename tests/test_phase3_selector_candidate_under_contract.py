# tests/test_phase3_selector_candidate_under_contract.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase3.selector_candidate_under_contract import (
    run_p78_selector_candidate_under_contract_probe,
    strip_audit_metadata,
    safe_divide,
    accuracy_summary,
)
from src.phase3.source_available_query_descriptor_contract import (
    build_p70a_source_available_query_descriptor_records,
    build_p70b_source_available_query_descriptor_records,
)
from src.phase3.pure_numeric_relation_testbed import run_p70a_pure_numeric_relation_testbed_probe
from src.phase3.synthetic_time_series_relation_testbed import run_p70b_synthetic_time_series_relation_testbed_probe


def test_p78_01_constants_exact():
    res = run_p78_selector_candidate_under_contract_probe()
    assert res["phase"] == "P78"
    assert res["phase_group"] == "PHASE_3"
    assert res["phase_name"] == "Selector Candidate Under Source-Available Contract"
    assert res["contract_version"] == "phase3_p78_selector_candidate_under_source_available_contract_v1"
    
    assert res["source_baseline_phase"] == "P69"
    assert res["source_vector_testbed_phase"] == "P70A"
    assert res["source_time_series_testbed_phase"] == "P70B"
    assert res["source_contrastive_phase"] == "P71"
    assert res["source_operator_bank_phase"] == "P72"
    assert res["source_transfer_audit_phase"] == "P73"
    assert res["source_composition_audit_phase"] == "P74"
    assert res["source_negative_control_phase"] == "P75"
    assert res["source_prebridge_leakage_phase"] == "P76"
    assert res["source_query_contract_phase"] == "P77"
    
    assert res["verdict"] == "P78_READY_FOR_REVIEW"


def test_p78_02_boundaries_and_leakage():
    res = run_p78_selector_candidate_under_contract_probe()
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
    
    assert res["rule_based_selector_candidate_allowed"] is True
    assert res["predictive_selector_implementation_allowed"] is False
    assert res["learned_selector_evidence_present"] is False
    assert res["predictive_selector_claims_allowed"] is False
    
    assert res["target_endpoint_used_for_selector"] is False
    assert res["target_delta_used_for_selector"] is False
    assert res["exact_relation_label_used_for_selector"] is False
    assert res["exact_operator_id_used_for_selector"] is False
    assert res["audit_metadata_used_for_selector"] is False
    assert res["audit_label_used_for_evaluation_only"] is True


def test_p78_03_bridge_status():
    res = run_p78_selector_candidate_under_contract_probe()
    assert res["valid_for_future_selector_experiment"] is True
    assert res["rule_based_selector_candidate_evaluated"] is True
    assert res["semantic_metric_ready"] is False
    assert res["bridge_ready"] is False


def test_p78_04_notes_preservation():
    res = run_p78_selector_candidate_under_contract_probe()
    assert res["p77_audit_metadata_boundary_preserved"] is True
    assert res["p77_family_uniqueness_risk_preserved"] is True
    assert res["p77_bridge_not_ready_preserved"] is True


def test_p78_05_source_validation():
    res = run_p78_selector_candidate_under_contract_probe()
    assert res["source_contracts_validated"] is True
    assert res["sanity_summary"]["source_contracts_validated"] is True


def test_p78_06_records_audit():
    res = run_p78_selector_candidate_under_contract_probe()
    audit = res["selector_candidate_audit"]["selector_contract_record_audit"]
    
    assert res["sanity_summary"]["rule_based_selector_candidate_evaluated"] is True
    assert res["sanity_summary"]["selector_contract_records_valid"] is True
    
    assert audit["stripped_records_free_of_audit_metadata"] is True
    assert audit["stripped_records_free_of_exact_relation_label"] is True
    assert audit["stripped_records_free_of_exact_operator_id"] is True
    assert audit["stripped_records_free_of_target_endpoint"] is True
    assert audit["stripped_records_free_of_target_delta"] is True
    assert audit["all_records_available_before_target_endpoint"] is True
    assert audit["diagnostic_pass"] is True


def test_p78_07_modes_evaluation_accuracy():
    res = run_p78_selector_candidate_under_contract_probe()
    audit = res["selector_candidate_audit"]
    
    p70a = audit["p70a_selector_results_by_mode"]
    p70b = audit["p70b_selector_results_by_mode"]
    
    # Confirm all modes evaluated
    for mode in [
        "null_majority_baseline",
        "full_contract_rule_selector",
        "no_family_hint_rule_selector",
        "no_family_or_transformation_hint_rule_selector",
        "source_only_summary_selector",
        "intensity_only_selector",
    ]:
        assert mode in p70a
        assert mode in p70b
        
    assert res["sanity_summary"]["null_baseline_accuracy_reported"] is True
    assert res["sanity_summary"]["full_contract_accuracy_reported"] is True
    assert res["sanity_summary"]["no_family_accuracy_reported"] is True
    assert res["sanity_summary"]["source_only_accuracy_reported"] is True
    
    # Check family hint pass through risk present
    assert audit["family_hint_pass_through_risk_present"] is True
    assert res["sanity_summary"]["family_hint_pass_through_risk_present"] is True
    
    assert audit["learned_selector_evidence_present"] is False
    assert res["sanity_summary"]["learned_selector_evidence_present"] is False


def test_p78_08_bridge_boundary():
    res = run_p78_selector_candidate_under_contract_probe()
    audit = res["bridge_boundary_after_selector_candidate_audit"]
    
    assert audit["bridge_ready"] is False
    assert res["sanity_summary"]["bridge_not_ready_preserved"] is True
    assert audit["semantic_metric_ready"] is False
    assert audit["rule_based_selector_candidate_evaluated"] is True
    assert audit["learned_selector_evidence_present"] is False
    assert audit["family_hint_pass_through_risk_present"] is True
    assert audit["diagnostic_pass"] is True
    
    expected_reasons = [
        "selector_candidate_is_rule_based_not_learned",
        "family_hint_pass_through_risk_present",
        "semantic_metric_not_ready",
        "bridge_not_ready_without_learned_selector_evidence",
    ]
    for r in expected_reasons:
        assert r in audit["blocking_reasons"]
    assert audit["next_required_phase"] == "learned_selector_or_metric_candidate_under_ablation_controls"


def test_p78_09_stripped_records_free_of_leaks():
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    recs_a = build_p70a_source_available_query_descriptor_records(p70a_probe)
    recs_b = build_p70b_source_available_query_descriptor_records(p70b_probe)
    
    forbidden_keys = [
        "_audit_metadata",
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
        stripped = strip_audit_metadata(r)
        
        def check_leaks(d: dict):
            for k, v in d.items():
                assert k not in forbidden_keys, f"Forbidden leak key '{k}' found in stripped record."
                if isinstance(v, dict):
                    check_leaks(v)
        check_leaks(stripped)


def test_p78_10_helpers():
    assert safe_divide(10.0, 2.0) == 5.0
    assert safe_divide(5.0, 0.0) == 0.0
    
    s = accuracy_summary([True, False, True])
    assert s["count"] == 3
    assert s["correct_count"] == 2
    assert s["incorrect_count"] == 1
    assert s["accuracy"] == 2.0 / 3.0


def test_p78_11_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase3/selector_candidate_under_contract.py")
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


def test_p78_12_forbidden_phrases_in_source():
    core_path = pathlib.Path("src/phase3/selector_candidate_under_contract.py")
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


def test_p78_13_forbidden_bridge_phrases_in_source():
    core_path = pathlib.Path("src/phase3/selector_candidate_under_contract.py")
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


def test_p78_14_scope_gate():
    # Only 5 allowed files in the branch relative to base commit 0c9160389463736daa3a90f3583d9d6c8c574944
    cmd = ["git", "diff", "--name-only", "0c9160389463736daa3a90f3583d9d6c8c574944"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase3/selector_candidate_under_contract.py",
        "tools/phase3/run_p78_selector_candidate_under_contract_smoke.py",
        "tests/test_phase3_selector_candidate_under_contract.py",
        "tests/test_phase3_p78_selector_candidate_under_contract_smoke.py",
        "reports/PHASE_3_P78_SELECTOR_CANDIDATE_UNDER_SOURCE_AVAILABLE_CONTRACT_NO_TRAINING_NO_MODEL_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P78."
