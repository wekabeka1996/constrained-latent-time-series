# tests/test_phase3_hard_ablated_selector_evidence_gate.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase3.hard_ablated_selector_evidence_gate import (
    run_p79_hard_ablated_selector_evidence_gate_probe,
    strip_for_hard_ablated_selector,
    build_hard_ablated_records,
)
from src.phase3.source_available_query_descriptor_contract import (
    build_p70a_source_available_query_descriptor_records,
    build_p70b_source_available_query_descriptor_records,
)
from src.phase3.pure_numeric_relation_testbed import run_p70a_pure_numeric_relation_testbed_probe
from src.phase3.synthetic_time_series_relation_testbed import run_p70b_synthetic_time_series_relation_testbed_probe


def test_p79_01_constants_exact():
    res = run_p79_hard_ablated_selector_evidence_gate_probe()
    assert res["phase"] == "P79"
    assert res["phase_group"] == "PHASE_3"
    assert res["phase_name"] == "Hard-Ablated Selector Evidence Gate"
    assert res["contract_version"] == "phase3_p79_hard_ablated_selector_evidence_gate_v1"
    
    assert res["source_baseline_phase"] == "P69"
    assert res["source_vector_testbed_phase"] == "P70A"
    assert res["source_time_series_testbed_phase"] == "P70B"
    assert res["source_contrastive_phase"] == "P71"
    assert res["source_operator_bank_phase"] == "P72"
    assert res["source_transfer_audit_phase"] == "P73"
    assert res["source_composition_phase"] == "P74"
    assert res["source_negative_control_phase"] == "P75"
    assert res["source_prebridge_leakage_phase"] == "P76"
    assert res["source_query_contract_phase"] == "P77"
    assert res["source_selector_candidate_phase"] == "P78"
    
    assert res["verdict"] == "P79_READY_FOR_REVIEW"


def test_p79_02_boundaries_and_leakage():
    res = run_p79_hard_ablated_selector_evidence_gate_probe()
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
    
    assert res["hard_ablated_selector_evidence_gate_allowed"] is True
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
    
    assert res["relation_family_hint_used_for_selector"] is False
    assert res["transformation_class_hint_used_for_selector"] is False
    assert res["relation_axis_hint_used_for_selector"] is False
    assert res["parameter_group_hint_used_for_selector"] is False


def test_p79_03_bridge_status():
    res = run_p79_hard_ablated_selector_evidence_gate_probe()
    assert res["relation_specific_hints_removed"] is True
    assert res["hard_ablated_selector_evidence_gate_evaluated"] is True
    assert res["hard_ablated_selector_signal_present"] is False
    assert res["hard_ablated_selector_beats_null_baseline"] is False
    assert res["semantic_metric_ready"] is False
    assert res["bridge_ready"] is False


def test_p79_04_notes_preservation():
    res = run_p79_hard_ablated_selector_evidence_gate_probe()
    assert res["p78_hint_pass_through_note_preserved"] is True
    assert res["p78_no_learned_selector_evidence_preserved"] is True
    assert res["p78_bridge_not_ready_preserved"] is True


def test_p79_05_source_validation():
    res = run_p79_hard_ablated_selector_evidence_gate_probe()
    assert res["source_contracts_validated"] is True
    assert res["sanity_summary"]["source_contracts_validated"] is True


def test_p79_06_records_audit():
    res = run_p79_hard_ablated_selector_evidence_gate_probe()
    audit = res["hard_ablated_selector_evidence_gate_audit"]
    
    assert res["sanity_summary"]["hard_ablated_selector_evidence_gate_evaluated"] is True
    assert res["sanity_summary"]["relation_specific_hints_removed"] is True
    assert res["sanity_summary"]["hard_ablated_records_valid"] is True
    
    assert audit["diagnostic_pass"] is True
    
    for domain_audit in [audit["p70a_hard_ablated_input_audit"], audit["p70b_hard_ablated_input_audit"]]:
        assert domain_audit["audit_metadata_present_count"] == 0
        assert domain_audit["exact_label_present_count"] == 0
        assert domain_audit["exact_operator_id_present_count"] == 0
        assert domain_audit["target_endpoint_present_count"] == 0
        assert domain_audit["target_delta_present_count"] == 0
        assert domain_audit["relation_family_hint_present_count"] == 0
        assert domain_audit["transformation_class_hint_present_count"] == 0
        assert domain_audit["axis_or_group_hint_present_count"] == 0
        assert domain_audit["all_records_available_before_target_endpoint"] is True
        assert domain_audit["diagnostic_pass"] is True


def test_p79_07_modes_evaluation_accuracy():
    res = run_p79_hard_ablated_selector_evidence_gate_probe()
    audit = res["hard_ablated_selector_evidence_gate_audit"]
    
    p70a = audit["p70a_results_by_mode"]
    p70b = audit["p70b_results_by_mode"]
    
    # Confirm all modes evaluated
    expected_modes = [
        "null_majority_baseline",
        "source_only_summary_selector",
        "intensity_only_selector",
        "source_plus_intensity_selector",
        "context_only_selector",
        "split_only_selector",
    ]
    for mode in expected_modes:
        assert mode in p70a
        assert mode in p70b
        
    assert res["sanity_summary"]["null_baseline_accuracy_reported"] is True
    assert res["sanity_summary"]["source_only_accuracy_reported"] is True
    assert res["sanity_summary"]["intensity_only_accuracy_reported"] is True
    assert res["sanity_summary"]["source_plus_intensity_accuracy_reported"] is True
    assert res["sanity_summary"]["context_only_accuracy_reported"] is True
    assert res["sanity_summary"]["split_only_accuracy_reported"] is True


def test_p79_08_null_comparison():
    res = run_p79_hard_ablated_selector_evidence_gate_probe()
    audit = res["hard_ablated_selector_evidence_gate_audit"]
    
    comp_a = audit["p70a_null_comparison"]
    comp_b = audit["p70b_null_comparison"]
    
    assert comp_a["any_mode_beats_null_by_material_margin"] is False
    assert comp_b["any_mode_beats_null_by_material_margin"] is False
    
    assert comp_a["best_lift_over_null"] == 0.0
    assert comp_b["best_lift_over_null"] == 0.0


def test_p79_09_bridge_boundary():
    res = run_p79_hard_ablated_selector_evidence_gate_probe()
    audit = res["bridge_boundary_after_hard_ablation_audit"]
    
    assert audit["bridge_ready"] is False
    assert res["sanity_summary"]["bridge_not_ready_preserved"] is True
    assert audit["semantic_metric_ready"] is False
    assert audit["hard_ablated_selector_evidence_gate_evaluated"] is True
    assert audit["learned_selector_evidence_present"] is False
    assert audit["hard_ablated_selector_signal_present"] is False
    assert audit["diagnostic_pass"] is True
    
    expected_reasons = [
        "hard_ablated_selector_signal_not_established",
        "learned_selector_evidence_not_present",
        "semantic_metric_not_ready",
        "bridge_not_ready_without_selector_evidence",
    ]
    for r in expected_reasons:
        assert r in audit["blocking_reasons"]
    assert audit["next_required_phase"] == "learned_selector_or_metric_candidate_with_clean_source_observation_context"


def test_p79_10_hard_ablated_records_free_of_hints_and_leaks():
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    recs_a = build_p70a_source_available_query_descriptor_records(p70a_probe)
    recs_b = build_p70b_source_available_query_descriptor_records(p70b_probe)
    
    ablated_a = build_hard_ablated_records(recs_a)
    ablated_b = build_hard_ablated_records(recs_b)
    
    forbidden_keys = [
        "_audit_metadata",
        "relation_type",
        "operator_id",
        "z_b",
        "params_b",
        "series_b",
        "z_end",
        "params_end",
        "series_end",
        "target_endpoint",
        "target_midpoint",
        "z_b_minus_z_a",
        "params_b_minus_params_a",
        "series_summary_delta",
        "relation_family_hint",
        "transformation_class_hint",
        "relation_axis_hint",
        "parameter_group_hint",
    ]
    
    for r in ablated_a + ablated_b:
        def check_leaks(d: dict):
            for k, v in d.items():
                assert k not in forbidden_keys, f"Forbidden leak key '{k}' found in ablated record."
                if isinstance(v, dict):
                    check_leaks(v)
        check_leaks(r)


def test_p79_11_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase3/hard_ablated_selector_evidence_gate.py")
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


def test_p79_12_forbidden_phrases_in_source():
    core_path = pathlib.Path("src/phase3/hard_ablated_selector_evidence_gate.py")
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


def test_p79_13_forbidden_bridge_phrases_in_source():
    core_path = pathlib.Path("src/phase3/hard_ablated_selector_evidence_gate.py")
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


def test_p79_14_scope_gate():
    # Only 5 allowed files in the branch relative to base commit b1d2fa70edcbe5cb250d20ce81dbe7138aabe9ed
    cmd = ["git", "diff", "--name-only", "b1d2fa70edcbe5cb250d20ce81dbe7138aabe9ed"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase3/hard_ablated_selector_evidence_gate.py",
        "tools/phase3/run_p79_hard_ablated_selector_evidence_gate_smoke.py",
        "tests/test_phase3_hard_ablated_selector_evidence_gate.py",
        "tests/test_phase3_p79_hard_ablated_selector_evidence_gate_smoke.py",
        "reports/PHASE_3_P79_HARD_ABLATED_SELECTOR_EVIDENCE_GATE_NO_TRAINING_NO_MODEL_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P79."
