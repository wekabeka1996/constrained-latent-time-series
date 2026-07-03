# tests/test_phase3_global_negative_controls_collapse_audit.py

import inspect
import json
import math
import pathlib
import pytest

from src.phase3.global_negative_controls_collapse_audit import (
    PHASE,
    PHASE_GROUP,
    PHASE_NAME,
    CONTRACT_VERSION,
    SOURCE_BASELINE_PHASE,
    SOURCE_VECTOR_TESTBED_PHASE,
    SOURCE_TIME_SERIES_TESTBED_PHASE,
    SOURCE_CONTRASTIVE_PHASE,
    SOURCE_OPERATOR_BANK_PHASE,
    SOURCE_TRANSFER_AUDIT_PHASE,
    SOURCE_COMPOSITION_AUDIT_PHASE,
    TRAINING_ALLOWED,
    MODEL_IMPLEMENTATION_ALLOWED,
    NEURAL_ENCODER_IMPLEMENTATION_ALLOWED,
    NEURAL_OPERATOR_SELECTOR_ALLOWED,
    OPTIMIZATION_ALLOWED,
    TORCH_ALLOWED,
    NUMPY_ALLOWED,
    STOCHASTIC_RANDOM_ALLOWED,
    BRIDGE_IMPLEMENTATION_ALLOWED,
    LEARNED_METRIC_ALLOWED,
    GLOBAL_NEGATIVE_CONTROL_AUDIT_ALLOWED,
    COLLAPSE_AUDIT_ALLOWED,
    ORACLE_OPERATOR_BANK_REUSE_ALLOWED,
    ORACLE_RELATION_TYPE_SELECTION_ALLOWED_FOR_POSITIVE_CONTROL,
    LEARNED_RELATION_TYPE_SELECTION_ALLOWED,
    LEARNED_NEGATIVE_CONTROL_CLAIMS_ALLOWED,
    TARGET_ENDPOINT_USED_FOR_PREDICTION,
    TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY,
    TARGET_MIDPOINT_USED_FOR_PREDICTION,
    TARGET_MIDPOINT_USED_FOR_EVALUATION_ONLY,
    PRIMARY_EMPIRICAL_TARGET,
    VERDICT,
    AUDIT_DOMAINS,
    NEGATIVE_CONTROL_FAMILIES,
    FORBIDDEN_CLAIMS,
    ALLOWED_CLAIMS,
    safe_divide,
    summarize_values,
    summary_is_zero,
    summary_is_nonzero,
    vectors_max_abs_error,
    parameter_keys,
    assign_wrong_p70a_relation_type,
    assign_wrong_p70b_relation_type,
    extract_p70a_relation_records,
    extract_p70b_relation_records,
    extract_p70a_negative_records,
    extract_p70b_negative_records,
    audit_p70a_negative_controls_and_collapse,
    audit_p70b_negative_controls_and_collapse,
    audit_structural_negative_records,
    audit_midpoint_target_availability,
    validate_source_contracts_deep_with_transitive_notes,
    run_p75_global_negative_controls_collapse_audit_probe,
)

from src.phase3.pure_numeric_relation_testbed import run_p70a_pure_numeric_relation_testbed_probe
from src.phase3.synthetic_time_series_relation_testbed import run_p70b_synthetic_time_series_relation_testbed_probe


def test_p75_01_constants_exact():
    assert PHASE == "P75"
    assert PHASE_GROUP == "PHASE_3"
    assert PHASE_NAME == "Global Negative Controls and Collapse Audit"
    assert CONTRACT_VERSION == "phase3_p75_global_negative_controls_collapse_audit_contract_v1"
    assert SOURCE_BASELINE_PHASE == "P69"
    assert SOURCE_VECTOR_TESTBED_PHASE == "P70A"
    assert SOURCE_TIME_SERIES_TESTBED_PHASE == "P70B"
    assert SOURCE_CONTRASTIVE_PHASE == "P71"
    assert SOURCE_OPERATOR_BANK_PHASE == "P72"
    assert SOURCE_TRANSFER_AUDIT_PHASE == "P73"
    assert SOURCE_COMPOSITION_AUDIT_PHASE == "P74"
    
    assert TRAINING_ALLOWED is False
    assert MODEL_IMPLEMENTATION_ALLOWED is False
    assert NEURAL_ENCODER_IMPLEMENTATION_ALLOWED is False
    assert NEURAL_OPERATOR_SELECTOR_ALLOWED is False
    assert OPTIMIZATION_ALLOWED is False
    assert TORCH_ALLOWED is False
    assert NUMPY_ALLOWED is False
    assert STOCHASTIC_RANDOM_ALLOWED is False
    assert BRIDGE_IMPLEMENTATION_ALLOWED is False
    assert LEARNED_METRIC_ALLOWED is False
    
    assert GLOBAL_NEGATIVE_CONTROL_AUDIT_ALLOWED is True
    assert COLLAPSE_AUDIT_ALLOWED is True
    assert ORACLE_OPERATOR_BANK_REUSE_ALLOWED is True
    assert ORACLE_RELATION_TYPE_SELECTION_ALLOWED_FOR_POSITIVE_CONTROL is True
    assert LEARNED_RELATION_TYPE_SELECTION_ALLOWED is False
    assert LEARNED_NEGATIVE_CONTROL_CLAIMS_ALLOWED is False
    
    assert TARGET_ENDPOINT_USED_FOR_PREDICTION is False
    assert TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY is True
    assert TARGET_MIDPOINT_USED_FOR_PREDICTION is False
    assert TARGET_MIDPOINT_USED_FOR_EVALUATION_ONLY is True
    
    assert PRIMARY_EMPIRICAL_TARGET == "global_negative_controls_and_collapse_resistance_diagnostics"
    assert VERDICT == "P75_READY_FOR_REVIEW"


def test_p75_02_probe_returns_serializable_verdict():
    probe = run_p75_global_negative_controls_collapse_audit_probe()
    assert isinstance(probe, dict)
    assert probe["verdict"] == "P75_READY_FOR_REVIEW"
    assert probe["source_contracts_deep_validated"] is True
    
    serialized = json.dumps(probe)
    assert len(serialized) > 0


def test_p75_03_probe_arrays():
    probe = run_p75_global_negative_controls_collapse_audit_probe()
    assert probe["audit_domains"] == AUDIT_DOMAINS
    assert probe["negative_control_families"] == NEGATIVE_CONTROL_FAMILIES
    assert probe["forbidden_claims"] == FORBIDDEN_CLAIMS
    assert probe["allowed_claims"] == ALLOWED_CLAIMS


def test_p75_04_source_validation_depth_addressed():
    res = validate_source_contracts_deep_with_transitive_notes()
    assert res["source_contracts_deep_validated"] is True
    assert res["p69_validated"] is True
    assert res["p70a_validated"] is True
    assert res["p70b_validated"] is True
    assert res["p71_validated"] is True
    assert res["p72_validated"] is True
    assert res["p73_validated"] is True
    assert res["p74_validated"] is True
    
    assert res["p74_source_validation_depth_note_addressed"] is True
    assert len(res["direct_validation_checks_performed"]) > 0
    assert len(res["transitive_validation_checks_relied_on"]) > 0
    assert isinstance(res["field_missing_but_transitively_guarded"], list)


def test_p75_05_p70a_negative_controls_and_collapse():
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    res = audit_p70a_negative_controls_and_collapse(p70a_probe)
    
    assert res["domain"] == "p70a_vector_world"
    assert res["positive_oracle_exact"] is True
    assert res["wrong_operator_unexpected_false_pass_count"] == 0
    assert res["no_op_false_pass_count"] == 0
    assert res["constant_output_unexpected_false_pass_count"] == 0
    
    assert res["wrong_operator_collision_or_ambiguity_count"] > 0
    assert res["known_ambiguity_reported"] is True
    assert res["unexpected_false_pass_count"] == 0
    assert res["negative_controls_diagnostic_pass"] is True


def test_p75_06_p70b_negative_controls_and_collapse():
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    res = audit_p70b_negative_controls_and_collapse(p70b_probe)
    
    assert res["domain"] == "p70b_time_series_parameter_world"
    assert res["positive_oracle_exact"] is True
    assert res["wrong_operator_false_pass_count"] == 0
    assert res["no_op_false_pass_count"] == 0
    assert res["constant_output_unexpected_false_pass_count"] == 0
    assert res["source_agnostic_false_pass_count"] == 15
    assert res["source_agnostic_unexpected_false_pass_count"] == 0
    
    assert res["unexpected_false_pass_count"] == 0
    assert res["negative_controls_diagnostic_pass"] is True


def test_p75_07_structural_negative_controls():
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    res = audit_structural_negative_records(p70a_probe, p70b_probe)
    
    assert res["label_permutation_record_count"] > 0
    assert res["mismatched_pair_record_count"] > 0
    assert res["all_label_permutation_true_labels_differ"] is True
    assert res["all_negative_controls_nondegenerate"] is True
    assert res["all_mismatched_pairs_expected_to_fail"] is True
    assert res["assigned_operator_application_possible"] is False
    assert res["failure_rate_measured"] is False
    assert res["failure_rate_structural_only"] is True
    assert res["structural_negative_controls_ready"] is True
    assert res["diagnostic_pass"] is True


def test_p75_08_midpoint_target_honesty():
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    res = audit_midpoint_target_availability(p70a_probe, p70b_probe)
    
    assert res["p74_target_midpoint_note_addressed"] is True
    assert res["p70a_midpoint_target_fields_available"] is True
    assert res["p70b_midpoint_target_fields_available"] is True
    assert res["midpoint_target_fields_available"] is True
    assert res["midpoint_target_evaluation_direct"] is True
    assert res["midpoint_target_evaluation_structural_only"] is False


def test_p75_09_no_forbidden_imports_in_source():
    p = pathlib.Path("src/phase3/global_negative_controls_collapse_audit.py")
    content = p.read_text(encoding="utf-8")
    
    assert "import " + "torch" not in content
    assert "from " + "torch" not in content
    assert "import " + "numpy" not in content
    assert "from " + "numpy" not in content
    assert "import " + "pandas" not in content
    assert "import " + "sklearn" not in content
    assert "import " + "random" not in content
    
    assert "torch." not in content
    assert "np." not in content


def test_p75_10_no_unsupported_phrases_in_source():
    p = pathlib.Path("src/phase3/global_negative_controls_collapse_audit.py")
    content = p.read_text(encoding="utf-8")
    
    assert "DataLoader" not in content
    assert "Dataset(" not in content
    assert "Optimizer" not in content
    assert "Schrodinger" not in content
    assert "Schrödinger" not in content
    assert "Bridge" not in content
    assert "nn.Module" not in content
    assert ".backward(" not in content
    assert ".step(" not in content
    assert "fit(" not in content
    assert "train(" not in content


def test_p75_11_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase3/global_negative_controls_collapse_audit.py",
        "tools/phase3/run_p75_global_negative_controls_collapse_audit_smoke.py",
        "tests/test_phase3_global_negative_controls_collapse_audit.py",
        "tests/test_phase3_p75_global_negative_controls_collapse_audit_smoke.py",
        "reports/PHASE_3_P75_GLOBAL_NEGATIVE_CONTROLS_AND_COLLAPSE_AUDIT_NO_TRAINING_NO_MODEL_NO_OPTIMIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase3/p75-global-negative-controls-collapse-audit-no-training-no-model-no-optimization",
        base_commit="5ae8cec9e53d66eda4e56b595c66803cb48528e7",
        allowed_files=allowed,
        phase_label="P75",
    )
