# tests/test_phase3_composition_order_sensitivity_audit.py

import inspect
import json
import math
import pathlib
import pytest

from src.phase3.composition_order_sensitivity_audit import (
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
    ORACLE_COMPOSITION_AUDIT_ALLOWED,
    ORACLE_OPERATOR_BANK_REUSE_ALLOWED,
    ORACLE_RELATION_TYPE_SELECTION_ALLOWED,
    LEARNED_RELATION_TYPE_SELECTION_ALLOWED,
    LEARNED_COMPOSITION_CLAIMS_ALLOWED,
    TARGET_ENDPOINT_USED_FOR_PREDICTION,
    TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY,
    TARGET_MIDPOINT_USED_FOR_PREDICTION,
    TARGET_MIDPOINT_USED_FOR_EVALUATION_ONLY,
    PRIMARY_EMPIRICAL_TARGET,
    VERDICT,
    AUDIT_DOMAINS,
    COMPOSITION_FAMILIES,
    FORBIDDEN_CLAIMS,
    ALLOWED_CLAIMS,
    safe_divide,
    summarize_values,
    summary_is_zero,
    vectors_max_abs_error,
    count_by_key,
    build_p70a_composition_cases,
    build_p70b_composition_cases,
    apply_p70a_forward_composition,
    apply_p70a_reverse_composition,
    apply_p70b_forward_composition,
    apply_p70b_reverse_composition,
    compute_p70a_composite_invariant_indices,
    compute_p70b_composite_invariant_keys,
    audit_p70a_composition_and_order_sensitivity,
    audit_p70b_composition_and_order_sensitivity,
    validate_source_contracts_with_fidelity,
    run_p74_composition_order_sensitivity_audit_probe,
)

from src.phase3.pure_numeric_relation_testbed import run_p70a_pure_numeric_relation_testbed_probe
from src.phase3.synthetic_time_series_relation_testbed import run_p70b_synthetic_time_series_relation_testbed_probe


def test_p74_01_constants_exact():
    assert PHASE == "P74"
    assert PHASE_GROUP == "PHASE_3"
    assert PHASE_NAME == "Composition and Order-Sensitivity Audit"
    assert CONTRACT_VERSION == "phase3_p74_composition_order_sensitivity_audit_contract_v1"
    assert SOURCE_BASELINE_PHASE == "P69"
    assert SOURCE_VECTOR_TESTBED_PHASE == "P70A"
    assert SOURCE_TIME_SERIES_TESTBED_PHASE == "P70B"
    assert SOURCE_CONTRASTIVE_PHASE == "P71"
    assert SOURCE_OPERATOR_BANK_PHASE == "P72"
    assert SOURCE_TRANSFER_AUDIT_PHASE == "P73"
    
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
    
    assert ORACLE_COMPOSITION_AUDIT_ALLOWED is True
    assert ORACLE_OPERATOR_BANK_REUSE_ALLOWED is True
    assert ORACLE_RELATION_TYPE_SELECTION_ALLOWED is True
    assert LEARNED_RELATION_TYPE_SELECTION_ALLOWED is False
    assert LEARNED_COMPOSITION_CLAIMS_ALLOWED is False
    
    assert TARGET_ENDPOINT_USED_FOR_PREDICTION is False
    assert TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY is True
    assert TARGET_MIDPOINT_USED_FOR_PREDICTION is False
    assert TARGET_MIDPOINT_USED_FOR_EVALUATION_ONLY is True
    
    assert PRIMARY_EMPIRICAL_TARGET == "oracle_composition_and_order_sensitivity_diagnostics"
    assert VERDICT == "P74_READY_FOR_REVIEW"


def test_p74_02_probe_returns_serializable_verdict():
    probe = run_p74_composition_order_sensitivity_audit_probe()
    assert isinstance(probe, dict)
    assert probe["verdict"] == "P74_READY_FOR_REVIEW"
    assert probe["source_contracts_fidelity_validated"] is True
    
    serialized = json.dumps(probe)
    assert len(serialized) > 0


def test_p74_03_probe_arrays():
    probe = run_p74_composition_order_sensitivity_audit_probe()
    assert probe["audit_domains"] == AUDIT_DOMAINS
    assert probe["composition_families"] == COMPOSITION_FAMILIES
    assert probe["forbidden_claims"] == FORBIDDEN_CLAIMS
    assert probe["allowed_claims"] == ALLOWED_CLAIMS


def test_p74_04_validation_fidelity_note_addressed():
    res = validate_source_contracts_with_fidelity()
    assert res["source_contracts_fidelity_validated"] is True
    assert res["p69_validated"] is True
    assert res["p70a_validated"] is True
    assert res["p70b_validated"] is True
    assert res["p71_validated"] is True
    assert res["p72_validated"] is True
    assert res["p73_validated"] is True
    assert res["p73_validation_fidelity_note_addressed"] is True
    
    # Test that missing optional fields list is returned
    assert isinstance(res["field_missing_but_transitively_guarded"], list)


def test_p74_05_p70a_composition_audit():
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    res = audit_p70a_composition_and_order_sensitivity(p70a_probe)
    
    assert res["domain"] == "p70a_vector_world"
    assert res["order_sensitive_expected_count"] == 6
    assert res["order_sensitive_detected_count"] == 6
    assert res["commutative_expected_count"] == 8
    assert res["commutative_detected_count"] == 8
    assert res["expected_order_match_count"] == 14
    assert res["expected_order_mismatch_count"] == 0
    
    assert res["all_composite_invariants_preserved"] is True
    assert res["order_sensitivity_diagnostic_pass"] is True
    assert res["target_endpoint_used_for_prediction"] is False
    assert res["target_midpoint_used_for_prediction"] is False
    assert res["diagnostic_pass"] is True


def test_p74_06_p70b_composition_audit():
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    res = audit_p70b_composition_and_order_sensitivity(p70b_probe)
    
    assert res["domain"] == "p70b_time_series_parameter_world"
    # All 6 generated and 6 source cases are commutative since they change different disjoint parameter keys
    assert res["order_sensitive_detected_count"] == 0
    assert res["expected_order_mismatch_count"] == 0
    
    assert res["all_composite_invariants_preserved"] is True
    assert res["order_sensitivity_diagnostic_pass"] is True
    assert res["target_endpoint_used_for_prediction"] is False
    assert res["target_midpoint_used_for_prediction"] is False
    assert res["diagnostic_pass"] is True
    
    # Targets must match forward end prediction when available
    assert res["all_forward_targets_exact_when_available"] is True


def test_p74_07_no_forbidden_imports_in_source():
    p = pathlib.Path("src/phase3/composition_order_sensitivity_audit.py")
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


def test_p74_08_no_unsupported_phrases_in_source():
    p = pathlib.Path("src/phase3/composition_order_sensitivity_audit.py")
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


def test_p74_09_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase3/composition_order_sensitivity_audit.py",
        "tools/phase3/run_p74_composition_order_sensitivity_audit_smoke.py",
        "tests/test_phase3_composition_order_sensitivity_audit.py",
        "tests/test_phase3_p74_composition_order_sensitivity_audit_smoke.py",
        "reports/PHASE_3_P74_COMPOSITION_AND_ORDER_SENSITIVITY_AUDIT_NO_TRAINING_NO_MODEL_NO_OPTIMIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase3/p74-composition-order-sensitivity-audit-no-training-no-model-no-optimization",
        base_commit="9c2a6196b7cb973b415a838b0aa5739e3c75fcc8",
        allowed_files=allowed,
        phase_label="P74",
    )
