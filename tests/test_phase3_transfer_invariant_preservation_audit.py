# tests/test_phase3_transfer_invariant_preservation_audit.py

import inspect
import json
import math
import pathlib
import pytest

from src.phase3.transfer_invariant_preservation_audit import (
    PHASE,
    PHASE_GROUP,
    PHASE_NAME,
    CONTRACT_VERSION,
    SOURCE_BASELINE_PHASE,
    SOURCE_VECTOR_TESTBED_PHASE,
    SOURCE_TIME_SERIES_TESTBED_PHASE,
    SOURCE_CONTRASTIVE_PHASE,
    SOURCE_OPERATOR_BANK_PHASE,
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
    ORACLE_TRANSFER_AUDIT_ALLOWED,
    ORACLE_OPERATOR_BANK_REUSE_ALLOWED,
    ORACLE_RELATION_TYPE_SELECTION_ALLOWED,
    LEARNED_RELATION_TYPE_SELECTION_ALLOWED,
    LEARNED_TRANSFER_CLAIMS_ALLOWED,
    TARGET_ENDPOINT_USED_FOR_PREDICTION,
    TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY,
    PRIMARY_EMPIRICAL_TARGET,
    VERDICT,
    AUDIT_DOMAINS,
    AUDIT_SPLITS,
    NEGATIVE_CONTROL_MODES,
    FORBIDDEN_CLAIMS,
    ALLOWED_CLAIMS,
    safe_divide,
    summarize_values,
    summary_is_zero,
    count_by_split,
    extract_p70a_audit_records,
    extract_p70b_audit_records,
    extract_p70b_negative_control_records,
    audit_p70a_oracle_transfer_and_invariants,
    audit_p70b_oracle_transfer_and_invariants,
    audit_p70b_negative_control_failures,
    validate_source_contracts_complete,
    run_p73_transfer_invariant_preservation_audit_probe,
)

from src.phase3.pure_numeric_relation_testbed import run_p70a_pure_numeric_relation_testbed_probe
from src.phase3.synthetic_time_series_relation_testbed import run_p70b_synthetic_time_series_relation_testbed_probe


def test_p73_01_constants_exact():
    assert PHASE == "P73"
    assert PHASE_GROUP == "PHASE_3"
    assert PHASE_NAME == "Transfer and Invariant Preservation Audit"
    assert CONTRACT_VERSION == "phase3_p73_transfer_invariant_preservation_audit_contract_v1"
    assert SOURCE_BASELINE_PHASE == "P69"
    assert SOURCE_VECTOR_TESTBED_PHASE == "P70A"
    assert SOURCE_TIME_SERIES_TESTBED_PHASE == "P70B"
    assert SOURCE_CONTRASTIVE_PHASE == "P71"
    assert SOURCE_OPERATOR_BANK_PHASE == "P72"
    
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
    
    assert ORACLE_TRANSFER_AUDIT_ALLOWED is True
    assert ORACLE_OPERATOR_BANK_REUSE_ALLOWED is True
    assert ORACLE_RELATION_TYPE_SELECTION_ALLOWED is True
    assert LEARNED_RELATION_TYPE_SELECTION_ALLOWED is False
    assert LEARNED_TRANSFER_CLAIMS_ALLOWED is False
    
    assert TARGET_ENDPOINT_USED_FOR_PREDICTION is False
    assert TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY is True
    
    assert PRIMARY_EMPIRICAL_TARGET == "oracle_transfer_and_invariant_preservation_diagnostics"
    assert VERDICT == "P73_READY_FOR_REVIEW"


def test_p73_02_probe_returns_serializable_verdict():
    probe = run_p73_transfer_invariant_preservation_audit_probe()
    assert isinstance(probe, dict)
    assert probe["verdict"] == "P73_READY_FOR_REVIEW"
    assert probe["source_contracts_complete_validated"] is True
    
    serialized = json.dumps(probe)
    assert len(serialized) > 0


def test_p73_03_probe_arrays():
    probe = run_p73_transfer_invariant_preservation_audit_probe()
    assert probe["audit_domains"] == AUDIT_DOMAINS
    assert probe["audit_splits"] == AUDIT_SPLITS
    assert probe["negative_control_modes"] == NEGATIVE_CONTROL_MODES
    assert probe["forbidden_claims"] == FORBIDDEN_CLAIMS
    assert probe["allowed_claims"] == ALLOWED_CLAIMS


def test_p73_04_deep_source_validation_completeness_addressed():
    res = validate_source_contracts_complete()
    assert res["source_contracts_complete_validated"] is True
    assert res["p69_validated"] is True
    assert res["p70a_validated"] is True
    assert res["p70b_validated"] is True
    assert res["p71_validated"] is True
    assert res["p72_validated"] is True
    assert res["p72_validation_completeness_note_addressed"] is True
    
    # Verify we list fields missing but transitively guarded (e.g. from P70A/P70B/P69 optional fields)
    assert isinstance(res["field_missing_but_transitively_guarded"], list)


def test_p73_05_p70a_transfer_audit():
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    res = audit_p70a_oracle_transfer_and_invariants(p70a_probe)
    
    assert res["domain"] == "p70a_vector_world"
    assert res["all_splits_exact"] is True
    assert res["all_invariants_preserved"] is True
    assert res["heldout_base_exact"] is True
    assert res["heldout_magnitude_exact"] is True
    assert res["target_endpoint_used_for_prediction"] is False
    assert res["target_endpoint_used_for_evaluation_only"] is True
    assert res["diagnostic_pass"] is True
    
    # In P72 exact operator evaluations, transfer gaps must be exactly zero
    assert abs(res["transfer_gap_heldout_base_vs_train_mean_l2"]) < 1e-12
    assert abs(res["transfer_gap_heldout_magnitude_vs_train_mean_l2"]) < 1e-12


def test_p73_06_p70b_transfer_audit():
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    res = audit_p70b_oracle_transfer_and_invariants(p70b_probe)
    
    assert res["domain"] == "p70b_time_series_parameter_world"
    assert res["all_splits_exact"] is True
    assert res["all_invariants_preserved"] is True
    assert res["heldout_base_exact"] is True
    assert res["heldout_magnitude_exact"] is True
    assert res["target_endpoint_used_for_prediction"] is False
    assert res["target_endpoint_used_for_evaluation_only"] is True
    assert res["diagnostic_pass"] is True
    
    # Transfer gaps should be zero
    assert abs(res["transfer_gap_heldout_base_vs_train_parameter_mean_l2"]) < 1e-12
    assert abs(res["transfer_gap_heldout_magnitude_vs_train_parameter_mean_l2"]) < 1e-12
    assert abs(res["transfer_gap_heldout_base_vs_train_series_mean_l2"]) < 1e-12
    assert abs(res["transfer_gap_heldout_magnitude_vs_train_series_mean_l2"]) < 1e-12


def test_p73_07_negative_control_audit():
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    res = audit_p70b_negative_control_failures(p70b_probe)
    
    assert res["negative_control_record_count"] > 0
    assert res["label_permutation_record_count"] > 0
    assert res["mismatched_pair_record_count"] > 0
    assert res["structural_negative_controls_ready"] is True
    assert res["all_label_permutation_true_labels_differ"] is True
    assert res["all_negative_controls_nondegenerate"] is True
    assert res["all_mismatched_pairs_expected_to_fail"] is True
    assert res["diagnostic_pass"] is True


def test_p73_08_no_forbidden_imports_in_source():
    p = pathlib.Path("src/phase3/transfer_invariant_preservation_audit.py")
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


def test_p73_09_no_unsupported_phrases_in_source():
    p = pathlib.Path("src/phase3/transfer_invariant_preservation_audit.py")
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


def test_p73_10_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase3/transfer_invariant_preservation_audit.py",
        "tools/phase3/run_p73_transfer_invariant_preservation_audit_smoke.py",
        "tests/test_phase3_transfer_invariant_preservation_audit.py",
        "tests/test_phase3_p73_transfer_invariant_preservation_audit_smoke.py",
        "reports/PHASE_3_P73_TRANSFER_AND_INVARIANT_PRESERVATION_AUDIT_NO_TRAINING_NO_MODEL_NO_OPTIMIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase3/p73-transfer-invariant-preservation-audit-no-training-no-model-no-optimization",
        base_commit="f756ed33b3b4947251e2ce42469b6129f70220a7",
        allowed_files=allowed,
        phase_label="P73",
    )
