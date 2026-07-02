# tests/test_phase3_oracle_sparse_operator_bank_mvp.py

import inspect
import json
import math
import pathlib
import pytest

from src.phase3.oracle_sparse_operator_bank_mvp import (
    PHASE,
    PHASE_GROUP,
    PHASE_NAME,
    CONTRACT_VERSION,
    SOURCE_BASELINE_PHASE,
    SOURCE_VECTOR_TESTBED_PHASE,
    SOURCE_TIME_SERIES_TESTBED_PHASE,
    SOURCE_CONTRASTIVE_PHASE,
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
    ORACLE_OPERATOR_BANK_ALLOWED,
    ORACLE_RELATION_TYPE_SELECTION_ALLOWED,
    LEARNED_RELATION_TYPE_SELECTION_ALLOWED,
    SPARSE_OPERATOR_APPLICATION_ALLOWED,
    OPERATOR_BANK_TRAINING_ALLOWED,
    PRIMARY_EMPIRICAL_TARGET,
    VERDICT,
    OPERATOR_DOMAINS,
    P70A_VECTOR_OPERATOR_TYPES,
    P70B_TIME_SERIES_OPERATOR_TYPES,
    EVALUATION_SPLITS,
    FORBIDDEN_CLAIMS,
    ALLOWED_CLAIMS,
    assert_finite_vector,
    assert_parameter_state,
    safe_divide,
    vector_l2_error,
    parameter_l2_error,
    parameter_max_abs_error,
    build_p70a_vector_operator_specs,
    build_p70b_time_series_operator_specs,
    build_oracle_sparse_operator_bank,
    apply_p70a_sparse_vector_operator,
    apply_p70b_sparse_parameter_operator,
    p70a_no_relation_baseline,
    p70b_no_relation_parameter_baseline,
    build_p70a_mean_delta_by_relation,
    apply_p70a_mean_delta_baseline,
    build_p70b_mean_parameter_delta_by_relation,
    apply_p70b_mean_parameter_delta_baseline,
    extract_p70a_operator_evaluation_records,
    extract_p70b_operator_evaluation_records,
    summarize_error_values,
    evaluate_p70a_sparse_operator_bank,
    evaluate_p70b_sparse_operator_bank,
    validate_source_contracts_deep,
    run_p72_oracle_sparse_operator_bank_mvp_probe,
)

from src.phase3.pure_numeric_relation_testbed import run_p70a_pure_numeric_relation_testbed_probe
from src.phase3.synthetic_time_series_relation_testbed import run_p70b_synthetic_time_series_relation_testbed_probe


def test_p72_01_constants_exact():
    assert PHASE == "P72"
    assert PHASE_GROUP == "PHASE_3"
    assert PHASE_NAME == "Oracle Sparse Operator Bank MVP"
    assert CONTRACT_VERSION == "phase3_p72_oracle_sparse_operator_bank_mvp_contract_v1"
    assert SOURCE_BASELINE_PHASE == "P69"
    assert SOURCE_VECTOR_TESTBED_PHASE == "P70A"
    assert SOURCE_TIME_SERIES_TESTBED_PHASE == "P70B"
    assert SOURCE_CONTRASTIVE_PHASE == "P71"
    
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
    
    assert ORACLE_OPERATOR_BANK_ALLOWED is True
    assert ORACLE_RELATION_TYPE_SELECTION_ALLOWED is True
    assert LEARNED_RELATION_TYPE_SELECTION_ALLOWED is False
    assert SPARSE_OPERATOR_APPLICATION_ALLOWED is True
    assert OPERATOR_BANK_TRAINING_ALLOWED is False
    
    assert PRIMARY_EMPIRICAL_TARGET == "oracle_sparse_operator_application_and_invariant_preservation"
    assert VERDICT == "P72_READY_FOR_REVIEW"


def test_p72_02_probe_returns_serializable_verdict():
    probe = run_p72_oracle_sparse_operator_bank_mvp_probe()
    assert isinstance(probe, dict)
    assert probe["verdict"] == "P72_READY_FOR_REVIEW"
    assert probe["source_contracts_deep_validated"] is True
    
    serialized = json.dumps(probe)
    assert len(serialized) > 0


def test_p72_03_probe_arrays():
    probe = run_p72_oracle_sparse_operator_bank_mvp_probe()
    assert probe["operator_domains"] == OPERATOR_DOMAINS
    assert probe["p70a_vector_operator_types"] == P70A_VECTOR_OPERATOR_TYPES
    assert probe["p70b_time_series_operator_types"] == P70B_TIME_SERIES_OPERATOR_TYPES
    assert probe["evaluation_splits"] == EVALUATION_SPLITS
    assert probe["forbidden_claims"] == FORBIDDEN_CLAIMS
    assert probe["allowed_claims"] == ALLOWED_CLAIMS


def test_p72_04_deep_source_validation_checked():
    res = validate_source_contracts_deep()
    assert res["source_contracts_deep_validated"] is True
    assert res["p69_validated"] is True
    assert res["p70a_validated"] is True
    assert res["p70b_validated"] is True
    assert res["p71_validated"] is True
    assert res["p71_validation_depth_note_addressed"] is True


def test_p72_05_specs_fields():
    bank = build_oracle_sparse_operator_bank()
    assert bank["bank_kind"] == "oracle_sparse_operator_bank_mvp"
    assert bank["learned"] is False
    assert bank["oracle_relation_type_selection_allowed"] is True
    assert bank["learned_relation_type_selection_allowed"] is False
    
    p70a_specs = bank["p70a_vector_operator_specs"]
    for t in P70A_VECTOR_OPERATOR_TYPES:
        spec = p70a_specs[t]
        assert spec["operator_type"] == t
        assert spec["domain"] == "p70a_vector_world"
        assert spec["state_dim"] == 3
        assert spec["sparse"] is True
        assert spec["oracle_selected"] is True
        assert spec["learned"] is False
        
    p70b_specs = bank["p70b_time_series_operator_specs"]
    for t in P70B_TIME_SERIES_OPERATOR_TYPES:
        spec = p70b_specs[t]
        assert spec["operator_type"] == t
        assert spec["domain"] == "p70b_time_series_parameter_world"
        assert spec["sparse"] is True
        assert spec["oracle_selected"] is True
        assert spec["learned"] is False


def test_p72_06_p70a_operator_reconstruction_exact():
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    recs = extract_p70a_operator_evaluation_records(p70a_probe)
    assert len(recs) > 0
    
    for r in recs:
        z_a = r["z_a"]
        z_b = r["z_b"]
        r_type = r["relation_type"]
        intens = r["intensity"]
        
        z_pred = apply_p70a_sparse_vector_operator(z_a, r_type, intens)
        assert vector_l2_error(z_pred, z_b) < 1e-12


def test_p72_07_p70b_operator_reconstruction_exact():
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    recs = extract_p70b_operator_evaluation_records(p70b_probe)
    assert len(recs) > 0
    
    for r in recs:
        params_a = r["params_a"]
        params_b = r["params_b"]
        r_type = r["relation_type"]
        intens = r["intensity"]
        
        params_pred = apply_p70b_sparse_parameter_operator(params_a, r_type, intens)
        assert parameter_l2_error(params_pred, params_b) < 1e-12


def test_p72_08_p70a_evaluation_summary():
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    res = evaluate_p70a_sparse_operator_bank(p70a_probe)
    
    assert res["domain"] == "p70a_vector_world"
    assert res["oracle_endpoint_l2_error"]["mean"] < 1e-12
    assert res["oracle_invariant_violation"]["max"] < 1e-12
    assert res["oracle_beats_no_relation_mean_l2"] is True
    assert res["diagnostic_pass"] is True
    assert res["heldout_base_record_count"] > 0
    assert res["heldout_magnitude_record_count"] > 0


def test_p72_09_p70b_evaluation_summary():
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    res = evaluate_p70b_sparse_operator_bank(p70b_probe)
    
    assert res["domain"] == "p70b_time_series_parameter_world"
    assert res["oracle_parameter_l2_error"]["mean"] < 1e-12
    assert res["oracle_series_l2_error"]["mean"] < 1e-12
    assert res["oracle_invariant_violation"]["max"] < 1e-12
    assert res["oracle_beats_no_relation_mean_l2"] is True
    assert res["diagnostic_pass"] is True
    assert res["heldout_base_record_count"] > 0
    assert res["heldout_magnitude_record_count"] > 0


def test_p72_10_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        assert_finite_vector("v", [1.0, float("inf")])
    with pytest.raises(ValueError):
        assert_parameter_state("p", {"amplitude": -1.0, "frequency": 1.0, "phase": 0.0, "volatility_envelope": 0.0, "trend": 0.0})


def test_p72_11_no_forbidden_imports_in_source():
    p = pathlib.Path("src/phase3/oracle_sparse_operator_bank_mvp.py")
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


def test_p72_12_no_unsupported_phrases_in_source():
    p = pathlib.Path("src/phase3/oracle_sparse_operator_bank_mvp.py")
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


def test_p72_13_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase3/oracle_sparse_operator_bank_mvp.py",
        "tools/phase3/run_p72_oracle_sparse_operator_bank_mvp_smoke.py",
        "tests/test_phase3_oracle_sparse_operator_bank_mvp.py",
        "tests/test_phase3_p72_oracle_sparse_operator_bank_mvp_smoke.py",
        "reports/PHASE_3_P72_ORACLE_SPARSE_OPERATOR_BANK_MVP_NO_TRAINING_NO_MODEL_NO_OPTIMIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase3/p72-oracle-sparse-operator-bank-mvp-no-training-no-model-no-optimization",
        base_commit="bd5efa86ebb64894536db0015cedb00f58687e4c",
        allowed_files=allowed,
        phase_label="P72",
    )
