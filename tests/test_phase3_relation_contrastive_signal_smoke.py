# tests/test_phase3_relation_contrastive_signal_smoke.py

import inspect
import json
import math
import pathlib
import pytest

from src.phase3.relation_contrastive_signal_smoke import (
    PHASE,
    PHASE_GROUP,
    PHASE_NAME,
    CONTRACT_VERSION,
    SOURCE_BASELINE_PHASE,
    SOURCE_VECTOR_TESTBED_PHASE,
    SOURCE_TIME_SERIES_TESTBED_PHASE,
    TRAINING_ALLOWED,
    MODEL_IMPLEMENTATION_ALLOWED,
    NEURAL_ENCODER_IMPLEMENTATION_ALLOWED,
    OPTIMIZATION_ALLOWED,
    TORCH_ALLOWED,
    NUMPY_ALLOWED,
    STOCHASTIC_RANDOM_ALLOWED,
    BRIDGE_IMPLEMENTATION_ALLOWED,
    LEARNED_METRIC_ALLOWED,
    DESCRIPTOR_CONSTRUCTION_ALLOWED,
    CONTRASTIVE_EVALUATION_ALLOWED,
    RELATION_LABELS_ALLOWED_FOR_EVALUATION_ONLY,
    RELATION_LABELS_ALLOWED_FOR_DESCRIPTOR_CONSTRUCTION,
    PRIMARY_EMPIRICAL_TARGET,
    VERDICT,
    DESCRIPTOR_VIEWS,
    CONTROL_VIEWS,
    FORBIDDEN_CLAIMS,
    ALLOWED_CLAIMS,
    assert_finite_vector,
    assert_nonempty_records,
    safe_divide,
    encode_p70a_vector_delta_descriptor,
    encode_p70b_parameter_delta_descriptor,
    encode_p70b_series_summary_delta_descriptor,
    descriptor_l2_distance,
    descriptor_close,
    extract_p70a_train_style_records,
    extract_p70b_train_style_records,
    extract_p70b_negative_control_records,
    build_p70a_descriptor_records,
    build_p70b_parameter_descriptor_records,
    build_p70b_series_summary_descriptor_records,
    evaluate_contrastive_descriptor_view,
    evaluate_p70b_label_permutation_damage,
    evaluate_p70b_mismatched_pair_controls,
    validate_source_contracts,
    run_p71_relation_contrastive_signal_smoke_probe,
)

from src.phase3.pure_numeric_relation_testbed import run_p70a_pure_numeric_relation_testbed_probe
from src.phase3.synthetic_time_series_relation_testbed import run_p70b_synthetic_time_series_relation_testbed_probe


def test_p71_01_constants_exact():
    assert PHASE == "P71"
    assert PHASE_GROUP == "PHASE_3"
    assert PHASE_NAME == "Relation Descriptor Contrastive Signal Smoke"
    assert CONTRACT_VERSION == "phase3_p71_relation_descriptor_contrastive_signal_smoke_contract_v1"
    assert SOURCE_BASELINE_PHASE == "P69"
    assert SOURCE_VECTOR_TESTBED_PHASE == "P70A"
    assert SOURCE_TIME_SERIES_TESTBED_PHASE == "P70B"
    
    assert TRAINING_ALLOWED is False
    assert MODEL_IMPLEMENTATION_ALLOWED is False
    assert NEURAL_ENCODER_IMPLEMENTATION_ALLOWED is False
    assert OPTIMIZATION_ALLOWED is False
    assert TORCH_ALLOWED is False
    assert NUMPY_ALLOWED is False
    assert STOCHASTIC_RANDOM_ALLOWED is False
    assert BRIDGE_IMPLEMENTATION_ALLOWED is False
    assert LEARNED_METRIC_ALLOWED is False
    
    assert DESCRIPTOR_CONSTRUCTION_ALLOWED is True
    assert CONTRASTIVE_EVALUATION_ALLOWED is True
    assert RELATION_LABELS_ALLOWED_FOR_EVALUATION_ONLY is True
    assert RELATION_LABELS_ALLOWED_FOR_DESCRIPTOR_CONSTRUCTION is False
    
    assert PRIMARY_EMPIRICAL_TARGET == "relation_signal_availability_under_contrastive_diagnostics"
    assert VERDICT == "P71_READY_FOR_REVIEW"


def test_p71_02_probe_returns_serializable_verdict():
    probe = run_p71_relation_contrastive_signal_smoke_probe()
    assert isinstance(probe, dict)
    assert probe["verdict"] == "P71_READY_FOR_REVIEW"
    assert probe["source_contracts_validated"] is True
    
    serialized = json.dumps(probe)
    assert len(serialized) > 0


def test_p71_03_probe_arrays():
    probe = run_p71_relation_contrastive_signal_smoke_probe()
    assert probe["descriptor_views"] == DESCRIPTOR_VIEWS
    assert probe["control_views"] == CONTROL_VIEWS
    assert probe["forbidden_claims"] == FORBIDDEN_CLAIMS
    assert probe["allowed_claims"] == ALLOWED_CLAIMS


def test_p71_04_no_relation_type_argument_in_descriptors():
    # Verify that descriptor signatures do not accept relation_type or label parameters
    for func in [
        encode_p70a_vector_delta_descriptor,
        encode_p70b_parameter_delta_descriptor,
        encode_p70b_series_summary_delta_descriptor,
    ]:
        sig = inspect.signature(func)
        assert "relation_type" not in sig.parameters
        assert "label" not in sig.parameters


def test_p71_05_p70a_descriptor_exact():
    z_a = [1.0, 2.0, 3.0]
    z_b = [2.0, 2.0, 3.0] # delta is [1, 0, 0], L2 norm is 1
    
    desc = encode_p70a_vector_delta_descriptor(z_a, z_b)
    assert len(desc) == 9
    assert desc[:3] == [1.0, 0.0, 0.0]
    assert desc[3:6] == [1.0, 0.0, 0.0]
    assert desc[6:9] == [1.0, 0.0, 0.0]


def test_p71_06_p70b_param_descriptor_exact():
    params_a = {
        "amplitude": 1.0,
        "frequency": 1.0,
        "phase": 0.0,
        "volatility_envelope": 0.10,
        "trend": 0.0,
    }
    params_b = {
        "amplitude": 1.0,
        "frequency": 1.5,
        "phase": 0.0,
        "volatility_envelope": 0.10,
        "trend": 0.0,
    } # delta in frequency is 0.5, L2 norm is 0.5
    
    desc = encode_p70b_parameter_delta_descriptor(params_a, params_b)
    assert len(desc) == 15
    assert math.isclose(desc[1], 0.5) # frequency delta
    assert math.isclose(desc[6], 0.5) # frequency abs_delta
    assert math.isclose(desc[11], 1.0) # frequency direction


def test_p71_07_p70b_series_summary_descriptor_exact():
    series_a = [1.0] * 48
    series_b = [2.0] * 48
    
    desc = encode_p70b_series_summary_delta_descriptor(series_a, series_b)
    assert len(desc) == 18
    # mean_a = 1.0, mean_b = 2.0 -> mean_delta = 1.0
    # first_a = 1.0, first_b = 2.0 -> first_delta = 1.0
    # last_a = 1.0, last_b = 2.0 -> last_delta = 1.0
    assert math.isclose(desc[0], 1.0)
    assert math.isclose(desc[4], 1.0)
    assert math.isclose(desc[5], 1.0)


def test_p71_08_contrastive_metrics_evaluator():
    records = [
        {"record_id": "r1", "relation_type": "type1", "descriptor": [1.0, 0.0]},
        {"record_id": "r2", "relation_type": "type1", "descriptor": [1.1, 0.0]},
        {"record_id": "r3", "relation_type": "type2", "descriptor": [5.0, 5.0]},
    ]
    
    res = evaluate_contrastive_descriptor_view(records)
    assert res["record_count"] == 3
    assert res["relation_type_count"] == 2
    assert res["positive_pair_count"] == 1
    assert res["negative_pair_count"] == 2
    assert res["separation_margin"] > 0.0
    assert res["nearest_same_relation_top1_accuracy"] > 0.5
    assert res["different_relation_collision_count"] == 0


def test_p71_09_label_permutation_evaluation():
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    negatives = extract_p70b_negative_control_records(p70b_probe)
    
    res = evaluate_p70b_label_permutation_damage(negatives)
    assert res["label_permutation_record_count"] > 0
    assert res["all_true_labels_differ_from_assigned"] is True
    assert res["all_records_nondegenerate"] is True
    assert res["damage_expected"] is True


def test_p71_10_mismatched_pair_evaluation():
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    negatives = extract_p70b_negative_control_records(p70b_probe)
    
    res = evaluate_p70b_mismatched_pair_controls(negatives)
    assert res["mismatched_pair_record_count"] > 0
    assert res["all_records_nondegenerate"] is True
    assert res["all_expected_to_fail_relation_identity"] is True


def test_p71_11_contract_checks():
    val = validate_source_contracts()
    assert val["source_contracts_validated"] is True
    assert val["p69_validated"] is True
    assert val["p70a_validated"] is True
    assert val["p70b_validated"] is True


def test_p71_12_no_forbidden_imports_in_source():
    p = pathlib.Path("src/phase3/relation_contrastive_signal_smoke.py")
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


def test_p71_13_no_unsupported_phrases_in_source():
    p = pathlib.Path("src/phase3/relation_contrastive_signal_smoke.py")
    content = p.read_text(encoding="utf-8")
    
    # Check for forbidden mechanics
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


def test_p71_14_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase3/relation_contrastive_signal_smoke.py",
        "tools/phase3/run_p71_relation_contrastive_signal_smoke.py",
        "tests/test_phase3_relation_contrastive_signal_smoke.py",
        "tests/test_phase3_p71_relation_contrastive_signal_smoke.py",
        "reports/PHASE_3_P71_RELATION_DESCRIPTOR_CONTRASTIVE_SIGNAL_SMOKE_NO_TRAINING_NO_MODEL_NO_OPTIMIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase3/p71-relation-descriptor-contrastive-signal-smoke-no-training-no-model-no-optimization",
        base_commit="8e96be8de31075a3d8815d6c7e635b6b9361a9ef",
        allowed_files=allowed,
        phase_label="P71",
    )
