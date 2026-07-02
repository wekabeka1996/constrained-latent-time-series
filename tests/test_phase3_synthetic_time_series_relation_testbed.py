# tests/test_phase3_synthetic_time_series_relation_testbed.py

import json
import math
import pathlib
import pytest

from src.phase3.synthetic_time_series_relation_testbed import (
    PHASE,
    PHASE_GROUP,
    PHASE_NAME,
    CONTRACT_VERSION,
    SOURCE_TESTBED_PHASE,
    SOURCE_BASELINE_PHASE,
    SERIES_LENGTH,
    PARAMETER_KEYS,
    TRAINING_ALLOWED,
    MODEL_IMPLEMENTATION_ALLOWED,
    OPTIMIZATION_ALLOWED,
    TORCH_ALLOWED,
    NUMPY_ALLOWED,
    STOCHASTIC_RANDOM_ALLOWED,
    BRIDGE_IMPLEMENTATION_ALLOWED,
    TESTBED_RECORD_CONSTRUCTION_ALLOWED,
    TRAINING_DATASET_GENERATION_ALLOWED,
    PRIMARY_EMPIRICAL_TARGET,
    VERDICT,
    RELATION_TYPES,
    SPLIT_NAMES,
    FUTURE_USE_CASES,
    FORBIDDEN_CLAIMS,
    ALLOWED_CLAIMS,
    assert_finite_scalar,
    assert_finite_series,
    assert_parameter_state,
    parameter_close,
    parameter_l1_distance,
    generate_synthetic_series,
    apply_change_frequency,
    apply_scale_amplitude,
    apply_shift_phase,
    apply_scale_volatility_envelope,
    apply_shift_trend,
    build_time_series_relation_specs,
    build_time_series_base_parameter_states,
    build_time_series_intensity_values,
    apply_time_series_relation,
    compute_parameter_invariant_violation,
    series_l1_distance,
    series_l2_distance,
    series_max_abs_distance,
    summarize_series,
    build_time_series_relation_record,
    is_nondegenerate_transition,
    has_endpoint_collision,
    build_p70b_synthetic_time_series_relation_testbed,
    run_p70b_synthetic_time_series_relation_testbed_probe,
)


def test_p70b_01_constants_exact():
    assert PHASE == "P70B"
    assert PHASE_GROUP == "PHASE_3"
    assert PHASE_NAME == "Synthetic Time-Series Relation Testbed"
    assert CONTRACT_VERSION == "phase3_p70b_synthetic_time_series_relation_testbed_contract_v1"
    assert SOURCE_TESTBED_PHASE == "P70A"
    assert SOURCE_BASELINE_PHASE == "P69"
    assert SERIES_LENGTH == 48
    assert PARAMETER_KEYS == ["amplitude", "frequency", "phase", "volatility_envelope", "trend"]
    
    assert TRAINING_ALLOWED is False
    assert MODEL_IMPLEMENTATION_ALLOWED is False
    assert OPTIMIZATION_ALLOWED is False
    assert TORCH_ALLOWED is False
    assert NUMPY_ALLOWED is False
    assert STOCHASTIC_RANDOM_ALLOWED is False
    assert BRIDGE_IMPLEMENTATION_ALLOWED is False
    
    assert TESTBED_RECORD_CONSTRUCTION_ALLOWED is True
    assert TRAINING_DATASET_GENERATION_ALLOWED is False
    
    assert PRIMARY_EMPIRICAL_TARGET == "synthetic_time_series_repeated_relation_instances_for_future_operator_tests"
    assert VERDICT == "P70B_READY_FOR_REVIEW"


def test_p70b_02_probe_returns_serializable_verdict():
    probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    assert isinstance(probe, dict)
    assert probe["verdict"] == "P70B_READY_FOR_REVIEW"
    assert probe["p70a_contract_validated"] is True
    
    serialized = json.dumps(probe)
    assert len(serialized) > 0


def test_p70b_03_probe_arrays():
    probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    assert probe["relation_types"] == RELATION_TYPES
    assert probe["split_names"] == SPLIT_NAMES
    assert probe["future_use_cases"] == FUTURE_USE_CASES
    assert probe["forbidden_claims"] == FORBIDDEN_CLAIMS
    assert probe["allowed_claims"] == ALLOWED_CLAIMS


def test_p70b_04_generate_synthetic_series():
    params = {
        "amplitude": 1.0,
        "frequency": 1.0,
        "phase": 0.0,
        "volatility_envelope": 0.10,
        "trend": 0.0,
    }
    
    series = generate_synthetic_series(params, series_length=48)
    assert len(series) == 48
    assert all(math.isfinite(x) for x in series)
    
    # Assert first element: t = 0
    # carrier = amplitude * sin(phase) = 1.0 * sin(0) = 0.0
    # value = envelope * carrier + trend * (0 - 0.5) = 0 + 0 * -0.5 = 0.0
    assert math.isclose(series[0], 0.0)


def test_p70b_05_apply_change_frequency():
    params = {
        "amplitude": 1.0,
        "frequency": 1.0,
        "phase": 0.0,
        "volatility_envelope": 0.10,
        "trend": 0.0,
    }
    
    res = apply_change_frequency(params, 0.5)
    assert res["frequency"] == 1.5
    assert all(res[k] == params[k] for k in PARAMETER_KEYS if k != "frequency")


def test_p70b_06_apply_scale_amplitude():
    params = {
        "amplitude": 1.0,
        "frequency": 1.0,
        "phase": 0.0,
        "volatility_envelope": 0.10,
        "trend": 0.0,
    }
    
    res = apply_scale_amplitude(params, 0.5)
    assert res["amplitude"] == 1.5
    assert all(res[k] == params[k] for k in PARAMETER_KEYS if k != "amplitude")


def test_p70b_07_apply_shift_phase():
    params = {
        "amplitude": 1.0,
        "frequency": 1.0,
        "phase": 0.0,
        "volatility_envelope": 0.10,
        "trend": 0.0,
    }
    
    res = apply_shift_phase(params, 0.5)
    assert res["phase"] == 0.5
    assert all(res[k] == params[k] for k in PARAMETER_KEYS if k != "phase")


def test_p70b_08_apply_scale_volatility_envelope():
    params = {
        "amplitude": 1.0,
        "frequency": 1.0,
        "phase": 0.0,
        "volatility_envelope": 0.10,
        "trend": 0.0,
    }
    
    res = apply_scale_volatility_envelope(params, 0.5)
    assert math.isclose(res["volatility_envelope"], 0.15)
    assert all(res[k] == params[k] for k in PARAMETER_KEYS if k != "volatility_envelope")


def test_p70b_09_apply_shift_trend():
    params = {
        "amplitude": 1.0,
        "frequency": 1.0,
        "phase": 0.0,
        "volatility_envelope": 0.10,
        "trend": 0.0,
    }
    
    res = apply_shift_trend(params, 0.5)
    assert res["trend"] == 0.5
    assert all(res[k] == params[k] for k in PARAMETER_KEYS if k != "trend")


def test_p70b_10_invariant_violation_always_zero():
    testbed = build_p70b_synthetic_time_series_relation_testbed()
    records = testbed["relation_records"]
    assert len(records) > 0
    for rec in records:
        assert rec["parameter_invariant_violation"] == 0.0


def test_p70b_11_repeated_instances():
    testbed = build_p70b_synthetic_time_series_relation_testbed()
    summary = testbed["repeated_relation_instance_summary"]
    for rel_type in RELATION_TYPES:
        assert summary[rel_type] >= 2


def test_p70b_12_splits_exist():
    testbed = build_p70b_synthetic_time_series_relation_testbed()
    records = testbed["relation_records"]
    
    heldout_bases = [r for r in records if r["split"] == "heldout_base_state"]
    heldout_mags = [r for r in records if r["split"] == "heldout_magnitude"]
    
    assert len(heldout_bases) > 0
    assert len(heldout_mags) > 0


def test_p70b_13_composition_records():
    testbed = build_p70b_synthetic_time_series_relation_testbed()
    comps = testbed["composition_records"]
    
    assert len(comps) == 6
    for c in comps:
        assert len(c["series_start"]) == SERIES_LENGTH
        assert len(c["series_mid"]) == SERIES_LENGTH
        assert len(c["series_end"]) == SERIES_LENGTH


def test_p70b_14_negative_control_records_nondegenerate():
    testbed = build_p70b_synthetic_time_series_relation_testbed()
    neg_controls = testbed["negative_control_records"]
    
    perms = [n for n in neg_controls if n["negative_control_type"] == "label_permutation"]
    mismatches = [n for n in neg_controls if n["negative_control_type"] == "mismatched_pair"]
    
    assert len(perms) > 0
    for p in perms:
        assert p["true_relation_type"] != p["assigned_relation_type"]
        assert p["expected_to_fail_relation_identity"] is True
        assert p["nondegenerate"] is True
        
    assert len(mismatches) > 0
    for m in mismatches:
        assert m["expected_to_fail_relation_identity"] is True
        assert m["nondegenerate"] is True
        
    assert testbed["sanity_summary"]["negative_control_endpoint_collision_count"] == 0


def test_p70b_15_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        assert_finite_scalar("s", float("nan"))
    with pytest.raises(ValueError):
        assert_finite_series("v", [])
    with pytest.raises(ValueError):
        assert_finite_series("v", [1.0, float("nan")])
    with pytest.raises(ValueError):
        assert_parameter_state("p", {})
    with pytest.raises(ValueError):
        # amplitude must be > 0
        assert_parameter_state("p", {"amplitude": 0.0, "frequency": 1.0, "phase": 0.0, "volatility_envelope": 0.0, "trend": 0.0})


def test_p70b_16_no_forbidden_imports_in_source():
    p = pathlib.Path("src/phase3/synthetic_time_series_relation_testbed.py")
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


def test_p70b_17_no_unsupported_phrases_in_source():
    p = pathlib.Path("src/phase3/synthetic_time_series_relation_testbed.py")
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


def test_p70b_18_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase3/synthetic_time_series_relation_testbed.py",
        "tools/phase3/run_p70b_synthetic_time_series_relation_testbed_smoke.py",
        "tests/test_phase3_synthetic_time_series_relation_testbed.py",
        "tests/test_phase3_p70b_synthetic_time_series_relation_testbed_smoke.py",
        "reports/PHASE_3_P70B_SYNTHETIC_TIME_SERIES_RELATION_TESTBED_NO_TRAINING_NO_MODEL_NO_OPTIMIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase3/p70b-synthetic-time-series-relation-testbed-no-training-no-model-no-optimization",
        base_commit="0a2d0b855fd6e8d8dec00122639991ae10bc6b10",
        allowed_files=allowed,
        phase_label="P70B",
    )
