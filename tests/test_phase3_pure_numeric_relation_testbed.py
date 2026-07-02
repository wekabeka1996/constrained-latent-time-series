# tests/test_phase3_pure_numeric_relation_testbed.py

import json
import math
import pathlib
import pytest

from src.phase3.pure_numeric_relation_testbed import (
    PHASE,
    PHASE_GROUP,
    PHASE_NAME,
    CONTRACT_VERSION,
    SOURCE_BASELINE_PHASE,
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
    assert_finite_vector,
    assert_finite_scalar,
    vector_close,
    apply_translate_x,
    apply_translate_y,
    apply_scale_s,
    apply_reflect_x,
    apply_nonlinear_x_from_y,
    build_relation_specs,
    build_base_states,
    build_intensity_values,
    apply_relation,
    compute_invariant_violation,
    build_relation_record,
    build_p70a_pure_numeric_relation_testbed,
    run_p70a_pure_numeric_relation_testbed_probe,
)


def test_p70a_01_constants_exact():
    assert PHASE == "P70A"
    assert PHASE_GROUP == "PHASE_3"
    assert PHASE_NAME == "Pure Numeric Relation Testbed"
    assert CONTRACT_VERSION == "phase3_p70a_pure_numeric_relation_testbed_contract_v1"
    assert SOURCE_BASELINE_PHASE == "P69"
    
    assert TRAINING_ALLOWED is False
    assert MODEL_IMPLEMENTATION_ALLOWED is False
    assert OPTIMIZATION_ALLOWED is False
    assert TORCH_ALLOWED is False
    assert NUMPY_ALLOWED is False
    assert STOCHASTIC_RANDOM_ALLOWED is False
    assert BRIDGE_IMPLEMENTATION_ALLOWED is False
    
    assert TESTBED_RECORD_CONSTRUCTION_ALLOWED is True
    assert TRAINING_DATASET_GENERATION_ALLOWED is False
    
    assert PRIMARY_EMPIRICAL_TARGET == "repeated_relation_instances_for_future_operator_tests"
    assert VERDICT == "P70A_READY_FOR_REVIEW"


def test_p70a_02_probe_returns_serializable_verdict():
    probe = run_p70a_pure_numeric_relation_testbed_probe()
    assert isinstance(probe, dict)
    assert probe["verdict"] == "P70A_READY_FOR_REVIEW"
    assert probe["p69_contract_validated"] is True
    
    serialized = json.dumps(probe)
    assert len(serialized) > 0


def test_p70a_03_probe_arrays():
    probe = run_p70a_pure_numeric_relation_testbed_probe()
    assert probe["relation_types"] == RELATION_TYPES
    assert probe["split_names"] == SPLIT_NAMES
    assert probe["future_use_cases"] == FUTURE_USE_CASES
    assert probe["forbidden_claims"] == FORBIDDEN_CLAIMS
    assert probe["allowed_claims"] == ALLOWED_CLAIMS


def test_p70a_04_relation_specs():
    specs = build_relation_specs()
    assert len(specs) == len(RELATION_TYPES)
    for r in RELATION_TYPES:
        assert r in specs
        spec = specs[r]
        assert spec["state_dim"] == 3
        assert isinstance(spec["changed_indices"], list)
        assert isinstance(spec["invariant_indices"], list)
        assert len(spec["changed_indices"]) + len(spec["invariant_indices"]) == 3


def test_p70a_05_apply_translate_x():
    z = [1.0, 2.0, 3.0]
    res = apply_translate_x(z, 0.5)
    assert res == [1.5, 2.0, 3.0]


def test_p70a_06_apply_translate_y():
    z = [1.0, 2.0, 3.0]
    res = apply_translate_y(z, 0.5)
    assert res == [1.0, 2.5, 3.0]


def test_p70a_07_apply_scale_s():
    z = [1.0, 2.0, 3.0]
    res = apply_scale_s(z, 0.5)
    assert res == [1.0, 2.0, 4.5]


def test_p70a_08_apply_reflect_x():
    z = [1.0, 2.0, 3.0]
    res = apply_reflect_x(z, 999.0)
    assert res == [-1.0, 2.0, 3.0]


def test_p70a_09_apply_nonlinear_x_from_y():
    z = [1.0, 2.0, 3.0]
    # x + intensity * (y^2 + 1) = 1.0 + 0.5 * (4.0 + 1.0) = 1.0 + 2.5 = 3.5
    res = apply_nonlinear_x_from_y(z, 0.5)
    assert res == [3.5, 2.0, 3.0]


def test_p70a_10_invariant_violation_always_zero():
    testbed = build_p70a_pure_numeric_relation_testbed()
    records = testbed["relation_records"]
    assert len(records) > 0
    for rec in records:
        assert rec["invariant_violation"] == 0.0


def test_p70a_11_repeated_instances():
    testbed = build_p70a_pure_numeric_relation_testbed()
    summary = testbed["repeated_relation_instance_summary"]
    for rel_type in RELATION_TYPES:
        assert summary[rel_type] >= 2


def test_p70a_12_splits_exist():
    testbed = build_p70a_pure_numeric_relation_testbed()
    records = testbed["relation_records"]
    
    heldout_bases = [r for r in records if r["split"] == "heldout_base_state"]
    heldout_mags = [r for r in records if r["split"] == "heldout_magnitude"]
    
    assert len(heldout_bases) > 0
    assert len(heldout_mags) > 0


def test_p70a_13_composition_records_commutativity():
    testbed = build_p70a_pure_numeric_relation_testbed()
    comps = testbed["composition_records"]
    
    assert len(comps) >= 6
    
    # Check that translate_x and translate_y composition matches expectation
    tx_ty = next(c for c in comps if c["first_relation_type"] == "translate_x" and c["second_relation_type"] == "translate_y")
    assert tx_ty["expected_order_sensitive"] is False
    
    # Check that reflect_x and translate_x composition matches expectation
    rx_tx = next(c for c in comps if c["first_relation_type"] == "reflect_x" and c["second_relation_type"] == "translate_x")
    assert rx_tx["expected_order_sensitive"] is True


def test_p70a_14_negative_control_records():
    testbed = build_p70a_pure_numeric_relation_testbed()
    neg_controls = testbed["negative_control_records"]
    
    perms = [n for n in neg_controls if n["negative_control_type"] == "label_permutation"]
    mismatches = [n for n in neg_controls if n["negative_control_type"] == "mismatched_pair"]
    
    assert len(perms) > 0
    for p in perms:
        assert p["true_relation_type"] != p["assigned_relation_type"]
        assert p["expected_to_fail_relation_identity"] is True
        
    assert len(mismatches) > 0
    for m in mismatches:
        assert m["expected_to_fail_relation_identity"] is True


def test_p70a_15_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        assert_finite_vector("v", [])
    with pytest.raises(ValueError):
        assert_finite_vector("v", [1.0, float("nan")])
    with pytest.raises(ValueError):
        assert_finite_vector("v", [1.0, 2.0], expected_dim=3)
    with pytest.raises(ValueError):
        assert_finite_scalar("s", float("inf"))
    with pytest.raises(ValueError):
        apply_relation([1.0, 2.0, 3.0], "unknown_relation", 1.0)


def test_p70a_16_no_forbidden_imports_in_source():
    p = pathlib.Path("src/phase3/pure_numeric_relation_testbed.py")
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


def test_p70a_17_no_unsupported_phrases_in_source():
    p = pathlib.Path("src/phase3/pure_numeric_relation_testbed.py")
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


def test_p70a_18_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase3/pure_numeric_relation_testbed.py",
        "tools/phase3/run_p70a_pure_numeric_relation_testbed_smoke.py",
        "tests/test_phase3_pure_numeric_relation_testbed.py",
        "tests/test_phase3_p70a_pure_numeric_relation_testbed_smoke.py",
        "reports/PHASE_3_P70A_PURE_NUMERIC_RELATION_TESTBED_NO_TRAINING_NO_MODEL_NO_OPTIMIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase3/p70a-pure-numeric-relation-testbed-no-training-no-model-no-optimization",
        base_commit="a5c732a70af3fea94a10e2a5107c7cad321aab47",
        allowed_files=allowed,
        phase_label="P70A",
    )
