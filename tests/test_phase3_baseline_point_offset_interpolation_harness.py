# tests/test_phase3_baseline_point_offset_interpolation_harness.py

import json
import math
import pathlib
import pytest

from src.phase3.baseline_point_offset_interpolation_harness import (
    PHASE,
    PHASE_GROUP,
    PHASE_NAME,
    CONTRACT_VERSION,
    SOURCE_CONTRACT_PHASE,
    TRAINING_ALLOWED,
    DATASET_GENERATION_ALLOWED,
    MODEL_IMPLEMENTATION_ALLOWED,
    OPTIMIZATION_ALLOWED,
    TORCH_ALLOWED,
    NUMPY_ALLOWED,
    BRIDGE_IMPLEMENTATION_ALLOWED,
    PRIMARY_EMPIRICAL_TARGET,
    VERDICT,
    MANDATORY_BASELINES,
    FUTURE_OPERATOR_MUST_BEAT,
    FORBIDDEN_CLAIMS,
    ALLOWED_CLAIMS,
    assert_finite_vector,
    assert_same_length,
    vector_delta,
    vector_add,
    vector_scale,
    linear_latent_interpolation,
    z_b_minus_z_a_offset_transfer,
    mean_offset_per_relation_type,
    no_relation_apply_or_decoder_baseline,
    deterministic_random_relation_vector_baseline,
    l1_distance,
    l2_distance,
    max_abs_distance,
    build_p69_static_sanity_fixtures,
    run_p69_baseline_point_offset_interpolation_harness_probe,
)


def test_p69_01_constants_exact():
    assert PHASE == "P69"
    assert PHASE_GROUP == "PHASE_3"
    assert PHASE_NAME == "Baseline Point Offset Interpolation Harness"
    assert CONTRACT_VERSION == "phase3_p69_baseline_point_offset_interpolation_harness_contract_v1"
    assert SOURCE_CONTRACT_PHASE == "P68"
    
    assert TRAINING_ALLOWED is False
    assert DATASET_GENERATION_ALLOWED is False
    assert MODEL_IMPLEMENTATION_ALLOWED is False
    assert OPTIMIZATION_ALLOWED is False
    assert TORCH_ALLOWED is False
    assert NUMPY_ALLOWED is False
    assert BRIDGE_IMPLEMENTATION_ALLOWED is False
    
    assert PRIMARY_EMPIRICAL_TARGET == "baseline_null_hypotheses_for_systematic_relational_operator_structure"
    assert VERDICT == "P69_READY_FOR_REVIEW"


def test_p69_02_probe_returns_serializable_verdict():
    probe = run_p69_baseline_point_offset_interpolation_harness_probe()
    assert isinstance(probe, dict)
    assert probe["verdict"] == "P69_READY_FOR_REVIEW"
    assert probe["p68_contract_validated"] is True
    
    serialized = json.dumps(probe)
    assert len(serialized) > 0


def test_p69_03_probe_arrays():
    probe = run_p69_baseline_point_offset_interpolation_harness_probe()
    assert probe["mandatory_baselines"] == MANDATORY_BASELINES
    assert probe["future_operator_must_beat"] == FUTURE_OPERATOR_MUST_BEAT
    assert probe["forbidden_claims"] == FORBIDDEN_CLAIMS
    assert probe["allowed_claims"] == ALLOWED_CLAIMS


def test_p69_04_linear_latent_interpolation():
    z_a = [0.0, 1.0, 2.0]
    z_b = [2.0, 3.0, 4.0]
    lambdas = [0.0, 0.5, 1.0]
    
    records = linear_latent_interpolation(z_a, z_b, lambdas)
    assert len(records) == 3
    
    assert records[0]["lambda"] == 0.0
    assert records[0]["z_lambda"] == [0.0, 1.0, 2.0]
    
    assert records[1]["lambda"] == 0.5
    assert records[1]["z_lambda"] == [1.0, 2.0, 3.0]
    
    assert records[2]["lambda"] == 1.0
    assert records[2]["z_lambda"] == [2.0, 3.0, 4.0]


def test_p69_05_z_b_minus_z_a_offset_transfer():
    z_a = [0.0, 1.0]
    z_b = [2.0, 1.0]
    z_c = [5.0, 5.0]
    
    res = z_b_minus_z_a_offset_transfer(z_a, z_b, z_c)
    assert res["delta_ab"] == [2.0, 0.0]
    assert res["z_c_transferred"] == [7.0, 5.0]


def test_p69_06_mean_offset_per_relation_type():
    records = [
        {"relation_type": "rel1", "z_a": [0.0, 0.0], "z_b": [2.0, 4.0]},
        {"relation_type": "rel1", "z_a": [1.0, 1.0], "z_b": [5.0, 7.0]},
        {"relation_type": "rel2", "z_a": [0.0, 0.0], "z_b": [1.0, -1.0]}
    ]
    
    res = mean_offset_per_relation_type(records)
    assert len(res) == 2
    
    # rel1 has deltas [2, 4] and [4, 6] -> mean is [3, 5]
    assert res["rel1"]["count"] == 2
    assert math.isclose(res["rel1"]["mean_delta"][0], 3.0)
    assert math.isclose(res["rel1"]["mean_delta"][1], 5.0)
    
    # rel2 has delta [1, -1] -> mean is [1, -1]
    assert res["rel2"]["count"] == 1
    assert math.isclose(res["rel2"]["mean_delta"][0], 1.0)
    assert math.isclose(res["rel2"]["mean_delta"][1], -1.0)


def test_p69_07_no_relation_baseline():
    z_a = [1.23, 4.56]
    lambdas = [0.0, 0.5, 1.0]
    
    records = no_relation_apply_or_decoder_baseline(z_a, lambdas)
    assert len(records) == 3
    for r in records:
        assert r["z_lambda"] == z_a


def test_p69_08_deterministic_random_relation_vector_baseline():
    z_a = [0.0, 0.0, 0.0]
    lambdas = [0.0, 1.0]
    
    # direction: direction[i] = ((-1.0) ** i) * (1.0 / (i + 2.0))
    # index 0: 1 * (1/2) = 0.5
    # index 1: -1 * (1/3) = -0.3333333333333333
    # index 2: 1 * (1/4) = 0.25
    
    records1 = deterministic_random_relation_vector_baseline(z_a, lambdas)
    records2 = deterministic_random_relation_vector_baseline(z_a, lambdas)
    
    assert records1 == records2  # deterministic
    assert len(records1) == 2
    
    assert records1[0]["lambda"] == 0.0
    assert records1[0]["z_lambda"] == [0.0, 0.0, 0.0]
    
    assert records1[1]["lambda"] == 1.0
    assert math.isclose(records1[1]["z_lambda"][0], 0.5)
    assert math.isclose(records1[1]["z_lambda"][1], -1.0 / 3.0)
    assert math.isclose(records1[1]["z_lambda"][2], 0.25)


def test_p69_09_distance_helpers():
    a = [0.0, 0.0]
    b = [3.0, 4.0]
    
    assert math.isclose(l1_distance(a, b), 7.0)
    assert math.isclose(l2_distance(a, b), 5.0)
    assert math.isclose(max_abs_distance(a, b), 4.0)


def test_p69_10_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        assert_finite_vector("v", [])
    with pytest.raises(ValueError):
        assert_finite_vector("v", [1.0, float("nan")])
    with pytest.raises(ValueError):
        assert_finite_vector("v", [1.0, float("inf")])
    with pytest.raises(ValueError):
        assert_finite_vector("v", [1.0, "not_numeric"])
    with pytest.raises(ValueError):
        assert_same_length("a", [1.0], "b", [1.0, 2.0])
    with pytest.raises(ValueError):
        linear_latent_interpolation([1.0], [1.0], [-0.1])
    with pytest.raises(ValueError):
        linear_latent_interpolation([1.0], [1.0], [1.1])


def test_p69_11_no_forbidden_imports_in_source():
    p = pathlib.Path("src/phase3/baseline_point_offset_interpolation_harness.py")
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


def test_p69_12_no_unsupported_phrases_in_source():
    p = pathlib.Path("src/phase3/baseline_point_offset_interpolation_harness.py")
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


def test_p69_13_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase3/baseline_point_offset_interpolation_harness.py",
        "tools/phase3/run_p69_baseline_point_offset_interpolation_harness_smoke.py",
        "tests/test_phase3_baseline_point_offset_interpolation_harness.py",
        "tests/test_phase3_p69_baseline_point_offset_interpolation_harness_smoke.py",
        "reports/PHASE_3_P69_BASELINE_POINT_OFFSET_INTERPOLATION_HARNESS_NO_TRAINING_NO_DATASET_NO_MODEL_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase3/p69-baseline-point-offset-interpolation-harness-no-training-no-dataset-no-model",
        base_commit="0890ada1222319a304ea4db00abec79b302f7b5f",
        allowed_files=allowed,
        phase_label="P69",
    )
