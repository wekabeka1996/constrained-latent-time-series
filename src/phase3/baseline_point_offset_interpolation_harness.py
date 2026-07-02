# src/phase3/baseline_point_offset_interpolation_harness.py

import json
import math
from typing import Any

from src.phase3.latent_operator_genesis_research_contract import (
    run_p68_latent_operator_genesis_research_contract_probe,
)

PHASE = "P69"
PHASE_GROUP = "PHASE_3"
PHASE_NAME = "Baseline Point Offset Interpolation Harness"
CONTRACT_VERSION = "phase3_p69_baseline_point_offset_interpolation_harness_contract_v1"
SOURCE_CONTRACT_PHASE = "P68"

TRAINING_ALLOWED = False
DATASET_GENERATION_ALLOWED = False
MODEL_IMPLEMENTATION_ALLOWED = False
OPTIMIZATION_ALLOWED = False
TORCH_ALLOWED = False
NUMPY_ALLOWED = False
BRIDGE_IMPLEMENTATION_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "baseline_null_hypotheses_for_systematic_relational_operator_structure"
VERDICT = "P69_READY_FOR_REVIEW"

MANDATORY_BASELINES = [
    "linear_latent_interpolation",
    "z_b_minus_z_a_offset_transfer",
    "mean_offset_per_relation_type",
    "no_relation_apply_or_decoder_baseline",
    "random_relation_vector_baseline",
]

FUTURE_OPERATOR_MUST_BEAT = [
    "linear_latent_interpolation",
    "z_b_minus_z_a_offset_transfer",
    "mean_offset_per_relation_type",
    "no_relation_apply_or_decoder_baseline",
    "random_relation_vector_baseline",
]

FORBIDDEN_CLAIMS = [
    "semantic_geometry_is_proven",
    "meaning_is_learned",
    "operator_identity_is_proven",
    "relation_encoder_is_validated",
    "sparse_operator_bank_is_validated",
    "schrodinger_bridge_creates_meaning",
    "linear_interpolation_failure_is_assumed_without_measurement",
    "z_b_minus_z_a_failure_is_assumed_without_measurement",
]

ALLOWED_CLAIMS = [
    "p69_defines_required_phase3_baseline_null_hypotheses",
    "p69_provides_deterministic_pure_python_baseline_primitives",
    "future_operator_modules_must_compare_against_p69_baselines",
    "p69_does_not_train_models_or_generate_datasets",
    "p69_does_not_establish_operator_level_evidence",
]


def assert_finite_vector(name: str, vector: list[float]) -> None:
    if not isinstance(vector, list):
        raise ValueError(f"Vector {name} must be a list of floats.")
    if len(vector) == 0:
        raise ValueError(f"Vector {name} cannot be empty.")
    for i, val in enumerate(vector):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Element {i} in vector {name} is not numeric.")
        if not math.isfinite(val):
            raise ValueError(f"Element {i} in vector {name} is not finite (NaN or Inf).")


def assert_same_length(name_a: str, a: list[float], name_b: str, b: list[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"Mismatched dimensions: {name_a} (length {len(a)}) and {name_b} (length {len(b)}).")


def vector_delta(a: list[float], b: list[float]) -> list[float]:
    assert_finite_vector("a", a)
    assert_finite_vector("b", b)
    assert_same_length("a", a, "b", b)
    return [float(b_val - a_val) for a_val, b_val in zip(a, b)]


def vector_add(a: list[float], delta: list[float]) -> list[float]:
    assert_finite_vector("a", a)
    assert_finite_vector("delta", delta)
    assert_same_length("a", a, "delta", delta)
    return [float(a_val + d_val) for a_val, d_val in zip(a, delta)]


def vector_scale(vector: list[float], scalar: float) -> list[float]:
    assert_finite_vector("vector", vector)
    if not isinstance(scalar, (int, float)) or isinstance(scalar, bool) or not math.isfinite(scalar):
        raise ValueError("Scale factor must be a finite numeric value.")
    return [float(val * scalar) for val in vector]


def linear_latent_interpolation(
    z_a: list[float],
    z_b: list[float],
    lambdas: list[float],
) -> list[dict]:
    assert_finite_vector("z_a", z_a)
    assert_finite_vector("z_b", z_b)
    assert_same_length("z_a", z_a, "z_b", z_b)
    
    if not isinstance(lambdas, list) or len(lambdas) == 0:
        raise ValueError("lambdas must be a non-empty list of floats.")
        
    records = []
    for idx, l in enumerate(lambdas):
        if not isinstance(l, (int, float)) or isinstance(l, bool) or not math.isfinite(l):
            raise ValueError(f"lambda at index {idx} must be a finite numeric value.")
        if l < 0.0 or l > 1.0:
            raise ValueError(f"lambda at index {idx} ({l}) is outside range [0.0, 1.0].")
            
        term_a = vector_scale(z_a, 1.0 - l)
        term_b = vector_scale(z_b, l)
        z_lambda = vector_add(term_a, term_b)
        
        records.append({
            "lambda": float(l),
            "z_lambda": z_lambda
        })
        
    return records


def z_b_minus_z_a_offset_transfer(
    z_a: list[float],
    z_b: list[float],
    z_c: list[float],
) -> dict:
    assert_finite_vector("z_a", z_a)
    assert_finite_vector("z_b", z_b)
    assert_finite_vector("z_c", z_c)
    assert_same_length("z_a", z_a, "z_b", z_b)
    assert_same_length("z_a", z_a, "z_c", z_c)
    
    delta_ab = vector_delta(z_a, z_b)
    z_c_transferred = vector_add(z_c, delta_ab)
    
    return {
        "delta_ab": delta_ab,
        "z_c_transferred": z_c_transferred
    }


def mean_offset_per_relation_type(
    transition_records: list[dict],
) -> dict:
    if not isinstance(transition_records, list) or len(transition_records) == 0:
        raise ValueError("transition_records must be a non-empty list of dicts.")
        
    deltas_by_relation = {}
    dim_by_relation = {}
    
    for idx, record in enumerate(transition_records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {idx} must be a dictionary.")
        if "relation_type" not in record or "z_a" not in record or "z_b" not in record:
            raise ValueError(f"Record at index {idx} must contain keys relation_type, z_a, and z_b.")
            
        rel_type = record["relation_type"]
        z_a = record["z_a"]
        z_b = record["z_b"]
        
        assert_finite_vector(f"record[{idx}].z_a", z_a)
        assert_finite_vector(f"record[{idx}].z_b", z_b)
        assert_same_length(f"record[{idx}].z_a", z_a, f"record[{idx}].z_b", z_b)
        
        dim = len(z_a)
        if rel_type in dim_by_relation:
            if dim != dim_by_relation[rel_type]:
                raise ValueError(f"Mismatched dimensions for relation type '{rel_type}' in record {idx}.")
        else:
            dim_by_relation[rel_type] = dim
            
        delta = vector_delta(z_a, z_b)
        if rel_type not in deltas_by_relation:
            deltas_by_relation[rel_type] = []
        deltas_by_relation[rel_type].append(delta)
        
    output = {}
    for rel_type, deltas in deltas_by_relation.items():
        n = len(deltas)
        dim = len(deltas[0])
        mean_delta = [0.0] * dim
        for d in deltas:
            for i in range(dim):
                mean_delta[i] += d[i]
        for i in range(dim):
            mean_delta[i] = float(mean_delta[i] / n)
            
        output[rel_type] = {
            "count": int(n),
            "mean_delta": mean_delta
        }
        
    return output


def no_relation_apply_or_decoder_baseline(
    z_a: list[float],
    lambdas: list[float],
) -> list[dict]:
    assert_finite_vector("z_a", z_a)
    if not isinstance(lambdas, list) or len(lambdas) == 0:
        raise ValueError("lambdas must be a non-empty list of floats.")
        
    records = []
    for idx, l in enumerate(lambdas):
        if not isinstance(l, (int, float)) or isinstance(l, bool) or not math.isfinite(l):
            raise ValueError(f"lambda at index {idx} must be a finite numeric value.")
        if l < 0.0 or l > 1.0:
            raise ValueError(f"lambda at index {idx} ({l}) is outside range [0.0, 1.0].")
            
        # Returns unchanged z_a
        records.append({
            "lambda": float(l),
            "z_lambda": [float(val) for val in z_a]
        })
        
    return records


def deterministic_random_relation_vector_baseline(
    z_a: list[float],
    lambdas: list[float],
) -> list[dict]:
    assert_finite_vector("z_a", z_a)
    if not isinstance(lambdas, list) or len(lambdas) == 0:
        raise ValueError("lambdas must be a non-empty list of floats.")
        
    dim = len(z_a)
    direction = [0.0] * dim
    for i in range(dim):
        direction[i] = float(((-1.0) ** i) * (1.0 / (i + 2.0)))
        
    records = []
    for idx, l in enumerate(lambdas):
        if not isinstance(l, (int, float)) or isinstance(l, bool) or not math.isfinite(l):
            raise ValueError(f"lambda at index {idx} must be a finite numeric value.")
        if l < 0.0 or l > 1.0:
            raise ValueError(f"lambda at index {idx} ({l}) is outside range [0.0, 1.0].")
            
        scaled_dir = vector_scale(direction, l)
        z_lambda = vector_add(z_a, scaled_dir)
        
        records.append({
            "lambda": float(l),
            "direction": direction,
            "z_lambda": z_lambda
        })
        
    return records


def l1_distance(a: list[float], b: list[float]) -> float:
    assert_finite_vector("a", a)
    assert_finite_vector("b", b)
    assert_same_length("a", a, "b", b)
    return float(sum(abs(x - y) for x, y in zip(a, b)))


def l2_distance(a: list[float], b: list[float]) -> float:
    assert_finite_vector("a", a)
    assert_finite_vector("b", b)
    assert_same_length("a", a, "b", b)
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def max_abs_distance(a: list[float], b: list[float]) -> float:
    assert_finite_vector("a", a)
    assert_finite_vector("b", b)
    assert_same_length("a", a, "b", b)
    return float(max(abs(x - y) for x, y in zip(a, b)))


def build_p69_static_sanity_fixtures() -> dict:
    return {
        "fixture_kind": "static_sanity_vectors_not_dataset",
        "vectors": {
            "z_a": [0.0, 0.0],
            "z_b": [2.0, 0.0],
            "z_c": [1.0, 1.0],
        },
        "lambdas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "transition_records": [
            {
                "relation_type": "translate_x",
                "z_a": [0.0, 0.0],
                "z_b": [2.0, 0.0],
            },
            {
                "relation_type": "translate_x",
                "z_a": [1.0, 1.0],
                "z_b": [3.0, 1.0],
            },
            {
                "relation_type": "translate_y",
                "z_a": [0.0, 0.0],
                "z_b": [0.0, 2.0],
            },
            {
                "relation_type": "translate_y",
                "z_a": [1.0, 1.0],
                "z_b": [1.0, 3.0],
            },
        ],
    }


def run_p69_baseline_point_offset_interpolation_harness_probe() -> dict:
    # 1. Validate P68 contract
    p68_res = run_p68_latent_operator_genesis_research_contract_probe()
    p68_valid = True
    
    if p68_res.get("phase") != "P68":
        p68_valid = False
    if p68_res.get("verdict") != "P68_READY_FOR_REVIEW":
        p68_valid = False
        
    seq = p68_res.get("phase3_package_sequence", [])
    if "P69_BASELINE_POINT_OFFSET_INTERPOLATION_HARNESS" not in seq:
        p68_valid = False
        
    baselines = p68_res.get("mandatory_baselines", [])
    for b in MANDATORY_BASELINES:
        if b not in baselines:
            p68_valid = False
            
    allowed = p68_res.get("allowed_claims", [])
    if "bridge_methods_are_deferred_until_operator_and_metric_evidence_exist" not in allowed:
        p68_valid = False
        
    if p68_res.get("permutation_control_required_per_module") is not True:
        p68_valid = False
    if p68_res.get("relation_labels_decoder_forbidden") is not True:
        p68_valid = False
        
    # 2. Build static fixtures
    fixtures = build_p69_static_sanity_fixtures()
    z_a = fixtures["vectors"]["z_a"]
    z_b = fixtures["vectors"]["z_b"]
    z_c = fixtures["vectors"]["z_c"]
    lambdas = fixtures["lambdas"]
    records = fixtures["transition_records"]
    
    # 3. Run all five baselines
    linear_out = linear_latent_interpolation(z_a, z_b, lambdas)
    offset_out = z_b_minus_z_a_offset_transfer(z_a, z_b, z_c)
    mean_out = mean_offset_per_relation_type(records)
    no_relation_out = no_relation_apply_or_decoder_baseline(z_a, lambdas)
    random_out = deterministic_random_relation_vector_baseline(z_a, lambdas)
    
    baseline_outputs = {
        "linear_latent_interpolation": linear_out,
        "z_b_minus_z_a_offset_transfer": offset_out,
        "mean_offset_per_relation_type": mean_out,
        "no_relation_apply_or_decoder_baseline": no_relation_out,
        "random_relation_vector_baseline": random_out
    }
    
    # 4. Compute simple sanity distances for output validation
    # Calculate difference between linear interpolation at lambda=0.5 and random relation at lambda=0.5
    linear_05 = next(r["z_lambda"] for r in linear_out if r["lambda"] == 0.5)
    random_05 = next(r["z_lambda"] for r in random_out if r["lambda"] == 0.5)
    
    sanity_distances = {
        "linear_to_random_05_l1": l1_distance(linear_05, random_05),
        "linear_to_random_05_l2": l2_distance(linear_05, random_05),
        "linear_to_random_05_max": max_abs_distance(linear_05, random_05),
        "offset_transferred_l2": l2_distance(offset_out["z_c_transferred"], [3.0, 1.0])
    }
    
    # 5. Formulate probe output
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,
        "source_contract_phase": SOURCE_CONTRACT_PHASE,
        "verdict": VERDICT if p68_valid else "FAIL",
        
        "training_allowed": TRAINING_ALLOWED,
        "dataset_generation_allowed": DATASET_GENERATION_ALLOWED,
        "model_implementation_allowed": MODEL_IMPLEMENTATION_ALLOWED,
        "optimization_allowed": OPTIMIZATION_ALLOWED,
        "torch_allowed": TORCH_ALLOWED,
        "numpy_allowed": NUMPY_ALLOWED,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        
        "primary_empirical_target": PRIMARY_EMPIRICAL_TARGET,
        "mandatory_baselines": MANDATORY_BASELINES,
        "future_operator_must_beat": FUTURE_OPERATOR_MUST_BEAT,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "allowed_claims": ALLOWED_CLAIMS,
        
        "p68_contract_validated": p68_valid,
        "static_fixture_kind": fixtures["fixture_kind"],
        "baseline_outputs": baseline_outputs,
        "sanity_distances": sanity_distances,
        
        "json_safe": True,
        "diagnostic_only": True
    }
    
    # Assert JSON-safe
    json.dumps(output)
    
    return output
