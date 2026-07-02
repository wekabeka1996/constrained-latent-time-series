# src/phase3/pure_numeric_relation_testbed.py

import json
import math
from typing import Any

from src.phase3.baseline_point_offset_interpolation_harness import (
    run_p69_baseline_point_offset_interpolation_harness_probe,
    l1_distance,
    l2_distance,
    max_abs_distance,
)

PHASE = "P70A"
PHASE_GROUP = "PHASE_3"
PHASE_NAME = "Pure Numeric Relation Testbed"
CONTRACT_VERSION = "phase3_p70a_pure_numeric_relation_testbed_contract_v1"
SOURCE_BASELINE_PHASE = "P69"

TRAINING_ALLOWED = False
MODEL_IMPLEMENTATION_ALLOWED = False
OPTIMIZATION_ALLOWED = False
TORCH_ALLOWED = False
NUMPY_ALLOWED = False
STOCHASTIC_RANDOM_ALLOWED = False
BRIDGE_IMPLEMENTATION_ALLOWED = False

TESTBED_RECORD_CONSTRUCTION_ALLOWED = True
TRAINING_DATASET_GENERATION_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "repeated_relation_instances_for_future_operator_tests"
VERDICT = "P70A_READY_FOR_REVIEW"

RELATION_TYPES = [
    "translate_x",
    "translate_y",
    "scale_s",
    "reflect_x",
    "nonlinear_x_from_y",
]

SPLIT_NAMES = [
    "train_style_repeated_instances",
    "heldout_base_state",
    "heldout_magnitude",
    "heldout_composition",
    "negative_control",
]

FUTURE_USE_CASES = [
    "p71_relation_encoder_contrastive_signal_smoke",
    "p72_sparse_operator_bank_mvp",
    "p73_transfer_and_invariant_preservation_audit",
    "p74_composition_and_order_sensitivity_audit",
    "p75_negative_control_and_collapse_audit",
]

FORBIDDEN_CLAIMS = [
    "semantic_geometry_is_proven",
    "meaning_is_learned",
    "operator_identity_is_proven",
    "relation_encoder_is_validated",
    "sparse_operator_bank_is_validated",
    "transfer_is_proven",
    "composition_is_proven",
    "schrodinger_bridge_creates_meaning",
    "geometric_schrodinger_bridge_creates_meaning",
    "testbed_records_are_training_data",
]

ALLOWED_CLAIMS = [
    "p70a_constructs_pure_numeric_relation_testbed_records",
    "p70a_provides_repeated_relation_instances",
    "p70a_defines_heldout_base_state_records",
    "p70a_defines_heldout_magnitude_records",
    "p70a_defines_composition_records",
    "p70a_defines_negative_control_records",
    "p70a_does_not_train_models",
    "p70a_does_not_establish_operator_level_evidence",
]


def assert_finite_vector(name: str, vector: list[float], expected_dim: int | None = None) -> None:
    if not isinstance(vector, list):
        raise ValueError(f"Vector {name} must be a list.")
    if len(vector) == 0:
        raise ValueError(f"Vector {name} cannot be empty.")
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(f"Vector {name} length {len(vector)} does not match expected dimension {expected_dim}.")
    for i, val in enumerate(vector):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Element {i} in vector {name} is not numeric.")
        if not math.isfinite(val):
            raise ValueError(f"Element {i} in vector {name} is not finite.")


def assert_finite_scalar(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Scalar {name} must be a numeric value.")
    if not math.isfinite(value):
        raise ValueError(f"Scalar {name} must be finite.")


def vector_close(a: list[float], b: list[float], tolerance: float = 1e-12) -> bool:
    assert_finite_vector("a", a)
    assert_finite_vector("b", b)
    if len(a) != len(b):
        raise ValueError("Mismatched dimensions in vector_close.")
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def apply_translate_x(z: list[float], intensity: float) -> list[float]:
    assert_finite_vector("z", z, expected_dim=3)
    assert_finite_scalar("intensity", intensity)
    return [float(z[0] + intensity), float(z[1]), float(z[2])]


def apply_translate_y(z: list[float], intensity: float) -> list[float]:
    assert_finite_vector("z", z, expected_dim=3)
    assert_finite_scalar("intensity", intensity)
    return [float(z[0]), float(z[1] + intensity), float(z[2])]


def apply_scale_s(z: list[float], intensity: float) -> list[float]:
    assert_finite_vector("z", z, expected_dim=3)
    assert_finite_scalar("intensity", intensity)
    return [float(z[0]), float(z[1]), float(z[2] * (1.0 + intensity))]


def apply_reflect_x(z: list[float], intensity: float) -> list[float]:
    assert_finite_vector("z", z, expected_dim=3)
    assert_finite_scalar("intensity", intensity)
    return [float(-z[0]), float(z[1]), float(z[2])]


def apply_nonlinear_x_from_y(z: list[float], intensity: float) -> list[float]:
    assert_finite_vector("z", z, expected_dim=3)
    assert_finite_scalar("intensity", intensity)
    return [float(z[0] + intensity * ((z[1] * z[1]) + 1.0)), float(z[1]), float(z[2])]


def build_relation_specs() -> dict:
    return {
        "translate_x": {
            "relation_type": "translate_x",
            "state_dim": 3,
            "changed_indices": [0],
            "invariant_indices": [1, 2],
            "has_intensity": True,
            "is_linear": True,
            "is_invertible": True,
            "is_self_inverse": False,
            "composition_family": "translation",
            "order_sensitive_with": ["reflect_x"]
        },
        "translate_y": {
            "relation_type": "translate_y",
            "state_dim": 3,
            "changed_indices": [1],
            "invariant_indices": [0, 2],
            "has_intensity": True,
            "is_linear": True,
            "is_invertible": True,
            "is_self_inverse": False,
            "composition_family": "translation",
            "order_sensitive_with": []
        },
        "scale_s": {
            "relation_type": "scale_s",
            "state_dim": 3,
            "changed_indices": [2],
            "invariant_indices": [0, 1],
            "has_intensity": True,
            "is_linear": False,
            "is_invertible": True,
            "is_self_inverse": False,
            "composition_family": "scale",
            "order_sensitive_with": []
        },
        "reflect_x": {
            "relation_type": "reflect_x",
            "state_dim": 3,
            "changed_indices": [0],
            "invariant_indices": [1, 2],
            "has_intensity": True,
            "is_linear": True,
            "is_invertible": True,
            "is_self_inverse": True,
            "composition_family": "reflection",
            "order_sensitive_with": ["translate_x", "nonlinear_x_from_y"]
        },
        "nonlinear_x_from_y": {
            "relation_type": "nonlinear_x_from_y",
            "state_dim": 3,
            "changed_indices": [0],
            "invariant_indices": [1, 2],
            "has_intensity": True,
            "is_linear": False,
            "is_invertible": True,
            "is_self_inverse": False,
            "composition_family": "nonlinear_shear",
            "order_sensitive_with": ["reflect_x"]
        }
    }


def build_base_states() -> dict:
    return {
        "state_dim": 3,
        "train_style_base_states": [
            [0.0, 0.0, 1.0],
            [1.0, 0.5, 2.0],
            [-1.0, 1.0, 1.5],
            [2.0, -0.5, 0.75],
        ],
        "heldout_base_states": [
            [0.25, -1.0, 1.25],
            [-2.0, 0.75, 2.5],
        ],
    }


def build_intensity_values() -> dict:
    return {
        "train_style_intensities": [0.5, 1.0],
        "heldout_intensities": [2.0],
        "reflection_schema_intensity": [1.0],
    }


def apply_relation(z: list[float], relation_type: str, intensity: float) -> list[float]:
    if relation_type == "translate_x":
        return apply_translate_x(z, intensity)
    elif relation_type == "translate_y":
        return apply_translate_y(z, intensity)
    elif relation_type == "scale_s":
        return apply_scale_s(z, intensity)
    elif relation_type == "reflect_x":
        return apply_reflect_x(z, intensity)
    elif relation_type == "nonlinear_x_from_y":
        return apply_nonlinear_x_from_y(z, intensity)
    else:
        raise ValueError(f"Unknown relation type: '{relation_type}'")


def compute_invariant_violation(
    z_before: list[float],
    z_after: list[float],
    invariant_indices: list[int],
) -> float:
    assert_finite_vector("z_before", z_before, expected_dim=3)
    assert_finite_vector("z_after", z_after, expected_dim=3)
    
    max_drift = 0.0
    for idx in invariant_indices:
        if idx < 0 or idx >= 3:
            raise ValueError(f"Invalid invariant index: {idx}")
        drift = abs(z_before[idx] - z_after[idx])
        if drift > max_drift:
            max_drift = drift
            
    return float(max_drift)


def build_relation_record(
    record_id: str,
    split: str,
    relation_type: str,
    z_a: list[float],
    intensity: float,
    role: str,
) -> dict:
    z_b = apply_relation(z_a, relation_type, intensity)
    specs = build_relation_specs()
    spec = specs[relation_type]
    
    violation = compute_invariant_violation(z_a, z_b, spec["invariant_indices"])
    
    return {
        "record_id": record_id,
        "split": split,
        "role": role,
        "relation_type": relation_type,
        "z_a": [float(val) for val in z_a],
        "z_b": z_b,
        "intensity": float(intensity),
        "state_dim": 3,
        "changed_indices": spec["changed_indices"],
        "invariant_indices": spec["invariant_indices"],
        "invariant_violation": violation,
        "p69_baseline_compatible": True,
    }


def build_p70a_pure_numeric_relation_testbed() -> dict:
    specs = build_relation_specs()
    bases = build_base_states()
    intensities = build_intensity_values()
    
    relation_records = []
    record_counter = 0
    
    # 1. Train-style repeated instances
    for rel_type in RELATION_TYPES:
        int_list = intensities["reflection_schema_intensity"] if rel_type == "reflect_x" else intensities["train_style_intensities"]
        for base in bases["train_style_base_states"]:
            for intens in int_list:
                record_id = f"rec_{record_counter:04d}"
                record_counter += 1
                rec = build_relation_record(
                    record_id=record_id,
                    split="train_style_repeated_instances",
                    relation_type=rel_type,
                    z_a=base,
                    intensity=intens,
                    role="train_style"
                )
                relation_records.append(rec)
                
    # 2. Heldout base records
    for rel_type in RELATION_TYPES:
        intens = 1.0 if rel_type == "reflect_x" else 0.5
        for base in bases["heldout_base_states"]:
            record_id = f"rec_{record_counter:04d}"
            record_counter += 1
            rec = build_relation_record(
                record_id=record_id,
                split="heldout_base_state",
                relation_type=rel_type,
                z_a=base,
                intensity=intens,
                role="heldout_base"
            )
            relation_records.append(rec)
            
    # 3. Heldout magnitude records
    for rel_type in RELATION_TYPES:
        if rel_type == "reflect_x":
            continue
        intens = intensities["heldout_intensities"][0] # 2.0
        # Use at least 2 train-style base states
        for base in bases["train_style_base_states"][:2]:
            record_id = f"rec_{record_counter:04d}"
            record_counter += 1
            rec = build_relation_record(
                record_id=record_id,
                split="heldout_magnitude",
                relation_type=rel_type,
                z_a=base,
                intensity=intens,
                role="heldout_magnitude"
            )
            relation_records.append(rec)
            
    # 4. Composition records
    composition_records = []
    comp_counter = 0
    
    # We define base states to compose on: e.g. [1.0, 0.5, 2.0]
    base_comp = [1.0, 0.5, 2.0]
    
    compositions_def = [
        # (rel1, intens1, rel2, intens2, expected_order_sensitive)
        ("translate_x", 1.0, "translate_y", 0.5, False),
        ("translate_y", 0.5, "translate_x", 1.0, False),
        ("reflect_x", 1.0, "translate_x", 1.0, True),
        ("translate_x", 1.0, "reflect_x", 1.0, True),
        ("translate_x", 1.0, "nonlinear_x_from_y", 0.5, False),
        ("nonlinear_x_from_y", 0.5, "translate_x", 1.0, False),
    ]
    
    for first_rel, first_int, second_rel, second_int, order_sens in compositions_def:
        z_start = base_comp
        z_mid = apply_relation(z_start, first_rel, first_int)
        z_end = apply_relation(z_mid, second_rel, second_int)
        
        comp_id = f"comp_{comp_counter:04d}"
        comp_counter += 1
        
        composition_records.append({
            "composition_id": comp_id,
            "split": "heldout_composition",
            "z_start": [float(val) for val in z_start],
            "first_relation_type": first_rel,
            "first_intensity": float(first_int),
            "z_mid": z_mid,
            "second_relation_type": second_rel,
            "second_intensity": float(second_int),
            "z_end": z_end,
            "reverse_order_available": True,
            "expected_order_sensitive": order_sens,
            "p74_ready": True,
        })
        
    # 5. Negative control records
    negative_control_records = []
    neg_counter = 0
    
    # 5.1 Label permutation records
    # For each relation type, cycle to another
    for idx, rel_type in enumerate(RELATION_TYPES):
        permuted_type = RELATION_TYPES[(idx + 1) % len(RELATION_TYPES)]
        base = bases["train_style_base_states"][0]
        intens = 1.0 if rel_type == "reflect_x" else 0.5
        z_b = apply_relation(base, rel_type, intens)
        
        neg_id = f"neg_{neg_counter:04d}"
        neg_counter += 1
        negative_control_records.append({
            "record_id": neg_id,
            "negative_control_type": "label_permutation",
            "z_a": [float(val) for val in base],
            "z_b": z_b,
            "true_relation_type": rel_type,
            "assigned_relation_type": permuted_type,
            "expected_to_fail_relation_identity": True,
        })
        
    # 5.2 Mismatched pair records
    # Pair z_a from base_states[0] with z_b from base_states[1] under translate_x
    base_a = bases["train_style_base_states"][0]
    base_b = bases["train_style_base_states"][1]
    z_b_mismatch = apply_relation(base_b, "translate_x", 0.5)
    
    neg_id = f"neg_{neg_counter:04d}"
    neg_counter += 1
    negative_control_records.append({
        "record_id": neg_id,
        "negative_control_type": "mismatched_pair",
        "z_a": [float(val) for val in base_a],
        "z_b": z_b_mismatch,
        "true_relation_type": "translate_x",
        "assigned_relation_type": "translate_x",
        "expected_to_fail_relation_identity": True,
    })
    
    # Calculate split summary
    split_counts = {}
    for rec in relation_records:
        split = rec["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
    split_counts["heldout_composition"] = len(composition_records)
    split_counts["negative_control"] = len(negative_control_records)
    
    # Check repeated instances count per relation type
    instances_per_relation = {}
    for rec in relation_records:
        if rec["split"] == "train_style_repeated_instances":
            r_type = rec["relation_type"]
            instances_per_relation[r_type] = instances_per_relation.get(r_type, 0) + 1
            
    min_instances = min(instances_per_relation.values()) if instances_per_relation else 0
    max_violation = max(rec["invariant_violation"] for rec in relation_records)
    
    sanity_summary = {
        "all_relation_types_present": len(instances_per_relation) == len(RELATION_TYPES),
        "all_relation_types_have_repeated_instances": all(val >= 2 for val in instances_per_relation.values()),
        "min_train_style_instances_per_relation_type": int(min_instances),
        "heldout_base_records_present": any(rec["split"] == "heldout_base_state" for rec in relation_records),
        "heldout_magnitude_records_present": any(rec["split"] == "heldout_magnitude" for rec in relation_records),
        "composition_records_present": len(composition_records) > 0,
        "negative_control_records_present": len(negative_control_records) > 0,
        "max_invariant_violation": float(max_violation),
        "p69_baseline_compatible_records_present": all(rec["p69_baseline_compatible"] for rec in relation_records),
        "json_safe": True,
    }
    
    return {
        "testbed_kind": "pure_numeric_relation_testbed",
        "state_dim": 3,
        "relation_specs": specs,
        "base_states": bases,
        "intensity_values": intensities,
        "relation_records": relation_records,
        "composition_records": composition_records,
        "negative_control_records": negative_control_records,
        "split_summary": split_counts,
        "repeated_relation_instance_summary": instances_per_relation,
        "sanity_summary": sanity_summary,
    }


def validate_p69_baseline_contract() -> dict:
    p69_res = run_p69_baseline_point_offset_interpolation_harness_probe()
    p69_valid = True
    missing_or_invalid = []
    
    if p69_res.get("phase") != "P69":
        p69_valid = False
        missing_or_invalid.append("phase_mismatch")
    if p69_res.get("verdict") != "P69_READY_FOR_REVIEW":
        p69_valid = False
        missing_or_invalid.append("verdict_mismatch")
        
    baselines = p69_res.get("mandatory_baselines", [])
    expected_baselines = [
        "linear_latent_interpolation",
        "z_b_minus_z_a_offset_transfer",
        "mean_offset_per_relation_type",
        "no_relation_apply_or_decoder_baseline",
        "random_relation_vector_baseline"
    ]
    for b in expected_baselines:
        if b not in baselines:
            p69_valid = False
            missing_or_invalid.append(f"missing_baseline_{b}")
            
    beat_list = p69_res.get("future_operator_must_beat", [])
    for b in expected_baselines:
        if b not in beat_list:
            p69_valid = False
            missing_or_invalid.append(f"missing_beat_{b}")
            
    if p69_res.get("torch_allowed") is not False:
        p69_valid = False
        missing_or_invalid.append("torch_allowed_must_be_false")
    if p69_res.get("numpy_allowed") is not False:
        p69_valid = False
        missing_or_invalid.append("numpy_allowed_must_be_false")
    if p69_res.get("model_implementation_allowed") is not False:
        p69_valid = False
        missing_or_invalid.append("model_allowed_must_be_false")
    if p69_res.get("bridge_implementation_allowed") is not False:
        p69_valid = False
        missing_or_invalid.append("bridge_allowed_must_be_false")
        
    if p69_res.get("p68_contract_validated") is not True:
        p69_valid = False
        missing_or_invalid.append("p68_contract_invalid")
        
    return {
        "p69_contract_validated": p69_valid,
        "missing_or_invalid": missing_or_invalid
    }


def run_p70a_pure_numeric_relation_testbed_probe() -> dict:
    # 1. Validate P69 contract
    p69_val = validate_p69_baseline_contract()
    p69_valid = p69_val["p69_contract_validated"]
    
    # 2. Build specs and testbed
    specs = build_relation_specs()
    testbed = build_p70a_pure_numeric_relation_testbed()
    sanity = testbed["sanity_summary"]
    
    verdict_str = VERDICT if p69_valid else "P70A_BLOCKED_BY_P69_CONTRACT"
    
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,
        "source_baseline_phase": SOURCE_BASELINE_PHASE,
        "verdict": verdict_str,
        
        "training_allowed": TRAINING_ALLOWED,
        "model_implementation_allowed": MODEL_IMPLEMENTATION_ALLOWED,
        "optimization_allowed": OPTIMIZATION_ALLOWED,
        "torch_allowed": TORCH_ALLOWED,
        "numpy_allowed": NUMPY_ALLOWED,
        "stochastic_random_allowed": STOCHASTIC_RANDOM_ALLOWED,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        
        "testbed_record_construction_allowed": TESTBED_RECORD_CONSTRUCTION_ALLOWED,
        "training_dataset_generation_allowed": TRAINING_DATASET_GENERATION_ALLOWED,
        
        "primary_empirical_target": PRIMARY_EMPIRICAL_TARGET,
        "relation_types": RELATION_TYPES,
        "split_names": SPLIT_NAMES,
        "future_use_cases": FUTURE_USE_CASES,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "allowed_claims": ALLOWED_CLAIMS,
        
        "p69_contract_validated": p69_valid,
        "testbed": testbed,
        "sanity_summary": sanity,
        "json_safe": True,
        "diagnostic_only": True
    }
    
    # Assert JSON safe
    json.dumps(output)
    
    return output
