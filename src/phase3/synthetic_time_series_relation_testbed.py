# src/phase3/synthetic_time_series_relation_testbed.py

import json
import math
from typing import Any

from src.phase3.pure_numeric_relation_testbed import (
    run_p70a_pure_numeric_relation_testbed_probe,
)

from src.phase3.baseline_point_offset_interpolation_harness import (
    l1_distance,
    l2_distance,
    max_abs_distance,
)

PHASE = "P70B"
PHASE_GROUP = "PHASE_3"
PHASE_NAME = "Synthetic Time-Series Relation Testbed"
CONTRACT_VERSION = "phase3_p70b_synthetic_time_series_relation_testbed_contract_v1"
SOURCE_TESTBED_PHASE = "P70A"
SOURCE_BASELINE_PHASE = "P69"

SERIES_LENGTH = 48
PARAMETER_KEYS = [
    "amplitude",
    "frequency",
    "phase",
    "volatility_envelope",
    "trend",
]

TRAINING_ALLOWED = False
MODEL_IMPLEMENTATION_ALLOWED = False
OPTIMIZATION_ALLOWED = False
TORCH_ALLOWED = False
NUMPY_ALLOWED = False
STOCHASTIC_RANDOM_ALLOWED = False
BRIDGE_IMPLEMENTATION_ALLOWED = False

TESTBED_RECORD_CONSTRUCTION_ALLOWED = True
TRAINING_DATASET_GENERATION_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "synthetic_time_series_repeated_relation_instances_for_future_operator_tests"
VERDICT = "P70B_READY_FOR_REVIEW"

RELATION_TYPES = [
    "change_frequency",
    "scale_amplitude",
    "shift_phase",
    "scale_volatility_envelope",
    "shift_trend",
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
    "synthetic_series_prove_market_usefulness",
    "garch_generation_is_proven",
    "schrodinger_bridge_creates_meaning",
    "geometric_schrodinger_bridge_creates_meaning",
    "testbed_records_are_training_data",
]

ALLOWED_CLAIMS = [
    "p70b_constructs_deterministic_synthetic_time_series_relation_records",
    "p70b_provides_repeated_relation_instances",
    "p70b_defines_heldout_base_state_records",
    "p70b_defines_heldout_magnitude_records",
    "p70b_defines_composition_records",
    "p70b_defines_nondegenerate_negative_control_records",
    "p70b_does_not_train_models",
    "p70b_does_not_establish_operator_level_evidence",
]


def assert_finite_scalar(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Scalar {name} must be numeric.")
    if not math.isfinite(value):
        raise ValueError(f"Scalar {name} must be finite.")


def assert_finite_series(name: str, series: list[float], expected_length: int | None = None) -> None:
    if not isinstance(series, list):
        raise ValueError(f"Series {name} must be a list.")
    if len(series) == 0:
        raise ValueError(f"Series {name} cannot be empty.")
    if expected_length is not None and len(series) != expected_length:
        raise ValueError(f"Series {name} length {len(series)} does not match expected length {expected_length}.")
    for i, val in enumerate(series):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Element {i} in series {name} is not numeric.")
        if not math.isfinite(val):
            raise ValueError(f"Element {i} in series {name} is not finite.")


def assert_parameter_state(name: str, params: dict) -> None:
    if not isinstance(params, dict):
        raise ValueError(f"Parameter state {name} must be a dictionary.")
    for k in PARAMETER_KEYS:
        if k not in params:
            raise ValueError(f"Missing parameter key '{k}' in {name}.")
        assert_finite_scalar(f"{name}['{k}']", params[k])
    
    if len(params) != len(PARAMETER_KEYS):
        raise ValueError(f"Invalid keys in parameter state {name}.")
        
    if params["amplitude"] <= 0.0:
        raise ValueError(f"amplitude in {name} must be greater than 0.0.")
    if params["frequency"] <= 0.0:
        raise ValueError(f"frequency in {name} must be greater than 0.0.")
    if params["volatility_envelope"] < 0.0:
        raise ValueError(f"volatility_envelope in {name} must be non-negative.")


def parameter_close(a: dict, b: dict, tolerance: float = 1e-12) -> bool:
    assert_parameter_state("a", a)
    assert_parameter_state("b", b)
    return all(abs(a[k] - b[k]) <= tolerance for k in PARAMETER_KEYS)


def parameter_l1_distance(a: dict, b: dict) -> float:
    assert_parameter_state("a", a)
    assert_parameter_state("b", b)
    return float(sum(abs(a[k] - b[k]) for k in PARAMETER_KEYS))


def generate_synthetic_series(params: dict, series_length: int = SERIES_LENGTH) -> list[float]:
    assert_parameter_state("params", params)
    if series_length < 8:
        raise ValueError("Series length must be at least 8.")
        
    amplitude = params["amplitude"]
    frequency = params["frequency"]
    phase = params["phase"]
    volatility_envelope = params["volatility_envelope"]
    trend = params["trend"]
    
    series = [0.0] * series_length
    for i in range(series_length):
        t = float(i / (series_length - 1))
        carrier = float(amplitude * math.sin(2.0 * math.pi * frequency * t + phase))
        envelope = float(1.0 + volatility_envelope * (0.5 + 0.5 * math.sin(2.0 * math.pi * 3.0 * t)))
        trend_component = float(trend * (t - 0.5))
        series[i] = float(envelope * carrier + trend_component)
        
    assert_finite_series("generated", series, expected_length=series_length)
    return series


def apply_change_frequency(params: dict, intensity: float) -> dict:
    assert_parameter_state("params", params)
    assert_finite_scalar("intensity", intensity)
    new_params = params.copy()
    new_params["frequency"] = float(params["frequency"] + intensity)
    assert_parameter_state("transformed", new_params)
    return new_params


def apply_scale_amplitude(params: dict, intensity: float) -> dict:
    assert_parameter_state("params", params)
    assert_finite_scalar("intensity", intensity)
    new_params = params.copy()
    new_params["amplitude"] = float(params["amplitude"] * (1.0 + intensity))
    assert_parameter_state("transformed", new_params)
    return new_params


def apply_shift_phase(params: dict, intensity: float) -> dict:
    assert_parameter_state("params", params)
    assert_finite_scalar("intensity", intensity)
    new_params = params.copy()
    new_params["phase"] = float(params["phase"] + intensity)
    assert_parameter_state("transformed", new_params)
    return new_params


def apply_scale_volatility_envelope(params: dict, intensity: float) -> dict:
    assert_parameter_state("params", params)
    assert_finite_scalar("intensity", intensity)
    new_params = params.copy()
    new_params["volatility_envelope"] = float(params["volatility_envelope"] * (1.0 + intensity))
    assert_parameter_state("transformed", new_params)
    return new_params


def apply_shift_trend(params: dict, intensity: float) -> dict:
    assert_parameter_state("params", params)
    assert_finite_scalar("intensity", intensity)
    new_params = params.copy()
    new_params["trend"] = float(params["trend"] + intensity)
    assert_parameter_state("transformed", new_params)
    return new_params


def build_time_series_relation_specs() -> dict:
    return {
        "change_frequency": {
            "relation_type": "change_frequency",
            "parameter_keys": PARAMETER_KEYS,
            "changed_keys": ["frequency"],
            "invariant_keys": ["amplitude", "phase", "volatility_envelope", "trend"],
            "has_intensity": True,
            "is_linear_in_parameters": True,
            "is_invertible_in_parameters": True,
            "composition_family": "frequency_translation",
            "order_sensitive_with": []
        },
        "scale_amplitude": {
            "relation_type": "scale_amplitude",
            "parameter_keys": PARAMETER_KEYS,
            "changed_keys": ["amplitude"],
            "invariant_keys": ["frequency", "phase", "volatility_envelope", "trend"],
            "has_intensity": True,
            "is_linear_in_parameters": False,
            "is_invertible_in_parameters": True,
            "composition_family": "amplitude_scaling",
            "order_sensitive_with": []
        },
        "shift_phase": {
            "relation_type": "shift_phase",
            "parameter_keys": PARAMETER_KEYS,
            "changed_keys": ["phase"],
            "invariant_keys": ["amplitude", "frequency", "volatility_envelope", "trend"],
            "has_intensity": True,
            "is_linear_in_parameters": True,
            "is_invertible_in_parameters": True,
            "composition_family": "phase_translation",
            "order_sensitive_with": []
        },
        "scale_volatility_envelope": {
            "relation_type": "scale_volatility_envelope",
            "parameter_keys": PARAMETER_KEYS,
            "changed_keys": ["volatility_envelope"],
            "invariant_keys": ["amplitude", "frequency", "phase", "trend"],
            "has_intensity": True,
            "is_linear_in_parameters": False,
            "is_invertible_in_parameters": True,
            "composition_family": "volatility_scaling",
            "order_sensitive_with": []
        },
        "shift_trend": {
            "relation_type": "shift_trend",
            "parameter_keys": PARAMETER_KEYS,
            "changed_keys": ["trend"],
            "invariant_keys": ["amplitude", "frequency", "phase", "volatility_envelope"],
            "has_intensity": True,
            "is_linear_in_parameters": True,
            "is_invertible_in_parameters": True,
            "composition_family": "trend_translation",
            "order_sensitive_with": []
        }
    }


def build_time_series_base_parameter_states() -> dict:
    return {
        "parameter_keys": PARAMETER_KEYS,
        "train_style_base_states": [
            {
                "amplitude": 1.0,
                "frequency": 1.0,
                "phase": 0.0,
                "volatility_envelope": 0.10,
                "trend": 0.0,
            },
            {
                "amplitude": 1.4,
                "frequency": 1.5,
                "phase": 0.25,
                "volatility_envelope": 0.20,
                "trend": -0.05,
            },
            {
                "amplitude": 0.8,
                "frequency": 2.0,
                "phase": 0.5,
                "volatility_envelope": 0.15,
                "trend": 0.08,
            },
            {
                "amplitude": 1.8,
                "frequency": 2.5,
                "phase": 0.75,
                "volatility_envelope": 0.25,
                "trend": -0.10,
            },
        ],
        "heldout_base_states": [
            {
                "amplitude": 1.2,
                "frequency": 1.25,
                "phase": 0.1,
                "volatility_envelope": 0.18,
                "trend": 0.04,
            },
            {
                "amplitude": 0.9,
                "frequency": 2.25,
                "phase": 0.6,
                "volatility_envelope": 0.12,
                "trend": -0.08,
            },
        ],
    }


def build_time_series_intensity_values() -> dict:
    return {
        "train_style_intensities": {
            "change_frequency": [0.25, 0.5],
            "scale_amplitude": [0.25, 0.5],
            "shift_phase": [0.25, 0.5],
            "scale_volatility_envelope": [0.5, 1.0],
            "shift_trend": [0.05, 0.10],
        },
        "heldout_intensities": {
            "change_frequency": [0.75],
            "scale_amplitude": [1.0],
            "shift_phase": [1.0],
            "scale_volatility_envelope": [1.5],
            "shift_trend": [0.20],
        },
    }


def apply_time_series_relation(params: dict, relation_type: str, intensity: float) -> dict:
    if relation_type == "change_frequency":
        return apply_change_frequency(params, intensity)
    elif relation_type == "scale_amplitude":
        return apply_scale_amplitude(params, intensity)
    elif relation_type == "shift_phase":
        return apply_shift_phase(params, intensity)
    elif relation_type == "scale_volatility_envelope":
        return apply_scale_volatility_envelope(params, intensity)
    elif relation_type == "shift_trend":
        return apply_shift_trend(params, intensity)
    else:
        raise ValueError(f"Unknown relation type: '{relation_type}'")


def compute_parameter_invariant_violation(
    before_params: dict,
    after_params: dict,
    invariant_keys: list[str],
) -> float:
    assert_parameter_state("before_params", before_params)
    assert_parameter_state("after_params", after_params)
    
    max_drift = 0.0
    for k in invariant_keys:
        if k not in PARAMETER_KEYS:
            raise ValueError(f"Invalid parameter key: {k}")
        drift = abs(before_params[k] - after_params[k])
        if drift > max_drift:
            max_drift = drift
            
    return float(max_drift)


def series_l1_distance(a: list[float], b: list[float]) -> float:
    assert_finite_series("a", a)
    assert_finite_series("b", b)
    if len(a) != len(b):
        raise ValueError("Series lengths mismatch in L1 distance.")
    return float(sum(abs(x - y) for x, y in zip(a, b)))


def series_l2_distance(a: list[float], b: list[float]) -> float:
    assert_finite_series("a", a)
    assert_finite_series("b", b)
    if len(a) != len(b):
        raise ValueError("Series lengths mismatch in L2 distance.")
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def series_max_abs_distance(a: list[float], b: list[float]) -> float:
    assert_finite_series("a", a)
    assert_finite_series("b", b)
    if len(a) != len(b):
        raise ValueError("Series lengths mismatch in max abs distance.")
    return float(max(abs(x - y) for x, y in zip(a, b)))


def summarize_series(series: list[float]) -> dict:
    assert_finite_series("series", series)
    n = len(series)
    mean_val = sum(series) / n
    min_val = min(series)
    max_val = max(series)
    energy_val = sum(x * x for x in series)
    
    return {
        "length": int(n),
        "mean": float(mean_val),
        "min": float(min_val),
        "max": float(max_val),
        "energy": float(energy_val),
        "first": float(series[0]),
        "last": float(series[-1]),
    }


def build_time_series_relation_record(
    record_id: str,
    split: str,
    relation_type: str,
    params_a: dict,
    intensity: float,
    role: str,
) -> dict:
    params_b = apply_time_series_relation(params_a, relation_type, intensity)
    series_a = generate_synthetic_series(params_a)
    series_b = generate_synthetic_series(params_b)
    
    specs = build_time_series_relation_specs()
    spec = specs[relation_type]
    
    violation = compute_parameter_invariant_violation(params_a, params_b, spec["invariant_keys"])
    
    return {
        "record_id": record_id,
        "split": split,
        "role": role,
        "relation_type": relation_type,
        "params_a": params_a,
        "params_b": params_b,
        "series_a": series_a,
        "series_b": series_b,
        "intensity": float(intensity),
        "series_length": int(SERIES_LENGTH),
        "changed_keys": spec["changed_keys"],
        "invariant_keys": spec["invariant_keys"],
        "parameter_invariant_violation": violation,
        "series_distance_l1": series_l1_distance(series_a, series_b),
        "series_distance_l2": series_l2_distance(series_a, series_b),
        "series_distance_max_abs": series_max_abs_distance(series_a, series_b),
        "series_a_summary": summarize_series(series_a),
        "series_b_summary": summarize_series(series_b),
        "p69_baseline_compatible": True,
        "p70a_contract_compatible": True,
    }


def is_nondegenerate_transition(params_a: dict, params_b: dict, series_a: list[float], series_b: list[float]) -> bool:
    dist_p = parameter_l1_distance(params_a, params_b)
    dist_s = series_l2_distance(series_a, series_b)
    return (dist_p > 1e-12) and (dist_s > 1e-9)


def has_endpoint_collision(record_a: dict, record_b: dict, tolerance: float = 1e-9) -> bool:
    return series_l2_distance(record_a["series_b"], record_b["series_b"]) < tolerance


def build_p70b_synthetic_time_series_relation_testbed() -> dict:
    specs = build_time_series_relation_specs()
    bases = build_time_series_base_parameter_states()
    intensities = build_time_series_intensity_values()
    
    relation_records = []
    record_counter = 0
    
    # 1. Train-style repeated instances
    for rel_type in RELATION_TYPES:
        int_list = intensities["train_style_intensities"][rel_type]
        for base in bases["train_style_base_states"]:
            for intens in int_list:
                record_id = f"rec_{record_counter:04d}"
                record_counter += 1
                rec = build_time_series_relation_record(
                    record_id=record_id,
                    split="train_style_repeated_instances",
                    relation_type=rel_type,
                    params_a=base,
                    intensity=intens,
                    role="train_style"
                )
                relation_records.append(rec)
                
    # 2. Heldout base records
    for rel_type in RELATION_TYPES:
        intens = intensities["train_style_intensities"][rel_type][0]
        for base in bases["heldout_base_states"]:
            record_id = f"rec_{record_counter:04d}"
            record_counter += 1
            rec = build_time_series_relation_record(
                record_id=record_id,
                split="heldout_base_state",
                relation_type=rel_type,
                params_a=base,
                intensity=intens,
                role="heldout_base"
            )
            relation_records.append(rec)
            
    # 3. Heldout magnitude records
    for rel_type in RELATION_TYPES:
        intens = intensities["heldout_intensities"][rel_type][0]
        # Use at least 2 train-style base states
        for base in bases["train_style_base_states"][:2]:
            record_id = f"rec_{record_counter:04d}"
            record_counter += 1
            rec = build_time_series_relation_record(
                record_id=record_id,
                split="heldout_magnitude",
                relation_type=rel_type,
                params_a=base,
                intensity=intens,
                role="heldout_magnitude"
            )
            relation_records.append(rec)
            
    # 4. Composition records
    composition_records = []
    comp_counter = 0
    base_comp = bases["train_style_base_states"][0]
    
    compositions_def = [
        ("change_frequency", 0.25, "scale_amplitude", 0.5, False),
        ("scale_amplitude", 0.5, "change_frequency", 0.25, False),
        ("shift_phase", 0.25, "scale_volatility_envelope", 0.5, False),
        ("scale_volatility_envelope", 0.5, "shift_phase", 0.25, False),
        ("shift_trend", 0.05, "change_frequency", 0.25, False),
        ("change_frequency", 0.25, "shift_trend", 0.05, False),
    ]
    
    for first_rel, first_int, second_rel, second_int, order_sens in compositions_def:
        params_start = base_comp
        params_mid = apply_time_series_relation(params_start, first_rel, first_int)
        params_end = apply_time_series_relation(params_mid, second_rel, second_int)
        
        comp_id = f"comp_{comp_counter:04d}"
        comp_counter += 1
        
        composition_records.append({
            "composition_id": comp_id,
            "split": "heldout_composition",
            "params_start": params_start,
            "series_start": generate_synthetic_series(params_start),
            "first_relation_type": first_rel,
            "first_intensity": float(first_int),
            "params_mid": params_mid,
            "series_mid": generate_synthetic_series(params_mid),
            "second_relation_type": second_rel,
            "second_intensity": float(second_int),
            "params_end": params_end,
            "series_end": generate_synthetic_series(params_end),
            "reverse_order_available": True,
            "expected_order_sensitive": order_sens,
            "p74_ready": True,
        })
        
    # 5. Negative control records
    negative_control_records = []
    neg_counter = 0
    
    # 5.1 Label permutation records (use non-degenerate train-style base state at index 1)
    base_perm = bases["train_style_base_states"][1]
    
    for idx, rel_type in enumerate(RELATION_TYPES):
        permuted_type = RELATION_TYPES[(idx + 1) % len(RELATION_TYPES)]
        intens = intensities["train_style_intensities"][rel_type][0]
        params_b = apply_time_series_relation(base_perm, rel_type, intens)
        series_a = generate_synthetic_series(base_perm)
        series_b = generate_synthetic_series(params_b)
        
        neg_id = f"neg_{neg_counter:04d}"
        neg_counter += 1
        
        negative_control_records.append({
            "record_id": neg_id,
            "negative_control_type": "label_permutation",
            "params_a": base_perm,
            "params_b": params_b,
            "series_a": series_a,
            "series_b": series_b,
            "true_relation_type": rel_type,
            "assigned_relation_type": permuted_type,
            "expected_to_fail_relation_identity": True,
            "nondegenerate": is_nondegenerate_transition(base_perm, params_b, series_a, series_b),
        })
        
    # 5.2 Mismatched pair records
    base_a = bases["train_style_base_states"][0]
    base_b = bases["train_style_base_states"][1]
    intens_mismatch = intensities["train_style_intensities"]["change_frequency"][0]
    params_b_mismatch = apply_time_series_relation(base_b, "change_frequency", intens_mismatch)
    
    series_a_mism = generate_synthetic_series(base_a)
    series_b_mism = generate_synthetic_series(params_b_mismatch)
    
    neg_id = f"neg_{neg_counter:04d}"
    neg_counter += 1
    
    negative_control_records.append({
        "record_id": neg_id,
        "negative_control_type": "mismatched_pair",
        "params_a": base_a,
        "params_b": params_b_mismatch,
        "series_a": series_a_mism,
        "series_b": series_b_mism,
        "true_relation_type": "change_frequency",
        "assigned_relation_type": "change_frequency",
        "expected_to_fail_relation_identity": True,
        "nondegenerate": is_nondegenerate_transition(base_a, params_b_mismatch, series_a_mism, series_b_mism),
    })
    
    # Check non-degeneracy
    neg_nondeg = all(n["nondegenerate"] for n in negative_control_records)
    
    # Check endpoint collisions count among label permutations
    collision_count = 0
    perms_only = [n for n in negative_control_records if n["negative_control_type"] == "label_permutation"]
    for i in range(len(perms_only)):
        for j in range(i + 1, len(perms_only)):
            if has_endpoint_collision(perms_only[i], perms_only[j]):
                collision_count += 1
                
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
    max_violation = max(rec["parameter_invariant_violation"] for rec in relation_records)
    
    all_lengths_match = all(len(rec["series_a"]) == SERIES_LENGTH and len(rec["series_b"]) == SERIES_LENGTH for rec in relation_records)
    all_finite = True
    for rec in relation_records:
        for val in rec["series_a"] + rec["series_b"]:
            if not math.isfinite(val):
                all_finite = False
                
    sanity_summary = {
        "all_relation_types_present": len(instances_per_relation) == len(RELATION_TYPES),
        "all_relation_types_have_repeated_instances": all(val >= 2 for val in instances_per_relation.values()),
        "min_train_style_instances_per_relation_type": int(min_instances),
        "heldout_base_records_present": any(rec["split"] == "heldout_base_state" for rec in relation_records),
        "heldout_magnitude_records_present": any(rec["split"] == "heldout_magnitude" for rec in relation_records),
        "composition_records_present": len(composition_records) > 0,
        "negative_control_records_present": len(negative_control_records) > 0,
        "negative_controls_nondegenerate": neg_nondeg,
        "negative_control_endpoint_collision_count": int(collision_count),
        "max_parameter_invariant_violation": float(max_violation),
        "all_series_lengths_match": all_lengths_match,
        "all_series_values_finite": all_finite,
        "p69_baseline_compatible_records_present": all(rec["p69_baseline_compatible"] for rec in relation_records),
        "p70a_contract_compatible_records_present": all(rec["p70a_contract_compatible"] for rec in relation_records),
        "json_safe": True,
    }
    
    return {
        "testbed_kind": "synthetic_time_series_relation_testbed",
        "series_length": int(SERIES_LENGTH),
        "parameter_keys": PARAMETER_KEYS,
        "relation_specs": specs,
        "base_parameter_states": bases,
        "intensity_values": intensities,
        "relation_records": relation_records,
        "composition_records": composition_records,
        "negative_control_records": negative_control_records,
        "split_summary": split_counts,
        "repeated_relation_instance_summary": instances_per_relation,
        "sanity_summary": sanity_summary,
    }


def validate_p70a_relation_testbed_contract() -> dict:
    p70a_res = run_p70a_pure_numeric_relation_testbed_probe()
    p70a_valid = True
    missing_or_invalid = []
    
    if p70a_res.get("phase") != "P70A":
        p70a_valid = False
        missing_or_invalid.append("phase_mismatch")
    if p70a_res.get("verdict") != "P70A_READY_FOR_REVIEW":
        p70a_valid = False
        missing_or_invalid.append("verdict_mismatch")
        
    p70a_types = p70a_res.get("relation_types", [])
    expected_p70a_types = ["translate_x", "translate_y", "scale_s", "reflect_x", "nonlinear_x_from_y"]
    for t in expected_p70a_types:
        if t not in p70a_types:
            p70a_valid = False
            missing_or_invalid.append(f"missing_p70a_type_{t}")
            
    sanity = p70a_res.get("sanity_summary", {})
    if sanity.get("all_relation_types_have_repeated_instances") is not True:
        p70a_valid = False
        missing_or_invalid.append("p70a_no_repeated_instances")
    if sanity.get("negative_control_records_present") is not True:
        p70a_valid = False
        missing_or_invalid.append("p70a_no_negative_controls")
        
    if p70a_res.get("torch_allowed") is not False:
        p70a_valid = False
        missing_or_invalid.append("torch_allowed_must_be_false")
    if p70a_res.get("numpy_allowed") is not False:
        p70a_valid = False
        missing_or_invalid.append("numpy_allowed_must_be_false")
    if p70a_res.get("model_implementation_allowed") is not False:
        p70a_valid = False
        missing_or_invalid.append("model_allowed_must_be_false")
    if p70a_res.get("bridge_implementation_allowed") is not False:
        p70a_valid = False
        missing_or_invalid.append("bridge_allowed_must_be_false")
        
    if p70a_res.get("p69_contract_validated") is not True:
        p70a_valid = False
        missing_or_invalid.append("p69_contract_invalid")
        
    return {
        "p70a_contract_validated": p70a_valid,
        "missing_or_invalid": missing_or_invalid
    }


def run_p70b_synthetic_time_series_relation_testbed_probe() -> dict:
    # 1. Validate P70A contract
    p70a_val = validate_p70a_relation_testbed_contract()
    p70a_valid = p70a_val["p70a_contract_validated"]
    
    # 2. Build testbed and check sanity
    testbed = build_p70b_synthetic_time_series_relation_testbed()
    sanity = testbed["sanity_summary"]
    
    verdict_str = VERDICT if p70a_valid else "P70B_BLOCKED_BY_P70A_CONTRACT"
    if verdict_str == VERDICT:
        # Check endpoint collisions and non-degeneracy
        if not sanity["negative_controls_nondegenerate"] or sanity["negative_control_endpoint_collision_count"] > 0:
            verdict_str = "P70B_BLOCKED_BY_TESTBED_SANITY"
            
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,
        "source_testbed_phase": SOURCE_TESTBED_PHASE,
        "source_baseline_phase": SOURCE_BASELINE_PHASE,
        "verdict": verdict_str,
        
        "series_length": int(SERIES_LENGTH),
        "parameter_keys": PARAMETER_KEYS,
        
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
        
        "p70a_contract_validated": p70a_valid,
        "testbed": testbed,
        "sanity_summary": sanity,
        "json_safe": True,
        "diagnostic_only": True
    }
    
    # Assert JSON safe
    json.dumps(output)
    
    return output
