# src/phase3/oracle_sparse_operator_bank_mvp.py

import json
import math
from typing import Any

from src.phase3.baseline_point_offset_interpolation_harness import (
    run_p69_baseline_point_offset_interpolation_harness_probe,
    l1_distance,
    l2_distance,
    max_abs_distance,
)

from src.phase3.pure_numeric_relation_testbed import (
    run_p70a_pure_numeric_relation_testbed_probe,
    apply_relation,
    compute_invariant_violation,
)

from src.phase3.synthetic_time_series_relation_testbed import (
    run_p70b_synthetic_time_series_relation_testbed_probe,
    apply_time_series_relation,
    generate_synthetic_series,
    compute_parameter_invariant_violation,
    series_l1_distance,
    series_l2_distance,
    series_max_abs_distance,
)

from src.phase3.relation_contrastive_signal_smoke import (
    run_p71_relation_contrastive_signal_smoke_probe,
)

PHASE = "P72"
PHASE_GROUP = "PHASE_3"
PHASE_NAME = "Oracle Sparse Operator Bank MVP"
CONTRACT_VERSION = "phase3_p72_oracle_sparse_operator_bank_mvp_contract_v1"

SOURCE_BASELINE_PHASE = "P69"
SOURCE_VECTOR_TESTBED_PHASE = "P70A"
SOURCE_TIME_SERIES_TESTBED_PHASE = "P70B"
SOURCE_CONTRASTIVE_PHASE = "P71"

TRAINING_ALLOWED = False
MODEL_IMPLEMENTATION_ALLOWED = False
NEURAL_ENCODER_IMPLEMENTATION_ALLOWED = False
NEURAL_OPERATOR_SELECTOR_ALLOWED = False
OPTIMIZATION_ALLOWED = False
TORCH_ALLOWED = False
NUMPY_ALLOWED = False
STOCHASTIC_RANDOM_ALLOWED = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
LEARNED_METRIC_ALLOWED = False

ORACLE_OPERATOR_BANK_ALLOWED = True
ORACLE_RELATION_TYPE_SELECTION_ALLOWED = True
LEARNED_RELATION_TYPE_SELECTION_ALLOWED = False
SPARSE_OPERATOR_APPLICATION_ALLOWED = True
OPERATOR_BANK_TRAINING_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "oracle_sparse_operator_application_and_invariant_preservation"
VERDICT = "P72_READY_FOR_REVIEW"

OPERATOR_DOMAINS = [
    "p70a_vector_world",
    "p70b_time_series_parameter_world",
]

P70A_VECTOR_OPERATOR_TYPES = [
    "translate_x",
    "translate_y",
    "scale_s",
    "reflect_x",
    "nonlinear_x_from_y",
]

P70B_TIME_SERIES_OPERATOR_TYPES = [
    "change_frequency",
    "scale_amplitude",
    "shift_phase",
    "scale_volatility_envelope",
    "shift_trend",
]

EVALUATION_SPLITS = [
    "train_style_repeated_instances",
    "heldout_base_state",
    "heldout_magnitude",
]

FORBIDDEN_CLAIMS = [
    "semantic_geometry_is_proven",
    "meaning_is_learned",
    "operator_identity_is_proven",
    "relation_encoder_is_trained",
    "relation_encoder_is_validated",
    "sparse_operator_bank_is_learned",
    "sparse_operator_bank_is_validated_as_learned_model",
    "learned_operator_selection_is_proven",
    "transfer_is_proven",
    "composition_is_proven",
    "bridge_method_is_validated",
    "schrodinger_bridge_creates_meaning",
    "geometric_schrodinger_bridge_creates_meaning",
    "oracle_operator_bank_proves_semantics",
]

ALLOWED_CLAIMS = [
    "p72_constructs_oracle_sparse_operator_bank",
    "p72_uses_known_relation_type_metadata_only_as_oracle_selector",
    "p72_evaluates_sparse_operator_application",
    "p72_evaluates_invariant_preservation",
    "p72_evaluates_heldout_base_records",
    "p72_evaluates_heldout_magnitude_records",
    "p72_compares_against_null_baselines",
    "p72_repairs_p71_source_validation_depth_boundary",
    "p72_does_not_train_models",
    "p72_does_not_establish_learned_operator_evidence",
]


def assert_finite_vector(name: str, vector: list[float]) -> None:
    if not isinstance(vector, list):
        raise ValueError(f"Vector {name} must be a list.")
    if len(vector) == 0:
        raise ValueError(f"Vector {name} cannot be empty.")
    for i, val in enumerate(vector):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Element {i} in vector {name} is not numeric.")
        if not math.isfinite(val):
            raise ValueError(f"Element {i} in vector {name} is not finite.")


def assert_parameter_state(name: str, params: dict) -> None:
    keys = ["amplitude", "frequency", "phase", "volatility_envelope", "trend"]
    if not isinstance(params, dict):
        raise ValueError(f"Parameter state {name} must be a dictionary.")
    for k in keys:
        if k not in params:
            raise ValueError(f"Missing parameter key '{k}' in {name}.")
        val = params[k]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Parameter {k} in {name} is not numeric.")
        if not math.isfinite(val):
            raise ValueError(f"Parameter {k} in {name} is not finite.")
            
    if params["amplitude"] <= 0.0:
        raise ValueError(f"amplitude in {name} must be greater than 0.0.")
    if params["frequency"] <= 0.0:
        raise ValueError(f"frequency in {name} must be greater than 0.0.")
    if params["volatility_envelope"] < 0.0:
        raise ValueError(f"volatility_envelope in {name} must be non-negative.")


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-15:
        return 0.0
    return float(numerator / denominator)


def vector_l2_error(a: list[float], b: list[float]) -> float:
    assert_finite_vector("a", a)
    assert_finite_vector("b", b)
    if len(a) != len(b):
        raise ValueError("Dimensions mismatch in vector_l2_error.")
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def parameter_l2_error(a: dict, b: dict) -> float:
    assert_parameter_state("a", a)
    assert_parameter_state("b", b)
    keys = ["amplitude", "frequency", "phase", "volatility_envelope", "trend"]
    return float(math.sqrt(sum((a[k] - b[k]) ** 2 for k in keys)))


def parameter_max_abs_error(a: dict, b: dict) -> float:
    assert_parameter_state("a", a)
    assert_parameter_state("b", b)
    keys = ["amplitude", "frequency", "phase", "volatility_envelope", "trend"]
    return float(max(abs(a[k] - b[k]) for k in keys))


def build_p70a_vector_operator_specs() -> dict:
    specs = {}
    
    # translate_x: changes index 0, preserves 1, 2
    specs["translate_x"] = {
        "operator_type": "translate_x",
        "domain": "p70a_vector_world",
        "state_dim": 3,
        "changed_indices": [0],
        "invariant_indices": [1, 2],
        "sparse": True,
        "oracle_selected": True,
        "learned": False,
    }
    
    # translate_y: changes index 1, preserves 0, 2
    specs["translate_y"] = {
        "operator_type": "translate_y",
        "domain": "p70a_vector_world",
        "state_dim": 3,
        "changed_indices": [1],
        "invariant_indices": [0, 2],
        "sparse": True,
        "oracle_selected": True,
        "learned": False,
    }
    
    # scale_s: changes index 2, preserves 0, 1
    specs["scale_s"] = {
        "operator_type": "scale_s",
        "domain": "p70a_vector_world",
        "state_dim": 3,
        "changed_indices": [2],
        "invariant_indices": [0, 1],
        "sparse": True,
        "oracle_selected": True,
        "learned": False,
    }
    
    # reflect_x: changes index 0, preserves 1, 2
    specs["reflect_x"] = {
        "operator_type": "reflect_x",
        "domain": "p70a_vector_world",
        "state_dim": 3,
        "changed_indices": [0],
        "invariant_indices": [1, 2],
        "sparse": True,
        "oracle_selected": True,
        "learned": False,
    }
    
    # nonlinear_x_from_y: changes index 0, preserves 1, 2
    specs["nonlinear_x_from_y"] = {
        "operator_type": "nonlinear_x_from_y",
        "domain": "p70a_vector_world",
        "state_dim": 3,
        "changed_indices": [0],
        "invariant_indices": [1, 2],
        "sparse": True,
        "oracle_selected": True,
        "learned": False,
    }
    
    return specs


def build_p70b_time_series_operator_specs() -> dict:
    specs = {}
    keys = ["amplitude", "frequency", "phase", "volatility_envelope", "trend"]
    
    # change_frequency
    specs["change_frequency"] = {
        "operator_type": "change_frequency",
        "domain": "p70b_time_series_parameter_world",
        "parameter_keys": keys,
        "changed_keys": ["frequency"],
        "invariant_keys": ["amplitude", "phase", "volatility_envelope", "trend"],
        "sparse": True,
        "oracle_selected": True,
        "learned": False,
    }
    
    # scale_amplitude
    specs["scale_amplitude"] = {
        "operator_type": "scale_amplitude",
        "domain": "p70b_time_series_parameter_world",
        "parameter_keys": keys,
        "changed_keys": ["amplitude"],
        "invariant_keys": ["frequency", "phase", "volatility_envelope", "trend"],
        "sparse": True,
        "oracle_selected": True,
        "learned": False,
    }
    
    # shift_phase
    specs["shift_phase"] = {
        "operator_type": "shift_phase",
        "domain": "p70b_time_series_parameter_world",
        "parameter_keys": keys,
        "changed_keys": ["phase"],
        "invariant_keys": ["amplitude", "frequency", "volatility_envelope", "trend"],
        "sparse": True,
        "oracle_selected": True,
        "learned": False,
    }
    
    # scale_volatility_envelope
    specs["scale_volatility_envelope"] = {
        "operator_type": "scale_volatility_envelope",
        "domain": "p70b_time_series_parameter_world",
        "parameter_keys": keys,
        "changed_keys": ["volatility_envelope"],
        "invariant_keys": ["amplitude", "frequency", "phase", "trend"],
        "sparse": True,
        "oracle_selected": True,
        "learned": False,
    }
    
    # shift_trend
    specs["shift_trend"] = {
        "operator_type": "shift_trend",
        "domain": "p70b_time_series_parameter_world",
        "parameter_keys": keys,
        "changed_keys": ["trend"],
        "invariant_keys": ["amplitude", "frequency", "phase", "volatility_envelope"],
        "sparse": True,
        "oracle_selected": True,
        "learned": False,
    }
    
    return specs


def build_oracle_sparse_operator_bank() -> dict:
    return {
        "bank_kind": "oracle_sparse_operator_bank_mvp",
        "learned": False,
        "oracle_relation_type_selection_allowed": True,
        "learned_relation_type_selection_allowed": False,
        "domains": OPERATOR_DOMAINS,
        "p70a_vector_operator_specs": build_p70a_vector_operator_specs(),
        "p70b_time_series_operator_specs": build_p70b_time_series_operator_specs(),
    }


def apply_p70a_sparse_vector_operator(
    z_input: list[float],
    operator_type: str,
    intensity: float,
) -> list[float]:
    assert_finite_vector("z_input", z_input)
    if operator_type not in P70A_VECTOR_OPERATOR_TYPES:
        raise ValueError(f"Unknown vector operator type: '{operator_type}'")
    return apply_relation(z_input, operator_type, intensity)


def apply_p70b_sparse_parameter_operator(
    params_input: dict,
    operator_type: str,
    intensity: float,
) -> dict:
    assert_parameter_state("params_input", params_input)
    if operator_type not in P70B_TIME_SERIES_OPERATOR_TYPES:
        raise ValueError(f"Unknown time series parameter operator type: '{operator_type}'")
    return apply_time_series_relation(params_input, operator_type, intensity)


def p70a_no_relation_baseline(z_a: list[float]) -> list[float]:
    assert_finite_vector("z_a", z_a)
    return list(z_a)


def p70b_no_relation_parameter_baseline(params_a: dict) -> dict:
    assert_parameter_state("params_a", params_a)
    return dict(params_a)


def build_p70a_mean_delta_by_relation(train_style_records: list[dict]) -> dict:
    deltas = {}
    counts = {}
    for rec in train_style_records:
        r_type = rec["relation_type"]
        z_a = rec["z_a"]
        z_b = rec["z_b"]
        diff = [float(y - x) for x, y in zip(z_a, z_b)]
        
        if r_type not in deltas:
            deltas[r_type] = [0.0] * len(z_a)
            counts[r_type] = 0
            
        for k in range(len(diff)):
            deltas[r_type][k] += diff[k]
        counts[r_type] += 1
        
    mean_deltas = {}
    for r_type in deltas:
        mean_deltas[r_type] = [float(val / counts[r_type]) for val in deltas[r_type]]
    return mean_deltas


def apply_p70a_mean_delta_baseline(
    z_a: list[float],
    relation_type: str,
    mean_delta_by_relation: dict,
) -> list[float]:
    assert_finite_vector("z_a", z_a)
    mean_delta = mean_delta_by_relation.get(relation_type, [0.0] * len(z_a))
    return [float(x + d) for x, d in zip(z_a, mean_delta)]


def build_p70b_mean_parameter_delta_by_relation(train_style_records: list[dict]) -> dict:
    keys = ["amplitude", "frequency", "phase", "volatility_envelope", "trend"]
    deltas = {}
    counts = {}
    for rec in train_style_records:
        r_type = rec["relation_type"]
        pa = rec["params_a"]
        pb = rec["params_b"]
        
        if r_type not in deltas:
            deltas[r_type] = {k: 0.0 for k in keys}
            counts[r_type] = 0
            
        for k in keys:
            deltas[r_type][k] += pb[k] - pa[k]
        counts[r_type] += 1
        
    mean_deltas = {}
    for r_type in deltas:
        mean_deltas[r_type] = {k: float(deltas[r_type][k] / counts[r_type]) for k in keys}
    return mean_deltas


def apply_p70b_mean_parameter_delta_baseline(
    params_a: dict,
    relation_type: str,
    mean_delta_by_relation: dict,
) -> dict:
    assert_parameter_state("params_a", params_a)
    mean_delta = mean_delta_by_relation.get(relation_type, {k: 0.0 for k in params_a})
    
    new_params = {}
    for k in params_a:
        new_params[k] = float(params_a[k] + mean_delta[k])
        
    # Boundary correction so parameter validators do not reject baseline output
    if new_params["amplitude"] <= 0.0:
        new_params["amplitude"] = 1e-5
    if new_params["frequency"] <= 0.0:
        new_params["frequency"] = 1e-5
    if new_params["volatility_envelope"] < 0.0:
        new_params["volatility_envelope"] = 0.0
        
    return new_params


def extract_p70a_operator_evaluation_records(p70a_probe: dict) -> list[dict]:
    records = p70a_probe.get("testbed", {}).get("relation_records", [])
    return [rec for rec in records if rec.get("split") in EVALUATION_SPLITS]


def extract_p70b_operator_evaluation_records(p70b_probe: dict) -> list[dict]:
    records = p70b_probe.get("testbed", {}).get("relation_records", [])
    return [rec for rec in records if rec.get("split") in EVALUATION_SPLITS]


def summarize_error_values(values: list[float]) -> dict:
    if len(values) == 0:
        return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": int(len(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def evaluate_p70a_sparse_operator_bank(
    p70a_probe: dict,
) -> dict:
    records = extract_p70a_operator_evaluation_records(p70a_probe)
    train_style_recs = [r for r in records if r["split"] == "train_style_repeated_instances"]
    mean_delta_lookup = build_p70a_mean_delta_by_relation(train_style_recs)
    
    specs = build_p70a_vector_operator_specs()
    
    oracle_l2_errors = []
    oracle_max_abs_errors = []
    oracle_invariant_violations = []
    
    no_relation_l2_errors = []
    mean_delta_l2_errors = []
    
    split_counts = {}
    heldout_base_count = 0
    heldout_mag_count = 0
    
    heldout_base_oracle_l2s = []
    heldout_magnitude_oracle_l2s = []
    
    operator_types = set()
    
    for rec in records:
        z_a = rec["z_a"]
        z_b = rec["z_b"]
        r_type = rec["relation_type"]
        intens = rec["intensity"]
        split = rec["split"]
        
        split_counts[split] = split_counts.get(split, 0) + 1
        operator_types.add(r_type)
        
        if split == "heldout_base_state":
            heldout_base_count += 1
        elif split == "heldout_magnitude":
            heldout_mag_count += 1
            
        # 1. Oracle Sparse Operator
        z_pred = apply_p70a_sparse_vector_operator(z_a, r_type, intens)
        
        err_l2 = vector_l2_error(z_pred, z_b)
        err_max_abs = float(max(abs(x - y) for x, y in zip(z_pred, z_b)))
        
        oracle_l2_errors.append(err_l2)
        oracle_max_abs_errors.append(err_max_abs)
        
        if split == "heldout_base_state":
            heldout_base_oracle_l2s.append(err_l2)
        elif split == "heldout_magnitude":
            heldout_magnitude_oracle_l2s.append(err_l2)
            
        # Invariants
        spec = specs[r_type]
        violation = compute_invariant_violation(z_a, z_pred, spec["invariant_indices"])
        oracle_invariant_violations.append(violation)
        
        # 2. No relation baseline
        z_no_rel = p70a_no_relation_baseline(z_a)
        no_relation_l2_errors.append(vector_l2_error(z_no_rel, z_b))
        
        # 3. Mean delta baseline
        z_mean_del = apply_p70a_mean_delta_baseline(z_a, r_type, mean_delta_lookup)
        mean_delta_l2_errors.append(vector_l2_error(z_mean_del, z_b))
        
    sum_oracle_l2 = summarize_error_values(oracle_l2_errors)
    sum_no_relation_l2 = summarize_error_values(no_relation_l2_errors)
    sum_mean_delta_l2 = summarize_error_values(mean_delta_l2_errors)
    
    oracle_beats_no_relation = sum_oracle_l2["mean"] < sum_no_relation_l2["mean"]
    oracle_beats_mean_delta = sum_oracle_l2["mean"] < sum_mean_delta_l2["mean"]
    
    # Exactness check: oracle endpoint L2 error mean should be ~0.0
    diag_pass = (
        sum_oracle_l2["mean"] < 1e-12
        and max(oracle_invariant_violations) < 1e-12
        and oracle_beats_no_relation
    )
    
    return {
        "domain": "p70a_vector_world",
        "record_count": len(records),
        "split_counts": split_counts,
        "operator_types_present": sorted(list(operator_types)),
        "oracle_endpoint_l2_error": sum_oracle_l2,
        "oracle_endpoint_max_abs_error": summarize_error_values(oracle_max_abs_errors),
        "oracle_invariant_violation": summarize_error_values(oracle_invariant_violations),
        "no_relation_endpoint_l2_error": sum_no_relation_l2,
        "mean_delta_endpoint_l2_error": sum_mean_delta_l2,
        "oracle_beats_no_relation_mean_l2": oracle_beats_no_relation,
        "oracle_beats_mean_delta_mean_l2": oracle_beats_mean_delta,
        "heldout_base_record_count": heldout_base_count,
        "heldout_magnitude_record_count": heldout_mag_count,
        "heldout_base_oracle_mean_l2": float(sum(heldout_base_oracle_l2s) / len(heldout_base_oracle_l2s)) if heldout_base_oracle_l2s else 0.0,
        "heldout_magnitude_oracle_mean_l2": float(sum(heldout_magnitude_oracle_l2s) / len(heldout_magnitude_oracle_l2s)) if heldout_magnitude_oracle_l2s else 0.0,
        "diagnostic_pass": diag_pass,
    }


def evaluate_p70b_sparse_operator_bank(
    p70b_probe: dict,
) -> dict:
    records = extract_p70b_operator_evaluation_records(p70b_probe)
    train_style_recs = [r for r in records if r["split"] == "train_style_repeated_instances"]
    mean_delta_lookup = build_p70b_mean_parameter_delta_by_relation(train_style_recs)
    
    specs = build_p70b_time_series_operator_specs()
    
    oracle_parameter_l2_errors = []
    oracle_parameter_max_abs_errors = []
    oracle_series_l2_errors = []
    oracle_series_max_abs_errors = []
    oracle_invariant_violations = []
    
    no_relation_parameter_l2_errors = []
    mean_delta_parameter_l2_errors = []
    
    split_counts = {}
    heldout_base_count = 0
    heldout_mag_count = 0
    
    heldout_base_oracle_l2s = []
    heldout_magnitude_oracle_l2s = []
    
    operator_types = set()
    
    for rec in records:
        params_a = rec["params_a"]
        params_b = rec["params_b"]
        series_b = rec["series_b"]
        r_type = rec["relation_type"]
        intens = rec["intensity"]
        split = rec["split"]
        
        split_counts[split] = split_counts.get(split, 0) + 1
        operator_types.add(r_type)
        
        if split == "heldout_base_state":
            heldout_base_count += 1
        elif split == "heldout_magnitude":
            heldout_mag_count += 1
            
        # 1. Oracle Sparse Operator application
        params_pred = apply_p70b_sparse_parameter_operator(params_a, r_type, intens)
        series_pred = generate_synthetic_series(params_pred)
        
        err_param_l2 = parameter_l2_error(params_pred, params_b)
        err_param_max_abs = parameter_max_abs_error(params_pred, params_b)
        err_series_l2 = series_l2_distance(series_pred, series_b)
        err_series_max_abs = series_max_abs_distance(series_pred, series_b)
        
        oracle_parameter_l2_errors.append(err_param_l2)
        oracle_parameter_max_abs_errors.append(err_param_max_abs)
        oracle_series_l2_errors.append(err_series_l2)
        oracle_series_max_abs_errors.append(err_series_max_abs)
        
        if split == "heldout_base_state":
            heldout_base_oracle_l2s.append(err_param_l2)
        elif split == "heldout_magnitude":
            heldout_magnitude_oracle_l2s.append(err_param_l2)
            
        # Invariants
        spec = specs[r_type]
        violation = compute_parameter_invariant_violation(params_a, params_pred, spec["invariant_keys"])
        oracle_invariant_violations.append(violation)
        
        # 2. No relation baseline
        params_no_rel = p70b_no_relation_parameter_baseline(params_a)
        no_relation_parameter_l2_errors.append(parameter_l2_error(params_no_rel, params_b))
        
        # 3. Mean parameter delta baseline
        params_mean_del = apply_p70b_mean_parameter_delta_baseline(params_a, r_type, mean_delta_lookup)
        mean_delta_parameter_l2_errors.append(parameter_l2_error(params_mean_del, params_b))
        
    sum_oracle_param_l2 = summarize_error_values(oracle_parameter_l2_errors)
    sum_no_relation_param_l2 = summarize_error_values(no_relation_parameter_l2_errors)
    sum_mean_delta_param_l2 = summarize_error_values(mean_delta_parameter_l2_errors)
    
    oracle_beats_no_relation = sum_oracle_param_l2["mean"] < sum_no_relation_param_l2["mean"]
    oracle_beats_mean_delta = sum_oracle_param_l2["mean"] < sum_mean_delta_param_l2["mean"]
    
    diag_pass = (
        sum_oracle_param_l2["mean"] < 1e-12
        and max(oracle_invariant_violations) < 1e-12
        and oracle_beats_no_relation
    )
    
    return {
        "domain": "p70b_time_series_parameter_world",
        "record_count": len(records),
        "split_counts": split_counts,
        "operator_types_present": sorted(list(operator_types)),
        "oracle_parameter_l2_error": sum_oracle_param_l2,
        "oracle_parameter_max_abs_error": summarize_error_values(oracle_parameter_max_abs_errors),
        "oracle_series_l2_error": summarize_error_values(oracle_series_l2_errors),
        "oracle_series_max_abs_error": summarize_error_values(oracle_series_max_abs_errors),
        "oracle_invariant_violation": summarize_error_values(oracle_invariant_violations),
        "no_relation_parameter_l2_error": sum_no_relation_param_l2,
        "mean_delta_parameter_l2_error": sum_mean_delta_param_l2,
        "oracle_beats_no_relation_mean_l2": oracle_beats_no_relation,
        "oracle_beats_mean_delta_mean_l2": oracle_beats_mean_delta,
        "heldout_base_record_count": heldout_base_count,
        "heldout_magnitude_record_count": heldout_mag_count,
        "heldout_base_oracle_mean_l2": float(sum(heldout_base_oracle_l2s) / len(heldout_base_oracle_l2s)) if heldout_base_oracle_l2s else 0.0,
        "heldout_magnitude_oracle_mean_l2": float(sum(heldout_magnitude_oracle_l2s) / len(heldout_magnitude_oracle_l2s)) if heldout_magnitude_oracle_l2s else 0.0,
        "diagnostic_pass": diag_pass,
    }


def validate_source_contracts_deep() -> dict:
    deep_validated = True
    missing_or_invalid = []
    
    # 1. P69 deep checks
    p69_val = True
    try:
        p69_res = run_p69_baseline_point_offset_interpolation_harness_probe()
        if p69_res.get("phase") != "P69" or p69_res.get("verdict") != "P69_READY_FOR_REVIEW":
            p69_val = False
            missing_or_invalid.append("p69_invalid_phase_or_verdict")
            
        baselines = p69_res.get("mandatory_baselines", [])
        expected_baselines = [
            "linear_latent_interpolation",
            "z_b_minus_z_a_offset_transfer",
            "mean_offset_per_relation_type",
            "no_relation_apply_or_decoder_baseline",
            "random_relation_vector_baseline",
        ]
        for b in expected_baselines:
            if b not in baselines:
                p69_val = False
                missing_or_invalid.append(f"p69_missing_baseline_{b}")
                
        if p69_res.get("torch_allowed") is not False or p69_res.get("numpy_allowed") is not False:
            p69_val = False
            missing_or_invalid.append("p69_torch_or_numpy_allowed")
            
        if p69_res.get("p68_contract_validated") is not True:
            p69_val = False
            missing_or_invalid.append("p69_p68_contract_not_validated")
    except Exception as e:
        p69_val = False
        missing_or_invalid.append(f"p69_probe_raised_exception_{str(e)}")
        
    # 2. P70A deep checks
    p70a_val = True
    try:
        p70a_res = run_p70a_pure_numeric_relation_testbed_probe()
        if p70a_res.get("phase") != "P70A" or p70a_res.get("verdict") != "P70A_READY_FOR_REVIEW":
            p70a_val = False
            missing_or_invalid.append("p70a_invalid_phase_or_verdict")
            
        r_types = p70a_res.get("relation_types", [])
        expected_r_types = ["translate_x", "translate_y", "scale_s", "reflect_x", "nonlinear_x_from_y"]
        for t in expected_r_types:
            if t not in r_types:
                p70a_val = False
                missing_or_invalid.append(f"p70a_missing_relation_type_{t}")
                
        sanity = p70a_res.get("sanity_summary", {})
        if sanity.get("all_relation_types_have_repeated_instances") is not True:
            p70a_val = False
            missing_or_invalid.append("p70a_repeated_instances_failed")
        if sanity.get("heldout_base_records_present") is not True:
            p70a_val = False
            missing_or_invalid.append("p70a_heldout_base_records_missing")
        if sanity.get("heldout_magnitude_records_present") is not True:
            p70a_val = False
            missing_or_invalid.append("p70a_heldout_magnitude_records_missing")
        if sanity.get("negative_control_records_present") is not True:
            p70a_val = False
            missing_or_invalid.append("p70a_negative_controls_missing")
        if sanity.get("max_invariant_violation") != 0.0:
            p70a_val = False
            missing_or_invalid.append("p70a_max_invariant_violation_not_zero")
            
        if p70a_res.get("p69_contract_validated") is not True:
            p70a_val = False
            missing_or_invalid.append("p70a_p69_contract_not_validated")
        if p70a_res.get("torch_allowed") is not False or p70a_res.get("numpy_allowed") is not False:
            p70a_val = False
            missing_or_invalid.append("p70a_torch_or_numpy_allowed")
    except Exception as e:
        p70a_val = False
        missing_or_invalid.append(f"p70a_probe_raised_exception_{str(e)}")
        
    # 3. P70B deep checks
    p70b_val = True
    try:
        p70b_res = run_p70b_synthetic_time_series_relation_testbed_probe()
        if p70b_res.get("phase") != "P70B" or p70b_res.get("verdict") != "P70B_READY_FOR_REVIEW":
            p70b_val = False
            missing_or_invalid.append("p70b_invalid_phase_or_verdict")
            
        r_types_b = p70b_res.get("relation_types", [])
        expected_r_types_b = ["change_frequency", "scale_amplitude", "shift_phase", "scale_volatility_envelope", "shift_trend"]
        for t in expected_r_types_b:
            if t not in r_types_b:
                p70b_val = False
                missing_or_invalid.append(f"p70b_missing_relation_type_{t}")
                
        sanity_b = p70b_res.get("sanity_summary", {})
        if sanity_b.get("all_relation_types_have_repeated_instances") is not True:
            p70b_val = False
            missing_or_invalid.append("p70b_repeated_instances_failed")
        if sanity_b.get("heldout_base_records_present") is not True:
            p70b_val = False
            missing_or_invalid.append("p70b_heldout_base_records_missing")
        if sanity_b.get("heldout_magnitude_records_present") is not True:
            p70b_val = False
            missing_or_invalid.append("p70b_heldout_magnitude_records_missing")
        if sanity_b.get("negative_controls_nondegenerate") is not True:
            p70b_val = False
            missing_or_invalid.append("p70b_negative_controls_degenerate")
        if sanity_b.get("negative_control_endpoint_collision_count") != 0:
            p70b_val = False
            missing_or_invalid.append("p70b_endpoint_collisions_found")
        if sanity_b.get("max_parameter_invariant_violation") != 0.0:
            p70b_val = False
            missing_or_invalid.append("p70b_max_invariant_violation_not_zero")
            
        if p70b_res.get("p70a_contract_validated") is not True:
            p70b_val = False
            missing_or_invalid.append("p70b_p70a_contract_not_validated")
        if p70b_res.get("torch_allowed") is not False or p70b_res.get("numpy_allowed") is not False:
            p70b_val = False
            missing_or_invalid.append("p70b_torch_or_numpy_allowed")
    except Exception as e:
        p70b_val = False
        missing_or_invalid.append(f"p70b_probe_raised_exception_{str(e)}")
        
    # 4. P71 deep checks
    p71_val = True
    try:
        p71_res = run_p71_relation_contrastive_signal_smoke_probe()
        if p71_res.get("phase") != "P71" or p71_res.get("verdict") != "P71_READY_FOR_REVIEW":
            p71_val = False
            missing_or_invalid.append("p71_invalid_phase_or_verdict")
            
        views = p71_res.get("descriptor_views", [])
        for v in ["p70a_vector_delta_descriptor", "p70b_parameter_delta_descriptor", "p70b_series_summary_delta_descriptor"]:
            if v not in views:
                p71_val = False
                missing_or_invalid.append(f"p71_missing_descriptor_view_{v}")
                
        if p71_res.get("source_contracts_validated") is not True:
            p71_val = False
            missing_or_invalid.append("p71_source_contracts_not_validated")
            
        sanity_p71 = p71_res.get("sanity_summary", {})
        if sanity_p71.get("p70b_parameter_descriptor_strong_pass") is not True:
            p71_val = False
            missing_or_invalid.append("p71_p70b_parameter_descriptor_strong_pass_failed")
        if sanity_p71.get("labels_used_for_descriptor_construction") is not False:
            p71_val = False
            missing_or_invalid.append("p71_labels_used_for_descriptor_construction_must_be_false")
        if sanity_p71.get("labels_used_for_evaluation_only") is not True:
            p71_val = False
            missing_or_invalid.append("p71_labels_used_for_evaluation_only_must_be_true")
            
        for flag in [
            "training_allowed", "model_implementation_allowed", "neural_encoder_implementation_allowed",
            "optimization_allowed", "torch_allowed", "numpy_allowed", "bridge_implementation_allowed",
            "learned_metric_allowed"
        ]:
            if p71_res.get(flag) is not False:
                p71_val = False
                missing_or_invalid.append(f"p71_{flag}_must_be_false")
    except Exception as e:
        p71_val = False
        missing_or_invalid.append(f"p71_probe_raised_exception_{str(e)}")
        
    deep_validated = p69_val and p70a_val and p70b_val and p71_val
    
    return {
        "source_contracts_deep_validated": deep_validated,
        "p69_validated": p69_val,
        "p70a_validated": p70a_val,
        "p70b_validated": p70b_val,
        "p71_validated": p71_val,
        "p71_validation_depth_note_addressed": True,
        "missing_or_invalid": missing_or_invalid,
    }


def run_p72_oracle_sparse_operator_bank_mvp_probe() -> dict:
    # 1. Deep source contract validation
    deep_val = validate_source_contracts_deep()
    contracts_valid = deep_val["source_contracts_deep_validated"]
    
    # 2. Build operator bank
    bank = build_oracle_sparse_operator_bank()
    
    # 3. Load probes
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    # 4. Evaluate P70A and P70B
    eval_p70a = evaluate_p70a_sparse_operator_bank(p70a_probe)
    eval_p70b = evaluate_p70b_sparse_operator_bank(p70b_probe)
    
    # Formulate verdict
    verdict_str = VERDICT if contracts_valid else "P72_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        # Check oracle exactness: L2 mean errors and invariant violations must be exactly zero (~1e-12)
        oracle_ok = (
            eval_p70a["oracle_endpoint_l2_error"]["mean"] < 1e-12
            and eval_p70a["oracle_invariant_violation"]["max"] < 1e-12
            and eval_p70b["oracle_parameter_l2_error"]["mean"] < 1e-12
            and eval_p70b["oracle_series_l2_error"]["mean"] < 1e-12
            and eval_p70b["oracle_invariant_violation"]["max"] < 1e-12
        )
        if not oracle_ok:
            verdict_str = "P72_BLOCKED_BY_OPERATOR_APPLICATION_SANITY"
            
    sanity_summary = {
        "source_contracts_deep_validated": contracts_valid,
        "p71_validation_depth_note_addressed": True,
        "operator_bank_present": True,
        "operator_bank_learned": False,
        "oracle_selection_used": True,
        "learned_selection_used": False,
        
        "p70a_operator_diagnostic_pass": eval_p70a["diagnostic_pass"],
        "p70b_operator_diagnostic_pass": eval_p70b["diagnostic_pass"],
        
        "p70a_oracle_endpoint_l2_mean_zero": eval_p70a["oracle_endpoint_l2_error"]["mean"] < 1e-12,
        "p70a_oracle_invariant_violation_max_zero": eval_p70a["oracle_invariant_violation"]["max"] < 1e-12,
        "p70b_oracle_parameter_l2_mean_zero": eval_p70b["oracle_parameter_l2_error"]["mean"] < 1e-12,
        "p70b_oracle_series_l2_mean_zero": eval_p70b["oracle_series_l2_error"]["mean"] < 1e-12,
        "p70b_oracle_invariant_violation_max_zero": eval_p70b["oracle_invariant_violation"]["max"] < 1e-12,
        
        "heldout_base_evaluated": eval_p70a["heldout_base_record_count"] > 0 and eval_p70b["heldout_base_record_count"] > 0,
        "heldout_magnitude_evaluated": eval_p70a["heldout_magnitude_record_count"] > 0 and eval_p70b["heldout_magnitude_record_count"] > 0,
        
        "oracle_beats_no_relation_baseline": eval_p70a["oracle_beats_no_relation_mean_l2"] and eval_p70b["oracle_beats_no_relation_mean_l2"],
        
        "training_or_model_added": False,
        "json_safe": True,
    }
    
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,
        
        "source_baseline_phase": SOURCE_BASELINE_PHASE,
        "source_vector_testbed_phase": SOURCE_VECTOR_TESTBED_PHASE,
        "source_time_series_testbed_phase": SOURCE_TIME_SERIES_TESTBED_PHASE,
        "source_contrastive_phase": SOURCE_CONTRASTIVE_PHASE,
        
        "verdict": verdict_str,
        
        "training_allowed": TRAINING_ALLOWED,
        "model_implementation_allowed": MODEL_IMPLEMENTATION_ALLOWED,
        "neural_encoder_implementation_allowed": NEURAL_ENCODER_IMPLEMENTATION_ALLOWED,
        "neural_operator_selector_allowed": NEURAL_OPERATOR_SELECTOR_ALLOWED,
        "optimization_allowed": OPTIMIZATION_ALLOWED,
        "torch_allowed": TORCH_ALLOWED,
        "numpy_allowed": NUMPY_ALLOWED,
        "stochastic_random_allowed": STOCHASTIC_RANDOM_ALLOWED,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "learned_metric_allowed": LEARNED_METRIC_ALLOWED,
        
        "oracle_operator_bank_allowed": ORACLE_OPERATOR_BANK_ALLOWED,
        "oracle_relation_type_selection_allowed": ORACLE_RELATION_TYPE_SELECTION_ALLOWED,
        "learned_relation_type_selection_allowed": LEARNED_RELATION_TYPE_SELECTION_ALLOWED,
        "sparse_operator_application_allowed": SPARSE_OPERATOR_APPLICATION_ALLOWED,
        "operator_bank_training_allowed": OPERATOR_BANK_TRAINING_ALLOWED,
        
        "primary_empirical_target": PRIMARY_EMPIRICAL_TARGET,
        "operator_domains": OPERATOR_DOMAINS,
        "p70a_vector_operator_types": P70A_VECTOR_OPERATOR_TYPES,
        "p70b_time_series_operator_types": P70B_TIME_SERIES_OPERATOR_TYPES,
        "evaluation_splits": EVALUATION_SPLITS,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "allowed_claims": ALLOWED_CLAIMS,
        
        "source_contracts_deep_validated": contracts_valid,
        "p71_validation_depth_note_addressed": True,
        
        "operator_bank": bank,
        "p70a_operator_evaluation": eval_p70a,
        "p70b_operator_evaluation": eval_p70b,
        
        "sanity_summary": sanity_summary,
        "json_safe": True,
        "diagnostic_only": True
    }
    
    # Assert JSON safe
    json.dumps(output)
    
    return output
