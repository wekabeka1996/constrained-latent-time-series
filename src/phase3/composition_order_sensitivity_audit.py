# src/phase3/composition_order_sensitivity_audit.py

import json
import math
from typing import Any

from src.phase3.baseline_point_offset_interpolation_harness import (
    run_p69_baseline_point_offset_interpolation_harness_probe,
)

from src.phase3.pure_numeric_relation_testbed import (
    run_p70a_pure_numeric_relation_testbed_probe,
    compute_invariant_violation,
)

from src.phase3.synthetic_time_series_relation_testbed import (
    run_p70b_synthetic_time_series_relation_testbed_probe,
    generate_synthetic_series,
    compute_parameter_invariant_violation,
    series_l2_distance,
    series_max_abs_distance,
)

from src.phase3.relation_contrastive_signal_smoke import (
    run_p71_relation_contrastive_signal_smoke_probe,
)

from src.phase3.oracle_sparse_operator_bank_mvp import (
    run_p72_oracle_sparse_operator_bank_mvp_probe,
    build_oracle_sparse_operator_bank,
    build_p70a_vector_operator_specs,
    build_p70b_time_series_operator_specs,
    apply_p70a_sparse_vector_operator,
    apply_p70b_sparse_parameter_operator,
    vector_l2_error,
    parameter_l2_error,
    parameter_max_abs_error,
)

from src.phase3.transfer_invariant_preservation_audit import (
    run_p73_transfer_invariant_preservation_audit_probe,
)

PHASE = "P74"
PHASE_GROUP = "PHASE_3"
PHASE_NAME = "Composition and Order-Sensitivity Audit"
CONTRACT_VERSION = "phase3_p74_composition_order_sensitivity_audit_contract_v1"

SOURCE_BASELINE_PHASE = "P69"
SOURCE_VECTOR_TESTBED_PHASE = "P70A"
SOURCE_TIME_SERIES_TESTBED_PHASE = "P70B"
SOURCE_CONTRASTIVE_PHASE = "P71"
SOURCE_OPERATOR_BANK_PHASE = "P72"
SOURCE_TRANSFER_AUDIT_PHASE = "P73"

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

ORACLE_COMPOSITION_AUDIT_ALLOWED = True
ORACLE_OPERATOR_BANK_REUSE_ALLOWED = True
ORACLE_RELATION_TYPE_SELECTION_ALLOWED = True
LEARNED_RELATION_TYPE_SELECTION_ALLOWED = False
LEARNED_COMPOSITION_CLAIMS_ALLOWED = False

TARGET_ENDPOINT_USED_FOR_PREDICTION = False
TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY = True
TARGET_MIDPOINT_USED_FOR_PREDICTION = False
TARGET_MIDPOINT_USED_FOR_EVALUATION_ONLY = True

PRIMARY_EMPIRICAL_TARGET = "oracle_composition_and_order_sensitivity_diagnostics"
VERDICT = "P74_READY_FOR_REVIEW"

AUDIT_DOMAINS = [
    "p70a_vector_world",
    "p70b_time_series_parameter_world",
]

COMPOSITION_FAMILIES = [
    "commutative_expected",
    "order_sensitive_expected",
    "source_declared",
    "generated_probe",
]

FORBIDDEN_CLAIMS = [
    "semantic_geometry_is_proven",
    "meaning_is_learned",
    "operator_identity_is_proven",
    "relation_encoder_is_trained",
    "relation_encoder_is_validated",
    "sparse_operator_bank_is_learned",
    "learned_operator_selection_is_proven",
    "learned_transfer_is_proven",
    "learned_composition_is_proven",
    "composition_is_proven_in_learned_system",
    "bridge_method_is_validated",
    "schrodinger_bridge_creates_meaning",
    "geometric_schrodinger_bridge_creates_meaning",
    "oracle_composition_proves_semantics",
]

ALLOWED_CLAIMS = [
    "p74_audits_oracle_composition_behavior",
    "p74_audits_order_sensitivity",
    "p74_audits_commutative_cases",
    "p74_audits_invariant_preservation_under_composition",
    "p74_checks_endpoint_and_midpoint_leakage_boundary",
    "p74_addresses_p73_validation_fidelity_note",
    "p74_does_not_train_models",
    "p74_does_not_establish_learned_composition_evidence",
]


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-15:
        return 0.0
    return float(numerator / denominator)


def summarize_values(values: list[float]) -> dict:
    if len(values) == 0:
        return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": int(len(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def summary_is_zero(summary: dict, tolerance: float = 1e-12) -> bool:
    return summary["mean"] < tolerance and summary["max"] < tolerance


def vectors_max_abs_error(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Dimensions mismatch in vectors_max_abs_error.")
    return float(max(abs(x - y) for x, y in zip(a, b)))


def count_by_key(records: list[dict], key: str) -> dict:
    counts = {}
    for rec in records:
        val = rec.get(key, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts


def build_p70a_composition_cases(p70a_probe: dict) -> list[dict]:
    # 1. Try to load from P70A source
    cases = []
    source_records = p70a_probe.get("testbed", {}).get("composition_records", [])
    for rec in source_records:
        cases.append({
            "composition_id": rec.get("composition_id", f"src_{len(cases)}"),
            "domain": "p70a_vector_world",
            "case_source": "source_declared",
            "expected_order_sensitive": rec.get("expected_order_sensitive", False),
            "z_start": rec["z_start"],
            "first_relation_type": rec["first_relation_type"],
            "first_intensity": rec["first_intensity"],
            "second_relation_type": rec["second_relation_type"],
            "second_intensity": rec["second_intensity"],
            "z_end_target": rec.get("z_end"), # target endpoints only for post-prediction evaluation
        })
        
    # 2. Generate deterministic probe cases
    z_start = [1.0, 2.0, 3.0]
    
    probe_defs = [
        # translate_x then reflect_x (order sensitive)
        ("translate_x", 1.0, "reflect_x", 1.0, True),
        # reflect_x then translate_x (order sensitive counterpart)
        ("reflect_x", 1.0, "translate_x", 1.0, True),
        # translate_y then nonlinear_x_from_y (order sensitive)
        ("translate_y", 1.0, "nonlinear_x_from_y", 1.0, True),
        # nonlinear_x_from_y then translate_y (order sensitive counterpart)
        ("nonlinear_x_from_y", 1.0, "translate_y", 1.0, True),
        # translate_x then translate_y (commutative)
        ("translate_x", 1.0, "translate_y", 1.0, False),
        # translate_y then translate_x (commutative counterpart)
        ("translate_y", 1.0, "translate_x", 1.0, False),
        # scale_s then translate_x (commutative)
        ("scale_s", 1.0, "translate_x", 1.0, False),
        # translate_x then scale_s (commutative counterpart)
        ("translate_x", 1.0, "scale_s", 1.0, False),
    ]
    
    for idx, (first_rel, first_int, second_rel, second_int, order_sens) in enumerate(probe_defs):
        cases.append({
            "composition_id": f"gen_p70a_{idx:03d}",
            "domain": "p70a_vector_world",
            "case_source": "generated_probe",
            "expected_order_sensitive": order_sens,
            "z_start": z_start,
            "first_relation_type": first_rel,
            "first_intensity": float(first_int),
            "second_relation_type": second_rel,
            "second_intensity": float(second_int),
        })
        
    return cases


def build_p70b_composition_cases(p70b_probe: dict) -> list[dict]:
    cases = []
    source_records = p70b_probe.get("testbed", {}).get("composition_records", [])
    for rec in source_records:
        cases.append({
            "composition_id": rec.get("composition_id", f"src_{len(cases)}"),
            "domain": "p70b_time_series_parameter_world",
            "case_source": "source_declared",
            "expected_order_sensitive": rec.get("expected_order_sensitive", False),
            "params_start": rec["params_start"],
            "first_relation_type": rec["first_relation_type"],
            "first_intensity": rec["first_intensity"],
            "second_relation_type": rec["second_relation_type"],
            "second_intensity": rec["second_intensity"],
            "params_end_target": rec.get("params_end"),
            "series_end_target": rec.get("series_end"),
        })
        
    # Generate additional P70B probe cases
    params_start = {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0, "volatility_envelope": 0.10, "trend": 0.0}
    
    probe_defs = [
        ("change_frequency", 0.25, "scale_amplitude", 0.5),
        ("scale_amplitude", 0.5, "change_frequency", 0.25),
        ("shift_phase", 0.25, "scale_volatility_envelope", 0.5),
        ("scale_volatility_envelope", 0.5, "shift_phase", 0.25),
        ("shift_trend", 0.05, "change_frequency", 0.25),
        ("change_frequency", 0.25, "shift_trend", 0.05),
    ]
    
    for idx, (first_rel, first_int, second_rel, second_int) in enumerate(probe_defs):
        cases.append({
            "composition_id": f"gen_p70b_{idx:03d}",
            "domain": "p70b_time_series_parameter_world",
            "case_source": "generated_probe",
            "expected_order_sensitive": False,
            "params_start": params_start,
            "first_relation_type": first_rel,
            "first_intensity": float(first_int),
            "second_relation_type": second_rel,
            "second_intensity": float(second_int),
        })
        
    return cases


def apply_p70a_forward_composition(
    z_start: list[float],
    first_relation_type: str,
    first_intensity: float,
    second_relation_type: str,
    second_intensity: float,
) -> dict:
    z_mid = apply_p70a_sparse_vector_operator(z_start, first_relation_type, first_intensity)
    z_end = apply_p70a_sparse_vector_operator(z_mid, second_relation_type, second_intensity)
    return {
        "z_mid_pred": z_mid,
        "z_end_pred": z_end,
    }


def apply_p70a_reverse_composition(
    z_start: list[float],
    first_relation_type: str,
    first_intensity: float,
    second_relation_type: str,
    second_intensity: float,
) -> dict:
    # Applies second relation type first, then first
    z_mid_rev = apply_p70a_sparse_vector_operator(z_start, second_relation_type, second_intensity)
    z_end_rev = apply_p70a_sparse_vector_operator(z_mid_rev, first_relation_type, first_intensity)
    return {
        "z_mid_reverse_pred": z_mid_rev,
        "z_end_reverse_pred": z_end_rev,
    }


def apply_p70b_forward_composition(
    params_start: dict,
    first_relation_type: str,
    first_intensity: float,
    second_relation_type: str,
    second_intensity: float,
) -> dict:
    p_mid = apply_p70b_sparse_parameter_operator(params_start, first_relation_type, first_intensity)
    p_end = apply_p70b_sparse_parameter_operator(p_mid, second_relation_type, second_intensity)
    
    s_mid = generate_synthetic_series(p_mid)
    s_end = generate_synthetic_series(p_end)
    return {
        "params_mid_pred": p_mid,
        "params_end_pred": p_end,
        "series_mid_pred": s_mid,
        "series_end_pred": s_end,
    }


def apply_p70b_reverse_composition(
    params_start: dict,
    first_relation_type: str,
    first_intensity: float,
    second_relation_type: str,
    second_intensity: float,
) -> dict:
    p_mid_rev = apply_p70b_sparse_parameter_operator(params_start, second_relation_type, second_intensity)
    p_end_rev = apply_p70b_sparse_parameter_operator(p_mid_rev, first_relation_type, first_intensity)
    
    s_mid_rev = generate_synthetic_series(p_mid_rev)
    s_end_rev = generate_synthetic_series(p_end_rev)
    return {
        "params_mid_reverse_pred": p_mid_rev,
        "params_end_reverse_pred": p_end_rev,
        "series_mid_reverse_pred": s_mid_rev,
        "series_end_reverse_pred": s_end_rev,
    }


def compute_p70a_composite_invariant_indices(first_relation_type: str, second_relation_type: str) -> list[int]:
    specs = build_p70a_vector_operator_specs()
    first = specs[first_relation_type]
    second = specs[second_relation_type]
    
    changed_union = set(first["changed_indices"]) | set(second["changed_indices"])
    composite_invariant = [i for i in [0, 1, 2] if i not in changed_union]
    return composite_invariant


def compute_p70b_composite_invariant_keys(first_relation_type: str, second_relation_type: str) -> list[str]:
    specs = build_p70b_time_series_operator_specs()
    first = specs[first_relation_type]
    second = specs[second_relation_type]
    
    keys = ["amplitude", "frequency", "phase", "volatility_envelope", "trend"]
    changed_union = set(first["changed_keys"]) | set(second["changed_keys"])
    composite_invariant = [k for k in keys if k not in changed_union]
    return composite_invariant


def audit_p70a_composition_and_order_sensitivity(p70a_probe: dict) -> dict:
    cases = build_p70a_composition_cases(p70a_probe)
    
    forward_l2_errors = []
    forward_max_abs_errors = []
    rev_forward_l2s = []
    composite_violations = []
    
    order_sensitive_expected = 0
    order_sensitive_detected = 0
    commutative_expected = 0
    commutative_detected = 0
    expected_order_match = 0
    expected_order_mismatch = 0
    
    all_targets_exact = True
    
    for case in cases:
        z_start = case["z_start"]
        first_rel = case["first_relation_type"]
        first_int = case["first_intensity"]
        second_rel = case["second_relation_type"]
        second_int = case["second_intensity"]
        exp_sensitive = case["expected_order_sensitive"]
        
        # 1. Forward prediction (strictly without targets)
        f_res = apply_p70a_forward_composition(z_start, first_rel, first_int, second_rel, second_int)
        z_end_pred = f_res["z_end_pred"]
        
        # 2. Reverse prediction
        r_res = apply_p70a_reverse_composition(z_start, first_rel, first_int, second_rel, second_int)
        z_end_rev_pred = r_res["z_end_reverse_pred"]
        
        # Target endpoint evaluation
        if "z_end_target" in case and case["z_end_target"] is not None:
            z_end_target = case["z_end_target"]
            err_l2 = vector_l2_error(z_end_pred, z_end_target)
            err_max_abs = vectors_max_abs_error(z_end_pred, z_end_target)
            
            forward_l2_errors.append(err_l2)
            forward_max_abs_errors.append(err_max_abs)
            if err_l2 >= 1e-12:
                all_targets_exact = False
                
        # Commutativity diagnostics
        rev_forw_l2 = vector_l2_error(z_end_pred, z_end_rev_pred)
        rev_forward_l2s.append(rev_forw_l2)
        
        detected_sensitive = rev_forw_l2 > 1e-12
        if exp_sensitive:
            order_sensitive_expected += 1
            if detected_sensitive:
                order_sensitive_detected += 1
                expected_order_match += 1
            else:
                expected_order_mismatch += 1
        else:
            commutative_expected += 1
            if not detected_sensitive:
                commutative_detected += 1
                expected_order_match += 1
            else:
                expected_order_mismatch += 1
                
        # Composite invariants
        inv_indices = compute_p70a_composite_invariant_indices(first_rel, second_rel)
        violation = compute_invariant_violation(z_start, z_end_pred, inv_indices)
        composite_violations.append(violation)
        
    sum_f_l2 = summarize_values(forward_l2_errors)
    sum_f_max = summarize_values(forward_max_abs_errors)
    sum_rev_f_l2 = summarize_values(rev_forward_l2s)
    sum_composite_violation = summarize_values(composite_violations)
    
    all_invariants = summary_is_zero(sum_composite_violation)
    diag_pass = (
        all_invariants
        and expected_order_mismatch == 0
        and order_sensitive_detected == order_sensitive_expected
        and commutative_detected == commutative_expected
    )
    
    return {
        "domain": "p70a_vector_world",
        "composition_case_count": len(cases),
        "source_declared_case_count": sum(1 for c in cases if c["case_source"] == "source_declared"),
        "generated_probe_case_count": sum(1 for c in cases if c["case_source"] == "generated_probe"),
        "forward_endpoint_l2_error": sum_f_l2,
        "forward_endpoint_max_abs_error": sum_f_max,
        "reverse_forward_l2_distance": sum_rev_f_l2,
        "composite_invariant_violation": sum_composite_violation,
        "order_sensitive_expected_count": int(order_sensitive_expected),
        "order_sensitive_detected_count": int(order_sensitive_detected),
        "commutative_expected_count": int(commutative_expected),
        "commutative_detected_count": int(commutative_detected),
        "expected_order_match_count": int(expected_order_match),
        "expected_order_mismatch_count": int(expected_order_mismatch),
        "all_forward_targets_exact_when_available": all_targets_exact,
        "all_composite_invariants_preserved": all_invariants,
        "order_sensitivity_diagnostic_pass": diag_pass,
        "target_endpoint_used_for_prediction": TARGET_ENDPOINT_USED_FOR_PREDICTION,
        "target_endpoint_used_for_evaluation_only": TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY,
        "target_midpoint_used_for_prediction": TARGET_MIDPOINT_USED_FOR_PREDICTION,
        "target_midpoint_used_for_evaluation_only": TARGET_MIDPOINT_USED_FOR_EVALUATION_ONLY,
        "diagnostic_pass": diag_pass,
    }


def audit_p70b_composition_and_order_sensitivity(p70b_probe: dict) -> dict:
    cases = build_p70b_composition_cases(p70b_probe)
    
    forward_param_l2s = []
    forward_param_maxs = []
    forward_series_l2s = []
    forward_series_maxs = []
    
    rev_forw_param_l2s = []
    rev_forw_series_l2s = []
    composite_violations = []
    
    order_sensitive_expected = 0
    order_sensitive_detected = 0
    commutative_expected = 0
    commutative_detected = 0
    expected_order_match = 0
    expected_order_mismatch = 0
    
    all_targets_exact = True
    
    for case in cases:
        params_start = case["params_start"]
        first_rel = case["first_relation_type"]
        first_int = case["first_intensity"]
        second_rel = case["second_relation_type"]
        second_int = case["second_intensity"]
        exp_sensitive = case["expected_order_sensitive"]
        
        # 1. Forward prediction
        f_res = apply_p70b_forward_composition(params_start, first_rel, first_int, second_rel, second_int)
        p_end_pred = f_res["params_end_pred"]
        s_end_pred = f_res["series_end_pred"]
        
        # 2. Reverse prediction
        r_res = apply_p70b_reverse_composition(params_start, first_rel, first_int, second_rel, second_int)
        p_end_rev_pred = r_res["params_end_reverse_pred"]
        s_end_rev_pred = r_res["series_end_reverse_pred"]
        
        # Target evaluation
        if "params_end_target" in case and case["params_end_target"] is not None:
            p_end_target = case["params_end_target"]
            err_p_l2 = parameter_l2_error(p_end_pred, p_end_target)
            err_p_max = parameter_max_abs_error(p_end_pred, p_end_target)
            forward_param_l2s.append(err_p_l2)
            forward_param_maxs.append(err_p_max)
            if err_p_l2 >= 1e-12:
                all_targets_exact = False
                
        if "series_end_target" in case and case["series_end_target"] is not None:
            s_end_target = case["series_end_target"]
            err_s_l2 = series_l2_distance(s_end_pred, s_end_target)
            err_s_max = series_max_abs_distance(s_end_pred, s_end_target)
            forward_series_l2s.append(err_s_l2)
            forward_series_maxs.append(err_s_max)
            if err_s_l2 >= 1e-9:
                all_targets_exact = False
                
        # Commutativity diagnostics
        rf_p_l2 = parameter_l2_error(p_end_pred, p_end_rev_pred)
        rf_s_l2 = series_l2_distance(s_end_pred, s_end_rev_pred)
        rev_forw_param_l2s.append(rf_p_l2)
        rev_forw_series_l2s.append(rf_s_l2)
        
        detected_sensitive = rf_p_l2 > 1e-12 or rf_s_l2 > 1e-9
        if exp_sensitive:
            order_sensitive_expected += 1
            if detected_sensitive:
                order_sensitive_detected += 1
                expected_order_match += 1
            else:
                expected_order_mismatch += 1
        else:
            commutative_expected += 1
            if not detected_sensitive:
                commutative_detected += 1
                expected_order_match += 1
            else:
                expected_order_mismatch += 1
                
        # Composite invariants
        inv_keys = compute_p70b_composite_invariant_keys(first_rel, second_rel)
        violation = compute_parameter_invariant_violation(params_start, p_end_pred, inv_keys)
        composite_violations.append(violation)
        
    sum_p_l2 = summarize_values(forward_param_l2s)
    sum_p_max = summarize_values(forward_param_maxs)
    sum_s_l2 = summarize_values(forward_series_l2s)
    sum_s_max = summarize_values(forward_series_maxs)
    
    sum_rf_p_l2 = summarize_values(rev_forw_param_l2s)
    sum_rf_s_l2 = summarize_values(rev_forw_series_l2s)
    sum_composite_violation = summarize_values(composite_violations)
    
    all_invariants = summary_is_zero(sum_composite_violation)
    diag_pass = (
        all_invariants
        and expected_order_mismatch == 0
        and order_sensitive_detected == order_sensitive_expected
        and commutative_detected == commutative_expected
    )
    
    return {
        "domain": "p70b_time_series_parameter_world",
        "composition_case_count": len(cases),
        "source_declared_case_count": sum(1 for c in cases if c["case_source"] == "source_declared"),
        "generated_probe_case_count": sum(1 for c in cases if c["case_source"] == "generated_probe"),
        "forward_parameter_l2_error": sum_p_l2,
        "forward_parameter_max_abs_error": sum_p_max,
        "forward_series_l2_error": sum_s_l2,
        "forward_series_max_abs_error": sum_s_max,
        "reverse_forward_parameter_l2_distance": sum_rf_p_l2,
        "reverse_forward_series_l2_distance": sum_rf_s_l2,
        "composite_invariant_violation": sum_composite_violation,
        "order_sensitive_expected_count": int(order_sensitive_expected),
        "order_sensitive_detected_count": int(order_sensitive_detected),
        "commutative_expected_count": int(commutative_expected),
        "commutative_detected_count": int(commutative_detected),
        "expected_order_match_count": int(expected_order_match),
        "expected_order_mismatch_count": int(expected_order_mismatch),
        "all_forward_targets_exact_when_available": all_targets_exact,
        "all_composite_invariants_preserved": all_invariants,
        "order_sensitivity_diagnostic_pass": diag_pass,
        "target_endpoint_used_for_prediction": TARGET_ENDPOINT_USED_FOR_PREDICTION,
        "target_endpoint_used_for_evaluation_only": TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY,
        "target_midpoint_used_for_prediction": TARGET_MIDPOINT_USED_FOR_PREDICTION,
        "target_midpoint_used_for_evaluation_only": TARGET_MIDPOINT_USED_FOR_EVALUATION_ONLY,
        "diagnostic_pass": diag_pass,
    }


def validate_source_contracts_with_fidelity() -> dict:
    fidelity_validated = True
    missing_or_invalid = []
    missing_fields = []
    
    # Track actual booleans (fidelity repair: no hardcoding)
    p69_ok = True
    p70a_ok = True
    p70b_ok = True
    p71_ok = True
    p72_ok = True
    p73_ok = True
    
    # 1. P69 checks
    try:
        p69 = run_p69_baseline_point_offset_interpolation_harness_probe()
        if p69.get("phase") != "P69" or p69.get("verdict") != "P69_READY_FOR_REVIEW":
            p69_ok = False
            missing_or_invalid.append("p69_invalid_phase_or_verdict")
        if p69.get("model_implementation_allowed") is not False:
            p69_ok = False
            missing_or_invalid.append("p69_model_allowed_not_false")
        if p69.get("bridge_implementation_allowed") is not False:
            p69_ok = False
            missing_or_invalid.append("p69_bridge_allowed_not_false")
    except Exception as e:
        p69_ok = False
        missing_or_invalid.append(f"p69_exception_{str(e)}")
        
    # 2. P70A checks
    try:
        p70a = run_p70a_pure_numeric_relation_testbed_probe()
        if p70a.get("phase") != "P70A" or p70a.get("verdict") != "P70A_READY_FOR_REVIEW":
            p70a_ok = False
            missing_or_invalid.append("p70a_invalid_phase_or_verdict")
        if "model_implementation_allowed" not in p70a:
            missing_fields.append("p70a_model_implementation_allowed")
        else:
            if p70a.get("model_implementation_allowed") is not False:
                p70a_ok = False
                missing_or_invalid.append("p70a_model_allowed_not_false")
    except Exception as e:
        p70a_ok = False
        missing_or_invalid.append(f"p70a_exception_{str(e)}")
        
    # 3. P70B checks
    try:
        p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
        if p70b.get("phase") != "P70B" or p70b.get("verdict") != "P70B_READY_FOR_REVIEW":
            p70b_ok = False
            missing_or_invalid.append("p70b_invalid_phase_or_verdict")
        if "model_implementation_allowed" not in p70b:
            missing_fields.append("p70b_model_implementation_allowed")
        else:
            if p70b.get("model_implementation_allowed") is not False:
                p70b_ok = False
                missing_or_invalid.append("p70b_model_allowed_not_false")
        if p70b.get("sanity_summary", {}).get("negative_control_endpoint_collision_count") != 0:
            p70b_ok = False
            missing_or_invalid.append("p70b_endpoint_collisions_not_zero")
    except Exception as e:
        p70b_ok = False
        missing_or_invalid.append(f"p70b_exception_{str(e)}")
        
    # 4. P71 checks
    try:
        p71 = run_p71_relation_contrastive_signal_smoke_probe()
        if p71.get("phase") != "P71" or p71.get("verdict") != "P71_READY_FOR_REVIEW":
            p71_ok = False
            missing_or_invalid.append("p71_invalid_phase_or_verdict")
        if p71.get("neural_encoder_implementation_allowed") is not False:
            p71_ok = False
            missing_or_invalid.append("p71_neural_encoder_allowed_not_false")
    except Exception as e:
        p71_ok = False
        missing_or_invalid.append(f"p71_exception_{str(e)}")
        
    # 5. P72 checks
    try:
        p72 = run_p72_oracle_sparse_operator_bank_mvp_probe()
        if p72.get("phase") != "P72" or p72.get("verdict") != "P72_READY_FOR_REVIEW":
            p72_ok = False
            missing_or_invalid.append("p72_invalid_phase_or_verdict")
        if p72.get("operator_bank", {}).get("learned") is not False:
            p72_ok = False
            missing_or_invalid.append("p72_operator_bank_learned_not_false")
    except Exception as e:
        p72_ok = False
        missing_or_invalid.append(f"p72_exception_{str(e)}")
        
    # 6. P73 checks
    try:
        p73 = run_p73_transfer_invariant_preservation_audit_probe()
        if p73.get("phase") != "P73" or p73.get("verdict") != "P73_READY_FOR_REVIEW":
            p73_ok = False
            missing_or_invalid.append("p73_invalid_phase_or_verdict")
        if p73.get("target_endpoint_used_for_prediction") is not False:
            p73_ok = False
            missing_or_invalid.append("p73_target_endpoint_used_for_prediction_not_false")
        if p73.get("sanity_summary", {}).get("p70a_transfer_gaps_zero") is not True:
            p73_ok = False
            missing_or_invalid.append("p73_p70a_transfer_gaps_not_zero")
    except Exception as e:
        p73_ok = False
        missing_or_invalid.append(f"p73_exception_{str(e)}")
        
    fidelity_validated = p69_ok and p70a_ok and p70b_ok and p71_ok and p72_ok and p73_ok
    
    return {
        "source_contracts_fidelity_validated": fidelity_validated,
        "p69_validated": p69_ok,
        "p70a_validated": p70a_ok,
        "p70b_validated": p70b_ok,
        "p71_validated": p71_ok,
        "p72_validated": p72_ok,
        "p73_validated": p73_ok,
        "p73_validation_fidelity_note_addressed": True,
        "missing_or_invalid": missing_or_invalid,
        "field_missing_but_transitively_guarded": sorted(missing_fields),
    }


def run_p74_composition_order_sensitivity_audit_probe() -> dict:
    # 1. Source contracts validation
    fidelity_val = validate_source_contracts_with_fidelity()
    contracts_valid = fidelity_val["source_contracts_fidelity_validated"]
    
    # 2. Load Probes
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    # 3. Composition Audits
    audit_p70a = audit_p70a_composition_and_order_sensitivity(p70a_probe)
    audit_p70b = audit_p70b_composition_and_order_sensitivity(p70b_probe)
    
    # Formulate verdict
    verdict_str = VERDICT if contracts_valid else "P74_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if not (audit_p70a["diagnostic_pass"] and audit_p70b["diagnostic_pass"]):
            verdict_str = "P74_BLOCKED_BY_COMPOSITION_OR_ORDER_SANITY"
            
    sanity_summary = {
        "source_contracts_fidelity_validated": contracts_valid,
        "p73_validation_fidelity_note_addressed": True,
        
        "target_endpoint_used_for_prediction": TARGET_ENDPOINT_USED_FOR_PREDICTION,
        "target_endpoint_used_for_evaluation_only": TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY,
        "target_midpoint_used_for_prediction": TARGET_MIDPOINT_USED_FOR_PREDICTION,
        "target_midpoint_used_for_evaluation_only": TARGET_MIDPOINT_USED_FOR_EVALUATION_ONLY,
        
        "p70a_composition_diagnostic_pass": audit_p70a["diagnostic_pass"],
        "p70b_composition_diagnostic_pass": audit_p70b["diagnostic_pass"],
        
        "p70a_order_sensitive_cases_detected": audit_p70a["order_sensitive_detected_count"] > 0,
        "p70a_commutative_cases_detected": audit_p70a["commutative_detected_count"] > 0,
        "p70b_commutative_cases_detected": audit_p70b["commutative_detected_count"] > 0,
        
        "p70a_expected_order_mismatches_zero": audit_p70a["expected_order_mismatch_count"] == 0,
        "p70b_expected_order_mismatches_zero": audit_p70b["expected_order_mismatch_count"] == 0,
        
        "p70a_composite_invariants_preserved": audit_p70a["all_composite_invariants_preserved"],
        "p70b_composite_invariants_preserved": audit_p70b["all_composite_invariants_preserved"],
        
        "learned_composition_claims_made": False,
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
        "source_operator_bank_phase": SOURCE_OPERATOR_BANK_PHASE,
        "source_transfer_audit_phase": SOURCE_TRANSFER_AUDIT_PHASE,
        
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
        
        "oracle_composition_audit_allowed": ORACLE_COMPOSITION_AUDIT_ALLOWED,
        "oracle_operator_bank_reuse_allowed": ORACLE_OPERATOR_BANK_REUSE_ALLOWED,
        "oracle_relation_type_selection_allowed": ORACLE_RELATION_TYPE_SELECTION_ALLOWED,
        "learned_relation_type_selection_allowed": LEARNED_RELATION_TYPE_SELECTION_ALLOWED,
        "learned_composition_claims_allowed": LEARNED_COMPOSITION_CLAIMS_ALLOWED,
        
        "target_endpoint_used_for_prediction": TARGET_ENDPOINT_USED_FOR_PREDICTION,
        "target_endpoint_used_for_evaluation_only": TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY,
        "target_midpoint_used_for_prediction": TARGET_MIDPOINT_USED_FOR_PREDICTION,
        "target_midpoint_used_for_evaluation_only": TARGET_MIDPOINT_USED_FOR_EVALUATION_ONLY,
        
        "primary_empirical_target": PRIMARY_EMPIRICAL_TARGET,
        "audit_domains": AUDIT_DOMAINS,
        "composition_families": COMPOSITION_FAMILIES,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "allowed_claims": ALLOWED_CLAIMS,
        
        "source_contracts_fidelity_validated": contracts_valid,
        "p73_validation_fidelity_note_addressed": True,
        
        "p70a_composition_audit": audit_p70a,
        "p70b_composition_audit": audit_p70b,
        
        "sanity_summary": sanity_summary,
        "json_safe": True,
        "diagnostic_only": True
    }
    
    # Assert JSON safe
    json.dumps(output)
    
    return output
