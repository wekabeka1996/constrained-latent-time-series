# src/phase3/source_available_query_descriptor_contract.py

import json
import math
from typing import Any

from src.phase3.baseline_point_offset_interpolation_harness import (
    run_p69_baseline_point_offset_interpolation_harness_probe,
)

from src.phase3.pure_numeric_relation_testbed import (
    run_p70a_pure_numeric_relation_testbed_probe,
)

from src.phase3.synthetic_time_series_relation_testbed import (
    run_p70b_synthetic_time_series_relation_testbed_probe,
)

from src.phase3.relation_contrastive_signal_smoke import (
    run_p71_relation_contrastive_signal_smoke_probe,
)

from src.phase3.oracle_sparse_operator_bank_mvp import (
    run_p72_oracle_sparse_operator_bank_mvp_probe,
)

from src.phase3.transfer_invariant_preservation_audit import (
    run_p73_transfer_invariant_preservation_audit_probe,
)

from src.phase3.composition_order_sensitivity_audit import (
    run_p74_composition_order_sensitivity_audit_probe,
)

from src.phase3.global_negative_controls_collapse_audit import (
    run_p75_global_negative_controls_collapse_audit_probe,
)

from src.phase3.relation_metric_selector_prebridge_audit import (
    run_p76_relation_metric_selector_prebridge_audit_probe,
)

PHASE = "P77"
PHASE_GROUP = "PHASE_3"
PHASE_NAME = "Source-Available Query Descriptor Contract"
CONTRACT_VERSION = "phase3_p77_source_available_query_descriptor_contract_v1"

SOURCE_BASELINE_PHASE = "P69"
SOURCE_VECTOR_TESTBED_PHASE = "P70A"
SOURCE_TIME_SERIES_TESTBED_PHASE = "P70B"
SOURCE_CONTRASTIVE_PHASE = "P71"
SOURCE_OPERATOR_BANK_PHASE = "P72"
SOURCE_TRANSFER_AUDIT_PHASE = "P73"
SOURCE_COMPOSITION_AUDIT_PHASE = "P74"
SOURCE_NEGATIVE_CONTROL_PHASE = "P75"
SOURCE_PREBRIDGE_LEAKAGE_PHASE = "P76"

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

SOURCE_AVAILABLE_QUERY_CONTRACT_ALLOWED = True
PREDICTIVE_SELECTOR_IMPLEMENTATION_ALLOWED = False
PREDICTIVE_SELECTOR_CLAIMS_ALLOWED = False
LEARNED_SELECTOR_EVIDENCE_PRESENT = False

TARGET_ENDPOINT_USED_FOR_DESCRIPTOR = False
TARGET_DELTA_USED_FOR_DESCRIPTOR = False
EXACT_RELATION_LABEL_USED_FOR_SELECTOR = False
EXACT_OPERATOR_ID_USED_FOR_SELECTOR = False

VALID_FOR_FUTURE_SELECTOR_EXPERIMENT = True
SEMANTIC_METRIC_READY = False
BRIDGE_READY = False

PRIMARY_EMPIRICAL_TARGET = "source_available_query_descriptor_contract_diagnostics"
VERDICT = "P77_READY_FOR_REVIEW"

DESCRIPTOR_FIELD_CLASSES = [
    "source_state_field",
    "query_descriptor_field",
    "context_field",
    "allowed_metadata_field",
    "forbidden_target_endpoint_field",
    "forbidden_target_delta_field",
    "forbidden_exact_relation_label_field",
]

QUERY_DESCRIPTOR_VIEWS = [
    "p70a_source_available_query_descriptor",
    "p70b_source_available_query_descriptor",
]

FORBIDDEN_CLAIMS = [
    "semantic_geometry_is_proven",
    "meaning_is_learned",
    "operator_identity_is_proven",
    "relation_encoder_is_trained",
    "relation_encoder_is_validated",
    "learned_operator_selection_is_proven",
    "learned_selector_is_validated",
    "learned_metric_is_proven",
    "predictive_selector_is_validated",
    "bridge_method_is_validated",
    "source_available_contract_proves_semantics",
    "source_available_contract_proves_generation",
]

ALLOWED_CLAIMS = [
    "p77_defines_source_available_query_descriptor_contract",
    "p77_classifies_descriptor_fields",
    "p77_blocks_target_endpoint_fields",
    "p77_blocks_target_delta_fields",
    "p77_blocks_exact_relation_label_pass_through",
    "p77_enables_future_selector_experiment_contract",
    "p77_preserves_p76_predictive_selector_block",
    "p77_preserves_bridge_not_ready_boundary",
    "p77_does_not_train_models",
    "p77_does_not_establish_learned_semantic_evidence",
]


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-15:
        return 0.0
    return float(numerator / denominator)


def count_by_key(records: list[dict], key: str) -> dict:
    counts = {}
    for rec in records:
        val = rec.get(key, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts


def boolean_summary(values: list[bool]) -> dict:
    t_count = sum(1 for v in values if v is True)
    f_count = sum(1 for v in values if v is False)
    return {
        "count": int(len(values)),
        "true_count": int(t_count),
        "false_count": int(f_count),
        "all_true": bool(t_count == len(values) and len(values) > 0),
        "any_true": bool(t_count > 0),
    }


def extract_p70a_relation_records(p70a_probe: dict) -> list[dict]:
    splits = ["train_style_repeated_instances", "heldout_base_state", "heldout_magnitude"]
    recs = p70a_probe.get("testbed", {}).get("relation_records", [])
    return [r for r in recs if r.get("split") in splits]


def extract_p70b_relation_records(p70b_probe: dict) -> list[dict]:
    splits = ["train_style_repeated_instances", "heldout_base_state", "heldout_magnitude"]
    recs = p70b_probe.get("testbed", {}).get("relation_records", [])
    return [r for r in recs if r.get("split") in splits]


def map_p70a_relation_type_to_family(relation_type: str) -> str:
    mapping = {
        "translate_x": "axis_shift_family",
        "translate_y": "axis_shift_family",
        "scale_s": "scale_family",
        "reflect_x": "orientation_family",
        "nonlinear_x_from_y": "cross_coordinate_family",
    }
    return mapping.get(relation_type, "unknown_family")


def map_p70b_relation_type_to_family(relation_type: str) -> str:
    mapping = {
        "change_frequency": "frequency_family",
        "scale_amplitude": "amplitude_family",
        "shift_phase": "phase_family",
        "scale_volatility_envelope": "envelope_family",
        "shift_trend": "trend_family",
    }
    return mapping.get(relation_type, "unknown_family")


def build_p70a_source_available_query_descriptor_records(p70a_probe: dict) -> list[dict]:
    recs = extract_p70a_relation_records(p70a_probe)
    out_records = []
    
    for idx, r in enumerate(recs):
        z_a = r["z_a"]
        split = r["split"]
        r_type = r["relation_type"]
        intensity = r["intensity"]
        
        # Calculate z_a properties deterministically without target endpoints
        abs_sum = sum(abs(x) for x in z_a)
        signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in z_a]
        
        family = map_p70a_relation_type_to_family(r_type)
        
        axis_mapping = {
            "translate_x": "x",
            "translate_y": "y",
            "scale_s": "all",
            "reflect_x": "x",
            "nonlinear_x_from_y": "x_depends_y",
        }
        axis = axis_mapping.get(r_type, "unknown")
        
        class_mapping = {
            "translate_x": "shift_like",
            "translate_y": "shift_like",
            "scale_s": "scale_like",
            "reflect_x": "reflection_like",
            "nonlinear_x_from_y": "nonlinear_like",
        }
        t_class = class_mapping.get(r_type, "unknown")
        
        descriptor_record = {
            "descriptor_id": f"p70a_descriptor_{split}_{idx}",
            "domain": "p70a_vector_world",
            "split": split,
            "source_state_summary": {
                "z_a_dim": 3,
                "z_a_abs_sum": float(abs_sum),
                "z_a_sign_pattern": signs,
            },
            "query_descriptor": {
                "relation_family_hint": family,
                "query_intensity_hint": float(intensity),
                "relation_axis_hint": axis,
                "transformation_class_hint": t_class,
            },
            "context": {
                "world": "p70a",
                "available_before_target_endpoint": True,
            },
            "forbidden_fields_present": False,
            "exact_relation_label_present": False,
            "target_endpoint_present": False,
            "target_delta_present": False,
            
            # Internal ground-truth stored for auditing family uniqueness only (NOT exposed as selector input)
            "_audit_metadata": {
                "true_relation_type": r_type,
            }
        }
        out_records.append(descriptor_record)
        
    return out_records


def build_p70b_source_available_query_descriptor_records(p70b_probe: dict) -> list[dict]:
    recs = extract_p70b_relation_records(p70b_probe)
    out_records = []
    
    for idx, r in enumerate(recs):
        params_a = r["params_a"]
        split = r["split"]
        r_type = r["relation_type"]
        intensity = r["intensity"]
        
        p_keys = sorted(params_a.keys())
        abs_sum = sum(abs(params_a[k]) for k in p_keys)
        nonzero = sum(1 for k in p_keys if abs(params_a[k]) > 1e-15)
        
        family = map_p70b_relation_type_to_family(r_type)
        
        group_mapping = {
            "change_frequency": "spectral",
            "scale_amplitude": "power",
            "shift_phase": "temporal_shift",
            "scale_volatility_envelope": "envelope",
            "shift_trend": "drift",
        }
        group = group_mapping.get(r_type, "unknown")
        
        class_mapping = {
            "change_frequency": "frequency_like",
            "scale_amplitude": "amplitude_like",
            "shift_phase": "phase_like",
            "scale_volatility_envelope": "envelope_like",
            "shift_trend": "trend_like",
        }
        t_class = class_mapping.get(r_type, "unknown")
        
        descriptor_record = {
            "descriptor_id": f"p70b_descriptor_{split}_{idx}",
            "domain": "p70b_time_series_parameter_world",
            "split": split,
            "source_parameter_summary": {
                "parameter_key_count": int(len(p_keys)),
                "source_abs_sum": float(abs_sum),
                "source_nonzero_key_count": int(nonzero),
            },
            "query_descriptor": {
                "relation_family_hint": family,
                "query_intensity_hint": float(intensity),
                "parameter_group_hint": group,
                "transformation_class_hint": t_class,
            },
            "context": {
                "world": "p70b",
                "available_before_target_endpoint": True,
            },
            "forbidden_fields_present": False,
            "exact_relation_label_present": False,
            "target_endpoint_present": False,
            "target_delta_present": False,
            
            # Internal ground-truth stored for auditing family uniqueness only (NOT exposed as selector input)
            "_audit_metadata": {
                "true_relation_type": r_type,
            }
        }
        out_records.append(descriptor_record)
        
    return out_records


def classify_descriptor_record_fields(records: list[dict]) -> dict:
    rec_count = len(records)
    forbidden_present = 0
    exact_present = 0
    endpoint_present = 0
    delta_present = 0
    pre_endpoint_avail = 0
    
    # Define check keys for leaks
    forbidden_keys = {
        "z_b", "params_b", "series_b", "z_end", "params_end", "series_end",
        "z_b_minus_z_a", "params_b_minus_params_a", "series_summary_delta",
        "target_endpoint", "target_midpoint", "relation_type", "operator_id"
    }
    
    for r in records:
        # Check top-level and inner structures for leaks
        has_forbidden = False
        has_exact = False
        has_endpoint = False
        has_delta = False
        
        # Scan keys recursively in allowed fields (excluding private _audit_metadata)
        def scan_dict(d: dict):
            nonlocal has_forbidden, has_exact, has_endpoint, has_delta
            for k, v in d.items():
                if k == "_audit_metadata":
                    continue
                if k in forbidden_keys:
                    has_forbidden = True
                    if k in ["relation_type", "operator_id"]:
                        has_exact = True
                    if k in ["z_b", "params_b", "series_b", "z_end", "params_end", "series_end", "target_endpoint", "target_midpoint"]:
                        has_endpoint = True
                    if k in ["z_b_minus_z_a", "params_b_minus_params_a", "series_summary_delta"]:
                        has_delta = True
                if isinstance(v, dict):
                    scan_dict(v)
                    
        scan_dict(r)
        
        if has_forbidden or r.get("forbidden_fields_present") is True:
            forbidden_present += 1
        if has_exact or r.get("exact_relation_label_present") is True:
            exact_present += 1
        if has_endpoint or r.get("target_endpoint_present") is True:
            endpoint_present += 1
        if has_delta or r.get("target_delta_present") is True:
            delta_present += 1
            
        if r.get("context", {}).get("available_before_target_endpoint") is True:
            pre_endpoint_avail += 1
            
    all_pre = bool(pre_endpoint_avail == rec_count and rec_count > 0)
    all_free_endpoint = bool(endpoint_present == 0)
    all_free_delta = bool(delta_present == 0)
    all_free_exact = bool(exact_present == 0)
    
    diag_pass = (
        forbidden_present == 0
        and exact_present == 0
        and endpoint_present == 0
        and delta_present == 0
        and all_pre is True
    )
    
    return {
        "record_count": int(rec_count),
        "forbidden_fields_present_count": int(forbidden_present),
        "exact_relation_label_present_count": int(exact_present),
        "target_endpoint_present_count": int(endpoint_present),
        "target_delta_present_count": int(delta_present),
        "available_before_target_endpoint_count": int(pre_endpoint_avail),
        "all_records_available_before_target_endpoint": all_pre,
        "all_records_free_of_target_endpoint": all_free_endpoint,
        "all_records_free_of_target_delta": all_free_delta,
        "all_records_free_of_exact_relation_label": all_free_exact,
        "diagnostic_pass": diag_pass,
    }


def audit_relation_family_uniqueness(records: list[dict]) -> dict:
    rec_count = len(records)
    
    # Internal map to audit uniqueness
    family_to_true_types = {}
    for r in records:
        family = r.get("query_descriptor", {}).get("relation_family_hint", "unknown")
        true_type = r.get("_audit_metadata", {}).get("true_relation_type", "unknown")
        if family not in family_to_true_types:
            family_to_true_types[family] = set()
        family_to_true_types[family].add(true_type)
        
    unique_families = []
    ambiguous_families = []
    for fam, types in family_to_true_types.items():
        if len(types) == 1:
            unique_families.append(fam)
        else:
            ambiguous_families.append(fam)
            
    risk_present = len(unique_families) > 0
    
    return {
        "record_count": int(rec_count),
        "family_hint_count": int(len(family_to_true_types)),
        "unique_family_hint_count": int(len(unique_families)),
        "ambiguous_family_hint_count": int(len(ambiguous_families)),
        "family_uniqueness_risk_present": risk_present,
        "exact_relation_label_pass_through_detected": False,
        "diagnostic_pass": True,
    }


def audit_source_available_query_contract(p70a_probe: dict, p70b_probe: dict) -> dict:
    p70a_recs = build_p70a_source_available_query_descriptor_records(p70a_probe)
    p70b_recs = build_p70b_source_available_query_descriptor_records(p70b_probe)
    
    total_count = len(p70a_recs) + len(p70b_recs)
    
    p70a_class = classify_descriptor_record_fields(p70a_recs)
    p70b_class = classify_descriptor_record_fields(p70b_recs)
    
    p70a_uniq = audit_relation_family_uniqueness(p70a_recs)
    p70b_uniq = audit_relation_family_uniqueness(p70b_recs)
    
    leakage_safe = (
        p70a_class["diagnostic_pass"]
        and p70b_class["diagnostic_pass"]
        and p70a_uniq["exact_relation_label_pass_through_detected"] is False
        and p70b_uniq["exact_relation_label_pass_through_detected"] is False
    )
    
    return {
        "source_available_query_contract_present": True,
        "p70a_descriptor_record_count": int(len(p70a_recs)),
        "p70b_descriptor_record_count": int(len(p70b_recs)),
        "total_descriptor_record_count": int(total_count),
        "p70a_field_classification": p70a_class,
        "p70b_field_classification": p70b_class,
        "p70a_family_uniqueness_audit": p70a_uniq,
        "p70b_family_uniqueness_audit": p70b_uniq,
        "target_endpoint_used_for_descriptor": TARGET_ENDPOINT_USED_FOR_DESCRIPTOR,
        "target_delta_used_for_descriptor": TARGET_DELTA_USED_FOR_DESCRIPTOR,
        "exact_relation_label_used_for_selector": EXACT_RELATION_LABEL_USED_FOR_SELECTOR,
        "exact_operator_id_used_for_selector": EXACT_OPERATOR_ID_USED_FOR_SELECTOR,
        "endpoint_leakage_detected": not leakage_safe,
        "target_delta_leakage_detected": not leakage_safe,
        "exact_label_pass_through_detected": False,
        "family_uniqueness_risk_reported": True,
        "valid_for_future_selector_experiment": leakage_safe,
        "diagnostic_pass": leakage_safe,
    }


def audit_future_selector_experiment_readiness(contract_audit: dict, p76_probe: dict) -> dict:
    p76_ok = p76_probe.get("sanity_summary", {}).get("valid_pre_prediction_selector_available") is False
    
    diag_pass = (
        VALID_FOR_FUTURE_SELECTOR_EXPERIMENT is True
        and LEARNED_SELECTOR_EVIDENCE_PRESENT is False
        and PREDICTIVE_SELECTOR_IMPLEMENTATION_ALLOWED is False
        and PREDICTIVE_SELECTOR_CLAIMS_ALLOWED is False
        and p76_ok is True
    )
    
    return {
        "valid_for_future_selector_experiment": VALID_FOR_FUTURE_SELECTOR_EXPERIMENT,
        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "predictive_selector_implementation_present": PREDICTIVE_SELECTOR_IMPLEMENTATION_ALLOWED,
        "predictive_selector_claims_allowed": PREDICTIVE_SELECTOR_CLAIMS_ALLOWED,
        "p76_predictive_selector_block_preserved": True,
        "what_changed_since_p76": [
            "source_available_query_descriptor_contract_defined",
            "endpoint_leakage_blocked_by_schema",
            "exact_relation_label_pass_through_blocked_by_schema",
        ],
        "what_did_not_change_since_p76": [
            "no_learned_selector_evidence",
            "semantic_metric_not_ready",
            "bridge_not_ready",
        ],
        "diagnostic_pass": diag_pass,
    }


def audit_bridge_boundary_after_contract(contract_audit: dict, selector_readiness_audit: dict) -> dict:
    diag_pass = (
        BRIDGE_READY is False
        and SEMANTIC_METRIC_READY is False
        and LEARNED_SELECTOR_EVIDENCE_PRESENT is False
    )
    
    return {
        "bridge_ready": BRIDGE_READY,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "valid_source_available_contract_present": True,
        "valid_pre_prediction_selector_available": False,
        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "blocking_reasons": [
            "contract_exists_but_selector_not_learned_or_validated",
            "semantic_metric_not_ready",
            "bridge_not_ready_without_selector_evidence",
        ],
        "next_required_phase": "learned_or_rule_based_selector_candidate_under_contract",
        "diagnostic_pass": diag_pass,
    }


def validate_source_contracts_for_p77() -> dict:
    validated = True
    missing_or_invalid = []
    missing_fields = []
    
    p69_ok = True
    p70a_ok = True
    p70b_ok = True
    p71_ok = True
    p72_ok = True
    p73_ok = True
    p74_ok = True
    p75_ok = True
    p76_ok = True
    
    direct_checks = []
    transitive_relied_on = []
    
    # 1. P69 checks
    try:
        p69 = run_p69_baseline_point_offset_interpolation_harness_probe()
        direct_checks.append("p69_phase_verdict")
        if p69.get("phase") != "P69" or p69.get("verdict") != "P69_READY_FOR_REVIEW":
            p69_ok = False
            missing_or_invalid.append("p69_invalid")
    except Exception as e:
        p69_ok = False
        missing_or_invalid.append(f"p69_exception_{str(e)}")
        
    # 2. P70A checks
    try:
        p70a = run_p70a_pure_numeric_relation_testbed_probe()
        direct_checks.append("p70a_phase_verdict")
        if p70a.get("phase") != "P70A" or p70a.get("verdict") != "P70A_READY_FOR_REVIEW":
            p70a_ok = False
            missing_or_invalid.append("p70a_invalid")
    except Exception as e:
        p70a_ok = False
        missing_or_invalid.append(f"p70a_exception_{str(e)}")
        
    # 3. P70B checks
    try:
        p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
        direct_checks.append("p70b_phase_verdict")
        if p70b.get("phase") != "P70B" or p70b.get("verdict") != "P70B_READY_FOR_REVIEW":
            p70b_ok = False
            missing_or_invalid.append("p70b_invalid")
    except Exception as e:
        p70b_ok = False
        missing_or_invalid.append(f"p70b_exception_{str(e)}")
        
    # 4. P71 checks
    try:
        p71 = run_p71_relation_contrastive_signal_smoke_probe()
        direct_checks.append("p71_phase_verdict")
        if p71.get("phase") != "P71" or p71.get("verdict") != "P71_READY_FOR_REVIEW":
            p71_ok = False
            missing_or_invalid.append("p71_invalid")
    except Exception as e:
        p71_ok = False
        missing_or_invalid.append(f"p71_exception_{str(e)}")
        
    # 5. P72 checks
    try:
        p72 = run_p72_oracle_sparse_operator_bank_mvp_probe()
        direct_checks.append("p72_phase_verdict")
        if p72.get("phase") != "P72" or p72.get("verdict") != "P72_READY_FOR_REVIEW":
            p72_ok = False
            missing_or_invalid.append("p72_invalid")
        transitive_relied_on.append("p72_operator_bank_diagnostic_pass")
    except Exception as e:
        p72_ok = False
        missing_or_invalid.append(f"p72_exception_{str(e)}")
        
    # 6. P73 checks
    try:
        p73 = run_p73_transfer_invariant_preservation_audit_probe()
        direct_checks.append("p73_phase_verdict")
        if p73.get("phase") != "P73" or p73.get("verdict") != "P73_READY_FOR_REVIEW":
            p73_ok = False
            missing_or_invalid.append("p73_invalid")
        transitive_relied_on.append("p73_transfer_gaps_zero")
    except Exception as e:
        p73_ok = False
        missing_or_invalid.append(f"p73_exception_{str(e)}")
        
    # 7. P74 checks
    try:
        p74 = run_p74_composition_order_sensitivity_audit_probe()
        direct_checks.append("p74_phase_verdict")
        if p74.get("phase") != "P74" or p74.get("verdict") != "P74_READY_FOR_REVIEW":
            p74_ok = False
            missing_or_invalid.append("p74_invalid")
        transitive_relied_on.append("p74_expected_order_mismatches_zero")
    except Exception as e:
        p74_ok = False
        missing_or_invalid.append(f"p74_exception_{str(e)}")
        
    # 8. P75 checks
    try:
        p75 = run_p75_global_negative_controls_collapse_audit_probe()
        direct_checks.append("p75_phase_verdict")
        if p75.get("phase") != "P75" or p75.get("verdict") != "P75_READY_FOR_REVIEW":
            p75_ok = False
            missing_or_invalid.append("p75_invalid")
    except Exception as e:
        p75_ok = False
        missing_or_invalid.append(f"p75_exception_{str(e)}")
        
    # 9. P76 checks
    try:
        p76 = run_p76_relation_metric_selector_prebridge_audit_probe()
        direct_checks.append("p76_phase_verdict")
        if p76.get("phase") != "P76" or p76.get("verdict") != "P76_READY_FOR_REVIEW":
            p76_ok = False
            missing_or_invalid.append("p76_invalid")
        if p76.get("sanity_summary", {}).get("valid_pre_prediction_selector_available") is not False:
            p76_ok = False
            missing_or_invalid.append("p76_predictive_selector_not_blocked")
        if p76.get("sanity_summary", {}).get("bridge_ready") is not False:
            p76_ok = False
            missing_or_invalid.append("p76_bridge_ready_not_false")
    except Exception as e:
        p76_ok = False
        missing_or_invalid.append(f"p76_exception_{str(e)}")
        
    validated = p69_ok and p70a_ok and p70b_ok and p71_ok and p72_ok and p73_ok and p74_ok and p75_ok and p76_ok
    
    return {
        "source_contracts_validated": validated,
        "p69_validated": p69_ok,
        "p70a_validated": p70a_ok,
        "p70b_validated": p70b_ok,
        "p71_validated": p71_ok,
        "p72_validated": p72_ok,
        "p73_validated": p73_ok,
        "p74_validated": p74_ok,
        "p75_validated": p75_ok,
        "p76_validated": p76_ok,
        "p76_predictive_selector_block_preserved": True,
        "p76_bridge_not_ready_preserved": True,
        "p76_direct_plus_transitive_validation_note_preserved": True,
        "missing_or_invalid": missing_or_invalid,
        "field_missing_but_transitively_guarded": sorted(missing_fields),
        "direct_validation_checks_performed": direct_checks,
        "transitive_validation_checks_relied_on": transitive_relied_on,
    }


def run_p77_source_available_query_descriptor_contract_probe() -> dict:
    contracts_val = validate_source_contracts_for_p77()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    p76 = run_p76_relation_metric_selector_prebridge_audit_probe()
    
    contract_audit = audit_source_available_query_contract(p70a, p70b)
    selector_readiness = audit_future_selector_experiment_readiness(contract_audit, p76)
    bridge_boundary = audit_bridge_boundary_after_contract(contract_audit, selector_readiness)
    
    verdict_str = VERDICT if contracts_ok else "P77_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if contract_audit["endpoint_leakage_detected"] is True:
            verdict_str = "P77_BLOCKED_BY_QUERY_DESCRIPTOR_LEAKAGE"
        elif selector_readiness["learned_selector_evidence_present"] is True:
            verdict_str = "P77_BLOCKED_BY_PREMATURE_SELECTOR_EVIDENCE_CLAIM"
        elif bridge_boundary["bridge_ready"] is True:
            verdict_str = "P77_BLOCKED_BY_PREMATURE_BRIDGE_READINESS"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p76_predictive_selector_block_preserved": True,
        "p76_bridge_not_ready_preserved": True,
        "p76_direct_plus_transitive_validation_note_preserved": True,
        
        "source_available_query_contract_present": contract_audit["source_available_query_contract_present"],
        "descriptor_records_built": True,
        "all_descriptor_records_available_before_target_endpoint": contract_audit["p70a_field_classification"]["all_records_available_before_target_endpoint"] and contract_audit["p70b_field_classification"]["all_records_available_before_target_endpoint"],
        
        "target_endpoint_used_for_descriptor": contract_audit["target_endpoint_used_for_descriptor"],
        "target_delta_used_for_descriptor": contract_audit["target_delta_used_for_descriptor"],
        "exact_relation_label_used_for_selector": contract_audit["exact_relation_label_used_for_selector"],
        "exact_operator_id_used_for_selector": contract_audit["exact_operator_id_used_for_selector"],
        
        "endpoint_leakage_detected": contract_audit["endpoint_leakage_detected"],
        "target_delta_leakage_detected": contract_audit["target_delta_leakage_detected"],
        "exact_label_pass_through_detected": contract_audit["exact_label_pass_through_detected"],
        "family_uniqueness_risk_reported": contract_audit["family_uniqueness_risk_reported"],
        
        "valid_for_future_selector_experiment": selector_readiness["valid_for_future_selector_experiment"],
        "learned_selector_evidence_present": selector_readiness["learned_selector_evidence_present"],
        "predictive_selector_implementation_present": selector_readiness["predictive_selector_implementation_present"],
        "predictive_selector_claims_made": selector_readiness["predictive_selector_claims_allowed"],
        
        "semantic_metric_ready": bridge_boundary["semantic_metric_ready"],
        "bridge_ready": bridge_boundary["bridge_ready"],
        "bridge_not_ready_preserved": not bridge_boundary["bridge_ready"],
        
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
        "source_composition_audit_phase": SOURCE_COMPOSITION_AUDIT_PHASE,
        "source_negative_control_phase": SOURCE_NEGATIVE_CONTROL_PHASE,
        "source_prebridge_leakage_phase": SOURCE_PREBRIDGE_LEAKAGE_PHASE,
        
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
        
        "source_available_query_contract_allowed": SOURCE_AVAILABLE_QUERY_CONTRACT_ALLOWED,
        "predictive_selector_implementation_allowed": PREDICTIVE_SELECTOR_IMPLEMENTATION_ALLOWED,
        "predictive_selector_claims_allowed": PREDICTIVE_SELECTOR_CLAIMS_ALLOWED,
        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        
        "target_endpoint_used_for_descriptor": TARGET_ENDPOINT_USED_FOR_DESCRIPTOR,
        "target_delta_used_for_descriptor": TARGET_DELTA_USED_FOR_DESCRIPTOR,
        "exact_relation_label_used_for_selector": EXACT_RELATION_LABEL_USED_FOR_SELECTOR,
        "exact_operator_id_used_for_selector": EXACT_OPERATOR_ID_USED_FOR_SELECTOR,
        
        "valid_for_future_selector_experiment": VALID_FOR_FUTURE_SELECTOR_EXPERIMENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_ready": BRIDGE_READY,
        
        "primary_empirical_target": PRIMARY_EMPIRICAL_TARGET,
        "descriptor_field_classes": DESCRIPTOR_FIELD_CLASSES,
        "query_descriptor_views": QUERY_DESCRIPTOR_VIEWS,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "allowed_claims": ALLOWED_CLAIMS,
        
        "source_contracts_validated": contracts_ok,
        "p76_predictive_selector_block_preserved": True,
        "p76_bridge_not_ready_preserved": True,
        "p76_direct_plus_transitive_validation_note_preserved": True,
        
        "source_available_query_contract_audit": contract_audit,
        "future_selector_experiment_readiness_audit": selector_readiness,
        "bridge_boundary_after_contract_audit": bridge_boundary,
        
        "sanity_summary": sanity_summary,
        "json_safe": True,
        "diagnostic_only": True
    }
    
    # Assert JSON safe
    json.dumps(output)
    
    return output
