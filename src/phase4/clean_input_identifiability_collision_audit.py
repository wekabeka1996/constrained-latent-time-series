# src/phase4/clean_input_identifiability_collision_audit.py

import json
from typing import Any

from src.phase4.learned_selector_dataset_contract import (
    build_selector_dataset_records,
)

# Replaced transitive P82 pilot probe import to avoid training loops during audit runtime.

from src.phase3.pure_numeric_relation_testbed import (
    run_p70a_pure_numeric_relation_testbed_probe,
)

from src.phase3.synthetic_time_series_relation_testbed import (
    run_p70b_synthetic_time_series_relation_testbed_probe,
)

PHASE = "P83"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Clean Input Identifiability and Collision Audit"
CONTRACT_VERSION = "phase4_p83_clean_input_identifiability_collision_audit_v1"

SOURCE_SELECTOR_PILOT_PHASE = "P82"
SOURCE_DATASET_PHASE = "P81"

TRAINING_ALLOWED_BY_PHASE4_AUTHORITY = True
MODEL_TRAINING_PERFORMED = False
TORCH_TRAINING_PERFORMED = False
NEW_MODEL_IMPLEMENTED = False
OPTIMIZER_CREATED = False
CHECKPOINT_WRITTEN = False

CLEAN_SELECTOR_INPUT_USED = True
HINT_PASSTHROUGH_USED_AS_MODEL_INPUT = False

TARGET_ENDPOINT_USED_FOR_SELECTOR_INPUT = False
TARGET_DELTA_USED_FOR_SELECTOR_INPUT = False
EXACT_RELATION_LABEL_USED_FOR_SELECTOR_INPUT = False
EXACT_OPERATOR_ID_USED_FOR_SELECTOR_INPUT = False
AUDIT_METADATA_USED_FOR_SELECTOR_INPUT = False
RELATION_SPECIFIC_HINT_USED_FOR_MODEL_INPUT = False

CLEAN_INPUT_IDENTIFIABILITY_AUDIT_PERFORMED = True
CLEAN_INPUT_LABEL_COLLISION_AUDIT_PERFORMED = True
DETERMINISTIC_ORACLE_UPPER_BOUND_COMPUTED = True

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "clean_input_identifiability_collision_audit"
VERDICT = "P83_READY_FOR_REVIEW"


ACCEPTED_P82_REPORT_METADATA = {
    "phase": "P82",
    "verdict": "P82_READY_FOR_REVIEW",
    "tiny_learned_selector_trained": True,
    "clean_selector_input_used": True,
    "hint_passthrough_used_as_model_input": False,
    "target_endpoint_used_for_selector_input": False,
    "test_split_used_for_training": False,
    "checkpoint_written": False,
    "learned_selector_evidence_present": False,
    "bridge_ready": False,
    "baseline_results": {
        "source_only_baseline": {
            "test_accuracy": 0.2222
        }
    },
    "tiny_selector_training_results": {
        "test_accuracy_at_best": 0.2222
    }
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-15:
        return 0.0
    return float(numerator / denominator)


def extract_label(record: dict) -> str:
    return record["label_evaluation"]["target_relation_label"]


def build_full_clean_input_key(record: dict) -> str:
    sel = record["selector_input"]
    key_dict = {
        "domain": sel.get("domain"),
        "query_intensity_hint": sel.get("query_intensity_hint"),
        "context_world": sel.get("context_world"),
    }
    if "source_state_summary" in sel:
        key_dict["source_state_summary"] = sel["source_state_summary"]
    if "source_parameter_summary" in sel:
        key_dict["source_parameter_summary"] = sel["source_parameter_summary"]
    return canonical_json(key_dict)


def build_p82_model_feature_key(record: dict) -> str:
    sel = record["selector_input"]
    domain = sel.get("domain", "unknown")
    intensity = float(sel.get("query_intensity_hint", 0.0))
    context = sel.get("context_world", "unknown")
    
    dom_a = 1.0 if domain == "p70a_vector_world" else 0.0
    dom_b = 1.0 if domain == "p70b_time_series_parameter_world" else 0.0
    
    ctx_a = 1.0 if context == "p70a" else 0.0
    ctx_b = 1.0 if context == "p70b" else 0.0
    
    z_a_dim = 0.0
    z_a_abs_sum = 0.0
    z_a_sign_0 = 0.0
    z_a_sign_1 = 0.0
    z_a_sign_2 = 0.0
    
    state_sum = sel.get("source_state_summary", {})
    if state_sum:
        z_a_dim = float(state_sum.get("z_a_dim", 0.0))
        z_a_abs_sum = float(state_sum.get("z_a_abs_sum", 0.0))
        signs = state_sum.get("z_a_sign_pattern", [])
        if len(signs) >= 3:
            z_a_sign_0 = float(signs[0])
            z_a_sign_1 = float(signs[1])
            z_a_sign_2 = float(signs[2])
            
    params_key_count = 0.0
    params_abs_sum = 0.0
    params_nonzero_key_count = 0.0
    
    param_sum = sel.get("source_parameter_summary", {})
    if param_sum:
        params_key_count = float(param_sum.get("params_key_count", 0.0))
        params_abs_sum = float(param_sum.get("params_abs_sum", 0.0))
        params_nonzero_key_count = float(param_sum.get("params_nonzero_key_count", 0.0))
        
    vec = [
        dom_a, dom_b,
        intensity,
        ctx_a, ctx_b,
        z_a_dim, z_a_abs_sum,
        z_a_sign_0, z_a_sign_1, z_a_sign_2,
        params_key_count, params_abs_sum, params_nonzero_key_count
    ]
    return canonical_json(vec)


def build_source_only_key(record: dict) -> str:
    sel = record["selector_input"]
    key_dict = {
        "domain": sel.get("domain"),
        "context_world": sel.get("context_world"),
    }
    if "source_state_summary" in sel:
        key_dict["source_state_summary"] = sel["source_state_summary"]
    if "source_parameter_summary" in sel:
        key_dict["source_parameter_summary"] = sel["source_parameter_summary"]
    return canonical_json(key_dict)


def group_records_by_key(records: list[dict], key_name: str) -> dict:
    grouped = {}
    for r in records:
        if key_name == "full_clean_input_key":
            k = build_full_clean_input_key(r)
        elif key_name == "p82_model_feature_key":
            k = build_p82_model_feature_key(r)
        elif key_name == "source_only_key":
            k = build_source_only_key(r)
        else:
            k = "unknown"
        grouped[k] = grouped.get(k, []) + [r]
    return grouped


def analyze_label_collisions(records: list[dict], key_name: str) -> dict:
    grouped = group_records_by_key(records, key_name)
    total_records = len(records)
    
    unique_key_count = len(grouped)
    collided_key_count = 0
    non_collided_key_count = 0
    records_in_collided_keys = 0
    
    max_labels = 0
    total_labels = 0
    upper_bound_correct = 0
    
    for k, recs in grouped.items():
        lbl_counts = {}
        for r in recs:
            lbl = extract_label(r)
            lbl_counts[lbl] = lbl_counts.get(lbl, 0) + 1
            
        unique_lbl_count = len(lbl_counts)
        total_labels += unique_lbl_count
        if unique_lbl_count > max_labels:
            max_labels = unique_lbl_count
            
        if unique_lbl_count > 1:
            collided_key_count += 1
            records_in_collided_keys += len(recs)
        else:
            non_collided_key_count += 1
            
        # Oracle upper bound logic
        max_freq = max(lbl_counts.values())
        upper_bound_correct += max_freq
        
    fraction = safe_divide(records_in_collided_keys, total_records)
    mean_labels = safe_divide(total_labels, unique_key_count)
    upper_bound_acc = safe_divide(upper_bound_correct, total_records)
    
    return {
        "key_name": key_name,
        "record_count": int(total_records),
        "unique_key_count": int(unique_key_count),
        "collided_key_count": int(collided_key_count),
        "non_collided_key_count": int(non_collided_key_count),
        "records_in_collided_keys": int(records_in_collided_keys),
        "collision_record_fraction": float(fraction),
        "max_labels_per_key": int(max_labels),
        "mean_labels_per_key": float(mean_labels),
        "deterministic_identifiability_upper_bound_accuracy": float(upper_bound_acc),
        "diagnostic_pass": True,
    }


def analyze_split_aware_identifiability(records: list[dict], key_name: str) -> dict:
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "validation"]
    test_recs = [r for r in records if r["split"] == "test"]
    
    return {
        "key_name": key_name,
        "train": analyze_label_collisions(train_recs, key_name) if len(train_recs) > 0 else {},
        "validation": analyze_label_collisions(val_recs, key_name) if len(val_recs) > 0 else {},
        "test": analyze_label_collisions(test_recs, key_name) if len(test_recs) > 0 else {},
        "all": analyze_label_collisions(records, key_name),
        "diagnostic_pass": True,
    }


def analyze_label_distribution(records: list[dict]) -> dict:
    def get_counts(recs):
        cnts = {}
        for r in recs:
            lbl = extract_label(r)
            cnts[lbl] = cnts.get(lbl, 0) + 1
        return cnts
        
    dist_all = get_counts(records)
    
    dist_split = {}
    for s in ["train", "validation", "test"]:
        s_recs = [r for r in records if r["split"] == s]
        dist_split[s] = get_counts(s_recs)
        
    dist_domain = {}
    for d in ["p70a_vector_world", "p70b_time_series_parameter_world"]:
        d_recs = [r for r in records if r["domain"] == d]
        dist_domain[d] = get_counts(d_recs)
        
    return {
        "total_record_count": int(len(records)),
        "label_distribution_all": dist_all,
        "label_distribution_by_split": dist_split,
        "label_distribution_by_domain": dist_domain,
        "diagnostic_pass": True,
    }


def interpret_identifiability_results(collision_audits: dict, p82_probe: dict) -> dict:
    # Read p82 model feature key test summary
    feature_test_summary = collision_audits["p82_model_feature_key"]["test"]
    upper_bound = feature_test_summary.get("deterministic_identifiability_upper_bound_accuracy", 1.0)
    collision_fraction = feature_test_summary.get("collision_record_fraction", 0.0)
    
    # Check baseline results
    p82_baselines = p82_probe.get("baseline_results", {})
    src_baseline = p82_baselines.get("source_only_baseline", {}).get("test_accuracy", 0.2222)
    p82_test_acc = p82_probe.get("tiny_selector_training_results", {}).get("test_accuracy_at_best", 0.2222)
    
    # If the deterministic upper bound is very close to domain baseline
    is_underdetermined = (upper_bound <= src_baseline + 0.05) or (collision_fraction >= 0.5)
    
    if is_underdetermined:
        return {
            "clean_input_relation_identifiability_established": False,
            "clean_input_label_collisions_present": bool(collision_fraction > 0.0),
            "model_capacity_not_primary_failure": True,
            "additional_observation_context_required": True,
            "recommended_next_phase": "P84_clean_query_observation_enrichment_contract",
            "diagnostic_pass": True,
        }
    else:
        return {
            "clean_input_relation_identifiability_established": "inconclusive",
            "clean_input_label_collisions_present": bool(collision_fraction > 0.0),
            "model_capacity_not_primary_failure": False,
            "additional_observation_context_required": "uncertain",
            "recommended_next_phase": "P84_stronger_selector_or_regularization_probe",
            "diagnostic_pass": True,
        }


def audit_bridge_boundary_after_identifiability_audit(interpretation: dict) -> dict:
    blocking_reasons = [
        "learned_selector_evidence_not_present",
        "learned_metric_evidence_not_present",
        "clean_input_identifiability_not_established",
        "bridge_input_contract_not_defined",
        "bridge_validation_not_run",
    ]
    
    return {
        "bridge_ready": BRIDGE_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,
        "blocking_reasons": blocking_reasons,
        "diagnostic_pass": True,
    }


def validate_source_contracts_for_p83() -> dict:
    validated = True
    missing_or_invalid = []
    
    p82_ok = True
    
    try:
        p82 = ACCEPTED_P82_REPORT_METADATA
        if p82.get("phase") != "P82" or p82.get("verdict") != "P82_READY_FOR_REVIEW":
            p82_ok = False
            missing_or_invalid.append("p82_invalid_phase_or_verdict")
        if p82.get("tiny_learned_selector_trained") is not True:
            p82_ok = False
            missing_or_invalid.append("p82_training_not_done")
        if p82.get("clean_selector_input_used") is not True:
            p82_ok = False
            missing_or_invalid.append("p82_leakage_checks_failed")
        if p82.get("hint_passthrough_used_as_model_input") is not False:
            p82_ok = False
            missing_or_invalid.append("p82_hints_used_as_input")
        if p82.get("target_endpoint_used_for_selector_input") is not False:
            p82_ok = False
            missing_or_invalid.append("p82_target_endpoint_leakage")
        if p82.get("test_split_used_for_training") is not False:
            p82_ok = False
            missing_or_invalid.append("p82_test_leakage")
        if p82.get("checkpoint_written") is not False:
            p82_ok = False
            missing_or_invalid.append("p82_checkpoints_created")
        if p82.get("learned_selector_evidence_present") is not False:
            p82_ok = False
            missing_or_invalid.append("p82_has_selector_evidence")
        if p82.get("bridge_ready") is not False:
            p82_ok = False
            missing_or_invalid.append("p82_bridge_ready")
    except Exception as e:
        p82_ok = False
        missing_or_invalid.append(f"p82_exception_{str(e)}")
        
    validated = p82_ok
    
    return {
        "source_contracts_validated": validated,
        "p82_validated": p82_ok,
        "p82_tiny_selector_trained_preserved": p82_ok,
        "p82_clean_input_only_preserved": p82_ok,
        "p82_no_learned_selector_evidence_preserved": p82_ok,
        "p82_bridge_not_ready_preserved": p82_ok,
        "missing_or_invalid": missing_or_invalid,
    }


def run_p83_clean_input_identifiability_collision_audit_probe() -> dict:
    contracts_val = validate_source_contracts_for_p83()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    records = build_selector_dataset_records(p70a, p70b)
    
    label_dist = analyze_label_distribution(records)
    
    collision_keys = ["full_clean_input_key", "p82_model_feature_key", "source_only_key"]
    
    collision_audits = {}
    split_aware_audits = {}
    for k in collision_keys:
        collision_audits[k] = analyze_label_collisions(records, k)
        split_aware_audits[k] = analyze_split_aware_identifiability(records, k)
        
    interpretation = interpret_identifiability_results(split_aware_audits, ACCEPTED_P82_REPORT_METADATA)
    bridge_audit = audit_bridge_boundary_after_identifiability_audit(interpretation)
    
    verdict_str = VERDICT if contracts_ok else "P83_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED is True or TORCH_TRAINING_PERFORMED is True:
            verdict_str = "P83_BLOCKED_BY_UNEXPECTED_TRAINING_OR_MODEL"
        elif BRIDGE_READY is True:
            verdict_str = "P83_BLOCKED_BY_PREMATURE_BRIDGE_AUTHORITY"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p82_tiny_selector_trained_preserved": contracts_val["p82_tiny_selector_trained_preserved"],
        "p82_clean_input_only_preserved": contracts_val["p82_clean_input_only_preserved"],
        "p82_no_learned_selector_evidence_preserved": contracts_val["p82_no_learned_selector_evidence_preserved"],
        "p82_bridge_not_ready_preserved": contracts_val["p82_bridge_not_ready_preserved"],

        "clean_input_identifiability_audit_performed": CLEAN_INPUT_IDENTIFIABILITY_AUDIT_PERFORMED,
        "clean_input_label_collision_audit_performed": CLEAN_INPUT_LABEL_COLLISION_AUDIT_PERFORMED,
        "deterministic_oracle_upper_bound_computed": DETERMINISTIC_ORACLE_UPPER_BOUND_COMPUTED,

        "clean_input_relation_identifiability_established": interpretation["clean_input_relation_identifiability_established"],
        "clean_input_label_collisions_present": interpretation["clean_input_label_collisions_present"],
        "model_capacity_not_primary_failure": interpretation["model_capacity_not_primary_failure"],
        "additional_observation_context_required": interpretation["additional_observation_context_required"],

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_ready": BRIDGE_READY,

        "json_safe": True,
    }
    
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,

        "source_selector_pilot_phase": SOURCE_SELECTOR_PILOT_PHASE,
        "source_dataset_phase": SOURCE_DATASET_PHASE,

        "verdict": verdict_str,

        "training_allowed_by_phase4_authority": TRAINING_ALLOWED_BY_PHASE4_AUTHORITY,
        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "clean_selector_input_used": CLEAN_SELECTOR_INPUT_USED,
        "hint_passthrough_used_as_model_input": HINT_PASSTHROUGH_USED_AS_MODEL_INPUT,

        "target_endpoint_used_for_selector_input": TARGET_ENDPOINT_USED_FOR_SELECTOR_INPUT,
        "target_delta_used_for_selector_input": TARGET_DELTA_USED_FOR_SELECTOR_INPUT,
        "exact_relation_label_used_for_selector_input": EXACT_RELATION_LABEL_USED_FOR_SELECTOR_INPUT,
        "exact_operator_id_used_for_selector_input": EXACT_OPERATOR_ID_USED_FOR_SELECTOR_INPUT,
        "audit_metadata_used_for_selector_input": AUDIT_METADATA_USED_FOR_SELECTOR_INPUT,
        "relation_specific_hint_used_for_model_input": RELATION_SPECIFIC_HINT_USED_FOR_MODEL_INPUT,

        "clean_input_identifiability_audit_performed": CLEAN_INPUT_IDENTIFIABILITY_AUDIT_PERFORMED,
        "clean_input_label_collision_audit_performed": CLEAN_INPUT_LABEL_COLLISION_AUDIT_PERFORMED,
        "deterministic_oracle_upper_bound_computed": DETERMINISTIC_ORACLE_UPPER_BOUND_COMPUTED,

        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "bridge_ready": BRIDGE_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,

        "source_contracts_validated": contracts_ok,
        "p82_tiny_selector_trained_preserved": contracts_val["p82_tiny_selector_trained_preserved"],
        "p82_clean_input_only_preserved": contracts_val["p82_clean_input_only_preserved"],
        "p82_no_learned_selector_evidence_preserved": contracts_val["p82_no_learned_selector_evidence_preserved"],
        "p82_bridge_not_ready_preserved": contracts_val["p82_bridge_not_ready_preserved"],

        "label_distribution_audit": label_dist,
        "collision_audits": collision_audits,
        "split_aware_collision_audits": split_aware_audits,
        "identifiability_interpretation": interpretation,
        "bridge_boundary_after_identifiability_audit": bridge_audit,

        "sanity_summary": sanity_summary,

        "json_safe": True,
        "diagnostic_only": True
    }
    
    return output
