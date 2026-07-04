# src/phase4/clean_query_observation_enrichment_contract.py

import json
from typing import Any

from src.phase4.clean_input_identifiability_collision_audit import (
    run_p83_clean_input_identifiability_collision_audit_probe,
    analyze_label_collisions,
)

from src.phase4.learned_selector_dataset_contract import (
    build_selector_dataset_records,
)

from src.phase3.pure_numeric_relation_testbed import (
    run_p70a_pure_numeric_relation_testbed_probe,
)

from src.phase3.synthetic_time_series_relation_testbed import (
    run_p70b_synthetic_time_series_relation_testbed_probe,
)

PHASE = "P84"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Clean Query-Observation Enrichment Contract"
CONTRACT_VERSION = "phase4_p84_clean_query_observation_enrichment_contract_v1"

SOURCE_IDENTIFIABILITY_PHASE = "P83"
SOURCE_SELECTOR_PILOT_PHASE = "P82"
SOURCE_DATASET_PHASE = "P81"

TRAINING_ALLOWED_BY_PHASE4_AUTHORITY = True
MODEL_TRAINING_PERFORMED = False
TORCH_TRAINING_PERFORMED = False
NEW_MODEL_IMPLEMENTED = False
OPTIMIZER_CREATED = False
CHECKPOINT_WRITTEN = False

QUERY_OBSERVATION_ENRICHMENT_CONTRACT_PRESENT = True
SUPPORT_OBSERVATION_CONTEXT_DEFINED = True
INVARIANT_OBSERVATION_CONTEXT_DEFINED = True
EXTERNAL_QUERY_CONTEXT_SCHEMA_DEFINED = True

TARGET_RECORD_ENDPOINT_USED_FOR_SELECTOR_INPUT = False
TARGET_RECORD_DELTA_USED_FOR_SELECTOR_INPUT = False
EXACT_RELATION_LABEL_USED_FOR_SELECTOR_INPUT = False
EXACT_OPERATOR_ID_USED_FOR_SELECTOR_INPUT = False
AUDIT_METADATA_USED_FOR_SELECTOR_INPUT = False
RELATION_SPECIFIC_HINT_USED_FOR_MODEL_INPUT = False

SYNTHETIC_SUPPORT_CONTEXT_FOR_FEASIBILITY_ONLY = True
SUPPORT_CONTEXT_CONSTRUCTED_FROM_AUDIT_LABEL = True
VALID_FOR_FINAL_SELECTOR_EVIDENCE = False
VALID_FOR_FEASIBILITY_COLLISION_REDUCTION_AUDIT = True

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "clean_query_observation_enrichment_contract"
VERDICT = "P84_READY_FOR_REVIEW"

ORIGINAL_CLEAN_SELECTOR_VIEW = "p81_clean_selector_input"
ENRICHED_SELECTOR_VIEW = "query_observation_enriched_selector_input"

ENRICHED_SELECTOR_ALLOWED_FIELDS = [
    "domain",
    "query_intensity_hint",
    "context_world",
    "source_state_summary",
    "source_parameter_summary",
    "support_observation_context",
    "invariant_observation_context",
    "query_observation_context_metadata"
]

ENRICHED_SELECTOR_FORBIDDEN_FIELDS = [
    "_audit_metadata",
    "true_relation_type",
    "target_relation_label",
    "relation_type",
    "operator_id",
    "relation_family_hint",
    "transformation_class_hint",
    "relation_axis_hint",
    "parameter_group_hint",
    "target_endpoint",
    "target_midpoint",
    "z_b",
    "params_b",
    "series_b",
    "z_b_minus_z_a",
    "params_b_minus_params_a",
    "series_summary_delta"
]

FORBIDDEN_LABEL_DERIVED_CONTROL_VIEW = "label_derived_context_forbidden_control"
ENRICHMENT_AUDIT_VIEW = "query_observation_enrichment_audit"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-15:
        return 0.0
    return float(numerator / denominator)


def extract_label(record: dict) -> str:
    return record["label_evaluation"]["target_relation_label"]


def scan_keys(value: Any) -> set[str]:
    keys = set()
    if isinstance(value, dict):
        for k, v in value.items():
            keys.add(k)
            keys.update(scan_keys(v))
    elif isinstance(value, list):
        for item in value:
            keys.update(scan_keys(item))
    return keys


def build_original_clean_key(record: dict) -> str:
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


def build_support_pool(records: list[dict]) -> dict:
    pool = {}
    for r in records:
        lbl = extract_label(r)
        pool[lbl] = pool.get(lbl, []) + [r]
    return pool


def select_support_records_for_record(record: dict, support_pool: dict, max_support: int = 2) -> list[dict]:
    lbl = extract_label(record)
    pool = support_pool.get(lbl, [])
    
    # Deterministic sorting (prefer train split, secondary sort by dataset_record_id)
    sorted_recs = sorted(pool, key=lambda r: (0 if r["split"] == "train" else 1, r["dataset_record_id"]))
    
    # Exclude evaluated record self
    filtered = [r for r in sorted_recs if r["dataset_record_id"] != record["dataset_record_id"]]
    
    return filtered[:max_support]


def summarize_support_record(record: dict) -> dict:
    sel = record["selector_input"]
    summary = {
        "support_domain": record["domain"],
        "support_context_world": sel.get("context_world", "unknown"),
    }
    
    # Source summaries are allowed
    if "source_state_summary" in sel:
        summary["support_source_summary"] = sel["source_state_summary"]
    if "source_parameter_summary" in sel:
        summary["support_source_summary"] = sel["source_parameter_summary"]
        
    # Result and delta summaries are missing in P81 clean input
    summary["support_result_summary"] = {}
    summary["support_delta_summary"] = {}
    
    return summary


def build_invariant_observation_context(support_summaries: list[dict]) -> dict:
    return {
        "support_count": int(len(support_summaries)),
        "source_summary_available_count": sum(1 for s in support_summaries if s.get("support_source_summary")),
        "result_summary_available_count": 0,
        "delta_summary_available_count": 0,
        "changed_dimensions_count": 0,
        "preserved_dimensions_count": 0,
        "support_delta_abs_sum_mean": 0.0,
        "support_delta_nonzero_count_mean": 0.0,
        
        "invariant_context_available": False,
        "support_observation_limited_by_missing_result_summaries": True
    }


def build_query_observation_enriched_selector_input(record: dict, support_pool: dict) -> dict:
    sel = record["selector_input"]
    
    support_recs = select_support_records_for_record(record, support_pool, max_support=2)
    support_summaries = [summarize_support_record(r) for r in support_recs]
    
    invariant_context = build_invariant_observation_context(support_summaries)
    
    enriched = {
        "domain": sel.get("domain"),
        "query_intensity_hint": sel.get("query_intensity_hint"),
        "context_world": sel.get("context_world"),
        "support_observation_context": support_summaries,
        "invariant_observation_context": invariant_context,
        "query_observation_context_metadata": {
            "context_source_kind": "synthetic_support_context_for_feasibility_only",
            "support_context_constructed_from_audit_label": True,
            "valid_for_final_selector_evidence": False,
            "valid_for_feasibility_collision_reduction_audit": True,
            "external_query_source_required_for_final_evidence": True
        }
    }
    
    if "source_state_summary" in sel:
        enriched["source_state_summary"] = sel["source_state_summary"]
    if "source_parameter_summary" in sel:
        enriched["source_parameter_summary"] = sel["source_parameter_summary"]
        
    return enriched


def audit_enriched_input_leakage(enriched_input: dict) -> dict:
    target_record_endpoint_used = False
    target_record_delta_used = False
    exact_relation_label_used = False
    exact_operator_id_used = False
    audit_metadata_used = False
    relation_specific_hint_used = False
    
    forbidden_endpoints = {"z_b", "params_b", "series_b", "z_end", "params_end", "series_end", "target_endpoint", "target_midpoint"}
    forbidden_deltas = {"z_b_minus_z_a", "params_b_minus_params_a", "series_summary_delta"}
    forbidden_labels = {"relation_type", "true_relation_type", "target_relation_label"}
    forbidden_operators = {"operator_id"}
    forbidden_metadata = {"_audit_metadata"}
    forbidden_hints = {"relation_family_hint", "transformation_class_hint", "relation_axis_hint", "parameter_group_hint"}
    
    all_keys = scan_keys(enriched_input)
    
    for k in all_keys:
        if k in forbidden_endpoints:
            target_record_endpoint_used = True
        if k in forbidden_deltas:
            target_record_delta_used = True
        if k in forbidden_labels:
            exact_relation_label_used = True
        if k in forbidden_operators:
            exact_operator_id_used = True
        if k in forbidden_metadata:
            audit_metadata_used = True
        if k in forbidden_hints:
            relation_specific_hint_used = True
            
    diag_pass = not (
        target_record_endpoint_used
        or target_record_delta_used
        or exact_relation_label_used
        or exact_operator_id_used
        or audit_metadata_used
        or relation_specific_hint_used
    )
    
    return {
        "target_record_endpoint_used_for_selector_input": target_record_endpoint_used,
        "target_record_delta_used_for_selector_input": target_record_delta_used,
        "exact_relation_label_used_for_selector_input": exact_relation_label_used,
        "exact_operator_id_used_for_selector_input": exact_operator_id_used,
        "audit_metadata_used_for_selector_input": audit_metadata_used,
        "relation_specific_hint_used_for_model_input": relation_specific_hint_used,
        
        "support_context_constructed_from_audit_label": True,
        "valid_for_final_selector_evidence": False,
        "valid_for_feasibility_collision_reduction_audit": True,
        "diagnostic_pass": diag_pass
    }


def build_query_observation_enriched_records(records: list[dict]) -> list[dict]:
    support_pool = build_support_pool(records)
    
    out = []
    for r in records:
        enriched_input = build_query_observation_enriched_selector_input(r, support_pool)
        leakage = audit_enriched_input_leakage(enriched_input)
        
        out.append({
            "dataset_record_id": r["dataset_record_id"],
            "split": r["split"],
            "domain": r["domain"],
            "original_clean_selector_input": r["selector_input"],
            "query_observation_enriched_selector_input": enriched_input,
            "label_evaluation": r["label_evaluation"],
            "enrichment_leakage_audit": leakage
        })
    return out


def build_enriched_selector_key(enriched_record: dict) -> str:
    # Full clean input + support/invariant context (only if invariant_context_available is True)
    sel = enriched_record["query_observation_enriched_selector_input"]
    key_dict = {
        "domain": sel.get("domain"),
        "query_intensity_hint": sel.get("query_intensity_hint"),
        "context_world": sel.get("context_world"),
    }
    
    inv = sel.get("invariant_observation_context", {})
    if inv.get("invariant_context_available") is True:
        key_dict["support_observation_context"] = sel.get("support_observation_context")
        key_dict["invariant_observation_context"] = inv
        
    if "source_state_summary" in sel:
        key_dict["source_state_summary"] = sel["source_state_summary"]
    if "source_parameter_summary" in sel:
        key_dict["source_parameter_summary"] = sel["source_parameter_summary"]
    return canonical_json(key_dict)


def analyze_enriched_collision_reduction(original_records: list[dict], enriched_records: list[dict]) -> dict:
    original_audit = analyze_label_collisions(original_records, "full_clean_input_key")
    
    # Calculate enriched key collisions
    grouped_enriched = {}
    for r in enriched_records:
        k = build_enriched_selector_key(r)
        grouped_enriched[k] = grouped_enriched.get(k, []) + [r]
        
    total_records = len(enriched_records)
    collided_key_count = 0
    records_in_collided_keys = 0
    upper_bound_correct = 0
    
    for k, recs in grouped_enriched.items():
        lbl_counts = {}
        for r in recs:
            lbl = extract_label(r)
            lbl_counts[lbl] = lbl_counts.get(lbl, 0) + 1
            
        unique_lbl_count = len(lbl_counts)
        if unique_lbl_count > 1:
            collided_key_count += 1
            records_in_collided_keys += len(recs)
            
        max_freq = max(lbl_counts.values())
        upper_bound_correct += max_freq
        
    enriched_fraction = safe_divide(records_in_collided_keys, total_records)
    enriched_upper_bound_acc = safe_divide(upper_bound_correct, total_records)
    
    orig_fraction = original_audit["collision_record_fraction"]
    orig_upper_bound_acc = original_audit["deterministic_identifiability_upper_bound_accuracy"]
    
    # Check if collision fraction reduced or upper bound improved
    reduced = (enriched_fraction < orig_fraction)
    improved = (enriched_upper_bound_acc > orig_upper_bound_acc)
    
    return {
        "original_collision_record_fraction": float(orig_fraction),
        "enriched_collision_record_fraction": float(enriched_fraction),
        "collision_fraction_reduced": reduced,
        
        "original_upper_bound_accuracy": float(orig_upper_bound_acc),
        "enriched_upper_bound_accuracy": float(enriched_upper_bound_acc),
        "upper_bound_improved": improved,
        
        "valid_for_final_selector_evidence": False,
        "valid_for_feasibility_collision_reduction_audit": True,
        "diagnostic_pass": True
    }


def interpret_query_observation_enrichment_contract(enrichment_audit: dict, collision_reduction_audit: dict) -> dict:
    improved = collision_reduction_audit["upper_bound_improved"]
    reduced = collision_reduction_audit["collision_fraction_reduced"]
    
    # Since P81 support sets lack delta/result endpoints, it cannot reduce collisions
    feasibility_supported = (improved or reduced)
    
    if feasibility_supported:
        return {
            "query_observation_context_reduces_collisions": True,
            "enrichment_feasibility_supported": True,
            "valid_for_final_selector_evidence": False,
            "reason_final_evidence_blocked": "support_context_constructed_from_audit_label",
            "external_pre_target_query_context_required": True,
            "recommended_next_phase": "P85_externalized_query_context_dataset_or_support_set_contract",
            "diagnostic_pass": True
        }
    else:
        return {
            "query_observation_context_reduces_collisions": False,
            "enrichment_feasibility_supported": False,
            "valid_for_final_selector_evidence": False,
            "reason_final_evidence_blocked": "no_collision_reduction_or_missing_support_observation_summaries",
            "external_pre_target_query_context_required": True,
            "recommended_next_phase": "P85_define_real_external_query_context_source",
            "diagnostic_pass": True
        }


def audit_bridge_boundary_after_query_observation_enrichment(interpretation: dict) -> dict:
    blocking_reasons = [
        "learned_selector_evidence_not_present",
        "learned_metric_evidence_not_present",
        "external_pre_target_query_context_not_yet_available",
        "bridge_input_contract_not_defined",
        "bridge_validation_not_run"
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
        "diagnostic_pass": True
    }


def validate_source_contracts_for_p84() -> dict:
    validated = True
    missing_or_invalid = []
    
    p83_ok = True
    
    try:
        p83 = run_p83_clean_input_identifiability_collision_audit_probe()
        if p83.get("phase") != "P83" or p83.get("verdict") != "P83_READY_FOR_REVIEW":
            p83_ok = False
            missing_or_invalid.append("p83_invalid_phase_or_verdict")
        if p83.get("clean_input_identifiability_audit_performed") is not True:
            p83_ok = False
            missing_or_invalid.append("p83_identifiability_audit_not_done")
        if p83.get("sanity_summary", {}).get("clean_input_relation_identifiability_established") is not False:
            p83_ok = False
            missing_or_invalid.append("p83_identifiability_established_prematurely")
        if p83.get("sanity_summary", {}).get("clean_input_label_collisions_present") is not True:
            p83_ok = False
            missing_or_invalid.append("p83_collisions_missing")
        if p83.get("sanity_summary", {}).get("model_capacity_not_primary_failure") is not True:
            p83_ok = False
            missing_or_invalid.append("p83_capacity_failure_not_flagged")
        if p83.get("sanity_summary", {}).get("additional_observation_context_required") is not True:
            p83_ok = False
            missing_or_invalid.append("p83_additional_context_not_flagged")
        if p83.get("model_training_performed") is not False:
            p83_ok = False
            missing_or_invalid.append("p83_unexpected_model_training")
        if p83.get("torch_training_performed") is not False:
            p83_ok = False
            missing_or_invalid.append("p83_unexpected_torch_training")
        if p83.get("bridge_ready") is not False:
            p83_ok = False
            missing_or_invalid.append("p83_bridge_ready")
    except Exception as e:
        p83_ok = False
        missing_or_invalid.append(f"p83_exception_{str(e)}")
        
    validated = p83_ok
    
    return {
        "source_contracts_validated": validated,
        "p83_validated": p83_ok,
        "p83_clean_input_underidentification_preserved": p83_ok,
        "p83_additional_context_required_preserved": p83_ok,
        "p83_bridge_not_ready_preserved": p83_ok,
        "missing_or_invalid": missing_or_invalid
    }


def run_p84_clean_query_observation_enrichment_contract_probe() -> dict:
    contracts_val = validate_source_contracts_for_p84()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    original_records = build_selector_dataset_records(p70a, p70b)
    enriched_records = build_query_observation_enriched_records(original_records)
    
    # Audit enriched leakage
    has_leakage = False
    endpoint_leak = False
    delta_leak = False
    label_leak = False
    op_leak = False
    metadata_leak = False
    hint_leak = False
    
    for r in enriched_records:
        audit = r["enrichment_leakage_audit"]
        if audit["diagnostic_pass"] is False:
            has_leakage = True
        if audit["target_record_endpoint_used_for_selector_input"] is True:
            endpoint_leak = True
        if audit["target_record_delta_used_for_selector_input"] is True:
            delta_leak = True
        if audit["exact_relation_label_used_for_selector_input"] is True:
            label_leak = True
        if audit["exact_operator_id_used_for_selector_input"] is True:
            op_leak = True
        if audit["audit_metadata_used_for_selector_input"] is True:
            metadata_leak = True
        if audit["relation_specific_hint_used_for_model_input"] is True:
            hint_leak = True
            
    enrichment_audit = {
        "target_record_endpoint_used_for_selector_input": endpoint_leak,
        "target_record_delta_used_for_selector_input": delta_leak,
        "exact_relation_label_used_for_selector_input": label_leak,
        "exact_operator_id_used_for_selector_input": op_leak,
        "audit_metadata_used_for_selector_input": metadata_leak,
        "relation_specific_hint_used_for_model_input": hint_leak,
        "support_context_constructed_from_audit_label": SUPPORT_CONTEXT_CONSTRUCTED_FROM_AUDIT_LABEL,
        "valid_for_final_selector_evidence": VALID_FOR_FINAL_SELECTOR_EVIDENCE,
        "valid_for_feasibility_collision_reduction_audit": VALID_FOR_FEASIBILITY_COLLISION_REDUCTION_AUDIT,
        "diagnostic_pass": not has_leakage
    }
    
    reduction_audit = analyze_enriched_collision_reduction(original_records, enriched_records)
    interpretation = interpret_query_observation_enrichment_contract(enrichment_audit, reduction_audit)
    bridge_audit = audit_bridge_boundary_after_query_observation_enrichment(interpretation)
    
    verdict_str = VERDICT if contracts_ok else "P84_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED is True or TORCH_TRAINING_PERFORMED is True:
            verdict_str = "P84_BLOCKED_BY_UNEXPECTED_TRAINING_OR_MODEL"
        elif has_leakage is True:
            if endpoint_leak or delta_leak:
                verdict_str = "P84_BLOCKED_BY_TARGET_RECORD_LEAKAGE"
            elif label_leak or op_leak or metadata_leak or hint_leak:
                verdict_str = "P84_BLOCKED_BY_LABEL_OR_HINT_PASS_THROUGH"
        elif BRIDGE_READY is True:
            verdict_str = "P84_BLOCKED_BY_PREMATURE_BRIDGE_AUTHORITY"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p83_clean_input_underidentification_preserved": contracts_val["p83_clean_input_underidentification_preserved"],
        "p83_additional_context_required_preserved": contracts_val["p83_additional_context_required_preserved"],
        "p83_bridge_not_ready_preserved": contracts_val["p83_bridge_not_ready_preserved"],

        "query_observation_enrichment_contract_present": QUERY_OBSERVATION_ENRICHMENT_CONTRACT_PRESENT,
        "support_observation_context_defined": SUPPORT_OBSERVATION_CONTEXT_DEFINED,
        "invariant_observation_context_defined": INVARIANT_OBSERVATION_CONTEXT_DEFINED,
        "external_query_context_schema_defined": EXTERNAL_QUERY_CONTEXT_SCHEMA_DEFINED,

        "target_record_endpoint_used_for_selector_input": TARGET_RECORD_ENDPOINT_USED_FOR_SELECTOR_INPUT,
        "target_record_delta_used_for_selector_input": TARGET_RECORD_DELTA_USED_FOR_SELECTOR_INPUT,
        "exact_relation_label_used_for_selector_input": EXACT_RELATION_LABEL_USED_FOR_SELECTOR_INPUT,
        "exact_operator_id_used_for_selector_input": EXACT_OPERATOR_ID_USED_FOR_SELECTOR_INPUT,
        "audit_metadata_used_for_selector_input": AUDIT_METADATA_USED_FOR_SELECTOR_INPUT,
        "relation_specific_hint_used_for_model_input": RELATION_SPECIFIC_HINT_USED_FOR_MODEL_INPUT,

        "synthetic_support_context_for_feasibility_only": SYNTHETIC_SUPPORT_CONTEXT_FOR_FEASIBILITY_ONLY,
        "support_context_constructed_from_audit_label": SUPPORT_CONTEXT_CONSTRUCTED_FROM_AUDIT_LABEL,
        "valid_for_final_selector_evidence": VALID_FOR_FINAL_SELECTOR_EVIDENCE,
        "valid_for_feasibility_collision_reduction_audit": VALID_FOR_FEASIBILITY_COLLISION_REDUCTION_AUDIT,

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_ready": BRIDGE_READY,

        "json_safe": True
    }
    
    sample_enriched = enriched_records[:3] if len(enriched_records) >= 3 else enriched_records
    
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,

        "source_identifiability_phase": SOURCE_IDENTIFIABILITY_PHASE,
        "source_selector_pilot_phase": SOURCE_SELECTOR_PILOT_PHASE,
        "source_dataset_phase": SOURCE_DATASET_PHASE,

        "verdict": verdict_str,

        "training_allowed_by_phase4_authority": TRAINING_ALLOWED_BY_PHASE4_AUTHORITY,
        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "query_observation_enrichment_contract_present": QUERY_OBSERVATION_ENRICHMENT_CONTRACT_PRESENT,
        "support_observation_context_defined": SUPPORT_OBSERVATION_CONTEXT_DEFINED,
        "invariant_observation_context_defined": INVARIANT_OBSERVATION_CONTEXT_DEFINED,
        "external_query_context_schema_defined": EXTERNAL_QUERY_CONTEXT_SCHEMA_DEFINED,

        "target_record_endpoint_used_for_selector_input": TARGET_RECORD_ENDPOINT_USED_FOR_SELECTOR_INPUT,
        "target_record_delta_used_for_selector_input": TARGET_RECORD_DELTA_USED_FOR_SELECTOR_INPUT,
        "exact_relation_label_used_for_selector_input": EXACT_RELATION_LABEL_USED_FOR_SELECTOR_INPUT,
        "exact_operator_id_used_for_selector_input": EXACT_OPERATOR_ID_USED_FOR_SELECTOR_INPUT,
        "audit_metadata_used_for_selector_input": AUDIT_METADATA_USED_FOR_SELECTOR_INPUT,
        "relation_specific_hint_used_for_model_input": RELATION_SPECIFIC_HINT_USED_FOR_MODEL_INPUT,

        "synthetic_support_context_for_feasibility_only": SYNTHETIC_SUPPORT_CONTEXT_FOR_FEASIBILITY_ONLY,
        "support_context_constructed_from_audit_label": SUPPORT_CONTEXT_CONSTRUCTED_FROM_AUDIT_LABEL,
        "valid_for_final_selector_evidence": VALID_FOR_FINAL_SELECTOR_EVIDENCE,
        "valid_for_feasibility_collision_reduction_audit": VALID_FOR_FEASIBILITY_COLLISION_REDUCTION_AUDIT,

        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "bridge_ready": BRIDGE_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,

        "source_contracts_validated": contracts_ok,
        "p83_clean_input_underidentification_preserved": contracts_val["p83_clean_input_underidentification_preserved"],
        "p83_additional_context_required_preserved": contracts_val["p83_additional_context_required_preserved"],
        "p83_bridge_not_ready_preserved": contracts_val["p83_bridge_not_ready_preserved"],

        "dataset_record_count": int(len(original_records)),
        "enriched_record_count": int(len(enriched_records)),
        "enrichment_leakage_audit": enrichment_audit,
        "collision_reduction_audit": reduction_audit,
        "enrichment_contract_interpretation": interpretation,
        "bridge_boundary_after_query_observation_enrichment": bridge_audit,

        "sample_enriched_records": sample_enriched,
        "sample_record_count": int(len(sample_enriched)),

        "sanity_summary": sanity_summary,

        "json_safe": True,
        "diagnostic_only": True
    }
    
    return output
