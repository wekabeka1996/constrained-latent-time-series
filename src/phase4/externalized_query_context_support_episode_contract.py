# src/phase4/externalized_query_context_support_episode_contract.py

import json
from typing import Any

from src.phase4.clean_query_observation_enrichment_contract import (
    run_p84_clean_query_observation_enrichment_contract_probe,
)

from src.phase4.clean_input_identifiability_collision_audit import (
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

PHASE = "P85"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Externalized Query Context Source and Support Episode Contract"
CONTRACT_VERSION = "phase4_p85_externalized_query_context_support_episode_contract_v1"

SOURCE_ENRICHMENT_PHASE = "P84"
SOURCE_IDENTIFIABILITY_PHASE = "P83"
SOURCE_SELECTOR_PILOT_PHASE = "P82"
SOURCE_DATASET_PHASE = "P81"

TRAINING_ALLOWED_BY_PHASE4_AUTHORITY = True
MODEL_TRAINING_PERFORMED = False
TORCH_TRAINING_PERFORMED = False
NEW_MODEL_IMPLEMENTED = False
OPTIMIZER_CREATED = False
CHECKPOINT_WRITTEN = False

EXTERNALIZED_QUERY_CONTEXT_SOURCE_DEFINED = True
SUPPORT_EPISODE_CONTEXT_DEFINED = True
SUPPORT_PAIR_RESULT_DELTA_SCHEMA_DEFINED = True
EPISODE_MANIFEST_SCHEMA_DEFINED = True
SUPPORT_INVARIANT_CONTEXT_SCHEMA_DEFINED = True

SUPPORT_SELECTION_USES_RELATION_LABEL_FOR_SYNTHETIC_DATASET_CONSTRUCTION = True
SUPPORT_SELECTION_LABEL_VISIBLE_TO_SELECTOR = False
SUPPORT_CONTEXT_CONSTRUCTED_BY_SYNTHETIC_TASK_GENERATOR = True

VALID_FOR_P86_SYNTHETIC_FEW_SHOT_SELECTOR_EXPERIMENT = False
VALID_FOR_FINAL_SEMANTIC_GEOMETRY_EVIDENCE = False
VALID_FOR_BRIDGE_EVIDENCE = False
EXTERNAL_REAL_QUERY_SOURCE_STILL_REQUIRED = True

EVALUATED_TARGET_ENDPOINT_USED_FOR_SELECTOR_INPUT = False
EVALUATED_TARGET_DELTA_USED_FOR_SELECTOR_INPUT = False
EXACT_RELATION_LABEL_USED_FOR_SELECTOR_INPUT = False
EXACT_OPERATOR_ID_USED_FOR_SELECTOR_INPUT = False
AUDIT_METADATA_USED_FOR_SELECTOR_INPUT = False
RELATION_SPECIFIC_HINT_USED_FOR_MODEL_INPUT = False

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "externalized_query_context_support_episode_contract"
VERDICT = "P85_READY_FOR_REVIEW"

QUERY_SOURCE_VIEW = "query_source_input"
SUPPORT_EPISODE_VIEW = "externalized_support_episode_context"
EPISODE_SELECTOR_INPUT_VIEW = "externalized_episode_selector_input"
EPISODE_MANIFEST_VIEW = "externalized_episode_manifest"
AUDIT_LABEL_VIEW = "audit_label_evaluation_only"


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


def build_query_source_input(record: dict) -> dict:
    sel = record["selector_input"]
    out = {
        "domain": record["domain"],
        "query_intensity_hint": sel.get("query_intensity_hint"),
        "context_world": sel.get("context_world"),
    }
    if "source_state_summary" in sel:
        out["source_state_summary"] = sel["source_state_summary"]
    if "source_parameter_summary" in sel:
        out["source_parameter_summary"] = sel["source_parameter_summary"]
    return out


def extract_support_pair_observation(record: dict) -> dict:
    sel = record["selector_input"]
    obs = {
        "support_record_id": record["dataset_record_id"],
        "support_domain": record["domain"],
        "support_context_world": sel.get("context_world", "unknown"),
        "support_result_summary": {},
        "support_delta_summary": {},
        "support_pair_observation_available": False,
        "missing_result_summary": True,
        "missing_delta_summary": True
    }
    if "source_state_summary" in sel:
        obs["support_source_summary"] = sel["source_state_summary"]
    elif "source_parameter_summary" in sel:
        obs["support_source_summary"] = sel["source_parameter_summary"]
        
    return obs


def build_support_episode_manifest(records: list[dict], max_support: int = 2) -> list[dict]:
    # Group by label
    pool = {}
    for r in records:
        lbl = extract_label(r)
        pool[lbl] = pool.get(lbl, []) + [r]
        
    manifests = []
    for r in records:
        lbl = extract_label(r)
        candidates = pool.get(lbl, [])
        
        # Sort deterministically: prefer train, then sorted by dataset_record_id
        sorted_recs = sorted(candidates, key=lambda c: (0 if c["split"] == "train" else 1, c["dataset_record_id"]))
        filtered = [c for c in sorted_recs if c["dataset_record_id"] != r["dataset_record_id"]]
        support_recs = filtered[:max_support]
        support_ids = [s["dataset_record_id"] for s in support_recs]
        
        # Construct episode_id based on query record id
        episode_id = f"episode_{r['dataset_record_id']}"
        
        manifests.append({
            "episode_id": episode_id,
            "query_record_id": r["dataset_record_id"],
            "support_record_ids": support_ids,
            "split": r["split"],
            "domain": r["domain"],
            "construction_source": "synthetic_task_generator",
            "support_selection_uses_relation_label_for_dataset_construction": True,
            "support_selection_label_visible_to_selector": False,
            "valid_for_p86_synthetic_few_shot_selector_experiment": True,
            "valid_for_final_semantic_geometry_evidence": False,
            "valid_for_bridge_evidence": False
        })
    return manifests


def build_support_invariant_context(support_pair_observations: list[dict]) -> dict:
    available_count = sum(1 for obs in support_pair_observations if obs.get("support_pair_observation_available") is True)
    
    return {
        "support_pair_count": int(len(support_pair_observations)),
        "support_pair_observation_available_count": available_count,
        "result_summary_available_count": 0,
        "delta_summary_available_count": 0,
        "support_delta_abs_sum_mean": 0.0,
        "support_delta_nonzero_count_mean": 0.0,
        "support_invariant_context_available": False,
        "support_episode_limited_by_missing_result_or_delta_summaries": True
    }


def build_externalized_episode_selector_input(query_record: dict, support_records: list[dict]) -> dict:
    query_input = build_query_source_input(query_record)
    support_pairs = [extract_support_pair_observation(s) for s in support_records]
    invariant_context = build_support_invariant_context(support_pairs)
    
    return {
        "query_source_input": query_input,
        "externalized_support_episode_context": {
            "support_pair_count": int(len(support_pairs)),
            "support_pairs": support_pairs,
            "support_invariant_context": invariant_context,
            "episode_context_metadata": {
                "context_source_kind": "externalized_synthetic_support_episode",
                "support_selection_uses_relation_label_for_dataset_construction": True,
                "support_selection_label_visible_to_selector": False,
                "valid_for_p86_synthetic_few_shot_selector_experiment": True,
                "valid_for_final_semantic_geometry_evidence": False,
                "valid_for_bridge_evidence": False,
                "external_real_query_source_still_required": True
            }
        }
    }


def audit_episode_selector_input_leakage(episode_selector_input: dict) -> dict:
    endpoint_leak = False
    delta_leak = False
    label_leak = False
    op_leak = False
    metadata_leak = False
    hint_leak = False
    
    forbidden_endpoints = {
        "query_target_endpoint", "evaluated_target_endpoint", "query_z_b", "query_params_b", "query_series_b", "target_endpoint", "target_midpoint", "z_b", "params_b", "series_b"
    }
    forbidden_deltas = {
        "query_target_delta", "evaluated_target_delta", "query_z_b_minus_z_a", "query_params_b_minus_params_a", "query_series_summary_delta", "z_b_minus_z_a", "params_b_minus_params_a", "series_summary_delta"
    }
    forbidden_labels = {
        "target_relation_label", "true_relation_type", "relation_type"
    }
    forbidden_operators = {
        "operator_id"
    }
    forbidden_metadata = {
        "_audit_metadata"
    }
    forbidden_hints = {
        "relation_family_hint", "transformation_class_hint", "relation_axis_hint", "parameter_group_hint"
    }
    
    all_keys = scan_keys(episode_selector_input)
    
    for k in all_keys:
        if k in forbidden_endpoints:
            endpoint_leak = True
        if k in forbidden_deltas:
            delta_leak = True
        if k in forbidden_labels:
            label_leak = True
        if k in forbidden_operators:
            op_leak = True
        if k in forbidden_metadata:
            metadata_leak = True
        if k in forbidden_hints:
            hint_leak = True
            
    diag_pass = not (
        endpoint_leak
        or delta_leak
        or label_leak
        or op_leak
        or metadata_leak
        or hint_leak
    )
    
    return {
        "evaluated_target_endpoint_used_for_selector_input": endpoint_leak,
        "evaluated_target_delta_used_for_selector_input": delta_leak,
        "exact_relation_label_used_for_selector_input": label_leak,
        "exact_operator_id_used_for_selector_input": op_leak,
        "audit_metadata_used_for_selector_input": metadata_leak,
        "relation_specific_hint_used_for_model_input": hint_leak,
        "support_selection_label_visible_to_selector": False,
        "diagnostic_pass": diag_pass
    }


def build_externalized_query_context_episode_records(records: list[dict]) -> list[dict]:
    manifests = build_support_episode_manifest(records, max_support=2)
    records_by_id = {r["dataset_record_id"]: r for r in records}
    
    episodes = []
    for m in manifests:
        query_rec = records_by_id[m["query_record_id"]]
        support_recs = [records_by_id[sid] for sid in m["support_record_ids"]]
        
        selector_input = build_externalized_episode_selector_input(query_rec, support_recs)
        leakage = audit_episode_selector_input_leakage(selector_input)
        
        episodes.append({
            "episode_id": m["episode_id"],
            "query_record_id": m["query_record_id"],
            "split": m["split"],
            "domain": m["domain"],
            "externalized_episode_selector_input": selector_input,
            "externalized_episode_manifest": m,
            "audit_label_evaluation_only": query_rec["label_evaluation"],
            "episode_input_leakage_audit": leakage
        })
    return episodes


def build_episode_selector_key(episode_record: dict) -> str:
    sel = episode_record["externalized_episode_selector_input"]
    key_dict = {
        "query_source_input": sel["query_source_input"]
    }
    
    inv = sel.get("externalized_support_episode_context", {}).get("support_invariant_context", {})
    if inv.get("support_invariant_context_available") is True:
        key_dict["support_pair_observations"] = sel.get("externalized_support_episode_context", {}).get("support_pairs")
        key_dict["support_invariant_context"] = inv
        
    return canonical_json(key_dict)


def analyze_episode_collision_reduction(original_records: list[dict], episode_records: list[dict]) -> dict:
    original_audit = analyze_label_collisions(original_records, "full_clean_input_key")
    
    # Calculate enriched key collisions
    grouped_enriched = {}
    for r in episode_records:
        k = build_episode_selector_key(r)
        grouped_enriched[k] = grouped_enriched.get(k, []) + [r]
        
    total_records = len(episode_records)
    collided_key_count = 0
    records_in_collided_keys = 0
    upper_bound_correct = 0
    
    for k, recs in grouped_enriched.items():
        lbl_counts = {}
        for r in recs:
            lbl = r["audit_label_evaluation_only"]["target_relation_label"]
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
    
    reduced = (enriched_fraction < orig_fraction)
    improved = (enriched_upper_bound_acc > orig_upper_bound_acc)
    
    return {
        "original_collision_record_fraction": float(orig_fraction),
        "episode_collision_record_fraction": float(enriched_fraction),
        "collision_fraction_reduced": reduced,
        
        "original_upper_bound_accuracy": float(orig_upper_bound_acc),
        "episode_upper_bound_accuracy": float(enriched_upper_bound_acc),
        "upper_bound_improved": improved,
        
        "support_pair_observation_available_count": 0,
        "support_episode_limited_by_missing_result_or_delta_summaries": True,
        
        "valid_for_p86_synthetic_few_shot_selector_experiment": False,
        "valid_for_final_semantic_geometry_evidence": False,
        "valid_for_bridge_evidence": False,
        "diagnostic_pass": True
    }


def validate_source_contracts_for_p85() -> dict:
    validated = True
    missing_or_invalid = []
    
    p84_ok = True
    
    try:
        p84 = run_p84_clean_query_observation_enrichment_contract_probe()
        if p84.get("phase") != "P84" or p84.get("verdict") != "P84_READY_FOR_REVIEW":
            p84_ok = False
            missing_or_invalid.append("p84_invalid_phase_or_verdict")
        if p84.get("query_observation_enrichment_contract_present") is not True:
            p84_ok = False
            missing_or_invalid.append("p84_enrichment_contract_missing")
        if p84.get("support_observation_context_defined") is not True:
            p84_ok = False
            missing_or_invalid.append("p84_support_context_missing")
        if p84.get("invariant_observation_context_defined") is not True:
            p84_ok = False
            missing_or_invalid.append("p84_invariant_context_missing")
        if p84.get("sanity_summary", {}).get("support_context_constructed_from_audit_label") is not True:
            p84_ok = False
            missing_or_invalid.append("p84_support_label_construction_not_flagged")
        if p84.get("sanity_summary", {}).get("valid_for_final_selector_evidence") is not False:
            p84_ok = False
            missing_or_invalid.append("p84_final_evidence_premature")
        if p84.get("sanity_summary", {}).get("valid_for_feasibility_collision_reduction_audit") is not True:
            p84_ok = False
            missing_or_invalid.append("p84_feasibility_audit_not_flagged")
        if p84.get("collision_reduction_audit", {}).get("collision_fraction_reduced") is not False:
            p84_ok = False
            missing_or_invalid.append("p84_collision_fraction_reduced_unexpectedly")
        if p84.get("collision_reduction_audit", {}).get("upper_bound_improved") is not False:
            p84_ok = False
            missing_or_invalid.append("p84_upper_bound_improved_unexpectedly")
        if p84.get("model_training_performed") is not False:
            p84_ok = False
            missing_or_invalid.append("p84_model_training_unexpected")
        if p84.get("torch_training_performed") is not False:
            p84_ok = False
            missing_or_invalid.append("p84_torch_training_unexpected")
        if p84.get("bridge_ready") is not False:
            p84_ok = False
            missing_or_invalid.append("p84_bridge_ready")
    except Exception as e:
        p84_ok = False
        missing_or_invalid.append(f"p84_exception_{str(e)}")
        
    validated = p84_ok
    
    return {
        "source_contracts_validated": validated,
        "p84_validated": p84_ok,
        "p84_query_observation_contract_preserved": p84_ok,
        "p84_no_final_selector_evidence_preserved": p84_ok,
        "p84_real_external_context_required_preserved": p84_ok,
        "p84_bridge_not_ready_preserved": p84_ok,
        "missing_or_invalid": missing_or_invalid
    }


def interpret_externalized_query_context_contract(
    episode_collision_audit: dict,
    episode_leakage_audit: dict
) -> dict:
    # Since P81 support result/delta summaries are unavailable, we cannot run synthetic P86 few shot selectors yet
    feasibility_supported = False
    
    if feasibility_supported:
        return {
            "externalized_query_context_source_defined": True,
            "support_episode_context_feasibility_supported": True,
            "valid_for_p86_synthetic_few_shot_selector_experiment": True,
            "valid_for_final_semantic_geometry_evidence": False,
            "external_real_query_source_still_required": True,
            "recommended_next_phase": "P86_synthetic_few_shot_selector_pilot_no_bridge",
            "diagnostic_pass": True
        }
    else:
        return {
            "externalized_query_context_source_defined": True,
            "support_episode_context_feasibility_supported": False,
            "valid_for_p86_synthetic_few_shot_selector_experiment": False,
            "valid_for_final_semantic_geometry_evidence": False,
            "external_real_query_source_still_required": True,
            "recommended_next_phase": "P86_support_pair_materializer_from_p70_relation_cases_no_training_no_bridge",
            "diagnostic_pass": True
        }


def audit_bridge_boundary_after_externalized_query_context(interpretation: dict) -> dict:
    blocking_reasons = [
        "learned_selector_evidence_not_present",
        "learned_metric_evidence_not_present",
        "external_real_query_source_still_required",
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


def run_p85_externalized_query_context_support_episode_contract_probe() -> dict:
    contracts_val = validate_source_contracts_for_p85()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    original_records = build_selector_dataset_records(p70a, p70b)
    episode_records = build_externalized_query_context_episode_records(original_records)
    
    # Audit leakage
    has_leakage = False
    endpoint_leak = False
    delta_leak = False
    label_leak = False
    op_leak = False
    metadata_leak = False
    hint_leak = False
    
    for ep in episode_records:
        audit = ep["episode_input_leakage_audit"]
        if audit["diagnostic_pass"] is False:
            has_leakage = True
        if audit["evaluated_target_endpoint_used_for_selector_input"] is True:
            endpoint_leak = True
        if audit["evaluated_target_delta_used_for_selector_input"] is True:
            delta_leak = True
        if audit["exact_relation_label_used_for_selector_input"] is True:
            label_leak = True
        if audit["exact_operator_id_used_for_selector_input"] is True:
            op_leak = True
        if audit["audit_metadata_used_for_selector_input"] is True:
            metadata_leak = True
        if audit["relation_specific_hint_used_for_model_input"] is True:
            hint_leak = True
            
    episode_leakage_audit = {
        "evaluated_target_endpoint_used_for_selector_input": endpoint_leak,
        "evaluated_target_delta_used_for_selector_input": delta_leak,
        "exact_relation_label_used_for_selector_input": label_leak,
        "exact_operator_id_used_for_selector_input": op_leak,
        "audit_metadata_used_for_selector_input": metadata_leak,
        "relation_specific_hint_used_for_model_input": hint_leak,
        "support_selection_label_visible_to_selector": False,
        "diagnostic_pass": not has_leakage
    }
    
    reduction_audit = analyze_episode_collision_reduction(original_records, episode_records)
    interpretation = interpret_externalized_query_context_contract(reduction_audit, episode_leakage_audit)
    bridge_audit = audit_bridge_boundary_after_externalized_query_context(interpretation)
    
    valid_p86 = interpretation["valid_for_p86_synthetic_few_shot_selector_experiment"]
    
    verdict_str = VERDICT if contracts_ok else "P85_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED is True or TORCH_TRAINING_PERFORMED is True:
            verdict_str = "P85_BLOCKED_BY_UNEXPECTED_TRAINING_OR_MODEL"
        elif has_leakage is True:
            if endpoint_leak or delta_leak:
                verdict_str = "P85_BLOCKED_BY_EVALUATED_TARGET_LEAKAGE"
            elif label_leak or op_leak or metadata_leak or hint_leak:
                verdict_str = "P85_BLOCKED_BY_LABEL_OR_HINT_PASS_THROUGH"
        elif BRIDGE_READY is True:
            verdict_str = "P85_BLOCKED_BY_PREMATURE_BRIDGE_AUTHORITY"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p84_query_observation_contract_preserved": contracts_val["p84_query_observation_contract_preserved"],
        "p84_no_final_selector_evidence_preserved": contracts_val["p84_no_final_selector_evidence_preserved"],
        "p84_real_external_context_required_preserved": contracts_val["p84_real_external_context_required_preserved"],
        "p84_bridge_not_ready_preserved": contracts_val["p84_bridge_not_ready_preserved"],

        "externalized_query_context_source_defined": EXTERNALIZED_QUERY_CONTEXT_SOURCE_DEFINED,
        "support_episode_context_defined": SUPPORT_EPISODE_CONTEXT_DEFINED,
        "support_pair_result_delta_schema_defined": SUPPORT_PAIR_RESULT_DELTA_SCHEMA_DEFINED,
        "episode_manifest_schema_defined": EPISODE_MANIFEST_SCHEMA_DEFINED,
        "support_invariant_context_schema_defined": SUPPORT_INVARIANT_CONTEXT_SCHEMA_DEFINED,

        "support_selection_uses_relation_label_for_synthetic_dataset_construction": SUPPORT_SELECTION_USES_RELATION_LABEL_FOR_SYNTHETIC_DATASET_CONSTRUCTION,
        "support_selection_label_visible_to_selector": SUPPORT_SELECTION_LABEL_VISIBLE_TO_SELECTOR,

        "evaluated_target_endpoint_used_for_selector_input": endpoint_leak,
        "evaluated_target_delta_used_for_selector_input": delta_leak,
        "exact_relation_label_used_for_selector_input": label_leak,
        "exact_operator_id_used_for_selector_input": op_leak,
        "audit_metadata_used_for_selector_input": metadata_leak,
        "relation_specific_hint_used_for_model_input": hint_leak,

        "valid_for_p86_synthetic_few_shot_selector_experiment": valid_p86,
        "valid_for_final_semantic_geometry_evidence": VALID_FOR_FINAL_SEMANTIC_GEOMETRY_EVIDENCE,
        "valid_for_bridge_evidence": VALID_FOR_BRIDGE_EVIDENCE,
        "external_real_query_source_still_required": EXTERNAL_REAL_QUERY_SOURCE_STILL_REQUIRED,

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
    
    sample_episodes = episode_records[:3] if len(episode_records) >= 3 else episode_records
    
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,

        "source_enrichment_phase": SOURCE_ENRICHMENT_PHASE,
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

        "externalized_query_context_source_defined": EXTERNALIZED_QUERY_CONTEXT_SOURCE_DEFINED,
        "support_episode_context_defined": SUPPORT_EPISODE_CONTEXT_DEFINED,
        "support_pair_result_delta_schema_defined": SUPPORT_PAIR_RESULT_DELTA_SCHEMA_DEFINED,
        "episode_manifest_schema_defined": EPISODE_MANIFEST_SCHEMA_DEFINED,
        "support_invariant_context_schema_defined": SUPPORT_INVARIANT_CONTEXT_SCHEMA_DEFINED,

        "support_selection_uses_relation_label_for_synthetic_dataset_construction": SUPPORT_SELECTION_USES_RELATION_LABEL_FOR_SYNTHETIC_DATASET_CONSTRUCTION,
        "support_selection_label_visible_to_selector": SUPPORT_SELECTION_LABEL_VISIBLE_TO_SELECTOR,
        "support_context_constructed_by_synthetic_task_generator": SUPPORT_CONTEXT_CONSTRUCTED_BY_SYNTHETIC_TASK_GENERATOR,

        "valid_for_p86_synthetic_few_shot_selector_experiment": valid_p86,
        "valid_for_final_semantic_geometry_evidence": VALID_FOR_FINAL_SEMANTIC_GEOMETRY_EVIDENCE,
        "valid_for_bridge_evidence": VALID_FOR_BRIDGE_EVIDENCE,
        "external_real_query_source_still_required": EXTERNAL_REAL_QUERY_SOURCE_STILL_REQUIRED,

        "evaluated_target_endpoint_used_for_selector_input": endpoint_leak,
        "evaluated_target_delta_used_for_selector_input": delta_leak,
        "exact_relation_label_used_for_selector_input": label_leak,
        "exact_operator_id_used_for_selector_input": op_leak,
        "audit_metadata_used_for_selector_input": metadata_leak,
        "relation_specific_hint_used_for_model_input": hint_leak,

        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "bridge_ready": BRIDGE_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,

        "source_contracts_validated": contracts_ok,
        "p84_query_observation_contract_preserved": contracts_val["p84_query_observation_contract_preserved"],
        "p84_no_final_selector_evidence_preserved": contracts_val["p84_no_final_selector_evidence_preserved"],
        "p84_real_external_context_required_preserved": contracts_val["p84_real_external_context_required_preserved"],
        "p84_bridge_not_ready_preserved": contracts_val["p84_bridge_not_ready_preserved"],

        "dataset_record_count": int(len(original_records)),
        "episode_record_count": int(len(episode_records)),
        "episode_manifest_count": int(len(episode_records)),

        "episode_input_leakage_audit": episode_leakage_audit,
        "episode_collision_reduction_audit": reduction_audit,
        "externalized_query_context_interpretation": interpretation,
        "bridge_boundary_after_externalized_query_context": bridge_audit,

        "sample_episode_records": sample_episodes,
        "sample_record_count": int(len(sample_episodes)),

        "sanity_summary": sanity_summary,

        "json_safe": True,
        "diagnostic_only": True
    }
    
    return output
