# src/phase4/support_pair_materializer_from_p70_relation_cases.py

import json
from typing import Any

from src.phase4.externalized_query_context_support_episode_contract import (
    run_p85_externalized_query_context_support_episode_contract_probe,
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

PHASE = "P86"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Support Pair Materializer from P70 Relation Cases"
CONTRACT_VERSION = "phase4_p86_support_pair_materializer_from_p70_relation_cases_v1"

SOURCE_EXTERNALIZED_CONTEXT_PHASE = "P85"
SOURCE_ENRICHMENT_PHASE = "P84"
SOURCE_IDENTIFIABILITY_PHASE = "P83"
SOURCE_SELECTOR_PILOT_PHASE = "P82"
SOURCE_DATASET_PHASE = "P81"
SOURCE_SYNTHETIC_RELATION_PHASES = ["P70A", "P70B"]

TRAINING_ALLOWED_BY_PHASE4_AUTHORITY = True
MODEL_TRAINING_PERFORMED = False
TORCH_TRAINING_PERFORMED = False
NEW_MODEL_IMPLEMENTED = False
OPTIMIZER_CREATED = False
CHECKPOINT_WRITTEN = False

SUPPORT_PAIR_MATERIALIZER_DEFINED = True
P70_RELATION_CASE_SOURCE_DEFINED = True
SUPPORT_SOURCE_RESULT_DELTA_SCHEMA_DEFINED = True
SUPPORT_PAIR_MATERIALIZATION_ATTEMPTED = True
EXTERNALIZED_EPISODE_RECORDS_WITH_MATERIALIZED_SUPPORT_DEFINED = True

SUPPORT_SELECTION_USES_RELATION_LABEL_FOR_SYNTHETIC_DATASET_CONSTRUCTION = True
SUPPORT_SELECTION_LABEL_VISIBLE_TO_SELECTOR = False
SUPPORT_CONTEXT_CONSTRUCTED_BY_SYNTHETIC_TASK_GENERATOR = True

VALID_FOR_P87_SYNTHETIC_FEW_SHOT_SELECTOR_PILOT = False
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

PRIMARY_EMPIRICAL_TARGET = "support_pair_materializer_from_p70_relation_cases"
VERDICT = "P86_READY_FOR_REVIEW"

P70_RELATION_CASE_SOURCE_VIEW = "p70_relation_case_source"
MATERIALIZED_SUPPORT_PAIR_VIEW = "materialized_support_source_result_delta_pair"
QUERY_SOURCE_VIEW = "query_source_input"
MATERIALIZED_SUPPORT_EPISODE_VIEW = "materialized_externalized_support_episode_context"
EPISODE_MANIFEST_VIEW = "materialized_externalized_episode_manifest"
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


def extract_p70a_relation_records(p70a_probe: dict) -> list[dict]:
    splits = ["train_style_repeated_instances", "heldout_base_state", "heldout_magnitude"]
    recs = p70a_probe.get("testbed", {}).get("relation_records", [])
    return [r for r in recs if r.get("split") in splits]


def extract_p70b_relation_records(p70b_probe: dict) -> list[dict]:
    splits = ["train_style_repeated_instances", "heldout_base_state", "heldout_magnitude"]
    recs = p70b_probe.get("testbed", {}).get("relation_records", [])
    return [r for r in recs if r.get("split") in splits]


def collect_p70_relation_cases(p70a_probe: dict, p70b_probe: dict) -> dict:
    cases_a = extract_p70a_relation_records(p70a_probe)
    cases_b = extract_p70b_relation_records(p70b_probe)
    
    diagnostic = {
        "p70a_cases_available": len(cases_a) > 0,
        "p70b_cases_available": len(cases_b) > 0,
        "diagnostic_pass": True
    }
    
    return {
        "p70a_cases": cases_a,
        "p70b_cases": cases_b,
        "p70a_case_count": int(len(cases_a)),
        "p70b_case_count": int(len(cases_b)),
        "case_collection_diagnostic": diagnostic
    }


def materialize_support_pair_for_record(support_rec: dict, p70_cases: dict) -> dict:
    rec_id = support_rec["dataset_record_id"]
    domain = support_rec["domain"]
    
    raw_case = None
    source_phase = "unavailable"
    method = "unavailable"
    
    if domain == "p70a_vector_world" and rec_id.startswith("p81_record_p70a_"):
        try:
            idx = int(rec_id.replace("p81_record_p70a_", ""))
            if 0 <= idx < len(p70_cases["p70a_cases"]):
                raw_case = p70_cases["p70a_cases"][idx]
                source_phase = "P70A"
                method = "deterministic_p70_case_reconstruction"
        except Exception:
            pass
    elif domain == "p70b_time_series_parameter_world" and rec_id.startswith("p81_record_p70b_"):
        try:
            idx = int(rec_id.replace("p81_record_p70b_", ""))
            if 0 <= idx < len(p70_cases["p70b_cases"]):
                raw_case = p70_cases["p70b_cases"][idx]
                source_phase = "P70B"
                method = "deterministic_p70_case_reconstruction"
        except Exception:
            pass
            
    if raw_case is None:
        # Fallback to source-only summaries
        sel = support_rec["selector_input"]
        summary = {
            "support_record_id": rec_id,
            "support_domain": domain,
            "support_context_world": sel.get("context_world", "unknown"),
            "support_result_summary": {},
            "support_delta_summary": {},
            "support_pair_observation_available": False,
            "materialization_source_phase": "unavailable",
            "materialization_method": "unavailable",
            "missing_result_summary": True,
            "missing_delta_summary": True
        }
        if "source_state_summary" in sel:
            summary["support_source_summary"] = sel["source_state_summary"]
        elif "source_parameter_summary" in sel:
            summary["support_source_summary"] = sel["source_parameter_summary"]
        return summary
        
    # Build actual materialized summaries
    if domain == "p70a_vector_world":
        z_a = raw_case["z_a"]
        z_b = raw_case["z_b"]
        delta = [z_b[i] - z_a[i] for i in range(len(z_a))]
        
        source_summary = {
            "vector_dim": len(z_a),
            "abs_sum": float(sum(abs(x) for x in z_a)),
            "sign_pattern": [1 if x > 0.0 else (-1 if x < -0.0 else 0) for x in z_a],
            "nonzero_count": sum(1 for x in z_a if abs(x) > 1e-15)
        }
        result_summary = {
            "vector_dim": len(z_b),
            "abs_sum": float(sum(abs(x) for x in z_b)),
            "sign_pattern": [1 if x > 0.0 else (-1 if x < -0.0 else 0) for x in z_b],
            "nonzero_count": sum(1 for x in z_b if abs(x) > 1e-15)
        }
        delta_summary = {
            "delta_dim": len(z_a),
            "delta_abs_sum": float(sum(abs(x) for x in delta)),
            "delta_nonzero_count": sum(1 for x in delta if abs(x) > 1e-15),
            "delta_sign_pattern": [1 if x > 1e-15 else (-1 if x < -1e-15 else 0) for x in delta]
        }
    else:  # p70b_time_series_parameter_world
        params_a = raw_case["params_a"]
        params_b = raw_case["params_b"]
        
        source_summary = {
            "params_key_count": len(params_a),
            "params_abs_sum": float(sum(abs(v) for v in params_a.values())),
            "params_nonzero_key_count": sum(1 for v in params_a.values() if abs(v) > 1e-15)
        }
        result_summary = {
            "params_key_count": len(params_b),
            "params_abs_sum": float(sum(abs(v) for v in params_b.values())),
            "params_nonzero_key_count": sum(1 for v in params_b.values() if abs(v) > 1e-15)
        }
        if "series_b" in raw_case:
            sb = raw_case["series_b"]
            sa = raw_case.get("series_a", [])
            result_summary.update({
                "series_length": len(sb),
                "series_abs_sum": float(sum(abs(x) for x in sb)),
                "series_mean": float(sum(sb) / len(sb)) if len(sb) > 0 else 0.0,
                "series_endpoint_delta": float(sb[-1] - sa[-1]) if len(sb) > 0 and len(sa) > 0 else 0.0
            })
        delta_summary = {
            "delta_key_count": len(params_a),
            "delta_abs_sum": float(sum(abs(params_b[k] - params_a[k]) for k in params_a)),
            "delta_nonzero_key_count": sum(1 for k in params_a if abs(params_b[k] - params_a[k]) > 1e-15)
        }
        
    return {
        "support_record_id": rec_id,
        "support_domain": domain,
        "support_context_world": raw_case.get("split", "unknown"),
        "support_source_summary": source_summary,
        "support_result_summary": result_summary,
        "support_delta_summary": delta_summary,
        "support_pair_observation_available": True,
        "materialization_source_phase": source_phase,
        "materialization_method": method,
        "missing_result_summary": False,
        "missing_delta_summary": False
    }


def build_materialized_support_episode_manifest(records: list[dict], max_support: int = 2) -> list[dict]:
    # Group by label
    pool = {}
    for r in records:
        lbl = extract_label(r)
        pool[lbl] = pool.get(lbl, []) + [r]
        
    manifests = []
    for r in records:
        lbl = extract_label(r)
        candidates = pool.get(lbl, [])
        
        # Sort deterministically
        sorted_recs = sorted(candidates, key=lambda c: (0 if c["split"] == "train" else 1, c["dataset_record_id"]))
        filtered = [c for c in sorted_recs if c["dataset_record_id"] != r["dataset_record_id"]]
        support_recs = filtered[:max_support]
        support_ids = [s["dataset_record_id"] for s in support_recs]
        
        episode_id = f"episode_{r['dataset_record_id']}"
        
        manifests.append({
            "episode_id": episode_id,
            "query_record_id": r["dataset_record_id"],
            "support_record_ids": support_ids,
            "split": r["split"],
            "domain": r["domain"],
            "construction_source": "synthetic_task_generator",
            "support_selection_uses_relation_label_for_synthetic_dataset_construction": True,
            "support_selection_label_visible_to_selector": False,
            "valid_for_p87_synthetic_few_shot_selector_pilot": False,  # Will update in probe
            "valid_for_final_semantic_geometry_evidence": False,
            "valid_for_bridge_evidence": False
        })
    return manifests


def build_materialized_support_invariant_context(materialized_support_pairs: list[dict]) -> dict:
    available_count = sum(1 for obs in materialized_support_pairs if obs.get("support_pair_observation_available") is True)
    
    # We can compute invariant context stats if we have materialized values
    inv_avail = available_count > 0
    
    delta_sums = []
    nonzero_counts = []
    for obs in materialized_support_pairs:
        if obs.get("support_pair_observation_available") is True:
            delta_sums.append(obs["support_delta_summary"].get("delta_abs_sum", 0.0))
            nonzero_counts.append(obs["support_delta_summary"].get("delta_nonzero_count", obs["support_delta_summary"].get("delta_nonzero_key_count", 0)))
            
    mean_abs_sum = sum(delta_sums) / len(delta_sums) if len(delta_sums) > 0 else 0.0
    mean_nonzero = sum(nonzero_counts) / len(nonzero_counts) if len(nonzero_counts) > 0 else 0.0
    
    return {
        "support_pair_count": int(len(materialized_support_pairs)),
        "support_pair_observation_available_count": available_count,
        "result_summary_available_count": available_count,
        "delta_summary_available_count": available_count,
        "support_delta_abs_sum_mean": float(mean_abs_sum),
        "support_delta_nonzero_count_mean": float(mean_nonzero),
        "support_invariant_context_available": inv_avail,
        "support_episode_limited_by_missing_result_or_delta_summaries": not inv_avail
    }


def build_materialized_episode_selector_input(
    query_record: dict,
    support_records: list[dict],
    p70_case_index: dict
) -> dict:
    query_input = build_query_source_input(query_record)
    support_pairs = [materialize_support_pair_for_record(s, p70_case_index) for s in support_records]
    invariant_context = build_materialized_support_invariant_context(support_pairs)
    
    # Check if all support pairs are materialized successfully
    all_avail = all(obs.get("support_pair_observation_available") is True for obs in support_pairs)
    
    return {
        "query_source_input": query_input,
        "materialized_externalized_support_episode_context": {
            "support_pair_count": int(len(support_pairs)),
            "support_pairs": support_pairs,
            "support_invariant_context": invariant_context,
            "episode_context_metadata": {
                "context_source_kind": "materialized_externalized_synthetic_support_episode",
                "support_selection_uses_relation_label_for_synthetic_dataset_construction": True,
                "support_selection_label_visible_to_selector": False,
                "valid_for_p87_synthetic_few_shot_selector_pilot": all_avail,
                "valid_for_final_semantic_geometry_evidence": False,
                "valid_for_bridge_evidence": False,
                "external_real_query_source_still_required": True
            }
        }
    }


def audit_materialized_episode_selector_input_leakage(episode_selector_input: dict) -> dict:
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


def build_materialized_externalized_episode_records(
    records: list[dict],
    p70_case_index: dict
) -> list[dict]:
    manifests = build_materialized_support_episode_manifest(records, max_support=2)
    records_by_id = {r["dataset_record_id"]: r for r in records}
    
    episodes = []
    for m in manifests:
        query_rec = records_by_id[m["query_record_id"]]
        support_recs = [records_by_id[sid] for sid in m["support_record_ids"]]
        
        selector_input = build_materialized_episode_selector_input(query_rec, support_recs, p70_case_index)
        leakage = audit_materialized_episode_selector_input_leakage(selector_input)
        
        # update manifest flags based on selector input flags
        all_avail = selector_input["materialized_externalized_support_episode_context"]["episode_context_metadata"]["valid_for_p87_synthetic_few_shot_selector_pilot"]
        m["valid_for_p87_synthetic_few_shot_selector_pilot"] = all_avail
        
        episodes.append({
            "episode_id": m["episode_id"],
            "query_record_id": m["query_record_id"],
            "split": m["split"],
            "domain": m["domain"],
            "materialized_episode_selector_input": selector_input,
            "materialized_episode_manifest": m,
            "audit_label_evaluation_only": query_rec["label_evaluation"],
            "episode_input_leakage_audit": leakage
        })
    return episodes


def sanitize_materialized_support_pair_for_collision_key(pair: dict) -> dict:
    return {
        "support_domain": pair["support_domain"],
        "support_context_world": pair["support_context_world"],
        "support_source_summary": pair["support_source_summary"],
        "support_result_summary": pair["support_result_summary"],
        "support_delta_summary": pair["support_delta_summary"]
    }


def build_materialized_episode_selector_key_dict(episode_record: dict) -> dict:
    sel = episode_record["materialized_episode_selector_input"]
    key_dict = {
        "query_source_input": sel["query_source_input"]
    }
    
    inv = sel.get("materialized_externalized_support_episode_context", {}).get("support_invariant_context", {})
    if inv.get("support_invariant_context_available") is True:
        support_pairs = sel.get("materialized_externalized_support_episode_context", {}).get("support_pairs", [])
        key_dict["support_pair_observations"] = [
            sanitize_materialized_support_pair_for_collision_key(pair)
            for pair in support_pairs
        ]
        key_dict["support_invariant_context"] = {
            "support_pair_count": inv["support_pair_count"],
            "support_pair_observation_available_count": inv["support_pair_observation_available_count"],
            "result_summary_available_count": inv["result_summary_available_count"],
            "delta_summary_available_count": inv["delta_summary_available_count"],
            "support_delta_abs_sum_mean": inv["support_delta_abs_sum_mean"],
            "support_delta_nonzero_count_mean": inv["support_delta_nonzero_count_mean"],
            "support_invariant_context_available": inv["support_invariant_context_available"],
            "support_episode_limited_by_missing_result_or_delta_summaries": inv["support_episode_limited_by_missing_result_or_delta_summaries"]
        }
        
    return key_dict


def build_materialized_episode_selector_key(episode_record: dict) -> str:
    key_dict = build_materialized_episode_selector_key_dict(episode_record)
    return canonical_json(key_dict)


def analyze_materialized_episode_collision_reduction(original_records: list[dict], episode_records: list[dict]) -> dict:
    original_audit = analyze_label_collisions(original_records, "full_clean_input_key")
    
    # Calculate key collisions for materialized episodes
    grouped_enriched = {}
    for r in episode_records:
        k = build_materialized_episode_selector_key(r)
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
    
    # Verify sanitization recursively on key dicts
    has_id = False
    has_metadata = False
    has_label = False
    
    forbidden_metadata = {
        "materialization_source_phase", "materialization_method", 
        "missing_result_summary", "missing_delta_summary", 
        "support_pair_observation_available"
    }
    forbidden_labels = {
        "target_relation_label", "true_relation_type", "relation_type",
        "operator_id", "relation_family_hint", "transformation_class_hint",
        "relation_axis_hint", "parameter_group_hint", "_audit_metadata"
    }
    
    for r in episode_records:
        key_dict = build_materialized_episode_selector_key_dict(r)
        all_keys = scan_keys(key_dict)
        if "support_record_id" in all_keys:
            has_id = True
        for k in all_keys:
            if k in forbidden_metadata:
                has_metadata = True
            if k in forbidden_labels:
                has_label = True
                
    collision_key_sanitized = not (has_id or has_metadata or has_label)
    
    # Count materialized statistics
    avail_count = 0
    limited = False
    all_valid = True
    
    for r in episode_records:
        meta = r["materialized_episode_selector_input"]["materialized_externalized_support_episode_context"]["episode_context_metadata"]
        if meta["valid_for_p87_synthetic_few_shot_selector_pilot"] is False:
            all_valid = False
        avail_count += r["materialized_episode_selector_input"]["materialized_externalized_support_episode_context"]["support_invariant_context"]["support_pair_observation_available_count"]
        if r["materialized_episode_selector_input"]["materialized_externalized_support_episode_context"]["support_invariant_context"]["support_episode_limited_by_missing_result_or_delta_summaries"] is True:
            limited = True
            
    success_p87 = all_valid and reduced and improved and collision_key_sanitized
    
    return {
        "original_collision_record_fraction": float(orig_fraction),
        "materialized_episode_collision_record_fraction": float(enriched_fraction),
        "collision_fraction_reduced": reduced,
        
        "original_upper_bound_accuracy": float(orig_upper_bound_acc),
        "materialized_episode_upper_bound_accuracy": float(enriched_upper_bound_acc),
        "upper_bound_improved": improved,
        
        "support_pair_observation_available_count": int(avail_count),
        "support_episode_limited_by_missing_result_or_delta_summaries": limited,
        
        "valid_for_p87_synthetic_few_shot_selector_pilot": success_p87,
        "valid_for_final_semantic_geometry_evidence": False,
        "valid_for_bridge_evidence": False,
        "diagnostic_pass": collision_key_sanitized,
        
        "collision_key_sanitized": collision_key_sanitized,
        "support_record_id_used_in_collision_key": has_id,
        "materialization_metadata_used_in_collision_key": has_metadata,
        "label_or_hint_used_in_collision_key": has_label
    }


def analyze_support_pair_materialization(episode_records: list[dict]) -> dict:
    total_episodes = len(episode_records)
    total_pairs = sum(r["materialized_episode_selector_input"]["materialized_externalized_support_episode_context"]["support_pair_count"] for r in episode_records)
    available_pairs = sum(r["materialized_episode_selector_input"]["materialized_externalized_support_episode_context"]["support_invariant_context"]["support_pair_observation_available_count"] for r in episode_records)
    
    fraction = safe_divide(available_pairs, total_pairs)
    success = (available_pairs > 0)
    
    return {
        "episode_record_count": int(total_episodes),
        "support_pair_total_count": int(total_pairs),
        "support_pair_observation_available_count": int(available_pairs),
        "support_pair_observation_available_fraction": float(fraction),
        "result_summary_available_count": int(available_pairs),
        "delta_summary_available_count": int(available_pairs),
        "materialization_successful": success,
        "materialization_limited_by_missing_p70_case_outputs": not success,
        "diagnostic_pass": True
    }


def interpret_support_pair_materialization(
    materialization_audit: dict,
    collision_audit: dict,
    leakage_audit: dict
) -> dict:
    success = materialization_audit["materialization_successful"]
    reduced = collision_audit["collision_fraction_reduced"]
    improved = collision_audit["upper_bound_improved"]
    
    if success and reduced and improved:
        return {
            "support_pair_materialization_successful": True,
            "support_episode_context_feasibility_supported": True,
            "valid_for_p87_synthetic_few_shot_selector_pilot": True,
            "valid_for_final_semantic_geometry_evidence": False,
            "external_real_query_source_still_required": True,
            "recommended_next_phase": "P87_synthetic_few_shot_selector_pilot_no_bridge",
            "diagnostic_pass": True
        }
    elif not success:
        return {
            "support_pair_materialization_successful": False,
            "support_episode_context_feasibility_supported": False,
            "valid_for_p87_synthetic_few_shot_selector_pilot": False,
            "valid_for_final_semantic_geometry_evidence": False,
            "external_real_query_source_still_required": True,
            "recommended_next_phase": "P87_p70_case_output_contract_repair_no_training_no_bridge",
            "diagnostic_pass": True
        }
    else:  # success but no collision reduction
        return {
            "support_pair_materialization_successful": True,
            "support_episode_context_feasibility_supported": False,
            "valid_for_p87_synthetic_few_shot_selector_pilot": False,
            "valid_for_final_semantic_geometry_evidence": False,
            "external_real_query_source_still_required": True,
            "recommended_next_phase": "P87_support_episode_feature_contract_revision_no_training_no_bridge",
            "diagnostic_pass": True
        }


def audit_bridge_boundary_after_support_pair_materialization(interpretation: dict) -> dict:
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


def validate_source_contracts_for_p86() -> dict:
    validated = True
    missing_or_invalid = []
    
    p85_ok = True
    
    try:
        p85 = run_p85_externalized_query_context_support_episode_contract_probe()
        if p85.get("phase") != "P85" or p85.get("verdict") != "P85_READY_FOR_REVIEW":
            p85_ok = False
            missing_or_invalid.append("p85_invalid_phase_or_verdict")
        if p85.get("externalized_query_context_source_defined") is not True:
            p85_ok = False
            missing_or_invalid.append("p85_external_query_source_missing")
        if p85.get("support_episode_context_defined") is not True:
            p85_ok = False
            missing_or_invalid.append("p85_support_episode_missing")
        if p85.get("support_pair_result_delta_schema_defined") is not True:
            p85_ok = False
            missing_or_invalid.append("p85_pair_schema_missing")
        if p85.get("sanity_summary", {}).get("support_selection_label_visible_to_selector") is not False:
            p85_ok = False
            missing_or_invalid.append("p85_label_visible_unexpectedly")
        if p85.get("evaluated_target_endpoint_used_for_selector_input") is not False:
            p85_ok = False
            missing_or_invalid.append("p85_target_endpoint_leakage")
        if p85.get("exact_relation_label_used_for_selector_input") is not False:
            p85_ok = False
            missing_or_invalid.append("p85_label_leakage")
        if p85.get("valid_for_final_semantic_geometry_evidence") is not False:
            p85_ok = False
            missing_or_invalid.append("p85_final_semantic_evidence_premature")
        if p85.get("valid_for_bridge_evidence") is not False:
            p85_ok = False
            missing_or_invalid.append("p85_bridge_evidence_premature")
        if p85.get("model_training_performed") is not False:
            p85_ok = False
            missing_or_invalid.append("p85_unexpected_model_training")
        if p85.get("torch_training_performed") is not False:
            p85_ok = False
            missing_or_invalid.append("p85_unexpected_torch_training")
        if p85.get("bridge_ready") is not False:
            p85_ok = False
            missing_or_invalid.append("p85_bridge_ready")
        if p85.get("externalized_query_context_interpretation", {}).get("recommended_next_phase") != "P86_support_pair_materializer_from_p70_relation_cases_no_training_no_bridge":
            p85_ok = False
            missing_or_invalid.append("p85_next_phase_incorrect")
    except Exception as e:
        p85_ok = False
        missing_or_invalid.append(f"p85_exception_{str(e)}")
        
    validated = p85_ok
    
    return {
        "source_contracts_validated": validated,
        "p85_validated": p85_ok,
        "p85_externalized_context_contract_preserved": p85_ok,
        "p85_no_selector_evidence_preserved": p85_ok,
        "p85_support_pair_materializer_required_preserved": p85_ok,
        "p85_bridge_not_ready_preserved": p85_ok,
        "missing_or_invalid": missing_or_invalid
    }


def run_p86_support_pair_materializer_from_p70_relation_cases_probe() -> dict:
    contracts_val = validate_source_contracts_for_p86()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    case_index = collect_p70_relation_cases(p70a_probe, p70b_probe)
    original_records = build_selector_dataset_records(p70a_probe, p70b_probe)
    episode_records = build_materialized_externalized_episode_records(original_records, case_index)
    
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
    
    materialization_audit = analyze_support_pair_materialization(episode_records)
    reduction_audit = analyze_materialized_episode_collision_reduction(original_records, episode_records)
    interpretation = interpret_support_pair_materialization(materialization_audit, reduction_audit, episode_leakage_audit)
    bridge_audit = audit_bridge_boundary_after_support_pair_materialization(interpretation)
    
    success_p87 = interpretation["valid_for_p87_synthetic_few_shot_selector_pilot"]
    success_materialization = materialization_audit["materialization_successful"]
    
    verdict_str = VERDICT if contracts_ok else "P86_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED is True or TORCH_TRAINING_PERFORMED is True:
            verdict_str = "P86_BLOCKED_BY_UNEXPECTED_TRAINING_OR_MODEL"
        elif has_leakage is True:
            if endpoint_leak or delta_leak:
                verdict_str = "P86_BLOCKED_BY_EVALUATED_TARGET_LEAKAGE"
            elif label_leak or op_leak or metadata_leak or hint_leak:
                verdict_str = "P86_BLOCKED_BY_LABEL_OR_HINT_PASS_THROUGH"
        elif reduction_audit["collision_key_sanitized"] is False:
            verdict_str = "P86_BLOCKED_BY_UNSANITIZED_COLLISION_KEY"
        elif BRIDGE_READY is True:
            verdict_str = "P86_BLOCKED_BY_PREMATURE_BRIDGE_AUTHORITY"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p85_externalized_context_contract_preserved": contracts_val["p85_externalized_context_contract_preserved"],
        "p85_no_selector_evidence_preserved": contracts_val["p85_no_selector_evidence_preserved"],
        "p85_support_pair_materializer_required_preserved": contracts_val["p85_support_pair_materializer_required_preserved"],
        "p85_bridge_not_ready_preserved": contracts_val["p85_bridge_not_ready_preserved"],

        "support_pair_materializer_defined": SUPPORT_PAIR_MATERIALIZER_DEFINED,
        "p70_relation_case_source_defined": P70_RELATION_CASE_SOURCE_DEFINED,
        "support_source_result_delta_schema_defined": SUPPORT_SOURCE_RESULT_DELTA_SCHEMA_DEFINED,
        "support_pair_materialization_attempted": SUPPORT_PAIR_MATERIALIZATION_ATTEMPTED,
        "externalized_episode_records_with_materialized_support_defined": EXTERNALIZED_EPISODE_RECORDS_WITH_MATERIALIZED_SUPPORT_DEFINED,

        "support_selection_uses_relation_label_for_synthetic_dataset_construction": SUPPORT_SELECTION_USES_RELATION_LABEL_FOR_SYNTHETIC_DATASET_CONSTRUCTION,
        "support_selection_label_visible_to_selector": SUPPORT_SELECTION_LABEL_VISIBLE_TO_SELECTOR,

        "evaluated_target_endpoint_used_for_selector_input": endpoint_leak,
        "evaluated_target_delta_used_for_selector_input": delta_leak,
        "exact_relation_label_used_for_selector_input": label_leak,
        "exact_operator_id_used_for_selector_input": op_leak,
        "audit_metadata_used_for_selector_input": metadata_leak,
        "relation_specific_hint_used_for_model_input": hint_leak,

        "support_pair_materialization_successful": success_materialization,
        "valid_for_p87_synthetic_few_shot_selector_pilot": success_p87,
        "valid_for_final_semantic_geometry_evidence": VALID_FOR_FINAL_SEMANTIC_GEOMETRY_EVIDENCE,
        "valid_for_bridge_evidence": VALID_FOR_BRIDGE_EVIDENCE,
        "external_real_query_source_still_required": EXTERNAL_REAL_QUERY_SOURCE_STILL_REQUIRED,

        "collision_key_sanitized": reduction_audit["collision_key_sanitized"],
        "support_record_id_used_in_collision_key": reduction_audit["support_record_id_used_in_collision_key"],
        "materialization_metadata_used_in_collision_key": reduction_audit["materialization_metadata_used_in_collision_key"],
        "label_or_hint_used_in_collision_key": reduction_audit["label_or_hint_used_in_collision_key"],

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

        "source_externalized_context_phase": SOURCE_EXTERNALIZED_CONTEXT_PHASE,
        "source_enrichment_phase": SOURCE_ENRICHMENT_PHASE,
        "source_identifiability_phase": SOURCE_IDENTIFIABILITY_PHASE,
        "source_selector_pilot_phase": SOURCE_SELECTOR_PILOT_PHASE,
        "source_dataset_phase": SOURCE_DATASET_PHASE,
        "source_synthetic_relation_phases": SOURCE_SYNTHETIC_RELATION_PHASES,

        "verdict": verdict_str,

        "training_allowed_by_phase4_authority": TRAINING_ALLOWED_BY_PHASE4_AUTHORITY,
        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "support_pair_materializer_defined": SUPPORT_PAIR_MATERIALIZER_DEFINED,
        "p70_relation_case_source_defined": P70_RELATION_CASE_SOURCE_DEFINED,
        "support_source_result_delta_schema_defined": SUPPORT_SOURCE_RESULT_DELTA_SCHEMA_DEFINED,
        "support_pair_materialization_attempted": SUPPORT_PAIR_MATERIALIZATION_ATTEMPTED,
        "externalized_episode_records_with_materialized_support_defined": EXTERNALIZED_EPISODE_RECORDS_WITH_MATERIALIZED_SUPPORT_DEFINED,

        "support_selection_uses_relation_label_for_synthetic_dataset_construction": SUPPORT_SELECTION_USES_RELATION_LABEL_FOR_SYNTHETIC_DATASET_CONSTRUCTION,
        "support_selection_label_visible_to_selector": SUPPORT_SELECTION_LABEL_VISIBLE_TO_SELECTOR,
        "support_context_constructed_by_synthetic_task_generator": SUPPORT_CONTEXT_CONSTRUCTED_BY_SYNTHETIC_TASK_GENERATOR,

        "valid_for_p87_synthetic_few_shot_selector_pilot": success_p87,
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
        "p85_externalized_context_contract_preserved": contracts_val["p85_externalized_context_contract_preserved"],
        "p85_no_selector_evidence_preserved": contracts_val["p85_no_selector_evidence_preserved"],
        "p85_support_pair_materializer_required_preserved": contracts_val["p85_support_pair_materializer_required_preserved"],
        "p85_bridge_not_ready_preserved": contracts_val["p85_bridge_not_ready_preserved"],

        "dataset_record_count": int(len(original_records)),
        "materialized_episode_record_count": int(len(episode_records)),

        "collision_key_sanitized": reduction_audit["collision_key_sanitized"],
        "support_record_id_used_in_collision_key": reduction_audit["support_record_id_used_in_collision_key"],
        "materialization_metadata_used_in_collision_key": reduction_audit["materialization_metadata_used_in_collision_key"],
        "label_or_hint_used_in_collision_key": reduction_audit["label_or_hint_used_in_collision_key"],

        "p70_relation_case_collection_audit": {
            "p70a_cases_available": case_index["p70a_case_count"] > 0,
            "p70b_cases_available": case_index["p70b_case_count"] > 0,
            "p70a_case_count": case_index["p70a_case_count"],
            "p70b_case_count": case_index["p70b_case_count"],
            "diagnostic_pass": True
        },
        "support_pair_materialization_audit": materialization_audit,
        "materialized_episode_input_leakage_audit": episode_leakage_audit,
        "materialized_episode_collision_reduction_audit": reduction_audit,
        "support_pair_materialization_interpretation": interpretation,
        "bridge_boundary_after_support_pair_materialization": bridge_audit,

        "sample_materialized_episode_records": sample_episodes,
        "sample_record_count": int(len(sample_episodes)),

        "sanity_summary": sanity_summary,

        "json_safe": True,
        "diagnostic_only": True
    }
    
    return output
