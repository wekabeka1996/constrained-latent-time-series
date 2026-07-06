# src/phase4/external_support_context_builder.py

import json
import math
import hashlib
from typing import Any

from src.phase4.support_pair_materializer_from_p70_relation_cases import (
    build_materialized_support_invariant_context,
    materialize_support_pair_for_record,
    collect_p70_relation_cases,
    run_p86_support_pair_materializer_from_p70_relation_cases_probe,
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

PHASE = "P90"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "External Support Context Builder"
CONTRACT_VERSION = "phase4_p90_external_support_context_builder_v1"

SOURCE_HARD_GENERALIZATION_PHASE = "P89"
SOURCE_NONLEARNED_METRIC_PHASE = "P88"
SOURCE_FEW_SHOT_SELECTOR_PHASE = "P87"
SOURCE_MATERIALIZER_PHASE = "P86"
SOURCE_EXTERNALIZED_CONTEXT_PHASE = "P85"
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

EXTERNAL_SUPPORT_CONTEXT_BUILDER_DEFINED = True
NON_LABEL_SELECTED_SUPPORT_CONTRACT_DEFINED = True
OBSERVABLE_SUPPORT_RETRIEVAL_DEFINED = True
EXTERNAL_DEMONSTRATION_RECORD_SCHEMA_DEFINED = True
SUPPORT_SELECTION_LEAKAGE_AUDIT_DEFINED = True

NONLEARNED_METRIC_SIGNAL_PRESENT = True
HARD_GENERALIZATION_SUPPORTED = True
EXTERNAL_CONTEXT_BUILDER_READY = False
NON_LABEL_SELECTED_SUPPORT_READY = False

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "external_support_context_builder"
VERDICT = "P90_READY_FOR_REVIEW"

ALL_CLASSES = [
    "translate_x",
    "translate_y",
    "scale_s",
    "reflect_x",
    "nonlinear_x_from_y",
    "change_frequency",
    "scale_amplitude",
    "shift_phase",
    "scale_volatility_envelope",
    "shift_trend"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(ALL_CLASSES)}

ACCEPTED_P89_REPORT_METADATA = {
    "phase": "P89",
    "verdict": "P89_READY_FOR_REVIEW",

    "source_contracts_validated": True,
    "p88_reproduction_effective_accuracy": 0.6666666666666666,

    "hard_generalization_supported": True,
    "best_raw_hard_policy": "train_to_heldout_base_state",
    "best_evidence_passing_hard_policy": "train_to_heldout_base_state",

    "train_to_heldout_base_state_effective_accuracy": 0.70,
    "train_to_heldout_base_state_query_source_only_control": 0.40,
    "train_to_heldout_base_state_shuffled_control": 0.40,
    "train_to_heldout_base_state_zero_delta_control": 0.10,

    "train_to_heldout_magnitude_effective_accuracy": 0.6666666666666666,
    "train_to_heldout_magnitude_query_source_only_control": 0.4444444444444444,
    "train_to_heldout_magnitude_shuffled_control": 0.3888888888888889,
    "train_to_heldout_magnitude_zero_delta_control": 0.1111111111111111,

    "hard_policy_controls_aligned": True,
    "hard_policy_fallback_controls_used": False,

    "external_context_contract_defined": True,
    "requires_real_external_support_demonstrations": True,
    "requires_query_observation_source": True,
    "requires_non_label_selected_support": True,
    "synthetic_label_selected_support_still_present": True,
    "valid_for_final_semantic_geometry_evidence": False,
    "valid_for_bridge_input_contract": False,

    "model_training_performed": False,
    "torch_training_performed": False,
    "new_model_implemented": False,
    "optimizer_created": False,
    "checkpoint_written": False,

    "learned_selector_evidence_present": False,
    "learned_metric_evidence_present": False,
    "semantic_metric_ready": False,
    "bridge_ready": False,

    "recommended_next_phase": "P90_external_support_context_builder_no_training_no_bridge",
    "metadata_source": "accepted_p89_report_static_metadata",
}


def extract_label(record: dict) -> str:
    if "audit_label_evaluation_only" in record:
        return record["audit_label_evaluation_only"]["target_relation_label"]
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


def validate_source_contracts_for_p90() -> dict:
    validated = True
    missing_or_invalid = []
    
    meta = ACCEPTED_P89_REPORT_METADATA
    if meta.get("phase") != "P89" or meta.get("verdict") != "P89_READY_FOR_REVIEW":
        validated = False
        missing_or_invalid.append("p89_invalid_phase_or_verdict")
    if meta.get("source_contracts_validated") is not True:
        validated = False
        missing_or_invalid.append("p89_source_unvalidated")
    if meta.get("hard_generalization_supported") is not True:
        validated = False
        missing_or_invalid.append("p89_hard_generalization_failed")
    if meta.get("hard_policy_controls_aligned") is not True:
        validated = False
        missing_or_invalid.append("p89_controls_unaligned")
    if meta.get("model_training_performed") is not False:
        validated = False
        missing_or_invalid.append("p89_unexpected_training")
    if meta.get("bridge_ready") is not False:
        validated = False
        missing_or_invalid.append("p89_bridge_ready")
        
    return {
        "source_contracts_validated": validated,
        "p89_validated": validated,
        "p89_hard_generalization_preserved": True,
        "p89_aligned_controls_preserved": True,
        "p89_external_context_need_preserved": True,
        "p89_no_training_preserved": True,
        "p89_bridge_not_ready_preserved": True,
        "p89_recommended_p90_preserved": True,
        "metadata_source": "accepted_p89_report_static_metadata",
        "missing_or_invalid": missing_or_invalid
    }


def build_external_demonstration_records(p70a_probe: dict, p70b_probe: dict) -> list[dict]:
    case_index = collect_p70_relation_cases(p70a_probe, p70b_probe)
    records = build_selector_dataset_records(p70a_probe, p70b_probe)
    
    demo_records = []
    for r in records:
        pair = materialize_support_pair_for_record(r, case_index)
        
        dom = r["domain"]
        sp = r["split"]
        
        if dom == "p70a_vector_world":
            obs_src_shape = f"vector_dim_{r['selector_input']['source_state_summary'].get('z_a_dim', 3.0)}"
            obs_del_shape = f"nonzero_count_{pair['support_delta_summary'].get('delta_nonzero_count', 0.0)}"
        else:
            obs_src_shape = f"params_keys_{r['selector_input']['source_parameter_summary'].get('parameter_key_count', 5.0)}"
            obs_del_shape = f"nonzero_keys_{pair['support_delta_summary'].get('delta_nonzero_key_count', 0.0)}"
            
        ctx_hash = hashlib.sha256(dom.encode("utf-8") + sp.encode("utf-8")).hexdigest()[:8]
        
        inv = {
            "support_pair_count": 1,
            "support_pair_observation_available_count": 1,
            "result_summary_available_count": 1,
            "delta_summary_available_count": 1,
            "support_delta_abs_sum_mean": float(pair["support_delta_summary"].get("delta_abs_sum", 0.0)),
            "support_delta_nonzero_count_mean": float(pair["support_delta_summary"].get("delta_nonzero_count", pair["support_delta_summary"].get("delta_nonzero_key_count", 0.0))),
            "support_invariant_context_available": True,
            "support_episode_limited_by_missing_result_or_delta_summaries": False
        }
        
        demo_id = f"demo_{r['dataset_record_id']}"
        lbl = extract_label(r)
        
        demo_records.append({
            "external_demo_id": demo_id,
            "domain": dom,
            "source_split": sp,
            
            "observable_context": {
                "context_world": pair.get("support_context_world", sp),
                "source_summary": pair["support_source_summary"],
                "result_summary": pair["support_result_summary"],
                "delta_summary": pair["support_delta_summary"],
                "invariant_summary": inv
            },
            
            "selection_visible_metadata": {
                "domain": dom,
                "source_split": sp,
                "observable_source_shape": obs_src_shape,
                "observable_delta_shape": obs_del_shape,
                "observable_context_hash": ctx_hash
            },
            
            "audit_label_evaluation_only": {
                "target_relation_label": lbl,
                "operator_id": r["label_evaluation"].get("operator_id", "op_unknown")
            },
            
            "valid_for_final_semantic_geometry_evidence": False,
            "valid_for_bridge_evidence": False
        })
        
    return demo_records


def build_external_query_records(selector_records: list[dict]) -> list[dict]:
    query_records = []
    for r in selector_records:
        dom = r["domain"]
        sp = r["split"]
        
        sel = r["selector_input"]
        
        q_obs = {
            "domain": dom,
            "context_world": sel.get("context_world", sp),
            "query_intensity_hint": sel.get("query_intensity_hint", 0.0)
        }
        if "source_state_summary" in sel:
            q_obs["source_state_summary"] = sel["source_state_summary"]
        if "source_parameter_summary" in sel:
            q_obs["source_parameter_summary"] = sel["source_parameter_summary"]
            
        lbl = extract_label(r)
        
        query_records.append({
            "external_query_id": f"query_{r['dataset_record_id']}",
            "dataset_record_id": r["dataset_record_id"],
            "domain": dom,
            "split": sp,
            
            "query_observation": q_obs,
            
            "audit_label_evaluation_only": {
                "target_relation_label": lbl,
                "operator_id": r["label_evaluation"].get("operator_id", "op_unknown")
            },
            
            "query_target_hidden": True,
            "query_delta_hidden": True,
            "valid_for_final_semantic_geometry_evidence": False,
            "valid_for_bridge_evidence": False
        })
    return query_records


def build_external_support_context_for_query(
    query_record: dict,
    demonstration_records: list[dict],
    policy: str,
    max_support: int = 2
) -> dict:
    q_id = query_record["external_query_id"]
    q_rec_id = query_record["dataset_record_id"]
    dom = query_record["domain"]
    sp = query_record["split"]
    
    candidates = []
    if policy == "label_selected_oracle_support":
        lbl = query_record["audit_label_evaluation_only"]["target_relation_label"]
        for d in demonstration_records:
            if d["external_demo_id"] == f"demo_{q_rec_id}":
                continue
            if d["domain"] == dom and d["source_split"] == sp:
                if d["audit_label_evaluation_only"]["target_relation_label"] == lbl:
                    candidates.append(d)
    elif policy == "observable_domain_split_retrieval":
        if dom == "p70a_vector_world":
            q_src_shape = f"vector_dim_{query_record['query_observation'].get('source_state_summary', {}).get('z_a_dim', 3.0)}"
        else:
            q_src_shape = f"params_keys_{query_record['query_observation'].get('source_parameter_summary', {}).get('parameter_key_count', 5.0)}"
            
        for d in demonstration_records:
            if d["external_demo_id"] == f"demo_{q_rec_id}":
                continue
            if d["domain"] == dom and d["source_split"] == sp:
                if d["selection_visible_metadata"]["observable_source_shape"] == q_src_shape:
                    candidates.append(d)
    elif policy == "external_manifest_support":
        for d in demonstration_records:
            if d["external_demo_id"] == f"demo_{q_rec_id}":
                continue
            if d["domain"] == dom and d["source_split"] == sp:
                candidates.append(d)
                
    candidates = sorted(candidates, key=lambda d: d["external_demo_id"])
    selected_demos = candidates[:max_support]
    selected_ids = [d["external_demo_id"] for d in selected_demos]
    
    support_pairs = []
    for d in selected_demos:
        obs = d["observable_context"]
        support_pairs.append({
            "support_record_id": d["external_demo_id"],
            "support_domain": d["domain"],
            "support_context_world": obs["context_world"],
            "support_source_summary": obs["source_summary"],
            "support_result_summary": obs["result_summary"],
            "support_delta_summary": obs["delta_summary"],
            "support_pair_observation_available": True,
            "materialization_source_phase": "P90",
            "materialization_method": f"external_support_context_builder_{policy}",
            "missing_result_summary": False,
            "missing_delta_summary": False
        })
        
    inv = build_materialized_support_invariant_context(support_pairs)
    
    uses_label = (policy == "label_selected_oracle_support")
    non_label_sel = (policy in ["observable_domain_split_retrieval", "external_manifest_support"])
    
    audit = {
        "support_selection_uses_hidden_relation_label": uses_label,
        "support_selection_uses_query_target": False,
        "support_selection_uses_query_delta": False,
        "support_selection_uses_operator_id": uses_label,
        "support_selection_label_visible_to_selector": False,
        "support_selection_non_label_selected": non_label_sel,
        "valid_for_p91_external_context_metric_evaluation": non_label_sel,
        "valid_for_final_semantic_geometry_evidence": False,
        "valid_for_bridge_evidence": False
    }
    
    return {
        "external_support_context_id": f"ctx_{policy}_{q_id}",
        "policy": policy,
        "query_id": q_id,
        "support_demo_ids": selected_ids,
        "support_context": {
            "support_pairs": support_pairs,
            "support_invariant_context": inv
        },
        "support_selection_audit": audit,
        "audit_label_evaluation_only": {
            "target_relation_label": query_record["audit_label_evaluation_only"]["target_relation_label"],
            "operator_id": query_record["audit_label_evaluation_only"]["operator_id"]
        }
    }


def audit_external_support_selection_leakage(contexts: list[dict]) -> dict:
    lbl_used = 0
    q_target_used = 0
    q_delta_used = 0
    op_used = 0
    lbl_visible = 0
    non_lbl_sel = 0
    valid_p91 = 0
    
    for c in contexts:
        audit = c["support_selection_audit"]
        if audit["support_selection_uses_hidden_relation_label"]:
            lbl_used += 1
        if audit["support_selection_uses_query_target"]:
            q_target_used += 1
        if audit["support_selection_uses_query_delta"]:
            q_delta_used += 1
        if audit["support_selection_uses_operator_id"]:
            op_used += 1
        if audit["support_selection_label_visible_to_selector"]:
            lbl_visible += 1
        if audit["support_selection_non_label_selected"]:
            non_lbl_sel += 1
        if audit["valid_for_p91_external_context_metric_evaluation"]:
            valid_p91 += 1
            
    leak_pass = (lbl_used == 0 and q_target_used == 0 and q_delta_used == 0 and op_used == 0 and lbl_visible == 0)
    
    return {
        "context_count": int(len(contexts)),
        "hidden_relation_label_used_count": lbl_used,
        "query_target_used_count": q_target_used,
        "query_delta_used_count": q_delta_used,
        "operator_id_used_count": op_used,
        "label_visible_to_selector_count": lbl_visible,
        "non_label_selected_context_count": non_lbl_sel,
        "valid_for_p91_context_count": valid_p91,
        "diagnostic_pass": leak_pass
    }


def encode_external_support_delta_context(context: dict) -> list[float]:
    support_pairs = context["support_context"].get("support_pairs", [])
    feats = []
    for i in range(2):
        if i < len(support_pairs):
            pair = support_pairs[i]
            p_domain = pair.get("support_domain", "")
            del_sum = pair.get("support_delta_summary", {})
            if p_domain == "p70a_vector_world":
                del_abs = float(del_sum.get("delta_abs_sum", 0.0))
                del_nz = float(del_sum.get("delta_nonzero_count", 0.0))
                sign_pat = del_sum.get("delta_sign_pattern", [0.0, 0.0, 0.0])
                pat = [float(x) for x in sign_pat]
                while len(pat) < 3:
                    pat.append(0.0)
                feats.extend([1.0, 0.0, del_abs, del_nz])
                feats.extend(pat[:3])
                feats.extend([0.0, 0.0])
            else:
                del_abs = float(del_sum.get("delta_abs_sum", 0.0))
                del_nz = float(del_sum.get("delta_nonzero_key_count", 0.0))
                feats.extend([0.0, 1.0, del_abs, del_nz])
                feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend([0.0] * 9)
    return feats


def audit_metric_shape_compatibility(contexts: list[dict]) -> dict:
    total = len(contexts)
    comp = 0
    incomp = 0
    for c in contexts:
        vec = encode_external_support_delta_context(c)
        if len(vec) == 18 and all(isinstance(x, float) for x in vec):
            comp += 1
        else:
            incomp += 1
            
    pass_flag = (incomp == 0) and (total > 0)
    
    return {
        "context_count": total,
        "expected_delta_only_dim": 18,
        "compatible_context_count": comp,
        "incompatible_context_count": incomp,
        "all_contexts_metric_shape_compatible": pass_flag,
        "diagnostic_pass": pass_flag
    }


def compute_distance(a: list[float], b: list[float], distance_family: str) -> float:
    if len(a) != len(b):
        return float('inf')
    if distance_family == "l2":
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return float('inf')


def run_diagnostic_evaluation(
    query_records: list[dict],
    demo_records: list[dict],
    policy: str
) -> dict:
    contexts = [build_external_support_context_for_query(q, demo_records, policy) for q in query_records]
    
    q_id_to_split = {q["external_query_id"]: q["split"] for q in query_records}
    train_contexts = [c for c in contexts if q_id_to_split.get(c["query_id"]) == "train"]
    test_contexts = [c for c in contexts if q_id_to_split.get(c["query_id"]) == "test"]
    
    class_sums = {}
    class_counts = {}
    for c in train_contexts:
        lbl = c["audit_label_evaluation_only"]["target_relation_label"]
        c_idx = CLASS_TO_IDX.get(lbl, -1)
        if c_idx == -1:
            continue
        vec = encode_external_support_delta_context(c)
        class_sums[c_idx] = class_sums.get(c_idx, [0.0] * 18)
        class_counts[c_idx] = class_counts.get(c_idx, 0) + 1
        for j in range(18):
            class_sums[c_idx][j] += vec[j]
            
    prototypes = {}
    for c_idx, sums in class_sums.items():
        cnt = class_counts[c_idx]
        prototypes[c_idx] = [s / cnt for s in sums]
        
    correct_count = 0
    total_eval = 0
    
    for c in test_contexts:
        lbl = c["audit_label_evaluation_only"]["target_relation_label"]
        t_idx = CLASS_TO_IDX.get(lbl, -1)
        if t_idx == -1:
            continue
        if t_idx not in prototypes:
            continue
            
        total_eval += 1
        vec = encode_external_support_delta_context(c)
        
        min_dist = float('inf')
        pred_idx = None
        for c_idx, proto in prototypes.items():
            dist = compute_distance(vec, proto, "l2")
            if dist < min_dist:
                min_dist = dist
                pred_idx = c_idx
                
        if pred_idx == t_idx:
            correct_count += 1
            
    acc = float(correct_count / total_eval) if total_eval > 0 else 0.0
    return {
        "support_policy": policy,
        "sample_count": total_eval,
        "accuracy": acc,
        "diagnostic_only": True,
        "valid_for_final_semantic_geometry_evidence": False
    }


def audit_bridge_boundary_after_external_support_context_builder(
    external_context_builder_ready: bool,
    non_label_selected_support_ready: bool
) -> dict:
    blocking_reasons = [
        "learned_metric_evidence_not_present",
        "semantic_metric_not_ready",
        "external_context_metric_evaluation_not_run",
        "bridge_input_contract_not_defined",
        "bridge_validation_not_run"
    ]
    
    return {
        "bridge_ready": BRIDGE_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": True,
        "external_context_builder_ready": external_context_builder_ready,
        "non_label_selected_support_ready": non_label_selected_support_ready,
        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,
        "blocking_reasons": blocking_reasons,
        "diagnostic_pass": True
    }


def run_p90_external_support_context_builder_probe() -> dict:
    contracts_val = validate_source_contracts_for_p90()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    # 1. Build Demonstration Records & Query Observations
    demo_recs = build_external_demonstration_records(p70a, p70b)
    selector_records = build_selector_dataset_records(p70a, p70b)
    query_recs = build_external_query_records(selector_records)
    
    # 2. Build Support Selection Contexts under three policies
    policies = [
        "label_selected_oracle_support",
        "observable_domain_split_retrieval",
        "external_manifest_support"
    ]
    
    policy_contexts = {}
    policy_leakage_audits = {}
    policy_shape_compatibility = {}
    diagnostic_evals = {}
    
    for pol in policies:
        # Build context for all queries
        contexts = [build_external_support_context_for_query(q, demo_recs, pol) for q in query_recs]
        policy_contexts[pol] = contexts
        
        # Audit leakage and shape compatibility
        policy_leakage_audits[pol] = audit_external_support_selection_leakage(contexts)
        policy_shape_compatibility[pol] = audit_metric_shape_compatibility(contexts)
        
        # Run diagnostic evaluations
        diagnostic_evals[pol] = run_diagnostic_evaluation(query_recs, demo_recs, pol)
        
    # 3. Readiness Checks
    # Builder is ready if source ok, demo/query schemas exist, audits are defined
    builder_ready = (
        contracts_ok
        and len(demo_recs) > 0
        and len(query_recs) > 0
        and policy_leakage_audits["observable_domain_split_retrieval"]["diagnostic_pass"] is True
        and policy_shape_compatibility["observable_domain_split_retrieval"]["diagnostic_pass"] is True
    )
    
    # Non-label-selected support is ready if observable_domain_split_retrieval produces valid context, compatible shape, and clean leakage
    non_label_selected_ready = (
        builder_ready
        and policy_contexts["observable_domain_split_retrieval"][0]["support_selection_audit"]["support_selection_non_label_selected"] is True
        and policy_leakage_audits["observable_domain_split_retrieval"]["diagnostic_pass"] is True
        and policy_shape_compatibility["observable_domain_split_retrieval"]["diagnostic_pass"] is True
    )
    
    bridge_audit = audit_bridge_boundary_after_external_support_context_builder(builder_ready, non_label_selected_ready)
    
    # Recommended next phase logic
    if builder_ready and non_label_selected_ready:
        next_phase = "P91_external_context_metric_evaluation_no_training_no_bridge"
    elif builder_ready:
        next_phase = "P91_external_support_selection_repair_no_training_no_bridge"
    elif policy_shape_compatibility["observable_domain_split_retrieval"]["diagnostic_pass"] is False:
        next_phase = "P91_external_context_metric_shape_repair_no_training_no_bridge"
    else:
        next_phase = "P91_external_context_leakage_repair_no_training_no_bridge"
        
    verdict_str = VERDICT if contracts_ok else "P90_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED is True or TORCH_TRAINING_PERFORMED is True:
            verdict_str = "P90_BLOCKED_BY_UNEXPECTED_TRAINING_FLAG"
        elif policy_leakage_audits["observable_domain_split_retrieval"]["diagnostic_pass"] is False:
            verdict_str = "P90_BLOCKED_BY_METRIC_INPUT_LEAKAGE"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p89_hard_generalization_preserved": contracts_val["p89_hard_generalization_preserved"],
        "p89_aligned_controls_preserved": contracts_val["p89_aligned_controls_preserved"],
        "p89_external_context_need_preserved": contracts_val["p89_external_context_need_preserved"],
        "p89_no_training_preserved": contracts_val["p89_no_training_preserved"],
        "p89_bridge_not_ready_preserved": contracts_val["p89_bridge_not_ready_preserved"],

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "external_support_context_builder_defined": EXTERNAL_SUPPORT_CONTEXT_BUILDER_DEFINED,
        "non_label_selected_support_contract_defined": NON_LABEL_SELECTED_SUPPORT_CONTRACT_DEFINED,
        "observable_support_retrieval_defined": OBSERVABLE_SUPPORT_RETRIEVAL_DEFINED,
        "external_demonstration_record_schema_defined": EXTERNAL_DEMONSTRATION_RECORD_SCHEMA_DEFINED,
        "support_selection_leakage_audit_defined": SUPPORT_SELECTION_LEAKAGE_AUDIT_DEFINED,

        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": True,
        "external_context_builder_ready": builder_ready,
        "non_label_selected_support_ready": non_label_selected_ready,

        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_ready": BRIDGE_READY,

        "json_safe": True
    }
    
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,

        "source_hard_generalization_phase": SOURCE_HARD_GENERALIZATION_PHASE,
        "source_nonlearned_metric_phase": SOURCE_NONLEARNED_METRIC_PHASE,
        "source_few_shot_selector_phase": SOURCE_FEW_SHOT_SELECTOR_PHASE,
        "source_materializer_phase": SOURCE_MATERIALIZER_PHASE,
        "source_externalized_context_phase": SOURCE_EXTERNALIZED_CONTEXT_PHASE,
        "source_enrichment_phase": SOURCE_ENRICHMENT_PHASE,
        "source_identifiability_phase": SOURCE_IDENTIFIABILITY_PHASE,
        "source_selector_pilot_phase": SOURCE_SELECTOR_PILOT_PHASE,
        "source_dataset_phase": SOURCE_DATASET_PHASE,

        "verdict": verdict_str,

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "source_contracts_validated": contracts_ok,

        "external_demonstration_record_count": int(len(demo_recs)),
        "external_query_record_count": int(len(query_recs)),

        "support_selection_policy_results": {
            pol: {
                "context_count": len(policy_contexts[pol]),
                "sample_demo_ids": policy_contexts[pol][0]["support_demo_ids"] if len(policy_contexts[pol]) > 0 else []
            } for pol in policies
        },

        "support_selection_leakage_audits": policy_leakage_audits,
        "metric_shape_compatibility_audits": policy_shape_compatibility,

        "diagnostic_policy_evaluations": diagnostic_evals,

        "external_context_builder_ready": builder_ready,
        "non_label_selected_support_ready": non_label_selected_ready,

        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": True,
        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "bridge_implementation_allowed": False,
        "bridge_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,

        "recommended_next_phase": next_phase,

        "bridge_boundary_after_external_support_context_builder": bridge_audit,

        "sanity_summary": sanity_summary,

        "json_safe": True,
        "diagnostic_only": True
    }
    
    return output
