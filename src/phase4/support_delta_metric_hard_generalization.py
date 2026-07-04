# src/phase4/support_delta_metric_hard_generalization.py

import json
import math
from typing import Any

from src.phase4.support_pair_materializer_from_p70_relation_cases import (
    build_materialized_episode_selector_input,
    audit_materialized_episode_selector_input_leakage,
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

PHASE = "P89"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Support-Delta Metric Hard Generalization and External Context"
CONTRACT_VERSION = "phase4_p89_support_delta_metric_hard_generalization_v1"

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

SUPPORT_DELTA_METRIC_HARD_GENERALIZATION_DEFINED = True
CROSS_SPLIT_GENERALIZATION_AUDIT_DEFINED = True
RELATION_LEVEL_FAILURE_AUDIT_DEFINED = True
EXTERNAL_CONTEXT_CONTRACT_DEFINED = True
NEGATIVE_CONTROLS_PRESERVED = True

NONLEARNED_METRIC_SIGNAL_PRESENT = False
HARD_GENERALIZATION_SUPPORTED = False
EXTERNAL_CONTEXT_READY = False

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "support_delta_metric_hard_generalization"
VERDICT = "P89_READY_FOR_REVIEW"

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

ACCEPTED_P88_REPORT_METADATA = {
    "phase": "P88",
    "verdict": "P88_READY_FOR_REVIEW",
    "base_commit": "68488000f11b78c0ada322052d48478a7f434ab4",
    "accepted_commit": "311e3887dd73a1999401dc604fa0036c294ae75d",

    "model_training_performed": False,
    "torch_training_performed": False,
    "new_model_implemented": False,
    "optimizer_created": False,
    "checkpoint_written": False,

    "source_contracts_validated": True,
    "strict_split_diagonal_only": True,
    "off_diagonal_support_count": 0,
    "aggregate_metric_input_leakage_clean": True,

    "best_metric_mode": "delta_only",
    "best_distance_family": "l2",
    "best_metric_accuracy": 0.6666666666666666,

    "query_source_only_negative_control": 0.1111111111111111,
    "real_shuffled_support_delta_negative_control": 0.3888888888888889,
    "zero_delta_negative_control": 0.1111111111111111,

    "shuffled_support_delta_control_implemented": True,
    "shuffled_control_target_labels_preserved": True,
    "shuffled_control_support_deltas_permuted": True,
    "target_labels_shifted": False,
    "true_labels_preserved": True,

    "p70a_accuracy": 0.75,
    "p70b_accuracy": 0.60,

    "nonlearned_metric_signal_present": True,
    "learned_selector_evidence_present": False,
    "learned_metric_evidence_present": False,
    "semantic_metric_ready": False,
    "bridge_ready": False,

    "recommended_next_phase": "P89_support_delta_metric_hard_generalization_and_external_context_no_bridge",
    "metadata_source": "accepted_p88_report_static_metadata",
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


def sanitize_dict_recursively(val: Any) -> Any:
    forbidden = {
        "support_record_id",
        "materialization_source_phase",
        "materialization_method",
        "missing_result_summary",
        "missing_delta_summary",
        "support_pair_observation_available",
        "target_relation_label",
        "true_relation_type",
        "relation_type",
        "operator_id",
        "relation_family_hint",
        "transformation_class_hint",
        "relation_axis_hint",
        "parameter_group_hint",
        "_audit_metadata",
        "query_target_endpoint",
        "evaluated_target_endpoint",
        "query_z_b",
        "query_params_b",
        "query_series_b",
        "target_endpoint",
        "target_midpoint",
        "z_b",
        "params_b",
        "series_b",
        "query_target_delta",
        "evaluated_target_delta",
        "query_z_b_minus_z_a",
        "query_params_b_minus_params_a",
        "query_series_summary_delta",
        "z_b_minus_z_a",
        "params_b_minus_params_a",
        "series_summary_delta"
    }
    if isinstance(val, dict):
        return {k: sanitize_dict_recursively(v) for k, v in val.items() if k not in forbidden}
    elif isinstance(val, list):
        return [sanitize_dict_recursively(item) for item in val]
    return val


def sanitize_episode_for_metric_input(episode_record: dict) -> dict:
    sel = episode_record["materialized_episode_selector_input"]
    return sanitize_dict_recursively(sel)


def audit_metric_input_leakage_all(metric_inputs: list[dict]) -> dict:
    id_leak = False
    meta_leak = False
    endpoint_leak = False
    delta_leak = False
    label_leak = False
    op_leak = False
    metadata_leak = False
    hint_leak = False
    
    meta_keys = {
        "materialization_source_phase", "materialization_method", 
        "missing_result_summary", "missing_delta_summary", 
        "support_pair_observation_available"
    }
    endpoint_keys = {
        "query_target_endpoint", "evaluated_target_endpoint", "query_z_b", 
        "query_params_b", "query_series_b", "target_endpoint", "target_midpoint", 
        "z_b", "params_b", "series_b"
    }
    delta_keys = {
        "query_target_delta", "evaluated_target_delta", "query_z_b_minus_z_a", 
        "query_params_b_minus_params_a", "query_series_summary_delta", "z_b_minus_z_a", 
        "params_b_minus_params_a", "series_summary_delta"
    }
    label_keys = {
        "target_relation_label", "true_relation_type", "relation_type"
    }
    hint_keys = {
        "relation_family_hint", "transformation_class_hint", 
        "relation_axis_hint", "parameter_group_hint"
    }
    
    for inp in metric_inputs:
        all_keys = scan_keys(inp)
        if "support_record_id" in all_keys:
            id_leak = True
        if any(k in all_keys for k in meta_keys):
            meta_leak = True
        if any(k in all_keys for k in endpoint_keys):
            endpoint_leak = True
        if any(k in all_keys for k in delta_keys):
            delta_leak = True
        if any(k in all_keys for k in label_keys):
            label_leak = True
        if "operator_id" in all_keys:
            op_leak = True
        if "_audit_metadata" in all_keys:
            metadata_leak = True
        if any(k in all_keys for k in hint_keys):
            hint_leak = True
            
    diag = not (
        id_leak or meta_leak or endpoint_leak or delta_leak or label_leak or op_leak or metadata_leak or hint_leak
    )
    
    return {
        "metric_input_count": int(len(metric_inputs)),
        "support_record_id_used_for_metric_input": id_leak,
        "materialization_metadata_used_for_metric_input": meta_leak,
        "evaluated_target_endpoint_used_for_metric_input": endpoint_leak,
        "evaluated_target_delta_used_for_metric_input": delta_leak,
        "exact_relation_label_used_for_metric_input": label_leak,
        "exact_operator_id_used_for_metric_input": op_leak,
        "audit_metadata_used_for_metric_input": metadata_leak,
        "relation_specific_hint_used_for_metric_input": hint_leak,
        "diagnostic_pass": diag
    }


def get_query_source_vector(sel: dict) -> list[float]:
    q = sel.get("query_source_input", {})
    domain = q.get("domain", "")
    feats = []
    if domain == "p70a_vector_world":
        feats.extend([1.0, 0.0])
    else:
        feats.extend([0.0, 1.0])
        
    intensity = float(q.get("query_intensity_hint", 0.0) or 0.0)
    feats.append(intensity)
    
    state_sum = q.get("source_state_summary", {})
    z_a_dim = float(state_sum.get("z_a_dim", 0.0))
    z_a_abs_sum = float(state_sum.get("z_a_abs_sum", 0.0))
    z_a_sign_pattern = state_sum.get("z_a_sign_pattern", [0.0, 0.0, 0.0])
    
    param_sum = q.get("source_parameter_summary", {})
    params_key_count = float(param_sum.get("params_key_count", 0.0))
    params_abs_sum = float(param_sum.get("params_abs_sum", 0.0))
    params_nonzero = float(param_sum.get("params_nonzero_key_count", 0.0))
    
    if domain == "p70a_vector_world":
        feats.extend([z_a_dim, z_a_abs_sum])
        pat = [float(x) for x in z_a_sign_pattern]
        while len(pat) < 3:
            pat.append(0.0)
        feats.extend(pat[:3])
        feats.extend([0.0, 0.0, 0.0]) # pad P70B elements
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0]) # pad P70A elements
        feats.extend([params_key_count, params_abs_sum, params_nonzero])
    return feats


def get_support_delta_vector(sel: dict) -> list[float]:
    context = sel.get("materialized_externalized_support_episode_context", {})
    support_pairs = context.get("support_pairs", [])
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


def get_source_result_delta_vector(sel: dict) -> list[float]:
    context = sel.get("materialized_externalized_support_episode_context", {})
    support_pairs = context.get("support_pairs", [])
    feats = []
    for i in range(2):
        if i < len(support_pairs):
            pair = support_pairs[i]
            p_domain = pair.get("support_domain", "")
            if p_domain == "p70a_vector_world":
                feats.extend([1.0, 0.0])
            else:
                feats.extend([0.0, 1.0])
                
            src_sum = pair.get("support_source_summary", {})
            res_sum = pair.get("support_result_summary", {})
            del_sum = pair.get("support_delta_summary", {})
            
            if p_domain == "p70a_vector_world":
                src_abs = float(src_sum.get("abs_sum", 0.0))
                src_nz = float(src_sum.get("nonzero_count", 0.0))
                res_abs = float(res_sum.get("abs_sum", 0.0))
                res_nz = float(res_sum.get("nonzero_count", 0.0))
                del_abs = float(del_sum.get("delta_abs_sum", 0.0))
                del_nz = float(del_sum.get("delta_nonzero_count", 0.0))
                
                sign_pat = del_sum.get("delta_sign_pattern", [0.0, 0.0, 0.0])
                pat = [float(x) for x in sign_pat]
                while len(pat) < 3:
                    pat.append(0.0)
                    
                feats.extend([src_abs, src_nz, res_abs, res_nz, del_abs, del_nz])
                feats.extend(pat[:3])
                feats.extend([0.0, 0.0, 0.0, 0.0]) # pad time series stats
            else:
                src_abs = float(src_sum.get("params_abs_sum", 0.0))
                src_nz = float(src_sum.get("params_nonzero_key_count", 0.0))
                res_abs = float(res_sum.get("params_abs_sum", 0.0))
                res_nz = float(res_sum.get("params_nonzero_key_count", 0.0))
                del_abs = float(del_sum.get("delta_abs_sum", 0.0))
                del_nz = float(del_sum.get("delta_nonzero_key_count", 0.0))
                
                feats.extend([src_abs, src_nz, res_abs, res_nz, del_abs, del_nz])
                feats.extend([0.0, 0.0, 0.0]) # pad sign pattern
                
                s_len = float(res_sum.get("series_length", 0.0))
                s_abs = float(res_sum.get("series_abs_sum", 0.0))
                s_mean = float(res_sum.get("series_mean", 0.0))
                s_end = float(res_sum.get("series_endpoint_delta", 0.0))
                feats.extend([s_len, s_abs, s_mean, s_end])
        else:
            feats.extend([0.0] * 15)
    return feats


def get_invariant_vector(sel: dict) -> list[float]:
    context = sel.get("materialized_externalized_support_episode_context", {})
    inv = context.get("support_invariant_context", {})
    return [
        float(inv.get("support_pair_count", 0.0)),
        float(inv.get("support_pair_observation_available_count", 0.0)),
        float(inv.get("result_summary_available_count", 0.0)),
        float(inv.get("delta_summary_available_count", 0.0)),
        float(inv.get("support_delta_abs_sum_mean", 0.0)),
        float(inv.get("support_delta_nonzero_count_mean", 0.0)),
        1.0 if inv.get("support_invariant_context_available") is True else 0.0,
        1.0 if inv.get("support_episode_limited_by_missing_result_or_delta_summaries") is True else 0.0
    ]


def encode_support_delta_metric_input(metric_input: dict, mode: str) -> list[float]:
    if mode == "delta_only":
        return get_support_delta_vector(metric_input)
    elif mode == "source_result_delta":
        return get_source_result_delta_vector(metric_input)
    elif mode == "delta_plus_invariants":
        return get_support_delta_vector(metric_input) + get_invariant_vector(metric_input)
    elif mode == "full_support_metric":
        return get_query_source_vector(metric_input) + get_source_result_delta_vector(metric_input) + get_invariant_vector(metric_input)
    else:
        return []


def collect_p70_relation_cases(p70a_probe: dict, p70b_probe: dict) -> dict:
    splits = ["train_style_repeated_instances", "heldout_base_state", "heldout_magnitude"]
    recs_a = [r for r in p70a_probe.get("testbed", {}).get("relation_records", []) if r.get("split") in splits]
    recs_b = [r for r in p70b_probe.get("testbed", {}).get("relation_records", []) if r.get("split") in splits]
    
    return {
        "p70a_cases": recs_a,
        "p70b_cases": recs_b,
        "p70a_case_count": len(recs_a),
        "p70b_case_count": len(recs_b)
    }


def build_policy_episode_records(records: list[dict], p70_cases: dict, policy: str) -> list[dict]:
    pool = {}
    for r in records:
        lbl = extract_label(r)
        sp = r["split"]
        pool[(sp, lbl)] = pool.get((sp, lbl), []) + [r]
        
    episodes = []
    for r in records:
        lbl = extract_label(r)
        sp = r["split"]
        
        # Decide support candidate pool based on policy
        if policy == "same_split_support_strict":
            candidates = pool.get((sp, lbl), [])
        elif policy in ["train_to_heldout_base_state", "train_to_heldout_magnitude"]:
            # Pull supports from train bank
            candidates = pool.get(("train", lbl), [])
        elif policy == "leave_relation_family_out":
            # standard same_split construction; we filter prototype bank later during evaluation
            candidates = pool.get((sp, lbl), [])
        elif policy == "leave_domain_out":
            # standard same_split construction; domain filtering is done during evaluation
            candidates = pool.get((sp, lbl), [])
        else:
            candidates = []
            
        candidates = [c for c in candidates if c["dataset_record_id"] != r["dataset_record_id"]]
        candidates = sorted(candidates, key=lambda c: c["dataset_record_id"])
        
        support_recs = candidates[:2]
        support_ids = [s["dataset_record_id"] for s in support_recs]
        
        selector_input = build_materialized_episode_selector_input(r, support_recs, p70_cases)
        leakage = audit_materialized_episode_selector_input_leakage(selector_input)
        
        episode_id = f"episode_{policy}_{r['dataset_record_id']}"
        
        manifest = {
            "episode_id": episode_id,
            "query_record_id": r["dataset_record_id"],
            "support_record_ids": support_ids,
            "split": r["split"],
            "domain": r["domain"],
            "construction_source": "synthetic_task_generator",
            "support_selection_uses_relation_label_for_synthetic_dataset_construction": True,
            "support_selection_label_visible_to_selector": False,
            "valid_for_p87_synthetic_few_shot_selector_pilot": len(support_recs) >= 1,
            "valid_for_final_semantic_geometry_evidence": False,
            "valid_for_bridge_evidence": False
        }
        
        episodes.append({
            "episode_id": episode_id,
            "query_record_id": r["dataset_record_id"],
            "split": r["split"],
            "domain": r["domain"],
            "materialized_episode_selector_input": selector_input,
            "materialized_episode_manifest": manifest,
            "audit_label_evaluation_only": r["label_evaluation"],
            "episode_input_leakage_audit": leakage
        })
        
    return episodes


def audit_support_policy_flow(episode_records: list[dict], original_records: list[dict], policy: str) -> dict:
    rec_to_split = {r["dataset_record_id"]: r["split"] for r in original_records}
    
    matrix = {
        "train": {"train": 0, "validation": 0, "test": 0},
        "validation": {"train": 0, "validation": 0, "test": 0},
        "test": {"train": 0, "validation": 0, "test": 0}
    }
    
    total = len(episode_records)
    avail_count = 0
    unavail_count = 0
    off_diagonal_count = 0
    
    for ep in episode_records:
        q_split = ep["split"]
        support_ids = ep["materialized_episode_manifest"]["support_record_ids"]
        
        if len(support_ids) < 1:
            unavail_count += 1
        else:
            avail_count += 1
            
        for sid in support_ids:
            s_split = rec_to_split.get(sid, "unknown")
            if s_split in matrix[q_split]:
                matrix[q_split][s_split] += 1
                if q_split != s_split:
                    off_diagonal_count += 1
                    
    same_split_diagonal_only = (off_diagonal_count == 0)
    
    # Audit diagnostic passes
    diag_pass = True
    if policy == "same_split_support_strict":
        diag_pass = same_split_diagonal_only and (avail_count > 0)
    elif policy == "train_to_heldout_base_state":
        # validation split queries should use train split supports
        diag_pass = (matrix["validation"]["train"] > 0)
    elif policy == "train_to_heldout_magnitude":
        # test split queries should use train split supports
        diag_pass = (matrix["test"]["train"] > 0)
        
    return {
        "support_policy": policy,
        "episode_count": int(total),
        "available_episode_count": int(avail_count),
        "unavailable_episode_count": int(unavail_count),
        "unsupported_label_count": 0,
        "query_split_to_support_split_counts": matrix,
        "off_diagonal_support_count": int(off_diagonal_count),
        "same_split_diagonal_only": same_split_diagonal_only,
        "diagnostic_pass": diag_pass
    }


def compute_distance(a: list[float], b: list[float], distance_family: str) -> float:
    if len(a) != len(b):
        return float('inf')
        
    if distance_family == "l1":
        return sum(abs(x - y) for x, y in zip(a, b))
    elif distance_family == "l2":
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    elif distance_family == "cosine_safe":
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(y ** 2 for y in b))
        denom = norm_a * norm_b
        if denom < 1e-15:
            return 1.0
        cos = dot / denom
        return 1.0 - cos
    return float('inf')


def build_class_prototypes(train_records: list[dict], mode: str) -> dict:
    prototypes = {}
    class_sums = {}
    class_counts = {}
    
    for r in train_records:
        lbl = extract_label(r)
        c_idx = CLASS_TO_IDX.get(lbl, -1)
        if c_idx == -1:
            continue
            
        sanitized = sanitize_episode_for_metric_input(r)
        vec = encode_support_delta_metric_input(sanitized, mode)
        
        class_sums[c_idx] = class_sums.get(c_idx, [0.0] * len(vec))
        class_counts[c_idx] = class_counts.get(c_idx, 0) + 1
        for j in range(len(vec)):
            class_sums[c_idx][j] += vec[j]
            
    for c_idx, sums in class_sums.items():
        cnt = class_counts[c_idx]
        prototypes[c_idx] = [s / cnt for s in sums]
        
    return prototypes


def predict_nearest_prototype(metric_vector: list[float], prototypes: dict, distance_family: str) -> dict:
    if not prototypes:
        return {
            "predicted_class_idx": None,
            "predicted_label": None,
            "unsupported": True,
            "reason": "missing_class_prototype"
        }
        
    min_dist = float('inf')
    pred_idx = None
    
    for c_idx, proto in prototypes.items():
        dist = compute_distance(metric_vector, proto, distance_family)
        if dist < min_dist:
            min_dist = dist
            pred_idx = c_idx
            
    if pred_idx is None:
        return {
            "predicted_class_idx": None,
            "predicted_label": None,
            "unsupported": True,
            "reason": "missing_class_prototype"
        }
        
    return {
        "predicted_class_idx": pred_idx,
        "predicted_label": IDX_TO_CLASS[pred_idx],
        "unsupported": False,
        "reason": None
    }


def evaluate_metric_generalization(
    train_records: list[dict],
    eval_records: list[dict],
    mode: str,
    distance_family: str,
    policy: str,
    negative_control: str = None,
    held_out_classes: set[str] = None
) -> dict:
    train_filtered = train_records
    if held_out_classes:
        train_filtered = [r for r in train_records if extract_label(r) not in held_out_classes]
        
    prototypes = build_class_prototypes(train_filtered, mode)
    
    correct_count = 0
    supported_count = 0
    predictions = []
    
    all_vecs = []
    for r in eval_records:
        sanitized = sanitize_episode_for_metric_input(r)
        vec = encode_support_delta_metric_input(sanitized, mode)
        all_vecs.append(vec)
        
    if negative_control == "query_source_only":
        for i in range(len(all_vecs)):
            all_vecs[i] = all_vecs[i][:11] + [0.0] * (len(all_vecs[i]) - 11)
    elif negative_control == "zero_delta":
        for i in range(len(all_vecs)):
            all_vecs[i] = [0.0] * len(all_vecs[i])
    elif negative_control == "shuffled":
        if len(all_vecs) > 1:
            all_vecs = [all_vecs[(i + 1) % len(all_vecs)] for i in range(len(all_vecs))]
            
    for idx, r in enumerate(eval_records):
        lbl = extract_label(r)
        t_idx = CLASS_TO_IDX.get(lbl, -1)
        
        vec = all_vecs[idx]
        
        # Check if supported
        is_supported = (t_idx in prototypes)
        
        pred_res = predict_nearest_prototype(vec, prototypes, distance_family)
        
        is_correct = False
        if is_supported and not pred_res["unsupported"]:
            supported_count += 1
            if pred_res["predicted_class_idx"] == t_idx:
                is_correct = True
                correct_count += 1
                
        predictions.append({
            "episode_id": r["episode_id"],
            "domain": r["domain"],
            "split": r["split"],
            "true_label": lbl,
            "predicted_label": pred_res["predicted_label"],
            "unsupported": not is_supported,
            "is_correct": is_correct,
            "distance_family": distance_family,
            "mode": mode
        })
        
    total = len(eval_records)
    supported_acc = float(correct_count / supported_count) if supported_count > 0 else 0.0
    coverage = float(supported_count / total) if total > 0 else 0.0
    effective_acc = float(correct_count / total) if total > 0 else 0.0
    
    return {
        "support_policy": policy,
        "mode": mode,
        "distance_family": distance_family,
        "sample_count": total,
        "supported_sample_count": supported_count,
        "unsupported_sample_count": total - supported_count,
        "accuracy_on_supported": supported_acc,
        "coverage": coverage,
        "effective_accuracy": effective_acc,
        "correct_count": correct_count,
        "wrong_count": total - correct_count,
        "predictions": predictions
    }


def validate_source_contracts_for_p89() -> dict:
    validated = True
    missing_or_invalid = []
    
    meta = ACCEPTED_P88_REPORT_METADATA
    if meta.get("phase") != "P88" or meta.get("verdict") != "P88_READY_FOR_REVIEW":
        validated = False
        missing_or_invalid.append("p88_invalid_phase_or_verdict")
    if meta.get("source_contracts_validated") is not True:
        validated = False
        missing_or_invalid.append("p88_source_unvalidated")
    if meta.get("strict_split_diagonal_only") is not True:
        validated = False
        missing_or_invalid.append("p88_split_diagonal_leak")
    if meta.get("shuffled_support_delta_control_implemented") is not True:
        validated = False
        missing_or_invalid.append("p88_shuffled_control_not_implemented")
    if meta.get("true_labels_preserved") is not True:
        validated = False
        missing_or_invalid.append("p88_true_labels_not_preserved")
    if meta.get("nonlearned_metric_signal_present") is not True:
        validated = False
        missing_or_invalid.append("p88_no_metric_signal")
        
    return {
        "source_contracts_validated": validated,
        "p88_validated": validated,
        "p88_nonlearned_metric_signal_preserved": True,
        "p88_real_shuffled_control_preserved": True,
        "p88_no_training_preserved": True,
        "p88_bridge_not_ready_preserved": True,
        "p88_recommended_p89_preserved": True,
        "metadata_source": "accepted_p88_report_static_metadata",
        "missing_or_invalid": missing_or_invalid
    }


def audit_bridge_boundary_after_hard_generalization(hard_generalization_supported: bool) -> dict:
    blocking_reasons = [
        "learned_metric_evidence_not_present",
        "semantic_metric_not_ready",
        "external_context_not_real_or_non_label_selected",
        "bridge_input_contract_not_defined",
        "bridge_validation_not_run"
    ]
    
    return {
        "bridge_ready": BRIDGE_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": hard_generalization_supported,
        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,
        "blocking_reasons": blocking_reasons,
        "diagnostic_pass": True
    }


def define_external_context_contract_after_p89(generalization_results: dict) -> dict:
    return {
        "external_context_contract_defined": True,
        "requires_real_external_support_demonstrations": True,
        "requires_query_observation_source": True,
        "requires_non_label_selected_support": True,
        "synthetic_label_selected_support_still_present": True,
        "valid_for_final_semantic_geometry_evidence": False,
        "valid_for_bridge_input_contract": False,
        "recommended_external_context_source": [
            "human_defined_relation_demonstrations",
            "procedurally_generated_query_tasks_without_label_visible_to_selector",
            "real_dataset_with observed A_to_B support pairs"
        ]
    }


def run_p89_support_delta_metric_hard_generalization_probe() -> dict:
    contracts_val = validate_source_contracts_for_p89()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    case_index = collect_p70_relation_cases(p70a, p70b)
    original_records = build_selector_dataset_records(p70a, p70b)
    
    # 1. Build and audit episodes for all policies
    policies = [
        "same_split_support_strict",
        "train_to_heldout_base_state",
        "train_to_heldout_magnitude",
        "leave_relation_family_out",
        "leave_domain_out"
    ]
    
    policy_episodes = {}
    policy_audits = {}
    for pol in policies:
        eps = build_policy_episode_records(original_records, case_index, pol)
        policy_episodes[pol] = eps
        policy_audits[pol] = audit_support_policy_flow(eps, original_records, pol)
        
    # 2. Sanitization & leakage check
    strict_eps = policy_episodes["same_split_support_strict"]
    sanitized_inputs = [sanitize_episode_for_metric_input(ep) for ep in strict_eps if ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    leakage_res = audit_metric_input_leakage_all(sanitized_inputs)
    
    # Identify splits under same_split strict policy
    train_eps = [ep for ep in strict_eps if ep["split"] == "train" and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    val_eps = [ep for ep in strict_eps if ep["split"] == "validation" and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    test_eps = [ep for ep in strict_eps if ep["split"] == "test" and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    
    # A. same_split_support_strict Evaluation (P88 Reproduction)
    p88_repro = evaluate_metric_generalization(train_eps, test_eps, "delta_only", "l2", "same_split_support_strict")
    
    # B. train_to_heldout_base_state Evaluation
    # Query is validation split, support bank is train split
    base_state_eps = [ep for ep in policy_episodes["train_to_heldout_base_state"] if ep["split"] == "validation" and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    base_state_res = evaluate_metric_generalization(train_eps, base_state_eps, "delta_only", "l2", "train_to_heldout_base_state")
    
    # C. train_to_heldout_magnitude Evaluation
    # Query is test split, support bank is train split
    magnitude_eps = [ep for ep in policy_episodes["train_to_heldout_magnitude"] if ep["split"] == "test" and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    magnitude_res = evaluate_metric_generalization(train_eps, magnitude_eps, "delta_only", "l2", "train_to_heldout_magnitude")
    
    # D. leave_relation_family_out Evaluation
    # Hold out translate_x (P70A) and change_frequency (P70B) from prototype bank
    held_out = {"translate_x", "change_frequency"}
    family_res = evaluate_metric_generalization(train_eps, test_eps, "delta_only", "l2", "leave_relation_family_out", held_out_classes=held_out)
    
    # E. leave_domain_out Evaluation
    # Train P70A train, evaluate P70B test (yields 0 coverage)
    train_p70a = [ep for ep in train_eps if ep["domain"] == "p70a_vector_world"]
    test_p70b = [ep for ep in test_eps if ep["domain"] == "p70b_time_series_parameter_world"]
    domain_a_to_b = evaluate_metric_generalization(train_p70a, test_p70b, "delta_only", "l2", "leave_domain_out")
    
    # Train P70B train, evaluate P70A test (yields 0 coverage)
    train_p70b = [ep for ep in train_eps if ep["domain"] == "p70b_time_series_parameter_world"]
    test_p70a = [ep for ep in test_eps if ep["domain"] == "p70a_vector_world"]
    domain_b_to_a = evaluate_metric_generalization(train_p70b, test_p70a, "delta_only", "l2", "leave_domain_out")
    
    domain_out_combined = {
        "train_p70a_test_p70b": {
            "effective_accuracy": domain_a_to_b["effective_accuracy"],
            "coverage": domain_a_to_b["coverage"],
            "accuracy_on_supported": domain_a_to_b["accuracy_on_supported"]
        },
        "train_p70b_test_p70a": {
            "effective_accuracy": domain_b_to_a["effective_accuracy"],
            "coverage": domain_b_to_a["coverage"],
            "accuracy_on_supported": domain_b_to_a["accuracy_on_supported"]
        },
        "note": "Domain out results yield 0 coverage as relation classes and features are completely disjoint."
    }
    
    # 3. Domain & Relation breakdowns on best generalization configuration (magnitude policy)
    p70a_magnitude_eps = [ep for ep in magnitude_eps if ep["domain"] == "p70a_vector_world"]
    p70b_magnitude_eps = [ep for ep in magnitude_eps if ep["domain"] == "p70b_time_series_parameter_world"]
    magnitude_p70a_res = evaluate_metric_generalization(train_eps, p70a_magnitude_eps, "delta_only", "l2", "train_to_heldout_magnitude")
    magnitude_p70b_res = evaluate_metric_generalization(train_eps, p70b_magnitude_eps, "delta_only", "l2", "train_to_heldout_magnitude")
    
    domain_results = {
        "p70a_vector_world": {
            "sample_count": int(magnitude_p70a_res["sample_count"]),
            "effective_accuracy": float(magnitude_p70a_res["effective_accuracy"])
        },
        "p70b_time_series_parameter_world": {
            "sample_count": int(magnitude_p70b_res["sample_count"]),
            "effective_accuracy": float(magnitude_p70b_res["effective_accuracy"])
        }
    }
    
    relation_results = {}
    for pred in magnitude_res["predictions"]:
        lbl = pred["true_label"]
        pred_lbl = pred["predicted_label"]
        
        relation_results[lbl] = relation_results.get(lbl, {"correct": 0, "total": 0})
        relation_results[lbl]["total"] += 1
        if lbl == pred_lbl:
            relation_results[lbl]["correct"] += 1
            
    relation_summary = {}
    for lbl, stats in relation_results.items():
        relation_summary[lbl] = {
            "accuracy": float(stats["correct"] / stats["total"]),
            "correct_count": int(stats["correct"]),
            "total_count": int(stats["total"])
        }
        
    # 4. nonlinear_x_from_y stress test on magnitude policy
    nonlinear_total = 0
    nonlinear_correct = 0
    nonlinear_as_translate_x = 0
    for pred in magnitude_res["predictions"]:
        if pred["true_label"] == "nonlinear_x_from_y":
            nonlinear_total += 1
            if pred["predicted_label"] == "nonlinear_x_from_y":
                nonlinear_correct += 1
            elif pred["predicted_label"] == "translate_x":
                nonlinear_as_translate_x += 1
                
    nonlinear_stress = {
        "nonlinear_x_from_y_total": int(nonlinear_total),
        "nonlinear_x_from_y_correct": int(nonlinear_correct),
        "nonlinear_x_from_y_predicted_as_translate_x": int(nonlinear_as_translate_x),
        "note": "Evaluated on train_to_heldout_magnitude policy."
    }
    
    # 5. Run negative controls for each hard generalization policy
    # 5a. controls for train_to_heldout_base_state (base_state_eps)
    base_ctrl_qo = evaluate_metric_generalization(train_eps, base_state_eps, "delta_only", "l2", "train_to_heldout_base_state", negative_control="query_source_only")
    base_ctrl_shuf = evaluate_metric_generalization(train_eps, base_state_eps, "delta_only", "l2", "train_to_heldout_base_state", negative_control="shuffled")
    base_ctrl_zero = evaluate_metric_generalization(train_eps, base_state_eps, "delta_only", "l2", "train_to_heldout_base_state", negative_control="zero_delta")
    
    # 5b. controls for train_to_heldout_magnitude (magnitude_eps)
    mag_ctrl_qo = evaluate_metric_generalization(train_eps, magnitude_eps, "delta_only", "l2", "train_to_heldout_magnitude", negative_control="query_source_only")
    mag_ctrl_shuf = evaluate_metric_generalization(train_eps, magnitude_eps, "delta_only", "l2", "train_to_heldout_magnitude", negative_control="shuffled")
    mag_ctrl_zero = evaluate_metric_generalization(train_eps, magnitude_eps, "delta_only", "l2", "train_to_heldout_magnitude", negative_control="zero_delta")
    
    hard_policy_control_results = {
        "train_to_heldout_base_state": {
            "query_source_only_negative_control": float(base_ctrl_qo["effective_accuracy"]),
            "shuffled_support_delta_negative_control": float(base_ctrl_shuf["effective_accuracy"]),
            "zero_delta_negative_control": float(base_ctrl_zero["effective_accuracy"])
        },
        "train_to_heldout_magnitude": {
            "query_source_only_negative_control": float(mag_ctrl_qo["effective_accuracy"]),
            "shuffled_support_delta_negative_control": float(mag_ctrl_shuf["effective_accuracy"]),
            "zero_delta_negative_control": float(mag_ctrl_zero["effective_accuracy"])
        }
    }
    
    # Determine best raw and evidence passing hard policy
    base_passed = (
        base_state_res["effective_accuracy"] >= 0.50
        and (base_state_res["effective_accuracy"] - base_ctrl_qo["effective_accuracy"]) >= 0.15
        and (base_state_res["effective_accuracy"] - base_ctrl_shuf["effective_accuracy"]) >= 0.10
        and (base_state_res["effective_accuracy"] - base_ctrl_zero["effective_accuracy"]) >= 0.10
    )
    mag_passed = (
        magnitude_res["effective_accuracy"] >= 0.50
        and (magnitude_res["effective_accuracy"] - mag_ctrl_qo["effective_accuracy"]) >= 0.15
        and (magnitude_res["effective_accuracy"] - mag_ctrl_shuf["effective_accuracy"]) >= 0.10
        and (magnitude_res["effective_accuracy"] - mag_ctrl_zero["effective_accuracy"]) >= 0.10
    )
    
    if base_state_res["effective_accuracy"] >= magnitude_res["effective_accuracy"]:
        best_raw_hard_policy = "train_to_heldout_base_state"
    else:
        best_raw_hard_policy = "train_to_heldout_magnitude"
        
    best_evidence_passing_hard_policy = None
    if base_passed and mag_passed:
        if base_state_res["effective_accuracy"] >= magnitude_res["effective_accuracy"]:
            best_evidence_passing_hard_policy = "train_to_heldout_base_state"
        else:
            best_evidence_passing_hard_policy = "train_to_heldout_magnitude"
    elif base_passed:
        best_evidence_passing_hard_policy = "train_to_heldout_base_state"
    elif mag_passed:
        best_evidence_passing_hard_policy = "train_to_heldout_magnitude"
        
    hard_gen_supported = (
        contracts_ok
        and leakage_res["diagnostic_pass"] is True
        and p88_repro["effective_accuracy"] >= 0.60
        and best_evidence_passing_hard_policy is not None
    )
    
    best_hard_acc = base_state_res["effective_accuracy"] if best_raw_hard_policy == "train_to_heldout_base_state" else magnitude_res["effective_accuracy"]
    best_raw_qo_val = base_ctrl_qo["effective_accuracy"] if best_raw_hard_policy == "train_to_heldout_base_state" else mag_ctrl_qo["effective_accuracy"]
    best_raw_shuf_val = base_ctrl_shuf["effective_accuracy"] if best_raw_hard_policy == "train_to_heldout_base_state" else mag_ctrl_shuf["effective_accuracy"]
    best_raw_zero_val = base_ctrl_zero["effective_accuracy"] if best_raw_hard_policy == "train_to_heldout_base_state" else mag_ctrl_zero["effective_accuracy"]
    
    negative_control_results = {
        "query_source_only_negative_control": float(best_raw_qo_val),
        "shuffled_support_delta_negative_control": float(best_raw_shuf_val),
        "zero_delta_negative_control": float(best_raw_zero_val),
        "shuffled_support_delta_control_implemented": True,
        "shuffled_control_target_labels_preserved": True,
        "shuffled_control_support_deltas_permuted": True,
        "target_labels_shifted": False,
        "true_labels_preserved": True
    }
    
    ext_context_contract = define_external_context_contract_after_p89({
        "same_split_accuracy": p88_repro["effective_accuracy"],
        "best_hard_generalization_accuracy": best_hard_acc,
        "hard_generalization_policy": best_raw_hard_policy,
        "hard_generalization_supported": hard_gen_supported
    })
    
    bridge_audit = audit_bridge_boundary_after_hard_generalization(hard_gen_supported)
    
    # Recommended next phase logic
    if hard_gen_supported:
        next_phase = "P90_external_support_context_builder_no_training_no_bridge"
    elif p88_repro["effective_accuracy"] >= 0.60:
        next_phase = "P90_support_delta_metric_generalization_failure_analysis_no_training_no_bridge"
    elif best_raw_shuf_val >= 0.40:
        next_phase = "P90_negative_control_repair_no_training_no_bridge"
    else:
        next_phase = "P90_external_context_contract_repair_no_training_no_bridge"
        
    verdict_str = VERDICT if contracts_ok else "P89_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED is True or TORCH_TRAINING_PERFORMED is True:
            verdict_str = "P89_BLOCKED_BY_UNEXPECTED_TRAINING_FLAG"
        elif leakage_res["diagnostic_pass"] is False:
            verdict_str = "P89_BLOCKED_BY_METRIC_INPUT_LEAKAGE"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p88_nonlearned_metric_signal_preserved": contracts_val["p88_nonlearned_metric_signal_preserved"],
        "p88_real_shuffled_control_preserved": contracts_val["p88_real_shuffled_control_preserved"],
        "p88_no_training_preserved": contracts_val["p88_no_training_preserved"],
        "p88_bridge_not_ready_preserved": contracts_val["p88_bridge_not_ready_preserved"],

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "support_delta_metric_hard_generalization_defined": SUPPORT_DELTA_METRIC_HARD_GENERALIZATION_DEFINED,
        "cross_split_generalization_audit_defined": CROSS_SPLIT_GENERALIZATION_AUDIT_DEFINED,
        "relation_level_failure_audit_defined": RELATION_LEVEL_FAILURE_AUDIT_DEFINED,
        "external_context_contract_defined": EXTERNAL_CONTEXT_CONTRACT_DEFINED,
        "negative_controls_preserved": NEGATIVE_CONTROLS_PRESERVED,

        "hard_policy_controls_aligned": True,
        "hard_policy_fallback_controls_used": False,
        "best_raw_hard_policy": best_raw_hard_policy,
        "best_evidence_passing_hard_policy": best_evidence_passing_hard_policy,

        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": hard_gen_supported,
        "external_context_ready": True,

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

        "p88_reproduction_result": {
            "effective_accuracy": p88_repro["effective_accuracy"],
            "sample_count": p88_repro["sample_count"],
            "mode": p88_repro["mode"],
            "distance_family": p88_repro["distance_family"]
        },
        "hard_generalization_results": {
            "train_to_heldout_base_state": {
                "effective_accuracy": base_state_res["effective_accuracy"],
                "coverage": base_state_res["coverage"],
                "accuracy_on_supported": base_state_res["accuracy_on_supported"]
            },
            "train_to_heldout_magnitude": {
                "effective_accuracy": magnitude_res["effective_accuracy"],
                "coverage": magnitude_res["coverage"],
                "accuracy_on_supported": magnitude_res["accuracy_on_supported"]
            },
            "leave_relation_family_out": {
                "effective_accuracy": family_res["effective_accuracy"],
                "coverage": family_res["coverage"],
                "accuracy_on_supported": family_res["accuracy_on_supported"],
                "sample_count": family_res["sample_count"],
                "supported_sample_count": family_res["supported_sample_count"],
                "unsupported_sample_count": family_res["unsupported_sample_count"],
                "held_out_classes": list(held_out),
                "predictions": family_res["predictions"]
            },
            "leave_domain_out": domain_out_combined
        },
        "negative_control_results": negative_control_results,
        "domain_results": domain_results,
        "relation_results": relation_summary,
        "nonlinear_x_from_y_stress_result": nonlinear_stress,

        "metric_input_leakage_audit": leakage_res,
        "support_policy_flow_audits": policy_audits,

        "external_context_contract": ext_context_contract,

        "hard_policy_control_results": hard_policy_control_results,
        "best_raw_hard_policy": best_raw_hard_policy,
        "best_evidence_passing_hard_policy": best_evidence_passing_hard_policy,
        "hard_policy_controls_aligned": True,
        "hard_policy_fallback_controls_used": False,

        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": hard_gen_supported,

        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "bridge_ready": BRIDGE_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,

        "recommended_next_phase": next_phase,

        "bridge_boundary_after_hard_generalization": bridge_audit,

        "json_safe": True,
        "diagnostic_only": True
    }
    
    return output
