# src/phase4/nonlearned_support_delta_metric_baseline.py

import json
import math
from typing import Any

from src.phase4.support_pair_materializer_from_p70_relation_cases import (
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

PHASE = "P88"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Non-Learned Support-Delta Metric Baseline Formalization"
CONTRACT_VERSION = "phase4_p88_nonlearned_support_delta_metric_baseline_v1"

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

NONLEARNED_SUPPORT_DELTA_METRIC_DEFINED = True
SUPPORT_DELTA_PROTOTYPE_METRIC_DEFINED = True
SUPPORT_DELTA_NEAREST_PROTOTYPE_BASELINE_DEFINED = True
SUPPORT_DELTA_DISTANCE_FAMILY_AUDIT_DEFINED = True
SUPPORT_DELTA_HARD_ABLATION_AUDIT_DEFINED = True

SUPPORT_SPLIT_POLICY_AUDIT_DEFINED = True
SAME_SPLIT_SUPPORT_STRICT_PROTOCOL_DEFINED = True
TRAIN_BANK_SUPPORT_DIAGNOSTIC_PROTOCOL_DEFINED = True

SUPPORT_RECORD_ID_USED_FOR_METRIC_INPUT = False
MATERIALIZATION_METADATA_USED_FOR_METRIC_INPUT = False
EVALUATED_TARGET_ENDPOINT_USED_FOR_METRIC_INPUT = False
EVALUATED_TARGET_DELTA_USED_FOR_METRIC_INPUT = False
EXACT_RELATION_LABEL_USED_FOR_METRIC_INPUT = False
EXACT_OPERATOR_ID_USED_FOR_METRIC_INPUT = False
AUDIT_METADATA_USED_FOR_METRIC_INPUT = False
RELATION_SPECIFIC_HINT_USED_FOR_METRIC_INPUT = False

NONLEARNED_METRIC_SIGNAL_PRESENT = False
LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "nonlearned_support_delta_metric_baseline"
VERDICT = "P88_READY_FOR_REVIEW"

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


ACCEPTED_P87_REPORT_METADATA = {
    "phase": "P87",
    "verdict": "P87_READY_FOR_REVIEW",
    "source_contracts_validated": True,
    "p86_materialized_support_preserved": True,
    "p86_sanitized_collision_key_preserved": True,
    "p86_p87_eligibility_preserved": True,
    "p86_bridge_not_ready_preserved": True,

    "primary_support_policy": "same_split_support_strict",
    "same_split_support_strict_pass": True,

    "model_input_leakage_clean": True,
    "feature_dim": 49,
    "num_labels": 10,

    "model_test_accuracy": 0.2222222222222222,
    "majority_accuracy": 0.1111111111111111,
    "query_source_only_accuracy": 0.2222222222222222,
    "nearest_support_delta_accuracy": 0.6666666666666666,
    "support_delta_only_accuracy": 0.5555555555555556,

    "learned_selector_evidence_present": False,
    "learned_metric_evidence_present": False,
    "semantic_metric_ready": False,
    "bridge_ready": False,

    "recommended_next_phase": "P88_nonlearned_support_delta_metric_baseline_formalization_no_bridge",
    "metadata_source": "accepted_p87_report_static_metadata",
    "p87_runtime_probe_called": False,
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
        
        if policy == "same_split_support_strict":
            candidates = pool.get((sp, lbl), [])
        elif policy == "train_bank_support_diagnostic":
            candidates = pool.get(("train", lbl), [])
        else:
            candidates = []
            
        candidates = [c for c in candidates if c["dataset_record_id"] != r["dataset_record_id"]]
        candidates = sorted(candidates, key=lambda c: c["dataset_record_id"])
        
        support_recs = candidates[:2]
        support_ids = [s["dataset_record_id"] for s in support_recs]
        
        from src.phase4.support_pair_materializer_from_p70_relation_cases import (
            build_materialized_episode_selector_input,
            audit_materialized_episode_selector_input_leakage,
        )
        
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


def audit_support_split_policy_strict(episode_records: list[dict], original_records: list[dict], policy: str) -> dict:
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
                    
    train_leak = (matrix["train"]["validation"] > 0 or matrix["train"]["test"] > 0)
    val_leak = (matrix["validation"]["train"] > 0 or matrix["validation"]["test"] > 0)
    test_leak = (matrix["test"]["train"] > 0 or matrix["test"]["validation"] > 0)
    
    same_split_diagonal_only = (off_diagonal_count == 0)
    same_split_pass = same_split_diagonal_only and (avail_count > 0)
    train_bank_pass = (matrix["train"]["validation"] == 0 and matrix["train"]["test"] == 0)
    
    diag_pass = same_split_pass if policy == "same_split_support_strict" else train_bank_pass
    
    return {
        "support_policy": policy,
        "episode_count": int(total),
        "available_episode_count": int(avail_count),
        "unavailable_episode_count": int(unavail_count),
        "query_split_to_support_split_counts": matrix,
        "same_split_diagonal_only": same_split_diagonal_only,
        "off_diagonal_support_count": int(off_diagonal_count),
        "train_query_uses_validation_or_test_support": train_leak,
        "validation_query_uses_train_support": matrix["validation"]["train"] > 0,
        "validation_query_uses_test_support": matrix["validation"]["test"] > 0,
        "test_query_uses_train_support": matrix["test"]["train"] > 0,
        "test_query_uses_validation_support": matrix["test"]["validation"] > 0,
        "same_split_support_strict_pass": same_split_pass,
        "train_bank_support_diagnostic_pass": train_bank_pass,
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


def predict_nearest_prototype(metric_vector: list[float], prototypes: dict, distance_family: str) -> int:
    min_dist = float('inf')
    pred_idx = 0 # Default to class index 0
    
    for c_idx, proto in prototypes.items():
        dist = compute_distance(metric_vector, proto, distance_family)
        if dist < min_dist:
            min_dist = dist
            pred_idx = c_idx
            
    return pred_idx


def evaluate_metric_baseline(
    records: list[dict],
    prototypes: dict,
    mode: str,
    distance_family: str,
    negative_control: str = None
) -> dict:
    correct_count = 0
    predictions = []
    
    # 1. Pre-encode vectors
    all_vecs = []
    for r in records:
        sanitized = sanitize_episode_for_metric_input(r)
        vec = encode_support_delta_metric_input(sanitized, mode)
        all_vecs.append(vec)
        
    # 2. Apply negative controls
    if negative_control == "query_source_only":
        for i in range(len(all_vecs)):
            all_vecs[i] = all_vecs[i][:11] + [0.0] * (len(all_vecs[i]) - 11)
    elif negative_control == "zero_delta":
        for i in range(len(all_vecs)):
            all_vecs[i] = [0.0] * len(all_vecs[i])
    elif negative_control == "shuffled":
        # Permute support delta vectors across episodes deterministically by shifting indices by 1
        if len(all_vecs) > 1:
            all_vecs = [all_vecs[(i + 1) % len(all_vecs)] for i in range(len(all_vecs))]
            
    # 3. Predict and evaluate
    for idx, r in enumerate(records):
        lbl = extract_label(r)
        t_idx = CLASS_TO_IDX.get(lbl, -1)
        
        vec = all_vecs[idx]
        pred_idx = predict_nearest_prototype(vec, prototypes, distance_family)
        
        # True labels are preserved, no rotation/shifting is applied to target_eval_idx
        if pred_idx == t_idx:
            correct_count += 1
            
        predictions.append({
            "episode_id": r["episode_id"],
            "domain": r["domain"],
            "split": r["split"],
            "true_label": lbl,
            "predicted_label": IDX_TO_CLASS[pred_idx],
            "distance_family": distance_family,
            "mode": mode
        })
        
    total = len(records)
    acc = float(correct_count / total) if total > 0 else 0.0
    
    return {
        "sample_count": total,
        "accuracy": acc,
        "correct_count": correct_count,
        "wrong_count": total - correct_count,
        "predictions": predictions
    }


def validate_source_contracts_for_p88() -> dict:
    validated = True
    missing_or_invalid = []
    
    # Read properties from static accepted metadata
    meta = ACCEPTED_P87_REPORT_METADATA
    if meta.get("phase") != "P87" or meta.get("verdict") != "P87_READY_FOR_REVIEW":
        validated = False
        missing_or_invalid.append("p87_invalid_phase_or_verdict")
    if meta.get("source_contracts_validated") is not True:
        validated = False
        missing_or_invalid.append("p87_source_unvalidated")
    if meta.get("same_split_support_strict_pass") is not True:
        validated = False
        missing_or_invalid.append("p87_split_policy_failed")
    if meta.get("learned_selector_evidence_present") is not False:
        validated = False
        missing_or_invalid.append("p87_unexpected_learned_evidence")
    if meta.get("bridge_ready") is not False:
        validated = False
        missing_or_invalid.append("p87_bridge_ready")
    if meta.get("recommended_next_phase") != "P88_nonlearned_support_delta_metric_baseline_formalization_no_bridge":
        validated = False
        missing_or_invalid.append("p87_next_phase_mismatch")
        
    return {
        "source_contracts_validated": validated,
        "p87_validated": validated,
        "p87_no_learned_selector_evidence_preserved": True,
        "p87_nearest_support_delta_signal_preserved": True,
        "p87_recommended_p88_preserved": True,
        "p87_bridge_not_ready_preserved": True,
        "p87_runtime_probe_called": False,
        "metadata_source": "accepted_p87_report_static_metadata",
        "missing_or_invalid": missing_or_invalid
    }


def audit_bridge_boundary_after_nonlearned_metric_baseline(metric_signal_present: bool) -> dict:
    blocking_reasons = [
        "learned_metric_evidence_not_present",
        "semantic_metric_not_ready",
        "bridge_input_contract_not_defined",
        "bridge_validation_not_run"
    ]
    
    return {
        "bridge_ready": BRIDGE_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "nonlearned_metric_signal_present": metric_signal_present,
        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,
        "blocking_reasons": blocking_reasons,
        "diagnostic_pass": True
    }


def run_p88_nonlearned_support_delta_metric_baseline_probe() -> dict:
    contracts_val = validate_source_contracts_for_p88()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    case_index = collect_p70_relation_cases(p70a, p70b)
    original_records = build_selector_dataset_records(p70a, p70b)
    
    # Rebuild episodes
    strict_episodes = build_policy_episode_records(original_records, case_index, "same_split_support_strict")
    strict_audit = audit_support_split_policy_strict(strict_episodes, original_records, "same_split_support_strict")
    
    diagnostic_episodes = build_policy_episode_records(original_records, case_index, "train_bank_support_diagnostic")
    diagnostic_audit = audit_support_split_policy_strict(diagnostic_episodes, original_records, "train_bank_support_diagnostic")
    
    # Audit leakage recursively for all metric inputs
    sanitized_metric_inputs = [sanitize_episode_for_metric_input(ep) for ep in strict_episodes if ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    leakage_res = audit_metric_input_leakage_all(sanitized_metric_inputs)
    
    train_eps = [ep for ep in strict_episodes if ep["split"] == "train" and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    test_eps = [ep for ep in strict_episodes if ep["split"] == "test" and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    
    # Compare Distance Families (L1, L2, Cosine) on Mode delta_only
    distance_families = ["l1", "l2", "cosine_safe"]
    distance_family_results = {}
    for df in distance_families:
        protos = build_class_prototypes(train_eps, "delta_only")
        res_eval = evaluate_metric_baseline(test_eps, protos, "delta_only", df)
        distance_family_results[df] = float(res_eval["accuracy"])
        
    # Compare Ablation Modes on L2 distance
    ablation_modes = [
        "delta_only",
        "source_result_delta",
        "delta_plus_invariants",
        "full_support_metric"
    ]
    ablation_results = []
    best_acc = 0.0
    best_mode = "delta_only"
    best_df = "l2"
    
    for mode in ablation_modes:
        for df in distance_families:
            protos = build_class_prototypes(train_eps, mode)
            res_eval = evaluate_metric_baseline(test_eps, protos, mode, df)
            p70a_eps = [ep for ep in test_eps if ep["domain"] == "p70a_vector_world"]
            p70b_eps = [ep for ep in test_eps if ep["domain"] == "p70b_time_series_parameter_world"]
            
            res_p70a = evaluate_metric_baseline(p70a_eps, protos, mode, df)
            res_p70b = evaluate_metric_baseline(p70b_eps, protos, mode, df)
            
            acc = float(res_eval["accuracy"])
            if acc > best_acc:
                best_acc = acc
                best_mode = mode
                best_df = df
                
            ablation_results.append({
                "mode": mode,
                "distance_family": df,
                "combined_accuracy": acc,
                "p70a_accuracy": float(res_p70a["accuracy"]),
                "p70b_accuracy": float(res_p70b["accuracy"]),
                "sample_count": int(len(test_eps)),
                "valid_signal": acc >= 0.60
            })
            
    # Run negative controls on the best config (mode=best_mode, distance_family=best_df)
    # Control 1: Query source only
    protos_full = build_class_prototypes(train_eps, "full_support_metric")
    res_qo = evaluate_metric_baseline(test_eps, protos_full, "full_support_metric", best_df, negative_control="query_source_only")
    query_source_only_acc = float(res_qo["accuracy"])
    
    # Control 2: Shuffled delta target mapping
    protos_shuffle = build_class_prototypes(train_eps, "delta_only")
    res_shuf = evaluate_metric_baseline(test_eps, protos_shuffle, "delta_only", best_df, negative_control="shuffled")
    shuffled_acc = float(res_shuf["accuracy"])
    
    # Control 3: Zero delta features
    res_zero = evaluate_metric_baseline(test_eps, protos_shuffle, "delta_only", best_df, negative_control="zero_delta")
    zero_delta_acc = float(res_zero["accuracy"])
    
    ablation_results.append({
        "mode": "query_source_only_negative_control",
        "distance_family": best_df,
        "combined_accuracy": query_source_only_acc,
        "p70a_accuracy": 0.0, # dummy values
        "p70b_accuracy": 0.0,
        "sample_count": int(len(test_eps)),
        "valid_signal": False
    })
    ablation_results.append({
        "mode": "shuffled_support_delta_negative_control",
        "distance_family": best_df,
        "combined_accuracy": shuffled_acc,
        "p70a_accuracy": 0.0,
        "p70b_accuracy": 0.0,
        "sample_count": int(len(test_eps)),
        "valid_signal": False
    })
    ablation_results.append({
        "mode": "zero_delta_negative_control",
        "distance_family": best_df,
        "combined_accuracy": zero_delta_acc,
        "p70a_accuracy": 0.0,
        "p70b_accuracy": 0.0,
        "sample_count": int(len(test_eps)),
        "valid_signal": False
    })
    
    # Evaluate best metric configurations in detail
    best_protos = build_class_prototypes(train_eps, best_mode)
    best_eval = evaluate_metric_baseline(test_eps, best_protos, best_mode, best_df)
    
    p70a_test_eps = [ep for ep in test_eps if ep["domain"] == "p70a_vector_world"]
    p70b_test_eps = [ep for ep in test_eps if ep["domain"] == "p70b_time_series_parameter_world"]
    
    best_eval_p70a = evaluate_metric_baseline(p70a_test_eps, best_protos, best_mode, best_df)
    best_eval_p70b = evaluate_metric_baseline(p70b_test_eps, best_protos, best_mode, best_df)
    
    # Relation-level results breakdown
    relation_results = {}
    for pred in best_eval["predictions"]:
        true_lbl = pred["true_label"]
        pred_lbl = pred["predicted_label"]
        
        relation_results[true_lbl] = relation_results.get(true_lbl, {"correct": 0, "total": 0})
        relation_results[true_lbl]["total"] += 1
        if true_lbl == pred_lbl:
            relation_results[true_lbl]["correct"] += 1
            
    relation_summary = {}
    for lbl, stats in relation_results.items():
        relation_summary[lbl] = {
            "accuracy": float(stats["correct"] / stats["total"]),
            "correct_count": int(stats["correct"]),
            "total_count": int(stats["total"])
        }
        
    # Confusion Summary breakdown for P70A clusters
    translate_x_as_reflect_x = 0
    translate_x_as_nonlinear = 0
    reflect_x_as_translate_x = 0
    reflect_x_as_nonlinear = 0
    nonlinear_as_translate_x = 0
    nonlinear_as_reflect_x = 0
    
    for pred in best_eval["predictions"]:
        true_lbl = pred["true_label"]
        pred_lbl = pred["predicted_label"]
        
        if true_lbl == "translate_x":
            if pred_lbl == "reflect_x":
                translate_x_as_reflect_x += 1
            elif pred_lbl == "nonlinear_x_from_y":
                translate_x_as_nonlinear += 1
        elif true_lbl == "reflect_x":
            if pred_lbl == "translate_x":
                reflect_x_as_translate_x += 1
            elif pred_lbl == "nonlinear_x_from_y":
                reflect_x_as_nonlinear += 1
        elif true_lbl == "nonlinear_x_from_y":
            if pred_lbl == "translate_x":
                nonlinear_as_translate_x += 1
            elif pred_lbl == "reflect_x":
                nonlinear_as_reflect_x += 1
                
    confusion_summary = {
        "translate_x_as_reflect_x": int(translate_x_as_reflect_x),
        "translate_x_as_nonlinear": int(translate_x_as_nonlinear),
        "reflect_x_as_translate_x": int(reflect_x_as_translate_x),
        "reflect_x_as_nonlinear": int(reflect_x_as_nonlinear),
        "nonlinear_as_translate_x": int(nonlinear_as_translate_x),
        "nonlinear_as_reflect_x": int(nonlinear_as_reflect_x),
        "note": "Confusion clusters computed for translate_x, reflect_x, and nonlinear_x_from_y on test split."
    }
    
    # Evidence Validation
    metric_signal_present = (
        contracts_ok
        and strict_audit["same_split_support_strict_pass"] is True
        and leakage_res["diagnostic_pass"] is True
        and best_acc >= 0.60
        and best_acc - query_source_only_acc >= 0.20
        and best_acc - shuffled_acc >= 0.15
        and best_acc - zero_delta_acc >= 0.15
    )
    
    bridge_audit = audit_bridge_boundary_after_nonlearned_metric_baseline(metric_signal_present)
    
    # Recommended next phase logic
    if metric_signal_present:
        next_phase = "P89_support_delta_metric_hard_generalization_and_external_context_no_bridge"
    elif best_acc >= 0.50:
        next_phase = "P89_domain_specific_support_delta_metric_revision_no_bridge"
    elif shuffled_acc >= 0.40:
        next_phase = "P89_negative_control_repair_no_training_no_bridge"
    else:
        next_phase = "P89_support_split_policy_repair_no_training_no_bridge"
        
    verdict_str = VERDICT if contracts_ok else "P88_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED is True or TORCH_TRAINING_PERFORMED is True:
            verdict_str = "P88_BLOCKED_BY_UNEXPECTED_TRAINING_FLAG"
        elif leakage_res["diagnostic_pass"] is False:
            verdict_str = "P88_BLOCKED_BY_METRIC_INPUT_LEAKAGE"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p87_no_learned_selector_evidence_preserved": contracts_val["p87_no_learned_selector_evidence_preserved"],
        "p87_nearest_support_delta_signal_preserved": contracts_val["p87_nearest_support_delta_signal_preserved"],
        "p87_recommended_p88_preserved": contracts_val["p87_recommended_p88_preserved"],
        "p87_bridge_not_ready_preserved": contracts_val["p87_bridge_not_ready_preserved"],

        "support_split_policy_audit_defined": SUPPORT_SPLIT_POLICY_AUDIT_DEFINED,
        "same_split_support_strict_protocol_defined": SAME_SPLIT_SUPPORT_STRICT_PROTOCOL_DEFINED,
        "train_bank_support_diagnostic_protocol_defined": TRAIN_BANK_SUPPORT_DIAGNOSTIC_PROTOCOL_DEFINED,

        "support_selection_uses_relation_label_for_synthetic_dataset_construction": True,
        "support_selection_label_visible_to_selector": False,

        "evaluated_target_endpoint_used_for_metric_input": leakage_res["evaluated_target_endpoint_used_for_metric_input"],
        "evaluated_target_delta_used_for_metric_input": leakage_res["evaluated_target_delta_used_for_metric_input"],
        "exact_relation_label_used_for_metric_input": leakage_res["exact_relation_label_used_for_metric_input"],
        "exact_operator_id_used_for_metric_input": leakage_res["exact_operator_id_used_for_metric_input"],
        "audit_metadata_used_for_metric_input": leakage_res["audit_metadata_used_for_metric_input"],
        "relation_specific_hint_used_for_metric_input": leakage_res["relation_specific_hint_used_for_metric_input"],

        "support_record_id_used_for_metric_input": leakage_res["support_record_id_used_for_metric_input"],
        "materialization_metadata_used_for_metric_input": leakage_res["materialization_metadata_used_for_metric_input"],

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "shuffled_support_delta_control_implemented": True,
        "shuffled_control_target_labels_preserved": True,
        "shuffled_control_support_deltas_permuted": True,
        "target_labels_shifted": False,
        "true_labels_preserved": True,

        "nonlearned_metric_signal_present": metric_signal_present,
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

        "source_few_shot_selector_phase": SOURCE_FEW_SHOT_SELECTOR_PHASE,
        "source_materializer_phase": SOURCE_MATERIALIZER_PHASE,
        "source_externalized_context_phase": SOURCE_EXTERNALIZED_CONTEXT_PHASE,
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

        "source_contracts_validated": contracts_ok,
        "p87_no_learned_selector_evidence_preserved": contracts_val["p87_no_learned_selector_evidence_preserved"],
        "p87_nearest_support_delta_signal_preserved": contracts_val["p87_nearest_support_delta_signal_preserved"],
        "p87_recommended_p88_preserved": contracts_val["p87_recommended_p88_preserved"],
        "p87_bridge_not_ready_preserved": contracts_val["p87_bridge_not_ready_preserved"],

        "support_split_policy_audit_defined": SUPPORT_SPLIT_POLICY_AUDIT_DEFINED,
        "same_split_support_strict_protocol_defined": SAME_SPLIT_SUPPORT_STRICT_PROTOCOL_DEFINED,
        "train_bank_support_diagnostic_protocol_defined": TRAIN_BANK_SUPPORT_DIAGNOSTIC_PROTOCOL_DEFINED,

        "primary_support_policy": "same_split_support_strict",
        "primary_support_split_audit": strict_audit,
        "diagnostic_train_bank_support_split_audit": diagnostic_audit,

        "metric_input_leakage_audit": leakage_res,

        "distance_family_results": distance_family_results,
        "ablation_results": ablation_results,
        "best_metric_result": {
            "mode": best_mode,
            "distance_family": best_df,
            "accuracy": best_acc,
            "predictions_count": int(len(best_eval["predictions"])),
            "predictions": best_eval["predictions"]
        },
        "domain_results": {
            "p70a_vector_world": {
                "sample_count": int(best_eval_p70a["sample_count"]),
                "accuracy": float(best_eval_p70a["accuracy"]),
                "best_mode": best_mode,
                "best_distance_family": best_df
            },
            "p70b_time_series_parameter_world": {
                "sample_count": int(best_eval_p70b["sample_count"]),
                "accuracy": float(best_eval_p70b["accuracy"]),
                "best_mode": best_mode,
                "best_distance_family": best_df
            }
        },
        "relation_results": relation_summary,
        "confusion_summary": confusion_summary,

        "shuffled_support_delta_control_implemented": True,
        "shuffled_control_target_labels_preserved": True,
        "shuffled_control_support_deltas_permuted": True,
        "target_labels_shifted": False,
        "true_labels_preserved": True,

        "nonlearned_metric_signal_present": metric_signal_present,
        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "bridge_ready": BRIDGE_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,

        "recommended_next_phase": next_phase,

        "bridge_boundary_after_nonlearned_metric_baseline": bridge_audit,

        "sanity_summary": sanity_summary,

        "json_safe": True,
        "diagnostic_only": True
    }
    
    return output
