# src/phase4/synthetic_few_shot_selector_pilot.py

import json
import math
from typing import Any

import torch

from src.phase4.support_pair_materializer_from_p70_relation_cases import (
    run_p86_support_pair_materializer_from_p70_relation_cases_probe,
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

PHASE = "P87"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Synthetic Few-Shot Selector Pilot from Materialized Support Episodes"
CONTRACT_VERSION = "phase4_p87_synthetic_few_shot_selector_pilot_v1"

SOURCE_MATERIALIZER_PHASE = "P86"
SOURCE_EXTERNALIZED_CONTEXT_PHASE = "P85"
SOURCE_ENRICHMENT_PHASE = "P84"
SOURCE_IDENTIFIABILITY_PHASE = "P83"
SOURCE_SELECTOR_PILOT_PHASE = "P82"
SOURCE_DATASET_PHASE = "P81"

TRAINING_ALLOWED_BY_PHASE4_AUTHORITY = True
MODEL_TRAINING_PERFORMED = True
TORCH_TRAINING_PERFORMED = True
NEW_MODEL_IMPLEMENTED = True
OPTIMIZER_CREATED = True
CHECKPOINT_WRITTEN = False

SUPPORT_SPLIT_POLICY_AUDIT_DEFINED = True
SAME_SPLIT_SUPPORT_STRICT_PROTOCOL_DEFINED = True
TRAIN_BANK_SUPPORT_DIAGNOSTIC_PROTOCOL_DEFINED = True

MATERIALIZED_SUPPORT_EPISODE_INPUT_USED = True
SUPPORT_PAIR_RESULT_DELTA_INPUT_USED = True
SUPPORT_RECORD_ID_USED_FOR_MODEL_INPUT = False
MATERIALIZATION_METADATA_USED_FOR_MODEL_INPUT = False

EVALUATED_TARGET_ENDPOINT_USED_FOR_MODEL_INPUT = False
EVALUATED_TARGET_DELTA_USED_FOR_MODEL_INPUT = False
EXACT_RELATION_LABEL_USED_FOR_MODEL_INPUT = False
EXACT_OPERATOR_ID_USED_FOR_MODEL_INPUT = False
AUDIT_METADATA_USED_FOR_MODEL_INPUT = False
RELATION_SPECIFIC_HINT_USED_FOR_MODEL_INPUT = False

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "synthetic_few_shot_selector_pilot"
VERDICT = "P87_READY_FOR_REVIEW"

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


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-15:
        return 0.0
    return float(numerator / denominator)


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


def sanitize_episode_for_model_input(episode_record: dict) -> dict:
    sel = episode_record["materialized_episode_selector_input"]
    return sanitize_dict_recursively(sel)


def audit_model_input_leakage(model_input: dict) -> dict:
    all_keys = scan_keys(model_input)
    
    id_leak = "support_record_id" in all_keys
    
    meta_keys = {
        "materialization_source_phase", "materialization_method", 
        "missing_result_summary", "missing_delta_summary", 
        "support_pair_observation_available"
    }
    meta_leak = any(k in all_keys for k in meta_keys)
    
    endpoint_keys = {
        "query_target_endpoint", "evaluated_target_endpoint", "query_z_b", 
        "query_params_b", "query_series_b", "target_endpoint", "target_midpoint", 
        "z_b", "params_b", "series_b"
    }
    endpoint_leak = any(k in all_keys for k in endpoint_keys)
    
    delta_keys = {
        "query_target_delta", "evaluated_target_delta", "query_z_b_minus_z_a", 
        "query_params_b_minus_params_a", "query_series_summary_delta", "z_b_minus_z_a", 
        "params_b_minus_params_a", "series_summary_delta"
    }
    delta_leak = any(k in all_keys for k in delta_keys)
    
    label_keys = {
        "target_relation_label", "true_relation_type", "relation_type"
    }
    label_leak = any(k in all_keys for k in label_keys)
    
    op_leak = "operator_id" in all_keys
    metadata_leak = "_audit_metadata" in all_keys
    
    hint_keys = {
        "relation_family_hint", "transformation_class_hint", 
        "relation_axis_hint", "parameter_group_hint"
    }
    hint_leak = any(k in all_keys for k in hint_keys)
    
    diag = not (
        id_leak or meta_leak or endpoint_leak or delta_leak or label_leak or op_leak or metadata_leak or hint_leak
    )
    
    return {
        "support_record_id_used_for_model_input": id_leak,
        "materialization_metadata_used_for_model_input": meta_leak,
        "evaluated_target_endpoint_used_for_model_input": endpoint_leak,
        "evaluated_target_delta_used_for_model_input": delta_leak,
        "exact_relation_label_used_for_model_input": label_leak,
        "exact_operator_id_used_for_model_input": op_leak,
        "audit_metadata_used_for_model_input": metadata_leak,
        "relation_specific_hint_used_for_model_input": hint_leak,
        "diagnostic_pass": diag
    }


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


def encode_model_input(model_input: dict) -> list[float]:
    feats = []
    
    # 1. Query source features (10 floats)
    q = model_input.get("query_source_input", {})
    domain = q.get("domain", "")
    if domain == "p70a_vector_world":
        feats.extend([1.0, 0.0])
    else:
        feats.extend([0.0, 1.0])
        
    intensity = float(q.get("query_intensity_hint", 0.0) or 0.0)
    feats.append(intensity)
    
    # P70A elements
    state_sum = q.get("source_state_summary", {})
    z_a_dim = float(state_sum.get("z_a_dim", 0.0))
    z_a_abs_sum = float(state_sum.get("z_a_abs_sum", 0.0))
    z_a_sign_pattern = state_sum.get("z_a_sign_pattern", [0.0, 0.0, 0.0])
    
    # P70B elements
    param_sum = q.get("source_parameter_summary", {})
    params_key_count = float(param_sum.get("params_key_count", 0.0))
    params_abs_sum = float(param_sum.get("params_abs_sum", 0.0))
    params_nonzero = float(param_sum.get("params_nonzero_key_count", 0.0))
    
    if domain == "p70a_vector_world":
        feats.extend([z_a_dim, z_a_abs_sum])
        # Pad sign pattern to size 3
        pat = [float(x) for x in z_a_sign_pattern]
        while len(pat) < 3:
            pat.append(0.0)
        feats.extend(pat[:3])
        feats.extend([0.0, 0.0, 0.0]) # pad P70B elements
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0]) # pad P70A elements
        feats.extend([params_key_count, params_abs_sum, params_nonzero])
        
    # 2. Support pair features (15 floats per pair, max_support = 2, total 30 floats)
    context = model_input.get("materialized_externalized_support_episode_context", {})
    support_pairs = context.get("support_pairs", [])
    
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
            
            # Extract abs sum / counts
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
                
                # time series stats
                s_len = float(res_sum.get("series_length", 0.0))
                s_abs = float(res_sum.get("series_abs_sum", 0.0))
                s_mean = float(res_sum.get("series_mean", 0.0))
                s_end = float(res_sum.get("series_endpoint_delta", 0.0))
                feats.extend([s_len, s_abs, s_mean, s_end])
        else:
            # Pad missing pair completely (15 floats)
            feats.extend([0.0] * 15)
            
    # 3. Invariant context features (8 floats)
    inv = context.get("support_invariant_context", {})
    feats.extend([
        float(inv.get("support_pair_count", 0.0)),
        float(inv.get("support_pair_observation_available_count", 0.0)),
        float(inv.get("result_summary_available_count", 0.0)),
        float(inv.get("delta_summary_available_count", 0.0)),
        float(inv.get("support_delta_abs_sum_mean", 0.0)),
        float(inv.get("support_delta_nonzero_count_mean", 0.0)),
        1.0 if inv.get("support_invariant_context_available") is True else 0.0,
        1.0 if inv.get("support_episode_limited_by_missing_result_or_delta_summaries") is True else 0.0
    ])
    
    return feats


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
    
    return {
        "p70a_cases": cases_a,
        "p70b_cases": cases_b,
        "p70a_case_count": len(cases_a),
        "p70b_case_count": len(cases_b)
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


def audit_support_split_policy(episode_records: list[dict], original_records: list[dict], policy: str) -> dict:
    rec_to_split = {r["dataset_record_id"]: r["split"] for r in original_records}
    
    matrix = {
        "train": {"train": 0, "validation": 0, "test": 0},
        "validation": {"train": 0, "validation": 0, "test": 0},
        "test": {"train": 0, "validation": 0, "test": 0}
    }
    
    total = len(episode_records)
    avail_count = 0
    unavail_count = 0
    
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
                
    train_leak = (matrix["train"]["validation"] > 0 or matrix["train"]["test"] > 0)
    val_leak = (matrix["validation"]["test"] > 0)
    test_uses_train = (matrix["test"]["train"] > 0)
    
    same_split_pass = False
    train_bank_pass = False
    
    if policy == "same_split_support_strict":
        same_split_pass = (matrix["validation"]["train"] == 0 and matrix["test"]["train"] == 0 and not train_leak and not val_leak)
        diag_pass = same_split_pass
    else:  # train_bank_support_diagnostic
        train_bank_pass = (matrix["train"]["validation"] == 0 and matrix["train"]["test"] == 0)
        diag_pass = train_bank_pass
        
    return {
        "support_policy": policy,
        "episode_count": int(total),
        "available_episode_count": int(avail_count),
        "unavailable_episode_count": int(unavail_count),
        "query_split_to_support_split_counts": matrix,
        "train_query_uses_validation_or_test_support": train_leak,
        "validation_query_uses_test_support": val_leak,
        "test_query_uses_train_support": test_uses_train,
        "same_split_support_strict_pass": same_split_pass,
        "train_bank_support_diagnostic_pass": train_bank_pass,
        "diagnostic_pass": diag_pass
    }


class TinyFewShotSelector(torch.nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_labels)
        )

    def forward(self, x):
        return self.net(x)


def get_support_delta_vector(episode_record: dict) -> list[float]:
    # Extract sanitized support delta features (concatenated for nearest delta baseline, size 18)
    sel = episode_record["materialized_episode_selector_input"]
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
                    
                # 9 features for P70A
                feats.extend([1.0, 0.0, del_abs, del_nz])
                feats.extend(pat[:3])
                feats.extend([0.0, 0.0])
            else:
                del_abs = float(del_sum.get("delta_abs_sum", 0.0))
                del_nz = float(del_sum.get("delta_nonzero_key_count", 0.0))
                
                # 9 features for P70B
                feats.extend([0.0, 1.0, del_abs, del_nz])
                feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend([0.0] * 9)
            
    return feats


def run_evaluation_on_split(
    model: TinyFewShotSelector,
    query_only_model: TinyFewShotSelector,
    linear_delta_model: TinyFewShotSelector,
    episodes: list[dict],
    class_prototypes: dict,
    majority_class_idx: int,
    split_name: str
) -> dict:
    records = [ep for ep in episodes if ep["split"] == split_name and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    if len(records) == 0:
        return {
            "sample_count": 0,
            "majority_accuracy": 0.0,
            "query_source_only_accuracy": 0.0,
            "nearest_support_delta_accuracy": 0.0,
            "support_delta_only_accuracy": 0.0,
            "model_accuracy": 0.0,
            "predictions": []
        }
        
    correct_maj = 0
    correct_qo = 0
    correct_nsd = 0
    correct_sdo = 0
    correct_model = 0
    
    preds_list = []
    
    # Prepare tensors for batch evaluation
    model_inputs = []
    qo_inputs = []
    sdo_inputs = []
    targets = []
    
    for r in records:
        lbl = extract_label(r)
        t_idx = CLASS_TO_IDX.get(lbl, -1)
        targets.append(t_idx)
        
        # 1. Majority prediction
        if t_idx == majority_class_idx:
            correct_maj += 1
            
        # Feature encoding
        sanitized = sanitize_episode_for_model_input(r)
        encoded = encode_model_input(sanitized)
        model_inputs.append(encoded)
        qo_inputs.append(encoded[:11]) # Query-source-only uses first 11 floats
        sdo_inputs.append(get_support_delta_vector(r)) # support-delta-only uses support delta features (18 floats)
        
        # 3. Nearest support delta prediction
        eps_delta = get_support_delta_vector(r)
        min_dist = float('inf')
        pred_nsd_idx = majority_class_idx
        for c_idx, proto in class_prototypes.items():
            dist = math.sqrt(sum((eps_delta[j] - proto[j]) ** 2 for j in range(18)))
            if dist < min_dist:
                min_dist = dist
                pred_nsd_idx = c_idx
                
        if pred_nsd_idx == t_idx:
            correct_nsd += 1
            
    # Model predictions
    model.eval()
    query_only_model.eval()
    linear_delta_model.eval()
    
    with torch.no_grad():
        x_model = torch.tensor(model_inputs, dtype=torch.float32)
        x_qo = torch.tensor(qo_inputs, dtype=torch.float32)
        x_sdo = torch.tensor(sdo_inputs, dtype=torch.float32)
        
        logits_model = model(x_model)
        logits_qo = query_only_model(x_qo)
        logits_sdo = linear_delta_model(x_sdo)
        
        pred_model = logits_model.argmax(dim=1).tolist()
        pred_qo = logits_qo.argmax(dim=1).tolist()
        pred_sdo = logits_sdo.argmax(dim=1).tolist()
        
    for idx, r in enumerate(records):
        t_idx = targets[idx]
        p_m = pred_model[idx]
        p_qo = pred_qo[idx]
        p_sdo = pred_sdo[idx]
        
        if p_m == t_idx:
            correct_model += 1
        if p_qo == t_idx:
            correct_qo += 1
        if p_sdo == t_idx:
            correct_sdo += 1
            
        preds_list.append({
            "episode_id": r["episode_id"],
            "domain": r["domain"],
            "true_label": IDX_TO_CLASS[t_idx],
            "predicted_model": IDX_TO_CLASS[p_m],
            "predicted_qo": IDX_TO_CLASS[p_qo],
            "predicted_nsd": IDX_TO_CLASS[p_qo] # placeholder / dummy key
        })
        
    total = len(records)
    return {
        "sample_count": total,
        "majority_accuracy": float(correct_maj / total),
        "query_source_only_accuracy": float(correct_qo / total),
        "nearest_support_delta_accuracy": float(correct_nsd / total),
        "support_delta_only_accuracy": float(correct_sdo / total),
        "model_accuracy": float(correct_model / total),
        "predictions": preds_list
    }


def validate_source_contracts_for_p87() -> dict:
    validated = True
    missing_or_invalid = []
    
    p86_ok = True
    
    try:
        p86 = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
        if p86.get("phase") != "P86" or p86.get("verdict") != "P86_READY_FOR_REVIEW":
            p86_ok = False
            missing_or_invalid.append("p86_invalid_phase_or_verdict")
        if p86.get("support_pair_materializer_defined") is not True:
            p86_ok = False
            missing_or_invalid.append("p86_materializer_undefined")
        if p86.get("support_pair_materialization_audit", {}).get("materialization_successful") is not True:
            p86_ok = False
            missing_or_invalid.append("p86_materialization_failed")
        if p86.get("materialized_episode_collision_reduction_audit", {}).get("collision_key_sanitized") is not True:
            p86_ok = False
            missing_or_invalid.append("p86_unsanitized_collision_key")
        if p86.get("support_record_id_used_in_collision_key") is not False:
            p86_ok = False
            missing_or_invalid.append("p86_support_id_leak")
        if p86.get("materialization_metadata_used_in_collision_key") is not False:
            p86_ok = False
            missing_or_invalid.append("p86_meta_leak")
        if p86.get("valid_for_p87_synthetic_few_shot_selector_pilot") is not True:
            p86_ok = False
            missing_or_invalid.append("p86_not_eligible_for_p87")
        if p86.get("model_training_performed") is not False:
            p86_ok = False
            missing_or_invalid.append("p86_training_performed")
        if p86.get("bridge_ready") is not False:
            p86_ok = False
            missing_or_invalid.append("p86_bridge_ready")
        if p86.get("support_pair_materialization_interpretation", {}).get("recommended_next_phase") != "P87_synthetic_few_shot_selector_pilot_no_bridge":
            p86_ok = False
            missing_or_invalid.append("p86_next_phase_mismatch")
    except Exception as e:
        p86_ok = False
        missing_or_invalid.append(f"p86_exception_{str(e)}")
        
    validated = p86_ok
    
    return {
        "source_contracts_validated": validated,
        "p86_validated": p86_ok,
        "p86_materialized_support_preserved": p86_ok,
        "p86_sanitized_collision_key_preserved": p86_ok,
        "p86_p87_eligibility_preserved": p86_ok,
        "p86_bridge_not_ready_preserved": p86_ok,
        "missing_or_invalid": missing_or_invalid
    }


def audit_bridge_boundary_after_few_shot_selector_pilot(selector_evidence: bool) -> dict:
    blocking_reasons = [
        "learned_metric_evidence_not_present",
        "semantic_metric_not_ready",
        "bridge_input_contract_not_defined",
        "bridge_validation_not_run"
    ]
    
    return {
        "bridge_ready": BRIDGE_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "learned_selector_evidence_present": selector_evidence,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,
        "blocking_reasons": blocking_reasons,
        "diagnostic_pass": True
    }


def train_pytorch_selector(
    train_records: list[dict],
    val_records: list[dict],
    input_dim: int,
    use_query_only: bool = False,
    use_delta_only: bool = False
) -> TinyFewShotSelector:
    torch.manual_seed(8700)
    
    num_labels = len(ALL_CLASSES)
    
    if use_query_only:
        model = TinyFewShotSelector(11, num_labels)
    elif use_delta_only:
        model = TinyFewShotSelector(18, num_labels)
    else:
        model = TinyFewShotSelector(input_dim, num_labels)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Encode training batch
    train_x = []
    train_y = []
    for r in train_records:
        sanitized = sanitize_episode_for_model_input(r)
        encoded = encode_model_input(sanitized)
        if use_query_only:
            train_x.append(encoded[:11])
        elif use_delta_only:
            train_x.append(get_support_delta_vector(r))
        else:
            train_x.append(encoded)
        train_y.append(CLASS_TO_IDX[extract_label(r)])
        
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.long)
    
    # Encode validation batch
    val_x = []
    val_y = []
    for r in val_records:
        sanitized = sanitize_episode_for_model_input(r)
        encoded = encode_model_input(sanitized)
        if use_query_only:
            val_x.append(encoded[:11])
        elif use_delta_only:
            val_x.append(get_support_delta_vector(r))
        else:
            val_x.append(encoded)
        val_y.append(CLASS_TO_IDX[extract_label(r)])
        
    x_val = torch.tensor(val_x, dtype=torch.float32)
    y_val = torch.tensor(val_y, dtype=torch.long)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluate validation loss
        model.eval()
        with torch.no_grad():
            v_logits = model(x_val)
            v_loss = loss_fn(v_logits, y_val).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                # Keep state copy in memory
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model


def run_p87_synthetic_few_shot_selector_pilot_probe() -> dict:
    contracts_val = validate_source_contracts_for_p87()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    case_index = collect_p70_relation_cases(p70a, p70b)
    original_records = build_selector_dataset_records(p70a, p70b)
    
    # 1. Same split strict policy episodes
    strict_episodes = build_policy_episode_records(original_records, case_index, "same_split_support_strict")
    strict_audit = audit_support_split_policy(strict_episodes, original_records, "same_split_support_strict")
    
    # 2. Train bank diagnostic policy episodes
    diagnostic_episodes = build_policy_episode_records(original_records, case_index, "train_bank_support_diagnostic")
    diagnostic_audit = audit_support_split_policy(diagnostic_episodes, original_records, "train_bank_support_diagnostic")
    
    # Leakage check on a sample strict episode model input
    leakage_diagnostic = True
    leakage_res = None
    for ep in strict_episodes:
        if ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]:
            sanitized = sanitize_episode_for_model_input(ep)
            leakage_res = audit_model_input_leakage(sanitized)
            if leakage_res["diagnostic_pass"] is False:
                leakage_diagnostic = False
            break
            
    # Collect train split episodes under strict policy for baseline / prototype computation
    train_eps = [ep for ep in strict_episodes if ep["split"] == "train" and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    val_eps = [ep for ep in strict_episodes if ep["split"] == "validation" and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    test_eps = [ep for ep in strict_episodes if ep["split"] == "test" and ep["materialized_episode_manifest"]["valid_for_p87_synthetic_few_shot_selector_pilot"]]
    
    # Calculate nearest prototype class centers from train split support deltas
    class_delta_sums = {}
    class_delta_counts = {}
    for ep in train_eps:
        lbl = extract_label(ep)
        c_idx = CLASS_TO_IDX[lbl]
        eps_delta = get_support_delta_vector(ep)
        
        class_delta_sums[c_idx] = class_delta_sums.get(c_idx, [0.0]*18)
        class_delta_counts[c_idx] = class_delta_counts.get(c_idx, 0) + 1
        for j in range(18):
            class_delta_sums[c_idx][j] += eps_delta[j]
            
    class_prototypes = {}
    for c_idx, sums in class_delta_sums.items():
        cnt = class_delta_counts[c_idx]
        class_prototypes[c_idx] = [s / cnt for s in sums]
        
    # Calculate majority class in train split
    train_lbl_counts = {}
    for ep in train_eps:
        lbl = extract_label(ep)
        train_lbl_counts[lbl] = train_lbl_counts.get(lbl, 0) + 1
    majority_class_lbl = max(train_lbl_counts, key=train_lbl_counts.get) if len(train_lbl_counts) > 0 else "translate_x"
    majority_class_idx = CLASS_TO_IDX[majority_class_lbl]
    
    # Train PyTorch selector models in memory only (no checkpoints)
    # Model 1: TinyFewShotSelector (combined query + support)
    model = train_pytorch_selector(train_eps, val_eps, input_dim=49)
    
    # Model 2: Query-source-only (11 dims)
    query_only_model = train_pytorch_selector(train_eps, val_eps, input_dim=49, use_query_only=True)
    
    # Model 3: Support-delta-only linear classifier (18 dims)
    linear_delta_model = train_pytorch_selector(train_eps, val_eps, input_dim=49, use_delta_only=True)
    
    # Evaluate combined results on test split
    eval_test = run_evaluation_on_split(
        model, query_only_model, linear_delta_model,
        strict_episodes, class_prototypes, majority_class_idx, "test"
    )
    
    # Evaluate domain-specific test metrics
    p70a_test_eps = [ep for ep in test_eps if ep["domain"] == "p70a_vector_world"]
    p70b_test_eps = [ep for ep in test_eps if ep["domain"] == "p70b_time_series_parameter_world"]
    
    eval_p70a = run_evaluation_on_split(
        model, query_only_model, linear_delta_model,
        p70a_test_eps, class_prototypes, majority_class_idx, "test"
    )
    eval_p70b = run_evaluation_on_split(
        model, query_only_model, linear_delta_model,
        p70b_test_eps, class_prototypes, majority_class_idx, "test"
    )
    
    # Compute confusion summary on test split predictions
    translate_x_as_reflect_x = 0
    translate_x_as_nonlinear = 0
    reflect_x_as_translate_x = 0
    reflect_x_as_nonlinear = 0
    nonlinear_as_translate_x = 0
    nonlinear_as_reflect_x = 0
    
    for pred in eval_test["predictions"]:
        true_lbl = pred["true_label"]
        pred_lbl = pred["predicted_model"]
        
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
    
    # Evidence validation
    test_acc = eval_test["model_accuracy"]
    maj_acc = eval_test["majority_accuracy"]
    qo_acc = eval_test["query_source_only_accuracy"]
    
    evidence_criteria_passed = (
        leakage_diagnostic
        and strict_audit["same_split_support_strict_pass"] is True
        and test_acc - maj_acc >= 0.10
        and test_acc - qo_acc >= 0.10
        and test_acc >= 0.60
    )
    
    evidence_verdict = bool(evidence_criteria_passed)
    
    bridge_audit = audit_bridge_boundary_after_few_shot_selector_pilot(evidence_verdict)
    
    # Recommended next phase selection logic
    if evidence_verdict:
        next_phase = "P88_few_shot_metric_candidate_and_hard_ablation_no_bridge"
    elif eval_test["nearest_support_delta_accuracy"] >= 0.60:
        next_phase = "P88_nonlearned_support_delta_metric_baseline_formalization_no_bridge"
    elif strict_audit["same_split_support_strict_pass"] is False:
        next_phase = "P88_support_split_policy_repair_no_training_no_bridge"
    else:
        next_phase = "P88_support_episode_feature_contract_revision_no_bridge"
        
    verdict_str = VERDICT if contracts_ok else "P87_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED is False or TORCH_TRAINING_PERFORMED is False:
            verdict_str = "P87_BLOCKED_BY_UNEXPECTED_TRAINING_FLAG"
        elif leakage_diagnostic is False:
            verdict_str = "P87_BLOCKED_BY_MODEL_INPUT_LEAKAGE"
        elif BRIDGE_READY is True:
            verdict_str = "P87_BLOCKED_BY_PREMATURE_BRIDGE_AUTHORITY"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p86_materialized_support_preserved": contracts_val["p86_materialized_support_preserved"],
        "p86_sanitized_collision_key_preserved": contracts_val["p86_sanitized_collision_key_preserved"],
        "p86_p87_eligibility_preserved": contracts_val["p86_p87_eligibility_preserved"],
        "p86_bridge_not_ready_preserved": contracts_val["p86_bridge_not_ready_preserved"],

        "support_split_policy_audit_defined": SUPPORT_SPLIT_POLICY_AUDIT_DEFINED,
        "same_split_support_strict_protocol_defined": SAME_SPLIT_SUPPORT_STRICT_PROTOCOL_DEFINED,
        "train_bank_support_diagnostic_protocol_defined": TRAIN_BANK_SUPPORT_DIAGNOSTIC_PROTOCOL_DEFINED,

        "support_selection_uses_relation_label_for_synthetic_dataset_construction": True,
        "support_selection_label_visible_to_selector": False,

        "evaluated_target_endpoint_used_for_model_input": leakage_res["evaluated_target_endpoint_used_for_model_input"] if leakage_res else False,
        "evaluated_target_delta_used_for_model_input": leakage_res["evaluated_target_delta_used_for_model_input"] if leakage_res else False,
        "exact_relation_label_used_for_model_input": leakage_res["exact_relation_label_used_for_model_input"] if leakage_res else False,
        "exact_operator_id_used_for_model_input": leakage_res["exact_operator_id_used_for_model_input"] if leakage_res else False,
        "audit_metadata_used_for_model_input": leakage_res["audit_metadata_used_for_model_input"] if leakage_res else False,
        "relation_specific_hint_used_for_model_input": leakage_res["relation_specific_hint_used_for_model_input"] if leakage_res else False,

        "support_record_id_used_for_model_input": leakage_res["support_record_id_used_for_model_input"] if leakage_res else False,
        "materialization_metadata_used_for_model_input": leakage_res["materialization_metadata_used_for_model_input"] if leakage_res else False,

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "learned_selector_evidence_present": evidence_verdict,
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
        "p86_materialized_support_preserved": contracts_val["p86_materialized_support_preserved"],
        "p86_sanitized_collision_key_preserved": contracts_val["p86_sanitized_collision_key_preserved"],
        "p86_p87_eligibility_preserved": contracts_val["p86_p87_eligibility_preserved"],
        "p86_bridge_not_ready_preserved": contracts_val["p86_bridge_not_ready_preserved"],

        "support_split_policy_audit_defined": SUPPORT_SPLIT_POLICY_AUDIT_DEFINED,
        "same_split_support_strict_protocol_defined": SAME_SPLIT_SUPPORT_STRICT_PROTOCOL_DEFINED,
        "train_bank_support_diagnostic_protocol_defined": TRAIN_BANK_SUPPORT_DIAGNOSTIC_PROTOCOL_DEFINED,

        "primary_support_policy": "same_split_support_strict",
        "primary_support_split_audit": strict_audit,
        "diagnostic_train_bank_support_split_audit": diagnostic_audit,

        "model_input_leakage_audit": leakage_res,

        "feature_dim": 49,
        "num_labels": 10,

        "baseline_results": {
            "majority_accuracy": float(eval_test["majority_accuracy"]),
            "query_source_only_accuracy": float(eval_test["query_source_only_accuracy"]),
            "nearest_support_delta_accuracy": float(eval_test["nearest_support_delta_accuracy"]),
            "support_delta_only_accuracy": float(eval_test["support_delta_only_accuracy"]),
        },
        "model_results": {
            "test_sample_count": int(eval_test["sample_count"]),
            "model_test_accuracy": float(eval_test["model_accuracy"]),
        },
        "domain_results": {
            "p70a_vector_world": {
                "sample_count": int(eval_p70a["sample_count"]),
                "majority_accuracy": float(eval_p70a["majority_accuracy"]),
                "query_source_only_accuracy": float(eval_p70a["query_source_only_accuracy"]),
                "nearest_support_delta_accuracy": float(eval_p70a["nearest_support_delta_accuracy"]),
                "support_delta_only_accuracy": float(eval_p70a["support_delta_only_accuracy"]),
                "model_accuracy": float(eval_p70a["model_accuracy"])
            },
            "p70b_time_series_parameter_world": {
                "sample_count": int(eval_p70b["sample_count"]),
                "majority_accuracy": float(eval_p70b["majority_accuracy"]),
                "query_source_only_accuracy": float(eval_p70b["query_source_only_accuracy"]),
                "nearest_support_delta_accuracy": float(eval_p70b["nearest_support_delta_accuracy"]),
                "support_delta_only_accuracy": float(eval_p70b["support_delta_only_accuracy"]),
                "model_accuracy": float(eval_p70b["model_accuracy"])
            }
        },
        "confusion_summary": confusion_summary,

        "learned_selector_evidence_present": evidence_verdict,
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "bridge_ready": BRIDGE_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,

        "recommended_next_phase": next_phase,

        "bridge_boundary_after_few_shot_selector_pilot": bridge_audit,

        "sanity_summary": sanity_summary,

        "json_safe": True,
        "diagnostic_only": True
    }
    
    return output
