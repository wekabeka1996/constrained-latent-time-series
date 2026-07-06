# src/phase4/non_label_support_retrieval_improvement.py

import json
import math
import hashlib
from typing import Any

from src.phase4.external_support_context_builder import (
    build_external_demonstration_records,
    build_external_query_records,
    build_external_support_context_for_query,
    audit_external_support_selection_leakage,
    audit_metric_shape_compatibility,
    encode_external_support_delta_context,
    build_materialized_support_invariant_context,
)

from src.phase4.external_context_metric_evaluation import (
    build_class_prototypes_from_contexts,
    predict_nearest_prototype,
    ALL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
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

# ── Constants ──────────────────────────────────────────────────────────────────

PHASE = "P92"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Non-Label-Selected Support Retrieval Improvement"
CONTRACT_VERSION = "phase4_p92_non_label_support_retrieval_improvement_v1"

SOURCE_EXTERNAL_CONTEXT_METRIC_PHASE = "P91"
SOURCE_EXTERNAL_CONTEXT_BUILDER_PHASE = "P90"
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

NON_LABEL_SELECTED_RETRIEVAL_IMPROVEMENT_DEFINED = True
RETRIEVAL_POLICY_COMPARISON_DEFINED = True
RETRIEVAL_LEAKAGE_AUDIT_DEFINED = True
RETRIEVAL_PROXY_AUDIT_DEFINED = True

NONLEARNED_METRIC_SIGNAL_PRESENT = True
HARD_GENERALIZATION_SUPPORTED = True
EXTERNAL_CONTEXT_BUILDER_READY = True
NON_LABEL_SELECTED_SUPPORT_READY = True
NON_LABEL_SELECTED_METRIC_SIGNAL_PRESENT = False
EXTERNAL_CONTEXT_METRIC_SUPPORTED = False

RETRIEVAL_IMPROVEMENT_SIGNAL_PRESENT = False  # computed
IMPROVED_NON_LABEL_SELECTED_SUPPORT_READY = False  # computed

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "non_label_selected_support_retrieval_improvement"
VERDICT = "P92_READY_FOR_REVIEW"

BASELINE_POLICIES = [
    "p91_observable_domain_split_retrieval_baseline",
    "p91_external_manifest_support_baseline",
]

IMPROVED_POLICIES = [
    "source_similarity_retrieval",
    "source_similarity_with_split_relaxation",
    "observable_diversity_manifest",
    "hybrid_source_similarity_diversity",
]

ALL_RETRIEVAL_POLICIES = BASELINE_POLICIES + IMPROVED_POLICIES

# ── Static accepted P91 metadata ──────────────────────────────────────────────

ACCEPTED_P91_REPORT_METADATA = {
    "phase": "P91",
    "verdict": "P91_READY_FOR_REVIEW",

    "source_contracts_validated": True,

    "label_selected_oracle_support_effective_accuracy": 0.6666666666666666,
    "observable_domain_split_retrieval_effective_accuracy": 0.2222222222222222,
    "external_manifest_support_effective_accuracy": 0.2222222222222222,

    "observable_domain_split_retrieval_coverage": 1.0,
    "external_manifest_support_coverage": 1.0,

    "observable_domain_split_retrieval_shuffled_control": 0.1111111111111111,
    "observable_domain_split_retrieval_zero_delta_control": 0.1111111111111111,
    "external_manifest_support_shuffled_control": 0.1111111111111111,
    "external_manifest_support_zero_delta_control": 0.1111111111111111,

    "proxy_leakage_risk_present": False,
    "observable_domain_split_retrieval_leakage_clean": True,
    "external_manifest_support_leakage_clean": True,
    "metric_shape_compatibility_pass": True,

    "non_label_selected_metric_signal_present": False,
    "external_context_metric_supported": False,

    "model_training_performed": False,
    "torch_training_performed": False,
    "new_model_implemented": False,
    "optimizer_created": False,
    "checkpoint_written": False,

    "learned_selector_evidence_present": False,
    "learned_metric_evidence_present": False,
    "semantic_metric_ready": False,
    "bridge_ready": False,

    "recommended_next_phase": "P92_non_label_selected_support_retrieval_improvement_no_training_no_bridge",
    "metadata_source": "accepted_p91_report_static_metadata",
}


# ── Source validation ──────────────────────────────────────────────────────────

def validate_source_contracts_for_p92() -> dict:
    validated = True
    missing = []
    m = ACCEPTED_P91_REPORT_METADATA

    if m.get("phase") != "P91" or m.get("verdict") != "P91_READY_FOR_REVIEW":
        validated = False
        missing.append("p91_invalid_phase_or_verdict")
    if m.get("source_contracts_validated") is not True:
        validated = False
        missing.append("p91_source_unvalidated")
    if m.get("non_label_selected_metric_signal_present") is not False:
        validated = False
        missing.append("p91_non_label_signal_should_be_false")
    if m.get("external_context_metric_supported") is not False:
        validated = False
        missing.append("p91_external_metric_should_be_false")
    if m.get("model_training_performed") is not False:
        validated = False
        missing.append("p91_unexpected_training")
    if m.get("bridge_ready") is not False:
        validated = False
        missing.append("p91_bridge_ready")
    if m.get("recommended_next_phase") != "P92_non_label_selected_support_retrieval_improvement_no_training_no_bridge":
        validated = False
        missing.append("p91_recommended_phase_mismatch")

    oracle_present = (m.get("label_selected_oracle_support_effective_accuracy", 0.0) > 0.40)
    non_label_collapse = (
        m.get("observable_domain_split_retrieval_effective_accuracy", 1.0) < 0.40
        and m.get("external_manifest_support_effective_accuracy", 1.0) < 0.40
    )
    proxy_ok = m.get("proxy_leakage_risk_present") is False

    return {
        "source_contracts_validated": validated,
        "p91_validated": validated,
        "p91_oracle_signal_preserved": oracle_present,
        "p91_non_label_collapse_preserved": non_label_collapse,
        "p91_proxy_audit_preserved": proxy_ok,
        "p91_no_training_preserved": m.get("model_training_performed") is False,
        "p91_bridge_not_ready_preserved": m.get("bridge_ready") is False,
        "p91_recommended_p92_preserved": m.get("recommended_next_phase") == "P92_non_label_selected_support_retrieval_improvement_no_training_no_bridge",
        "metadata_source": "accepted_p91_report_static_metadata",
        "missing_or_invalid": missing,
    }


# ── Source similarity ──────────────────────────────────────────────────────────

def _source_similarity_distance_p70a(query_obs: dict, demo_obs: dict) -> float:
    """Deterministic source similarity for P70A vector world.

    Compares: z_a_dim, z_a_abs_sum, z_a_sign_pattern, query_intensity_hint.
    Does NOT use target label, operator_id, query target, or query delta.
    """
    q_src = query_obs.get("source_state_summary", {})
    d_src = demo_obs.get("source_summary", {})

    q_dim = float(q_src.get("z_a_dim", 3.0))
    d_dim = float(d_src.get("vector_dim", d_src.get("z_a_dim", 3.0)))
    dim_diff = abs(q_dim - d_dim)

    q_abs = float(q_src.get("z_a_abs_sum", 0.0))
    d_abs = float(d_src.get("abs_sum", d_src.get("z_a_abs_sum", 0.0)))
    abs_diff = abs(q_abs - d_abs)

    q_sign = q_src.get("z_a_sign_pattern", [0, 0, 0])
    d_sign = d_src.get("sign_pattern", d_src.get("z_a_sign_pattern", [0, 0, 0]))
    sign_dist = sum(abs(float(a) - float(b)) for a, b in zip(q_sign, d_sign))

    q_int = float(query_obs.get("query_intensity_hint", 0.0))
    d_int_proxy = float(d_src.get("abs_sum", d_src.get("z_a_abs_sum", 0.0)))
    int_diff = abs(q_int - d_int_proxy * 0.5)

    return dim_diff * 10.0 + abs_diff + sign_dist * 2.0 + int_diff


def _source_similarity_distance_p70b(query_obs: dict, demo_obs: dict) -> float:
    """Deterministic source similarity for P70B time-series parameter world.

    Compares: parameter_key_count, params_abs_sum, params_nonzero_key_count, query_intensity_hint.
    Does NOT use target label, operator_id, query target, or query delta.
    """
    q_src = query_obs.get("source_parameter_summary", {})
    d_src = demo_obs.get("source_summary", {})

    q_kc = float(q_src.get("parameter_key_count", 5.0))
    d_kc = float(d_src.get("params_key_count", d_src.get("parameter_key_count", 5.0)))
    kc_diff = abs(q_kc - d_kc)

    q_abs = float(q_src.get("source_abs_sum", 0.0))
    d_abs = float(d_src.get("params_abs_sum", d_src.get("source_abs_sum", 0.0)))
    abs_diff = abs(q_abs - d_abs)

    q_nz = float(q_src.get("source_nonzero_key_count", 0.0))
    d_nz = float(d_src.get("params_nonzero_key_count", d_src.get("source_nonzero_key_count", 0.0)))
    nz_diff = abs(q_nz - d_nz)

    q_int = float(query_obs.get("query_intensity_hint", 0.0))
    d_int_proxy = d_abs * 0.1
    int_diff = abs(q_int - d_int_proxy)

    return kc_diff * 10.0 + abs_diff + nz_diff * 2.0 + int_diff


def source_similarity_distance(query_record: dict, demo_record: dict) -> float:
    """Compute deterministic source similarity between query and demo.

    Uses only observable source features. No labels, operator_id, query target, or query delta.
    """
    q_dom = query_record["domain"]
    d_dom = demo_record["domain"]
    if q_dom != d_dom:
        return float("inf")

    q_obs = query_record["query_observation"]
    d_obs = demo_record["observable_context"]

    if q_dom == "p70a_vector_world":
        return _source_similarity_distance_p70a(q_obs, d_obs)
    else:
        return _source_similarity_distance_p70b(q_obs, d_obs)


# ── Diversity selection ────────────────────────────────────────────────────────

def _diversity_key(demo: dict) -> str:
    """Generate a diversity bucket key from observable metadata."""
    meta = demo["selection_visible_metadata"]
    return f"{meta['observable_source_shape']}|{meta['observable_delta_shape']}|{meta['observable_context_hash']}"


# ── Improved context builder ──────────────────────────────────────────────────

def build_improved_support_context_for_query(
    query_record: dict,
    demonstration_records: list[dict],
    policy: str,
    max_support: int = 2,
) -> dict:
    """Build support context using an improved retrieval policy.

    For baseline policies, delegates to P90's builder.
    For improved policies, implements new retrieval strategies.
    """
    q_id = query_record["external_query_id"]
    q_rec_id = query_record["dataset_record_id"]
    dom = query_record["domain"]
    sp = query_record["split"]

    # Baseline delegation
    if policy == "p91_observable_domain_split_retrieval_baseline":
        return build_external_support_context_for_query(
            query_record, demonstration_records, "observable_domain_split_retrieval", max_support,
        )
    if policy == "p91_external_manifest_support_baseline":
        return build_external_support_context_for_query(
            query_record, demonstration_records, "external_manifest_support", max_support,
        )

    # Common: build candidate pool (exclude self)
    pool = [
        d for d in demonstration_records
        if d["external_demo_id"] != f"demo_{q_rec_id}"
    ]

    uses_delta_shape_meta = False

    if policy == "source_similarity_retrieval":
        # Filter by domain and split
        pool = [d for d in pool if d["domain"] == dom and d["source_split"] == sp]
        # Sort by source similarity
        pool = sorted(pool, key=lambda d: (source_similarity_distance(query_record, d), d["external_demo_id"]))
        selected = pool[:max_support]

    elif policy == "source_similarity_with_split_relaxation":
        # First try same split
        same_split = [d for d in pool if d["domain"] == dom and d["source_split"] == sp]
        same_split = sorted(same_split, key=lambda d: (source_similarity_distance(query_record, d), d["external_demo_id"]))
        if len(same_split) >= max_support:
            selected = same_split[:max_support]
        else:
            # Relax: include other splits from same domain
            other_split = [d for d in pool if d["domain"] == dom and d["source_split"] != sp]
            other_split = sorted(other_split, key=lambda d: (source_similarity_distance(query_record, d), d["external_demo_id"]))
            selected = same_split + other_split
            selected = selected[:max_support]

    elif policy == "observable_diversity_manifest":
        uses_delta_shape_meta = True
        # Filter by domain and split
        pool = [d for d in pool if d["domain"] == dom and d["source_split"] == sp]
        pool = sorted(pool, key=lambda d: d["external_demo_id"])
        # Greedy diverse selection
        selected = []
        seen_keys: set[str] = set()
        for d in pool:
            key = _diversity_key(d)
            if key not in seen_keys:
                selected.append(d)
                seen_keys.add(key)
                if len(selected) >= max_support:
                    break
        # Fill if not enough unique
        if len(selected) < max_support:
            for d in pool:
                if d not in selected:
                    selected.append(d)
                    if len(selected) >= max_support:
                        break

    elif policy == "hybrid_source_similarity_diversity":
        uses_delta_shape_meta = True
        # Filter by domain and split
        pool = [d for d in pool if d["domain"] == dom and d["source_split"] == sp]
        pool = sorted(pool, key=lambda d: (source_similarity_distance(query_record, d), d["external_demo_id"]))
        # Select top-k by similarity, then pick diverse from those
        top_k = pool[:max(max_support * 3, 6)]
        selected = []
        seen_keys: set[str] = set()
        for d in top_k:
            key = _diversity_key(d)
            if key not in seen_keys:
                selected.append(d)
                seen_keys.add(key)
                if len(selected) >= max_support:
                    break
        if len(selected) < max_support:
            for d in top_k:
                if d not in selected:
                    selected.append(d)
                    if len(selected) >= max_support:
                        break
    else:
        selected = []

    selected_ids = [d["external_demo_id"] for d in selected]

    # Build support pairs
    support_pairs = []
    for d in selected:
        obs = d["observable_context"]
        support_pairs.append({
            "support_record_id": d["external_demo_id"],
            "support_domain": d["domain"],
            "support_context_world": obs["context_world"],
            "support_source_summary": obs["source_summary"],
            "support_result_summary": obs["result_summary"],
            "support_delta_summary": obs["delta_summary"],
            "support_pair_observation_available": True,
            "materialization_source_phase": "P92",
            "materialization_method": f"non_label_support_retrieval_improvement_{policy}",
            "missing_result_summary": False,
            "missing_delta_summary": False,
        })

    inv = build_materialized_support_invariant_context(support_pairs)

    audit = {
        "support_selection_uses_hidden_relation_label": False,
        "support_selection_uses_query_target": False,
        "support_selection_uses_query_delta": False,
        "support_selection_uses_operator_id": False,
        "support_selection_label_visible_to_selector": False,
        "support_selection_non_label_selected": True,
        "support_selection_uses_support_delta_shape_metadata": uses_delta_shape_meta,
        "support_selection_uses_support_result_content": False,
        "support_selection_uses_support_delta_content": False,
        "valid_for_p91_external_context_metric_evaluation": True,
        "valid_for_final_semantic_geometry_evidence": False,
        "valid_for_bridge_evidence": False,
    }

    return {
        "external_support_context_id": f"ctx_{policy}_{q_id}",
        "policy": policy,
        "query_id": q_id,
        "support_demo_ids": selected_ids,
        "support_context": {
            "support_pairs": support_pairs,
            "support_invariant_context": inv,
        },
        "support_selection_audit": audit,
        "audit_label_evaluation_only": {
            "target_relation_label": query_record["audit_label_evaluation_only"]["target_relation_label"],
            "operator_id": query_record["audit_label_evaluation_only"]["operator_id"],
        },
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _extract_label(ctx: dict) -> str:
    return ctx["audit_label_evaluation_only"]["target_relation_label"]


def evaluate_retrieval_policy_metric(
    query_records: list[dict],
    demo_records: list[dict],
    policy: str,
) -> dict:
    """Evaluate a retrieval policy using the P88/P89-compatible 18-dim metric."""
    q_split = {q["external_query_id"]: q["split"] for q in query_records}

    contexts = [
        build_improved_support_context_for_query(q, demo_records, policy)
        for q in query_records
    ]

    train_ctx = [c for c in contexts if q_split.get(c["query_id"]) == "train"]
    test_ctx = [c for c in contexts if q_split.get(c["query_id"]) == "test"]

    prototypes = build_class_prototypes_from_contexts(train_ctx)

    predictions = []
    correct = 0
    wrong = 0
    supported = 0
    unsupported = 0

    for ctx in test_ctx:
        true_lbl = _extract_label(ctx)
        true_idx = CLASS_TO_IDX.get(true_lbl, -1)
        vec = encode_external_support_delta_context(ctx)

        if true_idx not in prototypes:
            unsupported += 1
            predictions.append({
                "query_id": ctx["query_id"],
                "true_label": true_lbl,
                "predicted_label": None,
                "unsupported": True,
                "is_correct": False,
            })
            continue

        pred = predict_nearest_prototype(vec, prototypes)
        supported += 1
        is_correct = pred["predicted_class_idx"] == true_idx
        if is_correct:
            correct += 1
        else:
            wrong += 1
        predictions.append({
            "query_id": ctx["query_id"],
            "true_label": true_lbl,
            "predicted_label": pred["predicted_label"],
            "unsupported": False,
            "is_correct": is_correct,
        })

    total_test = len(test_ctx)
    eff_acc = float(correct / total_test) if total_test > 0 else 0.0
    cov = float(supported / total_test) if total_test > 0 else 0.0
    acc_supp = float(correct / supported) if supported > 0 else 0.0

    return {
        "support_policy": policy,
        "sample_count": total_test,
        "train_context_count": len(train_ctx),
        "test_context_count": total_test,
        "supported_test_count": supported,
        "unsupported_test_count": unsupported,
        "accuracy_on_supported": acc_supp,
        "coverage": cov,
        "effective_accuracy": eff_acc,
        "correct_count": correct,
        "wrong_count": wrong,
        "diagnostic_only": False,
        "valid_for_final_semantic_geometry_evidence": False,
        "predictions": predictions,
        "contexts": contexts,
    }


# ── Negative controls ──────────────────────────────────────────────────────────

def compute_retrieval_policy_negative_controls(
    query_records: list[dict],
    demo_records: list[dict],
    policy: str,
    normal_eff_acc: float,
    eval_result: dict,
) -> dict:
    """Compute shuffled and zero-delta negative controls for a retrieval policy."""
    q_split = {q["external_query_id"]: q["split"] for q in query_records}
    contexts = eval_result["contexts"]

    train_ctx = [c for c in contexts if q_split.get(c["query_id"]) == "train"]
    test_ctx = [c for c in contexts if q_split.get(c["query_id"]) == "test"]

    prototypes = build_class_prototypes_from_contexts(train_ctx)

    # Shuffled control
    test_vecs = [encode_external_support_delta_context(c) for c in test_ctx]
    test_labels = [_extract_label(c) for c in test_ctx]
    n = len(test_vecs)
    shift = 7 % n if n > 0 else 0
    shuffled_vecs = test_vecs[shift:] + test_vecs[:shift]

    shuf_correct = 0
    shuf_total = 0
    for i in range(n):
        true_idx = CLASS_TO_IDX.get(test_labels[i], -1)
        if true_idx == -1 or true_idx not in prototypes:
            continue
        shuf_total += 1
        pred = predict_nearest_prototype(shuffled_vecs[i], prototypes)
        if pred["predicted_class_idx"] == true_idx:
            shuf_correct += 1
    shuf_acc = float(shuf_correct / shuf_total) if shuf_total > 0 else 0.0

    # Zero-delta control
    zero_vec = [0.0] * 18
    zero_correct = 0
    zero_total = 0
    for ctx in test_ctx:
        true_idx = CLASS_TO_IDX.get(_extract_label(ctx), -1)
        if true_idx == -1 or true_idx not in prototypes:
            continue
        zero_total += 1
        pred = predict_nearest_prototype(zero_vec, prototypes)
        if pred["predicted_class_idx"] == true_idx:
            zero_correct += 1
    zero_acc = float(zero_correct / zero_total) if zero_total > 0 else 0.0

    return {
        "support_policy": policy,
        "normal_effective_accuracy": normal_eff_acc,
        "shuffled_support_delta_negative_control": shuf_acc,
        "zero_delta_negative_control": zero_acc,
        "beats_shuffled_by": float(normal_eff_acc - shuf_acc),
        "beats_zero_delta_by": float(normal_eff_acc - zero_acc),
        "shuffled_support_delta_control_implemented": True,
        "shuffled_control_target_labels_preserved": True,
        "shuffled_control_support_deltas_permuted": True,
        "target_labels_shifted": False,
        "true_labels_preserved": True,
        "query_source_only_negative_control_applicable": False,
    }


# ── Retrieval leakage audit ───────────────────────────────────────────────────

def audit_retrieval_policy_leakage(contexts: list[dict]) -> dict:
    """Audit a retrieval policy's contexts for hidden information leakage."""
    lbl_used = 0
    q_target_used = 0
    q_delta_used = 0
    op_used = 0
    lbl_visible = 0
    result_content_used = 0
    delta_content_used = 0
    delta_shape_meta_used = 0
    non_lbl_sel = 0
    valid_ext = 0

    for c in contexts:
        audit = c["support_selection_audit"]
        if audit.get("support_selection_uses_hidden_relation_label", False):
            lbl_used += 1
        if audit.get("support_selection_uses_query_target", False):
            q_target_used += 1
        if audit.get("support_selection_uses_query_delta", False):
            q_delta_used += 1
        if audit.get("support_selection_uses_operator_id", False):
            op_used += 1
        if audit.get("support_selection_label_visible_to_selector", False):
            lbl_visible += 1
        if audit.get("support_selection_uses_support_result_content", False):
            result_content_used += 1
        if audit.get("support_selection_uses_support_delta_content", False):
            delta_content_used += 1
        if audit.get("support_selection_uses_support_delta_shape_metadata", False):
            delta_shape_meta_used += 1
        if audit.get("support_selection_non_label_selected", False):
            non_lbl_sel += 1
        if audit.get("valid_for_p91_external_context_metric_evaluation", False):
            valid_ext += 1

    leak_pass = (
        lbl_used == 0
        and q_target_used == 0
        and q_delta_used == 0
        and op_used == 0
        and lbl_visible == 0
        and result_content_used == 0
        and delta_content_used == 0
    )

    return {
        "context_count": len(contexts),
        "hidden_relation_label_used_count": lbl_used,
        "query_target_used_count": q_target_used,
        "query_delta_used_count": q_delta_used,
        "operator_id_used_count": op_used,
        "label_visible_to_selector_count": lbl_visible,
        "support_result_content_used_for_selection_count": result_content_used,
        "support_delta_content_used_for_selection_count": delta_content_used,
        "support_delta_shape_metadata_used_count": delta_shape_meta_used,
        "non_label_selected_context_count": non_lbl_sel,
        "valid_for_external_metric_context_count": valid_ext,
        "diagnostic_pass": leak_pass,
    }


# ── Proxy leakage audit ───────────────────────────────────────────────────────

def audit_retrieval_proxy_leakage(
    demo_records: list[dict],
    policy_eval_results: dict[str, dict],
) -> dict:
    """Audit observable metadata fields and policy selection distributions for proxy leakage."""
    from collections import Counter

    # Standard metadata field proxy check (from P91)
    fields_to_check = [
        ("observable_source_shape", lambda d: d["selection_visible_metadata"]["observable_source_shape"]),
        ("observable_delta_shape", lambda d: d["selection_visible_metadata"]["observable_delta_shape"]),
        ("observable_context_hash", lambda d: d["selection_visible_metadata"]["observable_context_hash"]),
        ("domain_split_source_shape", lambda d: f"{d['domain']}_{d['source_split']}_{d['selection_visible_metadata']['observable_source_shape']}"),
        ("domain_split_delta_shape", lambda d: f"{d['domain']}_{d['source_split']}_{d['selection_visible_metadata']['observable_delta_shape']}"),
    ]

    field_results = {}
    high_risk = []

    for field_name, extractor in fields_to_check:
        val_to_labels: dict[str, list[str]] = {}
        for d in demo_records:
            val = str(extractor(d))
            lbl = d["audit_label_evaluation_only"]["target_relation_label"]
            val_to_labels.setdefault(val, []).append(lbl)

        unique_vals = len(val_to_labels)
        max_purity = 0.0
        risk = False
        for val, labels in val_to_labels.items():
            if len(labels) < 2:
                continue
            counts = Counter(labels)
            dominant_count = counts.most_common(1)[0][1]
            purity = dominant_count / len(labels)
            if purity > max_purity:
                max_purity = purity
            if purity >= 0.95 and len(labels) >= 2:
                risk = True

        field_results[field_name] = {
            "unique_value_count": unique_vals,
            "max_relation_purity": float(max_purity),
            "relation_proxy_risk": risk,
        }
        if risk:
            high_risk.append(field_name)

    # Policy selection distribution proxy check
    policy_selection_proxy = {}
    for pol, er in policy_eval_results.items():
        if pol in BASELINE_POLICIES:
            continue
        contexts = er.get("contexts", [])
        demo_label_map: dict[str, list[str]] = {}
        for ctx in contexts:
            for did in ctx.get("support_demo_ids", []):
                lbl = ctx["audit_label_evaluation_only"]["target_relation_label"]
                demo_label_map.setdefault(did, []).append(lbl)

        pol_max_purity = 0.0
        pol_risk = False
        for did, labels in demo_label_map.items():
            if len(labels) < 2:
                continue
            counts = Counter(labels)
            dominant = counts.most_common(1)[0][1]
            purity = dominant / len(labels)
            if purity > pol_max_purity:
                pol_max_purity = purity
            if purity >= 0.95 and len(labels) >= 2:
                pol_risk = True

        policy_selection_proxy[pol] = {
            "selected_demo_id_count": len(demo_label_map),
            "max_selection_relation_purity": float(pol_max_purity),
            "selection_proxy_risk": pol_risk,
        }
        if pol_risk:
            high_risk.append(f"policy_selection_{pol}")

    # Delta shape distribution proxy per improved policy
    retrieval_delta_shape_proxy = {}
    for pol, er in policy_eval_results.items():
        if pol in BASELINE_POLICIES:
            continue
        contexts = er.get("contexts", [])
        shape_to_labels: dict[str, list[str]] = {}
        for ctx in contexts:
            pairs = ctx.get("support_context", {}).get("support_pairs", [])
            lbl = ctx["audit_label_evaluation_only"]["target_relation_label"]
            for p in pairs:
                dom = p.get("support_domain", "")
                ds = p.get("support_delta_summary", {})
                if dom == "p70a_vector_world":
                    shape = f"nz_{ds.get('delta_nonzero_count', 0)}"
                else:
                    shape = f"nzk_{ds.get('delta_nonzero_key_count', 0)}"
                shape_to_labels.setdefault(shape, []).append(lbl)

        ds_max_purity = 0.0
        ds_risk = False
        for shape, labels in shape_to_labels.items():
            if len(labels) < 2:
                continue
            counts = Counter(labels)
            dominant = counts.most_common(1)[0][1]
            purity = dominant / len(labels)
            if purity > ds_max_purity:
                ds_max_purity = purity
            if purity >= 0.95 and len(labels) >= 2:
                ds_risk = True

        retrieval_delta_shape_proxy[pol] = {
            "delta_shape_unique_count": len(shape_to_labels),
            "max_delta_shape_relation_purity": float(ds_max_purity),
            "delta_shape_proxy_risk": ds_risk,
        }
        if ds_risk:
            high_risk.append(f"retrieval_delta_shape_{pol}")

    proxy_present = len(high_risk) > 0

    return {
        "audit_defined": True,
        "metadata_field_results": field_results,
        "policy_selection_proxy_results": policy_selection_proxy,
        "retrieval_delta_shape_proxy_results": retrieval_delta_shape_proxy,
        "high_proxy_risk_fields": high_risk,
        "proxy_leakage_risk_present": proxy_present,
        "diagnostic_pass": True,
    }


# ── Bridge boundary ───────────────────────────────────────────────────────────

def audit_bridge_boundary_after_retrieval_improvement(
    retrieval_improvement_signal_present: bool,
    improved_non_label_selected_support_ready: bool,
) -> dict:
    return {
        "bridge_ready": False,
        "bridge_implementation_allowed": False,
        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": True,
        "external_context_builder_ready": True,
        "non_label_selected_support_ready": True,
        "external_context_metric_supported": False,
        "retrieval_improvement_signal_present": retrieval_improvement_signal_present,
        "improved_non_label_selected_support_ready": improved_non_label_selected_support_ready,
        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,
        "blocking_reasons": [
            "learned_metric_evidence_not_present",
            "semantic_metric_not_ready",
            "external_context_metric_not_supported",
            "bridge_input_contract_not_defined",
            "bridge_validation_not_run",
        ],
        "diagnostic_pass": True,
    }


# ── Public probe ───────────────────────────────────────────────────────────────

def run_p92_non_label_support_retrieval_improvement_probe() -> dict:
    # 0. Validate source contracts
    contracts_val = validate_source_contracts_for_p92()
    contracts_ok = contracts_val["source_contracts_validated"]

    # 1. Build demonstration & query records
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    demo_recs = build_external_demonstration_records(p70a, p70b)
    selector_recs = build_selector_dataset_records(p70a, p70b)
    query_recs = build_external_query_records(selector_recs)

    # 2. Evaluate all policies
    policy_metric_results = {}
    policy_negative_controls = {}
    policy_leakage_audits = {}
    policy_shape_audits = {}

    for pol in ALL_RETRIEVAL_POLICIES:
        er = evaluate_retrieval_policy_metric(query_recs, demo_recs, pol)
        policy_metric_results[pol] = er

        nc = compute_retrieval_policy_negative_controls(
            query_recs, demo_recs, pol, er["effective_accuracy"], er,
        )
        policy_negative_controls[pol] = nc

        # Leakage audit
        contexts = er["contexts"]
        la = audit_retrieval_policy_leakage(contexts)
        policy_leakage_audits[pol] = la

        # Shape compatibility
        sa = audit_metric_shape_compatibility(contexts)
        policy_shape_audits[pol] = sa

    # 3. Proxy leakage audit
    proxy_audit = audit_retrieval_proxy_leakage(demo_recs, policy_metric_results)

    # 4. Determine improvement signal
    any_signal = False
    all_improved_clean = True
    for pol in IMPROVED_POLICIES:
        mr = policy_metric_results[pol]
        nc = policy_negative_controls[pol]
        la = policy_leakage_audits[pol]
        sa = policy_shape_audits[pol]

        if not la["diagnostic_pass"] or not sa["diagnostic_pass"]:
            all_improved_clean = False
            continue

        if (
            mr["effective_accuracy"] >= 0.40
            and mr["coverage"] >= 0.80
            and nc["beats_shuffled_by"] >= 0.10
            and nc["beats_zero_delta_by"] >= 0.10
            and la["diagnostic_pass"]
            and sa["diagnostic_pass"]
        ):
            any_signal = True

    retrieval_signal = any_signal

    # improved_non_label_selected_support_ready: at least one improved policy
    # builds valid contexts, leakage clean, shape clean, no hidden info
    improved_ready = False
    for pol in IMPROVED_POLICIES:
        la = policy_leakage_audits[pol]
        sa = policy_shape_audits[pol]
        if la["diagnostic_pass"] and sa["diagnostic_pass"]:
            improved_ready = True
            break

    # 5. Recommended next phase
    if retrieval_signal:
        next_phase = "P93_external_context_metric_hardening_after_retrieval_improvement_no_training_no_bridge"
    elif improved_ready and not retrieval_signal:
        next_phase = "P93_query_observation_enrichment_no_training_no_bridge"
    elif not all_improved_clean:
        any_leak_fail = any(not policy_leakage_audits[p]["diagnostic_pass"] for p in IMPROVED_POLICIES)
        any_shape_fail = any(not policy_shape_audits[p]["diagnostic_pass"] for p in IMPROVED_POLICIES)
        if any_leak_fail:
            next_phase = "P93_retrieval_leakage_repair_no_training_no_bridge"
        elif any_shape_fail:
            next_phase = "P93_retrieval_metric_shape_repair_no_training_no_bridge"
        else:
            next_phase = "P93_query_observation_enrichment_no_training_no_bridge"
    else:
        next_phase = "P93_query_observation_enrichment_no_training_no_bridge"

    # 6. Bridge boundary
    bridge_audit = audit_bridge_boundary_after_retrieval_improvement(
        retrieval_signal, improved_ready,
    )

    # 7. Verdict
    verdict_str = VERDICT if contracts_ok else "P92_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED or TORCH_TRAINING_PERFORMED:
            verdict_str = "P92_BLOCKED_BY_UNEXPECTED_TRAINING_FLAG"

    # 8. Build output (strip contexts and predictions for compactness)
    clean_metric_results = {}
    clean_predictions = {}
    for pol in ALL_RETRIEVAL_POLICIES:
        mr = policy_metric_results[pol]
        clean_predictions[pol] = mr["predictions"]
        clean_metric_results[pol] = {
            k: v for k, v in mr.items() if k not in ("predictions", "contexts")
        }

    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,

        "source_external_context_metric_phase": SOURCE_EXTERNAL_CONTEXT_METRIC_PHASE,
        "source_external_context_builder_phase": SOURCE_EXTERNAL_CONTEXT_BUILDER_PHASE,
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

        "retrieval_policy_metric_results": clean_metric_results,
        "retrieval_policy_predictions": clean_predictions,
        "retrieval_policy_negative_controls": policy_negative_controls,
        "retrieval_policy_leakage_audits": policy_leakage_audits,
        "retrieval_policy_shape_compatibility_audits": policy_shape_audits,

        "proxy_leakage_audit": proxy_audit,

        "retrieval_improvement_signal_present": retrieval_signal,
        "improved_non_label_selected_support_ready": improved_ready,

        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": True,
        "external_context_builder_ready": True,
        "non_label_selected_support_ready": True,
        "non_label_selected_metric_signal_present": False,
        "external_context_metric_supported": False,

        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "bridge_implementation_allowed": False,
        "bridge_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,

        "recommended_next_phase": next_phase,

        "bridge_boundary_after_retrieval_improvement": bridge_audit,

        "json_safe": True,
        "diagnostic_only": True,
    }

    return output
