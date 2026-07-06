# src/phase4/external_task_context_enrichment.py

import json
import math
import hashlib
from typing import Any

from src.phase4.external_support_context_builder import (
    build_external_demonstration_records,
    build_external_query_records,
    audit_metric_shape_compatibility,
    encode_external_support_delta_context,
    build_materialized_support_invariant_context,
    build_external_support_context_for_query,
)

from src.phase4.external_context_metric_evaluation import (
    build_class_prototypes_from_contexts,
    predict_nearest_prototype,
    ALL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)

from src.phase4.query_observation_enrichment import (
    build_enriched_query_records,
    build_enriched_support_metadata_records,
    enriched_source_signature_distance,
    audit_query_enrichment_leakage,
    audit_enrichment_proxy_leakage,
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

PHASE = "P94"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "External Task Context or Support Demonstration Enrichment"
CONTRACT_VERSION = "phase4_p94_external_task_context_enrichment_v1"

SOURCE_QUERY_OBSERVATION_ENRICHMENT_PHASE = "P93"
SOURCE_RETRIEVAL_IMPROVEMENT_PHASE = "P92"
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

EXTERNAL_TASK_CONTEXT_ENRICHMENT_DEFINED = True
SUPPORT_DEMONSTRATION_CONTEXT_DEFINED = True
TASK_CONTEXT_LEAKAGE_AUDIT_DEFINED = True
TASK_CONTEXT_PROXY_AUDIT_DEFINED = True

NONLEARNED_METRIC_SIGNAL_PRESENT = True
HARD_GENERALIZATION_SUPPORTED = True
EXTERNAL_CONTEXT_BUILDER_READY = True
NON_LABEL_SELECTED_SUPPORT_READY = True
RETRIEVAL_IMPROVEMENT_SIGNAL_PRESENT = False
QUERY_OBSERVATION_ENRICHMENT_READY = True
ENRICHED_RETRIEVAL_SIGNAL_PRESENT = False
ENRICHED_EXTERNAL_CONTEXT_METRIC_SUPPORTED = False

VALID_EXTERNAL_TASK_CONTEXT_AVAILABLE = False  # computed
SUPPORT_DEMONSTRATION_CONTEXT_READY = False  # computed
TASK_CONTEXT_SIGNAL_PRESENT = False  # computed
TASK_CONTEXT_EXTERNAL_METRIC_SUPPORTED = False  # computed
EXTERNAL_TASK_CONTEXT_REQUIRED = True

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "external_task_context_or_support_demonstration_enrichment"
VERDICT = "P94_READY_FOR_REVIEW"

ALL_P94_POLICIES = [
    "p93_enriched_source_signature_baseline",
    "task_context_absent_baseline",
    "diagnostic_oracle_task_context_retrieval",
    "diagnostic_support_delta_oracle_retrieval",
    "non_label_external_task_manifest_retrieval",
    "task_manifest_plus_source_similarity_retrieval",
    "support_demo_context_without_query_task_card",
    "support_demo_context_with_non_label_task_manifest",
]


# ── Static accepted repaired P93 metadata ─────────────────────────────────────

ACCEPTED_P93_REPORT_METADATA = {
    "phase": "P93",
    "verdict": "P93_READY_FOR_REVIEW",

    "source_contracts_validated": True,

    "p92_source_similarity_retrieval_baseline_effective_accuracy": 0.3888888888888889,
    "p92_source_similarity_with_split_relaxation_baseline_effective_accuracy": 0.3888888888888889,

    "enriched_source_signature_retrieval_effective_accuracy": 0.3888888888888889,
    "enriched_source_signature_with_split_relaxation_effective_accuracy": 0.3888888888888889,
    "context_bucket_retrieval_effective_accuracy": 0.3333333333333333,
    "hybrid_enriched_signature_bucket_retrieval_effective_accuracy": 0.3888888888888889,
    "cross_domain_normalized_signature_retrieval_effective_accuracy": 0.2222222222222222,

    "best_enriched_effective_accuracy": 0.3888888888888889,
    "best_enriched_policy": "enriched_source_signature_retrieval",

    "query_observation_enrichment_ready": True,
    "enriched_retrieval_signal_present": False,
    "enriched_external_context_metric_supported": False,

    "proxy_leakage_risk_present": True,
    "enrichment_leakage_clean": True,
    "metric_shape_compatibility_pass": True,

    "model_training_performed": False,
    "torch_training_performed": False,
    "new_model_implemented": False,
    "optimizer_created": False,
    "checkpoint_written": False,

    "learned_selector_evidence_present": False,
    "learned_metric_evidence_present": False,
    "semantic_metric_ready": False,
    "bridge_ready": False,

    "recommended_next_phase": "P94_external_task_context_or_support_demonstration_enrichment_no_training_no_bridge",
    "metadata_source": "accepted_repaired_p93_report_static_metadata",
}


# ── Source validation ──────────────────────────────────────────────────────────

def validate_source_contracts_for_p94() -> dict:
    validated = True
    missing = []
    m = ACCEPTED_P93_REPORT_METADATA

    if m.get("phase") != "P93" or m.get("verdict") != "P93_READY_FOR_REVIEW":
        validated = False
        missing.append("p93_invalid_phase_or_verdict")
    if m.get("source_contracts_validated") is not True:
        validated = False
        missing.append("p93_source_unvalidated")
    if m.get("query_observation_enrichment_ready") is not True:
        validated = False
        missing.append("p93_enrichment_not_ready")
    if m.get("enriched_retrieval_signal_present") is not False:
        validated = False
        missing.append("p93_enriched_signal_should_be_false")
    if m.get("enriched_external_context_metric_supported") is not False:
        validated = False
        missing.append("p93_external_metric_should_be_false")
    if m.get("model_training_performed") is not False:
        validated = False
        missing.append("p93_unexpected_training")
    if m.get("bridge_ready") is not False:
        validated = False
        missing.append("p93_bridge_ready")
    if m.get("recommended_next_phase") != "P94_external_task_context_or_support_demonstration_enrichment_no_training_no_bridge":
        validated = False
        missing.append("p93_recommended_phase_mismatch")

    # Repaired baseline: 38.89%
    base_ok = (m.get("p92_source_similarity_retrieval_baseline_effective_accuracy", 0.0) > 0.35)
    ceiling_ok = (m.get("best_enriched_effective_accuracy", 1.0) < 0.40)

    return {
        "source_contracts_validated": validated,
        "p93_validated": validated,
        "p93_repaired_baseline_preserved": base_ok,
        "p93_source_side_ceiling_preserved": ceiling_ok,
        "p93_enrichment_ready_preserved": m.get("query_observation_enrichment_ready") is True,
        "p93_enriched_signal_absent_preserved": m.get("enriched_retrieval_signal_present") is False,
        "p93_external_metric_not_supported_preserved": m.get("enriched_external_context_metric_supported") is False,
        "p93_proxy_audit_preserved": m.get("proxy_leakage_risk_present") is True,
        "p93_no_training_preserved": m.get("model_training_performed") is False,
        "p93_bridge_not_ready_preserved": m.get("bridge_ready") is False,
        "p93_recommended_p94_preserved": m.get("recommended_next_phase") == "P94_external_task_context_or_support_demonstration_enrichment_no_training_no_bridge",
        "metadata_source": "accepted_repaired_p93_report_static_metadata",
        "missing_or_invalid": missing,
    }


# ── Context builders ──────────────────────────────────────────────────────────

def build_external_task_context_records(enriched_query_records: list[dict]) -> list[dict]:
    records = []
    for q in enriched_query_records:
        q_id = q["external_query_id"]
        rec_id = q["dataset_record_id"]
        dom = q["domain"]
        sp = q["split"]
        lbl = q["audit_label_evaluation_only"]["target_relation_label"]

        # 1. Non-label context (independent of query labels)
        task_context = {
            "task_context_available": False,
            "context_source": "none_available_in_dataset",
            "public_task_card_id": f"card_none_{rec_id}",
            "public_task_bucket": f"bucket_none_{dom}",
            "public_task_descriptor": {
                "source_side_fingerprint": q["enriched_query_observation"]["observable_context_bucket"],
            }
        }

        task_context_audit = {
            "uses_hidden_relation_label": False,
            "uses_operator_id": False,
            "uses_query_target": False,
            "uses_query_delta": False,
            "uses_query_result": False,
            "uses_hidden_relation_family": False,
            "valid_for_non_label_retrieval": True,
            "diagnostic_only": False,
            "valid_for_final_semantic_geometry_evidence": False,
            "valid_for_bridge_evidence": False,
        }

        # 2. Diagnostic oracle context
        diag_task_context = {
            "task_context_available": True,
            "context_source": "diagnostic_oracle_relation_label",
            "public_task_card_id": f"card_oracle_{lbl}",
            "public_task_bucket": f"bucket_oracle_{lbl}",
            "public_task_descriptor": {
                "relation_label": lbl,
                "operator_id": q["audit_label_evaluation_only"]["operator_id"],
            }
        }

        diag_task_context_audit = {
            "uses_hidden_relation_label": True,
            "uses_operator_id": True,
            "uses_query_target": False,
            "uses_query_delta": False,
            "uses_query_result": False,
            "uses_hidden_relation_family": True,
            "valid_for_non_label_retrieval": False,
            "diagnostic_only": True,
            "valid_for_final_semantic_geometry_evidence": False,
            "valid_for_bridge_evidence": False,
        }

        records.append({
            "external_task_context_id": f"tc_{q_id}",
            "enriched_query_id": q["enriched_query_id"],
            "external_query_id": q_id,
            "dataset_record_id": rec_id,
            "domain": dom,
            "split": sp,
            "task_context": task_context,
            "task_context_audit": task_context_audit,
            # For separating diagnostic from non-label task manifest
            "diagnostic_task_context": diag_task_context,
            "diagnostic_task_context_audit": diag_task_context_audit,
            "audit_label_evaluation_only": q["audit_label_evaluation_only"],
        })

    return records


def build_support_demonstration_context_records(demo_records: list[dict]) -> list[dict]:
    records = []
    for d in demo_records:
        dom = d["domain"]
        sp = d["source_split"]
        obs = d["observable_context"]

        inv_sum = obs["invariant_summary"]
        del_sum = obs["delta_summary"]

        descriptor_str = json.dumps(del_sum, sort_keys=True)
        desc_hash = hashlib.sha256(descriptor_str.encode("utf-8")).hexdigest()[:8]

        sdc = {
            "support_source_descriptor": obs["source_summary"],
            "support_result_descriptor": obs["result_summary"],
            "support_delta_descriptor": del_sum,
            "support_invariant_descriptor": inv_sum,
            "support_demo_descriptor_hash": desc_hash,
        }

        audit = {
            "uses_hidden_relation_label": False,
            "uses_operator_id": False,
            "uses_support_result_content": True,
            "uses_support_delta_content": True,
            "uses_support_delta_shape_metadata": True,
            "valid_as_demonstration_context": True,
            "valid_for_query_retrieval_scoring_without_external_task_card": False,
            "valid_for_final_semantic_geometry_evidence": False,
            "valid_for_bridge_evidence": False,
        }

        records.append({
            "support_demo_context_id": f"sdc_{d['external_demo_id']}",
            "external_demo_id": d["external_demo_id"],
            "domain": dom,
            "source_split": sp,
            "support_demonstration_context": sdc,
            "support_demo_context_audit": audit,
            "audit_label_evaluation_only": d["audit_label_evaluation_only"],
        })

    return records


# ── Support context builder ───────────────────────────────────────────────────

def build_task_context_support_context_for_query(
    enriched_query_record: dict,
    task_context_record: dict,
    demo_records: list[dict],
    support_demo_context_records: list[dict],
    policy: str,
    max_support: int = 2,
) -> dict:
    q_id = enriched_query_record["external_query_id"]
    q_rec_id = enriched_query_record["dataset_record_id"]
    dom = enriched_query_record["domain"]
    sp = enriched_query_record["split"]

    # Separated descriptors
    task_context_used = {}
    support_demo_context_used = []

    # Audit flags
    uses_lbl = False
    uses_op = False
    uses_ext_tc = False
    uses_non_lbl_tm = False
    uses_sdc = False
    uses_s_res = False
    uses_s_del = False
    uses_s_dshape = False
    non_lbl_sel = True
    valid_ext = True
    diag_only = False

    # Build enriched support metadata map for sorting by enriched signature distance
    enriched_support_recs = build_enriched_support_metadata_records(demo_records)
    id_to_enriched_meta = {r["external_demo_id"]: r for r in enriched_support_recs}

    selected = []

    if policy in ("p93_enriched_source_signature_baseline", "task_context_absent_baseline"):
        # Reconstruct P93 signature retrieval (same domain/split, sorted by signature distance)
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (enriched_source_signature_distance(enriched_query_record, id_to_enriched_meta[d["external_demo_id"]]), d["external_demo_id"]),
        )
        selected = candidates[:max_support]

    elif policy == "diagnostic_oracle_task_context_retrieval":
        uses_lbl = True
        uses_op = True
        uses_ext_tc = True
        non_lbl_sel = False
        valid_ext = False
        diag_only = True

        task_context_used = task_context_record["diagnostic_task_context"]
        lbl = enriched_query_record["audit_label_evaluation_only"]["target_relation_label"]
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
            and d["audit_label_evaluation_only"]["target_relation_label"] == lbl
        ]
        selected = sorted(candidates, key=lambda d: d["external_demo_id"])[:max_support]

    elif policy == "diagnostic_support_delta_oracle_retrieval":
        uses_lbl = True
        uses_ext_tc = True
        uses_sdc = True
        uses_s_del = True
        non_lbl_sel = False
        valid_ext = False
        diag_only = True

        lbl = enriched_query_record["audit_label_evaluation_only"]["target_relation_label"]
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
            and d["audit_label_evaluation_only"]["target_relation_label"] == lbl
        ]
        selected = sorted(candidates, key=lambda d: d["external_demo_id"])[:max_support]

    elif policy == "non_label_external_task_manifest_retrieval":
        uses_ext_tc = True
        uses_non_lbl_tm = True
        task_context_used = task_context_record["task_context"]
        # Since no real context exists, select by public card id distance or dummy matching
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        selected = sorted(candidates, key=lambda d: d["external_demo_id"])[:max_support]

    elif policy == "task_manifest_plus_source_similarity_retrieval":
        uses_ext_tc = True
        uses_non_lbl_tm = True
        task_context_used = task_context_record["task_context"]
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        selected = sorted(
            candidates,
            key=lambda d: (enriched_source_signature_distance(enriched_query_record, id_to_enriched_meta[d["external_demo_id"]]), d["external_demo_id"]),
        )
        selected = selected[:max_support]

    elif policy == "support_demo_context_without_query_task_card":
        uses_sdc = True
        uses_s_res = True
        uses_s_del = True
        uses_s_dshape = True
        valid_ext = False
        diag_only = True
        # Selection without task card falls back to manifest
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        selected = sorted(candidates, key=lambda d: d["external_demo_id"])[:max_support]

    elif policy == "support_demo_context_with_non_label_task_manifest":
        uses_ext_tc = True
        uses_non_lbl_tm = True
        uses_sdc = True
        uses_s_res = True
        uses_s_del = True
        uses_s_dshape = True
        task_context_used = task_context_record["task_context"]
        valid_ext = False
        diag_only = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        selected = sorted(candidates, key=lambda d: d["external_demo_id"])[:max_support]

    selected_ids = [d["external_demo_id"] for d in selected]

    # Populate support pairs
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
            "materialization_source_phase": "P94",
            "materialization_method": f"external_task_context_enrichment_{policy}",
            "missing_result_summary": False,
            "missing_delta_summary": False,
        })

    inv = build_materialized_support_invariant_context(support_pairs)

    # Populate support_demo_context_used list
    for sid in selected_ids:
        sdc_rec = [r for r in support_demo_context_records if r["external_demo_id"] == sid][0]
        support_demo_context_used.append(sdc_rec["support_demonstration_context"])

    audit = {
        "support_selection_uses_hidden_relation_label": uses_lbl,
        "support_selection_uses_operator_id": uses_op,
        "support_selection_uses_query_target": False,
        "support_selection_uses_query_delta": False,
        "support_selection_uses_query_result": False,
        "support_selection_label_visible_to_selector": False,
        "support_selection_uses_external_task_context": uses_ext_tc,
        "support_selection_uses_non_label_task_manifest": uses_non_lbl_tm,
        "support_selection_uses_support_demo_context": uses_sdc,
        "support_selection_uses_support_result_content": uses_s_res,
        "support_selection_uses_support_delta_content": uses_s_del,
        "support_selection_uses_support_delta_shape_metadata": uses_s_dshape,
        "support_selection_non_label_selected": non_lbl_sel,
        "valid_for_external_metric_context": valid_ext,
        "diagnostic_only": diag_only,
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
        "task_context_used": task_context_used,
        "support_demo_context_used": support_demo_context_used,
        "support_selection_audit": audit,
        "audit_label_evaluation_only": {
            "target_relation_label": enriched_query_record["audit_label_evaluation_only"]["target_relation_label"],
            "operator_id": enriched_query_record["audit_label_evaluation_only"]["operator_id"],
        },
    }


# ── Evaluations ───────────────────────────────────────────────────────────────

def _extract_label(ctx: dict) -> str:
    return ctx["audit_label_evaluation_only"]["target_relation_label"]


def evaluate_task_context_policy_metric(
    enriched_query_records: list[dict],
    task_context_records: list[dict],
    demo_records: list[dict],
    support_demo_context_records: list[dict],
    policy: str,
) -> dict:
    q_split = {q["external_query_id"]: q["split"] for q in enriched_query_records}
    tc_map = {tc["external_query_id"]: tc for tc in task_context_records}

    contexts = [
        build_task_context_support_context_for_query(
            q, tc_map[q["external_query_id"]], demo_records, support_demo_context_records, policy,
        )
        for q in enriched_query_records
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

    is_oracle = policy in ("diagnostic_oracle_task_context_retrieval", "diagnostic_support_delta_oracle_retrieval")
    is_sdc_policy = policy in ("support_demo_context_without_query_task_card", "support_demo_context_with_non_label_task_manifest")
    diag_only = is_oracle or is_sdc_policy
    valid_ev = not diag_only

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
        "diagnostic_only": diag_only,
        "valid_for_external_metric_evidence": valid_ev,
        "valid_for_final_semantic_geometry_evidence": False,
        "predictions": predictions,
        "contexts": contexts,
    }


# ── Negative controls ──────────────────────────────────────────────────────────

def compute_task_context_policy_negative_controls(
    enriched_query_records: list[dict],
    policy: str,
    normal_eff_acc: float,
    eval_result: dict,
) -> dict:
    q_split = {q["external_query_id"]: q["split"] for q in enriched_query_records}
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


# ── Leakage audit ─────────────────────────────────────────────────────────────

def audit_task_context_retrieval_leakage(contexts: list[dict]) -> dict:
    lbl_used = 0
    op_used = 0
    q_tgt_used = 0
    q_del_used = 0
    q_res_used = 0
    lbl_vis = 0

    ext_tc_used = 0
    non_lbl_tm_used = 0
    sdc_used = 0

    s_res_used = 0
    s_del_used = 0
    s_dshape_used = 0

    diag_count = 0
    valid_ext_count = 0
    non_lbl_sel_count = 0

    for c in contexts:
        sa = c["support_selection_audit"]
        if sa.get("support_selection_uses_hidden_relation_label", False):
            lbl_used += 1
        if sa.get("support_selection_uses_operator_id", False):
            op_used += 1
        if sa.get("support_selection_uses_query_target", False):
            q_tgt_used += 1
        if sa.get("support_selection_uses_query_delta", False):
            q_del_used += 1
        if sa.get("support_selection_uses_query_result", False):
            q_res_used += 1
        if sa.get("support_selection_label_visible_to_selector", False):
            lbl_vis += 1

        if sa.get("support_selection_uses_external_task_context", False):
            ext_tc_used += 1
        if sa.get("support_selection_uses_non_label_task_manifest", False):
            non_lbl_tm_used += 1
        if sa.get("support_selection_uses_support_demo_context", False):
            sdc_used += 1

        if sa.get("support_selection_uses_support_result_content", False):
            s_res_used += 1
        if sa.get("support_selection_uses_support_delta_content", False):
            s_del_used += 1
        if sa.get("support_selection_uses_support_delta_shape_metadata", False):
            s_dshape_used += 1

        if sa.get("diagnostic_only", False):
            diag_count += 1
        if sa.get("valid_for_external_metric_context", False):
            valid_ext_count += 1
        if sa.get("support_selection_non_label_selected", False):
            non_lbl_sel_count += 1

    # Check if this is a diagnostic-only policy or evidence-passing policy
    # If diagnostic_only context is processed, we separate the counts but keep track.
    # The check for evidence policies requires zero use of hidden info.
    diag_pass = True
    # For a specific selection context, if it claims to be non-label selected, it must have 0 hidden leakage.
    for c in contexts:
        sa = c["support_selection_audit"]
        if sa.get("support_selection_non_label_selected", False):
            if (
                sa.get("support_selection_uses_hidden_relation_label", False)
                or sa.get("support_selection_uses_operator_id", False)
                or sa.get("support_selection_uses_query_target", False)
                or sa.get("support_selection_uses_query_delta", False)
                or sa.get("support_selection_uses_query_result", False)
                or sa.get("support_selection_label_visible_to_selector", False)
            ):
                diag_pass = False

    return {
        "context_count": len(contexts),
        "hidden_relation_label_used_count": lbl_used,
        "operator_id_used_count": op_used,
        "query_target_used_count": q_tgt_used,
        "query_delta_used_count": q_del_used,
        "query_result_used_count": q_res_used,
        "label_visible_to_selector_count": lbl_vis,

        "external_task_context_used_count": ext_tc_used,
        "non_label_task_manifest_used_count": non_lbl_tm_used,
        "support_demo_context_used_count": sdc_used,

        "support_result_content_used_for_selection_count": s_res_used,
        "support_delta_content_used_for_selection_count": s_del_used,
        "support_delta_shape_metadata_used_for_selection_count": s_dshape_used,

        "diagnostic_context_count": diag_count,
        "valid_external_metric_context_count": valid_ext_count,
        "non_label_selected_context_count": non_lbl_sel_count,

        "diagnostic_pass": diag_pass,
    }


# ── Proxy leakage audit ───────────────────────────────────────────────────────

def audit_task_context_proxy_leakage(
    task_context_records: list[dict],
    support_demo_context_records: list[dict],
    policy_eval_results: dict,
) -> dict:
    from collections import Counter

    task_results = {}
    support_results = {}
    policy_selection_proxy = {}
    high_risk = []

    # Check task context fields
    tc_fields = [
        ("public_task_card_id", lambda r: r["task_context"]["public_task_card_id"]),
        ("public_task_bucket", lambda r: r["task_context"]["public_task_bucket"]),
    ]

    for field_name, extractor in tc_fields:
        val_to_labels = {}
        for r in task_context_records:
            val = str(extractor(r))
            lbl = r["audit_label_evaluation_only"]["target_relation_label"]
            val_to_labels.setdefault(val, []).append(lbl)

        max_purity = 0.0
        risk = False
        for val, labels in val_to_labels.items():
            if len(labels) < 2:
                continue
            counts = Counter(labels)
            dom = counts.most_common(1)[0][1]
            purity = dom / len(labels)
            if purity > max_purity:
                max_purity = purity
            if purity >= 0.95 and len(labels) >= 2:
                risk = True

        task_results[field_name] = {
            "unique_value_count": len(val_to_labels),
            "max_relation_purity": float(max_purity),
            "relation_proxy_risk": risk,
        }
        if risk:
            high_risk.append(f"task_{field_name}")

    # Check support demo context fields
    sdc_fields = [
        ("support_demo_descriptor_hash", lambda r: r["support_demonstration_context"]["support_demo_descriptor_hash"]),
    ]

    for field_name, extractor in sdc_fields:
        val_to_labels = {}
        for r in support_demo_context_records:
            val = str(extractor(r))
            lbl = r["audit_label_evaluation_only"]["target_relation_label"]
            val_to_labels.setdefault(val, []).append(lbl)

        max_purity = 0.0
        risk = False
        for val, labels in val_to_labels.items():
            if len(labels) < 2:
                continue
            counts = Counter(labels)
            dom = counts.most_common(1)[0][1]
            purity = dom / len(labels)
            if purity > max_purity:
                max_purity = purity
            if purity >= 0.95 and len(labels) >= 2:
                risk = True

        support_results[field_name] = {
            "unique_value_count": len(val_to_labels),
            "max_relation_purity": float(max_purity),
            "relation_proxy_risk": risk,
        }
        if risk:
            high_risk.append(f"support_{field_name}")

    # Check policy selections
    for pol, er in policy_eval_results.items():
        if pol in ("p93_enriched_source_signature_baseline", "task_context_absent_baseline"):
            continue
        contexts = er.get("contexts", [])
        demo_label_map = {}
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
            dom = counts.most_common(1)[0][1]
            purity = dom / len(labels)
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

    proxy_present = len(high_risk) > 0

    return {
        "audit_defined": True,
        "task_context_field_results": task_results,
        "support_demo_context_field_results": support_results,
        "policy_selection_proxy_results": policy_selection_proxy,
        "high_proxy_risk_fields": high_risk,
        "proxy_leakage_risk_present": proxy_present,
        "diagnostic_pass": True,
    }


# ── Bridge boundary ───────────────────────────────────────────────────────────

def audit_bridge_boundary_after_external_task_context_enrichment(
    valid_external_task_context_available: bool,
    support_demonstration_context_ready: bool,
    task_context_signal_present: bool,
    task_context_external_metric_supported: bool,
) -> dict:
    return {
        "bridge_ready": False,
        "bridge_implementation_allowed": False,

        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": True,
        "query_observation_enrichment_ready": True,
        "enriched_retrieval_signal_present": False,

        "valid_external_task_context_available": valid_external_task_context_available,
        "support_demonstration_context_ready": support_demonstration_context_ready,
        "task_context_signal_present": task_context_signal_present,
        "task_context_external_metric_supported": task_context_external_metric_supported,
        "external_task_context_required": True,

        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,

        "blocking_reasons": [
            "learned_metric_evidence_not_present",
            "semantic_metric_not_ready",
            "real_external_task_context_dataset_not_available",
            "external_context_metric_not_supported_for_bridge",
            "bridge_input_contract_not_defined",
            "bridge_validation_not_run",
        ],
        "diagnostic_pass": True,
    }


# ── Public probe ───────────────────────────────────────────────────────────────

def run_p94_external_task_context_enrichment_probe() -> dict:
    # 0. Validate source contracts
    contracts_val = validate_source_contracts_for_p94()
    contracts_ok = contracts_val["source_contracts_validated"]

    # 1. Build records
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    demo_recs = build_external_demonstration_records(p70a, p70b)
    selector_recs = build_selector_dataset_records(p70a, p70b)

    enriched_queries = build_enriched_query_records(selector_recs)
    enriched_support = build_enriched_support_metadata_records(demo_recs)

    task_context_records = build_external_task_context_records(enriched_queries)
    support_demo_context_records = build_support_demonstration_context_records(demo_recs)

    # 2. Evaluate all policies
    policy_metric_results = {}
    policy_negative_controls = {}
    policy_leakage_audits = {}
    policy_shape_audits = {}

    for pol in ALL_P94_POLICIES:
        er = evaluate_task_context_policy_metric(
            enriched_queries, task_context_records, demo_recs, support_demo_context_records, pol,
        )
        policy_metric_results[pol] = er

        nc = compute_task_context_policy_negative_controls(
            enriched_queries, pol, er["effective_accuracy"], er,
        )
        policy_negative_controls[pol] = nc

        contexts = er["contexts"]
        la = audit_task_context_retrieval_leakage(contexts)
        policy_leakage_audits[pol] = la

        sa = audit_metric_shape_compatibility(contexts)
        policy_shape_audits[pol] = sa

    # 3. Proxy leakage audit
    proxy_audit = audit_task_context_proxy_leakage(
        task_context_records, support_demo_context_records, policy_metric_results,
    )

    # 4. Flags construction
    # Since no real valid external task context exists:
    valid_etc_available = False
    support_demo_ready = (
        contracts_ok
        and len(support_demo_context_records) > 0
    )

    # Check for non-diagnostic signal
    task_sig = False
    for pol in ALL_P94_POLICIES:
        mr = policy_metric_results[pol]
        nc = policy_negative_controls[pol]
        la = policy_leakage_audits[pol]
        sa = policy_shape_audits[pol]

        # Ignore diagnostic and invalid evidence policies from evidence calculation
        if not mr["valid_for_external_metric_evidence"]:
            continue

        if (
            mr["effective_accuracy"] >= 0.40
            and mr["coverage"] >= 0.80
            and nc["beats_shuffled_by"] >= 0.10
            and nc["beats_zero_delta_by"] >= 0.10
            and la["diagnostic_pass"]
            and sa["diagnostic_pass"]
        ):
            task_sig = True

    task_metric_supported = (
        task_sig
        and valid_etc_available
    )

    # 5. Next phase logic
    if task_metric_supported:
        next_phase = "P95_external_task_context_metric_hardening_no_training_no_bridge"
    elif not valid_etc_available and support_demo_ready:
        next_phase = "P95_real_external_task_context_dataset_contract_no_training_no_bridge"
    else:
        # Fallbacks
        next_phase = "P95_real_external_task_context_dataset_contract_no_training_no_bridge"

    # 6. Bridge audit
    bridge_audit = audit_bridge_boundary_after_external_task_context_enrichment(
        valid_etc_available, support_demo_ready, task_sig, task_metric_supported,
    )

    # 7. Verdict
    verdict_str = VERDICT if contracts_ok else "P94_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED or TORCH_TRAINING_PERFORMED:
            verdict_str = "P94_BLOCKED_BY_UNEXPECTED_TRAINING_FLAG"

    # 8. Clean output
    clean_metric_results = {}
    clean_predictions = {}
    for pol in ALL_P94_POLICIES:
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

        "source_query_observation_enrichment_phase": SOURCE_QUERY_OBSERVATION_ENRICHMENT_PHASE,
        "source_retrieval_improvement_phase": SOURCE_RETRIEVAL_IMPROVEMENT_PHASE,
        "source_external_context_metric_phase": SOURCE_EXTERNAL_CONTEXT_METRIC_PHASE,
        "source_external_context_builder_phase": SOURCE_EXTERNAL_CONTEXT_BUILDER_PHASE,
        "source_hard_generalization_phase": SOURCE_HARD_GENERALIZATION_PHASE,
        "source_nonlearned_metric_phase": SOURCE_NONLEARNED_METRIC_PHASE,

        "verdict": verdict_str,

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "source_contracts_validated": contracts_ok,

        "external_task_context_record_count": len(task_context_records),
        "support_demonstration_context_record_count": len(support_demo_context_records),

        "task_context_policy_metric_results": clean_metric_results,
        "task_context_policy_predictions": clean_predictions,
        "task_context_policy_negative_controls": policy_negative_controls,
        "task_context_policy_leakage_audits": policy_leakage_audits,
        "task_context_policy_shape_compatibility_audits": policy_shape_audits,
        "task_context_proxy_leakage_audit": proxy_audit,

        "valid_external_task_context_available": valid_etc_available,
        "support_demonstration_context_ready": support_demo_ready,
        "task_context_signal_present": task_sig,
        "task_context_external_metric_supported": task_metric_supported,
        "external_task_context_required": True,

        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "bridge_implementation_allowed": False,
        "bridge_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,

        "recommended_next_phase": next_phase,

        "bridge_boundary_after_external_task_context_enrichment": bridge_audit,

        "json_safe": True,
        "diagnostic_only": True,
    }

    return output
