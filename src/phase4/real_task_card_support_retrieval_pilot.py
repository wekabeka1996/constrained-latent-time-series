# src/phase4/real_task_card_support_retrieval_pilot.py

import json
import math
import hashlib
from typing import Any

from src.phase4.real_external_task_context_dataset_contract import (
    validate_source_contracts_for_p95,
    run_p95_real_external_task_context_dataset_contract_probe,
    build_task_card_catalog,
    build_query_task_card_records,
    build_support_demonstration_card_records,
    build_real_external_task_context_dataset_contract,
)

from src.phase4.external_task_context_enrichment import (
    validate_source_contracts_for_p94,
)

from src.phase4.query_observation_enrichment import (
    build_enriched_query_records,
    build_enriched_support_metadata_records,
    enriched_source_signature_distance,
)

from src.phase4.external_support_context_builder import (
    build_external_demonstration_records,
    build_external_query_records,
    encode_external_support_delta_context,
    build_materialized_support_invariant_context,
    audit_metric_shape_compatibility,
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

PHASE = "P96"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Real Task Card Support Retrieval Pilot"
CONTRACT_VERSION = "phase4_p96_real_task_card_support_retrieval_pilot_v1"

SOURCE_REAL_TASK_CONTEXT_CONTRACT_PHASE = "P95"
SOURCE_EXTERNAL_TASK_CONTEXT_PHASE = "P94"
SOURCE_QUERY_OBSERVATION_ENRICHMENT_PHASE = "P93"
SOURCE_RETRIEVAL_IMPROVEMENT_PHASE = "P92"
SOURCE_EXTERNAL_CONTEXT_METRIC_PHASE = "P91"
SOURCE_EXTERNAL_CONTEXT_BUILDER_PHASE = "P90"
SOURCE_HARD_GENERALIZATION_PHASE = "P89"
SOURCE_NONLEARNED_METRIC_PHASE = "P88"

TRAINING_ALLOWED_BY_PHASE4_AUTHORITY = True
MODEL_TRAINING_PERFORMED = False
TORCH_TRAINING_PERFORMED = False
NEW_MODEL_IMPLEMENTED = False
OPTIMIZER_CREATED = False
CHECKPOINT_WRITTEN = False

REAL_TASK_CARD_RETRIEVAL_PILOT_DEFINED = True
TASK_CARD_DISTANCE_DEFINED = True
SUPPORT_DEMO_PUBLIC_VIEW_DISTANCE_DEFINED = True
PILOT_LEAKAGE_AUDIT_DEFINED = True
PILOT_PROXY_AUDIT_DEFINED = True

REAL_EXTERNAL_TASK_DATASET_CONTRACT_READY = True
VALID_EXTERNAL_TASK_CONTEXT_AVAILABLE = False
VALID_FOR_REAL_EXTERNAL_TASK_CONTEXT_EVIDENCE = False
READY_FOR_REAL_DATA_COLLECTION = True

TASK_CARD_RETRIEVAL_PIPELINE_READY = False  # computed
TASK_CARD_RETRIEVAL_SIGNAL_PRESENT = False  # computed
TASK_CARD_EXTERNAL_METRIC_SUPPORTED = False

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "real_task_card_support_retrieval_pilot"
VERDICT = "P96_READY_FOR_REVIEW"

ACCEPTED_P95_REPORT_METADATA = {
    "phase": "P95",
    "verdict": "P95_READY_FOR_REVIEW",

    "real_external_task_dataset_contract_ready": True,
    "task_card_records_built": True,
    "query_task_card_records_built": True,
    "support_demonstration_card_records_built": True,

    "valid_external_task_context_available": False,
    "valid_for_real_external_task_context_evidence": False,
    "ready_for_real_data_collection": True,

    "contract_diagnostic_only": True,
    "assignment_is_diagnostic_label_mapped": True,
    "proxy_risk_present": True,
    "real_data_required_to_remove_assignment_proxy": True,

    "model_training_performed": False,
    "torch_training_performed": False,
    "new_model_implemented": False,
    "optimizer_created": False,
    "checkpoint_written": False,

    "learned_selector_evidence_present": False,
    "learned_metric_evidence_present": False,
    "semantic_metric_ready": False,
    "bridge_ready": False,

    "recommended_next_phase": "P96_real_task_card_support_retrieval_pilot_no_training_no_bridge",
    "metadata_source": "accepted_p95_report_static_metadata",
}


ALL_P96_POLICIES = [
    "p93_enriched_source_signature_baseline",
    "task_card_absent_baseline",
    "diagnostic_task_card_id_exact_match",
    "diagnostic_compatible_label_mapping_oracle",
    "task_card_public_token_overlap_retrieval",
    "task_card_change_preserve_signature_retrieval",
    "support_demo_public_caption_retrieval",
    "hybrid_task_card_plus_source_signature_retrieval",
]


def validate_source_contracts_for_p96() -> dict:
    validated = True
    missing = []
    m = ACCEPTED_P95_REPORT_METADATA

    if m.get("phase") != "P95" or m.get("verdict") != "P95_READY_FOR_REVIEW":
        validated = False
        missing.append("p95_invalid_phase_or_verdict")
    if m.get("real_external_task_dataset_contract_ready") is not True:
        validated = False
        missing.append("p95_contract_not_ready")
    if m.get("task_card_records_built") is not True or m.get("query_task_card_records_built") is not True or m.get("support_demonstration_card_records_built") is not True:
        validated = False
        missing.append("p95_records_not_built")
    if m.get("valid_for_real_external_task_context_evidence") is not False:
        validated = False
        missing.append("p95_valid_real_evidence_should_be_false")
    if m.get("assignment_is_diagnostic_label_mapped") is not True:
        validated = False
        missing.append("p95_not_diagnostic_label_mapped")
    if m.get("proxy_risk_present") is not True:
        validated = False
        missing.append("p95_proxy_risk_should_be_true")
    if m.get("bridge_ready") is not False:
        validated = False
        missing.append("p95_bridge_ready")
    if m.get("recommended_next_phase") != "P96_real_task_card_support_retrieval_pilot_no_training_no_bridge":
        validated = False
        missing.append("p95_recommended_phase_mismatch")

    return {
        "source_contracts_validated": validated,
        "p95_validated": validated,
        "p95_valid_real_evidence_false": m.get("valid_for_real_external_task_context_evidence") is False,
        "p95_proxy_risk_true": m.get("proxy_risk_present") is True,
        "missing_or_invalid": missing,
    }


# ── Distance Helpers ──────────────────────────────────────────────────────────

def tokenize_public_text(text: str) -> list[str]:
    text = text.lower()
    for c in ".,()[]{}:;!?\"'-\\/_":
        text = text.replace(c, " ")
    words = text.split()
    stopwords = {"while", "and", "or", "of", "the", "a", "to", "in", "with", "by", "from", "for", "on", "at", "an", "is", "are"}
    tokens = [w for w in words if w not in stopwords]
    return sorted(list(set(tokens)))


def token_overlap_distance(query_tokens: list[str], support_tokens: list[str]) -> float:
    q_set = set(query_tokens)
    s_set = set(support_tokens)
    union = q_set.union(s_set)
    if not union:
        return 1.0
    return 1.0 - (len(q_set.intersection(s_set)) / len(union))


def preserve_constraint_distance(query_constraints: dict, support_constraints: dict) -> float:
    keys = set(query_constraints.keys()).union(set(support_constraints.keys()))
    if not keys:
        return 0.0
    mismatches = 0
    for k in keys:
        v1 = query_constraints.get(k, False)
        v2 = support_constraints.get(k, False)
        if v1 != v2:
            mismatches += 1
    return float(mismatches / len(keys))


def task_card_public_distance(query_card: dict, support_card: dict, mode: str) -> float:
    # Do not use task_card_id, audit_label_mapping_only, compatible_relation_labels
    if mode == "token_overlap":
        return token_overlap_distance(
            query_card.get("public_descriptor_tokens", []),
            support_card.get("public_descriptor_tokens", []),
        )
    elif mode == "change_preserve_signature":
        d_change = token_overlap_distance(
            tokenize_public_text(query_card.get("change_intent", {}).get("primary_change", "")),
            tokenize_public_text(support_card.get("change_intent", {}).get("primary_change", "")),
        )
        d_axis = token_overlap_distance(
            tokenize_public_text(query_card.get("change_intent", {}).get("change_axis_description", "")),
            tokenize_public_text(support_card.get("change_intent", {}).get("change_axis_description", "")),
        )
        d_pres = preserve_constraint_distance(
            query_card.get("preserve_constraints", {}),
            support_card.get("preserve_constraints", {}),
        )
        return 0.4 * d_change + 0.4 * d_axis + 0.2 * d_pres
    elif mode == "hybrid_public_card":
        d_overlap = token_overlap_distance(
            query_card.get("public_descriptor_tokens", []),
            support_card.get("public_descriptor_tokens", []),
        )
        d_sig = task_card_public_distance(query_card, support_card, "change_preserve_signature")
        return 0.5 * d_overlap + 0.5 * d_sig
    else:
        return 1.0


def support_demo_public_view_distance(query_card: dict, support_demo_card: dict) -> float:
    # Do not use target_relation_label, operator_id, query target, query delta, external_demo_id as semantic score
    q_tokens = query_card.get("public_descriptor_tokens", [])
    caption_tokens = tokenize_public_text(support_demo_card.get("support_demo_public_view", {}).get("human_demo_caption", ""))
    d_caption = token_overlap_distance(q_tokens, caption_tokens)

    s_tokens = support_demo_card.get("support_task_card", {}).get("public_descriptor_tokens", [])
    d_card = token_overlap_distance(q_tokens, s_tokens)

    return 0.5 * d_caption + 0.5 * d_card


# ── Support Context Builder ───────────────────────────────────────────────────

def build_real_task_card_support_context_for_query(
    enriched_query_record: dict,
    query_task_card_record: dict,
    demo_records: list[dict],
    support_demo_card_records: list[dict],
    enriched_support_metadata_records: list[dict],
    policy: str,
    max_support: int = 2,
) -> dict:
    q_id = enriched_query_record["external_query_id"]
    q_rec_id = enriched_query_record["dataset_record_id"]
    dom = enriched_query_record["domain"]
    sp = enriched_query_record["split"]

    q_card = query_task_card_record["query_task_card"]

    uses_lbl = False
    uses_tc_id = False
    uses_pub_tok = False
    uses_chg_pres = False
    uses_demo_pub = False
    uses_src_sig = False
    uses_lbl_map = False

    diag_only = False
    val_real = False
    val_pilot = False

    selected = []

    id_to_enriched_meta = {r["external_demo_id"]: r for r in enriched_support_metadata_records}
    id_to_demo_card = {r["external_demo_id"]: r for r in support_demo_card_records}

    if policy in ("p93_enriched_source_signature_baseline", "task_card_absent_baseline"):
        uses_src_sig = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (enriched_source_signature_distance(enriched_query_record, id_to_enriched_meta[d["external_demo_id"]]), d["external_demo_id"]),
        )
        selected = candidates[:max_support]

    elif policy == "diagnostic_task_card_id_exact_match":
        uses_tc_id = True
        diag_only = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        q_tc_id = q_card["task_card_id"]
        # Filter matching exact card id
        candidates = [
            d for d in candidates
            if id_to_demo_card[d["external_demo_id"]]["support_task_card"]["task_card_id"] == q_tc_id
        ]
        selected = sorted(candidates, key=lambda d: d["external_demo_id"])[:max_support]

    elif policy == "diagnostic_compatible_label_mapping_oracle":
        uses_lbl = True
        uses_lbl_map = True
        diag_only = True
        lbl = enriched_query_record["audit_label_evaluation_only"]["target_relation_label"]
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
            and d["audit_label_evaluation_only"]["target_relation_label"] == lbl
        ]
        selected = sorted(candidates, key=lambda d: d["external_demo_id"])[:max_support]

    elif policy == "task_card_public_token_overlap_retrieval":
        uses_pub_tok = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                task_card_public_distance(q_card, id_to_demo_card[d["external_demo_id"]]["support_task_card"], "token_overlap"),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "task_card_change_preserve_signature_retrieval":
        uses_chg_pres = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                task_card_public_distance(q_card, id_to_demo_card[d["external_demo_id"]]["support_task_card"], "change_preserve_signature"),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "support_demo_public_caption_retrieval":
        uses_demo_pub = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                support_demo_public_view_distance(q_card, id_to_demo_card[d["external_demo_id"]]),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "hybrid_task_card_plus_source_signature_retrieval":
        uses_pub_tok = True
        uses_src_sig = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                0.5 * task_card_public_distance(q_card, id_to_demo_card[d["external_demo_id"]]["support_task_card"], "hybrid_public_card")
                + 0.5 * enriched_source_signature_distance(enriched_query_record, id_to_enriched_meta[d["external_demo_id"]]),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

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
            "materialization_source_phase": "P96",
            "materialization_method": f"real_task_card_retrieval_pilot_{policy}",
            "missing_result_summary": False,
            "missing_delta_summary": False,
        })

    inv = build_materialized_support_invariant_context(support_pairs)

    support_demo_cards_used = [
        id_to_demo_card[sid] for sid in selected_ids
    ]

    audit = {
        "support_selection_uses_hidden_relation_label": uses_lbl,
        "support_selection_uses_operator_id": False,
        "support_selection_uses_query_target": False,
        "support_selection_uses_query_delta": False,
        "support_selection_uses_query_result": False,
        "support_selection_uses_task_card_id": uses_tc_id,
        "support_selection_uses_public_task_card_tokens": uses_pub_tok,
        "support_selection_uses_change_preserve_signature": uses_chg_pres,
        "support_selection_uses_support_demo_public_view": uses_demo_pub,
        "support_selection_uses_source_signature": uses_src_sig,
        "support_selection_uses_compatible_label_mapping": uses_lbl_map,
        "support_selection_non_label_selected": not uses_lbl,
        "diagnostic_only": diag_only,
        "valid_for_real_external_task_context_evidence": val_real,
        "valid_for_dataset_contract_pilot": val_pilot,
        "valid_for_final_semantic_geometry_evidence": False,
        "valid_for_bridge_evidence": False,
    }

    return {
        "external_support_context_id": f"ctx_p96_{policy}_{q_id}",
        "policy": policy,
        "query_id": q_id,
        "support_demo_ids": selected_ids,
        "support_context": {
            "support_pairs": support_pairs,
            "support_invariant_context": inv,
        },
        "query_task_card_used": query_task_card_record,
        "support_demo_cards_used": support_demo_cards_used,
        "support_selection_audit": audit,
        "audit_label_evaluation_only": {
            "target_relation_label": enriched_query_record["audit_label_evaluation_only"]["target_relation_label"],
            "operator_id": enriched_query_record["audit_label_evaluation_only"]["operator_id"],
        },
    }


# ── Evaluations ───────────────────────────────────────────────────────────────

def _extract_label(ctx: dict) -> str:
    return ctx["audit_label_evaluation_only"]["target_relation_label"]


def evaluate_real_task_card_retrieval_policy_metric(
    enriched_query_records: list[dict],
    query_task_card_records: list[dict],
    demo_records: list[dict],
    support_demo_card_records: list[dict],
    enriched_support_metadata_records: list[dict],
    policy: str,
) -> dict:
    q_split = {q["external_query_id"]: q["split"] for q in enriched_query_records}
    tc_map = {tc["external_query_id"]: tc for tc in query_task_card_records}

    contexts = [
        build_real_task_card_support_context_for_query(
            q, tc_map[q["external_query_id"]], demo_records, support_demo_card_records, enriched_support_metadata_records, policy,
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

    diag_only = (policy in ("diagnostic_task_card_id_exact_match", "diagnostic_compatible_label_mapping_oracle"))
    val_pilot = not diag_only

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
        "valid_for_dataset_contract_pilot": val_pilot,
        "valid_for_real_external_task_context_evidence": False,
        "valid_for_final_semantic_geometry_evidence": False,
        "predictions": predictions,
        "contexts": contexts,
    }


# ── Controls ──────────────────────────────────────────────────────────────────

def compute_real_task_card_policy_negative_controls(
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
        "shuffled_control_target_labels_preserved": True,
        "shuffled_control_support_deltas_permuted": True,
        "target_labels_shifted": False,
        "true_labels_preserved": True,
        "query_source_only_negative_control_applicable": False,
    }


# ── Leakage audit ─────────────────────────────────────────────────────────────

def audit_real_task_card_retrieval_leakage(contexts: list[dict]) -> dict:
    lbl_used = 0
    op_used = 0
    q_tgt_used = 0
    q_del_used = 0
    q_res_used = 0

    tc_id_used = 0
    pub_tok_used = 0
    chg_pres_used = 0
    demo_pub_used = 0
    src_sig_used = 0
    lbl_map_used = 0

    diag_count = 0
    pilot_count = 0
    val_real_count = 0

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

        if sa.get("support_selection_uses_task_card_id", False):
            tc_id_used += 1
        if sa.get("support_selection_uses_public_task_card_tokens", False):
            pub_tok_used += 1
        if sa.get("support_selection_uses_change_preserve_signature", False):
            chg_pres_used += 1
        if sa.get("support_selection_uses_support_demo_public_view", False):
            demo_pub_used += 1
        if sa.get("support_selection_uses_source_signature", False):
            src_sig_used += 1
        if sa.get("support_selection_uses_compatible_label_mapping", False):
            lbl_map_used += 1

        if sa.get("diagnostic_only", False):
            diag_count += 1
        if sa.get("valid_for_dataset_contract_pilot", False):
            pilot_count += 1
        if sa.get("valid_for_real_external_task_context_evidence", False):
            val_real_count += 1

    # Audit pass logic: for non-diagnostic (public) policies, no hidden keys used
    diag_pass = True
    for c in contexts:
        sa = c["support_selection_audit"]
        if sa.get("valid_for_dataset_contract_pilot", False):
            if (
                sa.get("support_selection_uses_hidden_relation_label", False)
                or sa.get("support_selection_uses_operator_id", False)
                or sa.get("support_selection_uses_query_target", False)
                or sa.get("support_selection_uses_query_delta", False)
                or sa.get("support_selection_uses_query_result", False)
                or sa.get("support_selection_uses_task_card_id", False)
                or sa.get("support_selection_uses_compatible_label_mapping", False)
            ):
                diag_pass = False

    return {
        "context_count": len(contexts),
        "hidden_relation_label_used_count": lbl_used,
        "operator_id_used_count": op_used,
        "query_target_used_count": q_tgt_used,
        "query_delta_used_count": q_del_used,
        "query_result_used_count": q_res_used,

        "task_card_id_used_count": tc_id_used,
        "public_task_card_tokens_used_count": pub_tok_used,
        "change_preserve_signature_used_count": chg_pres_used,
        "support_demo_public_view_used_count": demo_pub_used,
        "source_signature_used_count": src_sig_used,
        "compatible_label_mapping_used_count": lbl_map_used,

        "diagnostic_context_count": diag_count,
        "dataset_contract_pilot_context_count": pilot_count,
        "valid_real_evidence_context_count": val_real_count,

        "diagnostic_pass": diag_pass,
    }


# ── Proxy audit ───────────────────────────────────────────────────────────────

def audit_real_task_card_retrieval_proxy_risk(policy_results: dict) -> dict:
    risk = True
    return {
        "audit_defined": True,
        "proxy_risk_present": risk,
        "assignment_is_diagnostic_label_mapped": True,
        "task_card_id_exact_match_is_proxy": True,
        "public_token_policy_still_inherits_assignment_proxy": True,
        "real_data_required_to_remove_proxy": True,
        "diagnostic_pass": True,
    }


# ── Bridge boundary ───────────────────────────────────────────────────────────

def audit_bridge_boundary_after_real_task_card_retrieval_pilot(
    pipeline_ready: bool,
    diagnostic_pipeline_signal_present: bool,
) -> dict:
    return {
        "bridge_ready": False,
        "bridge_implementation_allowed": False,

        "task_card_retrieval_pipeline_ready": pipeline_ready,
        "diagnostic_task_card_retrieval_signal_present": diagnostic_pipeline_signal_present,

        "valid_for_real_external_task_context_evidence": False,
        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,

        "blocking_reasons": [
            "p96_is_diagnostic_pipeline_pilot_not_real_external_evidence",
            "task_card_assignments_are_label_mapped_from_synthetic_data",
            "valid_external_task_context_evidence_not_available",
            "learned_metric_evidence_not_present",
            "semantic_metric_not_ready",
            "bridge_input_contract_not_defined",
            "bridge_validation_not_run",
        ],
        "diagnostic_pass": True,
    }


# ── Public probe ───────────────────────────────────────────────────────────────

def run_p96_real_task_card_support_retrieval_pilot_probe() -> dict:
    # 0. Validate P95 source contracts
    contracts_val = validate_source_contracts_for_p95()
    contracts_ok = contracts_val["source_contracts_validated"]

    # 1. Build records
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    demo_recs = build_external_demonstration_records(p70a, p70b)
    selector_recs = build_selector_dataset_records(p70a, p70b)

    enriched_queries = build_enriched_query_records(selector_recs)
    enriched_support = build_enriched_support_metadata_records(demo_recs)

    task_card_catalog = build_task_card_catalog()
    query_task_card_records = build_query_task_card_records(enriched_queries, task_card_catalog)
    support_demo_card_records = build_support_demonstration_card_records(demo_recs, task_card_catalog)

    # 2. Evaluate all policies
    policy_metric_results = {}
    policy_negative_controls = {}
    policy_leakage_audits = {}
    policy_shape_audits = {}

    for pol in ALL_P96_POLICIES:
        er = evaluate_real_task_card_retrieval_policy_metric(
            enriched_queries, query_task_card_records, demo_recs, support_demo_card_records, enriched_support, pol,
        )
        policy_metric_results[pol] = er

        nc = compute_real_task_card_policy_negative_controls(
            enriched_queries, pol, er["effective_accuracy"], er,
        )
        policy_negative_controls[pol] = nc

        contexts = er["contexts"]
        la = audit_real_task_card_retrieval_leakage(contexts)
        policy_leakage_audits[pol] = la

        sa = audit_metric_shape_compatibility(contexts)
        policy_shape_audits[pol] = sa

    # 3. Audits
    proxy_audit = audit_real_task_card_retrieval_proxy_risk(policy_metric_results)

    # 4. Pipeline readiness
    pipeline_ready = (
        contracts_ok
        and len(task_card_catalog) == 10
        and len(query_task_card_records) > 0
        and len(support_demo_card_records) > 0
        and all(policy_metric_results[p]["sample_count"] > 0 for p in ALL_P96_POLICIES)
    )

    # Check for pilot signal crossing threshold
    pilot_sig = False
    for pol in ALL_P96_POLICIES:
        mr = policy_metric_results[pol]
        nc = policy_negative_controls[pol]
        la = policy_leakage_audits[pol]
        sa = policy_shape_audits[pol]

        # Check only public-token policies (not diagnostic)
        if mr["diagnostic_only"]:
            continue

        if (
            mr["effective_accuracy"] >= 0.40
            and mr["coverage"] >= 0.80
            and nc["beats_shuffled_by"] >= 0.10
            and nc["beats_zero_delta_by"] >= 0.10
            and la["diagnostic_pass"]
            and sa["diagnostic_pass"]
        ):
            pilot_sig = True

    # 5. Next phase logic
    if pilot_sig:
        next_phase = "P97_real_task_card_raw_data_collection_contract_no_training_no_bridge"
    else:
        next_phase = "P97_task_card_retrieval_policy_repair_no_training_no_bridge"

    # 6. Bridge boundary
    bridge_audit = audit_bridge_boundary_after_real_task_card_retrieval_pilot(
        pipeline_ready, pilot_sig,
    )

    # 7. Clean outputs
    clean_metric_results = {}
    clean_predictions = {}
    for pol in ALL_P96_POLICIES:
        mr = policy_metric_results[pol]
        clean_predictions[pol] = mr["predictions"]
        clean_metric_results[pol] = {
            k: v for k, v in mr.items() if k not in ("predictions", "contexts")
        }

    verdict_str = VERDICT if contracts_ok else "P96_BLOCKED_BY_SOURCE_CONTRACT"

    return {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,
        "verdict": verdict_str,

        "source_real_task_context_contract_phase": SOURCE_REAL_TASK_CONTEXT_CONTRACT_PHASE,
        "source_external_task_context_phase": SOURCE_EXTERNAL_TASK_CONTEXT_PHASE,

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "source_contracts_validated": contracts_ok,

        "task_card_catalog_count": len(task_card_catalog),
        "query_task_card_record_count": len(query_task_card_records),
        "support_demo_card_record_count": len(support_demo_card_records),

        "policy_metric_results": clean_metric_results,
        "policy_predictions": clean_predictions,
        "policy_negative_controls": policy_negative_controls,
        "policy_leakage_audits": policy_leakage_audits,
        "policy_shape_compatibility_audits": policy_shape_audits,
        "policy_proxy_audit": proxy_audit,

        "task_card_retrieval_pipeline_ready": pipeline_ready,
        "task_card_retrieval_signal_present": pilot_sig,
        "task_card_external_metric_supported": False,

        "valid_external_task_context_available": False,
        "valid_for_real_external_task_context_evidence": False,
        "ready_for_real_data_collection": True,

        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "bridge_implementation_allowed": False,
        "bridge_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,

        "recommended_next_phase": next_phase,

        "bridge_boundary_after_real_task_card_retrieval_pilot": bridge_audit,

        "json_safe": True,
        "diagnostic_only": True,
    }
