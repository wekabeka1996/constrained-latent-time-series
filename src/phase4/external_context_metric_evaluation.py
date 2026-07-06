# src/phase4/external_context_metric_evaluation.py

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

PHASE = "P91"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "External Context Metric Evaluation"
CONTRACT_VERSION = "phase4_p91_external_context_metric_evaluation_v1"

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

EXTERNAL_CONTEXT_METRIC_EVALUATION_DEFINED = True
POLICY_COMPARISON_DEFINED = True
PROXY_LEAKAGE_AUDIT_DEFINED = True
NON_LABEL_SELECTED_METRIC_EVIDENCE_DEFINED = True

NONLEARNED_METRIC_SIGNAL_PRESENT = True
HARD_GENERALIZATION_SUPPORTED = True
EXTERNAL_CONTEXT_BUILDER_READY = True
NON_LABEL_SELECTED_SUPPORT_READY = True

NON_LABEL_SELECTED_METRIC_SIGNAL_PRESENT = False  # computed
EXTERNAL_CONTEXT_METRIC_SUPPORTED = False  # computed

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "external_context_metric_evaluation"
VERDICT = "P91_READY_FOR_REVIEW"

ALL_CLASSES = [
    "translate_x", "translate_y", "scale_s", "reflect_x",
    "nonlinear_x_from_y", "change_frequency", "scale_amplitude",
    "shift_phase", "scale_volatility_envelope", "shift_trend",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(ALL_CLASSES)}

# ── Static accepted P90 metadata ──────────────────────────────────────────────

ACCEPTED_P90_REPORT_METADATA = {
    "phase": "P90",
    "verdict": "P90_READY_FOR_REVIEW",

    "source_contracts_validated": True,
    "external_context_builder_ready": True,
    "non_label_selected_support_ready": True,

    "external_demonstration_record_count": 114,
    "external_query_record_count": 114,

    "label_selected_oracle_support_accuracy": 0.6666666666666666,
    "observable_domain_split_retrieval_accuracy": 0.2222222222222222,
    "external_manifest_support_accuracy": 0.2222222222222222,

    "observable_domain_split_retrieval_leakage_clean": True,
    "external_manifest_support_leakage_clean": True,

    "expected_delta_only_dim": 18,
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

    "recommended_next_phase": "P91_external_context_metric_evaluation_no_training_no_bridge",
    "metadata_source": "accepted_p90_report_static_metadata",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_label(ctx: dict) -> str:
    return ctx["audit_label_evaluation_only"]["target_relation_label"]


# ── Source validation ──────────────────────────────────────────────────────────

def validate_source_contracts_for_p91() -> dict:
    validated = True
    missing = []
    m = ACCEPTED_P90_REPORT_METADATA

    if m.get("phase") != "P90" or m.get("verdict") != "P90_READY_FOR_REVIEW":
        validated = False
        missing.append("p90_invalid_phase_or_verdict")
    if m.get("source_contracts_validated") is not True:
        validated = False
        missing.append("p90_source_unvalidated")
    if m.get("external_context_builder_ready") is not True:
        validated = False
        missing.append("p90_builder_not_ready")
    if m.get("non_label_selected_support_ready") is not True:
        validated = False
        missing.append("p90_non_label_selected_not_ready")
    if m.get("metric_shape_compatibility_pass") is not True:
        validated = False
        missing.append("p90_metric_shape_incompatible")
    if m.get("model_training_performed") is not False:
        validated = False
        missing.append("p90_unexpected_training")
    if m.get("bridge_ready") is not False:
        validated = False
        missing.append("p90_bridge_ready")
    if m.get("recommended_next_phase") != "P91_external_context_metric_evaluation_no_training_no_bridge":
        validated = False
        missing.append("p90_recommended_phase_mismatch")

    return {
        "source_contracts_validated": validated,
        "p90_validated": validated,
        "p90_external_context_builder_ready_preserved": m.get("external_context_builder_ready") is True,
        "p90_non_label_selected_support_ready_preserved": m.get("non_label_selected_support_ready") is True,
        "p90_metric_shape_compatibility_preserved": m.get("metric_shape_compatibility_pass") is True,
        "p90_no_training_preserved": m.get("model_training_performed") is False,
        "p90_bridge_not_ready_preserved": m.get("bridge_ready") is False,
        "p90_recommended_p91_preserved": m.get("recommended_next_phase") == "P91_external_context_metric_evaluation_no_training_no_bridge",
        "metadata_source": "accepted_p90_report_static_metadata",
        "missing_or_invalid": missing,
    }


# ── Context builders ──────────────────────────────────────────────────────────

def build_policy_contexts(
    query_records: list[dict],
    demo_records: list[dict],
    policy: str,
) -> list[dict]:
    return [
        build_external_support_context_for_query(q, demo_records, policy)
        for q in query_records
    ]


# ── Prototype builder ─────────────────────────────────────────────────────────

def build_class_prototypes_from_contexts(contexts: list[dict]) -> dict:
    sums: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for ctx in contexts:
        lbl = _extract_label(ctx)
        idx = CLASS_TO_IDX.get(lbl, -1)
        if idx == -1:
            continue
        vec = encode_external_support_delta_context(ctx)
        if idx not in sums:
            sums[idx] = [0.0] * len(vec)
            counts[idx] = 0
        counts[idx] += 1
        for j, v in enumerate(vec):
            sums[idx][j] += v
    prototypes = {}
    for idx in sums:
        c = counts[idx]
        prototypes[idx] = [s / c for s in sums[idx]]
    return prototypes


# ── Prediction ─────────────────────────────────────────────────────────────────

def _l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def predict_nearest_prototype(
    vec: list[float],
    prototypes: dict,
) -> dict:
    if not prototypes:
        return {
            "predicted_class_idx": None,
            "predicted_label": None,
            "unsupported": True,
            "reason": "missing_class_prototype",
        }
    best_idx = None
    best_dist = float("inf")
    for idx, proto in prototypes.items():
        d = _l2(vec, proto)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return {
        "predicted_class_idx": best_idx,
        "predicted_label": IDX_TO_CLASS.get(best_idx),
        "unsupported": False,
        "distance": best_dist,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_policy_metric(
    query_records: list[dict],
    demo_records: list[dict],
    policy: str,
) -> dict:
    q_split = {q["external_query_id"]: q["split"] for q in query_records}
    contexts = build_policy_contexts(query_records, demo_records, policy)

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

    is_diag = (policy == "label_selected_oracle_support")

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
        "diagnostic_only": is_diag,
        "valid_for_final_semantic_geometry_evidence": False,
        "predictions": predictions,
    }


# ── Negative controls ──────────────────────────────────────────────────────────

def _run_shuffled_control(
    query_records: list[dict],
    demo_records: list[dict],
    policy: str,
) -> float:
    q_split = {q["external_query_id"]: q["split"] for q in query_records}
    contexts = build_policy_contexts(query_records, demo_records, policy)

    train_ctx = [c for c in contexts if q_split.get(c["query_id"]) == "train"]
    test_ctx = [c for c in contexts if q_split.get(c["query_id"]) == "test"]

    prototypes = build_class_prototypes_from_contexts(train_ctx)

    # encode test vectors
    test_vecs = [encode_external_support_delta_context(c) for c in test_ctx]
    test_labels = [_extract_label(c) for c in test_ctx]

    # deterministic permutation: shift by 7 positions
    n = len(test_vecs)
    if n == 0:
        return 0.0
    shift = 7 % n if n > 0 else 0
    shuffled_vecs = test_vecs[shift:] + test_vecs[:shift]

    correct = 0
    total_eval = 0
    for i in range(n):
        true_idx = CLASS_TO_IDX.get(test_labels[i], -1)
        if true_idx == -1 or true_idx not in prototypes:
            continue
        total_eval += 1
        pred = predict_nearest_prototype(shuffled_vecs[i], prototypes)
        if pred["predicted_class_idx"] == true_idx:
            correct += 1

    return float(correct / total_eval) if total_eval > 0 else 0.0


def _run_zero_delta_control(
    query_records: list[dict],
    demo_records: list[dict],
    policy: str,
) -> float:
    q_split = {q["external_query_id"]: q["split"] for q in query_records}
    contexts = build_policy_contexts(query_records, demo_records, policy)

    train_ctx = [c for c in contexts if q_split.get(c["query_id"]) == "train"]
    test_ctx = [c for c in contexts if q_split.get(c["query_id"]) == "test"]

    prototypes = build_class_prototypes_from_contexts(train_ctx)
    zero_vec = [0.0] * 18

    correct = 0
    total_eval = 0
    for ctx in test_ctx:
        true_idx = CLASS_TO_IDX.get(_extract_label(ctx), -1)
        if true_idx == -1 or true_idx not in prototypes:
            continue
        total_eval += 1
        pred = predict_nearest_prototype(zero_vec, prototypes)
        if pred["predicted_class_idx"] == true_idx:
            correct += 1

    return float(correct / total_eval) if total_eval > 0 else 0.0


def compute_policy_negative_controls(
    query_records: list[dict],
    demo_records: list[dict],
    policy: str,
    normal_eff_acc: float,
) -> dict:
    shuf_acc = _run_shuffled_control(query_records, demo_records, policy)
    zero_acc = _run_zero_delta_control(query_records, demo_records, policy)

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


# ── Proxy leakage audit ───────────────────────────────────────────────────────

def audit_observable_metadata_proxy_leakage(demo_records: list[dict]) -> dict:
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
        # value -> list of labels
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
            from collections import Counter
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

    proxy_present = len(high_risk) > 0

    return {
        "audit_defined": True,
        "metadata_field_results": field_results,
        "high_proxy_risk_fields": high_risk,
        "proxy_leakage_risk_present": proxy_present,
        "diagnostic_pass": True,  # proxy risk doesn't block P91
    }


# ── Bridge boundary ───────────────────────────────────────────────────────────

def audit_bridge_boundary_after_external_context_metric_evaluation(
    external_context_metric_supported: bool,
    non_label_selected_metric_signal_present: bool,
) -> dict:
    return {
        "bridge_ready": False,
        "bridge_implementation_allowed": False,
        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": True,
        "external_context_builder_ready": True,
        "non_label_selected_support_ready": True,
        "external_context_metric_supported": external_context_metric_supported,
        "non_label_selected_metric_signal_present": non_label_selected_metric_signal_present,
        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,
        "blocking_reasons": [
            "learned_metric_evidence_not_present",
            "semantic_metric_not_ready",
            "external_metric_not_sufficient_for_bridge",
            "bridge_input_contract_not_defined",
            "bridge_validation_not_run",
        ],
        "diagnostic_pass": True,
    }


# ── Public probe ───────────────────────────────────────────────────────────────

def run_p91_external_context_metric_evaluation_probe() -> dict:
    # 0. Validate source contracts
    contracts_val = validate_source_contracts_for_p91()
    contracts_ok = contracts_val["source_contracts_validated"]

    # 1. Build demonstration & query records
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    demo_recs = build_external_demonstration_records(p70a, p70b)
    selector_recs = build_selector_dataset_records(p70a, p70b)
    query_recs = build_external_query_records(selector_recs)

    # 2. Evaluate metric for each policy
    policies = [
        "label_selected_oracle_support",
        "observable_domain_split_retrieval",
        "external_manifest_support",
    ]

    policy_metric_results = {}
    policy_negative_controls = {}

    for pol in policies:
        mr = evaluate_policy_metric(query_recs, demo_recs, pol)
        policy_metric_results[pol] = mr
        nc = compute_policy_negative_controls(
            query_recs, demo_recs, pol, mr["effective_accuracy"],
        )
        policy_negative_controls[pol] = nc

    # 3. Leakage re-verification for non-label-selected policies
    obs_contexts = build_policy_contexts(query_recs, demo_recs, "observable_domain_split_retrieval")
    man_contexts = build_policy_contexts(query_recs, demo_recs, "external_manifest_support")
    obs_leak = audit_external_support_selection_leakage(obs_contexts)
    man_leak = audit_external_support_selection_leakage(man_contexts)
    obs_shape = audit_metric_shape_compatibility(obs_contexts)
    man_shape = audit_metric_shape_compatibility(man_contexts)

    # 4. Proxy leakage audit
    proxy_audit = audit_observable_metadata_proxy_leakage(demo_recs)

    # 5. Determine non-label-selected metric signal
    non_label_policies = ["observable_domain_split_retrieval", "external_manifest_support"]
    any_signal = False
    for pol in non_label_policies:
        mr = policy_metric_results[pol]
        nc = policy_negative_controls[pol]
        leak_ok = obs_leak["diagnostic_pass"] if pol == "observable_domain_split_retrieval" else man_leak["diagnostic_pass"]
        shape_ok = obs_shape["diagnostic_pass"] if pol == "observable_domain_split_retrieval" else man_shape["diagnostic_pass"]
        if (
            mr["effective_accuracy"] >= 0.40
            and mr["coverage"] >= 0.80
            and nc["beats_shuffled_by"] >= 0.10
            and nc["beats_zero_delta_by"] >= 0.10
            and leak_ok
            and shape_ok
        ):
            any_signal = True

    non_label_signal = any_signal
    ext_metric_supported = (
        contracts_ok
        and non_label_signal
        and proxy_audit["audit_defined"]
    )

    # 6. Bridge boundary
    bridge_audit = audit_bridge_boundary_after_external_context_metric_evaluation(
        ext_metric_supported, non_label_signal,
    )

    # 7. Recommended next phase
    if ext_metric_supported:
        next_phase = "P92_external_context_metric_hardening_no_training_no_bridge"
    elif not non_label_signal and contracts_ok:
        next_phase = "P92_non_label_selected_support_retrieval_improvement_no_training_no_bridge"
    elif not obs_leak["diagnostic_pass"] or not man_leak["diagnostic_pass"]:
        next_phase = "P92_external_context_leakage_repair_no_training_no_bridge"
    elif not obs_shape["diagnostic_pass"] or not man_shape["diagnostic_pass"]:
        next_phase = "P92_external_context_metric_shape_repair_no_training_no_bridge"
    else:
        next_phase = "P92_non_label_selected_support_retrieval_improvement_no_training_no_bridge"

    # 8. Verdict
    verdict_str = VERDICT if contracts_ok else "P91_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED or TORCH_TRAINING_PERFORMED:
            verdict_str = "P91_BLOCKED_BY_UNEXPECTED_TRAINING_FLAG"

    # 9. Build output
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,

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

        "policy_metric_results": {
            pol: {k: v for k, v in policy_metric_results[pol].items() if k != "predictions"}
            for pol in policies
        },
        "policy_metric_predictions": {
            pol: policy_metric_results[pol]["predictions"]
            for pol in policies
        },
        "policy_negative_control_results": policy_negative_controls,

        "leakage_verification": {
            "observable_domain_split_retrieval": obs_leak,
            "external_manifest_support": man_leak,
        },
        "shape_compatibility_verification": {
            "observable_domain_split_retrieval": obs_shape,
            "external_manifest_support": man_shape,
        },

        "proxy_leakage_audit": proxy_audit,

        "non_label_selected_metric_signal_present": non_label_signal,
        "external_context_metric_supported": ext_metric_supported,

        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": True,
        "external_context_builder_ready": True,
        "non_label_selected_support_ready": True,

        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "bridge_implementation_allowed": False,
        "bridge_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,

        "recommended_next_phase": next_phase,

        "bridge_boundary_after_external_context_metric_evaluation": bridge_audit,

        "json_safe": True,
        "diagnostic_only": True,
    }

    return output
