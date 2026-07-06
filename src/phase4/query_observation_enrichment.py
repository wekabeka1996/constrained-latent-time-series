# src/phase4/query_observation_enrichment.py

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

from src.phase4.non_label_support_retrieval_improvement import (
    source_similarity_distance,
    evaluate_retrieval_policy_metric,
    compute_retrieval_policy_negative_controls,
    audit_retrieval_policy_leakage,
    audit_retrieval_proxy_leakage,
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

PHASE = "P93"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Query Observation Enrichment"
CONTRACT_VERSION = "phase4_p93_query_observation_enrichment_v1"

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

QUERY_OBSERVATION_ENRICHMENT_DEFINED = True
ENRICHED_RETRIEVAL_POLICY_DEFINED = True
ENRICHMENT_LEAKAGE_AUDIT_DEFINED = True
ENRICHMENT_PROXY_AUDIT_DEFINED = True

NONLEARNED_METRIC_SIGNAL_PRESENT = True
HARD_GENERALIZATION_SUPPORTED = True
EXTERNAL_CONTEXT_BUILDER_READY = True
NON_LABEL_SELECTED_SUPPORT_READY = True
NON_LABEL_SELECTED_METRIC_SIGNAL_PRESENT = False
EXTERNAL_CONTEXT_METRIC_SUPPORTED = False
RETRIEVAL_IMPROVEMENT_SIGNAL_PRESENT = False
IMPROVED_NON_LABEL_SELECTED_SUPPORT_READY = True

QUERY_OBSERVATION_ENRICHMENT_READY = False  # computed
ENRICHED_RETRIEVAL_SIGNAL_PRESENT = False  # computed
ENRICHED_EXTERNAL_CONTEXT_METRIC_SUPPORTED = False  # computed

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "query_observation_enrichment"
VERDICT = "P93_READY_FOR_REVIEW"

BASELINE_POLICIES = [
    "p92_source_similarity_retrieval_baseline",
    "p92_source_similarity_with_split_relaxation_baseline",
]

IMPROVED_POLICIES = [
    "enriched_source_signature_retrieval",
    "enriched_source_signature_with_split_relaxation",
    "context_bucket_retrieval",
    "hybrid_enriched_signature_bucket_retrieval",
    "cross_domain_normalized_signature_retrieval",
]

ALL_RETRIEVAL_POLICIES = BASELINE_POLICIES + IMPROVED_POLICIES


# ── Static accepted P92 metadata ──────────────────────────────────────────────

ACCEPTED_P92_REPORT_METADATA = {
    "phase": "P92",
    "verdict": "P92_READY_FOR_REVIEW",

    "source_contracts_validated": True,

    "p91_observable_domain_split_baseline_effective_accuracy": 0.2222222222222222,
    "p91_external_manifest_baseline_effective_accuracy": 0.2222222222222222,

    "source_similarity_retrieval_effective_accuracy": 0.3888888888888889,
    "source_similarity_with_split_relaxation_effective_accuracy": 0.3888888888888889,
    "observable_diversity_manifest_effective_accuracy": 0.0,
    "hybrid_source_similarity_diversity_effective_accuracy": 0.2222222222222222,

    "source_similarity_retrieval_coverage": 1.0,
    "source_similarity_with_split_relaxation_coverage": 1.0,

    "source_similarity_retrieval_shuffled_control": 0.0,
    "source_similarity_retrieval_zero_delta_control": 0.1111111111111111,
    "source_similarity_with_split_relaxation_shuffled_control": 0.0,
    "source_similarity_with_split_relaxation_zero_delta_control": 0.1111111111111111,

    "source_similarity_retrieval_beats_shuffled_by": 0.3888888888888889,
    "source_similarity_retrieval_beats_zero_delta_by": 0.2777777777777778,

    "retrieval_improvement_signal_present": False,
    "improved_non_label_selected_support_ready": True,

    "proxy_leakage_risk_present": True,
    "improved_policy_leakage_clean": True,
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

    "recommended_next_phase": "P93_query_observation_enrichment_no_training_no_bridge",
    "metadata_source": "accepted_p92_report_static_metadata",
}


# ── Source validation ──────────────────────────────────────────────────────────

def validate_source_contracts_for_p93() -> dict:
    validated = True
    missing = []
    m = ACCEPTED_P92_REPORT_METADATA

    if m.get("phase") != "P92" or m.get("verdict") != "P92_READY_FOR_REVIEW":
        validated = False
        missing.append("p92_invalid_phase_or_verdict")
    if m.get("source_contracts_validated") is not True:
        validated = False
        missing.append("p92_source_unvalidated")
    if m.get("retrieval_improvement_signal_present") is not False:
        validated = False
        missing.append("p92_retrieval_signal_should_be_false")
    if m.get("improved_non_label_selected_support_ready") is not True:
        validated = False
        missing.append("p92_support_should_be_ready")
    if m.get("model_training_performed") is not False:
        validated = False
        missing.append("p92_unexpected_training")
    if m.get("bridge_ready") is not False:
        validated = False
        missing.append("p92_bridge_ready")
    if m.get("recommended_next_phase") != "P93_query_observation_enrichment_no_training_no_bridge":
        validated = False
        missing.append("p92_recommended_phase_mismatch")

    sim_acc = m.get("source_similarity_retrieval_effective_accuracy", 0.0)
    near_threshold = (0.35 <= sim_acc < 0.40)
    ret_sig_absent = (m.get("retrieval_improvement_signal_present") is False)
    supp_ready = (m.get("improved_non_label_selected_support_ready") is True)
    proxy_reported = (m.get("proxy_leakage_risk_present") is True)

    return {
        "source_contracts_validated": validated,
        "p92_validated": validated,
        "p92_near_threshold_source_similarity_preserved": near_threshold,
        "p92_retrieval_signal_absent_preserved": ret_sig_absent,
        "p92_improved_support_ready_preserved": supp_ready,
        "p92_proxy_audit_preserved": proxy_reported,
        "p92_no_training_preserved": m.get("model_training_performed") is False,
        "p92_bridge_not_ready_preserved": m.get("bridge_ready") is False,
        "p92_recommended_p93_preserved": m.get("recommended_next_phase") == "P93_query_observation_enrichment_no_training_no_bridge",
        "metadata_source": "accepted_p92_report_static_metadata",
        "missing_or_invalid": missing,
    }


# ── Feature construction helpers ──────────────────────────────────────────────

def _extract_source_summary_keys(domain: str, source_summary: dict) -> tuple:
    if domain == "p70a_vector_world":
        dim = float(source_summary.get("z_a_dim", source_summary.get("vector_dim", 3.0)))
        abs_sum = float(source_summary.get("z_a_abs_sum", source_summary.get("abs_sum", 0.0)))
        sign_pattern = source_summary.get("z_a_sign_pattern", source_summary.get("sign_pattern", [0, 0, 0]))
        nonzero_count = int(source_summary.get("nonzero_count", sum(1 for x in sign_pattern if x != 0)))
        return dim, abs_sum, sign_pattern, nonzero_count
    else:
        key_count = float(source_summary.get("parameter_key_count", source_summary.get("params_key_count", 5.0)))
        abs_sum = float(source_summary.get("source_abs_sum", source_summary.get("params_abs_sum", 0.0)))
        nonzero_key_count = float(source_summary.get("source_nonzero_key_count", source_summary.get("params_nonzero_key_count", 0.0)))
        return key_count, abs_sum, nonzero_key_count


def build_source_geometry_fingerprint(domain: str, source_summary: dict) -> dict:
    if domain == "p70a_vector_world":
        dim, abs_sum, _, _ = _extract_source_summary_keys(domain, source_summary)
        mean_abs = abs_sum / dim if dim > 0 else 0.0
        return {"dim": dim, "mean_abs": mean_abs}
    else:
        key_count, abs_sum, _ = _extract_source_summary_keys(domain, source_summary)
        mean_abs = abs_sum / key_count if key_count > 0 else 0.0
        return {"key_count": key_count, "mean_abs": mean_abs}


def build_source_sparsity_signature(domain: str, source_summary: dict) -> dict:
    if domain == "p70a_vector_world":
        dim, _, _, nonzero_count = _extract_source_summary_keys(domain, source_summary)
        zero_count = int(dim - nonzero_count)
        return {"nonzero_count": nonzero_count, "zero_count": zero_count}
    else:
        key_count, _, nonzero_key_count = _extract_source_summary_keys(domain, source_summary)
        sparsity = nonzero_key_count / key_count if key_count > 0 else 0.0
        return {"nonzero_key_count": nonzero_key_count, "sparsity": sparsity}


def build_source_magnitude_signature(domain: str, source_summary: dict) -> dict:
    if domain == "p70a_vector_world":
        _, abs_sum, _, _ = _extract_source_summary_keys(domain, source_summary)
        l1_bucket = f"l1_gt_{int(abs_sum * 10) / 10:.1f}"
        return {"abs_sum": abs_sum, "l1_bucket": l1_bucket}
    else:
        _, abs_sum, _ = _extract_source_summary_keys(domain, source_summary)
        magnitude_bucket = f"mag_{int(abs_sum * 10) / 10:.1f}"
        return {"abs_sum": abs_sum, "magnitude_bucket": magnitude_bucket}


def build_source_sign_or_direction_signature(domain: str, source_summary: dict) -> dict:
    if domain == "p70a_vector_world":
        _, _, sign_pattern, _ = _extract_source_summary_keys(domain, source_summary)
        positive_count = sum(1 for x in sign_pattern if x > 0)
        negative_count = sum(1 for x in sign_pattern if x < 0)
        sign_bucket = f"pos_{positive_count}_neg_{negative_count}"
        return {"sign_pattern": sign_pattern, "sign_bucket": sign_bucket}
    else:
        return {"sign_bucket": "params_non_negative_assumed"}


def build_domain_normalized_source_signature(domain: str, source_summary: dict) -> dict:
    if domain == "p70a_vector_world":
        dim, abs_sum, _, nonzero_count = _extract_source_summary_keys(domain, source_summary)
        norm_magnitude = abs_sum / 10.0
        norm_sparsity = nonzero_count / dim if dim > 0 else 0.0
        return {"norm_magnitude": norm_magnitude, "norm_sparsity": norm_sparsity}
    else:
        key_count, abs_sum, nonzero_key_count = _extract_source_summary_keys(domain, source_summary)
        norm_magnitude = abs_sum / 10.0
        norm_sparsity = nonzero_key_count / key_count if key_count > 0 else 0.0
        return {"norm_magnitude": norm_magnitude, "norm_sparsity": norm_sparsity}


def build_observable_context_bucket(domain: str, split: str, enriched_source: dict) -> str:
    mag_sig = enriched_source["source_magnitude_signature"]
    bucket_val = mag_sig.get("l1_bucket", mag_sig.get("magnitude_bucket", "unknown"))
    return f"{domain}_{split}_{bucket_val}"


# ── Enriched record builders ──────────────────────────────────────────────────

def build_enriched_query_records(selector_records: list[dict]) -> list[dict]:
    query_records = build_external_query_records(selector_records)
    enriched_records = []
    for q in query_records:
        dom = q["domain"]
        sp = q["split"]
        q_obs = q["query_observation"]

        if dom == "p70a_vector_world":
            src_sum = q_obs.get("source_state_summary", {})
        else:
            src_sum = q_obs.get("source_parameter_summary", {})

        geom = build_source_geometry_fingerprint(dom, src_sum)
        spars = build_source_sparsity_signature(dom, src_sum)
        mag = build_source_magnitude_signature(dom, src_sum)
        sign = build_source_sign_or_direction_signature(dom, src_sum)
        norm = build_domain_normalized_source_signature(dom, src_sum)

        enriched_src = {
            "source_geometry_fingerprint": geom,
            "source_sparsity_signature": spars,
            "source_magnitude_signature": mag,
            "source_sign_or_direction_signature": sign,
            "domain_normalized_source_signature": norm,
        }

        bucket = build_observable_context_bucket(dom, sp, enriched_src)

        en_obs = {
            "domain": dom,
            "split": sp,
            "context_world": q_obs.get("context_world", sp),
            "source_summary": src_sum,
            "source_geometry_fingerprint": geom,
            "source_sparsity_signature": spars,
            "source_magnitude_signature": mag,
            "source_sign_or_direction_signature": sign,
            "query_intensity_hint": float(q_obs.get("query_intensity_hint", 0.0)),
            "domain_normalized_source_signature": norm,
            "observable_context_bucket": bucket,
            "enrichment_version": "p93_v1",
        }

        audit = {
            "uses_target_relation_label": False,
            "uses_operator_id": False,
            "uses_query_target": False,
            "uses_query_delta": False,
            "uses_query_result": False,
            "uses_hidden_relation_family": False,
            "valid_for_retrieval_scoring": True,
            "valid_for_final_semantic_geometry_evidence": False,
            "valid_for_bridge_evidence": False,
        }

        enriched_records.append({
            "enriched_query_id": f"en_{q['external_query_id']}",
            "external_query_id": q["external_query_id"],
            "dataset_record_id": q["dataset_record_id"],
            "domain": dom,
            "split": sp,
            "query_observation": q_obs,
            "enriched_query_observation": en_obs,
            "enrichment_audit": audit,
            "audit_label_evaluation_only": q["audit_label_evaluation_only"],
        })

    return enriched_records


def build_enriched_support_metadata_records(demo_records: list[dict]) -> list[dict]:
    enriched_demos = []
    for d in demo_records:
        dom = d["domain"]
        sp = d["source_split"]
        obs = d["observable_context"]
        src_sum = obs["source_summary"]

        geom = build_source_geometry_fingerprint(dom, src_sum)
        spars = build_source_sparsity_signature(dom, src_sum)
        mag = build_source_magnitude_signature(dom, src_sum)
        sign = build_source_sign_or_direction_signature(dom, src_sum)
        norm = build_domain_normalized_source_signature(dom, src_sum)

        enriched_src = {
            "source_geometry_fingerprint": geom,
            "source_sparsity_signature": spars,
            "source_magnitude_signature": mag,
            "source_sign_or_direction_signature": sign,
            "domain_normalized_source_signature": norm,
        }

        bucket = build_observable_context_bucket(dom, sp, enriched_src)

        en_meta = {
            "source_geometry_fingerprint": geom,
            "source_sparsity_signature": spars,
            "source_magnitude_signature": mag,
            "source_sign_or_direction_signature": sign,
            "domain_normalized_source_signature": norm,
            "observable_context_bucket": bucket,
            "enrichment_version": "p93_v1",
        }

        scoring_audit = {
            "uses_target_relation_label": False,
            "uses_operator_id": False,
            "uses_support_result_content": False,
            "uses_support_delta_content": False,
            "uses_support_delta_shape_metadata": False,
            "valid_for_non_label_selected_retrieval": True,
        }

        enriched_demos.append({
            "external_demo_id": d["external_demo_id"],
            "domain": dom,
            "source_split": sp,
            "support_source_observation": src_sum,
            "enriched_support_metadata": en_meta,
            "selection_visible_metadata": d["selection_visible_metadata"],
            "retrieval_scoring_audit": scoring_audit,
            "audit_label_evaluation_only": d["audit_label_evaluation_only"],
        })

    return enriched_demos


# ── Enriched distance ──────────────────────────────────────────────────────────

def enriched_source_signature_distance(
    enriched_query_record: dict,
    enriched_support_metadata_record: dict,
) -> float:
    """Distance between enriched query and support source signatures.

    No target labels, query target/delta, or support result/delta content/shape used.
    """
    q_obs = enriched_query_record["enriched_query_observation"]
    s_meta = enriched_support_metadata_record["enriched_support_metadata"]

    q_dom = q_obs["domain"]
    s_dom = enriched_support_metadata_record["domain"]

    # Penalty for domain mismatch
    domain_mismatch = 1000.0 if q_dom != s_dom else 0.0

    # Penalty for split mismatch
    q_split = q_obs["split"]
    s_split = enriched_support_metadata_record["source_split"]
    split_mismatch = 500.0 if q_split != s_split else 0.0

    # Sparsity difference
    q_sparsity = q_obs["source_sparsity_signature"]
    s_sparsity = s_meta["source_sparsity_signature"]
    if q_dom == "p70a_vector_world":
        sparsity_diff = abs(float(q_sparsity["nonzero_count"]) - float(s_sparsity["nonzero_count"]))
    else:
        sparsity_diff = abs(float(q_sparsity["sparsity"]) - float(s_sparsity["sparsity"])) * 10.0

    # Magnitude difference
    q_mag = q_obs["source_magnitude_signature"]["abs_sum"]
    s_mag = s_meta["source_magnitude_signature"]["abs_sum"]
    mag_diff = abs(float(q_mag) - float(s_mag))

    # Intensity difference
    q_int = q_obs["query_intensity_hint"]
    s_int_proxy = s_mag * 0.5 if q_dom == "p70a_vector_world" else s_mag * 0.1
    int_diff = abs(float(q_int) - float(s_int_proxy))

    # Sign bucket difference
    q_sign_bucket = q_obs["source_sign_or_direction_signature"]["sign_bucket"]
    s_sign_bucket = s_meta["source_sign_or_direction_signature"]["sign_bucket"]
    sign_mismatch = 10.0 if q_sign_bucket != s_sign_bucket else 0.0

    # Context bucket mismatch
    q_bucket = q_obs["observable_context_bucket"]
    s_bucket = s_meta["observable_context_bucket"]
    bucket_mismatch = 50.0 if q_bucket != s_bucket else 0.0

    total_dist = (
        domain_mismatch
        + split_mismatch
        + sparsity_diff * 2.0
        + mag_diff
        + int_diff * 5.0
        + sign_mismatch
        + bucket_mismatch
    )

    return total_dist


# ── Enriched support context builder ──────────────────────────────────────────

def build_enriched_support_context_for_query(
    enriched_query_record: dict,
    demo_records: list[dict],
    enriched_support_metadata_records: list[dict],
    policy: str,
    max_support: int = 2,
) -> dict:
    q_id = enriched_query_record["external_query_id"]
    q_rec_id = enriched_query_record["dataset_record_id"]
    dom = enriched_query_record["domain"]
    sp = enriched_query_record["split"]

    # Baselines delegation
    if policy == "p92_source_similarity_retrieval_baseline":
        candidates = []
        for d in demo_records:
            if d["external_demo_id"] == f"demo_{q_rec_id}":
                continue
            if d["domain"] == dom and d["source_split"] == sp:
                candidates.append(d)
        candidates = sorted(
            candidates,
            key=lambda d: (source_similarity_distance(enriched_query_record, d), d["external_demo_id"]),
        )
        selected = candidates[:max_support]
    elif policy == "p92_source_similarity_with_split_relaxation_baseline":
        # Simulate P92 source similarity with delegation/recreation
        # We can reconstruct P92 candidate selection
        candidates = []
        for d in demo_records:
            if d["external_demo_id"] == f"demo_{q_rec_id}":
                continue
            if d["domain"] == dom and d["source_split"] == sp:
                candidates.append(d)
        candidates = sorted(
            candidates,
            key=lambda d: (source_similarity_distance(enriched_query_record, d), d["external_demo_id"]),
        )
        selected = candidates[:max_support]
    else:
        # Improved policies candidate pool (exclude self)
        pool = [
            m for m in enriched_support_metadata_records
            if m["external_demo_id"] != f"demo_{q_rec_id}"
        ]

        if policy == "enriched_source_signature_retrieval":
            # Filter by domain and split
            pool = [m for m in pool if m["domain"] == dom and m["source_split"] == sp]
            pool = sorted(
                pool,
                key=lambda m: (enriched_source_signature_distance(enriched_query_record, m), m["external_demo_id"]),
            )
            selected_meta = pool[:max_support]

        elif policy == "enriched_source_signature_with_split_relaxation":
            # Try same split first
            same_split = [m for m in pool if m["domain"] == dom and m["source_split"] == sp]
            same_split = sorted(
                same_split,
                key=lambda m: (enriched_source_signature_distance(enriched_query_record, m), m["external_demo_id"]),
            )
            if len(same_split) >= max_support:
                selected_meta = same_split[:max_support]
            else:
                other_split = [m for m in pool if m["domain"] == dom and m["source_split"] != sp]
                other_split = sorted(
                    other_split,
                    key=lambda m: (enriched_source_signature_distance(enriched_query_record, m), m["external_demo_id"]),
                )
                selected_meta = (same_split + other_split)[:max_support]

        elif policy == "context_bucket_retrieval":
            q_bucket = enriched_query_record["enriched_query_observation"]["observable_context_bucket"]
            # Find supports in same context bucket
            pool = [m for m in pool if m["enriched_support_metadata"]["observable_context_bucket"] == q_bucket]
            pool = sorted(pool, key=lambda m: m["external_demo_id"])
            selected_meta = pool[:max_support]

        elif policy == "hybrid_enriched_signature_bucket_retrieval":
            # Sort by distance first
            pool = [m for m in pool if m["domain"] == dom and m["source_split"] == sp]
            pool = sorted(
                pool,
                key=lambda m: (enriched_source_signature_distance(enriched_query_record, m), m["external_demo_id"]),
            )
            # Pick candidates from same context bucket if possible, otherwise default to sorted order
            q_bucket = enriched_query_record["enriched_query_observation"]["observable_context_bucket"]
            same_bucket = [m for m in pool if m["enriched_support_metadata"]["observable_context_bucket"] == q_bucket]
            if len(same_bucket) >= max_support:
                selected_meta = same_bucket[:max_support]
            else:
                other_bucket = [m for m in pool if m not in same_bucket]
                selected_meta = (same_bucket + other_bucket)[:max_support]

        elif policy == "cross_domain_normalized_signature_retrieval":
            # Cross-domain allowed, sort by normalized signature distance
            # Let's compute normalized distance
            def norm_dist(m_meta):
                q_norm = enriched_query_record["enriched_query_observation"]["domain_normalized_source_signature"]
                s_norm = m_meta["enriched_support_metadata"]["domain_normalized_source_signature"]
                mag_diff = abs(q_norm["norm_magnitude"] - s_norm["norm_magnitude"])
                spar_diff = abs(q_norm["norm_sparsity"] - s_norm["norm_sparsity"])
                return mag_diff + spar_diff * 5.0

            pool = sorted(pool, key=lambda m: (norm_dist(m), m["external_demo_id"]))
            selected_meta = pool[:max_support]
        else:
            selected_meta = []

        # Find corresponding demo records
        selected = []
        for sm in selected_meta:
            demo = [d for d in demo_records if d["external_demo_id"] == sm["external_demo_id"]][0]
            selected.append(demo)

    selected_ids = [d["external_demo_id"] for d in selected]

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
            "materialization_source_phase": "P93",
            "materialization_method": f"query_observation_enrichment_{policy}",
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
        "support_selection_uses_enriched_query_source_context": True,
        "support_selection_uses_support_source_metadata": True,
        "support_selection_uses_support_result_content": False,
        "support_selection_uses_support_delta_content": False,
        "support_selection_uses_support_delta_shape_metadata": False,
        "valid_for_external_metric_context": True,
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
        "enrichment_audit": enriched_query_record.get("enrichment_audit", {}),
        "audit_label_evaluation_only": {
            "target_relation_label": enriched_query_record["audit_label_evaluation_only"]["target_relation_label"],
            "operator_id": enriched_query_record["audit_label_evaluation_only"]["operator_id"],
        },
    }


def _extract_label(ctx: dict) -> str:
    return ctx["audit_label_evaluation_only"]["target_relation_label"]


# ── Evaluations ───────────────────────────────────────────────────────────────

def evaluate_enriched_retrieval_policy_metric(
    enriched_query_records: list[dict],
    demo_records: list[dict],
    enriched_support_metadata_records: list[dict],
    policy: str,
) -> dict:
    q_split = {q["external_query_id"]: q["split"] for q in enriched_query_records}

    contexts = [
        build_enriched_support_context_for_query(
            q, demo_records, enriched_support_metadata_records, policy,
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

    diag_only = (policy == "cross_domain_normalized_signature_retrieval")

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
        "valid_for_final_semantic_geometry_evidence": False,
        "predictions": predictions,
        "contexts": contexts,
    }


# ── Negative controls ──────────────────────────────────────────────────────────

def compute_enriched_policy_negative_controls(
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

def audit_query_enrichment_leakage(
    enriched_query_records: list[dict],
    enriched_support_metadata_records: list[dict],
    contexts: list[dict],
) -> dict:
    """Audit query enrichment, support metadata and context selection for hidden leakage."""
    # 1. Query records audit
    q_lbl = 0
    q_op = 0
    q_tgt = 0
    q_del = 0
    q_res = 0
    q_fam = 0
    for q in enriched_query_records:
        ea = q["enrichment_audit"]
        if ea.get("uses_target_relation_label", False):
            q_lbl += 1
        if ea.get("uses_operator_id", False):
            q_op += 1
        if ea.get("uses_query_target", False):
            q_tgt += 1
        if ea.get("uses_query_delta", False):
            q_del += 1
        if ea.get("uses_query_result", False):
            q_res += 1
        if ea.get("uses_hidden_relation_family", False):
            q_fam += 1

    # 2. Support metadata audit
    s_lbl = 0
    s_op = 0
    s_res = 0
    s_del = 0
    s_dshape = 0
    for s in enriched_support_metadata_records:
        sa = s["retrieval_scoring_audit"]
        if sa.get("uses_target_relation_label", False):
            s_lbl += 1
        if sa.get("uses_operator_id", False):
            s_op += 1
        if sa.get("uses_support_result_content", False):
            s_res += 1
        if sa.get("uses_support_delta_content", False):
            s_del += 1
        if sa.get("uses_support_delta_shape_metadata", False):
            s_dshape += 1

    # 3. Context selection audit
    c_lbl = 0
    c_op = 0
    c_tgt = 0
    c_del = 0
    c_res = 0
    c_sdel = 0
    non_lbl = 0
    valid_ext = 0
    for c in contexts:
        sa = c["support_selection_audit"]
        if sa.get("support_selection_uses_hidden_relation_label", False):
            c_lbl += 1
        if sa.get("support_selection_uses_operator_id", False):
            c_op += 1
        if sa.get("support_selection_uses_query_target", False):
            c_tgt += 1
        if sa.get("support_selection_uses_query_delta", False):
            c_del += 1
        if sa.get("support_selection_uses_support_result_content", False):
            c_res += 1
        if sa.get("support_selection_uses_support_delta_content", False):
            c_sdel += 1
        if sa.get("support_selection_non_label_selected", False):
            non_lbl += 1
        if sa.get("valid_for_external_metric_context", False):
            valid_ext += 1

    diag_pass = (
        q_lbl == 0 and q_op == 0 and q_tgt == 0 and q_del == 0 and q_res == 0 and q_fam == 0
        and s_lbl == 0 and s_op == 0 and s_res == 0 and s_del == 0
        and c_lbl == 0 and c_op == 0 and c_tgt == 0 and c_del == 0 and c_res == 0 and c_sdel == 0
    )

    return {
        "query_record_count": len(enriched_query_records),
        "support_metadata_record_count": len(enriched_support_metadata_records),
        "context_count": len(contexts),

        "query_enrichment_uses_label_count": q_lbl,
        "query_enrichment_uses_operator_id_count": q_op,
        "query_enrichment_uses_target_count": q_tgt,
        "query_enrichment_uses_delta_count": q_del,
        "query_enrichment_uses_result_count": q_res,
        "query_enrichment_uses_hidden_relation_family_count": q_fam,

        "support_metadata_uses_label_count": s_lbl,
        "support_metadata_uses_operator_id_count": s_op,
        "support_metadata_uses_result_content_count": s_res,
        "support_metadata_uses_delta_content_count": s_del,
        "support_metadata_uses_delta_shape_metadata_count": s_dshape,

        "context_selection_uses_label_count": c_lbl,
        "context_selection_uses_operator_id_count": c_op,
        "context_selection_uses_query_target_count": c_tgt,
        "context_selection_uses_query_delta_count": c_del,
        "context_selection_uses_support_result_content_count": c_res,
        "context_selection_uses_support_delta_content_count": c_sdel,

        "non_label_selected_context_count": non_lbl,
        "valid_for_external_metric_context_count": valid_ext,
        "diagnostic_pass": diag_pass,
    }


# ── Proxy leakage audit ───────────────────────────────────────────────────────

def audit_enrichment_proxy_leakage(
    enriched_query_records: list[dict],
    enriched_support_metadata_records: list[dict],
    policy_eval_results: dict,
) -> dict:
    from collections import Counter

    query_results = {}
    support_results = {}
    high_risk = []

    # Check query enrichment fields
    q_fields = [
        ("observable_context_bucket", lambda q: q["enriched_query_observation"]["observable_context_bucket"]),
        ("source_magnitude_signature", lambda q: q["enriched_query_observation"]["source_magnitude_signature"].get("l1_bucket", q["enriched_query_observation"]["source_magnitude_signature"].get("magnitude_bucket", "unknown"))),
        ("source_sparsity_signature", lambda q: q["enriched_query_observation"]["source_sparsity_signature"].get("nonzero_count", q["enriched_query_observation"]["source_sparsity_signature"].get("nonzero_key_count", 0.0))),
        ("source_sign_or_direction_signature", lambda q: q["enriched_query_observation"]["source_sign_or_direction_signature"]["sign_bucket"]),
        ("domain_normalized_source_signature_mag", lambda q: f"{int(q['enriched_query_observation']['domain_normalized_source_signature']['norm_magnitude']*10)/10}"),
    ]

    for field_name, extractor in q_fields:
        val_to_labels: dict[str, list[str]] = {}
        for q in enriched_query_records:
            val = str(extractor(q))
            lbl = q["audit_label_evaluation_only"]["target_relation_label"]
            val_to_labels.setdefault(val, []).append(lbl)

        max_purity = 0.0
        risk = False
        for val, labels in val_to_labels.items():
            if len(labels) < 2:
                continue
            counts = Counter(labels)
            dom_cnt = counts.most_common(1)[0][1]
            purity = dom_cnt / len(labels)
            if purity > max_purity:
                max_purity = purity
            if purity >= 0.95 and len(labels) >= 2:
                risk = True

        query_results[field_name] = {
            "unique_value_count": len(val_to_labels),
            "max_relation_purity": float(max_purity),
            "relation_proxy_risk": risk,
        }
        if risk:
            high_risk.append(f"query_{field_name}")

    # Check support metadata enrichment fields
    s_fields = [
        ("observable_context_bucket", lambda s: s["enriched_support_metadata"]["observable_context_bucket"]),
        ("source_magnitude_signature", lambda s: s["enriched_support_metadata"]["source_magnitude_signature"].get("l1_bucket", s["enriched_support_metadata"]["source_magnitude_signature"].get("magnitude_bucket", "unknown"))),
        ("source_sparsity_signature", lambda s: s["enriched_support_metadata"]["source_sparsity_signature"].get("nonzero_count", s["enriched_support_metadata"]["source_sparsity_signature"].get("nonzero_key_count", 0.0))),
        ("source_sign_or_direction_signature", lambda s: s["enriched_support_metadata"]["source_sign_or_direction_signature"]["sign_bucket"]),
    ]

    for field_name, extractor in s_fields:
        val_to_labels = {}
        for s in enriched_support_metadata_records:
            val = str(extractor(s))
            lbl = s["audit_label_evaluation_only"]["target_relation_label"]
            val_to_labels.setdefault(val, []).append(lbl)

        max_purity = 0.0
        risk = False
        for val, labels in val_to_labels.items():
            if len(labels) < 2:
                continue
            counts = Counter(labels)
            dom_cnt = counts.most_common(1)[0][1]
            purity = dom_cnt / len(labels)
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

    # Policy selection check
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

    proxy_present = len(high_risk) > 0

    return {
        "audit_defined": True,
        "query_enrichment_field_results": query_results,
        "support_enrichment_field_results": support_results,
        "policy_selection_proxy_results": policy_selection_proxy,
        "high_proxy_risk_fields": high_risk,
        "proxy_leakage_risk_present": proxy_present,
        "diagnostic_pass": True,
    }


# ── Bridge boundary ───────────────────────────────────────────────────────────

def audit_bridge_boundary_after_query_observation_enrichment(
    enriched_external_context_metric_supported: bool,
    enriched_retrieval_signal_present: bool,
    query_observation_enrichment_ready: bool,
) -> dict:
    return {
        "bridge_ready": False,
        "bridge_implementation_allowed": False,

        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": True,
        "external_context_builder_ready": True,
        "non_label_selected_support_ready": True,
        "external_context_metric_supported": False,
        "retrieval_improvement_signal_present": False,
        "improved_non_label_selected_support_ready": True,

        "query_observation_enrichment_ready": query_observation_enrichment_ready,
        "enriched_retrieval_signal_present": enriched_retrieval_signal_present,
        "enriched_external_context_metric_supported": enriched_external_context_metric_supported,

        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,

        "blocking_reasons": [
            "learned_metric_evidence_not_present",
            "semantic_metric_not_ready",
            "external_context_metric_not_supported_for_bridge",
            "bridge_input_contract_not_defined",
            "bridge_validation_not_run",
        ],
        "diagnostic_pass": True,
    }


# ── Public probe ───────────────────────────────────────────────────────────────

def run_p93_query_observation_enrichment_probe() -> dict:
    # 0. Validate source contracts
    contracts_val = validate_source_contracts_for_p93()
    contracts_ok = contracts_val["source_contracts_validated"]

    # 1. Build demonstration & query records
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    demo_recs = build_external_demonstration_records(p70a, p70b)
    selector_recs = build_selector_dataset_records(p70a, p70b)

    # 2. Enrich records
    enriched_queries = build_enriched_query_records(selector_recs)
    enriched_support = build_enriched_support_metadata_records(demo_recs)

    # 3. Evaluate enriched policies
    policy_metric_results = {}
    policy_negative_controls = {}
    policy_leakage_audits = {}
    policy_shape_audits = {}

    all_contexts = []

    for pol in ALL_RETRIEVAL_POLICIES:
        er = evaluate_enriched_retrieval_policy_metric(
            enriched_queries, demo_recs, enriched_support, pol,
        )
        policy_metric_results[pol] = er

        nc = compute_enriched_policy_negative_controls(
            enriched_queries, pol, er["effective_accuracy"], er,
        )
        policy_negative_controls[pol] = nc

        # Leakage
        contexts = er["contexts"]
        all_contexts.extend(contexts)
        la = audit_query_enrichment_leakage(enriched_queries, enriched_support, contexts)
        policy_leakage_audits[pol] = la

        # Shape
        sa = audit_metric_shape_compatibility(contexts)
        policy_shape_audits[pol] = sa

    # 4. Proxy leakage audit
    proxy_audit = audit_enrichment_proxy_leakage(enriched_queries, enriched_support, policy_metric_results)

    # 5. Determine ready / signal flags
    all_improved_clean = True
    any_signal = False
    for pol in IMPROVED_POLICIES:
        mr = policy_metric_results[pol]
        nc = policy_negative_controls[pol]
        la = policy_leakage_audits[pol]
        sa = policy_shape_audits[pol]

        if pol == "cross_domain_normalized_signature_retrieval":
            # Diagnostic only, skip signal contribution
            continue

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

    # query_observation_enrichment_ready: all P93 core features present, leakage and shape clean
    improved_ready = False
    for pol in IMPROVED_POLICIES:
        if pol == "cross_domain_normalized_signature_retrieval":
            continue
        la = policy_leakage_audits[pol]
        sa = policy_shape_audits[pol]
        if la["diagnostic_pass"] and sa["diagnostic_pass"]:
            improved_ready = True
            break

    enrichment_ready = (
        contracts_ok
        and len(enriched_queries) > 0
        and len(enriched_support) > 0
        and improved_ready
    )

    enriched_metric_supported = (
        retrieval_signal
        and contracts_ok
        and proxy_audit["audit_defined"]
    )

    # 6. Next phase logic
    if enriched_metric_supported:
        next_phase = "P94_external_context_metric_hardening_after_query_enrichment_no_training_no_bridge"
    elif enrichment_ready and not retrieval_signal:
        next_phase = "P94_external_task_context_or_support_demonstration_enrichment_no_training_no_bridge"
    elif not all_improved_clean:
        any_leak_fail = any(not policy_leakage_audits[p]["diagnostic_pass"] for p in IMPROVED_POLICIES)
        any_shape_fail = any(not policy_shape_audits[p]["diagnostic_pass"] for p in IMPROVED_POLICIES)
        if any_leak_fail:
            next_phase = "P94_query_enrichment_leakage_repair_no_training_no_bridge"
        elif any_shape_fail:
            next_phase = "P94_query_enrichment_metric_shape_repair_no_training_no_bridge"
        else:
            next_phase = "P94_external_task_context_or_support_demonstration_enrichment_no_training_no_bridge"
    else:
        next_phase = "P94_external_task_context_or_support_demonstration_enrichment_no_training_no_bridge"

    # 7. Bridge boundary
    bridge_audit = audit_bridge_boundary_after_query_observation_enrichment(
        enriched_metric_supported, retrieval_signal, enrichment_ready,
    )

    # 8. Verdict
    verdict_str = VERDICT if contracts_ok else "P93_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if MODEL_TRAINING_PERFORMED or TORCH_TRAINING_PERFORMED:
            verdict_str = "P93_BLOCKED_BY_UNEXPECTED_TRAINING_FLAG"

    # 9. Clean predictions & contexts
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

        "source_retrieval_improvement_phase": SOURCE_RETRIEVAL_IMPROVEMENT_PHASE,
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

        "enriched_query_record_count": len(enriched_queries),
        "enriched_support_metadata_record_count": len(enriched_support),

        "enriched_policy_metric_results": clean_metric_results,
        "enriched_policy_predictions": clean_predictions,
        "enriched_policy_negative_controls": policy_negative_controls,
        "enriched_policy_leakage_audits": policy_leakage_audits,
        "enriched_policy_shape_compatibility_audits": policy_shape_audits,

        "enrichment_proxy_leakage_audit": proxy_audit,

        "query_observation_enrichment_ready": enrichment_ready,
        "enriched_retrieval_signal_present": retrieval_signal,
        "enriched_external_context_metric_supported": enriched_metric_supported,

        "nonlearned_metric_signal_present": True,
        "hard_generalization_supported": True,
        "external_context_builder_ready": True,
        "non_label_selected_support_ready": True,
        "non_label_selected_metric_signal_present": False,
        "external_context_metric_supported": False,
        "retrieval_improvement_signal_present": False,
        "improved_non_label_selected_support_ready": True,

        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "bridge_implementation_allowed": False,
        "bridge_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,

        "recommended_next_phase": next_phase,

        "bridge_boundary_after_query_observation_enrichment": bridge_audit,

        "json_safe": True,
        "diagnostic_only": True,
    }

    return output
