# src/phase4/task_card_retrieval_policy_repair.py

import json
import math
import hashlib
from typing import Any

from src.phase4.real_task_card_support_retrieval_pilot import (
    validate_source_contracts_for_p96,
    run_p96_real_task_card_support_retrieval_pilot_probe,
    tokenize_public_text,
    token_overlap_distance,
    preserve_constraint_distance,
)

from src.phase4.real_external_task_context_dataset_contract import (
    build_task_card_catalog,
    build_query_task_card_records,
    build_support_demonstration_card_records,
)

from src.phase4.query_observation_enrichment import (
    build_enriched_query_records,
    build_enriched_support_metadata_records,
    enriched_source_signature_distance,
)

from src.phase4.external_support_context_builder import (
    build_external_demonstration_records,
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

PHASE = "P97"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Task Card Retrieval Policy Repair"
CONTRACT_VERSION = "phase4_p97_task_card_retrieval_policy_repair_v1"

SOURCE_REAL_TASK_CARD_RETRIEVAL_PILOT_PHASE = "P96"
SOURCE_REAL_TASK_CONTEXT_CONTRACT_PHASE = "P95"
SOURCE_EXTERNAL_TASK_CONTEXT_PHASE = "P94"
SOURCE_QUERY_OBSERVATION_ENRICHMENT_PHASE = "P93"
SOURCE_RETRIEVAL_IMPROVEMENT_PHASE = "P92"
SOURCE_EXTERNAL_CONTEXT_METRIC_PHASE = "P91"
SOURCE_EXTERNAL_CONTEXT_BUILDER_PHASE = "P90"
SOURCE_NONLEARNED_METRIC_PHASE = "P88"

TRAINING_ALLOWED_BY_PHASE4_AUTHORITY = True
MODEL_TRAINING_PERFORMED = False
TORCH_TRAINING_PERFORMED = False
NEW_MODEL_IMPLEMENTED = False
OPTIMIZER_CREATED = False
CHECKPOINT_WRITTEN = False

TYPED_OPERATOR_INTENT_GRAPH_DEFINED = True
AXIS_TAXONOMY_DEFINED = True
CHANGE_PRESERVE_SLOT_SCHEMA_DEFINED = True
DEMO_DELTA_COMPATIBILITY_DEFINED = True
REPAIRED_RETRIEVAL_POLICIES_DEFINED = True
REPAIR_LEAKAGE_AUDIT_DEFINED = True
REPAIR_PROXY_AUDIT_DEFINED = True

REAL_EXTERNAL_TASK_DATASET_CONTRACT_READY = True
VALID_EXTERNAL_TASK_CONTEXT_AVAILABLE = False
VALID_FOR_REAL_EXTERNAL_TASK_CONTEXT_EVIDENCE = False
READY_FOR_REAL_DATA_COLLECTION = True

TASK_CARD_RETRIEVAL_PIPELINE_READY = True
P96_TASK_CARD_RETRIEVAL_SIGNAL_PRESENT = False

REPAIRED_TASK_CARD_RETRIEVAL_SIGNAL_PRESENT = False  # computed
REPAIRED_TASK_CARD_EXTERNAL_METRIC_SUPPORTED = False
REPAIRED_RETRIEVAL_POLICY_READY = False  # computed

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "task_card_retrieval_policy_repair"
VERDICT = "P97_READY_FOR_REVIEW"

ALL_P97_POLICIES = [
    "p93_enriched_source_signature_baseline",
    "task_card_absent_baseline",
    "p96_public_token_overlap_baseline",
    "p96_change_preserve_baseline",
    "diagnostic_task_card_id_exact_match",
    "diagnostic_compatible_label_mapping_oracle",
    "typed_operator_intent_signature_retrieval",
    "typed_intent_plus_preserve_constraint_retrieval",
    "demo_visible_transformation_signature_retrieval",
    "typed_intent_demo_compatibility_retrieval",
    "hybrid_typed_intent_demo_source_retrieval",
    "preserve_constraints_only_retrieval",
    "change_axis_only_retrieval",
    "demo_caption_intent_only_retrieval",
]


# ── Static accepted P96 metadata ──────────────────────────────────────────────

ACCEPTED_P96_REPORT_METADATA = {
    "phase": "P96",
    "verdict": "P96_READY_FOR_REVIEW",

    "task_card_retrieval_pipeline_ready": True,
    "task_card_retrieval_signal_present": False,
    "task_card_external_metric_supported": False,

    "valid_external_task_context_available": False,
    "valid_for_real_external_task_context_evidence": False,
    "ready_for_real_data_collection": True,

    "diagnostic_task_card_id_exact_match_effective_accuracy": 0.6666666666666666,
    "diagnostic_compatible_label_mapping_oracle_effective_accuracy": 0.6666666666666666,

    "p93_enriched_source_signature_baseline_effective_accuracy": 0.3888888888888889,
    "task_card_absent_baseline_effective_accuracy": 0.3888888888888889,

    "task_card_public_token_overlap_retrieval_effective_accuracy": 0.2777777777777778,
    "task_card_change_preserve_signature_retrieval_effective_accuracy": 0.2777777777777778,
    "support_demo_public_caption_retrieval_effective_accuracy": 0.2777777777777778,
    "hybrid_task_card_plus_source_signature_retrieval_effective_accuracy": 0.1111111111111111,

    "public_policies_use_task_card_id": False,
    "public_policies_use_compatible_label_mapping": False,
    "proxy_risk_present": True,
    "assignment_is_diagnostic_label_mapped": True,

    "model_training_performed": False,
    "torch_training_performed": False,
    "new_model_implemented": False,
    "optimizer_created": False,
    "checkpoint_written": False,

    "learned_selector_evidence_present": False,
    "learned_metric_evidence_present": False,
    "semantic_metric_ready": False,
    "bridge_ready": False,

    "recommended_next_phase": "P97_task_card_retrieval_policy_repair_no_training_no_bridge",
    "metadata_source": "accepted_p96_report_static_metadata",
}


# ── Source contract validation ─────────────────────────────────────────────────

def validate_source_contracts_for_p97() -> dict:
    validated = True
    missing = []
    m = ACCEPTED_P96_REPORT_METADATA

    if m.get("phase") != "P96" or m.get("verdict") != "P96_READY_FOR_REVIEW":
        validated = False
        missing.append("p96_invalid_phase_or_verdict")
    if m.get("task_card_retrieval_pipeline_ready") is not True:
        validated = False
        missing.append("p96_pipeline_not_ready")
    if m.get("task_card_retrieval_signal_present") is not False:
        validated = False
        missing.append("p96_signal_present_should_be_false")
    if m.get("diagnostic_task_card_id_exact_match_effective_accuracy", 0.0) < 0.65:
        validated = False
        missing.append("p96_oracle_accuracy_mismatch")
    if m.get("valid_for_real_external_task_context_evidence") is not False:
        validated = False
        missing.append("p96_valid_real_evidence_should_be_false")
    if m.get("recommended_next_phase") != "P97_task_card_retrieval_policy_repair_no_training_no_bridge":
        validated = False
        missing.append("p96_recommended_phase_mismatch")

    return {
        "source_contracts_validated": validated,
        "p96_validated": validated,
        "p96_pipeline_ready": m.get("task_card_retrieval_pipeline_ready") is True,
        "p96_public_token_signal_absent": m.get("task_card_retrieval_signal_present") is False,
        "p96_diagnostic_oracle_upper_bound_exists": (m.get("diagnostic_task_card_id_exact_match_effective_accuracy", 0.0) > 0.65),
        "p96_public_policies_below_40": (m.get("task_card_public_token_overlap_retrieval_effective_accuracy", 1.0) < 0.40),
        "p96_real_evidence_false": m.get("valid_for_real_external_task_context_evidence") is False,
        "p96_public_policies_no_task_card_id": m.get("public_policies_use_task_card_id") is False,
        "p96_proxy_risk_true": m.get("proxy_risk_present") is True,
        "p96_bridge_false": m.get("bridge_ready") is False,
        "p96_recommends_p97": m.get("recommended_next_phase") == "P97_task_card_retrieval_policy_repair_no_training_no_bridge",
        "metadata_source": "accepted_p96_report_static_metadata",
        "missing_or_invalid": missing,
    }


# ── Typed Operator Intent Signature Schema ──────────────────────────────────────

def build_typed_operator_intent_signature(task_card: dict) -> dict:
    inst = task_card.get("human_readable_instruction", "").lower()
    axis_desc = task_card.get("change_intent", {}).get("change_axis_description", "").lower()
    prim_chg = task_card.get("change_intent", {}).get("primary_change", "").lower()

    # Intent family
    if "phase" in inst or "phase" in prim_chg:
        family = "phase_shift"
    elif "trend" in inst or "trend" in prim_chg:
        family = "trend_shift"
    elif "volatility" in inst or "volatility" in prim_chg:
        family = "volatility_modulation"
    elif "translate" in inst or "translate" in prim_chg or "offset" in inst or "offset" in prim_chg or "shift" in inst or "shift" in prim_chg:
        family = "translation"
    elif "scale" in inst or "scale" in prim_chg or "amplitude" in inst or "amplitude" in prim_chg:
        family = "scaling"
    elif "reflect" in inst or "reflect" in prim_chg or "reversal" in inst or "reversal" in prim_chg or "invert" in inst or "invert" in prim_chg:
        family = "reflection"
    elif "nonlinear" in inst or "nonlinear" in prim_chg or "distortion" in inst or "distortion" in prim_chg or "warp" in inst or "warp" in prim_chg:
        family = "nonlinear_distortion"
    elif "frequency" in inst or "frequency" in prim_chg:
        family = "frequency_modulation"
    else:
        family = "unknown"

    # Change axis type
    if "frequency" in inst or "frequency" in prim_chg:
        axis_type = "frequency"
    elif "phase" in inst or "phase" in prim_chg:
        axis_type = "temporal_phase"
    elif "volatility" in inst or "volatility" in prim_chg:
        axis_type = "volatility_envelope"
    elif "amplitude" in inst or "amplitude" in prim_chg:
        axis_type = "amplitude"
    elif "trend" in inst or "trend" in prim_chg or "slope" in inst or "slope" in prim_chg:
        axis_type = "trend_slope"
    elif "length" in inst or "length" in prim_chg:
        axis_type = "sequence_length"
    elif "warp" in inst or "warp" in prim_chg or "distortion" in inst or "distortion" in prim_chg:
        axis_type = "local_warp"
    elif "horizontal" in inst or "horizontal" in prim_chg or "coordinate" in axis_desc or "coordinate" in prim_chg:
        axis_type = "horizontal_coordinate"
    elif "vertical" in inst or "vertical" in prim_chg or "baseline" in axis_desc or "baseline" in prim_chg or "value" in inst or "value" in prim_chg:
        axis_type = "vertical_value"
    else:
        axis_type = "unknown"

    # Change direction
    if "increase" in inst or "increase" in prim_chg:
        direction = "increase"
    elif "decrease" in inst or "decrease" in prim_chg:
        direction = "decrease"
    elif "invert" in inst or "invert" in prim_chg or "reversal" in inst or "reversal" in prim_chg:
        direction = "invert"
    elif "shift" in inst or "shift" in prim_chg or "translate" in inst or "translate" in prim_chg:
        direction = "shift"
    elif "compress" in inst or "compress" in prim_chg or "stretch" in inst or "stretch" in prim_chg:
        direction = "compress_or_stretch"
    elif "nonlinear" in inst or "nonlinear" in prim_chg:
        direction = "nonlinear"
    else:
        direction = "unknown"

    # Geometry action
    if "translate" in inst or "translate" in prim_chg:
        geom = "translate"
    elif "scale" in inst or "scale" in prim_chg:
        geom = "scale"
    elif "reflect" in inst or "reflect" in prim_chg or "invert" in inst or "invert" in prim_chg:
        geom = "reflect"
    elif "warp" in inst or "warp" in prim_chg or "distortion" in inst or "distortion" in prim_chg:
        geom = "warp"
    else:
        geom = "none"

    # Temporal action
    if "reversal" in inst or "reversal" in prim_chg or "reflect_x" in prim_chg:
        temp = "reverse_order"
    elif "phase" in inst or "phase" in prim_chg:
        temp = "shift_phase"
    elif "frequency" in inst or "frequency" in prim_chg:
        temp = "modulate_frequency"
    else:
        temp = "none"

    # Value action
    if "constant offset" in inst or "baseline offset" in axis_desc or "basline" in inst or "value baseline" in axis_desc:
        val = "offset"
    elif "amplitude" in inst or "amplitude" in prim_chg or "volatility" in inst or "volatility" in prim_chg:
        val = "scale_deviation"
    elif "value inversion" in inst or "value inversion" in prim_chg:
        val = "invert_value"
    elif "trend" in inst or "trend" in prim_chg or "slope" in inst or "slope" in prim_chg:
        val = "slope_change"
    else:
        val = "none"

    # Locality type
    if "local" in inst or "local" in prim_chg or "warp" in inst or "warp" in prim_chg:
        loc = "local"
    elif "envelope" in inst or "envelope" in prim_chg or "volatility" in inst or "volatility" in prim_chg:
        loc = "envelope"
    else:
        loc = "global"

    # Periodicity type
    if "periodic" in inst or "periodic" in prim_chg or "frequency" in inst or "frequency" in prim_chg or "phase" in inst or "phase" in prim_chg:
        period = "periodic"
    else:
        period = "non_periodic"

    return {
        "intent_family": family,
        "change_axis_type": axis_type,
        "change_direction_type": direction,
        "geometry_action_type": geom,
        "temporal_action_type": temp,
        "value_action_type": val,
        "locality_type": loc,
        "periodicity_type": period,
        "preserve_vector": task_card.get("preserve_constraints", {}),
        "forbid_vector": {f: True for f in task_card.get("forbidden_changes", [])},
        "descriptor_evidence": {
            "primary_change": prim_chg,
            "change_axis_description": axis_desc,
        },
        "signature_audit": {
            "uses_task_card_id": False,
            "uses_hidden_relation_label": False,
            "uses_operator_id": False,
            "uses_compatible_label_mapping": False,
            "uses_query_target": False,
            "uses_query_delta": False,
            "valid_public_signature": True,
            "valid_for_real_evidence": False,
            "valid_for_dataset_contract_pilot": True,
        }
    }


# ── Support Demo Visible Transformation Signature Schema ───────────────────────

def build_support_demo_visible_transformation_signature(support_demo_card: dict) -> dict:
    pub = support_demo_card.get("support_demo_public_view", {})
    chg = pub.get("visible_change_summary", {})
    pres = pub.get("visible_preservation_summary", {})

    mag = chg.get("delta_magnitude", 0.0)
    dim = chg.get("delta_dimension", 0)

    # Scale type
    if mag > 1.5:
        scale_type = "large"
    elif mag > 0.5:
        scale_type = "medium"
    else:
        scale_type = "small"

    # Caption intent signature parsing
    caption = pub.get("human_demo_caption", "")
    caption_card = {
        "human_readable_instruction": caption,
        "change_intent": {"primary_change": caption, "change_axis_description": caption},
        "preserve_constraints": {k: True for k in pres.get("invariant_keys", [])},
        "forbidden_changes": [],
    }
    cap_sig = build_typed_operator_intent_signature(caption_card)

    task_sig = build_typed_operator_intent_signature(support_demo_card["support_task_card"])

    return {
        "demo_change_scale_type": scale_type,
        "demo_dimension_type": f"dim_{dim}",
        "demo_preservation_type": "invariant_overlap",
        "demo_caption_intent_signature": cap_sig,
        "demo_task_intent_signature": task_sig,
        "signature_audit": {
            "uses_external_demo_id": False,
            "uses_hidden_relation_label": False,
            "uses_operator_id": False,
            "uses_compatible_label_mapping": False,
            "uses_support_result_content_publicly": True,
            "uses_support_delta_content_publicly": True,
            "valid_public_demo_signature": True,
            "valid_for_real_evidence": False,
            "valid_for_dataset_contract_pilot": True,
        }
    }


# ── Structured Distance Calculators ───────────────────────────────────────────

def typed_slot_distance(query_sig: dict, support_sig: dict) -> float:
    # Weighted slot comparisons
    weights = {
        "intent_family": 0.20,
        "change_axis_type": 0.20,
        "change_direction_type": 0.10,
        "geometry_action_type": 0.10,
        "temporal_action_type": 0.10,
        "value_action_type": 0.10,
        "locality_type": 0.05,
        "periodicity_type": 0.05,
        "preserve_vector": 0.07,
        "forbid_vector": 0.03,
    }

    dist = 0.0
    for slot, w in weights.items():
        if slot in ("preserve_vector", "forbid_vector"):
            # Compare constraints dict
            q_v = query_sig.get(slot, {})
            s_v = support_sig.get(slot, {})
            dist += w * preserve_constraint_distance(q_v, s_v)
        else:
            if query_sig.get(slot) != support_sig.get(slot):
                dist += w

    return dist


def demo_compatibility_distance(query_sig: dict, support_demo_sig: dict) -> float:
    d_task = typed_slot_distance(query_sig, support_demo_sig["demo_task_intent_signature"])
    d_caption = typed_slot_distance(query_sig, support_demo_sig["demo_caption_intent_signature"])
    return 0.5 * d_task + 0.5 * d_caption


# ── Support Context Builder ───────────────────────────────────────────────────

def build_repaired_task_card_support_context_for_query(
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
    q_sig = build_typed_operator_intent_signature(q_card)

    uses_lbl = False
    uses_tc_id = False
    uses_lbl_map = False
    uses_pub_fields = False
    uses_typed_sig = False
    uses_demo_sig = False
    uses_pres = False
    uses_src_sig = False

    diag_only = False
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

    elif policy == "p96_public_token_overlap_baseline":
        uses_pub_fields = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                token_overlap_distance(q_card.get("public_descriptor_tokens", []), id_to_demo_card[d["external_demo_id"]]["support_task_card"].get("public_descriptor_tokens", [])),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "p96_change_preserve_baseline":
        uses_pub_fields = True
        uses_pres = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                token_overlap_distance(tokenize_public_text(q_card["change_intent"]["primary_change"]), tokenize_public_text(id_to_demo_card[d["external_demo_id"]]["support_task_card"]["change_intent"]["primary_change"]))
                + preserve_constraint_distance(q_card["preserve_constraints"], id_to_demo_card[d["external_demo_id"]]["support_task_card"]["preserve_constraints"]),
                d["external_demo_id"]
            )
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

    elif policy == "typed_operator_intent_signature_retrieval":
        uses_pub_fields = True
        uses_typed_sig = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                typed_slot_distance(q_sig, build_typed_operator_intent_signature(id_to_demo_card[d["external_demo_id"]]["support_task_card"])),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "typed_intent_plus_preserve_constraint_retrieval":
        uses_pub_fields = True
        uses_typed_sig = True
        uses_pres = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                typed_slot_distance(q_sig, build_typed_operator_intent_signature(id_to_demo_card[d["external_demo_id"]]["support_task_card"]))
                + 0.5 * preserve_constraint_distance(q_card["preserve_constraints"], id_to_demo_card[d["external_demo_id"]]["support_task_card"]["preserve_constraints"]),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "demo_visible_transformation_signature_retrieval":
        uses_pub_fields = True
        uses_demo_sig = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                typed_slot_distance(q_sig, build_support_demo_visible_transformation_signature(id_to_demo_card[d["external_demo_id"]])["demo_task_intent_signature"]),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "typed_intent_demo_compatibility_retrieval":
        uses_pub_fields = True
        uses_typed_sig = True
        uses_demo_sig = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                demo_compatibility_distance(q_sig, build_support_demo_visible_transformation_signature(id_to_demo_card[d["external_demo_id"]])),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "hybrid_typed_intent_demo_source_retrieval":
        uses_pub_fields = True
        uses_typed_sig = True
        uses_demo_sig = True
        uses_src_sig = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                0.45 * typed_slot_distance(q_sig, build_typed_operator_intent_signature(id_to_demo_card[d["external_demo_id"]]["support_task_card"]))
                + 0.35 * demo_compatibility_distance(q_sig, build_support_demo_visible_transformation_signature(id_to_demo_card[d["external_demo_id"]]))
                + 0.20 * enriched_source_signature_distance(enriched_query_record, id_to_enriched_meta[d["external_demo_id"]]),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "preserve_constraints_only_retrieval":
        uses_pub_fields = True
        uses_pres = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                preserve_constraint_distance(q_card["preserve_constraints"], id_to_demo_card[d["external_demo_id"]]["support_task_card"]["preserve_constraints"]),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "change_axis_only_retrieval":
        uses_pub_fields = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                token_overlap_distance(tokenize_public_text(q_card["change_intent"]["change_axis_description"]), tokenize_public_text(id_to_demo_card[d["external_demo_id"]]["support_task_card"]["change_intent"]["change_axis_description"])),
                d["external_demo_id"]
            )
        )
        selected = candidates[:max_support]

    elif policy == "demo_caption_intent_only_retrieval":
        uses_pub_fields = True
        uses_demo_sig = True
        val_pilot = True
        candidates = [
            d for d in demo_records
            if d["external_demo_id"] != f"demo_{q_rec_id}" and d["domain"] == dom and d["source_split"] == sp
        ]
        candidates = sorted(
            candidates,
            key=lambda d: (
                typed_slot_distance(q_sig, build_support_demo_visible_transformation_signature(id_to_demo_card[d["external_demo_id"]])["demo_caption_intent_signature"]),
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
            "materialization_source_phase": "P97",
            "materialization_method": f"repaired_task_card_retrieval_{policy}",
            "missing_result_summary": False,
            "missing_delta_summary": False,
        })

    inv = build_materialized_support_invariant_context(support_pairs)

    support_demo_sigs_used = [
        build_support_demo_visible_transformation_signature(id_to_demo_card[sid])
        for sid in selected_ids
    ]

    audit = {
        "support_selection_uses_task_card_id": uses_tc_id,
        "support_selection_uses_compatible_label_mapping": uses_lbl_map,
        "support_selection_uses_hidden_relation_label": uses_lbl,
        "support_selection_uses_operator_id": False,
        "support_selection_uses_query_target": False,
        "support_selection_uses_query_delta": False,
        "support_selection_uses_public_task_card_fields": uses_pub_fields,
        "support_selection_uses_typed_operator_intent_signature": uses_typed_sig,
        "support_selection_uses_demo_visible_signature": uses_demo_sig,
        "support_selection_uses_preserve_constraints": uses_pres,
        "support_selection_uses_source_signature": uses_src_sig,
        "diagnostic_only": diag_only,
        "valid_for_dataset_contract_pilot": val_pilot,
        "valid_for_real_external_task_context_evidence": False,
        "valid_for_final_semantic_geometry_evidence": False,
        "valid_for_bridge_evidence": False,
    }

    return {
        "external_support_context_id": f"ctx_p97_{policy}_{q_id}",
        "policy": policy,
        "query_id": q_id,
        "support_demo_ids": selected_ids,
        "support_context": {
            "support_pairs": support_pairs,
            "support_invariant_context": inv,
        },
        "query_typed_intent_signature_used": q_sig,
        "support_demo_signatures_used": support_demo_sigs_used,
        "support_selection_audit": audit,
        "audit_label_evaluation_only": {
            "target_relation_label": enriched_query_record["audit_label_evaluation_only"]["target_relation_label"],
            "operator_id": enriched_query_record["audit_label_evaluation_only"]["operator_id"],
        },
    }


# ── Evaluations ───────────────────────────────────────────────────────────────

def _extract_label(ctx: dict) -> str:
    return ctx["audit_label_evaluation_only"]["target_relation_label"]


def evaluate_repaired_task_card_retrieval_policy_metric(
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
        build_repaired_task_card_support_context_for_query(
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

def compute_repaired_task_card_policy_negative_controls(
    enriched_query_records: list[dict],
    query_task_card_records: list[dict],
    demo_records: list[dict],
    support_demo_card_records: list[dict],
    enriched_support_metadata_records: list[dict],
    policy: str,
    normal_eff_acc: float,
    eval_result: dict,
) -> dict:
    q_split = {q["external_query_id"]: q["split"] for q in enriched_query_records}
    contexts = eval_result["contexts"]

    train_ctx = [c for c in contexts if q_split.get(c["query_id"]) == "train"]
    test_ctx = [c for c in contexts if q_split.get(c["query_id"]) == "test"]

    prototypes = build_class_prototypes_from_contexts(train_ctx)

    # 1. Shuffled control
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

    # 2. Zero control
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

    # 3. Typed Signature Shuffle Control
    # Permute query typed intent signatures across queries (shift them relative to query task cards)
    tc_map = {tc["external_query_id"]: tc for tc in query_task_card_records}
    m = len(enriched_query_records)
    sig_shift = 5 % m if m > 0 else 0

    shuffled_queries = enriched_query_records[sig_shift:] + enriched_query_records[:sig_shift]
    shuffled_tc_records = query_task_card_records[sig_shift:] + query_task_card_records[:sig_shift]

    shuf_sig_contexts = []
    for orig_q, shuf_tc in zip(enriched_query_records, shuffled_tc_records):
        orig_tc = tc_map[orig_q["external_query_id"]]
        # Swap query task card record
        shuf_ctx = build_repaired_task_card_support_context_for_query(
            orig_q, shuf_tc, demo_records, support_demo_card_records, enriched_support_metadata_records, policy,
        )
        shuf_sig_contexts.append(shuf_ctx)

    shuf_sig_train = [c for c in shuf_sig_contexts if q_split.get(c["query_id"]) == "train"]
    shuf_sig_test = [c for c in shuf_sig_contexts if q_split.get(c["query_id"]) == "test"]

    shuf_sig_prototypes = build_class_prototypes_from_contexts(shuf_sig_train)

    shuf_sig_correct = 0
    shuf_sig_total = 0
    for ctx in shuf_sig_test:
        true_lbl = _extract_label(ctx)
        true_idx = CLASS_TO_IDX.get(true_lbl, -1)
        if true_idx == -1 or true_idx not in shuf_sig_prototypes:
            continue
        shuf_sig_total += 1
        vec = encode_external_support_delta_context(ctx)
        pred = predict_nearest_prototype(vec, shuf_sig_prototypes)
        if pred["predicted_class_idx"] == true_idx:
            shuf_sig_correct += 1
    shuf_sig_acc = float(shuf_sig_correct / shuf_sig_total) if shuf_sig_total > 0 else 0.0

    return {
        "support_policy": policy,
        "normal_effective_accuracy": normal_eff_acc,
        "shuffled_support_delta_negative_control": shuf_acc,
        "zero_delta_negative_control": zero_acc,
        "typed_signature_shuffle_control": shuf_sig_acc,
        "beats_shuffled_by": float(normal_eff_acc - shuf_acc),
        "beats_zero_delta_by": float(normal_eff_acc - zero_acc),
        "typed_signature_shuffle_degradation": float(normal_eff_acc - shuf_sig_acc),
        "shuffled_control_target_labels_preserved": True,
        "shuffled_control_support_deltas_permuted": True,
        "target_labels_shifted": False,
        "true_labels_preserved": True,
        "query_source_only_negative_control_applicable": False,
    }


# ── Leakage audit ─────────────────────────────────────────────────────────────

def audit_repaired_task_card_retrieval_leakage(contexts: list[dict]) -> dict:
    lbl_used = 0
    op_used = 0
    q_tgt_used = 0
    q_del_used = 0
    q_res_used = 0

    tc_id_used = 0
    lbl_map_used = 0
    pub_fields_used = 0
    typed_sig_used = 0
    demo_sig_used = 0
    pres_used = 0
    src_sig_used = 0

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
        if sa.get("support_selection_uses_compatible_label_mapping", False):
            lbl_map_used += 1
        if sa.get("support_selection_uses_public_task_card_fields", False):
            pub_fields_used += 1
        if sa.get("support_selection_uses_typed_operator_intent_signature", False):
            typed_sig_used += 1
        if sa.get("support_selection_uses_demo_visible_signature", False):
            demo_sig_used += 1
        if sa.get("support_selection_uses_preserve_constraints", False):
            pres_used += 1
        if sa.get("support_selection_uses_source_signature", False):
            src_sig_used += 1

        if sa.get("diagnostic_only", False):
            diag_count += 1
        if sa.get("valid_for_dataset_contract_pilot", False):
            pilot_count += 1
        if sa.get("valid_for_real_external_task_context_evidence", False):
            val_real_count += 1

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
        "compatible_label_mapping_used_count": lbl_map_used,
        "public_task_card_fields_used_count": pub_fields_used,
        "typed_operator_intent_signature_used_count": typed_sig_used,
        "demo_visible_signature_used_count": demo_sig_used,
        "preserve_constraints_used_count": pres_used,
        "source_signature_used_count": src_sig_used,

        "diagnostic_context_count": diag_count,
        "dataset_contract_pilot_context_count": pilot_count,
        "valid_real_evidence_context_count": val_real_count,

        "diagnostic_pass": diag_pass,
    }


# ── Proxy audit ───────────────────────────────────────────────────────────────

def audit_repaired_task_card_retrieval_proxy_risk(policy_results: dict) -> dict:
    risk = True
    return {
        "audit_defined": True,
        "proxy_risk_present": risk,
        "assignment_is_diagnostic_label_mapped": True,
        "typed_operator_signature_inherits_assignment_proxy": True,
        "public_policy_signal_would_be_diagnostic_only": True,
        "real_data_required_to_remove_proxy": True,
        "diagnostic_pass": True,
    }


# ── Bridge boundary ───────────────────────────────────────────────────────────

def audit_bridge_boundary_after_repaired_task_card_retrieval(
    repaired_policy_ready: bool,
    repaired_diagnostic_signal_present: bool,
) -> dict:
    return {
        "bridge_ready": False,
        "bridge_implementation_allowed": False,
        "repaired_retrieval_policy_ready": repaired_policy_ready,
        "repaired_diagnostic_signal_present": repaired_diagnostic_signal_present,
        "valid_for_real_external_task_context_evidence": False,
        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,
        "blocking_reasons": [
            "p97_is_diagnostic_retrieval_repair_not_real_external_evidence",
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

def run_p97_task_card_retrieval_policy_repair_probe() -> dict:
    # 0. Validate P96 source contracts
    contracts_val = validate_source_contracts_for_p96()
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

    for pol in ALL_P97_POLICIES:
        er = evaluate_repaired_task_card_retrieval_policy_metric(
            enriched_queries, query_task_card_records, demo_recs, support_demo_card_records, enriched_support, pol,
        )
        policy_metric_results[pol] = er

        nc = compute_repaired_task_card_policy_negative_controls(
            enriched_queries, query_task_card_records, demo_recs, support_demo_card_records, enriched_support, pol, er["effective_accuracy"], er,
        )
        policy_negative_controls[pol] = nc

        contexts = er["contexts"]
        la = audit_repaired_task_card_retrieval_leakage(contexts)
        policy_leakage_audits[pol] = la

        sa = audit_metric_shape_compatibility(contexts)
        policy_shape_audits[pol] = sa

    # 3. Audits
    proxy_audit = audit_repaired_task_card_retrieval_proxy_risk(policy_metric_results)

    # 4. Pipeline readiness
    repaired_ready = (
        contracts_ok
        and len(task_card_catalog) == 10
        and len(query_task_card_records) > 0
        and len(support_demo_card_records) > 0
        and all(policy_metric_results[p]["sample_count"] > 0 for p in ALL_P97_POLICIES)
    )

    # Check for repaired public structured policies signal crossing threshold
    repaired_sig = False
    for pol in ALL_P97_POLICIES:
        mr = policy_metric_results[pol]
        nc = policy_negative_controls[pol]
        la = policy_leakage_audits[pol]
        sa = policy_shape_audits[pol]

        if mr["diagnostic_only"]:
            continue

        if (
            mr["effective_accuracy"] >= 0.40
            and mr["coverage"] >= 0.80
            and nc["beats_shuffled_by"] >= 0.10
            and nc["beats_zero_delta_by"] >= 0.10
            and nc["typed_signature_shuffle_degradation"] >= 0.10
            and la["diagnostic_pass"]
            and sa["diagnostic_pass"]
        ):
            repaired_sig = True

    # 5. Next phase logic
    # Set default recommendations
    next_phase = "P98_task_card_schema_enrichment_or_controlled_real_data_seed_no_training_no_bridge"
    for pol in ALL_P97_POLICIES:
        la = policy_leakage_audits[pol]
        if not la["diagnostic_pass"]:
            next_phase = "P98_repaired_task_card_retrieval_leakage_repair_no_training_no_bridge"

    if next_phase != "P98_repaired_task_card_retrieval_leakage_repair_no_training_no_bridge":
        if repaired_sig:
            next_phase = "P98_real_task_card_raw_data_collection_contract_no_training_no_bridge"

    # 6. Bridge boundary
    bridge_audit = audit_bridge_boundary_after_repaired_task_card_retrieval(
        repaired_ready, repaired_sig,
    )

    # 7. Clean outputs
    clean_metric_results = {}
    clean_predictions = {}
    for pol in ALL_P97_POLICIES:
        mr = policy_metric_results[pol]
        clean_predictions[pol] = mr["predictions"]
        clean_metric_results[pol] = {
            k: v for k, v in mr.items() if k not in ("predictions", "contexts")
        }

    verdict_str = VERDICT if contracts_ok else "P97_BLOCKED_BY_SOURCE_CONTRACT"

    return {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,
        "verdict": verdict_str,

        "source_real_task_card_retrieval_pilot_phase": SOURCE_REAL_TASK_CARD_RETRIEVAL_PILOT_PHASE,
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

        "repaired_retrieval_policy_ready": repaired_ready,
        "repaired_task_card_retrieval_signal_present": repaired_sig,
        "repaired_task_card_external_metric_supported": False,

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

        "bridge_boundary_after_repaired_task_card_retrieval": bridge_audit,

        "json_safe": True,
        "diagnostic_only": True,
    }
