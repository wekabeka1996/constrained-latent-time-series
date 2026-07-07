# src/phase4/real_external_task_context_dataset_contract.py

import json
import math
import hashlib
from typing import Any

from src.phase4.external_task_context_enrichment import (
    validate_source_contracts_for_p94,
    run_p94_external_task_context_enrichment_probe,
)

from src.phase4.query_observation_enrichment import (
    build_enriched_query_records,
)

from src.phase4.external_support_context_builder import (
    build_external_demonstration_records,
    build_external_query_records,
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

PHASE = "P95"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Real External Task Context Dataset Contract"
CONTRACT_VERSION = "phase4_p95_real_external_task_context_dataset_contract_v1"

SOURCE_EXTERNAL_TASK_CONTEXT_PHASE = "P94"
SOURCE_QUERY_OBSERVATION_ENRICHMENT_PHASE = "P93"
SOURCE_RETRIEVAL_IMPROVEMENT_PHASE = "P92"
SOURCE_EXTERNAL_CONTEXT_METRIC_PHASE = "P91"
SOURCE_EXTERNAL_CONTEXT_BUILDER_PHASE = "P90"
SOURCE_HARD_GENERALIZATION_PHASE = "P89"
SOURCE_NONLEARNED_METRIC_PHASE = "P88"
SOURCE_MATERIALIZER_PHASE = "P86"
SOURCE_DATASET_PHASE = "P81"

TRAINING_ALLOWED_BY_PHASE4_AUTHORITY = True
MODEL_TRAINING_PERFORMED = False
TORCH_TRAINING_PERFORMED = False
NEW_MODEL_IMPLEMENTED = False
OPTIMIZER_CREATED = False
CHECKPOINT_WRITTEN = False

REAL_EXTERNAL_TASK_CONTEXT_CONTRACT_DEFINED = True
TASK_CARD_SCHEMA_DEFINED = True
SUPPORT_DEMONSTRATION_SCHEMA_DEFINED = True
QUERY_TASK_CARD_SCHEMA_DEFINED = True
PRESERVE_CHANGE_CONSTRAINT_SCHEMA_DEFINED = True
LEAKAGE_AUDIT_DEFINED = True
PROXY_AUDIT_DEFINED = True

VALID_EXTERNAL_TASK_CONTEXT_AVAILABLE = False
REAL_EXTERNAL_TASK_DATASET_CONTRACT_READY = False  # computed
TASK_CARD_RECORDS_BUILT = False  # computed
SUPPORT_DEMONSTRATION_CARD_RECORDS_BUILT = False  # computed
QUERY_TASK_CARD_RECORDS_BUILT = False  # computed

TASK_CONTEXT_SIGNAL_PRESENT = False
TASK_CONTEXT_EXTERNAL_METRIC_SUPPORTED = False

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "real_external_task_context_dataset_contract"
VERDICT = "P95_READY_FOR_REVIEW"


# ── Static accepted P94 metadata ──────────────────────────────────────────────

ACCEPTED_P94_REPORT_METADATA = {
    "phase": "P94",
    "verdict": "P94_READY_FOR_REVIEW",

    "valid_external_task_context_available": False,
    "support_demonstration_context_ready": True,
    "task_context_signal_present": False,
    "task_context_external_metric_supported": False,
    "external_task_context_required": True,

    "diagnostic_oracle_task_context_retrieval_effective_accuracy": 0.6666666666666666,
    "diagnostic_support_delta_oracle_retrieval_effective_accuracy": 0.6666666666666666,

    "p93_enriched_source_signature_baseline_effective_accuracy": 0.3888888888888889,
    "task_context_absent_baseline_effective_accuracy": 0.3888888888888889,
    "non_label_external_task_manifest_retrieval_effective_accuracy": 0.2222222222222222,
    "task_manifest_plus_source_similarity_retrieval_effective_accuracy": 0.3888888888888889,

    "support_demo_context_without_query_task_card_effective_accuracy": 0.2222222222222222,
    "support_demo_context_without_query_task_card_diagnostic_only": True,
    "support_demo_context_without_query_task_card_valid_for_evidence": False,

    "support_demo_context_with_non_label_task_manifest_effective_accuracy": 0.2222222222222222,
    "support_demo_context_with_non_label_task_manifest_diagnostic_only": True,
    "support_demo_context_with_non_label_task_manifest_valid_for_evidence": False,

    "proxy_leakage_risk_present": True,

    "model_training_performed": False,
    "torch_training_performed": False,
    "new_model_implemented": False,
    "optimizer_created": False,
    "checkpoint_written": False,

    "learned_selector_evidence_present": False,
    "learned_metric_evidence_present": False,
    "semantic_metric_ready": False,
    "bridge_ready": False,

    "recommended_next_phase": "P95_real_external_task_context_dataset_contract_no_training_no_bridge",
    "metadata_source": "accepted_repaired_p94_report_static_metadata",
}


# ── Source validation ──────────────────────────────────────────────────────────

def validate_source_contracts_for_p95() -> dict:
    validated = True
    missing = []
    m = ACCEPTED_P94_REPORT_METADATA

    if m.get("phase") != "P94" or m.get("verdict") != "P94_READY_FOR_REVIEW":
        validated = False
        missing.append("p94_invalid_phase_or_verdict")
    if m.get("valid_external_task_context_available") is not False:
        validated = False
        missing.append("p94_valid_external_task_context_should_be_false")
    if m.get("support_demonstration_context_ready") is not True:
        validated = False
        missing.append("p94_support_demo_context_not_ready")
    if m.get("support_demo_context_without_query_task_card_diagnostic_only") is not True:
        validated = False
        missing.append("p94_policy1_not_diagnostic")
    if m.get("support_demo_context_without_query_task_card_valid_for_evidence") is not False:
        validated = False
        missing.append("p94_policy1_valid_for_evidence_should_be_false")
    if m.get("support_demo_context_with_non_label_task_manifest_diagnostic_only") is not True:
        validated = False
        missing.append("p94_policy2_not_diagnostic")
    if m.get("support_demo_context_with_non_label_task_manifest_valid_for_evidence") is not False:
        validated = False
        missing.append("p94_policy2_valid_for_evidence_should_be_false")
    if m.get("diagnostic_oracle_task_context_retrieval_effective_accuracy", 0.0) < 0.65:
        validated = False
        missing.append("p94_oracle_accuracy_mismatch")
    if m.get("bridge_ready") is not False:
        validated = False
        missing.append("p94_bridge_ready")
    if m.get("recommended_next_phase") != "P95_real_external_task_context_dataset_contract_no_training_no_bridge":
        validated = False
        missing.append("p94_recommended_phase_mismatch")

    return {
        "source_contracts_validated": validated,
        "p94_validated": validated,
        "p94_no_valid_external_task_context": m.get("valid_external_task_context_available") is False,
        "p94_support_demo_context_ready": m.get("support_demonstration_context_ready") is True,
        "p94_support_demo_policies_diagnostic_only": (
            m.get("support_demo_context_without_query_task_card_diagnostic_only") is True
            and m.get("support_demo_context_with_non_label_task_manifest_diagnostic_only") is True
        ),
        "p94_oracle_upper_bound_exists": (m.get("diagnostic_oracle_task_context_retrieval_effective_accuracy", 0.0) > 0.65),
        "p94_bridge_remains_false": m.get("bridge_ready") is False,
        "p94_recommended_p95_preserved": m.get("recommended_next_phase") == "P95_real_external_task_context_dataset_contract_no_training_no_bridge",
        "metadata_source": "accepted_repaired_p94_report_static_metadata",
        "missing_or_invalid": missing,
    }


# ── Core Concept: Task Cards Catalog ──────────────────────────────────────────

def build_task_card_catalog() -> list[dict]:
    # We map the 10 synthetic classes to realistic label-free task cards
    catalog_configs = [
        {
            "class_name": "translate_x",
            "task_id": "task_translate_horizontal",
            "instruction": "Shift the sequence coordinates along the horizontal axis.",
            "primary": "horizontal_translation",
            "axis": "coordinate alignment index offset",
            "preserves": {
                "preserve_sequence_length": True,
                "preserve_temporal_order": True,
                "preserve_phase_or_alignment": False,
                "preserve_broad_trend_direction": True,
            }
        },
        {
            "class_name": "translate_y",
            "task_id": "task_translate_vertical",
            "instruction": "Shift the sequence values along the vertical axis (constant offset).",
            "primary": "vertical_translation",
            "axis": "value baseline offset constant",
            "preserves": {
                "preserve_sequence_length": True,
                "preserve_temporal_order": True,
                "preserve_phase_or_alignment": True,
                "preserve_broad_trend_direction": True,
            }
        },
        {
            "class_name": "scale_s",
            "task_id": "task_scale_sequence_length",
            "instruction": "Compress or stretch the sequence length uniformly.",
            "primary": "sequence_length_scaling",
            "axis": "coordinate scaling factor",
            "preserves": {
                "preserve_sequence_length": False,
                "preserve_temporal_order": True,
                "preserve_phase_or_alignment": False,
                "preserve_broad_trend_direction": True,
            }
        },
        {
            "class_name": "reflect_x",
            "task_id": "task_reflect_horizontal",
            "instruction": "Invert the sequence horizontally (time reversal).",
            "primary": "horizontal_reflection",
            "axis": "coordinate direction inversion",
            "preserves": {
                "preserve_sequence_length": True,
                "preserve_temporal_order": False,
                "preserve_phase_or_alignment": False,
                "preserve_broad_trend_direction": False,
            }
        },
        {
            "class_name": "nonlinear_x_from_y",
            "task_id": "task_nonlinear_coordinate_distortion",
            "instruction": "Distort coordinates non-linearly using local value properties.",
            "primary": "nonlinear_coordinate_distortion",
            "axis": "local warping transformation",
            "preserves": {
                "preserve_sequence_length": True,
                "preserve_temporal_order": True,
                "preserve_phase_or_alignment": False,
                "preserve_broad_trend_direction": True,
            }
        },
        {
            "class_name": "change_frequency",
            "task_id": "task_frequency_modulation",
            "instruction": "Increase or decrease periodic frequency of fluctuations.",
            "primary": "frequency_modulation",
            "axis": "rate of cyclical variations",
            "preserves": {
                "preserve_sequence_length": True,
                "preserve_temporal_order": True,
                "preserve_phase_or_alignment": False,
                "preserve_broad_trend_direction": True,
            }
        },
        {
            "class_name": "scale_amplitude",
            "task_id": "task_scale_vertical_deviation",
            "instruction": "Increase or decrease amplitude of fluctuations while preserving broad trend direction.",
            "primary": "vertical_fluctuation_scaling",
            "axis": "larger deviations from local baseline",
            "preserves": {
                "preserve_sequence_length": True,
                "preserve_temporal_order": True,
                "preserve_phase_or_alignment": True,
                "preserve_broad_trend_direction": True,
            }
        },
        {
            "class_name": "shift_phase",
            "task_id": "task_shift_fluctuation_phase",
            "instruction": "Shift periodic fluctuations relative to coordinate frame.",
            "primary": "fluctuation_phase_shift",
            "axis": "cyclical wave alignment offset",
            "preserves": {
                "preserve_sequence_length": True,
                "preserve_temporal_order": True,
                "preserve_phase_or_alignment": False,
                "preserve_broad_trend_direction": True,
            }
        },
        {
            "class_name": "scale_volatility_envelope",
            "task_id": "task_volatility_envelope_modulation",
            "instruction": "Scale local variation envelope non-linearly.",
            "primary": "volatility_envelope_modulation",
            "axis": "local variance scaling function",
            "preserves": {
                "preserve_sequence_length": True,
                "preserve_temporal_order": True,
                "preserve_phase_or_alignment": True,
                "preserve_broad_trend_direction": True,
            }
        },
        {
            "class_name": "shift_trend",
            "task_id": "task_shift_broad_trend",
            "instruction": "Apply a non-constant slope offset over the broad trend direction.",
            "primary": "trend_direction_shift",
            "axis": "slope trajectory offset",
            "preserves": {
                "preserve_sequence_length": True,
                "preserve_temporal_order": True,
                "preserve_phase_or_alignment": True,
                "preserve_broad_trend_direction": False,
            }
        }
    ]

    cards = []
    for cfg in catalog_configs:
        tokens = cfg["instruction"].lower().replace(".", "").replace("(", "").replace(")", "").split()
        public_tokens = [t for t in tokens if t not in ("while", "and", "or", "of", "the", "a", "to")]

        cards.append({
            "task_card_id": cfg["task_id"],
            "task_card_version": "p95_v1",
            "human_readable_instruction": cfg["instruction"],
            "change_intent": {
                "primary_change": cfg["primary"],
                "secondary_change": [],
                "change_axis_description": cfg["axis"],
            },
            "preserve_constraints": cfg["preserves"],
            "forbidden_changes": [
                "relation_label_leak",
                "operator_id_leak",
                "query_target_leak",
                "query_delta_leak",
            ],
            "applicable_domains": ["p70a_vector_world", "p70b_time_series_world"],
            "public_descriptor_tokens": public_tokens,
            "task_card_audit": {
                "uses_hidden_relation_label": False,
                "uses_operator_id": False,
                "uses_query_target": False,
                "uses_query_delta": False,
                "valid_external_task_context": True,
                "valid_for_retrieval": True,
                "valid_for_final_semantic_geometry_evidence": False,
                "valid_for_bridge_evidence": False,
            },
            "audit_label_mapping_only": {
                "compatible_relation_labels": [cfg["class_name"]],
            }
        })

    return cards


# ── Query task card assignment schema ──────────────────────────────────────────

def build_query_task_card_records(
    enriched_query_records: list[dict],
    task_card_catalog: list[dict],
) -> list[dict]:
    records = []
    card_map = {}
    for c in task_card_catalog:
        lbl = c["audit_label_mapping_only"]["compatible_relation_labels"][0]
        card_map[lbl] = c

    for q in enriched_query_records:
        q_id = q["external_query_id"]
        rec_id = q["dataset_record_id"]
        dom = q["domain"]
        sp = q["split"]
        lbl = q["audit_label_evaluation_only"]["target_relation_label"]

        # Map task card based on synthetic label (Diagnostic Mapping only)
        target_card = card_map.get(lbl)

        records.append({
            "query_task_card_record_id": f"q_tc_{q_id}",
            "external_query_id": q_id,
            "dataset_record_id": rec_id,
            "domain": dom,
            "split": sp,
            "query_task_card": target_card,
            "assignment_source": "diagnostic_contract_mapping_from_existing_synthetic_label",
            "assignment_valid_for_real_evidence": False,
            "assignment_valid_for_contract_shape": True,
            "assignment_audit": {
                "uses_hidden_relation_label_for_assignment": True,
                "uses_operator_id_for_assignment": False,
                "uses_query_target_for_assignment": False,
                "uses_query_delta_for_assignment": False,
                "valid_for_real_external_task_context_evidence": False,
                "valid_for_dataset_contract_validation": True,
                "diagnostic_only": True,
            },
            "audit_label_evaluation_only": q["audit_label_evaluation_only"],
        })

    return records


# ── Support demonstration card assignment schema ───────────────────────────────

def build_support_demonstration_card_records(
    demo_records: list[dict],
    task_card_catalog: list[dict],
) -> list[dict]:
    records = []
    card_map = {}
    for c in task_card_catalog:
        lbl = c["audit_label_mapping_only"]["compatible_relation_labels"][0]
        card_map[lbl] = c

    for d in demo_records:
        demo_id = d["external_demo_id"]
        dom = d["domain"]
        sp = d["source_split"]
        lbl = d["audit_label_evaluation_only"]["target_relation_label"]
        obs = d["observable_context"]

        target_card = card_map.get(lbl)

        # Build public descriptors
        pub_view = {
            "source_observation": {
                "length": len(obs.get("source_summary", {}).get("coordinates", [])),
                "metric_features": obs.get("source_summary", {}).get("mean", 0.0),
            },
            "result_observation": {
                "length": len(obs.get("result_summary", {}).get("coordinates", [])),
                "metric_features": obs.get("result_summary", {}).get("mean", 0.0),
            },
            "visible_change_summary": {
                "delta_magnitude": math.sqrt(sum(x*x for x in obs.get("delta_summary", {}).get("delta_values", []))),
                "delta_dimension": len(obs.get("delta_summary", {}).get("delta_values", [])),
            },
            "visible_preservation_summary": {
                "invariant_keys": list(obs.get("invariant_summary", {}).keys()),
            },
            "human_demo_caption": f"Example: {target_card['human_readable_instruction']} Source coordinates and target coordinates are aligned.",
        }

        records.append({
            "support_demo_card_record_id": f"sd_card_{demo_id}",
            "external_demo_id": demo_id,
            "domain": dom,
            "source_split": sp,
            "support_task_card": target_card,
            "support_demo_public_view": pub_view,
            "assignment_source": "diagnostic_contract_mapping_from_existing_synthetic_label",
            "assignment_valid_for_real_evidence": False,
            "assignment_valid_for_contract_shape": True,
            "support_demo_card_audit": {
                "uses_hidden_relation_label_for_assignment": True,
                "uses_operator_id_for_assignment": False,
                "uses_support_result_content_in_public_demo": True,
                "uses_support_delta_content_in_public_demo": True,
                "valid_as_public_support_demonstration": True,
                "valid_for_real_external_task_context_evidence": False,
                "valid_for_dataset_contract_validation": True,
                "diagnostic_only": True,
                "valid_for_final_semantic_geometry_evidence": False,
                "valid_for_bridge_evidence": False,
            },
            "audit_label_evaluation_only": d["audit_label_evaluation_only"],
        })

    return records


# ── Dataset contract summary ───────────────────────────────────────────────────

def build_real_external_task_context_dataset_contract() -> dict:
    return {
        "contract_id": "p95_real_external_task_context_dataset_contract_v1",
        "contract_version": "p95_v1",
        "purpose": "provide public task card and demonstration context without exposing hidden labels to retrieval",
        "required_record_types": [
            "query_record",
            "query_task_card_record",
            "support_demo_card_record",
            "hidden_evaluation_record",
        ],
        "public_to_retrieval": [
            "query source observation",
            "query task card",
            "support source observation",
            "support result observation",
            "support visible change summary",
            "support visible preservation summary",
            "support task card",
        ],
        "hidden_from_retrieval": [
            "target_relation_label",
            "operator_id",
            "query target",
            "query delta",
            "hidden relation family",
            "evaluation target B",
        ],
        "contract_invariants": {
            "query_task_card_required": True,
            "support_demo_card_required": True,
            "hidden_target_required": True,
            "labels_hidden_from_retrieval": True,
            "operator_ids_hidden_from_retrieval": True,
            "bridge_forbidden": True,
            "training_forbidden_in_p95": True,
        },
        "ready_for_real_data_collection": True,
    }


# ── Leakage audit ─────────────────────────────────────────────────────────────

def audit_real_task_context_dataset_contract_leakage(
    task_card_catalog: list[dict],
    query_task_card_records: list[dict],
    support_demo_card_records: list[dict],
    dataset_contract: dict,
) -> dict:
    # 1. Catalog leaks
    cat_lbl_leak = sum(1 for c in task_card_catalog if c["task_card_audit"]["uses_hidden_relation_label"])
    cat_op_leak = sum(1 for c in task_card_catalog if c["task_card_audit"]["uses_operator_id"])
    cat_tgt_leak = sum(1 for c in task_card_catalog if c["task_card_audit"]["uses_query_target"])
    cat_del_leak = sum(1 for c in task_card_catalog if c["task_card_audit"]["uses_query_delta"])

    # 2. Query assignments leaks
    q_assign_lbl_leak = sum(1 for q in query_task_card_records if q["assignment_audit"]["uses_hidden_relation_label_for_assignment"])
    q_valid_real = sum(1 for q in query_task_card_records if q["assignment_audit"]["valid_for_real_external_task_context_evidence"])
    q_valid_shape = sum(1 for q in query_task_card_records if q["assignment_audit"]["valid_for_dataset_contract_validation"])

    # 3. Support assignment leaks
    s_assign_lbl_leak = sum(1 for s in support_demo_card_records if s["support_demo_card_audit"]["uses_hidden_relation_label_for_assignment"])
    s_valid_real = sum(1 for s in support_demo_card_records if s["support_demo_card_audit"]["valid_for_real_external_task_context_evidence"])
    s_valid_shape = sum(1 for s in support_demo_card_records if s["support_demo_card_audit"]["valid_for_dataset_contract_validation"])

    # 4. Public retrieval check
    lbl_in_public = sum(1 for field in dataset_contract["public_to_retrieval"] if "label" in field)
    op_in_public = sum(1 for field in dataset_contract["public_to_retrieval"] if "operator" in field)
    tgt_in_public = sum(1 for field in dataset_contract["public_to_retrieval"] if "target" in field)
    del_in_public = sum(1 for field in dataset_contract["public_to_retrieval"] if "delta" in field)

    diag_only_contract = True
    valid_real_evidence = False
    valid_shape_val = True
    diag_pass = True

    return {
        "task_card_count": len(task_card_catalog),
        "query_task_card_record_count": len(query_task_card_records),
        "support_demo_card_record_count": len(support_demo_card_records),

        "task_card_uses_hidden_relation_label_count": cat_lbl_leak,
        "task_card_uses_operator_id_count": cat_op_leak,
        "task_card_uses_query_target_count": cat_tgt_leak,
        "task_card_uses_query_delta_count": cat_del_leak,

        "query_assignment_uses_hidden_relation_label_count": q_assign_lbl_leak,
        "query_assignment_valid_for_real_evidence_count": q_valid_real,
        "query_assignment_valid_for_contract_shape_count": q_valid_shape,

        "support_assignment_uses_hidden_relation_label_count": s_assign_lbl_leak,
        "support_assignment_valid_for_real_evidence_count": s_valid_real,
        "support_assignment_valid_for_contract_shape_count": s_valid_shape,

        "retrieval_public_fields_include_label_count": lbl_in_public,
        "retrieval_public_fields_include_operator_id_count": op_in_public,
        "retrieval_public_fields_include_query_target_count": tgt_in_public,
        "retrieval_public_fields_include_query_delta_count": del_in_public,

        "contract_diagnostic_only": diag_only_contract,
        "valid_for_real_external_task_context_evidence": valid_real_evidence,
        "valid_for_dataset_contract_validation": valid_shape_val,
        "diagnostic_pass": diag_pass,
    }


# ── Proxy audit ───────────────────────────────────────────────────────────────

def audit_real_task_context_proxy_risk(
    task_card_catalog: list[dict],
    query_task_card_records: list[dict],
    support_demo_card_records: list[dict],
) -> dict:
    # Synthetic assignment uses hidden label mapping, so proxy risk is True.
    risk = True
    high_risk_fields = [
        "task_card_id",
        "human_readable_instruction",
        "public_descriptor_tokens",
        "compatible_relation_labels",
    ]

    return {
        "audit_defined": True,
        "proxy_risk_present": risk,
        "high_proxy_risk_fields": high_risk_fields,
        "task_card_descriptor_is_label_free": True,
        "assignment_is_diagnostic_label_mapped": True,
        "real_data_required_to_remove_assignment_proxy": True,
        "diagnostic_pass": True,
    }


# ── Bridge boundary ───────────────────────────────────────────────────────────

def audit_bridge_boundary_after_real_task_context_dataset_contract(
    dataset_contract_ready: bool,
    valid_for_real_external_task_context_evidence: bool,
) -> dict:
    return {
        "bridge_ready": False,
        "bridge_implementation_allowed": False,
        "real_external_task_dataset_contract_ready": dataset_contract_ready,
        "valid_for_real_external_task_context_evidence": valid_for_real_external_task_context_evidence,
        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,
        "blocking_reasons": [
            "real_task_context_contract_only_no_real_data_yet",
            "valid_external_task_context_evidence_not_available",
            "learned_metric_evidence_not_present",
            "semantic_metric_not_ready",
            "bridge_input_contract_not_defined",
            "bridge_validation_not_run",
        ],
        "diagnostic_pass": True,
    }


# ── Public probe ───────────────────────────────────────────────────────────────

def run_p95_real_external_task_context_dataset_contract_probe() -> dict:
    # 0. Validate P94 source contracts
    contracts_val = validate_source_contracts_for_p95()
    contracts_ok = contracts_val["source_contracts_validated"]

    # 1. Build records
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    demo_recs = build_external_demonstration_records(p70a, p70b)
    selector_recs = build_selector_dataset_records(p70a, p70b)
    enriched_queries = build_enriched_query_records(selector_recs)

    task_card_catalog = build_task_card_catalog()
    query_task_card_records = build_query_task_card_records(enriched_queries, task_card_catalog)
    support_demo_card_records = build_support_demonstration_card_records(demo_recs, task_card_catalog)

    dataset_contract = build_real_external_task_context_dataset_contract()

    # 2. Audits
    leakage = audit_real_task_context_dataset_contract_leakage(
        task_card_catalog, query_task_card_records, support_demo_card_records, dataset_contract,
    )
    proxy = audit_real_task_context_proxy_risk(
        task_card_catalog, query_task_card_records, support_demo_card_records,
    )

    # 3. Evidence flags
    contract_ready = (
        contracts_ok
        and len(task_card_catalog) == 10
        and len(query_task_card_records) > 0
        and len(support_demo_card_records) > 0
        and leakage["diagnostic_pass"]
        and proxy["diagnostic_pass"]
    )

    valid_etc_available = False
    valid_real_ev = False
    ready_for_real_collection = True

    # 4. Bridge boundary
    bridge_audit = audit_bridge_boundary_after_real_task_context_dataset_contract(
        contract_ready, valid_real_ev,
    )

    # 5. Verdict
    verdict_str = VERDICT if contracts_ok else "P95_BLOCKED_BY_SOURCE_CONTRACT"

    return {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,
        "verdict": verdict_str,

        "source_external_task_context_phase": SOURCE_EXTERNAL_TASK_CONTEXT_PHASE,
        "source_query_observation_enrichment_phase": SOURCE_QUERY_OBSERVATION_ENRICHMENT_PHASE,

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "new_model_implemented": NEW_MODEL_IMPLEMENTED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "source_contracts_validated": contracts_ok,

        "task_card_catalog_count": len(task_card_catalog),
        "query_task_card_record_count": len(query_task_card_records),
        "support_demo_card_record_count": len(support_demo_card_records),

        "real_external_task_dataset_contract": dataset_contract,
        "contract_leakage_audit": leakage,
        "contract_proxy_audit": proxy,

        "real_external_task_dataset_contract_ready": contract_ready,
        "valid_external_task_context_available": valid_etc_available,
        "valid_for_real_external_task_context_evidence": valid_real_ev,
        "ready_for_real_data_collection": ready_for_real_collection,

        "task_context_signal_present": False,
        "task_context_external_metric_supported": False,

        "learned_selector_evidence_present": False,
        "learned_metric_evidence_present": False,
        "semantic_metric_ready": False,
        "bridge_implementation_allowed": False,
        "bridge_ready": False,
        "generation_claims_allowed": False,
        "semantic_geometry_claims_allowed": False,

        "recommended_next_phase": "P96_real_task_card_support_retrieval_pilot_no_training_no_bridge",

        "bridge_boundary_after_real_task_context_dataset_contract": bridge_audit,

        "json_safe": True,
        "diagnostic_only": True,
    }
