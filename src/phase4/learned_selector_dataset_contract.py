# src/phase4/learned_selector_dataset_contract.py

import json
from typing import Any

from src.phase3.pure_numeric_relation_testbed import (
    run_p70a_pure_numeric_relation_testbed_probe,
)

from src.phase3.synthetic_time_series_relation_testbed import (
    run_p70b_synthetic_time_series_relation_testbed_probe,
)

from src.phase3.source_available_query_descriptor_contract import (
    build_p70a_source_available_query_descriptor_records,
    build_p70b_source_available_query_descriptor_records,
)

from src.phase4.learned_experiment_authority_contract import (
    run_p80_learned_experiment_authority_contract_probe,
)

PHASE = "P81"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Learned Selector Dataset Contract and Split Builder"
CONTRACT_VERSION = "phase4_p81_learned_selector_dataset_contract_v1"

SOURCE_AUTHORITY_PHASE = "P80"
SOURCE_VECTOR_TESTBED_PHASE = "P70A"
SOURCE_TIME_SERIES_TESTBED_PHASE = "P70B"
SOURCE_QUERY_CONTRACT_PHASE = "P77"

PHASE4_LEARNED_EXPERIMENTS_ALLOWED = True
TRAINING_ALLOWED_BY_PHASE4_AUTHORITY = True
NUMPY_ALLOWED_BY_PHASE4_AUTHORITY = True
TORCH_ALLOWED_BY_PHASE4_AUTHORITY = True
LEARNED_SELECTOR_ALLOWED_BY_PHASE4_AUTHORITY = True
LEARNED_METRIC_ALLOWED_BY_PHASE4_AUTHORITY = True

MODEL_TRAINING_PERFORMED = False
TORCH_TRAINING_PERFORMED = False
OPTIMIZER_CREATED = False
CHECKPOINT_WRITTEN = False

BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
SEMANTIC_METRIC_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

LEARNED_SELECTOR_DATASET_CONTRACT_PRESENT = True
SELECTOR_TRAIN_VAL_TEST_SPLIT_BUILT = True
TARGET_LABEL_AVAILABLE_FOR_SUPERVISED_TRAINING = True
LEARNED_SELECTOR_EVIDENCE_PRESENT = False

TARGET_ENDPOINT_USED_FOR_SELECTOR_INPUT = False
TARGET_DELTA_USED_FOR_SELECTOR_INPUT = False
EXACT_RELATION_LABEL_USED_FOR_SELECTOR_INPUT = False
EXACT_OPERATOR_ID_USED_FOR_SELECTOR_INPUT = False
AUDIT_METADATA_USED_FOR_SELECTOR_INPUT = False
RELATION_SPECIFIC_HINT_USED_FOR_MAIN_SELECTOR_INPUT = False

PRIMARY_EMPIRICAL_TARGET = "learned_selector_dataset_contract_and_split_builder"
VERDICT = "P81_READY_FOR_REVIEW"

MAIN_SELECTOR_VIEW = "clean_selector_input"

MAIN_SELECTOR_ALLOWED_INPUT_FIELDS = [
    "domain",
    "source_state_summary",
    "source_parameter_summary",
    "query_intensity_hint",
    "context_world",
    "source_split_origin",
]

MAIN_SELECTOR_FORBIDDEN_INPUT_FIELDS = [
    "_audit_metadata",
    "true_relation_type",
    "relation_type",
    "operator_id",
    "z_b",
    "params_b",
    "series_b",
    "z_end",
    "params_end",
    "series_end",
    "target_endpoint",
    "target_midpoint",
    "z_b_minus_z_a",
    "params_b_minus_params_a",
    "series_summary_delta",
    "relation_family_hint",
    "transformation_class_hint",
    "relation_axis_hint",
    "parameter_group_hint",
]

HINT_PASSTHROUGH_BASELINE_VIEW = "hint_passthrough_baseline_input"

HINT_BASELINE_ALLOWED_FIELDS = [
    "relation_family_hint",
    "transformation_class_hint",
    "relation_axis_hint",
    "parameter_group_hint",
    "query_intensity_hint",
    "domain",
]

LABEL_EVALUATION_VIEW = "label_evaluation_only"

LABEL_FIELDS = [
    "target_relation_label",
    "domain",
    "descriptor_id",
    "split",
]

SPLIT_MAPPING = {
    "train_style_repeated_instances": "train",
    "heldout_base_state": "validation",
    "heldout_magnitude": "test",
}

REQUIRED_SPLITS = ["train", "validation", "test"]

TRAIN_VAL_TEST_SPLIT_ISOLATED = True
TEST_SPLIT_USED_FOR_TRAINING = False
VALIDATION_LABELS_USED_FOR_MODEL_SELECTION_ONLY = True


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-15:
        return 0.0
    return float(numerator / denominator)


def deep_copy_without_keys(value: Any, forbidden_keys: set[str]) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            if k in forbidden_keys:
                continue
            cleaned[k] = deep_copy_without_keys(v, forbidden_keys)
        return cleaned
    elif isinstance(value, list):
        return [deep_copy_without_keys(x, forbidden_keys) for x in value]
    else:
        return value


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


def map_source_split_to_dataset_split(source_split: str) -> str:
    return SPLIT_MAPPING.get(source_split, "unknown_holdout")


def get_target_relation_label_from_audit_metadata(record: dict) -> str:
    # Used only to construct evaluation label target
    return record["_audit_metadata"]["true_relation_type"]


def build_main_selector_input(record: dict) -> dict:
    inp = {
        "domain": record["domain"],
        "query_intensity_hint": record["query_descriptor"]["query_intensity_hint"],
        "context_world": record.get("context", {}).get("world", record.get("context", {}).get("world_type", "unknown")),
        "source_split_origin": record["split"],
    }
    if "source_state_summary" in record:
        inp["source_state_summary"] = record["source_state_summary"]
    if "source_parameter_summary" in record:
        inp["source_parameter_summary"] = record["source_parameter_summary"]
    return inp


def build_hint_passthrough_baseline_input(record: dict) -> dict:
    qd = record.get("query_descriptor", {})
    inp = {
        "query_intensity_hint": qd.get("query_intensity_hint", 0.0),
        "domain": record["domain"],
    }
    if "relation_family_hint" in qd:
        inp["relation_family_hint"] = qd["relation_family_hint"]
    if "transformation_class_hint" in qd:
        inp["transformation_class_hint"] = qd["transformation_class_hint"]
    if "relation_axis_hint" in qd:
        inp["relation_axis_hint"] = qd["relation_axis_hint"]
    if "parameter_group_hint" in qd:
        inp["parameter_group_hint"] = qd["parameter_group_hint"]
    return inp


def build_selector_dataset_record(record: dict, domain_prefix: str, index: int) -> dict:
    split = map_source_split_to_dataset_split(record["split"])
    selector_input = build_main_selector_input(record)
    hint_passthrough = build_hint_passthrough_baseline_input(record)
    
    label_eval = {
        "target_relation_label": get_target_relation_label_from_audit_metadata(record),
        "domain": record["domain"],
        "descriptor_id": record["descriptor_id"],
        "split": split,
    }
    
    leakage_audit = audit_selector_input_leakage(selector_input)
    
    return {
        "dataset_record_id": f"p81_record_{domain_prefix}_{index}",
        "domain": record["domain"],
        "source_descriptor_id": record["descriptor_id"],
        "source_split_origin": record["split"],
        "split": split,

        "selector_input": selector_input,
        "hint_passthrough_baseline_input": hint_passthrough,
        "label_evaluation": label_eval,
        "input_leakage_audit": leakage_audit,
    }


def build_selector_dataset_records(p70a_probe: dict, p70b_probe: dict) -> list[dict]:
    recs_a = build_p70a_source_available_query_descriptor_records(p70a_probe)
    recs_b = build_p70b_source_available_query_descriptor_records(p70b_probe)
    
    out = []
    for idx, r in enumerate(recs_a):
        out.append(build_selector_dataset_record(r, "p70a", idx))
    for idx, r in enumerate(recs_b):
        out.append(build_selector_dataset_record(r, "p70b", idx))
    return out


def audit_selector_input_leakage(selector_input: dict) -> dict:
    target_endpoint_present = False
    target_delta_present = False
    exact_label_present = False
    exact_operator_present = False
    audit_metadata_present = False
    relation_specific_hint_present = False
    
    forbidden_endpoints = {"z_b", "params_b", "series_b", "z_end", "params_end", "series_end", "target_endpoint", "target_midpoint"}
    forbidden_deltas = {"z_b_minus_z_a", "params_b_minus_params_a", "series_summary_delta"}
    forbidden_labels = {"relation_type", "true_relation_type", "target_relation_label"}
    forbidden_operators = {"operator_id"}
    forbidden_metadata = {"_audit_metadata"}
    forbidden_hints = {"relation_family_hint", "transformation_class_hint", "relation_axis_hint", "parameter_group_hint"}
    
    all_keys = scan_keys(selector_input)
    
    for k in all_keys:
        if k in forbidden_endpoints:
            target_endpoint_present = True
        if k in forbidden_deltas:
            target_delta_present = True
        if k in forbidden_labels:
            exact_label_present = True
        if k in forbidden_operators:
            exact_operator_present = True
        if k in forbidden_metadata:
            audit_metadata_present = True
        if k in forbidden_hints:
            relation_specific_hint_present = True
            
    diag_pass = not (
        target_endpoint_present
        or target_delta_present
        or exact_label_present
        or exact_operator_present
        or audit_metadata_present
        or relation_specific_hint_present
    )
    
    return {
        "target_endpoint_present": target_endpoint_present,
        "target_delta_present": target_delta_present,
        "exact_label_present_in_selector_input": exact_label_present,
        "exact_operator_id_present_in_selector_input": exact_operator_present,
        "audit_metadata_present_in_selector_input": audit_metadata_present,
        "relation_specific_hint_present_in_main_selector_input": relation_specific_hint_present,
        "diagnostic_pass": diag_pass,
    }


def audit_learned_selector_dataset_contract(records: list[dict]) -> dict:
    total = len(records)
    p70a_count = sum(1 for r in records if r["domain"] == "p70a_vector_world")
    p70b_count = sum(1 for r in records if r["domain"] == "p70b_time_series_parameter_world")
    
    train_count = sum(1 for r in records if r["split"] == "train")
    val_count = sum(1 for r in records if r["split"] == "validation")
    test_count = sum(1 for r in records if r["split"] == "test")
    unknown_count = sum(1 for r in records if r["split"] == "unknown_holdout")
    
    leakage_detected = False
    target_endpoint_leak = False
    target_delta_leak = False
    exact_label_leak = False
    exact_operator_leak = False
    audit_metadata_leak = False
    hint_leak = False
    
    for r in records:
        audit = r["input_leakage_audit"]
        if audit["diagnostic_pass"] is False:
            leakage_detected = True
        if audit["target_endpoint_present"] is True:
            target_endpoint_leak = True
        if audit["target_delta_present"] is True:
            target_delta_leak = True
        if audit["exact_label_present_in_selector_input"] is True:
            exact_label_leak = True
        if audit["exact_operator_id_present_in_selector_input"] is True:
            exact_operator_leak = True
        if audit["audit_metadata_present_in_selector_input"] is True:
            audit_metadata_leak = True
        if audit["relation_specific_hint_present_in_main_selector_input"] is True:
            hint_leak = True
            
    splits_ok = (train_count > 0 and val_count > 0 and test_count > 0)
    diag_pass = (splits_ok and not leakage_detected)
    
    return {
        "dataset_record_count": int(total),
        "p70a_record_count": int(p70a_count),
        "p70b_record_count": int(p70b_count),

        "train_count": int(train_count),
        "validation_count": int(val_count),
        "test_count": int(test_count),
        "unknown_holdout_count": int(unknown_count),

        "all_required_splits_present": splits_ok,
        "selector_input_leakage_detected": leakage_detected,
        "label_available_for_supervised_training": True,
        "hint_passthrough_baseline_present": True,

        "target_endpoint_used_for_selector_input": target_endpoint_leak,
        "target_delta_used_for_selector_input": target_delta_leak,
        "exact_relation_label_used_for_selector_input": exact_label_leak,
        "exact_operator_id_used_for_selector_input": exact_operator_leak,
        "audit_metadata_used_for_selector_input": audit_metadata_leak,
        "relation_specific_hint_used_for_main_selector_input": hint_leak,

        "train_val_test_split_isolated": TRAIN_VAL_TEST_SPLIT_ISOLATED,
        "test_split_used_for_training": TEST_SPLIT_USED_FOR_TRAINING,

        "diagnostic_pass": diag_pass,
    }


def build_required_baseline_contract() -> dict:
    return {
        "required_baselines": [
            "majority_selector_baseline",
            "source_only_baseline",
            "intensity_only_baseline",
            "source_plus_intensity_baseline",
            "hint_passthrough_baseline",
            "oracle_operator_upper_bound"
        ],
        "main_selector_must_beat": [
            "majority_selector_baseline",
            "source_only_baseline",
            "intensity_only_baseline"
        ],
        "hint_passthrough_is_control_not_main_input": True,
        "oracle_operator_is_upper_bound_not_training_input": True,
    }


def validate_source_contracts_for_p81() -> dict:
    validated = True
    missing_or_invalid = []
    
    p80_ok = True
    
    try:
        p80 = run_p80_learned_experiment_authority_contract_probe()
        if p80.get("phase") != "P80" or p80.get("verdict") != "P80_READY_FOR_REVIEW":
            p80_ok = False
            missing_or_invalid.append("p80_invalid_phase_or_verdict")
        if p80.get("phase4_learned_experiments_allowed") is not True:
            p80_ok = False
            missing_or_invalid.append("p80_experiments_not_allowed")
        if p80.get("training_allowed") is not True:
            p80_ok = False
            missing_or_invalid.append("p80_training_not_allowed")
        if p80.get("torch_allowed") is not True:
            p80_ok = False
            missing_or_invalid.append("p80_torch_not_allowed")
        if p80.get("numpy_allowed") is not True:
            p80_ok = False
            missing_or_invalid.append("p80_numpy_not_allowed")
        if p80.get("learned_selector_allowed") is not True:
            p80_ok = False
            missing_or_invalid.append("p80_selector_not_allowed")
        if p80.get("learned_metric_allowed") is not True:
            p80_ok = False
            missing_or_invalid.append("p80_metric_not_allowed")
        if p80.get("bridge_implementation_allowed") is not False:
            p80_ok = False
            missing_or_invalid.append("p80_bridge_allowed")
        if p80.get("bridge_ready") is not False:
            p80_ok = False
            missing_or_invalid.append("p80_bridge_ready")
        if p80.get("leakage_gates_required") is not True:
            p80_ok = False
            missing_or_invalid.append("p80_leakage_gates_not_required")
        if p80.get("baseline_comparison_required") is not True:
            p80_ok = False
            missing_or_invalid.append("p80_baseline_not_required")
        if p80.get("train_val_test_split_required") is not True:
            p80_ok = False
            missing_or_invalid.append("p80_split_not_required")
        if p80.get("negative_controls_required") is not True:
            p80_ok = False
            missing_or_invalid.append("p80_neg_controls_not_required")
    except Exception as e:
        p80_ok = False
        missing_or_invalid.append(f"p80_exception_{str(e)}")
        
    validated = p80_ok
    
    return {
        "source_contracts_validated": validated,
        "p80_validated": p80_ok,
        "p80_phase4_authority_preserved": p80_ok,
        "p80_bridge_not_ready_preserved": p80_ok,
        "p80_leakage_gates_preserved": p80_ok,
        "missing_or_invalid": missing_or_invalid,
    }


def run_p81_learned_selector_dataset_contract_probe() -> dict:
    contracts_val = validate_source_contracts_for_p81()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    dataset_records = build_selector_dataset_records(p70a, p70b)
    dataset_audit = audit_learned_selector_dataset_contract(dataset_records)
    baselines = build_required_baseline_contract()
    
    verdict_str = VERDICT if contracts_ok else "P81_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if dataset_audit["diagnostic_pass"] is False:
            if dataset_audit["selector_input_leakage_detected"] is True:
                verdict_str = "P81_BLOCKED_BY_SELECTOR_INPUT_LEAKAGE"
            elif dataset_audit["all_required_splits_present"] is False:
                verdict_str = "P81_BLOCKED_BY_SPLIT_CONTRACT"
        elif MODEL_TRAINING_PERFORMED is True:
            verdict_str = "P81_BLOCKED_BY_UNEXPECTED_MODEL_TRAINING"
        elif BRIDGE_READY is True:
            verdict_str = "P81_BLOCKED_BY_PREMATURE_BRIDGE_AUTHORITY"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p80_phase4_authority_preserved": contracts_val["p80_phase4_authority_preserved"],
        "p80_bridge_not_ready_preserved": contracts_val["p80_bridge_not_ready_preserved"],
        "p80_leakage_gates_preserved": contracts_val["p80_leakage_gates_preserved"],

        "learned_selector_dataset_contract_present": LEARNED_SELECTOR_DATASET_CONTRACT_PRESENT,
        "selector_train_val_test_split_built": SELECTOR_TRAIN_VAL_TEST_SPLIT_BUILT,
        "all_required_splits_present": dataset_audit["all_required_splits_present"],
        "selector_input_leakage_detected": dataset_audit["selector_input_leakage_detected"],

        "target_label_available_for_supervised_training": TARGET_LABEL_AVAILABLE_FOR_SUPERVISED_TRAINING,
        "hint_passthrough_baseline_present": True,
        "baseline_contract_present": True,

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "bridge_ready": BRIDGE_READY,

        "json_safe": True,
    }
    
    # 3 sample records for compact output
    sample_records = dataset_records[:3] if len(dataset_records) >= 3 else dataset_records
    
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,

        "source_authority_phase": SOURCE_AUTHORITY_PHASE,
        "source_vector_testbed_phase": SOURCE_VECTOR_TESTBED_PHASE,
        "source_time_series_testbed_phase": SOURCE_TIME_SERIES_TESTBED_PHASE,
        "source_query_contract_phase": SOURCE_QUERY_CONTRACT_PHASE,

        "verdict": verdict_str,

        "phase4_learned_experiments_allowed": PHASE4_LEARNED_EXPERIMENTS_ALLOWED,
        "training_allowed_by_phase4_authority": TRAINING_ALLOWED_BY_PHASE4_AUTHORITY,
        "numpy_allowed_by_phase4_authority": NUMPY_ALLOWED_BY_PHASE4_AUTHORITY,
        "torch_allowed_by_phase4_authority": TORCH_ALLOWED_BY_PHASE4_AUTHORITY,
        "learned_selector_allowed_by_phase4_authority": LEARNED_SELECTOR_ALLOWED_BY_PHASE4_AUTHORITY,
        "learned_metric_allowed_by_phase4_authority": LEARNED_METRIC_ALLOWED_BY_PHASE4_AUTHORITY,

        "model_training_performed": MODEL_TRAINING_PERFORMED,
        "torch_training_performed": TORCH_TRAINING_PERFORMED,
        "optimizer_created": OPTIMIZER_CREATED,
        "checkpoint_written": CHECKPOINT_WRITTEN,

        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "bridge_ready": BRIDGE_READY,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,

        "learned_selector_dataset_contract_present": LEARNED_SELECTOR_DATASET_CONTRACT_PRESENT,
        "selector_train_val_test_split_built": SELECTOR_TRAIN_VAL_TEST_SPLIT_BUILT,
        "target_label_available_for_supervised_training": TARGET_LABEL_AVAILABLE_FOR_SUPERVISED_TRAINING,
        "learned_selector_evidence_present": LEARNED_SELECTOR_EVIDENCE_PRESENT,

        "target_endpoint_used_for_selector_input": TARGET_ENDPOINT_USED_FOR_SELECTOR_INPUT,
        "target_delta_used_for_selector_input": TARGET_DELTA_USED_FOR_SELECTOR_INPUT,
        "exact_relation_label_used_for_selector_input": EXACT_RELATION_LABEL_USED_FOR_SELECTOR_INPUT,
        "exact_operator_id_used_for_selector_input": EXACT_OPERATOR_ID_USED_FOR_SELECTOR_INPUT,
        "audit_metadata_used_for_selector_input": AUDIT_METADATA_USED_FOR_SELECTOR_INPUT,
        "relation_specific_hint_used_for_main_selector_input": RELATION_SPECIFIC_HINT_USED_FOR_MAIN_SELECTOR_INPUT,

        "dataset_views": [
            MAIN_SELECTOR_VIEW,
            HINT_PASSTHROUGH_BASELINE_VIEW,
            LABEL_EVALUATION_VIEW
        ],

        "split_mapping": SPLIT_MAPPING,
        "required_splits": REQUIRED_SPLITS,

        "source_contracts_validated": contracts_ok,
        "p80_phase4_authority_preserved": contracts_val["p80_phase4_authority_preserved"],
        "p80_bridge_not_ready_preserved": contracts_val["p80_bridge_not_ready_preserved"],
        "p80_leakage_gates_preserved": contracts_val["p80_leakage_gates_preserved"],

        "dataset_contract_audit": dataset_audit,
        "baseline_contract": baselines,

        "sample_dataset_records": sample_records,
        "sample_record_count": int(len(sample_records)),

        "sanity_summary": sanity_summary,

        "json_safe": True,
        "diagnostic_only": True
    }
    
    # Assert JSON safe
    json.dumps(output)
    
    return output
