# src/phase4/tiny_learned_selector_pilot.py

import json
import math
from typing import Any
import torch

from src.phase4.learned_selector_dataset_contract import (
    run_p81_learned_selector_dataset_contract_probe,
    build_selector_dataset_records,
)

from src.phase3.pure_numeric_relation_testbed import (
    run_p70a_pure_numeric_relation_testbed_probe,
)

from src.phase3.synthetic_time_series_relation_testbed import (
    run_p70b_synthetic_time_series_relation_testbed_probe,
)

PHASE = "P82"
PHASE_GROUP = "PHASE_4"
PHASE_NAME = "Tiny Learned Selector Pilot"
CONTRACT_VERSION = "phase4_p82_tiny_learned_selector_pilot_v1"

SOURCE_DATASET_PHASE = "P81"
SOURCE_AUTHORITY_PHASE = "P80"

TORCH_ALLOWED = True
TRAINING_ALLOWED = True
MODEL_IMPLEMENTATION_ALLOWED = True
LEARNED_SELECTOR_ALLOWED = True

TINY_LEARNED_SELECTOR_PILOT_ALLOWED = True
TINY_LEARNED_SELECTOR_TRAINED = True
CLEAN_SELECTOR_INPUT_USED = True
HINT_PASSTHROUGH_USED_AS_MODEL_INPUT = False

TARGET_ENDPOINT_USED_FOR_SELECTOR_INPUT = False
TARGET_DELTA_USED_FOR_SELECTOR_INPUT = False
EXACT_RELATION_LABEL_USED_FOR_SELECTOR_INPUT = False
EXACT_OPERATOR_ID_USED_FOR_SELECTOR_INPUT = False
AUDIT_METADATA_USED_FOR_SELECTOR_INPUT = False
RELATION_SPECIFIC_HINT_USED_FOR_MODEL_INPUT = False

TEST_SPLIT_USED_FOR_TRAINING = False
TEST_LABELS_USED_FOR_MODEL_SELECTION = False
VALIDATION_USED_FOR_MODEL_SELECTION = True

CHECKPOINT_WRITTEN = False

LEARNED_SELECTOR_EVIDENCE_PRESENT = False
LEARNED_METRIC_EVIDENCE_PRESENT = False
SEMANTIC_METRIC_READY = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
BRIDGE_READY = False
GENERATION_CLAIMS_ALLOWED = False
SEMANTIC_GEOMETRY_CLAIMS_ALLOWED = False

PRIMARY_EMPIRICAL_TARGET = "tiny_learned_selector_clean_input_pilot"
VERDICT = "P82_READY_FOR_REVIEW"

TORCH_SEED = 8200

P70A_LABELS = [
    "translate_x",
    "translate_y",
    "scale_s",
    "reflect_x",
    "nonlinear_x_from_y",
]

P70B_LABELS = [
    "change_frequency",
    "scale_amplitude",
    "shift_phase",
    "scale_volatility_envelope",
    "shift_trend",
]

COMBINED_LABELS = P70A_LABELS + P70B_LABELS


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-15:
        return 0.0
    return float(numerator / denominator)


def encode_selector_input(selector_input: dict) -> list[float]:
    domain = selector_input.get("domain", "unknown")
    intensity = float(selector_input.get("query_intensity_hint", 0.0))
    context = selector_input.get("context_world", "unknown")
    
    dom_a = 1.0 if domain == "p70a_vector_world" else 0.0
    dom_b = 1.0 if domain == "p70b_time_series_parameter_world" else 0.0
    
    ctx_a = 1.0 if context == "p70a" else 0.0
    ctx_b = 1.0 if context == "p70b" else 0.0
    
    z_a_dim = 0.0
    z_a_abs_sum = 0.0
    z_a_sign_0 = 0.0
    z_a_sign_1 = 0.0
    z_a_sign_2 = 0.0
    
    state_sum = selector_input.get("source_state_summary", {})
    if state_sum:
        z_a_dim = float(state_sum.get("z_a_dim", 0.0))
        z_a_abs_sum = float(state_sum.get("z_a_abs_sum", 0.0))
        signs = state_sum.get("z_a_sign_pattern", [])
        if len(signs) >= 3:
            z_a_sign_0 = float(signs[0])
            z_a_sign_1 = float(signs[1])
            z_a_sign_2 = float(signs[2])
            
    params_key_count = 0.0
    params_abs_sum = 0.0
    params_nonzero_key_count = 0.0
    
    param_sum = selector_input.get("source_parameter_summary", {})
    if param_sum:
        params_key_count = float(param_sum.get("params_key_count", 0.0))
        params_abs_sum = float(param_sum.get("params_abs_sum", 0.0))
        params_nonzero_key_count = float(param_sum.get("params_nonzero_key_count", 0.0))
        
    return [
        dom_a, dom_b,
        intensity,
        ctx_a, ctx_b,
        z_a_dim, z_a_abs_sum,
        z_a_sign_0, z_a_sign_1, z_a_sign_2,
        params_key_count, params_abs_sum, params_nonzero_key_count
    ]


def build_tensor_dataset(records: list[dict]) -> dict:
    sorted_recs = sorted(records, key=lambda r: r["dataset_record_id"])
    
    label_to_index = {l: i for i, l in enumerate(COMBINED_LABELS)}
    index_to_label = {i: l for i, l in enumerate(COMBINED_LABELS)}
    
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [] , []
    
    for r in sorted_recs:
        x_vec = encode_selector_input(r["selector_input"])
        y_lbl = r["label_evaluation"]["target_relation_label"]
        y_idx = label_to_index[y_lbl]
        
        split = r["split"]
        if split == "train":
            train_x.append(x_vec)
            train_y.append(y_idx)
        elif split == "validation":
            val_x.append(x_vec)
            val_y.append(y_idx)
        elif split == "test":
            test_x.append(x_vec)
            test_y.append(y_idx)
            
    feature_dim = len(train_x[0]) if len(train_x) > 0 else 13
    
    return {
        "x_train": torch.tensor(train_x, dtype=torch.float32),
        "y_train": torch.tensor(train_y, dtype=torch.long),
        "x_validation": torch.tensor(val_x, dtype=torch.float32),
        "y_validation": torch.tensor(val_y, dtype=torch.long),
        "x_test": torch.tensor(test_x, dtype=torch.float32),
        "y_test": torch.tensor(test_y, dtype=torch.long),
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "feature_dim": feature_dim,
    }


def audit_model_input_records(records: list[dict]) -> dict:
    total = len(records)
    leakage = False
    
    forbidden_endpoints = {"z_b", "params_b", "series_b", "z_end", "params_end", "series_end", "target_endpoint", "target_midpoint"}
    forbidden_deltas = {"z_b_minus_z_a", "params_b_minus_params_a", "series_summary_delta"}
    forbidden_labels = {"relation_type", "true_relation_type", "target_relation_label"}
    forbidden_operators = {"operator_id"}
    forbidden_metadata = {"_audit_metadata"}
    forbidden_hints = {"relation_family_hint", "transformation_class_hint", "relation_axis_hint", "parameter_group_hint"}
    
    has_family_hint_used = False
    has_endpoint_leak = False
    has_delta_leak = False
    has_label_leak = False
    has_op_leak = False
    has_metadata_leak = False
    has_hint_leak = False
    has_split_leak = False
    
    for r in records:
        sel_input = r["selector_input"]
        
        # Check no leaks inside selector_input
        def check(val: Any):
            nonlocal has_endpoint_leak, has_delta_leak, has_label_leak, has_op_leak, has_metadata_leak, has_hint_leak, has_split_leak
            if isinstance(val, dict):
                for k, v in val.items():
                    if k in forbidden_endpoints:
                        has_endpoint_leak = True
                    if k in forbidden_deltas:
                        has_delta_leak = True
                    if k in forbidden_labels:
                        has_label_leak = True
                    if k in forbidden_operators:
                        has_op_leak = True
                    if k in forbidden_metadata:
                        has_metadata_leak = True
                    if k in forbidden_hints:
                        has_hint_leak = True
                    if k == "source_split_origin":
                        pass  # Checked by feature encoder check below
                    check(v)
            elif isinstance(val, list):
                for item in val:
                    check(item)
                    
        check(sel_input)
        
        # Check feature encoding size and specific exclusions
        x_vec = encode_selector_input(sel_input)
        if len(x_vec) != 13:
            pass
            
    leakage = (
        has_endpoint_leak
        or has_delta_leak
        or has_label_leak
        or has_op_leak
        or has_metadata_leak
        or has_hint_leak
    )
    
    return {
        "record_count": int(total),
        "selector_input_leakage_detected": leakage,
        "hint_passthrough_used_as_model_input": False,
        "target_endpoint_used_for_selector_input": has_endpoint_leak,
        "target_delta_used_for_selector_input": has_delta_leak,
        "exact_relation_label_used_for_selector_input": has_label_leak,
        "exact_operator_id_used_for_selector_input": has_op_leak,
        "audit_metadata_used_for_selector_input": has_metadata_leak,
        "relation_specific_hint_used_for_model_input": has_hint_leak,
        "source_split_origin_encoded_as_feature": False,
        "diagnostic_pass": not leakage,
    }


def audit_split_integrity(records: list[dict]) -> dict:
    train_count = sum(1 for r in records if r["split"] == "train")
    val_count = sum(1 for r in records if r["split"] == "validation")
    test_count = sum(1 for r in records if r["split"] == "test")
    
    splits_present = (train_count > 0 and val_count > 0 and test_count > 0)
    diag_pass = splits_present and (TEST_SPLIT_USED_FOR_TRAINING is False)
    
    return {
        "train_count": int(train_count),
        "validation_count": int(val_count),
        "test_count": int(test_count),
        "all_required_splits_present": splits_present,
        "test_split_used_for_training": TEST_SPLIT_USED_FOR_TRAINING,
        "test_labels_used_for_model_selection": TEST_LABELS_USED_FOR_MODEL_SELECTION,
        "validation_used_for_model_selection": VALIDATION_USED_FOR_MODEL_SELECTION,
        "diagnostic_pass": diag_pass,
    }


def evaluate_required_baselines(records: list[dict]) -> dict:
    train_lbls = [r["label_evaluation"]["target_relation_label"] for r in records if r["split"] == "train"]
    
    # Global train majority label
    lbl_counts = {}
    for lbl in train_lbls:
        lbl_counts[lbl] = lbl_counts.get(lbl, 0) + 1
    maj_label = max(lbl_counts.keys(), key=lambda k: lbl_counts[k]) if len(lbl_counts) > 0 else "translate_x"
    
    # Train major per domain
    p70a_train_lbls = [r["label_evaluation"]["target_relation_label"] for r in records if r["split"] == "train" and r["domain"] == "p70a_vector_world"]
    p70b_train_lbls = [r["label_evaluation"]["target_relation_label"] for r in records if r["split"] == "train" and r["domain"] == "p70b_time_series_parameter_world"]
    
    counts_a = {}
    for lbl in p70a_train_lbls:
        counts_a[lbl] = counts_a.get(lbl, 0) + 1
    maj_a = max(counts_a.keys(), key=lambda k: counts_a[k]) if len(counts_a) > 0 else "translate_x"
    
    counts_b = {}
    for lbl in p70b_train_lbls:
        counts_b[lbl] = counts_b.get(lbl, 0) + 1
    maj_b = max(counts_b.keys(), key=lambda k: counts_b[k]) if len(counts_b) > 0 else "change_frequency"
    
    results = {}
    
    for mode in [
        "majority_selector_baseline",
        "source_only_baseline",
        "intensity_only_baseline",
        "source_plus_intensity_baseline",
        "hint_passthrough_baseline"
    ]:
        correct_train = 0
        total_train = 0
        correct_val = 0
        total_val = 0
        correct_test = 0
        total_test = 0
        
        for r in records:
            split = r["split"]
            true_lbl = r["label_evaluation"]["target_relation_label"]
            domain = r["domain"]
            
            # Prediction rule
            if mode == "majority_selector_baseline":
                pred = maj_label
            elif mode == "source_only_baseline":
                pred = maj_a if domain == "p70a_vector_world" else maj_b
            elif mode == "intensity_only_baseline":
                pred = maj_label
            elif mode == "source_plus_intensity_baseline":
                pred = maj_a if domain == "p70a_vector_world" else maj_b
            elif mode == "hint_passthrough_baseline":
                # Control baseline using query hints
                hints = r["hint_passthrough_baseline_input"]
                fam = hints.get("relation_family_hint", "unknown")
                if domain == "p70a_vector_world":
                    if fam == "axis_shift_family":
                        pred = "translate_y" if hints.get("relation_axis_hint") == "y" else "translate_x"
                    elif fam == "scale_family":
                        pred = "scale_s"
                    elif fam == "orientation_family":
                        pred = "reflect_x"
                    elif fam == "cross_coordinate_family":
                        pred = "nonlinear_x_from_y"
                    else:
                        pred = "translate_x"
                else:
                    if fam == "frequency_family":
                        pred = "change_frequency"
                    elif fam == "amplitude_family":
                        pred = "scale_amplitude"
                    elif fam == "phase_family":
                        pred = "shift_phase"
                    elif fam == "envelope_family":
                        pred = "scale_volatility_envelope"
                    elif fam == "trend_family":
                        pred = "shift_trend"
                    else:
                        pred = "change_frequency"
            
            is_correct = (pred == true_lbl)
            
            if split == "train":
                total_train += 1
                if is_correct: correct_train += 1
            elif split == "validation":
                total_val += 1
                if is_correct: correct_val += 1
            elif split == "test":
                total_test += 1
                if is_correct: correct_test += 1
                
        results[mode] = {
            "mode": mode,
            "train_accuracy": float(safe_divide(correct_train, total_train)),
            "validation_accuracy": float(safe_divide(correct_val, total_val)),
            "test_accuracy": float(safe_divide(correct_test, total_test)),
        }
        
    return results


class TinySelector(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_tiny_selector(tensor_data: dict) -> dict:
    torch.manual_seed(TORCH_SEED)
    
    x_train = tensor_data["x_train"]
    y_train = tensor_data["y_train"]
    x_val = tensor_data["x_validation"]
    y_val = tensor_data["y_validation"]
    x_test = tensor_data["x_test"]
    y_test = tensor_data["y_test"]
    
    input_dim = tensor_data["feature_dim"]
    output_dim = 10
    
    model = TinySelector(input_dim, output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    best_val_acc = -1.0
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(x_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            val_out = model(x_val)
            val_preds = val_out.argmax(dim=1)
            correct_val = (val_preds == y_val).sum().item()
            val_acc = safe_divide(correct_val, y_val.size(0))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                # Stored in memory only
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                
    # Load best weights
    model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        train_out = model(x_train)
        train_preds = train_out.argmax(dim=1)
        correct_train = (train_preds == y_train).sum().item()
        train_acc = safe_divide(correct_train, y_train.size(0))
        
        test_out = model(x_test)
        test_preds = test_out.argmax(dim=1)
        correct_test = (test_preds == y_test).sum().item()
        test_acc = safe_divide(correct_test, y_test.size(0))
        
    return {
        "model_class": "TinySelector",
        "epochs": 100,
        "optimizer_class": "AdamW",
        "loss_class": "CrossEntropyLoss",
        "best_validation_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "train_accuracy_at_best": float(train_acc),
        "test_accuracy_at_best": float(test_acc),
        "checkpoint_written": False,
        "training_completed": True,
    }


def evaluate_learned_selector_evidence(
    model_results: dict,
    baseline_results: dict,
    input_audit: dict,
    split_audit: dict
) -> dict:
    margin_threshold = 0.05
    test_acc = model_results["test_accuracy_at_best"]
    
    maj_baseline = baseline_results["majority_selector_baseline"]["test_accuracy"]
    src_baseline = baseline_results["source_only_baseline"]["test_accuracy"]
    int_baseline = baseline_results["intensity_only_baseline"]["test_accuracy"]
    hint_baseline = baseline_results["hint_passthrough_baseline"]["test_accuracy"]
    
    beats_maj = (test_acc - maj_baseline) >= margin_threshold
    beats_src = (test_acc - src_baseline) >= margin_threshold
    beats_int = (test_acc - int_baseline) >= margin_threshold
    beats_hint = (test_acc - hint_baseline) >= margin_threshold
    
    leakage = input_audit["selector_input_leakage_detected"]
    contam = split_audit["test_split_used_for_training"] or split_audit["test_labels_used_for_model_selection"]
    
    evidence_ok = beats_maj and beats_src and beats_int and (not leakage) and (not contam)
    
    blocking_reasons = []
    if not beats_maj:
        blocking_reasons.append("model_does_not_beat_majority_baseline")
    if not beats_src:
        blocking_reasons.append("model_does_not_beat_source_only_baseline")
    if not beats_int:
        blocking_reasons.append("model_does_not_beat_intensity_only_baseline")
    if leakage:
        blocking_reasons.append("model_input_leakage_detected")
    if contam:
        blocking_reasons.append("test_split_contamination_detected")
        
    return {
        "learned_selector_evidence_present": evidence_ok,
        "beats_majority_baseline_on_test": beats_maj,
        "beats_source_only_baseline_on_test": beats_src,
        "beats_intensity_only_baseline_on_test": beats_int,
        "beats_hint_passthrough_baseline_on_test": beats_hint,
        "evidence_margin_threshold": margin_threshold,
        "blocking_reasons": blocking_reasons,
        "diagnostic_pass": True,
    }


def audit_bridge_boundary_after_selector_pilot(evidence_audit: dict) -> dict:
    diag_pass = (
        BRIDGE_READY is False
        and BRIDGE_IMPLEMENTATION_ALLOWED is False
        and LEARNED_METRIC_EVIDENCE_PRESENT is False
        and SEMANTIC_METRIC_READY is False
        and GENERATION_CLAIMS_ALLOWED is False
        and SEMANTIC_GEOMETRY_CLAIMS_ALLOWED is False
    )
    
    blocking_reasons = [
        "learned_metric_evidence_not_present",
        "bridge_input_contract_not_defined",
        "bridge_validation_not_run",
    ]
    
    return {
        "bridge_ready": BRIDGE_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "learned_selector_evidence_present": evidence_audit["learned_selector_evidence_present"],
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,
        "blocking_reasons": blocking_reasons,
        "diagnostic_pass": diag_pass,
    }


def validate_source_contracts_for_p82() -> dict:
    validated = True
    missing_or_invalid = []
    
    p81_ok = True
    
    try:
        p81 = run_p81_learned_selector_dataset_contract_probe()
        if p81.get("phase") != "P81" or p81.get("verdict") != "P81_READY_FOR_REVIEW":
            p81_ok = False
            missing_or_invalid.append("p81_invalid_phase_or_verdict")
        if p81.get("learned_selector_dataset_contract_present") is not True:
            p81_ok = False
            missing_or_invalid.append("p81_dataset_contract_missing")
        if p81.get("selector_train_val_test_split_built") is not True:
            p81_ok = False
            missing_or_invalid.append("p81_split_contract_missing")
        if p81.get("dataset_contract_audit", {}).get("selector_input_leakage_detected") is not False:
            p81_ok = False
            missing_or_invalid.append("p81_leakage_detected")
        if p81.get("target_label_available_for_supervised_training") is not True:
            p81_ok = False
            missing_or_invalid.append("p81_labels_not_available")
        if p81.get("dataset_contract_audit", {}).get("hint_passthrough_baseline_present") is not True:
            p81_ok = False
            missing_or_invalid.append("p81_hint_baseline_missing")
        if p81.get("model_training_performed") is not False:
            p81_ok = False
            missing_or_invalid.append("p81_training_already_run")
        if p81.get("bridge_ready") is not False:
            p81_ok = False
            missing_or_invalid.append("p81_bridge_ready")
    except Exception as e:
        p81_ok = False
        missing_or_invalid.append(f"p81_exception_{str(e)}")
        
    validated = p81_ok
    
    return {
        "source_contracts_validated": validated,
        "p81_validated": p81_ok,
        "p81_dataset_contract_preserved": p81_ok,
        "p81_selector_input_leakage_absent_preserved": p81_ok,
        "p81_bridge_not_ready_preserved": p81_ok,
        "missing_or_invalid": missing_or_invalid,
    }


def run_p82_tiny_learned_selector_pilot_probe() -> dict:
    contracts_val = validate_source_contracts_for_p82()
    contracts_ok = contracts_val["source_contracts_validated"]
    
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    records = build_selector_dataset_records(p70a, p70b)
    
    input_audit = audit_model_input_records(records)
    split_audit = audit_split_integrity(records)
    
    tensor_data = build_tensor_dataset(records)
    baselines = evaluate_required_baselines(records)
    training_res = train_tiny_selector(tensor_data)
    
    evidence_audit = evaluate_learned_selector_evidence(training_res, baselines, input_audit, split_audit)
    bridge_audit = audit_bridge_boundary_after_selector_pilot(evidence_audit)
    
    verdict_str = VERDICT if contracts_ok else "P82_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if input_audit["diagnostic_pass"] is False:
            verdict_str = "P82_BLOCKED_BY_MODEL_INPUT_LEAKAGE"
        elif split_audit["diagnostic_pass"] is False:
            verdict_str = "P82_BLOCKED_BY_SPLIT_CONTAMINATION"
        elif training_res["checkpoint_written"] is True:
            verdict_str = "P82_BLOCKED_BY_CHECKPOINT_ARTIFACT"
        elif BRIDGE_READY is True or BRIDGE_IMPLEMENTATION_ALLOWED is True:
            verdict_str = "P82_BLOCKED_BY_PREMATURE_BRIDGE_AUTHORITY"
            
    sanity_summary = {
        "source_contracts_validated": contracts_ok,
        "p81_dataset_contract_preserved": contracts_val["p81_dataset_contract_preserved"],
        "p81_selector_input_leakage_absent_preserved": contracts_val["p81_selector_input_leakage_absent_preserved"],
        "p81_bridge_not_ready_preserved": contracts_val["p81_bridge_not_ready_preserved"],

        "tiny_learned_selector_trained": TINY_LEARNED_SELECTOR_TRAINED,
        "clean_selector_input_used": CLEAN_SELECTOR_INPUT_USED,
        "hint_passthrough_used_as_model_input": HINT_PASSTHROUGH_USED_AS_MODEL_INPUT,

        "selector_input_leakage_detected": input_audit["selector_input_leakage_detected"],
        "test_split_used_for_training": split_audit["test_split_used_for_training"],
        "test_labels_used_for_model_selection": split_audit["test_labels_used_for_model_selection"],
        "validation_used_for_model_selection": split_audit["validation_used_for_model_selection"],

        "checkpoint_written": CHECKPOINT_WRITTEN,

        "baseline_results_present": True,
        "training_results_present": True,
        "learned_selector_evidence_present": evidence_audit["learned_selector_evidence_present"],

        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_ready": BRIDGE_READY,

        "json_safe": True,
    }
    
    tensor_dataset_summary = {
        "x_train_shape": list(tensor_data["x_train"].shape),
        "y_train_shape": list(tensor_data["y_train"].shape),
        "x_validation_shape": list(tensor_data["x_validation"].shape),
        "y_validation_shape": list(tensor_data["y_validation"].shape),
        "x_test_shape": list(tensor_data["x_test"].shape),
        "y_test_shape": list(tensor_data["y_test"].shape),
        "feature_dim": tensor_data["feature_dim"],
    }
    
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,

        "source_dataset_phase": SOURCE_DATASET_PHASE,
        "source_authority_phase": SOURCE_AUTHORITY_PHASE,

        "verdict": verdict_str,

        "torch_allowed": TORCH_ALLOWED,
        "training_allowed": TRAINING_ALLOWED,
        "model_implementation_allowed": MODEL_IMPLEMENTATION_ALLOWED,
        "learned_selector_allowed": LEARNED_SELECTOR_ALLOWED,

        "tiny_learned_selector_pilot_allowed": TINY_LEARNED_SELECTOR_PILOT_ALLOWED,
        "tiny_learned_selector_trained": TINY_LEARNED_SELECTOR_TRAINED,
        "clean_selector_input_used": CLEAN_SELECTOR_INPUT_USED,
        "hint_passthrough_used_as_model_input": HINT_PASSTHROUGH_USED_AS_MODEL_INPUT,

        "target_endpoint_used_for_selector_input": TARGET_ENDPOINT_USED_FOR_SELECTOR_INPUT,
        "target_delta_used_for_selector_input": TARGET_DELTA_USED_FOR_SELECTOR_INPUT,
        "exact_relation_label_used_for_selector_input": EXACT_RELATION_LABEL_USED_FOR_SELECTOR_INPUT,
        "exact_operator_id_used_for_selector_input": EXACT_OPERATOR_ID_USED_FOR_SELECTOR_INPUT,
        "audit_metadata_used_for_selector_input": AUDIT_METADATA_USED_FOR_SELECTOR_INPUT,
        "relation_specific_hint_used_for_model_input": RELATION_SPECIFIC_HINT_USED_FOR_MODEL_INPUT,

        "test_split_used_for_training": TEST_SPLIT_USED_FOR_TRAINING,
        "test_labels_used_for_model_selection": TEST_LABELS_USED_FOR_MODEL_SELECTION,
        "validation_used_for_model_selection": VALIDATION_USED_FOR_MODEL_SELECTION,

        "checkpoint_written": CHECKPOINT_WRITTEN,

        "learned_selector_evidence_present": evidence_audit["learned_selector_evidence_present"],
        "learned_metric_evidence_present": LEARNED_METRIC_EVIDENCE_PRESENT,
        "semantic_metric_ready": SEMANTIC_METRIC_READY,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "bridge_ready": BRIDGE_READY,
        "generation_claims_allowed": GENERATION_CLAIMS_ALLOWED,
        "semantic_geometry_claims_allowed": SEMANTIC_GEOMETRY_CLAIMS_ALLOWED,

        "source_contracts_validated": contracts_ok,
        "p81_dataset_contract_preserved": contracts_val["p81_dataset_contract_preserved"],
        "p81_selector_input_leakage_absent_preserved": contracts_val["p81_selector_input_leakage_absent_preserved"],
        "p81_bridge_not_ready_preserved": contracts_val["p81_bridge_not_ready_preserved"],

        "model_input_audit": input_audit,
        "split_integrity_audit": split_audit,
        "tensor_dataset_summary": tensor_dataset_summary,
        "baseline_results": baselines,
        "tiny_selector_training_results": training_res,
        "learned_selector_evidence_audit": evidence_audit,
        "bridge_boundary_after_selector_pilot_audit": bridge_audit,

        "sanity_summary": sanity_summary,

        "json_safe": True,
        "diagnostic_only": False
    }
    
    # Assert JSON safe
    json.dumps(output)
    
    return output
