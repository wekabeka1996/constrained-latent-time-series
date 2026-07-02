# src/phase3/relation_contrastive_signal_smoke.py

import json
import math
from typing import Any

from src.phase3.baseline_point_offset_interpolation_harness import (
    l1_distance,
    l2_distance,
    max_abs_distance,
    run_p69_baseline_point_offset_interpolation_harness_probe,
)

from src.phase3.pure_numeric_relation_testbed import (
    run_p70a_pure_numeric_relation_testbed_probe,
)

from src.phase3.synthetic_time_series_relation_testbed import (
    run_p70b_synthetic_time_series_relation_testbed_probe,
)

PHASE = "P71"
PHASE_GROUP = "PHASE_3"
PHASE_NAME = "Relation Descriptor Contrastive Signal Smoke"
CONTRACT_VERSION = "phase3_p71_relation_descriptor_contrastive_signal_smoke_contract_v1"

SOURCE_BASELINE_PHASE = "P69"
SOURCE_VECTOR_TESTBED_PHASE = "P70A"
SOURCE_TIME_SERIES_TESTBED_PHASE = "P70B"

TRAINING_ALLOWED = False
MODEL_IMPLEMENTATION_ALLOWED = False
NEURAL_ENCODER_IMPLEMENTATION_ALLOWED = False
OPTIMIZATION_ALLOWED = False
TORCH_ALLOWED = False
NUMPY_ALLOWED = False
STOCHASTIC_RANDOM_ALLOWED = False
BRIDGE_IMPLEMENTATION_ALLOWED = False
LEARNED_METRIC_ALLOWED = False

DESCRIPTOR_CONSTRUCTION_ALLOWED = True
CONTRASTIVE_EVALUATION_ALLOWED = True
RELATION_LABELS_ALLOWED_FOR_EVALUATION_ONLY = True
RELATION_LABELS_ALLOWED_FOR_DESCRIPTOR_CONSTRUCTION = False

PRIMARY_EMPIRICAL_TARGET = "relation_signal_availability_under_contrastive_diagnostics"
VERDICT = "P71_READY_FOR_REVIEW"

DESCRIPTOR_VIEWS = [
    "p70a_vector_delta_descriptor",
    "p70b_parameter_delta_descriptor",
    "p70b_series_summary_delta_descriptor",
]

CONTROL_VIEWS = [
    "true_relation_labels",
    "assigned_permuted_relation_labels",
    "mismatched_pair_controls",
]

FORBIDDEN_CLAIMS = [
    "semantic_geometry_is_proven",
    "meaning_is_learned",
    "operator_identity_is_proven",
    "relation_encoder_is_trained",
    "relation_encoder_is_validated",
    "sparse_operator_bank_is_validated",
    "transfer_is_proven",
    "composition_is_proven",
    "bridge_method_is_validated",
    "schrodinger_bridge_creates_meaning",
    "geometric_schrodinger_bridge_creates_meaning",
    "contrastive_smoke_proves_semantics",
]

ALLOWED_CLAIMS = [
    "p71_evaluates_relation_signal_availability",
    "p71_uses_deterministic_relation_descriptors",
    "p71_uses_labels_only_for_evaluation",
    "p71_checks_label_permutation_damage",
    "p71_checks_mismatched_pair_controls",
    "p71_reports_descriptor_collision_warnings",
    "p71_does_not_train_models",
    "p71_does_not_establish_operator_level_evidence",
]


def assert_finite_vector(name: str, vector: list[float]) -> None:
    if not isinstance(vector, list):
        raise ValueError(f"Vector {name} must be a list.")
    if len(vector) == 0:
        raise ValueError(f"Vector {name} cannot be empty.")
    for i, val in enumerate(vector):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Element {i} in vector {name} is not numeric.")
        if not math.isfinite(val):
            raise ValueError(f"Element {i} in vector {name} is not finite.")


def assert_nonempty_records(name: str, records: list[dict]) -> None:
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"Records list '{name}' must be non-empty.")


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-15:
        return 0.0
    return float(numerator / denominator)


def encode_p70a_vector_delta_descriptor(z_a: list[float], z_b: list[float]) -> list[float]:
    assert_finite_vector("z_a", z_a)
    assert_finite_vector("z_b", z_b)
    if len(z_a) != len(z_b):
        raise ValueError("Dimensions mismatch in encode_p70a_vector_delta_descriptor.")
        
    delta = [float(y - x) for x, y in zip(z_a, z_b)]
    abs_delta = [float(abs(d)) for d in delta]
    
    l2_norm = math.sqrt(sum(d * d for d in delta))
    if l2_norm > 1e-15:
        direction = [float(d / l2_norm) for d in delta]
    else:
        direction = [0.0] * len(delta)
        
    return delta + abs_delta + direction


def encode_p70b_parameter_delta_descriptor(params_a: dict, params_b: dict) -> list[float]:
    keys = ["amplitude", "frequency", "phase", "volatility_envelope", "trend"]
    for k in keys:
        if k not in params_a or k not in params_b:
            raise ValueError(f"Missing parameter key '{k}'.")
            
    val_a = [float(params_a[k]) for k in keys]
    val_b = [float(params_b[k]) for k in keys]
    
    delta = [float(y - x) for x, y in zip(val_a, val_b)]
    abs_delta = [float(abs(d)) for d in delta]
    
    l2_norm = math.sqrt(sum(d * d for d in delta))
    if l2_norm > 1e-15:
        direction = [float(d / l2_norm) for d in delta]
    else:
        direction = [0.0] * len(delta)
        
    return delta + abs_delta + direction


def encode_p70b_series_summary_delta_descriptor(series_a: list[float], series_b: list[float]) -> list[float]:
    assert_finite_series = lambda s: [float(val) for val in s]
    
    def get_summary(s: list[float]) -> list[float]:
        assert_finite_vector("series", s)
        n = len(s)
        mean_val = sum(s) / n
        min_val = min(s)
        max_val = max(s)
        energy_val = sum(x * x for x in s)
        first_val = s[0]
        last_val = s[-1]
        return [mean_val, min_val, max_val, energy_val, first_val, last_val]
        
    sum_a = get_summary(series_a)
    sum_b = get_summary(series_b)
    
    delta = [float(y - x) for x, y in zip(sum_a, sum_b)]
    abs_delta = [float(abs(d)) for d in delta]
    
    l2_norm = math.sqrt(sum(d * d for d in delta))
    if l2_norm > 1e-15:
        direction = [float(d / l2_norm) for d in delta]
    else:
        direction = [0.0] * len(delta)
        
    return delta + abs_delta + direction


def descriptor_l2_distance(a: list[float], b: list[float]) -> float:
    assert_finite_vector("a", a)
    assert_finite_vector("b", b)
    if len(a) != len(b):
        raise ValueError("Dimensions mismatch in descriptor L2 distance.")
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def descriptor_close(a: list[float], b: list[float], tolerance: float = 1e-9) -> bool:
    assert_finite_vector("a", a)
    assert_finite_vector("b", b)
    if len(a) != len(b):
        raise ValueError("Dimensions mismatch in descriptor_close.")
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def extract_p70a_train_style_records(p70a_probe: dict) -> list[dict]:
    records = p70a_probe.get("testbed", {}).get("relation_records", [])
    return [rec for rec in records if rec.get("split") == "train_style_repeated_instances"]


def extract_p70b_train_style_records(p70b_probe: dict) -> list[dict]:
    records = p70b_probe.get("testbed", {}).get("relation_records", [])
    return [rec for rec in records if rec.get("split") == "train_style_repeated_instances"]


def extract_p70b_negative_control_records(p70b_probe: dict) -> list[dict]:
    return p70b_probe.get("testbed", {}).get("negative_control_records", [])


def build_p70a_descriptor_records(p70a_records: list[dict]) -> list[dict]:
    assert_nonempty_records("p70a_records", p70a_records)
    out = []
    for rec in p70a_records:
        z_a = rec["z_a"]
        z_b = rec["z_b"]
        descriptor = encode_p70a_vector_delta_descriptor(z_a, z_b)
        out.append({
            "record_id": rec["record_id"],
            "relation_type": rec["relation_type"],
            "descriptor_view": "p70a_vector_delta_descriptor",
            "descriptor": descriptor,
        })
    return out


def build_p70b_parameter_descriptor_records(p70b_records: list[dict]) -> list[dict]:
    assert_nonempty_records("p70b_records", p70b_records)
    out = []
    for rec in p70b_records:
        params_a = rec["params_a"]
        params_b = rec["params_b"]
        descriptor = encode_p70b_parameter_delta_descriptor(params_a, params_b)
        out.append({
            "record_id": rec["record_id"],
            "relation_type": rec["relation_type"],
            "descriptor_view": "p70b_parameter_delta_descriptor",
            "descriptor": descriptor,
        })
    return out


def build_p70b_series_summary_descriptor_records(p70b_records: list[dict]) -> list[dict]:
    assert_nonempty_records("p70b_records", p70b_records)
    out = []
    for rec in p70b_records:
        series_a = rec["series_a"]
        series_b = rec["series_b"]
        descriptor = encode_p70b_series_summary_delta_descriptor(series_a, series_b)
        out.append({
            "record_id": rec["record_id"],
            "relation_type": rec["relation_type"],
            "descriptor_view": "p70b_series_summary_delta_descriptor",
            "descriptor": descriptor,
        })
    return out


def evaluate_contrastive_descriptor_view(
    descriptor_records: list[dict],
    label_key: str = "relation_type",
    collision_tolerance: float = 1e-9,
) -> dict:
    assert_nonempty_records("descriptor_records", descriptor_records)
    
    n = len(descriptor_records)
    relation_types = set(rec[label_key] for rec in descriptor_records)
    
    pos_distances = []
    neg_distances = []
    collision_count = 0
    
    # Track metrics by relation type
    pos_distances_by_type = {}
    
    for i in range(n):
        rec_i = descriptor_records[i]
        label_i = rec_i[label_key]
        for j in range(i + 1, n):
            rec_j = descriptor_records[j]
            label_j = rec_j[label_key]
            
            dist = descriptor_l2_distance(rec_i["descriptor"], rec_j["descriptor"])
            
            if label_i == label_j:
                pos_distances.append(dist)
                if label_i not in pos_distances_by_type:
                    pos_distances_by_type[label_i] = []
                pos_distances_by_type[label_i].append(dist)
            else:
                neg_distances.append(dist)
                if dist <= collision_tolerance:
                    collision_count += 1
                    
    mean_pos = safe_divide(sum(pos_distances), len(pos_distances))
    mean_neg = safe_divide(sum(neg_distances), len(neg_distances))
    margin = float(mean_neg - mean_pos)
    
    # Nearest Top-1 Accuracy
    success_count = 0
    for i in range(n):
        rec_i = descriptor_records[i]
        label_i = rec_i[label_key]
        
        min_dist = float("inf")
        nearest_label = None
        for j in range(n):
            if j == i:
                continue
            rec_j = descriptor_records[j]
            dist = descriptor_l2_distance(rec_i["descriptor"], rec_j["descriptor"])
            if dist < min_dist:
                min_dist = dist
                nearest_label = rec_j[label_key]
                
        if nearest_label == label_i:
            success_count += 1
            
    accuracy = safe_divide(success_count, n)
    
    # Metrics by type
    same_relation_pair_count_by_type = {}
    mean_positive_l2_by_type = {}
    for r_type in relation_types:
        dists = pos_distances_by_type.get(r_type, [])
        same_relation_pair_count_by_type[r_type] = len(dists)
        mean_positive_l2_by_type[r_type] = safe_divide(sum(dists), len(dists))
        
    # Diagnostic pass
    diag_pass = (
        len(pos_distances) > 0
        and len(neg_distances) > 0
        and margin > 0.0
        and accuracy > 0.5
    )
    
    return {
        "record_count": int(n),
        "relation_type_count": len(relation_types),
        "positive_pair_count": len(pos_distances),
        "negative_pair_count": len(neg_distances),
        "mean_positive_l2": float(mean_pos),
        "mean_negative_l2": float(mean_neg),
        "separation_margin": margin,
        "nearest_same_relation_top1_accuracy": float(accuracy),
        "different_relation_collision_count": int(collision_count),
        "same_relation_pair_count_by_type": same_relation_pair_count_by_type,
        "mean_positive_l2_by_type": mean_positive_l2_by_type,
        "diagnostic_pass": diag_pass,
    }


def evaluate_p70b_label_permutation_damage(
    p70b_negative_controls: list[dict],
) -> dict:
    perms = [n for n in p70b_negative_controls if n.get("negative_control_type") == "label_permutation"]
    if len(perms) == 0:
        raise ValueError("No label permutation negative controls found.")
        
    all_differ = all(p["true_relation_type"] != p["assigned_relation_type"] for p in perms)
    all_nondeg = all(p.get("nondegenerate") is True for p in perms)
    
    # Calculate collision under assigned labels
    # Build parameter descriptors for the permuted records
    desc_records = []
    for idx, p in enumerate(perms):
        desc = encode_p70b_parameter_delta_descriptor(p["params_a"], p["params_b"])
        desc_records.append({
            "record_id": p.get("record_id", f"neg_{idx}"),
            "assigned_relation_type": p["assigned_relation_type"],
            "descriptor": desc
        })
        
    eval_res = evaluate_contrastive_descriptor_view(desc_records, label_key="assigned_relation_type")
    
    return {
        "label_permutation_record_count": len(perms),
        "all_true_labels_differ_from_assigned": all_differ,
        "all_records_nondegenerate": all_nondeg,
        "assigned_label_collision_count": eval_res["different_relation_collision_count"],
        "damage_expected": True,
    }


def evaluate_p70b_mismatched_pair_controls(
    p70b_negative_controls: list[dict],
) -> dict:
    mismatches = [n for n in p70b_negative_controls if n.get("negative_control_type") == "mismatched_pair"]
    if len(mismatches) == 0:
        raise ValueError("No mismatched pair negative controls found.")
        
    all_nondeg = all(m.get("nondegenerate") is True for m in mismatches)
    all_expected_fail = all(m.get("expected_to_fail_relation_identity") is True for m in mismatches)
    
    return {
        "mismatched_pair_record_count": len(mismatches),
        "all_records_nondegenerate": all_nondeg,
        "all_expected_to_fail_relation_identity": all_expected_fail,
    }


def validate_source_contracts() -> dict:
    # 1. P69
    p69_validated = True
    p69_res = run_p69_baseline_point_offset_interpolation_harness_probe()
    if p69_res.get("phase") != "P69" or p69_res.get("verdict") != "P69_READY_FOR_REVIEW":
        p69_validated = False
        
    # 2. P70A
    p70a_validated = True
    p70a_res = run_p70a_pure_numeric_relation_testbed_probe()
    if p70a_res.get("phase") != "P70A" or p70a_res.get("verdict") != "P70A_READY_FOR_REVIEW":
        p70a_validated = False
        
    # 3. P70B
    p70b_validated = True
    p70b_res = run_p70b_synthetic_time_series_relation_testbed_probe()
    if p70b_res.get("phase") != "P70B" or p70b_res.get("verdict") != "P70B_READY_FOR_REVIEW":
        p70b_validated = False
        
    source_valid = p69_validated and p70a_validated and p70b_validated
    
    return {
        "source_contracts_validated": source_valid,
        "p69_validated": p69_validated,
        "p70a_validated": p70a_validated,
        "p70b_validated": p70b_validated,
        "missing_or_invalid": [] if source_valid else ["contracts_failed"]
    }


def run_p71_relation_contrastive_signal_smoke_probe() -> dict:
    # 1. Validate contracts
    contract_res = validate_source_contracts()
    contracts_valid = contract_res["source_contracts_validated"]
    
    # 2. Load Probes
    p70a_probe = run_p70a_pure_numeric_relation_testbed_probe()
    p70b_probe = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    # 3. Extract Records
    p70a_records = extract_p70a_train_style_records(p70a_probe)
    p70b_records = extract_p70b_train_style_records(p70b_probe)
    p70b_negatives = extract_p70b_negative_control_records(p70b_probe)
    
    # 4. Build descriptors
    p70a_desc_recs = build_p70a_descriptor_records(p70a_records)
    p70b_param_desc_recs = build_p70b_parameter_descriptor_records(p70b_records)
    p70b_series_desc_recs = build_p70b_series_summary_descriptor_records(p70b_records)
    
    # 5. Evaluate contrastive views
    eval_p70a = evaluate_contrastive_descriptor_view(p70a_desc_recs)
    eval_p70b_param = evaluate_contrastive_descriptor_view(p70b_param_desc_recs)
    eval_p70b_series = evaluate_contrastive_descriptor_view(p70b_series_desc_recs)
    
    # 6. Evaluate Permutation & Mismatches
    eval_perm = evaluate_p70b_label_permutation_damage(p70b_negatives)
    eval_mism = evaluate_p70b_mismatched_pair_controls(p70b_negatives)
    
    # 7. Check passes
    p70b_param_strong = (
        eval_p70b_param["separation_margin"] > 0.0
        and eval_p70b_param["nearest_same_relation_top1_accuracy"] >= 0.8
        and eval_p70b_param["different_relation_collision_count"] == 0
    )
    
    p70a_ambiguity = eval_p70a["different_relation_collision_count"] > 0
    
    # Formulate verdict
    verdict_str = VERDICT if contracts_valid else "P71_BLOCKED_BY_SOURCE_CONTRACT"
    if verdict_str == VERDICT:
        if not p70b_param_strong:
            verdict_str = "P71_BLOCKED_BY_CONTRASTIVE_SIGNAL_SANITY"
            
    sanity_summary = {
        "source_contracts_validated": contracts_valid,
        "all_descriptor_views_present": True,
        "p70b_parameter_descriptor_strong_pass": p70b_param_strong,
        "p70b_series_summary_descriptor_diagnostic_pass": eval_p70b_series["diagnostic_pass"],
        "p70a_vector_descriptor_diagnostic_pass": eval_p70a["diagnostic_pass"],
        "p70a_descriptor_ambiguity_warning": p70a_ambiguity,
        "label_permutation_records_ready": eval_perm["label_permutation_record_count"] > 0,
        "mismatched_pair_records_ready": eval_mism["mismatched_pair_record_count"] > 0,
        "labels_used_for_descriptor_construction": False,
        "labels_used_for_evaluation_only": True,
        "json_safe": True,
    }
    
    output = {
        "phase": PHASE,
        "phase_group": PHASE_GROUP,
        "phase_name": PHASE_NAME,
        "contract_version": CONTRACT_VERSION,
        
        "source_baseline_phase": SOURCE_BASELINE_PHASE,
        "source_vector_testbed_phase": SOURCE_VECTOR_TESTBED_PHASE,
        "source_time_series_testbed_phase": SOURCE_TIME_SERIES_TESTBED_PHASE,
        
        "verdict": verdict_str,
        
        "training_allowed": TRAINING_ALLOWED,
        "model_implementation_allowed": MODEL_IMPLEMENTATION_ALLOWED,
        "neural_encoder_implementation_allowed": NEURAL_ENCODER_IMPLEMENTATION_ALLOWED,
        "optimization_allowed": OPTIMIZATION_ALLOWED,
        "torch_allowed": TORCH_ALLOWED,
        "numpy_allowed": NUMPY_ALLOWED,
        "stochastic_random_allowed": STOCHASTIC_RANDOM_ALLOWED,
        "bridge_implementation_allowed": BRIDGE_IMPLEMENTATION_ALLOWED,
        "learned_metric_allowed": LEARNED_METRIC_ALLOWED,
        
        "descriptor_construction_allowed": DESCRIPTOR_CONSTRUCTION_ALLOWED,
        "contrastive_evaluation_allowed": CONTRASTIVE_EVALUATION_ALLOWED,
        "relation_labels_allowed_for_evaluation_only": RELATION_LABELS_ALLOWED_FOR_EVALUATION_ONLY,
        "relation_labels_allowed_for_descriptor_construction": RELATION_LABELS_ALLOWED_FOR_DESCRIPTOR_CONSTRUCTION,
        
        "primary_empirical_target": PRIMARY_EMPIRICAL_TARGET,
        "descriptor_views": DESCRIPTOR_VIEWS,
        "control_views": CONTROL_VIEWS,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "allowed_claims": ALLOWED_CLAIMS,
        
        "source_contracts_validated": contracts_valid,
        
        "descriptor_evaluations": {
            "p70a_vector_delta_descriptor": eval_p70a,
            "p70b_parameter_delta_descriptor": eval_p70b_param,
            "p70b_series_summary_delta_descriptor": eval_p70b_series,
        },
        
        "label_permutation_evaluation": eval_perm,
        "mismatched_pair_evaluation": eval_mism,
        
        "sanity_summary": sanity_summary,
        "p70a_descriptor_ambiguity_warning": p70a_ambiguity,
        "json_safe": True,
        "diagnostic_only": True
    }
    
    # Assert JSON safe
    json.dumps(output)
    
    return output
