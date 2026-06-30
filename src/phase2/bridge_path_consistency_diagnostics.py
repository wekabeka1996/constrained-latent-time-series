# src/phase2/bridge_path_consistency_diagnostics.py

import json
import math
from typing import Any, Dict, Tuple

from src.phase2.torch_boundary import (
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)
from src.phase2.deterministic_endpoint_bridge_targets import (
    build_p49_endpoint_raw_tensors,
    build_endpoint_signatures,
)
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    P50_STEP_COUNT,
    P50_LEARNING_RATE,
    build_p50_bridge_targets,
    build_combined_signature_from_raw,
)
from src.phase2.post_fit_bridge_candidate_diagnostics import (
    run_single_p51_fit_with_final_raw,
)

# Constants
BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_CONTRACT_VERSION = "phase2_p52_bridge_path_consistency_diagnostics_contract_v1"
BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_KIND = "bridge_path_consistency_diagnostics_no_model_no_vae_no_gsb_no_science"
BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_MODULE_NAME = "src.phase2.bridge_path_consistency_diagnostics"
P52_TARGET_COUNT = 3
P52_LAMBDA_VALUES = (0.25, 0.50, 0.75)


def validate_non_empty_str(val: str, name: str) -> None:
    if type(val) is not str:
        raise TypeError(f"{name} must be exact str instance")
    if not val.strip():
        raise ValueError(f"{name} cannot be empty or whitespace only")


def assert_no_local_path_leakage(val: str, name: str = "field") -> None:
    forbidden = [":\\", "Users", "home", "/Users", "/home", ".git"]
    val_lower = val.lower()
    for item in forbidden:
        if item.lower() in val_lower:
            raise ValueError(f"Local path leak detected in {name}")


def assert_no_forbidden_claims(val: str, name: str = "field") -> None:
    val_lower = val.lower()
    if (("scientific" in val_lower) and ("success" in val_lower)) or \
       ("solved" in val_lower) or \
       ("best" in val_lower) or \
       ("winner" in val_lower) or \
       (("production" in val_lower) and ("ready" in val_lower)) or \
       (("state" in val_lower) and ("art" in val_lower)) or \
       (("semantic" in val_lower) and ("geometry" in val_lower)) or \
       (("synthetic" in val_lower) and ("state" in val_lower)) or \
       (("gsb" in val_lower) and ("solved" in val_lower)) or \
       (("model" in val_lower) and ("learned" in val_lower)) or \
       (("vae" in val_lower) and ("works" in val_lower)) or \
       (("schrodinger" in val_lower) and ("bridge" in val_lower)) or \
       (("generated" in val_lower) and ("c" in val_lower)):
        raise ValueError(f"Forbidden claim detected in {name}")


def load_torch_for_p52_path_consistency() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P52 loader.")
    import torch
    return torch


def signature_tensor_field_names(signature: dict) -> Tuple[str, ...]:
    torch = load_torch_for_p52_path_consistency()
    fields = []
    for k, v in signature.items():
        if torch.is_tensor(v):
            fields.append(k)
    return tuple(sorted(fields))


def flatten_signature_tensor_fields(signature: dict, field_names: Tuple[str, ...]) -> Any:
    torch = load_torch_for_p52_path_consistency()
    vectors = []
    for name in field_names:
        if name not in signature:
            raise KeyError(f"Field {name} not found in signature")
        val = signature[name]
        if not torch.is_tensor(val):
            raise TypeError(f"Field {name} is not a PyTorch tensor")
        vectors.append(val.flatten().detach().clone())
    return torch.cat(vectors, dim=0)


def build_endpoint_signature_vectors() -> dict:
    raw_endpoints = build_p49_endpoint_raw_tensors()
    signatures = build_endpoint_signatures(raw_endpoints)
    sig_A = signatures["endpoint_A"]
    sig_B = signatures["endpoint_B"]
    
    field_names = signature_tensor_field_names(sig_A)
    field_names_B = signature_tensor_field_names(sig_B)
    if field_names != field_names_B:
        raise ValueError("Endpoints A and B signature fields mismatch")
        
    vec_A = flatten_signature_tensor_fields(sig_A, field_names)
    vec_B = flatten_signature_tensor_fields(sig_B, field_names)
    
    return {
        "endpoint_A_signature": sig_A,
        "endpoint_B_signature": sig_B,
        "endpoint_A_vector": vec_A,
        "endpoint_B_vector": vec_B,
        "field_names": field_names,
        "vector_length": len(vec_A),
    }


def build_p52_post_fit_candidate_signature(target: dict) -> dict:
    fit_res = run_single_p51_fit_with_final_raw(target)
    final_raw = fit_res.pop("final_candidate_raw")
    final_sig = build_combined_signature_from_raw(final_raw)
    
    return {
        "target_id": fit_res["target_id"],
        "lambda_value": fit_res["lambda_value"],
        "final_signature": final_sig,
        "initial_loss_value": fit_res["initial_loss_value"],
        "final_loss_value": fit_res["final_loss_value"],
        "loss_delta_value": fit_res["loss_delta_value"],
        "loss_decreased": fit_res["loss_decreased"],
        "initial_loss_finite": fit_res["initial_loss_finite"],
        "final_loss_finite": fit_res["final_loss_finite"],
        "step_count": int(P50_STEP_COUNT),
        "learning_rate": float(P50_LEARNING_RATE),
        "construction_method": "linear_interpolation_in_p45_combined_signature_space",
        "realizability_claim": "path_consistency_diagnostics_only_not_state_validity",
    }


def compute_l2_distance(vec_a: Any, vec_b: Any) -> float:
    torch = load_torch_for_p52_path_consistency()
    dist = torch.sqrt(torch.sum((vec_a - vec_b) ** 2))
    return float(dist.item())


def compute_projection_position(endpoint_A_vec: Any, endpoint_B_vec: Any, candidate_vec: Any) -> float:
    torch = load_torch_for_p52_path_consistency()
    diff_endpoints = endpoint_B_vec - endpoint_A_vec
    diff_cand = candidate_vec - endpoint_A_vec
    
    num = torch.dot(diff_cand, diff_endpoints)
    denom = torch.dot(diff_endpoints, diff_endpoints)
    
    if denom.item() <= 0.0 or not math.isfinite(denom.item()):
        raise ValueError("Projection unavailable due to zero or non-finite endpoint distance")
        
    t_hat = num / denom
    return float(t_hat.item())


def run_single_p52_path_point_diagnostics(target: dict, endpoint_vectors: dict) -> dict:
    cand_res = build_p52_post_fit_candidate_signature(target)
    
    vec_A = endpoint_vectors["endpoint_A_vector"]
    vec_B = endpoint_vectors["endpoint_B_vector"]
    field_names = endpoint_vectors["field_names"]
    
    vec_cand = flatten_signature_tensor_fields(cand_res["final_signature"], field_names)
    vec_target = flatten_signature_tensor_fields(target, field_names)
    
    dist_A = compute_l2_distance(vec_cand, vec_A)
    dist_B = compute_l2_distance(vec_cand, vec_B)
    dist_target = compute_l2_distance(vec_cand, vec_target)
    
    t_hat = compute_projection_position(vec_A, vec_B, vec_cand)
    
    proj_finite = math.isfinite(t_hat)
    dists_finite = math.isfinite(dist_A) and math.isfinite(dist_B) and math.isfinite(dist_target)
    
    lambda_val = cand_res["lambda_value"]
    proj_diff = t_hat - lambda_val
    abs_proj_err = abs(proj_diff)
    
    return {
        "target_id": cand_res["target_id"],
        "lambda_value": float(lambda_val),
        "projection_position_value": float(t_hat),
        "projection_position_finite": bool(proj_finite),
        "projection_minus_lambda_value": float(proj_diff),
        "abs_projection_error_value": float(abs_proj_err),
        "distance_to_endpoint_A_value": float(dist_A),
        "distance_to_endpoint_B_value": float(dist_B),
        "distance_to_own_target_value": float(dist_target),
        "distances_finite": bool(dists_finite),
        "initial_loss_value": float(cand_res["initial_loss_value"]),
        "final_loss_value": float(cand_res["final_loss_value"]),
        "loss_delta_value": float(cand_res["loss_delta_value"]),
        "loss_decreased": bool(cand_res["loss_decreased"]),
        "step_count": int(cand_res["step_count"]),
        "learning_rate": float(cand_res["learning_rate"]),
        "signature_vector_length": int(endpoint_vectors["vector_length"]),
        "signature_tensor_field_count": len(field_names),
        "construction_method": cand_res["construction_method"],
        "realizability_claim": "path_consistency_diagnostics_only_not_state_validity",
        "no_state_validity_claim": True,
    }


def is_strictly_increasing(values: Tuple[float, ...]) -> bool:
    if len(values) < 2:
        return True
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def is_non_decreasing(values: Tuple[float, ...]) -> bool:
    if len(values) < 2:
        return True
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def is_non_increasing(values: Tuple[float, ...]) -> bool:
    if len(values) < 2:
        return True
    return all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def run_bridge_path_consistency_diagnostics_probe() -> dict:
    try:
        torch = load_torch_for_p52_path_consistency()
    except Exception:
        return {
            "contract_version": BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_CONTRACT_VERSION,
            "status": "blocked_torch_unavailable",
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    endpoint_vectors = build_endpoint_signature_vectors()
    targets = build_p50_bridge_targets()
    
    path_points = []
    for t in targets:
        point = run_single_p52_path_point_diagnostics(t, endpoint_vectors)
        path_points.append(point)
        
    path_points.sort(key=lambda x: x["lambda_value"])
    
    proj_positions = tuple(p["projection_position_value"] for p in path_points)
    dist_A = tuple(p["distance_to_endpoint_A_value"] for p in path_points)
    dist_B = tuple(p["distance_to_endpoint_B_value"] for p in path_points)
    own_target_dists = tuple(p["distance_to_own_target_value"] for p in path_points)
    
    proj_positions_finite = all(math.isfinite(x) for x in proj_positions)
    own_target_distances_finite = all(math.isfinite(x) for x in own_target_dists)
    all_targets_loss_decreased = all(p["loss_decreased"] for p in path_points)
    all_losses_finite = all(math.isfinite(p["initial_loss_value"]) and math.isfinite(p["final_loss_value"]) for p in path_points)
    
    proj_strictly_increasing = is_strictly_increasing(proj_positions)
    proj_non_decreasing = is_non_decreasing(proj_positions)
    
    dist_A_non_decreasing = is_non_decreasing(dist_A)
    dist_B_non_increasing = is_non_increasing(dist_B)
    
    path_consistency_passed = (
        proj_positions_finite and
        proj_strictly_increasing and
        own_target_distances_finite and
        all_targets_loss_decreased and
        all_losses_finite
    )
    
    summary = {
        "contract_version": BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_CONTRACT_VERSION,
        "kind": BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_KIND,
        "status": "bridge_path_consistency_diagnostics_available_no_model_no_vae_no_gsb_no_science",
        "torch_available": True,
        "bridge_target_count": len(targets),
        "lambda_values": [p["lambda_value"] for p in path_points],
        "signature_vector_length": endpoint_vectors["vector_length"],
        "signature_tensor_field_count": len(endpoint_vectors["field_names"]),
        "projection_positions": list(proj_positions),
        "projection_positions_finite": proj_positions_finite,
        "projection_positions_strictly_increasing": proj_strictly_increasing,
        "projection_positions_non_decreasing": proj_non_decreasing,
        "distance_to_endpoint_A_values": list(dist_A),
        "distance_to_endpoint_A_non_decreasing": dist_A_non_decreasing,
        "distance_to_endpoint_B_values": list(dist_B),
        "distance_to_endpoint_B_non_increasing": dist_B_non_increasing,
        "own_target_distances": list(own_target_dists),
        "own_target_distances_finite": own_target_distances_finite,
        "all_targets_loss_decreased": all_targets_loss_decreased,
        "all_losses_finite": all_losses_finite,
        "path_points": path_points,
        "path_consistency_passed": path_consistency_passed,
        "no_model": True,
        "no_vae": True,
        "no_encoder": True,
        "no_decoder": True,
        "no_dataset": True,
        "no_dataloader": True,
        "no_torch_optimizer": True,
        "no_gsb_claim": True,
        "no_generation_claim": True,
        "no_scientific_conclusion": True,
        "realizability_claim": "path_consistency_diagnostics_only_not_state_validity",
        "reason": "p52_bridge_path_consistency_diagnostics_probe_success"
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def bridge_path_consistency_diagnostics_probe_to_json_dict(probe_res: dict) -> dict:
    validate_non_empty_str(probe_res["contract_version"], "contract_version")
    validate_non_empty_str(probe_res["status"], "status")
    validate_non_empty_str(probe_res["reason"], "reason")
    assert_no_local_path_leakage(probe_res["reason"], "reason")
    assert_no_forbidden_claims(probe_res["reason"], "reason")
    
    return {
        "contract_version": probe_res["contract_version"],
        "kind": probe_res.get("kind", BRIDGE_PATH_CONSISTENCY_DIAGNOSTICS_KIND),
        "status": probe_res["status"],
        "torch_available": bool(probe_res["torch_available"]),
        "bridge_target_count": int(probe_res.get("bridge_target_count", 0)),
        "lambda_values": list(probe_res.get("lambda_values", [])),
        "signature_vector_length": int(probe_res.get("signature_vector_length", 0)),
        "signature_tensor_field_count": int(probe_res.get("signature_tensor_field_count", 0)),
        "projection_positions": list(probe_res.get("projection_positions", [])),
        "projection_positions_finite": bool(probe_res.get("projection_positions_finite", False)),
        "projection_positions_strictly_increasing": bool(probe_res.get("projection_positions_strictly_increasing", False)),
        "projection_positions_non_decreasing": bool(probe_res.get("projection_positions_non_decreasing", False)),
        "distance_to_endpoint_A_values": list(probe_res.get("distance_to_endpoint_A_values", [])),
        "distance_to_endpoint_A_non_decreasing": bool(probe_res.get("distance_to_endpoint_A_non_decreasing", False)),
        "distance_to_endpoint_B_values": list(probe_res.get("distance_to_endpoint_B_values", [])),
        "distance_to_endpoint_B_non_increasing": bool(probe_res.get("distance_to_endpoint_B_non_increasing", False)),
        "own_target_distances": list(probe_res.get("own_target_distances", [])),
        "own_target_distances_finite": bool(probe_res.get("own_target_distances_finite", False)),
        "all_targets_loss_decreased": bool(probe_res.get("all_targets_loss_decreased", False)),
        "all_losses_finite": bool(probe_res.get("all_losses_finite", False)),
        "path_points": probe_res.get("path_points", []),
        "path_consistency_passed": bool(probe_res.get("path_consistency_passed", False)),
        "no_model": bool(probe_res.get("no_model", True)),
        "no_vae": bool(probe_res.get("no_vae", True)),
        "no_encoder": bool(probe_res.get("no_encoder", True)),
        "no_decoder": bool(probe_res.get("no_decoder", True)),
        "no_dataset": bool(probe_res.get("no_dataset", True)),
        "no_dataloader": bool(probe_res.get("no_dataloader", True)),
        "no_torch_optimizer": bool(probe_res.get("no_torch_optimizer", True)),
        "no_gsb_claim": bool(probe_res.get("no_gsb_claim", True)),
        "no_generation_claim": bool(probe_res.get("no_generation_claim", True)),
        "no_scientific_conclusion": bool(probe_res.get("no_scientific_conclusion", True)),
        "realizability_claim": probe_res.get("realizability_claim", "path_consistency_diagnostics_only_not_state_validity"),
        "reason": probe_res["reason"],
    }


def compact_bridge_path_consistency_diagnostics_json(probe_res: dict) -> str:
    d = bridge_path_consistency_diagnostics_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
