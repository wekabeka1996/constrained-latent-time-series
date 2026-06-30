# src/phase2/bridge_path_residual_curvature_diagnostics.py

import json
import math
from typing import Any, Dict, Tuple

from src.phase2.torch_boundary import (
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    P50_STEP_COUNT,
    P50_LEARNING_RATE,
    build_p50_bridge_targets,
)
from src.phase2.bridge_path_consistency_diagnostics import (
    load_torch_for_p52_path_consistency,
    build_endpoint_signature_vectors,
    build_p52_post_fit_candidate_signature,
    flatten_signature_tensor_fields,
    compute_l2_distance,
    compute_projection_position,
)

# Constants
BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_CONTRACT_VERSION = "phase2_p53_bridge_path_residual_curvature_diagnostics_contract_v1"
BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_KIND = "bridge_path_residual_curvature_diagnostics_no_model_no_vae_no_gsb_no_science"
BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_MODULE_NAME = "src.phase2.bridge_path_residual_curvature_diagnostics"
P53_TARGET_COUNT = 3
P53_LAMBDA_VALUES = (0.25, 0.50, 0.75)


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


def load_torch_for_p53_residual_curvature() -> Any:
    return load_torch_for_p52_path_consistency()


def build_p53_path_vectors() -> dict:
    torch = load_torch_for_p53_residual_curvature()
    
    endpoint_vectors = build_endpoint_signature_vectors()
    vec_A = endpoint_vectors["endpoint_A_vector"]
    vec_B = endpoint_vectors["endpoint_B_vector"]
    field_names = endpoint_vectors["field_names"]
    
    endpoint_dist = compute_l2_distance(vec_A, vec_B)
    dir_vec = vec_B - vec_A
    
    targets = build_p50_bridge_targets()
    
    path_points = []
    for t in targets:
        cand_sig = build_p52_post_fit_candidate_signature(t)
        
        vec_cand = flatten_signature_tensor_fields(cand_sig["final_signature"], field_names)
        vec_target = flatten_signature_tensor_fields(t, field_names)
        
        path_points.append({
            "target_id": cand_sig["target_id"],
            "lambda_value": float(cand_sig["lambda_value"]),
            "candidate_vector": vec_cand,
            "own_target_vector": vec_target,
            "initial_loss_value": float(cand_sig["initial_loss_value"]),
            "final_loss_value": float(cand_sig["final_loss_value"]),
            "loss_delta_value": float(cand_sig["loss_delta_value"]),
            "loss_decreased": bool(cand_sig["loss_decreased"]),
            "step_count": int(cand_sig["step_count"]),
            "learning_rate": float(cand_sig["learning_rate"]),
            "construction_method": cand_sig["construction_method"],
            "realizability_claim": cand_sig["realizability_claim"],
        })
        
    path_points.sort(key=lambda x: x["lambda_value"])
    
    return {
        "endpoint_A_vector": vec_A,
        "endpoint_B_vector": vec_B,
        "endpoint_direction_vector": dir_vec,
        "endpoint_distance_value": float(endpoint_dist),
        "field_names": field_names,
        "signature_vector_length": len(vec_A),
        "path_points_internal": path_points,
    }


def compute_line_closest_point(endpoint_A_vec: Any, endpoint_B_vec: Any, projection_position: float) -> Any:
    return endpoint_A_vec + projection_position * (endpoint_B_vec - endpoint_A_vec)


def run_single_p53_residual_point_diagnostics(point: dict, path_vectors: dict) -> dict:
    torch = load_torch_for_p53_residual_curvature()
    
    vec_A = path_vectors["endpoint_A_vector"]
    vec_B = path_vectors["endpoint_B_vector"]
    field_names = path_vectors["field_names"]
    endpoint_dist = path_vectors["endpoint_distance_value"]
    
    vec_cand = point["candidate_vector"]
    vec_target = point["own_target_vector"]
    
    t_hat = compute_projection_position(vec_A, vec_B, vec_cand)
    closest = compute_line_closest_point(vec_A, vec_B, t_hat)
    
    residual = compute_l2_distance(vec_cand, closest)
    dist_target = compute_l2_distance(vec_cand, vec_target)
    
    proj_finite = math.isfinite(t_hat)
    res_finite = math.isfinite(residual)
    dist_target_finite = math.isfinite(dist_target)
    
    if endpoint_dist > 0.0 and math.isfinite(endpoint_dist):
        norm_res = residual / endpoint_dist
    else:
        norm_res = float('nan')
        
    lambda_val = point["lambda_value"]
    proj_diff = t_hat - lambda_val
    abs_proj_err = abs(proj_diff)
    
    return {
        "target_id": point["target_id"],
        "lambda_value": float(lambda_val),
        "projection_position_value": float(t_hat),
        "projection_position_finite": bool(proj_finite),
        "projection_minus_lambda_value": float(proj_diff),
        "abs_projection_error_value": float(abs_proj_err),
        "line_residual_value": float(residual),
        "normalized_line_residual_value": float(norm_res) if math.isfinite(norm_res) else None,
        "line_residual_finite": bool(res_finite),
        "distance_to_own_target_value": float(dist_target),
        "distance_to_own_target_finite": bool(dist_target_finite),
        "initial_loss_value": float(point["initial_loss_value"]),
        "final_loss_value": float(point["final_loss_value"]),
        "loss_delta_value": float(point["loss_delta_value"]),
        "loss_decreased": bool(point["loss_decreased"]),
        "step_count": int(point["step_count"]),
        "learning_rate": float(point["learning_rate"]),
        "signature_vector_length": int(path_vectors["signature_vector_length"]),
        "signature_tensor_field_count": len(field_names),
        "construction_method": point["construction_method"],
        "realizability_claim": "residual_curvature_diagnostics_only_not_state_validity",
        "no_state_validity_claim": True,
    }


def run_p53_curvature_smoothness_diagnostics(path_vectors: dict) -> dict:
    torch = load_torch_for_p53_residual_curvature()
    
    points = path_vectors["path_points_internal"]
    if len(points) != 3:
        raise ValueError(f"Expected exactly 3 points, got {len(points)}")
        
    c0 = points[0]["candidate_vector"]
    c1 = points[1]["candidate_vector"]
    c2 = points[2]["candidate_vector"]
    
    endpoint_dist = path_vectors["endpoint_distance_value"]
    
    d01 = compute_l2_distance(c0, c1)
    d12 = compute_l2_distance(c1, c2)
    d02 = compute_l2_distance(c0, c2)
    
    path_len = d01 + d12
    excess_len = path_len - d02
    
    # bend_ratio & curvature_proxy
    if d02 > 0.0 and math.isfinite(d02):
        bend_ratio = path_len / d02
        curvature_proxy = excess_len / d02
    else:
        bend_ratio = float('nan')
        curvature_proxy = float('nan')
        
    # segment_length_ratio
    min_seg = min(d01, d12)
    max_seg = max(d01, d12)
    if min_seg > 0.0 and math.isfinite(min_seg) and math.isfinite(max_seg):
        seg_ratio = max_seg / min_seg
    else:
        seg_ratio = float('nan')
        
    # midpoint deviation
    mid_outer = 0.5 * (c0 + c2)
    midpoint_deviation = compute_l2_distance(c1, mid_outer)
    
    if endpoint_dist > 0.0 and math.isfinite(endpoint_dist):
        norm_midpoint_dev = midpoint_deviation / endpoint_dist
    else:
        norm_midpoint_dev = float('nan')
        
    segment_lengths_finite = math.isfinite(d01) and math.isfinite(d12) and math.isfinite(d02)
    curvature_finite = math.isfinite(bend_ratio) and math.isfinite(curvature_proxy)
    midpoint_dev_finite = math.isfinite(midpoint_deviation) and math.isfinite(norm_midpoint_dev)
    
    return {
        "segment_0_1_length_value": float(d01),
        "segment_1_2_length_value": float(d12),
        "outer_chord_0_2_length_value": float(d02),
        "path_length_value": float(path_len),
        "excess_path_length_value": float(excess_len),
        "bend_ratio_value": float(bend_ratio) if math.isfinite(bend_ratio) else None,
        "curvature_proxy_value": float(curvature_proxy) if math.isfinite(curvature_proxy) else None,
        "segment_length_ratio_value": float(seg_ratio) if math.isfinite(seg_ratio) else None,
        "midpoint_deviation_value": float(midpoint_deviation),
        "normalized_midpoint_deviation_value": float(norm_midpoint_dev) if math.isfinite(norm_midpoint_dev) else None,
        "curvature_values_finite": bool(curvature_finite),
        "segment_lengths_finite": bool(segment_lengths_finite),
        "midpoint_deviation_finite": bool(midpoint_dev_finite),
    }


def run_bridge_path_residual_curvature_diagnostics_probe() -> dict:
    try:
        torch = load_torch_for_p53_residual_curvature()
    except Exception:
        return {
            "contract_version": BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_CONTRACT_VERSION,
            "status": "blocked_torch_unavailable",
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    path_vectors = build_p53_path_vectors()
    
    path_points = []
    for p in path_vectors["path_points_internal"]:
        pt = run_single_p53_residual_point_diagnostics(p, path_vectors)
        path_points.append(pt)
        
    curv = run_p53_curvature_smoothness_diagnostics(path_vectors)
    
    proj_positions = tuple(pt["projection_position_value"] for pt in path_points)
    proj_positions_finite = all(math.isfinite(x) for x in proj_positions)
    
    from src.phase2.bridge_path_consistency_diagnostics import (
        is_strictly_increasing,
    )
    proj_strictly_increasing = is_strictly_increasing(proj_positions)
    
    line_residuals = [pt["line_residual_value"] for pt in path_points]
    norm_residuals = [pt["normalized_line_residual_value"] for pt in path_points]
    own_target_dists = [pt["distance_to_own_target_value"] for pt in path_points]
    
    line_residuals_finite = all(math.isfinite(x) for x in line_residuals)
    own_target_distances_finite = all(math.isfinite(x) for x in own_target_dists)
    
    max_norm_res = max(norm_residuals)
    mean_norm_res = sum(norm_residuals) / len(norm_residuals)
    
    max_own_target_dist = max(own_target_dists)
    mean_own_target_dist = sum(own_target_dists) / len(own_target_dists)
    
    all_targets_loss_decreased = all(pt["loss_decreased"] for pt in path_points)
    all_losses_finite = all(math.isfinite(pt["initial_loss_value"]) and math.isfinite(pt["final_loss_value"]) for pt in path_points)
    
    passed = (
        proj_positions_finite and
        proj_strictly_increasing and
        line_residuals_finite and
        own_target_distances_finite and
        curv["curvature_values_finite"] and
        curv["segment_lengths_finite"] and
        curv["midpoint_deviation_finite"] and
        all_targets_loss_decreased and
        all_losses_finite
    )
    
    summary = {
        "contract_version": BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_CONTRACT_VERSION,
        "kind": BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_KIND,
        "status": "bridge_path_residual_curvature_diagnostics_available_no_model_no_vae_no_gsb_no_science",
        "torch_available": True,
        "bridge_target_count": len(path_points),
        "lambda_values": [pt["lambda_value"] for pt in path_points],
        "signature_vector_length": path_vectors["signature_vector_length"],
        "signature_tensor_field_count": len(path_vectors["field_names"]),
        "projection_positions": list(proj_positions),
        "projection_positions_finite": proj_positions_finite,
        "projection_positions_strictly_increasing": proj_strictly_increasing,
        "line_residual_values": line_residuals,
        "normalized_line_residual_values": norm_residuals,
        "line_residuals_finite": line_residuals_finite,
        "max_normalized_line_residual_value": float(max_norm_res),
        "mean_normalized_line_residual_value": float(mean_norm_res),
        "own_target_distances": own_target_dists,
        "own_target_distances_finite": own_target_distances_finite,
        "max_own_target_distance_value": float(max_own_target_dist),
        "mean_own_target_distance_value": float(mean_own_target_dist),
        
        "segment_0_1_length_value": curv["segment_0_1_length_value"],
        "segment_1_2_length_value": curv["segment_1_2_length_value"],
        "outer_chord_0_2_length_value": curv["outer_chord_0_2_length_value"],
        "path_length_value": curv["path_length_value"],
        "excess_path_length_value": curv["excess_path_length_value"],
        "bend_ratio_value": curv["bend_ratio_value"],
        "curvature_proxy_value": curv["curvature_proxy_value"],
        "segment_length_ratio_value": curv["segment_length_ratio_value"],
        "midpoint_deviation_value": curv["midpoint_deviation_value"],
        "normalized_midpoint_deviation_value": curv["normalized_midpoint_deviation_value"],
        "curvature_values_finite": curv["curvature_values_finite"],
        "segment_lengths_finite": curv["segment_lengths_finite"],
        "midpoint_deviation_finite": curv["midpoint_deviation_finite"],
        
        "all_targets_loss_decreased": all_targets_loss_decreased,
        "all_losses_finite": all_losses_finite,
        "residual_curvature_diagnostics_passed": passed,
        "path_shape_diagnostics_passed": passed,
        "path_points": path_points,
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
        "realizability_claim": "residual_curvature_diagnostics_only_not_state_validity",
        "reason": "p53_bridge_path_residual_curvature_diagnostics_probe_success"
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def bridge_path_residual_curvature_diagnostics_probe_to_json_dict(probe_res: dict) -> dict:
    validate_non_empty_str(probe_res["contract_version"], "contract_version")
    validate_non_empty_str(probe_res["status"], "status")
    validate_non_empty_str(probe_res["reason"], "reason")
    assert_no_local_path_leakage(probe_res["reason"], "reason")
    assert_no_forbidden_claims(probe_res["reason"], "reason")
    
    return {
        "contract_version": probe_res["contract_version"],
        "kind": probe_res.get("kind", BRIDGE_PATH_RESIDUAL_CURVATURE_DIAGNOSTICS_KIND),
        "status": probe_res["status"],
        "torch_available": bool(probe_res["torch_available"]),
        "bridge_target_count": int(probe_res.get("bridge_target_count", 0)),
        "lambda_values": list(probe_res.get("lambda_values", [])),
        "signature_vector_length": int(probe_res.get("signature_vector_length", 0)),
        "signature_tensor_field_count": int(probe_res.get("signature_tensor_field_count", 0)),
        "projection_positions": list(probe_res.get("projection_positions", [])),
        "projection_positions_finite": bool(probe_res.get("projection_positions_finite", False)),
        "projection_positions_strictly_increasing": bool(probe_res.get("projection_positions_strictly_increasing", False)),
        "line_residual_values": list(probe_res.get("line_residual_values", [])),
        "normalized_line_residual_values": list(probe_res.get("normalized_line_residual_values", [])),
        "line_residuals_finite": bool(probe_res.get("line_residuals_finite", False)),
        "max_normalized_line_residual_value": float(probe_res.get("max_normalized_line_residual_value", 0.0)),
        "mean_normalized_line_residual_value": float(probe_res.get("mean_normalized_line_residual_value", 0.0)),
        "own_target_distances": list(probe_res.get("own_target_distances", [])),
        "own_target_distances_finite": bool(probe_res.get("own_target_distances_finite", False)),
        "max_own_target_distance_value": float(probe_res.get("max_own_target_distance_value", 0.0)),
        "mean_own_target_distance_value": float(probe_res.get("mean_own_target_distance_value", 0.0)),
        "segment_0_1_length_value": float(probe_res.get("segment_0_1_length_value", 0.0)),
        "segment_1_2_length_value": float(probe_res.get("segment_1_2_length_value", 0.0)),
        "outer_chord_0_2_length_value": float(probe_res.get("outer_chord_0_2_length_value", 0.0)),
        "path_length_value": float(probe_res.get("path_length_value", 0.0)),
        "excess_path_length_value": float(probe_res.get("excess_path_length_value", 0.0)),
        "bend_ratio_value": float(probe_res["bend_ratio_value"]) if probe_res.get("bend_ratio_value") is not None else None,
        "curvature_proxy_value": float(probe_res["curvature_proxy_value"]) if probe_res.get("curvature_proxy_value") is not None else None,
        "segment_length_ratio_value": float(probe_res["segment_length_ratio_value"]) if probe_res.get("segment_length_ratio_value") is not None else None,
        "midpoint_deviation_value": float(probe_res.get("midpoint_deviation_value", 0.0)),
        "normalized_midpoint_deviation_value": float(probe_res["normalized_midpoint_deviation_value"]) if probe_res.get("normalized_midpoint_deviation_value") is not None else None,
        "curvature_values_finite": bool(probe_res.get("curvature_values_finite", False)),
        "segment_lengths_finite": bool(probe_res.get("segment_lengths_finite", False)),
        "midpoint_deviation_finite": bool(probe_res.get("midpoint_deviation_finite", False)),
        "all_targets_loss_decreased": bool(probe_res.get("all_targets_loss_decreased", False)),
        "all_losses_finite": bool(probe_res.get("all_losses_finite", False)),
        "residual_curvature_diagnostics_passed": bool(probe_res.get("residual_curvature_diagnostics_passed", False)),
        "path_shape_diagnostics_passed": bool(probe_res.get("path_shape_diagnostics_passed", False)),
        "path_points": probe_res.get("path_points", []),
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
        "realizability_claim": probe_res.get("realizability_claim", "residual_curvature_diagnostics_only_not_state_validity"),
        "reason": probe_res["reason"],
    }


def compact_bridge_path_residual_curvature_diagnostics_json(probe_res: dict) -> str:
    d = bridge_path_residual_curvature_diagnostics_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
