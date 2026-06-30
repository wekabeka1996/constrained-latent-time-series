# src/phase2/direct_raw_fit_to_endpoint_bridge_targets.py

import json
from typing import Any, Dict, Tuple

from src.phase2.torch_boundary import (
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)
from src.phase2.tensor_native_constraint_primitives import (
    pacf_to_stable_ar_coefficients,
    garch_mass_allocation_from_logits,
    default_garch_allocation_bias,
)
from src.phase2.analytic_moment_spectral_signatures import (
    ar_spectral_signature,
    garch_moment_persistence_signature,
    combined_ar_garch_signature,
)
from src.phase2.moment_spectral_matching_loss import (
    combined_moment_spectral_matching_loss,
)
from src.phase2.deterministic_endpoint_bridge_targets import (
    build_p49_endpoint_raw_tensors,
    build_endpoint_signatures,
    build_bridge_targets,
)

# Constants
DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_CONTRACT_VERSION = "phase2_p50_direct_raw_fit_to_endpoint_bridge_targets_contract_v1"
DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_KIND = "direct_raw_fit_to_endpoint_bridge_targets_no_model_no_vae_no_gsb_no_science"
DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_MODULE_NAME = "src.phase2.direct_raw_fit_to_endpoint_bridge_targets"
P50_STEP_COUNT = 30
P50_LEARNING_RATE = 0.05
P50_TARGET_COUNT = 3


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


def load_torch_for_p50_direct_bridge_fit() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P50 loader.")
    import torch
    return torch


def build_p50_bridge_targets() -> Tuple[Dict[str, Any], ...]:
    ep_raw = build_p49_endpoint_raw_tensors()
    ep_sigs = build_endpoint_signatures(ep_raw)
    targets = build_bridge_targets(ep_sigs)
    return targets


def freeze_bridge_target_signature(target: Dict[str, Any]) -> Dict[str, Any]:
    torch = load_torch_for_p50_direct_bridge_fit()
    frozen = {}
    for k, v in target.items():
        if torch.is_tensor(v):
            frozen[k] = v.detach().clone()
        else:
            frozen[k] = v
    return frozen


def build_p50_candidate_raw_tensors(lambda_value: float) -> Dict[str, Any]:
    torch = load_torch_for_p50_direct_bridge_fit()
    endpoint_raw = build_p49_endpoint_raw_tensors()
    raw_A = endpoint_raw["endpoint_A"]
    raw_B = endpoint_raw["endpoint_B"]
    
    # Blend raw inputs to create candidate starting states near but not equal to the endpoints
    if abs(lambda_value - 0.25) < 1e-4:
        w_A, w_B, offset = 0.8, 0.2, 0.05
    elif abs(lambda_value - 0.50) < 1e-4:
        w_A, w_B, offset = 0.5, 0.5, -0.03
    elif abs(lambda_value - 0.75) < 1e-4:
        w_A, w_B, offset = 0.2, 0.8, 0.02
    else:
        w_A, w_B, offset = 1.0 - lambda_value, lambda_value, 0.0
        
    candidate = {}
    for k in ["raw_kappa", "raw_omega", "raw_total_mass", "raw_allocation_logits"]:
        val_A = raw_A[k]
        val_B = raw_B[k]
        c_val = w_A * val_A + w_B * val_B + offset
        candidate[k] = c_val.clone().detach().requires_grad_(True)
        
    return candidate


def build_combined_signature_from_raw(raw_tensors: Dict[str, Any]) -> Dict[str, Any]:
    ar_res = pacf_to_stable_ar_coefficients(raw_tensors["raw_kappa"])
    bias = default_garch_allocation_bias(
        alpha_count=1,
        beta_count=1,
        dtype=raw_tensors["raw_omega"].dtype,
        device=raw_tensors["raw_omega"].device
    )
    garch_res = garch_mass_allocation_from_logits(
        raw_omega=raw_tensors["raw_omega"],
        raw_total_mass=raw_tensors["raw_total_mass"],
        raw_allocation_logits=raw_tensors["raw_allocation_logits"],
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    ar_sig = ar_spectral_signature(ar_res["ar_coefficients"])
    garch_sig = garch_moment_persistence_signature(garch_res)
    return combined_ar_garch_signature(ar_sig, garch_sig)


def compute_p50_bridge_target_loss(candidate_signature: Dict[str, Any], frozen_bridge_target: Dict[str, Any]) -> Dict[str, Any]:
    return combined_moment_spectral_matching_loss(candidate_signature, frozen_bridge_target)


def run_single_p50_bridge_fit(target: Dict[str, Any]) -> Dict[str, Any]:
    torch = load_torch_for_p50_direct_bridge_fit()
    
    target_id = target["target_id"]
    lambda_value = target["lambda_value"]
    
    frozen_target = freeze_bridge_target_signature(target)
    candidate_raw = build_p50_candidate_raw_tensors(lambda_value)
    
    # Compute initial loss
    init_sig = build_combined_signature_from_raw(candidate_raw)
    init_loss_dict = compute_p50_bridge_target_loss(init_sig, frozen_target)
    init_loss_val = init_loss_dict["loss_total"].item()
    
    # Run manual gradient steps
    for _ in range(P50_STEP_COUNT):
        sig = build_combined_signature_from_raw(candidate_raw)
        loss_dict = compute_p50_bridge_target_loss(sig, frozen_target)
        loss = loss_dict["loss_total"]
        
        # Zero grads if any left
        for k, v in candidate_raw.items():
            if v.grad is not None:
                v.grad.zero_()
                
        loss.backward()
        
        # Update raw parameters manually
        with torch.no_grad():
            for k, v in candidate_raw.items():
                if v.grad is not None:
                    v.copy_(v - P50_LEARNING_RATE * v.grad)
                    
    # Compute final loss
    final_sig = build_combined_signature_from_raw(candidate_raw)
    final_loss_dict = compute_p50_bridge_target_loss(final_sig, frozen_target)
    final_loss_val = final_loss_dict["loss_total"].item()
    
    loss_delta = init_loss_val - final_loss_val
    loss_decreased = loss_delta > 0.0
    
    import math
    init_finite = math.isfinite(init_loss_val)
    final_finite = math.isfinite(final_loss_val)
    
    return {
        "target_id": target_id,
        "lambda_value": float(lambda_value),
        "initial_loss_value": float(init_loss_val),
        "final_loss_value": float(final_loss_val),
        "loss_delta_value": float(loss_delta),
        "loss_decreased": bool(loss_decreased),
        "initial_loss_finite": bool(init_finite),
        "final_loss_finite": bool(final_finite),
        "step_count": int(P50_STEP_COUNT),
        "learning_rate": float(P50_LEARNING_RATE),
        "construction_method": target.get("construction_method", "linear_interpolation_in_p45_combined_signature_space"),
        "realizability_claim": "optimization_approachability_only_not_state_validity"
    }


def run_direct_raw_fit_to_bridge_targets_probe() -> dict:
    try:
        torch = load_torch_for_p50_direct_bridge_fit()
    except Exception:
        return {
            "contract_version": DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_CONTRACT_VERSION,
            "status": "blocked_torch_unavailable",
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    targets = build_p50_bridge_targets()
    target_results = []
    for t in targets:
        res = run_single_p50_bridge_fit(t)
        target_results.append(res)
        
    passed_count = sum(1 for r in target_results if r["loss_decreased"])
    all_decreased = (passed_count == len(targets))
    all_finite = all(r["initial_loss_finite"] and r["final_loss_finite"] for r in target_results)
    
    deltas = [r["loss_delta_value"] for r in target_results]
    min_delta = min(deltas) if deltas else 0.0
    max_delta = max(deltas) if deltas else 0.0
    mean_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    
    summary = {
        "contract_version": DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_CONTRACT_VERSION,
        "kind": DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_KIND,
        "status": "direct_raw_fit_to_bridge_targets_available_no_model_no_vae_no_gsb_no_science",
        "torch_available": True,
        "bridge_target_count": len(targets),
        "lambda_values": [t.get("lambda_value") for t in targets],
        "targets_passed": passed_count,
        "all_targets_loss_decreased": all_decreased,
        "all_losses_finite": all_finite,
        "min_loss_delta_value": float(min_delta),
        "max_loss_delta_value": float(max_delta),
        "mean_loss_delta_value": float(mean_delta),
        "target_results": target_results,
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
        "realizability_claim": "optimization_approachability_only_not_state_validity",
        "reason": "p50_direct_raw_fit_to_endpoint_bridge_targets_probe_success"
    }
    
    # Validations
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def direct_raw_fit_to_bridge_targets_probe_to_json_dict(probe_res: dict) -> dict:
    validate_non_empty_str(probe_res["contract_version"], "contract_version")
    validate_non_empty_str(probe_res["status"], "status")
    validate_non_empty_str(probe_res["reason"], "reason")
    assert_no_local_path_leakage(probe_res["reason"], "reason")
    assert_no_forbidden_claims(probe_res["reason"], "reason")
    
    return {
        "contract_version": probe_res["contract_version"],
        "kind": probe_res.get("kind", DIRECT_RAW_FIT_TO_BRIDGE_TARGETS_KIND),
        "status": probe_res["status"],
        "torch_available": bool(probe_res["torch_available"]),
        "bridge_target_count": int(probe_res.get("bridge_target_count", 0)),
        "lambda_values": list(probe_res.get("lambda_values", [])),
        "targets_passed": int(probe_res.get("targets_passed", 0)),
        "all_targets_loss_decreased": bool(probe_res.get("all_targets_loss_decreased", False)),
        "all_losses_finite": bool(probe_res.get("all_losses_finite", False)),
        "min_loss_delta_value": float(probe_res.get("min_loss_delta_value", 0.0)),
        "max_loss_delta_value": float(probe_res.get("max_loss_delta_value", 0.0)),
        "mean_loss_delta_value": float(probe_res.get("mean_loss_delta_value", 0.0)),
        "target_results": probe_res.get("target_results", []),
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
        "realizability_claim": probe_res.get("realizability_claim", "optimization_approachability_only_not_state_validity"),
        "reason": probe_res["reason"],
    }


def compact_direct_raw_fit_to_bridge_targets_json(probe_res: dict) -> str:
    d = direct_raw_fit_to_bridge_targets_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
