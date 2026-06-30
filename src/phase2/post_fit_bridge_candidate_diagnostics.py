# src/phase2/post_fit_bridge_candidate_diagnostics.py

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
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    P50_STEP_COUNT,
    P50_LEARNING_RATE,
    build_p50_bridge_targets,
    freeze_bridge_target_signature,
    build_p50_candidate_raw_tensors,
    build_combined_signature_from_raw,
    compute_p50_bridge_target_loss,
)

# Constants
POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_CONTRACT_VERSION = "phase2_p51_post_fit_bridge_candidate_diagnostics_contract_v1"
POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_KIND = "post_fit_bridge_candidate_diagnostics_no_model_no_vae_no_gsb_no_science"
POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_MODULE_NAME = "src.phase2.post_fit_bridge_candidate_diagnostics"
P51_TARGET_COUNT = 3


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


def load_torch_for_p51_post_fit_diagnostics() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P51 loader.")
    import torch
    return torch


def run_single_p51_fit_with_final_raw(target: Dict[str, Any]) -> Dict[str, Any]:
    torch = load_torch_for_p51_post_fit_diagnostics()
    
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
        "final_candidate_raw": candidate_raw,
    }


def build_post_fit_constrained_diagnostics(final_candidate_raw: Dict[str, Any]) -> Dict[str, Any]:
    torch = load_torch_for_p51_post_fit_diagnostics()
    
    # AR P44 diagnostics
    ar_res = pacf_to_stable_ar_coefficients(final_candidate_raw["raw_kappa"])
    ar_coefs = ar_res["ar_coefficients"]
    ar_finite = torch.all(torch.isfinite(ar_coefs)).item()
    ar_shape = list(ar_coefs.shape)
    ar_max = torch.max(torch.abs(ar_coefs)).item()
    ar_mean = torch.mean(torch.abs(ar_coefs)).item()
    
    # GARCH P44 diagnostics
    bias = default_garch_allocation_bias(
        alpha_count=1,
        beta_count=1,
        dtype=final_candidate_raw["raw_omega"].dtype,
        device=final_candidate_raw["raw_omega"].device
    )
    garch_res = garch_mass_allocation_from_logits(
        raw_omega=final_candidate_raw["raw_omega"],
        raw_total_mass=final_candidate_raw["raw_total_mass"],
        raw_allocation_logits=final_candidate_raw["raw_allocation_logits"],
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    
    omega = garch_res["omega"]
    alpha = garch_res["alpha"]
    beta = garch_res["beta"]
    persistence = garch_res["alpha_beta_sum"]
    margin = garch_res["stationarity_margin"]
    alpha_sum = garch_res["alpha_sum"]
    beta_sum = garch_res["beta_sum"]
    
    garch_finite = (
        torch.all(torch.isfinite(omega)).item() and
        torch.all(torch.isfinite(alpha)).item() and
        torch.all(torch.isfinite(beta)).item() and
        torch.all(torch.isfinite(persistence)).item() and
        torch.all(torch.isfinite(margin)).item()
    )
    
    persistence_finite = torch.all(torch.isfinite(persistence)).item()
    persistence_max = torch.max(persistence).item()
    persistence_min = torch.min(persistence).item()
    
    margin_finite = torch.all(torch.isfinite(margin)).item()
    margin_min = torch.min(margin).item()
    
    # Compute alpha/beta shares
    eps = 1e-6
    alpha_share = alpha_sum / torch.clamp(persistence, min=eps)
    beta_share = beta_sum / torch.clamp(persistence, min=eps)
    
    alpha_share_finite = torch.all(torch.isfinite(alpha_share)).item()
    beta_share_finite = torch.all(torch.isfinite(beta_share)).item()
    
    return {
        "ar_coefficients_finite": bool(ar_finite),
        "ar_coefficients_tensor_shape": ar_shape,
        "ar_abs_max_value": float(ar_max),
        "ar_abs_mean_value": float(ar_mean),
        "garch_tensor_fields_finite": bool(garch_finite),
        "garch_persistence_available": True,
        "garch_persistence_finite": bool(persistence_finite),
        "garch_persistence_max_value": float(persistence_max),
        "garch_persistence_min_value": float(persistence_min),
        "garch_stationarity_margin_available": True,
        "garch_stationarity_margin_finite": bool(margin_finite),
        "garch_stationarity_margin_min_value": float(margin_min),
        "garch_alpha_share_available": True,
        "garch_alpha_share_finite": bool(alpha_share_finite),
        "garch_beta_share_available": True,
        "garch_beta_share_finite": bool(beta_share_finite),
    }


def build_post_fit_signature_diagnostics(final_candidate_raw: Dict[str, Any]) -> Dict[str, Any]:
    torch = load_torch_for_p51_post_fit_diagnostics()
    sig = build_combined_signature_from_raw(final_candidate_raw)
    
    tensor_field_count = 0
    all_finite = True
    scalar_field_count = 0
    
    for k, v in sig.items():
        if torch.is_tensor(v):
            tensor_field_count += 1
            if not torch.all(torch.isfinite(v)).item():
                all_finite = False
        else:
            scalar_field_count += 1
            
    return {
        "signature_tensor_field_count": int(tensor_field_count),
        "signature_all_tensor_fields_finite": bool(all_finite),
        "signature_scalar_field_count": int(scalar_field_count),
        "signature_has_ar_spectrum": "ar_spectrum" in sig,
        "signature_has_ar_spectrum_log": "ar_spectrum_log" in sig,
        "signature_has_garch_persistence": "garch_persistence" in sig,
        "signature_has_garch_stationarity_margin": "garch_stationarity_margin" in sig,
        "signature_has_garch_alpha_share": "garch_alpha_share" in sig,
        "signature_has_garch_beta_share": "garch_beta_share" in sig,
    }


def run_single_p51_post_fit_diagnostics(target: Dict[str, Any]) -> Dict[str, Any]:
    fit_res = run_single_p51_fit_with_final_raw(target)
    final_raw = fit_res.pop("final_candidate_raw")
    
    constrained_diag = build_post_fit_constrained_diagnostics(final_raw)
    sig_diag = build_post_fit_signature_diagnostics(final_raw)
    
    return {
        "target_id": fit_res["target_id"],
        "lambda_value": fit_res["lambda_value"],
        "initial_loss_value": fit_res["initial_loss_value"],
        "final_loss_value": fit_res["final_loss_value"],
        "loss_delta_value": fit_res["loss_delta_value"],
        "loss_decreased": fit_res["loss_decreased"],
        "initial_loss_finite": fit_res["initial_loss_finite"],
        "final_loss_finite": fit_res["final_loss_finite"],
        "step_count": int(P50_STEP_COUNT),
        "learning_rate": float(P50_LEARNING_RATE),
        "ar_coefficients_finite": constrained_diag["ar_coefficients_finite"],
        "ar_coefficients_tensor_shape": constrained_diag["ar_coefficients_tensor_shape"],
        "ar_abs_max_value": constrained_diag["ar_abs_max_value"],
        "ar_abs_mean_value": constrained_diag["ar_abs_mean_value"],
        "garch_tensor_fields_finite": constrained_diag["garch_tensor_fields_finite"],
        "garch_persistence_available": constrained_diag["garch_persistence_available"],
        "garch_persistence_finite": constrained_diag["garch_persistence_finite"],
        "garch_persistence_max_value": constrained_diag["garch_persistence_max_value"],
        "garch_persistence_min_value": constrained_diag["garch_persistence_min_value"],
        "signature_tensor_field_count": sig_diag["signature_tensor_field_count"],
        "signature_all_tensor_fields_finite": sig_diag["signature_all_tensor_fields_finite"],
        "signature_has_ar_spectrum": sig_diag["signature_has_ar_spectrum"],
        "signature_has_ar_spectrum_log": sig_diag["signature_has_ar_spectrum_log"],
        "signature_has_garch_persistence": sig_diag["signature_has_garch_persistence"],
        "signature_has_garch_stationarity_margin": sig_diag["signature_has_garch_stationarity_margin"],
        "realizability_claim": "post_fit_diagnostics_only_not_state_validity",
        "no_state_validity_claim": True,
        "construction_method": target.get("construction_method", "linear_interpolation_in_p45_combined_signature_space"),
    }


def run_post_fit_bridge_candidate_diagnostics_probe() -> dict:
    try:
        torch = load_torch_for_p51_post_fit_diagnostics()
    except Exception:
        return {
            "contract_version": POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_CONTRACT_VERSION,
            "status": "blocked_torch_unavailable",
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    targets = build_p50_bridge_targets()
    target_results = []
    for t in targets:
        res = run_single_p51_post_fit_diagnostics(t)
        target_results.append(res)
        
    passed_count = sum(1 for r in target_results if r["loss_decreased"])
    all_decreased = (passed_count == len(targets))
    all_losses_finite = all(r["initial_loss_finite"] and r["final_loss_finite"] for r in target_results)
    
    all_ar_finite = all(r["ar_coefficients_finite"] for r in target_results)
    all_garch_finite = all(r["garch_tensor_fields_finite"] for r in target_results)
    all_sig_finite = all(r["signature_all_tensor_fields_finite"] for r in target_results)
    
    summary = {
        "contract_version": POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_CONTRACT_VERSION,
        "kind": POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_KIND,
        "status": "post_fit_bridge_candidate_diagnostics_available_no_model_no_vae_no_gsb_no_science",
        "torch_available": True,
        "bridge_target_count": len(targets),
        "lambda_values": [t.get("lambda_value") for t in targets],
        "targets_passed_loss_decrease": passed_count,
        "all_targets_loss_decreased": all_decreased,
        "all_losses_finite": all_losses_finite,
        "all_ar_coefficients_finite": all_ar_finite,
        "all_garch_tensor_fields_finite": all_garch_finite,
        "all_signature_tensor_fields_finite": all_sig_finite,
        "target_diagnostics": target_results,
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
        "realizability_claim": "post_fit_diagnostics_only_not_state_validity",
        "reason": "p51_post_fit_bridge_candidate_diagnostics_probe_success"
    }
    
    # Validations
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def post_fit_bridge_candidate_diagnostics_probe_to_json_dict(probe_res: dict) -> dict:
    validate_non_empty_str(probe_res["contract_version"], "contract_version")
    validate_non_empty_str(probe_res["status"], "status")
    validate_non_empty_str(probe_res["reason"], "reason")
    assert_no_local_path_leakage(probe_res["reason"], "reason")
    assert_no_forbidden_claims(probe_res["reason"], "reason")
    
    return {
        "contract_version": probe_res["contract_version"],
        "kind": probe_res.get("kind", POST_FIT_BRIDGE_CANDIDATE_DIAGNOSTICS_KIND),
        "status": probe_res["status"],
        "torch_available": bool(probe_res["torch_available"]),
        "bridge_target_count": int(probe_res.get("bridge_target_count", 0)),
        "lambda_values": list(probe_res.get("lambda_values", [])),
        "targets_passed_loss_decrease": int(probe_res.get("targets_passed_loss_decrease", 0)),
        "all_targets_loss_decreased": bool(probe_res.get("all_targets_loss_decreased", False)),
        "all_losses_finite": bool(probe_res.get("all_losses_finite", False)),
        "all_ar_coefficients_finite": bool(probe_res.get("all_ar_coefficients_finite", False)),
        "all_garch_tensor_fields_finite": bool(probe_res.get("all_garch_tensor_fields_finite", False)),
        "all_signature_tensor_fields_finite": bool(probe_res.get("all_signature_tensor_fields_finite", False)),
        "target_diagnostics": probe_res.get("target_diagnostics", []),
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
        "realizability_claim": probe_res.get("realizability_claim", "post_fit_diagnostics_only_not_state_validity"),
        "reason": probe_res["reason"],
    }


def compact_post_fit_bridge_candidate_diagnostics_json(probe_res: dict) -> str:
    d = post_fit_bridge_candidate_diagnostics_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
