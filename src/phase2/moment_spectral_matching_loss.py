# src/phase2/moment_spectral_matching_loss.py

import dataclasses
import json
from typing import Any, Tuple

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

# Constants
MOMENT_SPECTRAL_MATCHING_LOSS_CONTRACT_VERSION = "phase2_p46_moment_spectral_matching_loss_contract_v1"

MOMENT_SPECTRAL_MATCHING_LOSS_KIND = "moment_spectral_matching_loss_no_model_no_training"

MOMENT_SPECTRAL_MATCHING_LOSS_MODULE_NAME = "src.phase2.moment_spectral_matching_loss"

P46_DEFAULT_EPS = 1.0e-8

P46_DEFAULT_AR_LOG_SPECTRUM_WEIGHT = 1.0

P46_DEFAULT_AR_SUMMARY_WEIGHT = 0.25

P46_DEFAULT_GARCH_MOMENT_WEIGHT = 1.0

P46_DEFAULT_GARCH_DECAY_WEIGHT = 0.5

# Statuses
FC_VAE_LOSS_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"
FC_VAE_LOSS_STATUS_AVAILABLE = "moment_spectral_matching_loss_available_no_model_no_training"
FC_VAE_LOSS_STATUS_CONTRACT_MISMATCH = "blocked_by_contract_mismatch"

SUPPORTED_FC_VAE_LOSS_STATUSES = (
    FC_VAE_LOSS_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_LOSS_STATUS_AVAILABLE,
    FC_VAE_LOSS_STATUS_CONTRACT_MISMATCH,
)


# Validators
def validate_non_empty_str(val: Any, name: str) -> None:
    if type(val) is not str:
        raise TypeError(f"{name} must be exact str instance")
    if not val.strip():
        raise ValueError(f"{name} cannot be empty or whitespace only")


def validate_bool(val: Any, name: str) -> None:
    if type(val) is not bool:
        raise TypeError(f"{name} must be exact bool instance")


def validate_positive_int(val: Any, name: str) -> None:
    if type(val) is not int:
        raise TypeError(f"{name} must be exact int instance")
    if val <= 0:
        raise ValueError(f"{name} must be positive")


def validate_non_negative_int(val: Any, name: str) -> None:
    if type(val) is not int:
        raise TypeError(f"{name} must be exact int instance")
    if val < 0:
        raise ValueError(f"{name} must be non-negative")


def assert_no_local_path_leakage(val: str, name: str = "field") -> None:
    forbidden = [":\\", "Users", "home", "/Users", "/home", ".git"]
    val_lower = val.lower()
    for item in forbidden:
        if item.lower() in val_lower:
            raise ValueError(f"Local path leak detected in {name}")


def assert_no_forbidden_claims(val: str, name: str = "field") -> None:
    forbidden = ["scientific success", "solved", "best", "winner", "production ready", "state of the art"]
    val_lower = val.lower()
    for item in forbidden:
        if item.lower() in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}")


# Torch Loader
def load_torch_for_p46_loss() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P46 loader.")
    import torch
    return torch


# safe tensor MSE
def safe_tensor_mse(candidate: Any, target: Any, eps: float = P46_DEFAULT_EPS) -> Any:
    torch = load_torch_for_p46_loss()
    if not torch.is_tensor(candidate) or not torch.is_tensor(target):
        raise TypeError("candidate and target must be PyTorch tensors")
    if candidate.shape != target.shape:
        raise ValueError(f"mismatched tensor shapes: {candidate.shape} vs {target.shape}")
        
    return torch.mean((candidate - target) ** 2)


# AR Spectral Matching Loss
def ar_spectral_matching_loss(candidate_ar_signature: dict, target_ar_signature: dict, eps: float = P46_DEFAULT_EPS) -> dict:
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
        
    log_spectrum_mse = safe_tensor_mse(candidate_ar_signature["spectrum_log"], target_ar_signature["spectrum_log"], eps=eps)
    mean_mse = safe_tensor_mse(candidate_ar_signature["spectrum_mean"], target_ar_signature["spectrum_mean"], eps=eps)
    std_mse = safe_tensor_mse(candidate_ar_signature["spectrum_std"], target_ar_signature["spectrum_std"], eps=eps)
    low_mse = safe_tensor_mse(candidate_ar_signature["spectrum_low_freq_power"], target_ar_signature["spectrum_low_freq_power"], eps=eps)
    high_mse = safe_tensor_mse(candidate_ar_signature["spectrum_high_freq_power"], target_ar_signature["spectrum_high_freq_power"], eps=eps)
    
    summary_mse = (mean_mse + std_mse + low_mse + high_mse) / 4.0
    total = P46_DEFAULT_AR_LOG_SPECTRUM_WEIGHT * log_spectrum_mse + P46_DEFAULT_AR_SUMMARY_WEIGHT * summary_mse
    
    return {
        "ar_loss_total": total,
        "ar_log_spectrum_mse": log_spectrum_mse,
        "ar_summary_mse": summary_mse,
        "ar_loss_kind": "ar_spectral_matching_loss_no_model_no_training",
    }


# GARCH Moment Matching Loss
def garch_moment_matching_loss(candidate_garch_signature: dict, target_garch_signature: dict, eps: float = P46_DEFAULT_EPS) -> dict:
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
        
    var_mse = safe_tensor_mse(candidate_garch_signature["unconditional_variance"], target_garch_signature["unconditional_variance"], eps=eps)
    persistence_mse = safe_tensor_mse(candidate_garch_signature["persistence"], target_garch_signature["persistence"], eps=eps)
    margin_mse = safe_tensor_mse(candidate_garch_signature["stationarity_margin"], target_garch_signature["stationarity_margin"], eps=eps)
    alpha_mse = safe_tensor_mse(candidate_garch_signature["alpha_share"], target_garch_signature["alpha_share"], eps=eps)
    beta_mse = safe_tensor_mse(candidate_garch_signature["beta_share"], target_garch_signature["beta_share"], eps=eps)
    
    moment_mse = (var_mse + persistence_mse + margin_mse + alpha_mse + beta_mse) / 5.0
    decay_mse = safe_tensor_mse(candidate_garch_signature["persistence_decay"], target_garch_signature["persistence_decay"], eps=eps)
    
    total = P46_DEFAULT_GARCH_MOMENT_WEIGHT * moment_mse + P46_DEFAULT_GARCH_DECAY_WEIGHT * decay_mse
    
    return {
        "garch_loss_total": total,
        "garch_moment_mse": moment_mse,
        "garch_decay_mse": decay_mse,
        "garch_loss_kind": "garch_moment_persistence_matching_loss_no_model_no_training",
    }


# Combined Moment Spectral Matching Loss
def combined_moment_spectral_matching_loss(
    candidate_combined_signature: dict,
    target_combined_signature: dict,
    eps: float = P46_DEFAULT_EPS,
) -> dict:
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
        
    ar_log_spectrum_mse = safe_tensor_mse(
        candidate_combined_signature["ar_spectrum_log"],
        target_combined_signature["ar_spectrum_log"],
        eps=eps
    )
    mean_mse = safe_tensor_mse(
        candidate_combined_signature["ar_spectrum_mean"],
        target_combined_signature["ar_spectrum_mean"],
        eps=eps
    )
    std_mse = safe_tensor_mse(
        candidate_combined_signature["ar_spectrum_std"],
        target_combined_signature["ar_spectrum_std"],
        eps=eps
    )
    ratio_mse = safe_tensor_mse(
        candidate_combined_signature["ar_low_high_ratio"],
        target_combined_signature["ar_low_high_ratio"],
        eps=eps
    )
    
    ar_summary_mse = (mean_mse + std_mse + ratio_mse) / 3.0
    ar_component_loss = P46_DEFAULT_AR_LOG_SPECTRUM_WEIGHT * ar_log_spectrum_mse + P46_DEFAULT_AR_SUMMARY_WEIGHT * ar_summary_mse
    
    var_mse = safe_tensor_mse(
        candidate_combined_signature["garch_unconditional_variance"],
        target_combined_signature["garch_unconditional_variance"],
        eps=eps
    )
    persistence_mse = safe_tensor_mse(
        candidate_combined_signature["garch_persistence"],
        target_combined_signature["garch_persistence"],
        eps=eps
    )
    margin_mse = safe_tensor_mse(
        candidate_combined_signature["garch_stationarity_margin"],
        target_combined_signature["garch_stationarity_margin"],
        eps=eps
    )
    alpha_mse = safe_tensor_mse(
        candidate_combined_signature["garch_alpha_share"],
        target_combined_signature["garch_alpha_share"],
        eps=eps
    )
    beta_mse = safe_tensor_mse(
        candidate_combined_signature["garch_beta_share"],
        target_combined_signature["garch_beta_share"],
        eps=eps
    )
    
    garch_moment_mse = (var_mse + persistence_mse + margin_mse + alpha_mse + beta_mse) / 5.0
    garch_decay_mse = safe_tensor_mse(
        candidate_combined_signature["garch_persistence_decay"],
        target_combined_signature["garch_persistence_decay"],
        eps=eps
    )
    
    garch_component_loss = P46_DEFAULT_GARCH_MOMENT_WEIGHT * garch_moment_mse + P46_DEFAULT_GARCH_DECAY_WEIGHT * garch_decay_mse
    
    loss_total = ar_component_loss + garch_component_loss
    
    return {
        "loss_total": loss_total,
        "ar_component_loss": ar_component_loss,
        "garch_component_loss": garch_component_loss,
        "ar_log_spectrum_mse": ar_log_spectrum_mse,
        "ar_summary_mse": ar_summary_mse,
        "garch_moment_mse": garch_moment_mse,
        "garch_decay_mse": garch_decay_mse,
        "loss_kind": "combined_moment_spectral_matching_loss_no_model_no_training",
    }


# Smoke Probe
def run_moment_spectral_matching_loss_probe() -> dict:
    try:
        torch = load_torch_for_p46_loss()
    except Exception:
        return {
            "contract_version": MOMENT_SPECTRAL_MATCHING_LOSS_CONTRACT_VERSION,
            "status": FC_VAE_LOSS_STATUS_TORCH_UNAVAILABLE,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    # Candidate raw inputs with requires_grad=True
    raw_kappa_cand = torch.tensor(
        [[0.1, -0.05, 0.02, -0.01, 0.005],
         [-0.1, 0.05, -0.02, 0.01, -0.005]],
        dtype=torch.float32, requires_grad=True
    )
    raw_omega_cand = torch.tensor([[-0.5], [0.5]], dtype=torch.float32, requires_grad=True)
    raw_total_mass_cand = torch.tensor([[1.0], [-1.0]], dtype=torch.float32, requires_grad=True)
    raw_allocation_logits_cand = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    bias = default_garch_allocation_bias(1, 1, dtype=torch.float32)
    
    # Target raw inputs (fixed target, requires_grad=False)
    raw_kappa_targ = torch.tensor(
        [[0.08, -0.04, 0.015, -0.008, 0.003],
         [-0.08, 0.04, -0.015, 0.008, -0.003]],
        dtype=torch.float32
    )
    raw_omega_targ = torch.tensor([[-0.4], [0.4]], dtype=torch.float32)
    raw_total_mass_targ = torch.tensor([[0.9], [-0.9]], dtype=torch.float32)
    raw_allocation_logits_targ = torch.zeros((2, 2), dtype=torch.float32)
    
    # Compute Candidate signatures
    ar_res_cand = pacf_to_stable_ar_coefficients(raw_kappa_cand)
    garch_res_cand = garch_mass_allocation_from_logits(
        raw_omega=raw_omega_cand,
        raw_total_mass=raw_total_mass_cand,
        raw_allocation_logits=raw_allocation_logits_cand,
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    ar_sig_cand = ar_spectral_signature(ar_res_cand["ar_coefficients"])
    garch_sig_cand = garch_moment_persistence_signature(garch_res_cand)
    comb_sig_cand = combined_ar_garch_signature(ar_sig_cand, garch_sig_cand)
    
    # Compute Target signatures
    ar_res_targ = pacf_to_stable_ar_coefficients(raw_kappa_targ)
    garch_res_targ = garch_mass_allocation_from_logits(
        raw_omega=raw_omega_targ,
        raw_total_mass=raw_total_mass_targ,
        raw_allocation_logits=raw_allocation_logits_targ,
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    ar_sig_targ = ar_spectral_signature(ar_res_targ["ar_coefficients"])
    garch_sig_targ = garch_moment_persistence_signature(garch_res_targ)
    comb_sig_targ = combined_ar_garch_signature(ar_sig_targ, garch_sig_targ)
    
    # Compute matching losses
    ar_loss_res = ar_spectral_matching_loss(ar_sig_cand, ar_sig_targ)
    garch_loss_res = garch_moment_matching_loss(garch_sig_cand, garch_sig_targ)
    comb_loss_res = combined_moment_spectral_matching_loss(comb_sig_cand, comb_sig_targ)
    
    loss_total_tensor = comb_loss_res["loss_total"]
    
    # Differentiability backward verification
    loss_total_tensor.backward()
    
    candidate_raw_kappa_grad_nonzero = bool(torch.any(raw_kappa_cand.grad != 0.0))
    candidate_raw_omega_grad_nonzero = bool(torch.any(raw_omega_cand.grad != 0.0))
    candidate_raw_total_mass_grad_nonzero = bool(torch.any(raw_total_mass_cand.grad != 0.0))
    candidate_raw_allocation_logits_grad_nonzero = bool(torch.any(raw_allocation_logits_cand.grad != 0.0))
    
    loss_total_value = float(loss_total_tensor.detach())
    ar_component_loss_value = float(comb_loss_res["ar_component_loss"].detach())
    garch_component_loss_value = float(comb_loss_res["garch_component_loss"].detach())
    
    total_loss_finite = bool(torch.isfinite(loss_total_tensor.detach()))
    total_loss_nonnegative = bool(loss_total_value >= 0.0)
    ar_component_loss_nonnegative = bool(ar_component_loss_value >= 0.0)
    garch_component_loss_nonnegative = bool(garch_component_loss_value >= 0.0)
    
    summary = {
        "contract_version": MOMENT_SPECTRAL_MATCHING_LOSS_CONTRACT_VERSION,
        "status": FC_VAE_LOSS_STATUS_AVAILABLE,
        "torch_available": True,
        "ar_loss_available": True,
        "garch_loss_available": True,
        "combined_loss_available": True,
        "total_loss_finite": total_loss_finite,
        "total_loss_nonnegative": total_loss_nonnegative,
        "ar_component_loss_nonnegative": ar_component_loss_nonnegative,
        "garch_component_loss_nonnegative": garch_component_loss_nonnegative,
        "candidate_raw_kappa_grad_nonzero": candidate_raw_kappa_grad_nonzero,
        "candidate_raw_omega_grad_nonzero": candidate_raw_omega_grad_nonzero,
        "candidate_raw_total_mass_grad_nonzero": candidate_raw_total_mass_grad_nonzero,
        "candidate_raw_allocation_logits_grad_nonzero": candidate_raw_allocation_logits_grad_nonzero,
        "loss_total_value": loss_total_value,
        "ar_component_loss_value": ar_component_loss_value,
        "garch_component_loss_value": garch_component_loss_value,
        "no_model": True,
        "no_forward_execution": True,
        "no_output_generation": True,
        "no_optimizer": True,
        "no_training_loop": True,
        "no_scientific_conclusion": True,
        "reason": "p46_moment_spectral_matching_loss_probe_success",
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


# Serialization
def moment_spectral_matching_loss_probe_to_json_dict(probe_res: dict) -> dict:
    validate_non_empty_str(probe_res["contract_version"], "contract_version")
    validate_non_empty_str(probe_res["status"], "status")
    validate_non_empty_str(probe_res["reason"], "reason")
    assert_no_local_path_leakage(probe_res["reason"], "reason")
    assert_no_forbidden_claims(probe_res["reason"], "reason")
    
    return {
        "contract_version": probe_res["contract_version"],
        "status": probe_res["status"],
        "torch_available": bool(probe_res["torch_available"]),
        "ar_loss_available": bool(probe_res.get("ar_loss_available", False)),
        "garch_loss_available": bool(probe_res.get("garch_loss_available", False)),
        "combined_loss_available": bool(probe_res.get("combined_loss_available", False)),
        "total_loss_finite": bool(probe_res.get("total_loss_finite", False)),
        "total_loss_nonnegative": bool(probe_res.get("total_loss_nonnegative", False)),
        "ar_component_loss_nonnegative": bool(probe_res.get("ar_component_loss_nonnegative", False)),
        "garch_component_loss_nonnegative": bool(probe_res.get("garch_component_loss_nonnegative", False)),
        "candidate_raw_kappa_grad_nonzero": bool(probe_res.get("candidate_raw_kappa_grad_nonzero", False)),
        "candidate_raw_omega_grad_nonzero": bool(probe_res.get("candidate_raw_omega_grad_nonzero", False)),
        "candidate_raw_total_mass_grad_nonzero": bool(probe_res.get("candidate_raw_total_mass_grad_nonzero", False)),
        "candidate_raw_allocation_logits_grad_nonzero": bool(probe_res.get("candidate_raw_allocation_logits_grad_nonzero", False)),
        "loss_total_value": float(probe_res.get("loss_total_value", 0.0)),
        "ar_component_loss_value": float(probe_res.get("ar_component_loss_value", 0.0)),
        "garch_component_loss_value": float(probe_res.get("garch_component_loss_value", 0.0)),
        "no_model": bool(probe_res["no_model"]),
        "no_forward_execution": bool(probe_res["no_forward_execution"]),
        "no_output_generation": bool(probe_res["no_output_generation"]),
        "no_optimizer": bool(probe_res["no_optimizer"]),
        "no_training_loop": bool(probe_res["no_training_loop"]),
        "no_scientific_conclusion": bool(probe_res["no_scientific_conclusion"]),
        "reason": probe_res["reason"],
    }


def compact_moment_spectral_matching_loss_json(probe_res: dict) -> str:
    d = moment_spectral_matching_loss_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
