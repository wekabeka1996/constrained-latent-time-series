# src/phase2/tensor_native_constraint_primitives.py

import dataclasses
import json
from typing import Any, Tuple

from src.phase2.torch_boundary import (
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)

# Constants
TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_CONTRACT_VERSION = "phase2_p44_tensor_native_constraint_primitives_contract_v1"

TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_KIND = "tensor_native_constraint_primitives_no_model_no_loss_no_training"

TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_MODULE_NAME = "src.phase2.tensor_native_constraint_primitives"

P44_MAX_AR_ORDER = 5

P44_DEFAULT_EPS = 1.0e-4

# Statuses
FC_VAE_PRIMITIVES_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"
FC_VAE_PRIMITIVES_STATUS_AVAILABLE = "tensor_native_constraint_primitives_available_no_model_no_loss_no_training"
FC_VAE_PRIMITIVES_STATUS_CONTRACT_MISMATCH = "blocked_by_contract_mismatch"

SUPPORTED_FC_VAE_PRIMITIVES_STATUSES = (
    FC_VAE_PRIMITIVES_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_PRIMITIVES_STATUS_AVAILABLE,
    FC_VAE_PRIMITIVES_STATUS_CONTRACT_MISMATCH,
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
def load_torch_for_p44_tensor_primitives() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P44 loader.")
    import torch
    return torch


# PACF -> Stable AR coefficients
def pacf_to_stable_ar_coefficients(raw_kappa: Any, eps: float = P44_DEFAULT_EPS) -> dict:
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    
    torch = load_torch_for_p44_tensor_primitives()
    if not torch.is_tensor(raw_kappa):
        raise TypeError("raw_kappa must be a PyTorch tensor")
    
    p = raw_kappa.shape[-1]
    if p < 1 or p > P44_MAX_AR_ORDER:
        raise ValueError(f"AR order p={p} must be between 1 and {P44_MAX_AR_ORDER}")
    
    # Constrain reflection coefficients strictly to (-1, 1) with eps margin
    kappa = (1.0 - eps) * torch.tanh(raw_kappa)
    
    # Durbin recursion (batched)
    ar = kappa[..., 0:1]  # shape (..., 1)
    
    for k in range(1, p):
        kappa_k = kappa[..., k:k+1]  # shape (..., 1)
        ar_reversed = torch.flip(ar, dims=[-1])
        ar_updated = ar - kappa_k * ar_reversed
        ar = torch.cat([ar_updated, kappa_k], dim=-1)
        
    max_abs_pacf = torch.max(torch.abs(kappa))
    
    return {
        "ar_coefficients": ar,
        "pacf_values": kappa,
        "max_abs_pacf": max_abs_pacf,
        "eps_margin": eps,
        "order": p,
        "stationarity_parameterization": "pacf_reflection_coefficients",
    }


# GARCH Mass Allocation
def garch_mass_allocation_from_logits(
    raw_omega: Any,
    raw_total_mass: Any,
    raw_allocation_logits: Any,
    alpha_count: int,
    beta_count: int,
    eps: float = P44_DEFAULT_EPS,
    allocation_bias: Any = None,
) -> dict:
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    validate_positive_int(alpha_count, "alpha_count")
    validate_positive_int(beta_count, "beta_count")
    
    torch = load_torch_for_p44_tensor_primitives()
    
    if not torch.is_tensor(raw_omega):
        raise TypeError("raw_omega must be a PyTorch tensor")
    if not torch.is_tensor(raw_total_mass):
        raise TypeError("raw_total_mass must be a PyTorch tensor")
    if not torch.is_tensor(raw_allocation_logits):
        raise TypeError("raw_allocation_logits must be a PyTorch tensor")
        
    omega = torch.nn.functional.softplus(raw_omega) + eps
    
    logits = raw_allocation_logits
    if allocation_bias is not None:
        if not torch.is_tensor(allocation_bias):
            raise TypeError("allocation_bias must be a PyTorch tensor")
        logits = logits + allocation_bias
        
    total_mass = (1.0 - eps) * torch.sigmoid(raw_total_mass)
    allocation = torch.softmax(logits, dim=-1)
    
    alpha_beta = total_mass * allocation
    alpha = alpha_beta[..., :alpha_count]
    beta = alpha_beta[..., alpha_count:]
    
    alpha_sum = alpha.sum(dim=-1, keepdim=True)
    beta_sum = beta.sum(dim=-1, keepdim=True)
    alpha_beta_sum = alpha_sum + beta_sum
    
    stationarity_margin = (1.0 - eps) - alpha_beta_sum
    
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "alpha_sum": alpha_sum,
        "beta_sum": beta_sum,
        "alpha_beta_sum": alpha_beta_sum,
        "stationarity_margin": stationarity_margin,
        "total_mass": total_mass,
        "allocation": allocation,
        "eps_margin": eps,
    }


# default GARCH allocation bias prior
def default_garch_allocation_bias(
    alpha_count: int,
    beta_count: int,
    beta_bias: float = 2.0,
    alpha_bias: float = -1.0,
    *,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    validate_positive_int(alpha_count, "alpha_count")
    validate_positive_int(beta_count, "beta_count")
    
    torch = load_torch_for_p44_tensor_primitives()
    
    alpha_part = torch.full((alpha_count,), float(alpha_bias), device=device, dtype=dtype)
    beta_part = torch.full((beta_count,), float(beta_bias), device=device, dtype=dtype)
    
    return torch.cat([alpha_part, beta_part], dim=0)


# Smoke Probe
def run_tensor_native_constraint_primitives_probe() -> dict:
    try:
        torch = load_torch_for_p44_tensor_primitives()
    except Exception:
        return {
            "contract_version": TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_CONTRACT_VERSION,
            "status": FC_VAE_PRIMITIVES_STATUS_TORCH_UNAVAILABLE,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    # batch_size = 2, ar_order = 5
    raw_kappa = torch.tensor(
        [[0.1, -0.05, 0.02, -0.01, 0.005],
         [-0.1, 0.05, -0.02, 0.01, -0.005]],
        dtype=torch.float32
    )
    ar_res = pacf_to_stable_ar_coefficients(raw_kappa)
    
    raw_omega = torch.tensor([[-0.5], [0.5]], dtype=torch.float32)
    raw_total_mass = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    bias = default_garch_allocation_bias(1, 1, dtype=torch.float32)
    raw_allocation_logits = torch.zeros((2, 2), dtype=torch.float32)
    
    garch_res = garch_mass_allocation_from_logits(
        raw_omega=raw_omega,
        raw_total_mass=raw_total_mass,
        raw_allocation_logits=raw_allocation_logits,
        alpha_count=1,
        beta_count=1,
        allocation_bias=bias,
    )
    
    ar_order = ar_res["order"]
    max_abs_pacf_val = float(ar_res["max_abs_pacf"].item())
    pacf_abs_max_lt_one = max_abs_pacf_val < 1.0
    ar_coefficients_shape = list(ar_res["ar_coefficients"].shape)
    
    garch_omega_positive = bool(torch.all(garch_res["omega"] > 0).item())
    garch_alpha_nonnegative = bool(torch.all(garch_res["alpha"] >= 0).item())
    garch_beta_nonnegative = bool(torch.all(garch_res["beta"] >= 0).item())
    garch_alpha_beta_sum_below_one_minus_eps = bool(
        torch.all(garch_res["alpha_beta_sum"] < 1.0 - garch_res["eps_margin"]).item()
    )
    garch_stationarity_margin_positive = bool(
        torch.all(garch_res["stationarity_margin"] > 0).item()
    )
    
    beta_prior_greater_than_alpha_prior = bias[1].item() > bias[0].item()
    
    summary = {
        "contract_version": TENSOR_NATIVE_CONSTRAINT_PRIMITIVES_CONTRACT_VERSION,
        "status": FC_VAE_PRIMITIVES_STATUS_AVAILABLE,
        "torch_available": True,
        "ar_order": ar_order,
        "pacf_abs_max_lt_one": pacf_abs_max_lt_one,
        "ar_coefficients_shape": ar_coefficients_shape,
        "garch_omega_positive": garch_omega_positive,
        "garch_alpha_nonnegative": garch_alpha_nonnegative,
        "garch_beta_nonnegative": garch_beta_nonnegative,
        "garch_alpha_beta_sum_below_one_minus_eps": garch_alpha_beta_sum_below_one_minus_eps,
        "garch_stationarity_margin_positive": garch_stationarity_margin_positive,
        "beta_prior_greater_than_alpha_prior": beta_prior_greater_than_alpha_prior,
        "no_model": True,
        "no_forward_execution": True,
        "no_output_generation": True,
        "no_loss": True,
        "no_training": True,
        "no_scientific_conclusion": True,
        "reason": "p44_tensor_native_constraint_primitives_probe_success",
    }
    
    # Assert validation
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


# Serialization
def tensor_native_constraint_primitives_probe_to_json_dict(probe_res: dict) -> dict:
    # Validate the dictionary structure
    validate_non_empty_str(probe_res["contract_version"], "contract_version")
    validate_non_empty_str(probe_res["status"], "status")
    validate_non_empty_str(probe_res["reason"], "reason")
    assert_no_local_path_leakage(probe_res["reason"], "reason")
    assert_no_forbidden_claims(probe_res["reason"], "reason")
    
    return {
        "contract_version": probe_res["contract_version"],
        "status": probe_res["status"],
        "torch_available": bool(probe_res["torch_available"]),
        "ar_order": int(probe_res["ar_order"]) if "ar_order" in probe_res else 0,
        "pacf_abs_max_lt_one": bool(probe_res["pacf_abs_max_lt_one"]) if "pacf_abs_max_lt_one" in probe_res else False,
        "ar_coefficients_shape": list(probe_res["ar_coefficients_shape"]) if "ar_coefficients_shape" in probe_res else [],
        "garch_omega_positive": bool(probe_res["garch_omega_positive"]) if "garch_omega_positive" in probe_res else False,
        "garch_alpha_nonnegative": bool(probe_res["garch_alpha_nonnegative"]) if "garch_alpha_nonnegative" in probe_res else False,
        "garch_beta_nonnegative": bool(probe_res["garch_beta_nonnegative"]) if "garch_beta_nonnegative" in probe_res else False,
        "garch_alpha_beta_sum_below_one_minus_eps": bool(probe_res["garch_alpha_beta_sum_below_one_minus_eps"]) if "garch_alpha_beta_sum_below_one_minus_eps" in probe_res else False,
        "garch_stationarity_margin_positive": bool(probe_res["garch_stationarity_margin_positive"]) if "garch_stationarity_margin_positive" in probe_res else False,
        "beta_prior_greater_than_alpha_prior": bool(probe_res["beta_prior_greater_than_alpha_prior"]) if "beta_prior_greater_than_alpha_prior" in probe_res else False,
        "no_model": bool(probe_res["no_model"]),
        "no_forward_execution": bool(probe_res["no_forward_execution"]),
        "no_output_generation": bool(probe_res["no_output_generation"]),
        "no_loss": bool(probe_res["no_loss"]),
        "no_training": bool(probe_res["no_training"]),
        "no_scientific_conclusion": bool(probe_res["no_scientific_conclusion"]),
        "reason": probe_res["reason"],
    }


def compact_tensor_native_constraint_primitives_json(probe_res: dict) -> str:
    d = tensor_native_constraint_primitives_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
