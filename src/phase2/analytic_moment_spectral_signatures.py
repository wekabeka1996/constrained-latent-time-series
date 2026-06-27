# src/phase2/analytic_moment_spectral_signatures.py

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

# Constants
ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_CONTRACT_VERSION = "phase2_p45_analytic_moment_spectral_signatures_contract_v1"

ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_KIND = "analytic_moment_spectral_signatures_no_loss_no_model_no_training"

ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_MODULE_NAME = "src.phase2.analytic_moment_spectral_signatures"

P45_DEFAULT_FREQ_COUNT = 16

P45_DEFAULT_GARCH_LAG_COUNT = 8

P45_DEFAULT_EPS = 1.0e-6

# Statuses
FC_VAE_SIGNATURES_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"
FC_VAE_SIGNATURES_STATUS_AVAILABLE = "analytic_moment_spectral_signatures_available_no_loss_no_model_no_training"
FC_VAE_SIGNATURES_STATUS_CONTRACT_MISMATCH = "blocked_by_contract_mismatch"

SUPPORTED_FC_VAE_SIGNATURES_STATUSES = (
    FC_VAE_SIGNATURES_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_SIGNATURES_STATUS_AVAILABLE,
    FC_VAE_SIGNATURES_STATUS_CONTRACT_MISMATCH,
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
def load_torch_for_p45_signatures() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P45 loader.")
    import torch
    return torch


# Frequency Grid
def build_frequency_grid(freq_count: int = P45_DEFAULT_FREQ_COUNT, *, device=None, dtype=None) -> Any:
    validate_positive_int(freq_count, "freq_count")
    torch = load_torch_for_p45_signatures()
    return torch.linspace(0.0, 3.141592653589793, steps=freq_count, device=device, dtype=dtype)


# AR Spectral Signature
def ar_spectral_signature(ar_coefficients: Any, freq_grid: Any = None, eps: float = P45_DEFAULT_EPS) -> dict:
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
        
    torch = load_torch_for_p45_signatures()
    if not torch.is_tensor(ar_coefficients):
        raise TypeError("ar_coefficients must be a PyTorch tensor")
        
    p = ar_coefficients.shape[-1]
    if p < 1 or p > 5:
        raise ValueError(f"AR order p={p} must be between 1 and 5")
        
    if freq_grid is None:
        freq_grid = build_frequency_grid(
            P45_DEFAULT_FREQ_COUNT,
            device=ar_coefficients.device,
            dtype=ar_coefficients.dtype
        )
    elif not torch.is_tensor(freq_grid):
        raise TypeError("freq_grid must be a PyTorch tensor")
        
    freq_count = freq_grid.shape[0]
    
    # Compute: A(w) = 1 - sum_{k=1}^{p} phi_k exp(-i w k)
    lags = torch.arange(1, p + 1, device=ar_coefficients.device, dtype=ar_coefficients.dtype)
    angle = freq_grid.unsqueeze(-1) * lags.unsqueeze(0)  # (freq_count, p)
    
    cos_term = torch.cos(angle)
    sin_term = torch.sin(angle)
    
    real = 1.0 - (ar_coefficients.unsqueeze(-2) * cos_term).sum(dim=-1)
    imag = (ar_coefficients.unsqueeze(-2) * sin_term).sum(dim=-1)
    
    denom_power = real**2 + imag**2 + eps
    spectrum = 1.0 / denom_power
    
    spectrum_log = torch.log(spectrum + eps)
    spectrum_mean = spectrum.mean(dim=-1, keepdim=True)
    spectrum_std = spectrum.std(dim=-1, keepdim=True)
    spectrum_low_freq_power = spectrum[..., 0:1]
    spectrum_high_freq_power = spectrum[..., -1:]
    
    return {
        "ar_spectrum": spectrum,
        "freq_grid": freq_grid,
        "denom_power": denom_power,
        "spectrum_log": spectrum_log,
        "spectrum_mean": spectrum_mean,
        "spectrum_std": spectrum_std,
        "spectrum_low_freq_power": spectrum_low_freq_power,
        "spectrum_high_freq_power": spectrum_high_freq_power,
        "ar_order": p,
        "freq_count": freq_count,
    }


# GARCH Moment / Persistence Signature
def garch_moment_persistence_signature(
    garch_params: dict,
    lag_count: int = P45_DEFAULT_GARCH_LAG_COUNT,
    eps: float = P45_DEFAULT_EPS,
) -> dict:
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    validate_positive_int(lag_count, "lag_count")
    
    torch = load_torch_for_p45_signatures()
    
    omega = garch_params["omega"]
    alpha_sum = garch_params["alpha_sum"]
    beta_sum = garch_params["beta_sum"]
    persistence = garch_params["alpha_beta_sum"]
    stationarity_margin = garch_params["stationarity_margin"]
    
    unconditional_variance = omega / torch.clamp(1.0 - persistence, min=eps)
    
    lags = torch.arange(1, lag_count + 1, device=persistence.device, dtype=persistence.dtype)
    persistence_decay = persistence.pow(lags)
    
    alpha_share = alpha_sum / torch.clamp(persistence, min=eps)
    beta_share = beta_sum / torch.clamp(persistence, min=eps)
    
    return {
        "unconditional_variance": unconditional_variance,
        "persistence": persistence,
        "stationarity_margin": stationarity_margin,
        "alpha_sum": alpha_sum,
        "beta_sum": beta_sum,
        "alpha_share": alpha_share,
        "beta_share": beta_share,
        "persistence_decay": persistence_decay,
        "lag_count": lag_count,
    }


# Combined AR GARCH Signature Summary
def combined_ar_garch_signature(
    ar_signature: dict,
    garch_signature: dict,
    eps: float = P45_DEFAULT_EPS,
) -> dict:
    torch = load_torch_for_p45_signatures()
    
    spectrum_low_freq_power = ar_signature["spectrum_low_freq_power"]
    spectrum_high_freq_power = ar_signature["spectrum_high_freq_power"]
    ar_low_high_ratio = spectrum_low_freq_power / torch.clamp(spectrum_high_freq_power, min=eps)
    
    return {
        "ar_spectrum": ar_signature["ar_spectrum"],
        "ar_spectrum_log": ar_signature["spectrum_log"],
        "ar_spectrum_mean": ar_signature["spectrum_mean"],
        "ar_spectrum_std": ar_signature["spectrum_std"],
        "ar_low_high_ratio": ar_low_high_ratio,
        "garch_unconditional_variance": garch_signature["unconditional_variance"],
        "garch_persistence": garch_signature["persistence"],
        "garch_stationarity_margin": garch_signature["stationarity_margin"],
        "garch_alpha_share": garch_signature["alpha_share"],
        "garch_beta_share": garch_signature["beta_share"],
        "garch_persistence_decay": garch_signature["persistence_decay"],
        "signature_kind": "combined_ar_garch_analytic_signature_no_loss",
    }


# Smoke Probe
def run_analytic_moment_spectral_signatures_probe() -> dict:
    try:
        torch = load_torch_for_p45_signatures()
    except Exception:
        return {
            "contract_version": ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_CONTRACT_VERSION,
            "status": FC_VAE_SIGNATURES_STATUS_TORCH_UNAVAILABLE,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
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
    
    ar_sig = ar_spectral_signature(ar_res["ar_coefficients"])
    garch_sig = garch_moment_persistence_signature(garch_res)
    comb_sig = combined_ar_garch_signature(ar_sig, garch_sig)
    
    ar_spectrum_shape = list(ar_sig["ar_spectrum"].shape)
    garch_persistence_shape = list(garch_sig["persistence"].shape)
    garch_persistence_decay_shape = list(garch_sig["persistence_decay"].shape)
    
    ar_spectrum_positive = bool(torch.all(ar_sig["ar_spectrum"] > 0))
    ar_spectrum_finite = bool(torch.all(torch.isfinite(ar_sig["ar_spectrum"])))
    garch_unconditional_variance_positive = bool(
        torch.all(garch_sig["unconditional_variance"] > 0)
    )
    garch_persistence_below_one = bool(torch.all(garch_sig["persistence"] < 1.0))
    garch_stationarity_margin_positive = bool(
        torch.all(garch_sig["stationarity_margin"] > 0)
    )
    
    beta_share_greater_than_alpha_share = bool(
        torch.all(garch_sig["beta_share"] > garch_sig["alpha_share"])
    )
    
    summary = {
        "contract_version": ANALYTIC_MOMENT_SPECTRAL_SIGNATURES_CONTRACT_VERSION,
        "status": FC_VAE_SIGNATURES_STATUS_AVAILABLE,
        "torch_available": True,
        "ar_signature_available": True,
        "garch_signature_available": True,
        "combined_signature_available": True,
        "ar_spectrum_shape": ar_spectrum_shape,
        "garch_persistence_shape": garch_persistence_shape,
        "garch_persistence_decay_shape": garch_persistence_decay_shape,
        "ar_spectrum_positive": ar_spectrum_positive,
        "ar_spectrum_finite": ar_spectrum_finite,
        "garch_unconditional_variance_positive": garch_unconditional_variance_positive,
        "garch_persistence_below_one": garch_persistence_below_one,
        "garch_stationarity_margin_positive": garch_stationarity_margin_positive,
        "beta_share_greater_than_alpha_share": beta_share_greater_than_alpha_share,
        "no_model": True,
        "no_forward_execution": True,
        "no_output_generation": True,
        "no_loss": True,
        "no_training": True,
        "no_scientific_conclusion": True,
        "reason": "p45_analytic_moment_spectral_signatures_probe_success",
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


# Serialization
def analytic_moment_spectral_signatures_probe_to_json_dict(probe_res: dict) -> dict:
    validate_non_empty_str(probe_res["contract_version"], "contract_version")
    validate_non_empty_str(probe_res["status"], "status")
    validate_non_empty_str(probe_res["reason"], "reason")
    assert_no_local_path_leakage(probe_res["reason"], "reason")
    assert_no_forbidden_claims(probe_res["reason"], "reason")
    
    return {
        "contract_version": probe_res["contract_version"],
        "status": probe_res["status"],
        "torch_available": bool(probe_res["torch_available"]),
        "ar_signature_available": bool(probe_res.get("ar_signature_available", False)),
        "garch_signature_available": bool(probe_res.get("garch_signature_available", False)),
        "combined_signature_available": bool(probe_res.get("combined_signature_available", False)),
        "ar_spectrum_shape": list(probe_res.get("ar_spectrum_shape", [])),
        "garch_persistence_shape": list(probe_res.get("garch_persistence_shape", [])),
        "garch_persistence_decay_shape": list(probe_res.get("garch_persistence_decay_shape", [])),
        "ar_spectrum_positive": bool(probe_res.get("ar_spectrum_positive", False)),
        "ar_spectrum_finite": bool(probe_res.get("ar_spectrum_finite", False)),
        "garch_unconditional_variance_positive": bool(probe_res.get("garch_unconditional_variance_positive", False)),
        "garch_persistence_below_one": bool(probe_res.get("garch_persistence_below_one", False)),
        "garch_stationarity_margin_positive": bool(probe_res.get("garch_stationarity_margin_positive", False)),
        "beta_share_greater_than_alpha_share": bool(probe_res.get("beta_share_greater_than_alpha_share", False)),
        "no_model": bool(probe_res["no_model"]),
        "no_forward_execution": bool(probe_res["no_forward_execution"]),
        "no_output_generation": bool(probe_res["no_output_generation"]),
        "no_loss": bool(probe_res["no_loss"]),
        "no_training": bool(probe_res["no_training"]),
        "no_scientific_conclusion": bool(probe_res["no_scientific_conclusion"]),
        "reason": probe_res["reason"],
    }


def compact_analytic_moment_spectral_signatures_json(probe_res: dict) -> str:
    d = analytic_moment_spectral_signatures_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
