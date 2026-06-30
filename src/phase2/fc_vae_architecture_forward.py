# src/phase2/fc_vae_architecture_forward.py

import json
import math
from typing import Any, Dict, Tuple

from src.phase2.torch_boundary import (
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    build_p50_bridge_targets,
)

# Constants
FC_VAE_ARCHITECTURE_FORWARD_CONTRACT_VERSION = "phase2_p54_fc_vae_architecture_forward_contract_v1"
FC_VAE_ARCHITECTURE_FORWARD_KIND = "fc_vae_architecture_and_forward_pass_no_training"
FC_VAE_ARCHITECTURE_FORWARD_MODULE_NAME = "src.phase2.fc_vae_architecture_forward"
DEFAULT_BETA = 0.0
BETA_STATUS = "declared_not_trained_not_tuned"
FC_VAE_ARCHITECTURE_FORWARD_STATUS_AVAILABLE = "fc_vae_architecture_and_forward_pass_available_no_training"
FC_VAE_ARCHITECTURE_FORWARD_STATUS_BLOCKED = "blocked_torch_unavailable"


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


def load_torch_for_p54_fc_vae() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P54 loader.")
    import torch
    return torch


def build_fc_vae_model(
    latent_mean_dim: int = 4,
    latent_volatility_dim: int = 4,
    latent_shared_dim: int = 4,
    hidden_dim: int = 16,
    ar_order: int = 5,
    alpha_count: int = 1,
    beta_count: int = 1,
) -> Any:
    torch = load_torch_for_p54_fc_vae()
    import torch.nn as nn
    
    class FactorisedConstrainedVAE(nn.Module):
        def __init__(
            self,
            latent_mean_dim: int,
            latent_volatility_dim: int,
            latent_shared_dim: int,
            hidden_dim: int,
            ar_order: int,
            alpha_count: int,
            beta_count: int,
        ):
            super().__init__()
            self.latent_mean_dim = latent_mean_dim
            self.latent_volatility_dim = latent_volatility_dim
            self.latent_shared_dim = latent_shared_dim
            self.hidden_dim = hidden_dim
            self.ar_order = ar_order
            self.alpha_count = alpha_count
            self.beta_count = beta_count
            
            # Shared trunk MLP
            in_dim = latent_mean_dim + latent_volatility_dim + latent_shared_dim
            self.shared_trunk = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            
            # Mean branch
            mean_in_dim = latent_mean_dim + latent_shared_dim + hidden_dim
            self.mean_head = nn.Sequential(
                nn.Linear(mean_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, ar_order),
            )
            
            # Volatility branch
            vol_in_dim = latent_volatility_dim + latent_shared_dim + hidden_dim
            self.volatility_trunk = nn.Sequential(
                nn.Linear(vol_in_dim, hidden_dim),
                nn.ReLU(),
            )
            self.omega_head = nn.Linear(hidden_dim, 1)
            self.mass_head = nn.Linear(hidden_dim, 1)
            self.logits_head = nn.Linear(hidden_dim, alpha_count + beta_count)
            
        def forward(
            self,
            z_mean: Any,
            z_volatility: Any,
            z_shared: Any,
            target_signature: dict = None,
        ) -> dict:
            z_all = torch.cat([z_mean, z_volatility, z_shared], dim=-1)
            shared_hidden = self.shared_trunk(z_all)
            
            # Mean branch
            mean_in = torch.cat([z_mean, z_shared, shared_hidden], dim=-1)
            raw_kappa = self.mean_head(mean_in)
            
            # Volatility branch
            vol_in = torch.cat([z_volatility, z_shared, shared_hidden], dim=-1)
            vol_hidden = self.volatility_trunk(vol_in)
            raw_omega = self.omega_head(vol_hidden)
            raw_total_mass = self.mass_head(vol_hidden)
            raw_allocation_logits = self.logits_head(vol_hidden)
            
            # Pass raw parameters through constraints (P44)
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
            
            ar_res = pacf_to_stable_ar_coefficients(raw_kappa)
            bias = default_garch_allocation_bias(
                alpha_count=self.alpha_count,
                beta_count=self.beta_count,
                dtype=raw_omega.dtype,
                device=raw_omega.device,
            )
            garch_res = garch_mass_allocation_from_logits(
                raw_omega=raw_omega,
                raw_total_mass=raw_total_mass,
                raw_allocation_logits=raw_allocation_logits,
                alpha_count=self.alpha_count,
                beta_count=self.beta_count,
                allocation_bias=bias,
            )
            
            # Produce signatures (P45)
            ar_sig = ar_spectral_signature(ar_res["ar_coefficients"])
            garch_sig = garch_moment_persistence_signature(garch_res)
            combined_sig = combined_ar_garch_signature(ar_sig, garch_sig)
            
            # Optional matching loss (P46)
            loss_dict = None
            if target_signature is not None:
                loss_dict = combined_moment_spectral_matching_loss(combined_sig, target_signature)
                
            return {
                "raw_tensors": {
                    "raw_kappa": raw_kappa,
                    "raw_omega": raw_omega,
                    "raw_total_mass": raw_total_mass,
                    "raw_allocation_logits": raw_allocation_logits,
                },
                "constrained_diagnostics": {
                    "ar_coefficients": ar_res["ar_coefficients"],
                    "omega": garch_res["omega"],
                    "alpha": garch_res["alpha"],
                    "beta": garch_res["beta"],
                    "persistence": garch_res["alpha_beta_sum"],
                    "stationarity_margin": garch_res["stationarity_margin"],
                },
                "combined_signature": combined_sig,
                "loss_dict": loss_dict,
            }
            
    return FactorisedConstrainedVAE(
        latent_mean_dim=latent_mean_dim,
        latent_volatility_dim=latent_volatility_dim,
        latent_shared_dim=latent_shared_dim,
        hidden_dim=hidden_dim,
        ar_order=ar_order,
        alpha_count=alpha_count,
        beta_count=beta_count,
    )


def run_fc_vae_architecture_forward_probe() -> dict:
    try:
        torch = load_torch_for_p54_fc_vae()
    except Exception:
        return {
            "contract_version": FC_VAE_ARCHITECTURE_FORWARD_CONTRACT_VERSION,
            "status": FC_VAE_ARCHITECTURE_FORWARD_STATUS_BLOCKED,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    targets = build_p50_bridge_targets()
    target_sig = targets[0]
    
    model = build_fc_vae_model(
        latent_mean_dim=4,
        latent_volatility_dim=4,
        latent_shared_dim=4,
        hidden_dim=16,
        ar_order=5,
    )
    
    z_mean = torch.tensor([[0.1, -0.05, 0.02, -0.01], [0.05, -0.1, 0.01, -0.02]], dtype=torch.float32)
    z_volatility = torch.tensor([[-0.2, 0.1, -0.05, 0.02], [-0.1, 0.2, -0.02, 0.05]], dtype=torch.float32)
    z_shared = torch.tensor([[0.3, -0.15, 0.05, -0.02], [0.15, -0.3, 0.02, -0.05]], dtype=torch.float32)
    
    fwd_res = model(z_mean, z_volatility, z_shared, target_signature=target_sig)
    
    raw_kappa = fwd_res["raw_tensors"]["raw_kappa"]
    raw_omega = fwd_res["raw_tensors"]["raw_omega"]
    raw_total_mass = fwd_res["raw_tensors"]["raw_total_mass"]
    raw_allocation_logits = fwd_res["raw_tensors"]["raw_allocation_logits"]
    
    ar_coefs = fwd_res["constrained_diagnostics"]["ar_coefficients"]
    garch_omega = fwd_res["constrained_diagnostics"]["omega"]
    garch_alpha = fwd_res["constrained_diagnostics"]["alpha"]
    garch_beta = fwd_res["constrained_diagnostics"]["beta"]
    garch_persistence = fwd_res["constrained_diagnostics"]["persistence"]
    garch_margin = fwd_res["constrained_diagnostics"]["stationarity_margin"]
    
    combined_sig = fwd_res["combined_signature"]
    loss_dict = fwd_res["loss_dict"]
    
    losses_finite = math.isfinite(loss_dict["loss_total"].item())
    
    raw_kappa_finite = torch.all(torch.isfinite(raw_kappa)).item()
    raw_omega_finite = torch.all(torch.isfinite(raw_omega)).item()
    raw_total_mass_finite = torch.all(torch.isfinite(raw_total_mass)).item()
    raw_allocation_logits_finite = torch.all(torch.isfinite(raw_allocation_logits)).item()
    
    ar_coefs_finite = torch.all(torch.isfinite(ar_coefs)).item()
    garch_omega_finite = torch.all(torch.isfinite(garch_omega)).item()
    garch_alpha_finite = torch.all(torch.isfinite(garch_alpha)).item()
    garch_beta_finite = torch.all(torch.isfinite(garch_beta)).item()
    garch_persistence_finite = torch.all(torch.isfinite(garch_persistence)).item()
    garch_margin_finite = torch.all(torch.isfinite(garch_margin)).item()
    
    sig_all_finite = True
    tensor_field_count = 0
    scalar_field_count = 0
    for k, v in combined_sig.items():
        if torch.is_tensor(v):
            tensor_field_count += 1
            if not torch.all(torch.isfinite(v)).item():
                sig_all_finite = False
        else:
            scalar_field_count += 1
            
    summary = {
        "contract_version": FC_VAE_ARCHITECTURE_FORWARD_CONTRACT_VERSION,
        "kind": FC_VAE_ARCHITECTURE_FORWARD_KIND,
        "status": FC_VAE_ARCHITECTURE_FORWARD_STATUS_AVAILABLE,
        "torch_available": True,
        
        "z_mean_shape": list(z_mean.shape),
        "z_volatility_shape": list(z_volatility.shape),
        "z_shared_shape": list(z_shared.shape),
        
        "raw_kappa_shape": list(raw_kappa.shape),
        "raw_omega_shape": list(raw_omega.shape),
        "raw_total_mass_shape": list(raw_total_mass.shape),
        "raw_allocation_logits_shape": list(raw_allocation_logits.shape),
        
        "raw_kappa_finite": bool(raw_kappa_finite),
        "raw_omega_finite": bool(raw_omega_finite),
        "raw_total_mass_finite": bool(raw_total_mass_finite),
        "raw_allocation_logits_finite": bool(raw_allocation_logits_finite),
        
        "ar_coefficients_shape": list(ar_coefs.shape),
        "ar_coefficients_finite": bool(ar_coefs_finite),
        "garch_omega_shape": list(garch_omega.shape),
        "garch_omega_finite": bool(garch_omega_finite),
        "garch_alpha_shape": list(garch_alpha.shape),
        "garch_alpha_finite": bool(garch_alpha_finite),
        "garch_beta_shape": list(garch_beta.shape),
        "garch_beta_finite": bool(garch_beta_finite),
        "garch_persistence_shape": list(garch_persistence.shape),
        "garch_persistence_finite": bool(garch_persistence_finite),
        "garch_margin_shape": list(garch_margin.shape),
        "garch_margin_finite": bool(garch_margin_finite),
        
        "signature_tensor_field_count": int(tensor_field_count),
        "signature_scalar_field_count": int(scalar_field_count),
        "signature_all_tensor_fields_finite": bool(sig_all_finite),
        
        "loss_total_value": float(loss_dict["loss_total"].item()),
        "loss_total_finite": bool(losses_finite),
        
        "default_beta": float(DEFAULT_BETA),
        "beta_status": BETA_STATUS,
        
        "no_model_claim": True,
        "no_training": True,
        "no_optimizer": True,
        "no_dataset": True,
        "no_dataloader": True,
        "no_gsb_claim": True,
        "no_generation_claim": True,
        "no_scientific_conclusion": True,
        "realizability_claim": "fc_vae_architecture_and_forward_pass_only_no_training",
        "reason": "fc_vae_forward_dry_run_success"
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def fc_vae_architecture_forward_probe_to_json_dict(probe_res: dict) -> dict:
    validate_non_empty_str(probe_res["contract_version"], "contract_version")
    validate_non_empty_str(probe_res["status"], "status")
    validate_non_empty_str(probe_res["reason"], "reason")
    assert_no_local_path_leakage(probe_res["reason"], "reason")
    assert_no_forbidden_claims(probe_res["reason"], "reason")
    
    return {
        "contract_version": probe_res["contract_version"],
        "kind": probe_res.get("kind", FC_VAE_ARCHITECTURE_FORWARD_KIND),
        "status": probe_res["status"],
        "torch_available": bool(probe_res["torch_available"]),
        "z_mean_shape": list(probe_res.get("z_mean_shape", [])),
        "z_volatility_shape": list(probe_res.get("z_volatility_shape", [])),
        "z_shared_shape": list(probe_res.get("z_shared_shape", [])),
        "raw_kappa_shape": list(probe_res.get("raw_kappa_shape", [])),
        "raw_omega_shape": list(probe_res.get("raw_omega_shape", [])),
        "raw_total_mass_shape": list(probe_res.get("raw_total_mass_shape", [])),
        "raw_allocation_logits_shape": list(probe_res.get("raw_allocation_logits_shape", [])),
        "raw_kappa_finite": bool(probe_res.get("raw_kappa_finite", False)),
        "raw_omega_finite": bool(probe_res.get("raw_omega_finite", False)),
        "raw_total_mass_finite": bool(probe_res.get("raw_total_mass_finite", False)),
        "raw_allocation_logits_finite": bool(probe_res.get("raw_allocation_logits_finite", False)),
        "ar_coefficients_shape": list(probe_res.get("ar_coefficients_shape", [])),
        "ar_coefficients_finite": bool(probe_res.get("ar_coefficients_finite", False)),
        "garch_omega_shape": list(probe_res.get("garch_omega_shape", [])),
        "garch_omega_finite": bool(probe_res.get("garch_omega_finite", False)),
        "garch_alpha_shape": list(probe_res.get("garch_alpha_shape", [])),
        "garch_alpha_finite": bool(probe_res.get("garch_alpha_finite", False)),
        "garch_beta_shape": list(probe_res.get("garch_beta_shape", [])),
        "garch_beta_finite": bool(probe_res.get("garch_beta_finite", False)),
        "garch_persistence_shape": list(probe_res.get("garch_persistence_shape", [])),
        "garch_persistence_finite": bool(probe_res.get("garch_persistence_finite", False)),
        "garch_margin_shape": list(probe_res.get("garch_margin_shape", [])),
        "garch_margin_finite": bool(probe_res.get("garch_margin_finite", False)),
        "signature_tensor_field_count": int(probe_res.get("signature_tensor_field_count", 0)),
        "signature_scalar_field_count": int(probe_res.get("signature_scalar_field_count", 0)),
        "signature_all_tensor_fields_finite": bool(probe_res.get("signature_all_tensor_fields_finite", False)),
        "loss_total_value": float(probe_res.get("loss_total_value", 0.0)) if probe_res.get("loss_total_value") is not None else None,
        "loss_total_finite": bool(probe_res.get("loss_total_finite", False)),
        "default_beta": float(probe_res.get("default_beta", 0.0)),
        "beta_status": probe_res.get("beta_status", BETA_STATUS),
        "no_model_claim": bool(probe_res.get("no_model_claim", True)),
        "no_training": bool(probe_res.get("no_training", True)),
        "no_optimizer": bool(probe_res.get("no_optimizer", True)),
        "no_dataset": bool(probe_res.get("no_dataset", True)),
        "no_dataloader": bool(probe_res.get("no_dataloader", True)),
        "no_gsb_claim": bool(probe_res.get("no_gsb_claim", True)),
        "no_generation_claim": bool(probe_res.get("no_generation_claim", True)),
        "no_scientific_conclusion": bool(probe_res.get("no_scientific_conclusion", True)),
        "realizability_claim": probe_res.get("realizability_claim", "fc_vae_architecture_and_forward_pass_only_no_training"),
        "reason": probe_res["reason"],
    }


def compact_fc_vae_architecture_forward_json(probe_res: dict) -> str:
    d = fc_vae_architecture_forward_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
