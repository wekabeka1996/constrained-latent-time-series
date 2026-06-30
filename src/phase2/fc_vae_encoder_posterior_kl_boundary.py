# src/phase2/fc_vae_encoder_posterior_kl_boundary.py

import json
import math
from typing import Any, Dict, List, Tuple

from src.phase2.torch_boundary import (
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    build_p50_bridge_targets,
)
from src.phase2.fc_vae_architecture_forward import (
    build_fc_vae_model,
    DEFAULT_BETA,
    BETA_STATUS,
)
from src.phase2.bridge_path_consistency_diagnostics import (
    signature_tensor_field_names,
)

# ── Constants ──

FC_VAE_ENCODER_POSTERIOR_KL_CONTRACT_VERSION = "phase2_p55_fc_vae_encoder_posterior_kl_boundary_contract_v1"
FC_VAE_ENCODER_POSTERIOR_KL_KIND = "fc_vae_encoder_posterior_kl_boundary_no_training"
FC_VAE_ENCODER_POSTERIOR_KL_MODULE_NAME = "src.phase2.fc_vae_encoder_posterior_kl_boundary"
FC_VAE_ENCODER_POSTERIOR_KL_STATUS_AVAILABLE = "fc_vae_encoder_posterior_kl_boundary_available_no_training"
FC_VAE_ENCODER_POSTERIOR_KL_STATUS_BLOCKED = "blocked_torch_unavailable"

P55_LATENT_MEAN_DIM = 4
P55_LATENT_VOLATILITY_DIM = 4
P55_LATENT_SHARED_DIM = 4
P55_ENCODER_HIDDEN_DIM = 16
P55_SIGNATURE_VECTOR_TOTAL_LENGTH = 96
P55_SIGNATURE_VECTOR_PER_SAMPLE_LENGTH = 48
P55_EXPECTED_BATCH_SIZE = 2
P55_TARGET_COUNT = 3
P55_EPS_MODE = "zero"
P55_BETA = 0.0
P55_BETA_STATUS = "declared_not_trained_not_tuned"

P55_SIGNATURE_TENSOR_FIELD_COUNT = 11


# ── Validation helpers ──

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
    forbidden_pairs = [
        ("scientific", "success"), ("gsb", "solved"), ("vae", "works"),
        ("latent", "learned"), ("semantic", "geometry"),
        ("production", "ready"), ("state", "art"),
        ("schrodinger", "bridge"), ("generated", "c"),
    ]
    for a, b in forbidden_pairs:
        if a in val_lower and b in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}")


# ── Torch loader ──

def load_torch_for_p55_encoder_kl() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P55 loader.")
    import torch
    return torch


# ── Signature vector contract ──

def build_p55_signature_vector_from_target(target_signature: dict) -> dict:
    """Flatten a P45 combined signature dict into a per-sample tensor vector.
    
    Each tensor field has shape (batch, ...). We flatten each field's non-batch
    dimensions and concatenate along the last axis, producing shape (batch, per_sample_length).
    """
    torch = load_torch_for_p55_encoder_kl()
    field_names = signature_tensor_field_names(target_signature)
    
    pieces = []
    for fname in field_names:
        t = target_signature[fname]
        if not torch.is_tensor(t):
            continue
        # shape is (batch, ...), flatten non-batch dims
        batch_size = t.shape[0]
        flat = t.reshape(batch_size, -1)
        pieces.append(flat)
    
    signature_vector = torch.cat(pieces, dim=-1)
    
    return {
        "field_names": field_names,
        "signature_vector": signature_vector,
        "signature_vector_batch_shape": list(signature_vector.shape),
        "signature_vector_per_sample_length": int(signature_vector.shape[-1]),
        "signature_vector_total_length": int(signature_vector.numel()),
        "signature_tensor_field_count": len(field_names),
    }


# ── Reparameterization ──

def reparameterize_latent(mu: Any, logvar: Any, eps_mode: str = "zero") -> dict:
    """Reparameterize: z = mu + eps * std, where std = exp(0.5 * logvar)."""
    torch = load_torch_for_p55_encoder_kl()
    
    std = (0.5 * logvar).exp()
    
    if eps_mode == "zero":
        eps = torch.zeros_like(std)
    elif eps_mode == "ones":
        eps = torch.ones_like(std)
    elif eps_mode == "sample":
        eps = torch.randn_like(std)
    else:
        raise ValueError(f"Unknown eps_mode: {eps_mode}")
    
    z = mu + eps * std
    
    return {
        "z": z,
        "std": std,
        "eps": eps,
    }


# ── KL divergence ──

def compute_diag_gaussian_kl_to_standard_normal(mu: Any, logvar: Any) -> Any:
    """KL(q(z|x) || p(z)) for diagonal Gaussian q against standard normal p.
    
    Returns tensor of shape (batch,).
    """
    # kl_per_sample = -0.5 * sum(1 + logvar - mu^2 - exp(logvar), dim=-1)
    torch = load_torch_for_p55_encoder_kl()
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
    return kl


def compute_factorised_kl_dict(posterior: dict) -> dict:
    """Compute per-group and total KL divergences from posterior parameters."""
    kl_mean = compute_diag_gaussian_kl_to_standard_normal(
        posterior["z_mean_mu"], posterior["z_mean_logvar"]
    )
    kl_volatility = compute_diag_gaussian_kl_to_standard_normal(
        posterior["z_volatility_mu"], posterior["z_volatility_logvar"]
    )
    kl_shared = compute_diag_gaussian_kl_to_standard_normal(
        posterior["z_shared_mu"], posterior["z_shared_logvar"]
    )
    
    kl_total = kl_mean + kl_volatility + kl_shared
    
    return {
        "kl_mean_per_sample": kl_mean,
        "kl_volatility_per_sample": kl_volatility,
        "kl_shared_per_sample": kl_shared,
        "kl_total_per_sample": kl_total,
        "kl_mean_batch_mean": kl_mean.mean(),
        "kl_volatility_batch_mean": kl_volatility.mean(),
        "kl_shared_batch_mean": kl_shared.mean(),
        "kl_total_batch_mean": kl_total.mean(),
    }


# ── Encoder + full VAE model builder ──

def build_fc_vae_encoder_posterior_model(
    signature_per_sample_length: int = P55_SIGNATURE_VECTOR_PER_SAMPLE_LENGTH,
    latent_mean_dim: int = P55_LATENT_MEAN_DIM,
    latent_volatility_dim: int = P55_LATENT_VOLATILITY_DIM,
    latent_shared_dim: int = P55_LATENT_SHARED_DIM,
    encoder_hidden_dim: int = P55_ENCODER_HIDDEN_DIM,
    decoder_hidden_dim: int = 16,
    ar_order: int = 5,
    alpha_count: int = 1,
    beta_count: int = 1,
) -> Any:
    """Build a FactorisedConstrainedVAEWithEncoderKL nn.Module.
    
    This module contains:
    - Encoder trunk (MLP)
    - Posterior heads (mu/logvar for z_mean, z_volatility, z_shared)
    - P54 decoder (build_fc_vae_model)
    """
    torch = load_torch_for_p55_encoder_kl()
    import torch.nn as nn
    
    # Build the P54 decoder
    decoder = build_fc_vae_model(
        latent_mean_dim=latent_mean_dim,
        latent_volatility_dim=latent_volatility_dim,
        latent_shared_dim=latent_shared_dim,
        hidden_dim=decoder_hidden_dim,
        ar_order=ar_order,
        alpha_count=alpha_count,
        beta_count=beta_count,
    )
    
    class FactorisedConstrainedVAEWithEncoderKL(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Encoder trunk
            self.encoder_trunk = nn.Sequential(
                nn.Linear(signature_per_sample_length, encoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
                nn.ReLU(),
            )
            
            # Posterior heads for z_mean
            self.z_mean_mu_head = nn.Linear(encoder_hidden_dim, latent_mean_dim)
            self.z_mean_logvar_head = nn.Linear(encoder_hidden_dim, latent_mean_dim)
            
            # Posterior heads for z_volatility
            self.z_volatility_mu_head = nn.Linear(encoder_hidden_dim, latent_volatility_dim)
            self.z_volatility_logvar_head = nn.Linear(encoder_hidden_dim, latent_volatility_dim)
            
            # Posterior heads for z_shared
            self.z_shared_mu_head = nn.Linear(encoder_hidden_dim, latent_shared_dim)
            self.z_shared_logvar_head = nn.Linear(encoder_hidden_dim, latent_shared_dim)
            
            # P54 decoder
            self.decoder = decoder
        
        def encode(self, signature_vector: Any) -> dict:
            """Encode a flattened signature vector into posterior parameters."""
            h = self.encoder_trunk(signature_vector)
            
            return {
                "z_mean_mu": self.z_mean_mu_head(h),
                "z_mean_logvar": self.z_mean_logvar_head(h),
                "z_volatility_mu": self.z_volatility_mu_head(h),
                "z_volatility_logvar": self.z_volatility_logvar_head(h),
                "z_shared_mu": self.z_shared_mu_head(h),
                "z_shared_logvar": self.z_shared_logvar_head(h),
            }
        
        def forward(
            self,
            target_signature: dict,
            eps_mode: str = "zero",
            beta: float = P55_BETA,
        ) -> dict:
            # 1. Build signature vector from target
            sig_contract = build_p55_signature_vector_from_target(target_signature)
            signature_vector = sig_contract["signature_vector"]
            
            # 2. Encode
            posterior = self.encode(signature_vector)
            
            # 3. Reparameterize each latent group
            reparam_mean = reparameterize_latent(
                posterior["z_mean_mu"], posterior["z_mean_logvar"], eps_mode
            )
            reparam_vol = reparameterize_latent(
                posterior["z_volatility_mu"], posterior["z_volatility_logvar"], eps_mode
            )
            reparam_shared = reparameterize_latent(
                posterior["z_shared_mu"], posterior["z_shared_logvar"], eps_mode
            )
            
            z_mean = reparam_mean["z"]
            z_volatility = reparam_vol["z"]
            z_shared = reparam_shared["z"]
            
            latent_samples = {
                "z_mean": z_mean,
                "z_volatility": z_volatility,
                "z_shared": z_shared,
            }
            
            # 4. Decode through P54 decoder
            decoder_fwd = self.decoder(
                z_mean, z_volatility, z_shared,
                target_signature=target_signature,
            )
            
            # 5. Compute factorised KL
            kl_dict = compute_factorised_kl_dict(posterior)
            
            # 6. Compute total loss if reconstruction loss available
            loss_dict = decoder_fwd.get("loss_dict")
            total_loss = None
            if loss_dict is not None and "loss_total" in loss_dict:
                recon_loss = loss_dict["loss_total"]
                kl_loss = kl_dict["kl_total_batch_mean"]
                total_loss = recon_loss + beta * kl_loss
            
            return {
                "signature_vector_contract": sig_contract,
                "posterior": posterior,
                "latent_samples": latent_samples,
                "decoder_forward": decoder_fwd,
                "kl_dict": kl_dict,
                "loss_dict": loss_dict,
                "total_loss": total_loss,
                "beta": beta,
            }
    
    return FactorisedConstrainedVAEWithEncoderKL()


# ── Dry-run probe ──

def run_fc_vae_encoder_posterior_kl_probe() -> dict:
    try:
        torch = load_torch_for_p55_encoder_kl()
    except Exception:
        return {
            "contract_version": FC_VAE_ENCODER_POSTERIOR_KL_CONTRACT_VERSION,
            "status": FC_VAE_ENCODER_POSTERIOR_KL_STATUS_BLOCKED,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
    
    # Load deterministic target
    targets = build_p50_bridge_targets()
    target = targets[0]  # bridge_lambda_0_25
    target_id = target.get("target_id", "bridge_lambda_0_25")
    target_sig = target
    
    # Build model
    model = build_fc_vae_encoder_posterior_model()
    
    # Forward pass
    fwd = model(target_sig, eps_mode=P55_EPS_MODE, beta=P55_BETA)
    
    # Extract results
    sig_contract = fwd["signature_vector_contract"]
    posterior = fwd["posterior"]
    latent_samples = fwd["latent_samples"]
    decoder_fwd = fwd["decoder_forward"]
    kl_dict = fwd["kl_dict"]
    loss_dict = fwd["loss_dict"]
    total_loss = fwd["total_loss"]
    
    # Posterior finiteness checks
    posterior_all_mu_finite = all(
        torch.all(torch.isfinite(posterior[k])).item()
        for k in ["z_mean_mu", "z_volatility_mu", "z_shared_mu"]
    )
    posterior_all_logvar_finite = all(
        torch.all(torch.isfinite(posterior[k])).item()
        for k in ["z_mean_logvar", "z_volatility_logvar", "z_shared_logvar"]
    )
    
    # Latent samples finiteness
    latent_samples_finite = all(
        torch.all(torch.isfinite(latent_samples[k])).item()
        for k in ["z_mean", "z_volatility", "z_shared"]
    )
    
    # KL values
    kl_mean_val = float(kl_dict["kl_mean_batch_mean"].item())
    kl_vol_val = float(kl_dict["kl_volatility_batch_mean"].item())
    kl_shared_val = float(kl_dict["kl_shared_batch_mean"].item())
    kl_total_val = float(kl_dict["kl_total_batch_mean"].item())
    
    kl_values_finite = all(math.isfinite(v) for v in [kl_mean_val, kl_vol_val, kl_shared_val, kl_total_val])
    kl_values_non_negative = all(v >= 0.0 for v in [kl_mean_val, kl_vol_val, kl_shared_val, kl_total_val])
    
    # Decoder raw shapes
    raw_tensors = decoder_fwd["raw_tensors"]
    
    # Decoder signature finiteness
    combined_sig = decoder_fwd["combined_signature"]
    decoder_sig_all_finite = True
    decoder_sig_field_count = 0
    for k, v in combined_sig.items():
        if torch.is_tensor(v):
            decoder_sig_field_count += 1
            if not torch.all(torch.isfinite(v)).item():
                decoder_sig_all_finite = False
    
    # Reconstruction loss
    recon_loss_available = (loss_dict is not None and "loss_total" in loss_dict)
    recon_loss_val = None
    recon_loss_finite = False
    if recon_loss_available:
        recon_loss_val = float(loss_dict["loss_total"].item())
        recon_loss_finite = math.isfinite(recon_loss_val)
    
    # Total loss
    total_loss_available = total_loss is not None
    total_loss_val = None
    total_loss_finite = False
    if total_loss_available:
        total_loss_val = float(total_loss.item())
        total_loss_finite = math.isfinite(total_loss_val)
    
    # Overall pass check
    all_checks_pass = (
        posterior_all_mu_finite
        and posterior_all_logvar_finite
        and latent_samples_finite
        and kl_values_finite
        and kl_values_non_negative
        and decoder_sig_all_finite
        and (not recon_loss_available or recon_loss_finite)
        and (not total_loss_available or total_loss_finite)
    )
    
    summary = {
        "contract_version": FC_VAE_ENCODER_POSTERIOR_KL_CONTRACT_VERSION,
        "kind": FC_VAE_ENCODER_POSTERIOR_KL_KIND,
        "status": FC_VAE_ENCODER_POSTERIOR_KL_STATUS_AVAILABLE,
        "torch_available": True,
        "source_phase": "P55",
        "target_id": target_id,
        
        # Signature vector contract
        "signature_tensor_field_count": sig_contract["signature_tensor_field_count"],
        "signature_vector_batch_shape": sig_contract["signature_vector_batch_shape"],
        "signature_vector_per_sample_length": sig_contract["signature_vector_per_sample_length"],
        "signature_vector_total_length": sig_contract["signature_vector_total_length"],
        
        # Posterior shapes
        "z_mean_mu_shape": list(posterior["z_mean_mu"].shape),
        "z_mean_logvar_shape": list(posterior["z_mean_logvar"].shape),
        "z_volatility_mu_shape": list(posterior["z_volatility_mu"].shape),
        "z_volatility_logvar_shape": list(posterior["z_volatility_logvar"].shape),
        "z_shared_mu_shape": list(posterior["z_shared_mu"].shape),
        "z_shared_logvar_shape": list(posterior["z_shared_logvar"].shape),
        
        # Latent sample shapes
        "z_mean_shape": list(latent_samples["z_mean"].shape),
        "z_volatility_shape": list(latent_samples["z_volatility"].shape),
        "z_shared_shape": list(latent_samples["z_shared"].shape),
        
        # Posterior finiteness
        "posterior_all_mu_finite": posterior_all_mu_finite,
        "posterior_all_logvar_finite": posterior_all_logvar_finite,
        
        # Latent finiteness
        "latent_samples_finite": latent_samples_finite,
        
        # KL values
        "kl_mean_batch_mean_value": kl_mean_val,
        "kl_volatility_batch_mean_value": kl_vol_val,
        "kl_shared_batch_mean_value": kl_shared_val,
        "kl_total_batch_mean_value": kl_total_val,
        "kl_values_finite": kl_values_finite,
        "kl_values_non_negative": kl_values_non_negative,
        
        # Decoder raw shapes
        "raw_kappa_shape": list(raw_tensors["raw_kappa"].shape),
        "raw_omega_shape": list(raw_tensors["raw_omega"].shape),
        "raw_total_mass_shape": list(raw_tensors["raw_total_mass"].shape),
        "raw_allocation_logits_shape": list(raw_tensors["raw_allocation_logits"].shape),
        
        # Decoder signature
        "decoder_signature_tensor_field_count": decoder_sig_field_count,
        "decoder_signature_all_tensor_fields_finite": decoder_sig_all_finite,
        
        # Reconstruction loss
        "reconstruction_loss_available": recon_loss_available,
        "reconstruction_loss_total_value": recon_loss_val,
        "reconstruction_loss_total_finite": recon_loss_finite,
        
        # Total loss
        "beta_value": float(P55_BETA),
        "beta_status": P55_BETA_STATUS,
        "total_loss_available": total_loss_available,
        "total_loss_value": total_loss_val,
        "total_loss_finite": total_loss_finite,
        
        # Boundary
        "no_training": True,
        "no_optimizer": True,
        "no_dataset": True,
        "no_dataloader": True,
        "no_generation_claim": True,
        "no_gsb_claim": True,
        "no_scientific_conclusion": True,
        "no_latent_learning_claim": True,
        "no_vae_success_claim": True,
        "realizability_claim": "encoder_posterior_kl_boundary_only_no_training",
        
        "reason": "fc_vae_encoder_posterior_kl_boundary_probe_success" if all_checks_pass else "fc_vae_encoder_posterior_kl_boundary_probe_failure",
        "verdict": "PASS" if all_checks_pass else "FAIL",
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


# ── Serialization ──

def fc_vae_encoder_posterior_kl_probe_to_json_dict(probe_res: dict) -> dict:
    """Convert probe result to a JSON-safe dict (no tensors, no arrays)."""
    validate_non_empty_str(probe_res["contract_version"], "contract_version")
    validate_non_empty_str(probe_res["status"], "status")
    validate_non_empty_str(probe_res["reason"], "reason")
    assert_no_local_path_leakage(probe_res["reason"], "reason")
    assert_no_forbidden_claims(probe_res["reason"], "reason")
    
    d = {}
    for k, v in probe_res.items():
        if v is None:
            d[k] = None
        elif isinstance(v, (bool, int, float, str)):
            d[k] = v
        elif isinstance(v, list):
            d[k] = v
        elif isinstance(v, tuple):
            d[k] = list(v)
        # skip any tensor values
    return d


def compact_fc_vae_encoder_posterior_kl_json(probe_res: dict) -> str:
    d = fc_vae_encoder_posterior_kl_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
