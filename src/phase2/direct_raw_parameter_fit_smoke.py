# src/phase2/direct_raw_parameter_fit_smoke.py
#
# P47: Direct raw parameter fit smoke.
# Objective-stack sanity check only.
# No neural model. No FC-VAE. No encoder. No decoder.
# No dataset. No dataloader. No torch optimizer.
# No scientific conclusion.

import json
from typing import Any, Dict

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECT_RAW_PARAMETER_FIT_CONTRACT_VERSION = (
    "phase2_p47_direct_raw_parameter_fit_smoke_contract_v1"
)

DIRECT_RAW_PARAMETER_FIT_KIND = (
    "direct_raw_parameter_fit_smoke_no_model_no_vae_no_science"
)

DIRECT_RAW_PARAMETER_FIT_MODULE_NAME = "src.phase2.direct_raw_parameter_fit_smoke"

P47_STEP_COUNT = 20

P47_LEARNING_RATE = 0.05

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

P47_STATUS_AVAILABLE = (
    "direct_raw_parameter_fit_smoke_available_no_model_no_vae_no_science"
)

P47_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"


# ---------------------------------------------------------------------------
# Torch loader
# ---------------------------------------------------------------------------

def load_torch_for_p47_direct_fit() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P47 loader.")
    import torch
    return torch


# ---------------------------------------------------------------------------
# Target raw tensors (no grad)
# ---------------------------------------------------------------------------

def build_p47_fixed_target_raw_tensors() -> Dict[str, Any]:
    """Fixed target raw tensors. Must not require gradients."""
    torch = load_torch_for_p47_direct_fit()
    raw_kappa = torch.tensor(
        [[0.15, -0.08, 0.03, -0.015, 0.007],
         [-0.12, 0.06, -0.025, 0.012, -0.006]],
        dtype=torch.float32,
        requires_grad=False,
    )
    raw_omega = torch.tensor(
        [[-0.3], [0.4]],
        dtype=torch.float32,
        requires_grad=False,
    )
    raw_total_mass = torch.tensor(
        [[0.8], [-0.7]],
        dtype=torch.float32,
        requires_grad=False,
    )
    raw_allocation_logits = torch.tensor(
        [[0.2, -0.2], [-0.1, 0.1]],
        dtype=torch.float32,
        requires_grad=False,
    )
    return {
        "raw_kappa": raw_kappa,
        "raw_omega": raw_omega,
        "raw_total_mass": raw_total_mass,
        "raw_allocation_logits": raw_allocation_logits,
    }


# ---------------------------------------------------------------------------
# Candidate raw tensors (require grad)
# ---------------------------------------------------------------------------

def build_p47_candidate_raw_tensors() -> Dict[str, Any]:
    """Candidate raw tensors initialized differently from target. Require gradients."""
    torch = load_torch_for_p47_direct_fit()
    raw_kappa = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    raw_omega = torch.tensor(
        [[0.0], [0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    raw_total_mass = torch.tensor(
        [[0.0], [0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    raw_allocation_logits = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    return {
        "raw_kappa": raw_kappa,
        "raw_omega": raw_omega,
        "raw_total_mass": raw_total_mass,
        "raw_allocation_logits": raw_allocation_logits,
    }


# ---------------------------------------------------------------------------
# Build combined signature from raw tensors
# ---------------------------------------------------------------------------

def build_combined_signature_from_raw(raw_tensors: Dict[str, Any]) -> dict:
    """Build combined AR+GARCH analytic signature from raw parameter tensors."""
    bias = default_garch_allocation_bias(1, 1, dtype=raw_tensors["raw_omega"].dtype)
    ar_res = pacf_to_stable_ar_coefficients(raw_tensors["raw_kappa"])
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


# ---------------------------------------------------------------------------
# Compute P47 loss
# ---------------------------------------------------------------------------

def compute_p47_loss(candidate_raw: Dict[str, Any], target_signature: dict) -> dict:
    """Compute combined moment/spectral matching loss between candidate and target."""
    candidate_signature = build_combined_signature_from_raw(candidate_raw)
    loss_res = combined_moment_spectral_matching_loss(candidate_signature, target_signature)
    return loss_res


# ---------------------------------------------------------------------------
# Run probe
# ---------------------------------------------------------------------------

def run_direct_raw_parameter_fit_probe() -> dict:
    """
    Objective-stack sanity check.
    Direct gradient descent on raw AR/GARCH tensors without any model or optimizer.
    """
    try:
        torch = load_torch_for_p47_direct_fit()
    except Exception:
        return {
            "contract_version": DIRECT_RAW_PARAMETER_FIT_CONTRACT_VERSION,
            "status": P47_STATUS_TORCH_UNAVAILABLE,
            "torch_available": False,
        }

    # Build fixed target tensors (no grad) and compute target signatures once
    target_raw = build_p47_fixed_target_raw_tensors()
    target_signature = build_combined_signature_from_raw(target_raw)

    # Detach target signature tensors so they are fixed comparison targets
    frozen_target = {
        k: v.detach() if torch.is_tensor(v) else v
        for k, v in target_signature.items()
    }

    # Build candidate tensors (with grad)
    candidate_raw = build_p47_candidate_raw_tensors()

    # Compute initial loss
    initial_loss_res = compute_p47_loss(candidate_raw, frozen_target)
    initial_loss_tensor = initial_loss_res["loss_total"]
    initial_loss_value = float(initial_loss_tensor.detach())

    # Direct gradient descent loop (no optimizer)
    for _step in range(P47_STEP_COUNT):
        loss_res = compute_p47_loss(candidate_raw, frozen_target)
        loss = loss_res["loss_total"]
        loss.backward()

        # Manual gradient descent update (no torch optimizer)
        with torch.no_grad():
            for param in [
                candidate_raw["raw_kappa"],
                candidate_raw["raw_omega"],
                candidate_raw["raw_total_mass"],
                candidate_raw["raw_allocation_logits"],
            ]:
                if param.grad is not None:
                    param -= P47_LEARNING_RATE * param.grad

        # Zero gradients manually
        for param in [
            candidate_raw["raw_kappa"],
            candidate_raw["raw_omega"],
            candidate_raw["raw_total_mass"],
            candidate_raw["raw_allocation_logits"],
        ]:
            if param.grad is not None:
                param.grad.zero_()

    # Compute final loss
    final_loss_res = compute_p47_loss(candidate_raw, frozen_target)
    final_loss_tensor = final_loss_res["loss_total"]
    final_loss_value = float(final_loss_tensor.detach())

    loss_delta_value = initial_loss_value - final_loss_value
    loss_decreased = bool(final_loss_value < initial_loss_value)
    loss_decrease_positive = bool(loss_delta_value > 0.0)

    return {
        "contract_version": DIRECT_RAW_PARAMETER_FIT_CONTRACT_VERSION,
        "status": P47_STATUS_AVAILABLE,
        "torch_available": True,
        "initial_loss_finite": bool(torch.isfinite(initial_loss_tensor.detach())),
        "final_loss_finite": bool(torch.isfinite(final_loss_tensor.detach())),
        "initial_loss_nonnegative": bool(initial_loss_value >= 0.0),
        "final_loss_nonnegative": bool(final_loss_value >= 0.0),
        "loss_decreased": loss_decreased,
        "loss_decrease_positive": loss_decrease_positive,
        "step_count": P47_STEP_COUNT,
        "initial_loss_value": initial_loss_value,
        "final_loss_value": final_loss_value,
        "loss_delta_value": loss_delta_value,
        "no_model": True,
        "no_vae": True,
        "no_encoder": True,
        "no_decoder": True,
        "no_dataset": True,
        "no_dataloader": True,
        "no_torch_optimizer": True,
        "no_scientific_conclusion": True,
        "reason": "p47_direct_raw_parameter_fit_probe_success",
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def direct_raw_parameter_fit_probe_to_json_dict(probe_res: dict) -> dict:
    """Convert probe result to a JSON-safe dictionary. No raw tensors."""
    return {
        "contract_version": probe_res["contract_version"],
        "status": probe_res["status"],
        "torch_available": bool(probe_res["torch_available"]),
        "initial_loss_finite": bool(probe_res.get("initial_loss_finite", False)),
        "final_loss_finite": bool(probe_res.get("final_loss_finite", False)),
        "initial_loss_nonnegative": bool(probe_res.get("initial_loss_nonnegative", False)),
        "final_loss_nonnegative": bool(probe_res.get("final_loss_nonnegative", False)),
        "loss_decreased": bool(probe_res.get("loss_decreased", False)),
        "loss_decrease_positive": bool(probe_res.get("loss_decrease_positive", False)),
        "step_count": int(probe_res.get("step_count", 0)),
        "initial_loss_value": float(probe_res.get("initial_loss_value", 0.0)),
        "final_loss_value": float(probe_res.get("final_loss_value", 0.0)),
        "loss_delta_value": float(probe_res.get("loss_delta_value", 0.0)),
        "no_model": bool(probe_res.get("no_model", False)),
        "no_vae": bool(probe_res.get("no_vae", False)),
        "no_encoder": bool(probe_res.get("no_encoder", False)),
        "no_decoder": bool(probe_res.get("no_decoder", False)),
        "no_dataset": bool(probe_res.get("no_dataset", False)),
        "no_dataloader": bool(probe_res.get("no_dataloader", False)),
        "no_torch_optimizer": bool(probe_res.get("no_torch_optimizer", False)),
        "no_scientific_conclusion": bool(probe_res.get("no_scientific_conclusion", False)),
        "reason": probe_res.get("reason", ""),
    }


def compact_direct_raw_parameter_fit_json(probe_res: dict) -> str:
    """Return compact sorted-keys JSON string for probe results."""
    d = direct_raw_parameter_fit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
