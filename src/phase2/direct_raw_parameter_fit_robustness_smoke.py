# src/phase2/direct_raw_parameter_fit_robustness_smoke.py
#
# P48: Deterministic direct raw parameter fit robustness smoke.
# Multi-scenario objective-stack sanity check only.
# No neural model. No FC-VAE. No encoder. No decoder.
# No dataset. No dataloader. No torch optimizer.
# No scientific conclusion.

import json
from typing import Any, Dict, List, Tuple

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

DIRECT_FIT_ROBUSTNESS_CONTRACT_VERSION = (
    "phase2_p48_deterministic_direct_fit_robustness_smoke_contract_v1"
)

DIRECT_FIT_ROBUSTNESS_KIND = (
    "deterministic_direct_fit_robustness_smoke_no_model_no_vae_no_science"
)

DIRECT_FIT_ROBUSTNESS_MODULE_NAME = (
    "src.phase2.direct_raw_parameter_fit_robustness_smoke"
)

P48_DEFAULT_STEP_COUNT = 20

P48_DEFAULT_LEARNING_RATE = 0.05

P48_SCENARIO_COUNT = 4

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

P48_STATUS_AVAILABLE = (
    "direct_fit_robustness_smoke_available_no_model_no_vae_no_science"
)

P48_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"


# ---------------------------------------------------------------------------
# Torch loader
# ---------------------------------------------------------------------------

def load_torch_for_p48_direct_fit_robustness() -> Any:
    """Load torch through guarded P26-style local loader."""
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError(
            "PyTorch is not available or safe to import in P48 loader."
        )
    import torch
    return torch


# ---------------------------------------------------------------------------
# Core signature helper
# ---------------------------------------------------------------------------

def build_combined_signature_from_raw(raw_tensors: Dict[str, Any]) -> dict:
    """Build combined AR+GARCH analytic signature from raw parameter tensors."""
    bias = default_garch_allocation_bias(
        1, 1, dtype=raw_tensors["raw_omega"].dtype
    )
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


def freeze_signature(signature: dict) -> dict:
    """Detach tensor values only. Keep non-tensor metadata unchanged."""
    import torch
    return {
        k: v.detach() if torch.is_tensor(v) else v
        for k, v in signature.items()
    }


def compute_combined_loss(
    candidate_raw: Dict[str, Any],
    frozen_target_signature: dict,
) -> dict:
    """Compute combined moment/spectral matching loss."""
    candidate_signature = build_combined_signature_from_raw(candidate_raw)
    return combined_moment_spectral_matching_loss(
        candidate_signature, frozen_target_signature
    )


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def build_p48_scenarios() -> Tuple[dict, ...]:
    """
    Return exactly 4 deterministic fixed scenarios.
    Each scenario has varied but stable AR/GARCH target/candidate configurations.
    """
    torch = load_torch_for_p48_direct_fit_robustness()

    # Scenario 1: mild AR/GARCH mismatch
    s1_target = {
        "raw_kappa": torch.tensor(
            [[0.15, -0.08, 0.03, -0.015, 0.007],
             [-0.12, 0.06, -0.025, 0.012, -0.006]],
            dtype=torch.float32, requires_grad=False,
        ),
        "raw_omega": torch.tensor(
            [[-0.3], [0.4]], dtype=torch.float32, requires_grad=False,
        ),
        "raw_total_mass": torch.tensor(
            [[0.8], [-0.7]], dtype=torch.float32, requires_grad=False,
        ),
        "raw_allocation_logits": torch.tensor(
            [[0.2, -0.2], [-0.1, 0.1]], dtype=torch.float32, requires_grad=False,
        ),
    }
    s1_candidate = {
        "raw_kappa": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32, requires_grad=True,
        ),
        "raw_omega": torch.tensor(
            [[0.0], [0.0]], dtype=torch.float32, requires_grad=True,
        ),
        "raw_total_mass": torch.tensor(
            [[0.0], [0.0]], dtype=torch.float32, requires_grad=True,
        ),
        "raw_allocation_logits": torch.tensor(
            [[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32, requires_grad=True,
        ),
    }

    # Scenario 2: sign-flipped AR mismatch
    s2_target = {
        "raw_kappa": torch.tensor(
            [[0.30, -0.15, 0.08, -0.04, 0.02],
             [0.25, -0.10, 0.05, -0.02, 0.01]],
            dtype=torch.float32, requires_grad=False,
        ),
        "raw_omega": torch.tensor(
            [[-0.5], [0.2]], dtype=torch.float32, requires_grad=False,
        ),
        "raw_total_mass": torch.tensor(
            [[0.6], [-0.4]], dtype=torch.float32, requires_grad=False,
        ),
        "raw_allocation_logits": torch.tensor(
            [[0.3, -0.3], [0.1, -0.1]], dtype=torch.float32, requires_grad=False,
        ),
    }
    s2_candidate = {
        "raw_kappa": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32, requires_grad=True,
        ),
        "raw_omega": torch.tensor(
            [[0.0], [0.0]], dtype=torch.float32, requires_grad=True,
        ),
        "raw_total_mass": torch.tensor(
            [[0.0], [0.0]], dtype=torch.float32, requires_grad=True,
        ),
        "raw_allocation_logits": torch.tensor(
            [[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32, requires_grad=True,
        ),
    }

    # Scenario 3: stronger volatility persistence mismatch
    s3_target = {
        "raw_kappa": torch.tensor(
            [[0.05, -0.03, 0.01, -0.005, 0.002],
             [0.04, -0.02, 0.008, -0.004, 0.002]],
            dtype=torch.float32, requires_grad=False,
        ),
        "raw_omega": torch.tensor(
            [[-0.8], [0.9]], dtype=torch.float32, requires_grad=False,
        ),
        "raw_total_mass": torch.tensor(
            [[1.2], [-1.1]], dtype=torch.float32, requires_grad=False,
        ),
        "raw_allocation_logits": torch.tensor(
            [[0.5, -0.5], [-0.4, 0.4]], dtype=torch.float32, requires_grad=False,
        ),
    }
    s3_candidate = {
        "raw_kappa": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32, requires_grad=True,
        ),
        "raw_omega": torch.tensor(
            [[0.0], [0.0]], dtype=torch.float32, requires_grad=True,
        ),
        "raw_total_mass": torch.tensor(
            [[0.0], [0.0]], dtype=torch.float32, requires_grad=True,
        ),
        "raw_allocation_logits": torch.tensor(
            [[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32, requires_grad=True,
        ),
    }

    # Scenario 4: mixed AR + GARCH allocation mismatch
    s4_target = {
        "raw_kappa": torch.tensor(
            [[0.20, -0.10, 0.05, -0.025, 0.012],
             [-0.18, 0.09, -0.045, 0.022, -0.011]],
            dtype=torch.float32, requires_grad=False,
        ),
        "raw_omega": torch.tensor(
            [[-0.4], [0.6]], dtype=torch.float32, requires_grad=False,
        ),
        "raw_total_mass": torch.tensor(
            [[0.9], [-0.8]], dtype=torch.float32, requires_grad=False,
        ),
        "raw_allocation_logits": torch.tensor(
            [[-0.3, 0.3], [0.2, -0.2]], dtype=torch.float32, requires_grad=False,
        ),
    }
    s4_candidate = {
        "raw_kappa": torch.tensor(
            [[0.05, -0.03, 0.01, -0.005, 0.002],
             [-0.04, 0.02, -0.01, 0.005, -0.002]],
            dtype=torch.float32, requires_grad=True,
        ),
        "raw_omega": torch.tensor(
            [[0.1], [0.1]], dtype=torch.float32, requires_grad=True,
        ),
        "raw_total_mass": torch.tensor(
            [[0.2], [0.2]], dtype=torch.float32, requires_grad=True,
        ),
        "raw_allocation_logits": torch.tensor(
            [[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32, requires_grad=True,
        ),
    }

    return (
        {
            "scenario_id": "s1_mild_mismatch",
            "target_raw": s1_target,
            "candidate_raw": s1_candidate,
            "step_count": P48_DEFAULT_STEP_COUNT,
            "learning_rate": P48_DEFAULT_LEARNING_RATE,
        },
        {
            "scenario_id": "s2_sign_flipped_ar",
            "target_raw": s2_target,
            "candidate_raw": s2_candidate,
            "step_count": P48_DEFAULT_STEP_COUNT,
            "learning_rate": 0.01,
        },
        {
            "scenario_id": "s3_strong_volatility_mismatch",
            "target_raw": s3_target,
            "candidate_raw": s3_candidate,
            "step_count": P48_DEFAULT_STEP_COUNT,
            "learning_rate": P48_DEFAULT_LEARNING_RATE,
        },
        {
            "scenario_id": "s4_mixed_ar_garch_allocation",
            "target_raw": s4_target,
            "candidate_raw": s4_candidate,
            "step_count": P48_DEFAULT_STEP_COUNT,
            "learning_rate": P48_DEFAULT_LEARNING_RATE,
        },
    )


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

def run_single_p48_scenario(scenario: dict) -> dict:
    """
    Run one P48 scenario: build frozen target sig, run manual gradient steps, return summary.
    No optimizer. No model. No VAE. No dataset.
    """
    torch = load_torch_for_p48_direct_fit_robustness()

    scenario_id = scenario["scenario_id"]
    target_raw = scenario["target_raw"]
    candidate_raw = scenario["candidate_raw"]
    step_count = scenario["step_count"]
    learning_rate = scenario["learning_rate"]

    # Build frozen target signature
    target_sig = build_combined_signature_from_raw(target_raw)
    frozen_target = freeze_signature(target_sig)

    # Compute initial loss
    initial_loss_res = compute_combined_loss(candidate_raw, frozen_target)
    initial_loss_tensor = initial_loss_res["loss_total"]
    initial_loss_value = float(initial_loss_tensor.detach())

    # Manual gradient descent loop (no optimizer)
    param_keys = ["raw_kappa", "raw_omega", "raw_total_mass", "raw_allocation_logits"]
    for _step in range(step_count):
        loss_res = compute_combined_loss(candidate_raw, frozen_target)
        loss = loss_res["loss_total"]
        loss.backward()

        with torch.no_grad():
            for k in param_keys:
                param = candidate_raw[k]
                if param.grad is not None:
                    param -= learning_rate * param.grad

        for k in param_keys:
            param = candidate_raw[k]
            if param.grad is not None:
                param.grad.zero_()

    # Compute final loss
    final_loss_res = compute_combined_loss(candidate_raw, frozen_target)
    final_loss_tensor = final_loss_res["loss_total"]
    final_loss_value = float(final_loss_tensor.detach())

    loss_delta_value = initial_loss_value - final_loss_value
    loss_decreased = bool(final_loss_value < initial_loss_value)

    return {
        "scenario_id": scenario_id,
        "initial_loss_value": initial_loss_value,
        "final_loss_value": final_loss_value,
        "loss_delta_value": loss_delta_value,
        "loss_decreased": loss_decreased,
        "initial_loss_finite": bool(torch.isfinite(initial_loss_tensor.detach())),
        "final_loss_finite": bool(torch.isfinite(final_loss_tensor.detach())),
        "step_count": step_count,
        "learning_rate": learning_rate,
    }


# ---------------------------------------------------------------------------
# Aggregate probe
# ---------------------------------------------------------------------------

def run_direct_fit_robustness_probe() -> dict:
    """
    Run all 4 deterministic scenarios and return aggregate robustness summary.
    No model. No VAE. No dataset. No optimizer. No scientific conclusion.
    """
    try:
        torch = load_torch_for_p48_direct_fit_robustness()
    except Exception:
        return {
            "contract_version": DIRECT_FIT_ROBUSTNESS_CONTRACT_VERSION,
            "kind": DIRECT_FIT_ROBUSTNESS_KIND,
            "status": P48_STATUS_TORCH_UNAVAILABLE,
            "torch_available": False,
        }

    scenarios = build_p48_scenarios()
    scenario_results: List[dict] = []

    for scenario in scenarios:
        result = run_single_p48_scenario(scenario)
        scenario_results.append(result)

    scenarios_passed = sum(1 for r in scenario_results if r["loss_decreased"])
    all_scenarios_loss_decreased = all(r["loss_decreased"] for r in scenario_results)
    all_losses_finite = all(
        r["initial_loss_finite"] and r["final_loss_finite"]
        for r in scenario_results
    )
    deltas = [r["loss_delta_value"] for r in scenario_results]
    min_loss_delta_value = float(min(deltas))
    max_loss_delta_value = float(max(deltas))
    mean_loss_delta_value = float(sum(deltas) / len(deltas))

    return {
        "contract_version": DIRECT_FIT_ROBUSTNESS_CONTRACT_VERSION,
        "kind": DIRECT_FIT_ROBUSTNESS_KIND,
        "status": P48_STATUS_AVAILABLE,
        "torch_available": True,
        "scenario_count": P48_SCENARIO_COUNT,
        "scenarios_passed": scenarios_passed,
        "all_scenarios_loss_decreased": all_scenarios_loss_decreased,
        "all_losses_finite": all_losses_finite,
        "min_loss_delta_value": min_loss_delta_value,
        "max_loss_delta_value": max_loss_delta_value,
        "mean_loss_delta_value": mean_loss_delta_value,
        "scenario_results": scenario_results,
        "no_model": True,
        "no_vae": True,
        "no_encoder": True,
        "no_decoder": True,
        "no_dataset": True,
        "no_dataloader": True,
        "no_torch_optimizer": True,
        "no_scientific_conclusion": True,
        "reason": "p48_direct_fit_robustness_probe_success",
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def direct_fit_robustness_probe_to_json_dict(probe_res: dict) -> dict:
    """Convert probe result to JSON-safe dictionary. No raw tensors."""
    scenario_results = probe_res.get("scenario_results", [])
    safe_scenarios = []
    for r in scenario_results:
        safe_scenarios.append({
            "scenario_id": str(r.get("scenario_id", "")),
            "initial_loss_value": float(r.get("initial_loss_value", 0.0)),
            "final_loss_value": float(r.get("final_loss_value", 0.0)),
            "loss_delta_value": float(r.get("loss_delta_value", 0.0)),
            "loss_decreased": bool(r.get("loss_decreased", False)),
            "initial_loss_finite": bool(r.get("initial_loss_finite", False)),
            "final_loss_finite": bool(r.get("final_loss_finite", False)),
            "step_count": int(r.get("step_count", 0)),
            "learning_rate": float(r.get("learning_rate", 0.0)),
        })

    return {
        "contract_version": probe_res.get("contract_version", ""),
        "kind": probe_res.get("kind", ""),
        "status": probe_res.get("status", ""),
        "torch_available": bool(probe_res.get("torch_available", False)),
        "scenario_count": int(probe_res.get("scenario_count", 0)),
        "scenarios_passed": int(probe_res.get("scenarios_passed", 0)),
        "all_scenarios_loss_decreased": bool(
            probe_res.get("all_scenarios_loss_decreased", False)
        ),
        "all_losses_finite": bool(probe_res.get("all_losses_finite", False)),
        "min_loss_delta_value": float(probe_res.get("min_loss_delta_value", 0.0)),
        "max_loss_delta_value": float(probe_res.get("max_loss_delta_value", 0.0)),
        "mean_loss_delta_value": float(probe_res.get("mean_loss_delta_value", 0.0)),
        "scenario_results": safe_scenarios,
        "no_model": bool(probe_res.get("no_model", False)),
        "no_vae": bool(probe_res.get("no_vae", False)),
        "no_encoder": bool(probe_res.get("no_encoder", False)),
        "no_decoder": bool(probe_res.get("no_decoder", False)),
        "no_dataset": bool(probe_res.get("no_dataset", False)),
        "no_dataloader": bool(probe_res.get("no_dataloader", False)),
        "no_torch_optimizer": bool(probe_res.get("no_torch_optimizer", False)),
        "no_scientific_conclusion": bool(
            probe_res.get("no_scientific_conclusion", False)
        ),
        "reason": probe_res.get("reason", ""),
    }


def compact_direct_fit_robustness_json(probe_res: dict) -> str:
    """Return compact sorted-keys JSON string for probe results."""
    d = direct_fit_robustness_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
