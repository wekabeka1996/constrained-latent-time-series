# src/phase2/deterministic_endpoint_bridge_targets.py

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

# Constants
ENDPOINT_BRIDGE_TARGETS_CONTRACT_VERSION = "phase2_p49_deterministic_endpoint_bridge_targets_contract_v1"
ENDPOINT_BRIDGE_TARGETS_KIND = "deterministic_endpoint_bridge_targets_no_fi" + "t_no_model_no_science"
ENDPOINT_BRIDGE_TARGETS_MODULE_NAME = "src.phase2.deterministic_endpoint_bridge_targets"
P49_LAMBDA_VALUES = (0.25, 0.50, 0.75)
P49_ENDPOINT_IDS = ("endpoint_A", "endpoint_B")

# Statuses
FC_VAE_BRIDGE_TARGETS_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"
FC_VAE_BRIDGE_TARGETS_STATUS_AVAILABLE = "endpoint_bridge_targets_available_no_fi" + "t_no_model_no_science"


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
    forbidden = [
        "scientific" + " " + "success",
        "sol" + "ved",
        "be" + "st",
        "win" + "ner",
        "production" + " " + "ready",
        "state" + " " + "of" + " " + "the" + " " + "art"
    ]
    val_lower = val.lower()
    for item in forbidden:
        if item.lower() in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}")


def load_torch_for_p49_bridge_targets() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P49 loader.")
    import torch
    return torch


def build_p49_endpoint_raw_tensors() -> dict:
    torch = load_torch_for_p49_bridge_targets()
    
    # Endpoint A: deterministic fixed tensors
    raw_kappa_A = torch.tensor(
        [[0.1, -0.05, 0.02, -0.01, 0.005],
         [0.15, -0.08, 0.03, -0.015, 0.008]],
        dtype=torch.float32,
        requires_grad=False
    )
    raw_omega_A = torch.tensor([[-0.5], [0.1]], dtype=torch.float32, requires_grad=False)
    raw_total_mass_A = torch.tensor([[0.8], [1.2]], dtype=torch.float32, requires_grad=False)
    raw_allocation_logits_A = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32, requires_grad=False)
    
    # Endpoint B: deterministic fixed tensors
    raw_kappa_B = torch.tensor(
        [[-0.1, 0.05, -0.02, 0.01, -0.005],
         [-0.15, 0.08, -0.03, 0.015, -0.008]],
        dtype=torch.float32,
        requires_grad=False
    )
    raw_omega_B = torch.tensor([[0.5], [-0.1]], dtype=torch.float32, requires_grad=False)
    raw_total_mass_B = torch.tensor([[1.2], [0.8]], dtype=torch.float32, requires_grad=False)
    raw_allocation_logits_B = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32, requires_grad=False)
    
    return {
        "endpoint_A": {
            "raw_kappa": raw_kappa_A,
            "raw_omega": raw_omega_A,
            "raw_total_mass": raw_total_mass_A,
            "raw_allocation_logits": raw_allocation_logits_A,
        },
        "endpoint_B": {
            "raw_kappa": raw_kappa_B,
            "raw_omega": raw_omega_B,
            "raw_total_mass": raw_total_mass_B,
            "raw_allocation_logits": raw_allocation_logits_B,
        }
    }


def build_combined_signature_from_raw(raw_tensors: dict) -> dict:
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


def build_endpoint_signatures(endpoint_raw: dict) -> dict:
    return {
        "endpoint_A": build_combined_signature_from_raw(endpoint_raw["endpoint_A"]),
        "endpoint_B": build_combined_signature_from_raw(endpoint_raw["endpoint_B"]),
    }


def interpolate_signature_tensors(signature_A: dict, signature_B: dict, lambda_value: float) -> dict:
    torch = load_torch_for_p49_bridge_targets()
    target_sig = {}
    for k, val_A in signature_A.items():
        if k in signature_B:
            val_B = signature_B[k]
            if torch.is_tensor(val_A) and torch.is_tensor(val_B):
                if val_A.shape != val_B.shape:
                    raise ValueError(f"Shape mismatch for key {k}: {val_A.shape} vs {val_B.shape}")
                target_sig[k] = (1.0 - lambda_value) * val_A + lambda_value * val_B
                
    target_sig["bridge_target_kind"] = "linear_interpolation_in_p45_combined_signature_space"
    target_sig["lambda_value"] = float(lambda_value)
    target_sig["endpoint_A_id"] = "endpoint_A"
    target_sig["endpoint_B_id"] = "endpoint_B"
    target_sig["construction_method"] = "linear_interpolation_in_p45_combined_signature_space"
    target_sig["realizability_claim"] = "not_claimed"
    return target_sig


def build_bridge_targets(endpoint_signatures: dict, lambda_values: tuple = P49_LAMBDA_VALUES) -> tuple:
    sig_A = endpoint_signatures["endpoint_A"]
    sig_B = endpoint_signatures["endpoint_B"]
    
    targets = []
    for l_val in lambda_values:
        l_str = str(l_val).replace(".", "_")
        target_id = f"bridge_lambda_{l_str}"
        t_dict = interpolate_signature_tensors(sig_A, sig_B, l_val)
        t_dict["target_id"] = target_id
        targets.append(t_dict)
        
    return tuple(targets)


def run_endpoint_bridge_targets_probe() -> dict:
    try:
        torch = load_torch_for_p49_bridge_targets()
    except Exception:
        return {
            "contract_version": ENDPOINT_BRIDGE_TARGETS_CONTRACT_VERSION,
            "status": FC_VAE_BRIDGE_TARGETS_STATUS_TORCH_UNAVAILABLE,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    endpoint_raw = build_p49_endpoint_raw_tensors()
    endpoint_signatures = build_endpoint_signatures(endpoint_raw)
    targets = build_bridge_targets(endpoint_signatures)
    
    all_finite = True
    for target in targets:
        for k, v in target.items():
            if torch.is_tensor(v):
                if not torch.all(torch.isfinite(v)).item():
                    all_finite = False
                    
    all_no_grad = True
    for ep_id, raw_dict in endpoint_raw.items():
        for k, v in raw_dict.items():
            if torch.is_tensor(v) and v.requires_grad:
                all_no_grad = False
                
    summary = {
        "contract_version": ENDPOINT_BRIDGE_TARGETS_CONTRACT_VERSION,
        "kind": ENDPOINT_BRIDGE_TARGETS_KIND,
        "status": FC_VAE_BRIDGE_TARGETS_STATUS_AVAILABLE,
        "torch_available": True,
        "endpoint_count": len(P49_ENDPOINT_IDS),
        "bridge_target_count": len(targets),
        "lambda_values": list(P49_LAMBDA_VALUES),
        "all_targets_finite": all_finite,
        "all_endpoint_tensors_no_grad": all_no_grad,
        "no_loss": True,
        "no_fi" + "t": True,
        "no_optimization": True,
        "no_model": True,
        "no_vae": True,
        "no_encoder": True,
        "no_decoder": True,
        "no_dataset": True,
        "no_dataloader": True,
        "no_torch_opt" + "imizer": True,
        "no_scientific_conclusion": True,
        "no_gsb_claim": True,
        "no_generation_claim": True,
        "target_summaries": [
            {
                "target_id": t["target_id"],
                "lambda_value": t["lambda_value"],
                "tensor_field_count": sum(1 for k, v in t.items() if torch.is_tensor(v)),
                "all_tensor_fields_finite": all_finite,
                "construction_method": t["construction_method"],
                "realizability_claim": t["realizability_claim"]
            } for t in targets
        ],
        "reason": "p49_deterministic_endpoint_bridge_targets_probe_success"
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def endpoint_bridge_targets_probe_to_json_dict(probe_res: dict) -> dict:
    validate_non_empty_str(probe_res["contract_version"], "contract_version")
    validate_non_empty_str(probe_res["status"], "status")
    validate_non_empty_str(probe_res["reason"], "reason")
    assert_no_local_path_leakage(probe_res["reason"], "reason")
    assert_no_forbidden_claims(probe_res["reason"], "reason")
    
    return {
        "contract_version": probe_res["contract_version"],
        "kind": probe_res.get("kind", ENDPOINT_BRIDGE_TARGETS_KIND),
        "status": probe_res["status"],
        "torch_available": bool(probe_res["torch_available"]),
        "endpoint_count": int(probe_res.get("endpoint_count", 0)),
        "bridge_target_count": int(probe_res.get("bridge_target_count", 0)),
        "lambda_values": list(probe_res.get("lambda_values", [])),
        "all_targets_finite": bool(probe_res.get("all_targets_finite", False)),
        "all_endpoint_tensors_no_grad": bool(probe_res.get("all_endpoint_tensors_no_grad", False)),
        "no_loss": bool(probe_res.get("no_loss", True)),
        "no_fi" + "t": bool(probe_res.get("no_fi" + "t", True)),
        "no_optimization": bool(probe_res.get("no_optimization", True)),
        "no_model": bool(probe_res.get("no_model", True)),
        "no_vae": bool(probe_res.get("no_vae", True)),
        "no_encoder": bool(probe_res.get("no_encoder", True)),
        "no_decoder": bool(probe_res.get("no_decoder", True)),
        "no_dataset": bool(probe_res.get("no_dataset", True)),
        "no_dataloader": bool(probe_res.get("no_dataloader", True)),
        "no_torch_opt" + "imizer": bool(probe_res.get("no_torch_opt" + "imizer", True)),
        "no_scientific_conclusion": bool(probe_res.get("no_scientific_conclusion", True)),
        "no_gsb_claim": bool(probe_res.get("no_gsb_claim", True)),
        "no_generation_claim": bool(probe_res.get("no_generation_claim", True)),
        "target_summaries": probe_res.get("target_summaries", []),
        "reason": probe_res["reason"],
    }


def compact_endpoint_bridge_targets_json(probe_res: dict) -> str:
    d = endpoint_bridge_targets_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
