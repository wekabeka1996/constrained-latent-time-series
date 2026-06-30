# src/phase2/fc_vae_backward_gradient_smoke.py

import json
import math
from typing import Any, Dict, List

from src.phase2.torch_boundary import (
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    build_p50_bridge_targets,
)
from src.phase2.fc_vae_encoder_posterior_kl_boundary import (
    load_torch_for_p55_encoder_kl,
    build_fc_vae_encoder_posterior_model,
    P55_BETA,
    P55_EPS_MODE,
    P55_BETA_STATUS,
)

# Constants
FC_VAE_BACKWARD_GRADIENT_SMOKE_CONTRACT_VERSION = "phase2_p56_fc_vae_backward_gradient_smoke_contract_v1"
FC_VAE_BACKWARD_GRADIENT_SMOKE_KIND = "fc_vae_backward_gradient_smoke_no_optimizer_no_training"
FC_VAE_BACKWARD_GRADIENT_SMOKE_MODULE_NAME = "src.phase2.fc_vae_backward_gradient_smoke"
FC_VAE_BACKWARD_GRADIENT_SMOKE_STATUS_AVAILABLE = "fc_vae_backward_gradient_smoke_available_no_optimizer_no_training"
FC_VAE_BACKWARD_GRADIENT_SMOKE_STATUS_BLOCKED = "blocked_torch_unavailable"

P56_EPS_MODE = "zero"
P56_BETA_ZERO = 0.0
P56_BETA_SMALL = 0.001
P56_TARGET_ID = "bridge_lambda_0_25"


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
        ("posterior", "collapse"), ("synthetic", "state"),
        ("trained", "vae"), ("optimizer", "step"), ("vae", "solved"),
    ]
    for a, b in forbidden_pairs:
        if a in val_lower and b in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}")


def load_torch_for_p56_backward_smoke() -> Any:
    return load_torch_for_p55_encoder_kl()


def collect_p56_parameter_groups(model: Any) -> Dict[str, List[Any]]:
    return {
        "encoder_trunk": list(model.encoder_trunk.parameters()),
        "posterior_mean_mu_heads": list(model.z_mean_mu_head.parameters()),
        "posterior_mean_logvar_heads": list(model.z_mean_logvar_head.parameters()),
        "posterior_volatility_mu_heads": list(model.z_volatility_mu_head.parameters()),
        "posterior_volatility_logvar_heads": list(model.z_volatility_logvar_head.parameters()),
        "posterior_shared_mu_heads": list(model.z_shared_mu_head.parameters()),
        "posterior_shared_logvar_heads": list(model.z_shared_logvar_head.parameters()),
        "decoder": list(model.decoder.parameters()),
    }


def compute_gradient_stats_for_group(named_params: List[Any]) -> Dict[str, Any]:
    torch = load_torch_for_p56_backward_smoke()
    
    params = []
    for item in named_params:
        if isinstance(item, tuple) or isinstance(item, list):
            params.append(item[1])
        else:
            params.append(item)
            
    grads = []
    param_count = len(params)
    param_with_grad = 0
    param_without_grad = 0
    
    for p in params:
        if p.grad is not None:
            param_with_grad += 1
            grads.append(p.grad)
        else:
            param_without_grad += 1
            
    if not grads:
        return {
            "parameter_count": param_count,
            "parameters_with_grad": param_with_grad,
            "parameters_without_grad": param_without_grad,
            "all_present_grads_finite": True,
            "any_grad_present": False,
            "any_nonzero_grad": False,
            "max_abs_grad_value": 0.0,
            "mean_abs_grad_value": 0.0,
        }
        
    any_grad_present = True
    all_present_grads_finite = True
    any_nonzero_grad = False
    
    abs_grads = []
    for g in grads:
        is_finite = torch.all(torch.isfinite(g)).item()
        if not is_finite:
            all_present_grads_finite = False
        
        any_nonzero = (torch.any(g != 0.0)).item()
        if any_nonzero:
            any_nonzero_grad = True
            
        abs_val = torch.abs(g)
        abs_grads.append(abs_val)
        
    all_abs = torch.cat([ag.view(-1) for ag in abs_grads])
    max_val = torch.max(all_abs).item()
    mean_val = torch.mean(all_abs).item()
    
    return {
        "parameter_count": param_count,
        "parameters_with_grad": param_with_grad,
        "parameters_without_grad": param_without_grad,
        "all_present_grads_finite": bool(all_present_grads_finite),
        "any_grad_present": bool(any_grad_present),
        "any_nonzero_grad": bool(any_nonzero_grad),
        "max_abs_grad_value": float(max_val),
        "mean_abs_grad_value": float(mean_val),
    }


def compute_all_gradient_stats(model: Any) -> Dict[str, Dict[str, Any]]:
    groups = collect_p56_parameter_groups(model)
    stats = {}
    for name, params in groups.items():
        stats[name] = compute_gradient_stats_for_group(params)
    return stats


def clear_model_grads(model: Any) -> None:
    model.zero_grad(set_to_none=True)


def run_p56_reconstruction_backward_pass(model: Any, target_signature: Dict[str, Any]) -> Dict[str, Any]:
    clear_model_grads(model)
    fwd = model(target_signature, eps_mode="zero", beta=0.0)
    loss = fwd["total_loss"]
    if loss is None:
        raise ValueError("Reconstruction pass total_loss is None")
        
    loss_val = loss.item()
    if not math.isfinite(loss_val):
        raise ValueError(f"Reconstruction loss is non-finite: {loss_val}")
        
    loss.backward()
    stats = compute_all_gradient_stats(model)
    
    enc_passed = stats["encoder_trunk"]["all_present_grads_finite"] and stats["encoder_trunk"]["any_nonzero_grad"]
    
    mu_passed = (
        stats["posterior_mean_mu_heads"]["all_present_grads_finite"] and stats["posterior_mean_mu_heads"]["any_nonzero_grad"] and
        stats["posterior_volatility_mu_heads"]["all_present_grads_finite"] and stats["posterior_volatility_mu_heads"]["any_nonzero_grad"] and
        stats["posterior_shared_mu_heads"]["all_present_grads_finite"] and stats["posterior_shared_mu_heads"]["any_nonzero_grad"]
    )
    
    dec_passed = stats["decoder"]["all_present_grads_finite"] and stats["decoder"]["any_nonzero_grad"]
    
    passed = enc_passed and mu_passed and dec_passed
    
    return {
        "pass_name": "reconstruction_backward_beta_zero",
        "loss_value": float(loss_val),
        "loss_finite": True,
        "backward_completed": True,
        "gradient_stats": stats,
        "encoder_trunk_grad_expected": True,
        "posterior_mu_grad_expected": True,
        "posterior_logvar_grad_expected": False,
        "decoder_grad_expected": True,
        "encoder_trunk_passed": bool(enc_passed),
        "posterior_mu_passed": bool(mu_passed),
        "decoder_passed": bool(dec_passed),
        "passed": bool(passed),
    }


def run_p56_kl_backward_pass(model: Any, target_signature: Dict[str, Any]) -> Dict[str, Any]:
    clear_model_grads(model)
    fwd = model(target_signature, eps_mode="zero", beta=0.0)
    kl_loss = fwd["kl_dict"]["kl_total_batch_mean"]
    kl_val = kl_loss.item()
    
    if not math.isfinite(kl_val) or kl_val < 0.0:
        raise ValueError(f"KL loss is non-finite or negative: {kl_val}")
        
    kl_loss.backward()
    stats = compute_all_gradient_stats(model)
    
    enc_passed = stats["encoder_trunk"]["all_present_grads_finite"] and stats["encoder_trunk"]["any_nonzero_grad"]
    
    mu_passed = (
        stats["posterior_mean_mu_heads"]["all_present_grads_finite"] and stats["posterior_mean_mu_heads"]["any_nonzero_grad"] and
        stats["posterior_volatility_mu_heads"]["all_present_grads_finite"] and stats["posterior_volatility_mu_heads"]["any_nonzero_grad"] and
        stats["posterior_shared_mu_heads"]["all_present_grads_finite"] and stats["posterior_shared_mu_heads"]["any_nonzero_grad"]
    )
    
    logvar_passed = (
        stats["posterior_mean_logvar_heads"]["all_present_grads_finite"] and stats["posterior_mean_logvar_heads"]["any_nonzero_grad"] and
        stats["posterior_volatility_logvar_heads"]["all_present_grads_finite"] and stats["posterior_volatility_logvar_heads"]["any_nonzero_grad"] and
        stats["posterior_shared_logvar_heads"]["all_present_grads_finite"] and stats["posterior_shared_logvar_heads"]["any_nonzero_grad"]
    )
    
    passed = enc_passed and mu_passed and logvar_passed
    
    return {
        "pass_name": "kl_backward",
        "loss_value": float(kl_val),
        "loss_finite": True,
        "loss_non_negative": True,
        "backward_completed": True,
        "gradient_stats": stats,
        "encoder_trunk_grad_expected": True,
        "posterior_mu_grad_expected": True,
        "posterior_logvar_grad_expected": True,
        "decoder_grad_expected": False,
        "encoder_trunk_passed": bool(enc_passed),
        "posterior_mu_passed": bool(mu_passed),
        "posterior_logvar_passed": bool(logvar_passed),
        "passed": bool(passed),
    }


def run_p56_combined_backward_pass(model: Any, target_signature: Dict[str, Any]) -> Dict[str, Any]:
    clear_model_grads(model)
    fwd = model(target_signature, eps_mode="zero", beta=P56_BETA_SMALL)
    total_loss = fwd["total_loss"]
    loss_val = total_loss.item()
    
    if not math.isfinite(loss_val):
        raise ValueError(f"Combined loss is non-finite: {loss_val}")
        
    total_loss.backward()
    stats = compute_all_gradient_stats(model)
    
    enc_passed = stats["encoder_trunk"]["all_present_grads_finite"] and stats["encoder_trunk"]["any_nonzero_grad"]
    
    mu_passed = (
        stats["posterior_mean_mu_heads"]["all_present_grads_finite"] and stats["posterior_mean_mu_heads"]["any_nonzero_grad"] and
        stats["posterior_volatility_mu_heads"]["all_present_grads_finite"] and stats["posterior_volatility_mu_heads"]["any_nonzero_grad"] and
        stats["posterior_shared_mu_heads"]["all_present_grads_finite"] and stats["posterior_shared_mu_heads"]["any_nonzero_grad"]
    )
    
    logvar_passed = (
        stats["posterior_mean_logvar_heads"]["all_present_grads_finite"] and stats["posterior_mean_logvar_heads"]["any_nonzero_grad"] and
        stats["posterior_volatility_logvar_heads"]["all_present_grads_finite"] and stats["posterior_volatility_logvar_heads"]["any_nonzero_grad"] and
        stats["posterior_shared_logvar_heads"]["all_present_grads_finite"] and stats["posterior_shared_logvar_heads"]["any_nonzero_grad"]
    )
    
    dec_passed = stats["decoder"]["all_present_grads_finite"] and stats["decoder"]["any_nonzero_grad"]
    
    passed = enc_passed and mu_passed and logvar_passed and dec_passed
    
    return {
        "pass_name": "combined_backward_beta_small",
        "beta_value": P56_BETA_SMALL,
        "loss_value": float(loss_val),
        "loss_finite": True,
        "backward_completed": True,
        "gradient_stats": stats,
        "encoder_trunk_grad_expected": True,
        "posterior_mu_grad_expected": True,
        "posterior_logvar_grad_expected": True,
        "decoder_grad_expected": True,
        "encoder_trunk_passed": bool(enc_passed),
        "posterior_mu_passed": bool(mu_passed),
        "posterior_logvar_passed": bool(logvar_passed),
        "decoder_passed": bool(dec_passed),
        "passed": bool(passed),
    }


def run_fc_vae_backward_gradient_smoke_probe() -> Dict[str, Any]:
    try:
        torch = load_torch_for_p56_backward_smoke()
    except Exception:
        return {
            "contract_version": FC_VAE_BACKWARD_GRADIENT_SMOKE_CONTRACT_VERSION,
            "status": FC_VAE_BACKWARD_GRADIENT_SMOKE_STATUS_BLOCKED,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    targets = build_p50_bridge_targets()
    target = targets[0]
    target_id = target.get("target_id", P56_TARGET_ID)
    
    # Run reconstruction pass on fresh model
    model_recon = build_fc_vae_encoder_posterior_model()
    recon_res = run_p56_reconstruction_backward_pass(model_recon, target)
    
    # Run KL pass on fresh model
    model_kl = build_fc_vae_encoder_posterior_model()
    kl_res = run_p56_kl_backward_pass(model_kl, target)
    
    # Run combined pass on fresh model
    model_comb = build_fc_vae_encoder_posterior_model()
    comb_res = run_p56_combined_backward_pass(model_comb, target)
    
    all_completed = recon_res["backward_completed"] and kl_res["backward_completed"] and comb_res["backward_completed"]
    
    recon_passed = recon_res["passed"]
    kl_passed = kl_res["passed"]
    comb_passed = comb_res["passed"]
    
    all_passed = recon_passed and kl_passed and comb_passed
    
    # Check if all required gradients are finite & present
    all_required_gradients_finite = True
    all_required_gradients_present = True
    
    # Let's inspect the groups in the combined pass (which requires all groups to have grads)
    comb_stats = comb_res["gradient_stats"]
    for group_name, gs in comb_stats.items():
        if not gs["all_present_grads_finite"]:
            all_required_gradients_finite = False
        if not gs["any_grad_present"] or not gs["any_nonzero_grad"]:
            all_required_gradients_present = False
            
    summary = {
        "contract_version": FC_VAE_BACKWARD_GRADIENT_SMOKE_CONTRACT_VERSION,
        "kind": FC_VAE_BACKWARD_GRADIENT_SMOKE_KIND,
        "status": FC_VAE_BACKWARD_GRADIENT_SMOKE_STATUS_AVAILABLE,
        "torch_available": True,
        "source_phase": "P56",
        "target_id": target_id,
        "backward_pass_count": 3,
        
        "reconstruction_backward_passed": bool(recon_passed),
        "kl_backward_passed": bool(kl_passed),
        "combined_backward_passed": bool(comb_passed),
        
        "all_backward_passes_completed": bool(all_completed),
        "all_required_gradients_finite": bool(all_required_gradients_finite),
        "all_required_gradients_present": bool(all_required_gradients_present),
        
        # Boundaries
        "no_optimizer": True,
        "no_optimizer_step": True,
        "no_parameter_update": True,
        "no_training_loop": True,
        "no_dataset": True,
        "no_dataloader": True,
        "no_generation_claim": True,
        "no_gsb_claim": True,
        "no_scientific_conclusion": True,
        "no_latent_learning_claim": True,
        "no_vae_success_claim": True,
        "realizability_claim": "backward_gradient_smoke_only_no_optimizer_no_training",
        
        "passes": [
            {
                "pass_name": recon_res["pass_name"],
                "loss_value": recon_res["loss_value"],
                "passed": recon_res["passed"],
            },
            {
                "pass_name": kl_res["pass_name"],
                "loss_value": kl_res["loss_value"],
                "passed": kl_res["passed"],
            },
            {
                "pass_name": comb_res["pass_name"],
                "loss_value": comb_res["loss_value"],
                "passed": comb_res["passed"],
            }
        ],
        
        "reason": "fc_vae_backward_gradient_smoke_probe_success" if all_passed else "fc_vae_backward_gradient_smoke_probe_failure",
        "verdict": "PASS" if all_passed else "FAIL",
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def fc_vae_backward_gradient_smoke_probe_to_json_dict(probe_res: Dict[str, Any]) -> Dict[str, Any]:
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
            
    return d


def compact_fc_vae_backward_gradient_smoke_json(probe_res: Dict[str, Any]) -> str:
    d = fc_vae_backward_gradient_smoke_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
