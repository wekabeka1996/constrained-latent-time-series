# src/phase2/fc_vae_one_step_optimizer_boundary.py

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
)
from src.phase2.fc_vae_backward_gradient_smoke import (
    collect_p56_parameter_groups,
    compute_all_gradient_stats,
    clear_model_grads,
)

# Constants
FC_VAE_ONE_STEP_OPTIMIZER_CONTRACT_VERSION = "phase2_p57_fc_vae_one_step_optimizer_boundary_contract_v1"
FC_VAE_ONE_STEP_OPTIMIZER_KIND = "fc_vae_one_step_optimizer_boundary_no_training_loop_no_dataset"
FC_VAE_ONE_STEP_OPTIMIZER_MODULE_NAME = "src.phase2.fc_vae_one_step_optimizer_boundary"
FC_VAE_ONE_STEP_OPTIMIZER_STATUS_AVAILABLE = "fc_vae_one_step_optimizer_boundary_available_no_training_loop_no_dataset"
FC_VAE_ONE_STEP_OPTIMIZER_STATUS_BLOCKED = "blocked_torch_unavailable"

P57_TARGET_ID = "bridge_lambda_0_25"
P57_EPS_MODE = "zero"
P57_BETA = 0.001
P57_LEARNING_RATE = 1e-4
P57_OPTIMIZER_NAME = "SGD"
P57_STEP_COUNT = 1
P57_SEED = 57057


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
        ("trained", "vae"), ("vae", "solved"),
        ("model", "converged"), ("model", "trained"),
    ]
    for a, b in forbidden_pairs:
        if a in val_lower and b in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}")


def load_torch_for_p57_optimizer_boundary() -> Any:
    return load_torch_for_p55_encoder_kl()


def collect_parameter_snapshot(model: Any) -> Dict[str, Any]:
    snapshot = {}
    for name, p in model.named_parameters():
        snapshot[name] = p.detach().clone()
    return snapshot


def compare_parameter_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    torch = load_torch_for_p57_optimizer_boundary()
    
    if set(before.keys()) != set(after.keys()):
        raise ValueError("Parameter names mismatch between before and after snapshots")
        
    param_count = len(before)
    param_changed = 0
    param_unchanged = 0
    all_deltas_finite = True
    any_parameter_changed = False
    
    deltas = []
    for name in before.keys():
        diff = (after[name] - before[name]).abs()
        is_finite = torch.all(torch.isfinite(diff)).item()
        if not is_finite:
            all_deltas_finite = False
            
        max_diff = torch.max(diff).item()
        if max_diff > 0.0:
            param_changed += 1
            any_parameter_changed = True
            deltas.append(diff)
        else:
            param_unchanged += 1
            
    if not deltas:
        return {
            "parameter_count": param_count,
            "parameters_changed": param_changed,
            "parameters_unchanged": param_unchanged,
            "all_deltas_finite": bool(all_deltas_finite),
            "any_parameter_changed": bool(any_parameter_changed),
            "max_abs_parameter_delta_value": 0.0,
            "mean_abs_parameter_delta_value": 0.0,
            "changed_parameter_ratio": 0.0,
        }
        
    all_deltas = torch.cat([d.view(-1) for d in deltas])
    max_delta = torch.max(all_deltas).item()
    mean_delta = torch.mean(all_deltas).item()
    
    return {
        "parameter_count": param_count,
        "parameters_changed": param_changed,
        "parameters_unchanged": param_unchanged,
        "all_deltas_finite": bool(all_deltas_finite),
        "any_parameter_changed": bool(any_parameter_changed),
        "max_abs_parameter_delta_value": float(max_delta),
        "mean_abs_parameter_delta_value": float(mean_delta),
        "changed_parameter_ratio": float(param_changed / param_count),
    }


def collect_group_named_parameters(model: Any) -> Dict[str, Dict[str, Any]]:
    groups = {
        "encoder_trunk": {},
        "posterior_mean_mu_heads": {},
        "posterior_mean_logvar_heads": {},
        "posterior_volatility_mu_heads": {},
        "posterior_volatility_logvar_heads": {},
        "posterior_shared_mu_heads": {},
        "posterior_shared_logvar_heads": {},
        "decoder": {},
    }
    
    for name, p in model.named_parameters():
        if name.startswith("encoder_trunk."):
            groups["encoder_trunk"][name] = p
        elif name.startswith("z_mean_mu_head."):
            groups["posterior_mean_mu_heads"][name] = p
        elif name.startswith("z_mean_logvar_head."):
            groups["posterior_mean_logvar_heads"][name] = p
        elif name.startswith("z_volatility_mu_head."):
            groups["posterior_volatility_mu_heads"][name] = p
        elif name.startswith("z_volatility_logvar_head."):
            groups["posterior_volatility_logvar_heads"][name] = p
        elif name.startswith("z_shared_mu_head."):
            groups["posterior_shared_mu_heads"][name] = p
        elif name.startswith("z_shared_logvar_head."):
            groups["posterior_shared_logvar_heads"][name] = p
        elif name.startswith("decoder."):
            groups["decoder"][name] = p
            
    return groups


def compare_parameter_group_deltas(model: Any, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    torch = load_torch_for_p57_optimizer_boundary()
    group_map = collect_group_named_parameters(model)
    
    stats = {}
    for gname, named_params in group_map.items():
        param_count = len(named_params)
        param_changed = 0
        all_deltas_finite = True
        any_parameter_changed = False
        
        deltas = []
        for name in named_params.keys():
            diff = (after[name] - before[name]).abs()
            is_finite = torch.all(torch.isfinite(diff)).item()
            if not is_finite:
                all_deltas_finite = False
                
            max_diff = torch.max(diff).item()
            if max_diff > 0.0:
                param_changed += 1
                any_parameter_changed = True
                deltas.append(diff)
                
        if not deltas:
            stats[gname] = {
                "parameter_count": param_count,
                "parameters_changed": param_changed,
                "all_deltas_finite": bool(all_deltas_finite),
                "any_parameter_changed": bool(any_parameter_changed),
                "max_abs_parameter_delta_value": 0.0,
                "mean_abs_parameter_delta_value": 0.0,
            }
        else:
            all_deltas = torch.cat([d.view(-1) for d in deltas])
            max_delta = torch.max(all_deltas).item()
            mean_delta = torch.mean(all_deltas).item()
            
            stats[gname] = {
                "parameter_count": param_count,
                "parameters_changed": param_changed,
                "all_deltas_finite": bool(all_deltas_finite),
                "any_parameter_changed": bool(any_parameter_changed),
                "max_abs_parameter_delta_value": float(max_delta),
                "mean_abs_parameter_delta_value": float(mean_delta),
            }
            
    return stats


def run_p57_one_step_optimizer_boundary_once() -> Dict[str, Any]:
    torch = load_torch_for_p57_optimizer_boundary()
    
    torch.manual_seed(P57_SEED)
    
    targets = build_p50_bridge_targets()
    target = targets[0]
    target_id = target.get("target_id", P57_TARGET_ID)
    
    model = build_fc_vae_encoder_posterior_model()
    model.train()
    
    # 1. Forward before step
    clear_model_grads(model)
    fwd_before = model(target, eps_mode=P57_EPS_MODE, beta=P57_BETA)
    
    loss_before = fwd_before["total_loss"]
    recon_before = fwd_before["loss_dict"]["loss_total"]
    kl_before = fwd_before["kl_dict"]["kl_total_batch_mean"]
    
    loss_before_val = loss_before.item()
    recon_before_val = recon_before.item()
    kl_before_val = kl_before.item()
    
    if not math.isfinite(loss_before_val):
        raise ValueError(f"Initial loss is non-finite: {loss_before_val}")
        
    # 2. Snapshot parameters before
    before_snapshot = collect_parameter_snapshot(model)
    
    # 3. Backward
    loss_before.backward()
    
    # 4. Compute gradient stats before step
    grad_stats_before = compute_all_gradient_stats(model)
    
    gradients_before_step_finite = True
    gradients_before_step_present = True
    for name, gs in grad_stats_before.items():
        if not gs["all_present_grads_finite"]:
            gradients_before_step_finite = False
        if not gs["any_grad_present"] or not gs["any_nonzero_grad"]:
            gradients_before_step_present = False
            
    # 5. One step
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=P57_LEARNING_RATE)
    optimizer.step()
    
    # 6. Clear grads after step
    clear_model_grads(model)
    
    # 7. Snapshot parameters after
    after_snapshot = collect_parameter_snapshot(model)
    
    # 8. Compare deltas
    global_deltas = compare_parameter_snapshots(before_snapshot, after_snapshot)
    group_deltas = compare_parameter_group_deltas(model, before_snapshot, after_snapshot)
    
    all_parameter_deltas_finite = global_deltas["all_deltas_finite"]
    any_parameter_changed = global_deltas["any_parameter_changed"]
    
    all_expected_groups_changed = True
    for gname, gs in group_deltas.items():
        if not gs["any_parameter_changed"] or not gs["all_deltas_finite"]:
            all_expected_groups_changed = False
            
    # 9. Forward after step
    fwd_after = model(target, eps_mode=P57_EPS_MODE, beta=P57_BETA)
    
    loss_after = fwd_after["total_loss"]
    recon_after = fwd_after["loss_dict"]["loss_total"]
    kl_after = fwd_after["kl_dict"]["kl_total_batch_mean"]
    
    loss_after_val = loss_after.item()
    recon_after_val = recon_after.item()
    kl_after_val = kl_after.item()
    
    after_forward_finite = math.isfinite(loss_after_val) and math.isfinite(recon_after_val) and math.isfinite(kl_after_val)
    
    loss_delta_value = loss_after_val - loss_before_val
    recon_delta_value = recon_after_val - recon_before_val
    kl_delta_value = kl_after_val - kl_before_val
    
    loss_decreased = (loss_after_val < loss_before_val)
    
    one_step_completed = True
    
    passed = (
        one_step_completed
        and math.isfinite(loss_before_val)
        and math.isfinite(loss_after_val)
        and gradients_before_step_finite
        and gradients_before_step_present
        and all_parameter_deltas_finite
        and any_parameter_changed
        and all_expected_groups_changed
        and after_forward_finite
    )
    
    return {
        "target_id": target_id,
        "seed_value": P57_SEED,
        "optimizer_name": P57_OPTIMIZER_NAME,
        "learning_rate": P57_LEARNING_RATE,
        "optimizer_step_count": P57_STEP_COUNT,
        
        "loss_before_value": float(loss_before_val),
        "loss_before_finite": True,
        "loss_after_value": float(loss_after_val),
        "loss_after_finite": True,
        "loss_delta_value": float(loss_delta_value),
        "loss_decreased": bool(loss_decreased),
        
        "reconstruction_loss_before_value": float(recon_before_val),
        "reconstruction_loss_after_value": float(recon_after_val),
        "reconstruction_loss_delta_value": float(recon_delta_value),
        
        "kl_before_value": float(kl_before_val),
        "kl_after_value": float(kl_after_val),
        "kl_delta_value": float(kl_delta_value),
        
        "gradients_before_step_finite": bool(gradients_before_step_finite),
        "gradients_before_step_present": bool(gradients_before_step_present),
        
        "global_parameter_delta_stats": global_deltas,
        "group_parameter_delta_stats": group_deltas,
        
        "all_parameter_deltas_finite": bool(all_parameter_deltas_finite),
        "any_parameter_changed": bool(any_parameter_changed),
        "all_expected_groups_changed": bool(all_expected_groups_changed),
        "after_forward_finite": bool(after_forward_finite),
        "one_step_completed": bool(one_step_completed),
        "passed": bool(passed),
    }


def run_fc_vae_one_step_optimizer_boundary_probe() -> Dict[str, Any]:
    try:
        torch = load_torch_for_p57_optimizer_boundary()
    except Exception:
        return {
            "contract_version": FC_VAE_ONE_STEP_OPTIMIZER_CONTRACT_VERSION,
            "status": FC_VAE_ONE_STEP_OPTIMIZER_STATUS_BLOCKED,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    res = run_p57_one_step_optimizer_boundary_once()
    
    summary = {
        "contract_version": FC_VAE_ONE_STEP_OPTIMIZER_CONTRACT_VERSION,
        "kind": FC_VAE_ONE_STEP_OPTIMIZER_KIND,
        "status": FC_VAE_ONE_STEP_OPTIMIZER_STATUS_AVAILABLE,
        "torch_available": True,
        "source_phase": "P57",
        
        "target_id": res["target_id"],
        "seed_value": res["seed_value"],
        "optimizer_name": res["optimizer_name"],
        "learning_rate": res["learning_rate"],
        "optimizer_step_count": res["optimizer_step_count"],
        
        "loss_before_value": res["loss_before_value"],
        "loss_before_finite": res["loss_before_finite"],
        "loss_after_value": res["loss_after_value"],
        "loss_after_finite": res["loss_after_finite"],
        "loss_delta_value": res["loss_delta_value"],
        "loss_decreased": res["loss_decreased"],
        
        "reconstruction_loss_before_value": res["reconstruction_loss_before_value"],
        "reconstruction_loss_after_value": res["reconstruction_loss_after_value"],
        
        "kl_before_value": res["kl_before_value"],
        "kl_after_value": res["kl_after_value"],
        
        "gradients_before_step_finite": res["gradients_before_step_finite"],
        "gradients_before_step_present": res["gradients_before_step_present"],
        
        "all_parameter_deltas_finite": res["all_parameter_deltas_finite"],
        "any_parameter_changed": res["any_parameter_changed"],
        "all_expected_groups_changed": res["all_expected_groups_changed"],
        "after_forward_finite": res["after_forward_finite"],
        "one_step_completed": res["one_step_completed"],
        
        "global_parameter_delta_stats": res["global_parameter_delta_stats"],
        "group_parameter_delta_stats": res["group_parameter_delta_stats"],
        
        # Boundaries
        "no_training_loop": True,
        "no_dataset": True,
        "no_dataloader": True,
        "no_epoch_loop": True,
        "no_batch_loop": True,
        "no_scheduler": True,
        "no_checkpointing": True,
        
        "no_generation_claim": True,
        "no_gsb_claim": True,
        "no_scientific_conclusion": True,
        "no_latent_learning_claim": True,
        "no_vae_success_claim": True,
        "no_convergence_claim": True,
        
        "realizability_claim": "one_step_optimizer_boundary_only_no_training_loop_no_dataset",
        "reason": "fc_vae_one_step_optimizer_boundary_probe_success" if res["passed"] else "fc_vae_one_step_optimizer_boundary_probe_failure",
        "verdict": "PASS" if res["passed"] else "FAIL",
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def fc_vae_one_step_optimizer_boundary_probe_to_json_dict(probe_res: Dict[str, Any]) -> Dict[str, Any]:
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
        elif isinstance(v, dict):
            # Safe nested dict copy
            d[k] = v
            
    return d


def compact_fc_vae_one_step_optimizer_boundary_json(probe_res: Dict[str, Any]) -> str:
    d = fc_vae_one_step_optimizer_boundary_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
