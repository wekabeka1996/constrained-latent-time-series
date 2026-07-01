# src/phase2/fc_vae_bounded_micro_training_harness.py

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
    compute_all_gradient_stats,
    clear_model_grads,
)
from src.phase2.fc_vae_one_step_optimizer_boundary import (
    collect_parameter_snapshot,
    compare_parameter_snapshots,
    compare_parameter_group_deltas,
    load_torch_for_p57_optimizer_boundary,
)

# Constants
FC_VAE_BOUNDED_MICRO_TRAINING_CONTRACT_VERSION = "phase2_p58_fc_vae_bounded_micro_training_harness_contract_v1"
FC_VAE_BOUNDED_MICRO_TRAINING_KIND = "fc_vae_bounded_micro_training_harness_no_dataset_no_generalization"
FC_VAE_BOUNDED_MICRO_TRAINING_MODULE_NAME = "src.phase2.fc_vae_bounded_micro_training_harness"
FC_VAE_BOUNDED_MICRO_TRAINING_STATUS_AVAILABLE = "fc_vae_bounded_micro_training_harness_available_no_dataset_no_generalization"
FC_VAE_BOUNDED_MICRO_TRAINING_STATUS_BLOCKED = "blocked_torch_unavailable"

P58_TARGET_COUNT = 3
P58_TARGET_IDS = ("bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75")
P58_EPS_MODE = "zero"
P58_BETA = 0.001
P58_LEARNING_RATE = 1e-4
P58_OPTIMIZER_NAME = "SGD"
P58_STEP_COUNT = 5
P58_SEED = 58058


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
        ("model", "converged"), ("model", "trained"), ("model", "generalized"),
        ("generalize", "claim"),
    ]
    for a, b in forbidden_pairs:
        if a in val_lower and b in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}")


def load_torch_for_p58_micro_training() -> Any:
    return load_torch_for_p55_encoder_kl()


def build_p58_deterministic_target_set() -> List[Dict[str, Any]]:
    targets = build_p50_bridge_targets()
    ordered = []
    for lval in [0.25, 0.50, 0.75]:
        found = None
        for t in targets:
            if abs(t["lambda_value"] - lval) < 1e-4:
                found = t
                break
        if found is None:
            raise ValueError(f"Target with lambda {lval} not found in bridge targets")
        ordered.append(found)
        
    if len(ordered) != P58_TARGET_COUNT:
        raise ValueError(f"Expected exactly {P58_TARGET_COUNT} targets, got {len(ordered)}")
        
    return ordered


def compute_p58_micro_training_objective(model: Any, targets: List[Dict[str, Any]], beta: float = P58_BETA) -> Dict[str, Any]:
    torch = load_torch_for_p58_micro_training()
    
    total_losses = []
    recon_losses = []
    kl_losses = []
    
    for target in targets:
        fwd = model(target, eps_mode=P58_EPS_MODE, beta=beta)
        total_losses.append(fwd["total_loss"])
        recon_losses.append(fwd["loss_dict"]["loss_total"])
        kl_losses.append(fwd["kl_dict"]["kl_total_batch_mean"])
        
    total_stack = torch.stack(total_losses)
    recon_stack = torch.stack(recon_losses)
    kl_stack = torch.stack(kl_losses)
    
    objective_loss = torch.mean(total_stack)
    recon_mean = torch.mean(recon_stack)
    kl_mean = torch.mean(kl_stack)
    
    return {
        "objective_loss": objective_loss,
        "reconstruction_mean": recon_mean,
        "kl_mean": kl_mean,
        "per_target_total_losses": total_losses,
        "per_target_reconstruction_losses": recon_losses,
        "per_target_kl_losses": kl_losses,
    }


def run_p58_bounded_micro_training_once() -> Dict[str, Any]:
    torch = load_torch_for_p58_micro_training()
    
    torch.manual_seed(P58_SEED)
    targets = build_p58_deterministic_target_set()
    model = build_fc_vae_encoder_posterior_model()
    model.train()
    
    before_snapshot = collect_parameter_snapshot(model)
    
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=P58_LEARNING_RATE)
    
    step_summaries = []
    all_step_losses_finite = True
    all_step_gradients_finite = True
    all_step_gradients_present = True
    
    for step_idx in range(P58_STEP_COUNT):
        clear_model_grads(model)
        
        obj = compute_p58_micro_training_objective(model, targets, beta=P58_BETA)
        
        obj_loss = obj["objective_loss"]
        recon_mean = obj["reconstruction_mean"]
        kl_mean = obj["kl_mean"]
        
        obj_val = obj_loss.item()
        recon_val = recon_mean.item()
        kl_val = kl_mean.item()
        
        loss_finite = math.isfinite(obj_val)
        recon_finite = math.isfinite(recon_val)
        kl_finite = math.isfinite(kl_val)
        
        if not (loss_finite and recon_finite and kl_finite):
            all_step_losses_finite = False
            
        obj_loss.backward()
        
        grad_stats = compute_all_gradient_stats(model)
        
        step_grad_finite = True
        step_grad_present = True
        for name, gs in grad_stats.items():
            if not gs["all_present_grads_finite"]:
                step_grad_finite = False
            if not gs["any_grad_present"] or not gs["any_nonzero_grad"]:
                step_grad_present = False
                
        if not step_grad_finite:
            all_step_gradients_finite = False
        if not step_grad_present:
            all_step_gradients_present = False
            
        optimizer.step()
        
        step_summaries.append({
            "step_index": step_idx,
            "loss_value": float(obj_val),
            "reconstruction_mean_value": float(recon_val),
            "kl_mean_value": float(kl_val),
            "loss_finite": bool(loss_finite),
            "reconstruction_finite": bool(recon_finite),
            "kl_finite": bool(kl_finite),
            "gradients_finite": bool(step_grad_finite),
            "gradients_present": bool(step_grad_present),
            "optimizer_step_completed": True,
        })
        
    clear_model_grads(model)
    after_snapshot = collect_parameter_snapshot(model)
    
    global_deltas = compare_parameter_snapshots(before_snapshot, after_snapshot)
    group_deltas = compare_parameter_group_deltas(model, before_snapshot, after_snapshot)
    
    all_parameter_deltas_finite = global_deltas["all_deltas_finite"]
    any_parameter_changed = global_deltas["any_parameter_changed"]
    
    all_expected_groups_changed = True
    for gname, gs in group_deltas.items():
        if not gs["any_parameter_changed"] or not gs["all_deltas_finite"]:
            all_expected_groups_changed = False
            
    fwd_final = compute_p58_micro_training_objective(model, targets, beta=P58_BETA)
    final_obj_loss = fwd_final["objective_loss"]
    final_recon_mean = fwd_final["reconstruction_mean"]
    final_kl_mean = fwd_final["kl_mean"]
    
    final_obj_val = final_obj_loss.item()
    final_recon_val = final_recon_mean.item()
    final_kl_val = final_kl_mean.item()
    
    final_objective_finite = math.isfinite(final_obj_val) and math.isfinite(final_recon_val) and math.isfinite(final_kl_val)
    
    initial_loss_val = step_summaries[0]["loss_value"]
    final_step_loss_val = step_summaries[-1]["loss_value"]
    
    loss_delta = final_step_loss_val - initial_loss_val
    loss_decreased = (final_step_loss_val < initial_loss_val)
    
    step_count_completed = len(step_summaries)
    
    passed = (
        step_count_completed == P58_STEP_COUNT
        and all_step_losses_finite
        and all_step_gradients_finite
        and all_step_gradients_present
        and all_parameter_deltas_finite
        and any_parameter_changed
        and all_expected_groups_changed
        and final_objective_finite
    )
    
    return {
        "target_count": len(targets),
        "target_ids": [t["target_id"] for t in targets],
        "seed_value": P58_SEED,
        "optimizer_name": P58_OPTIMIZER_NAME,
        "learning_rate": P58_LEARNING_RATE,
        "requested_step_count": P58_STEP_COUNT,
        "completed_step_count": step_count_completed,
        "all_optimizer_steps_completed": (step_count_completed == P58_STEP_COUNT),
        
        "initial_loss_value": float(initial_loss_val),
        "final_step_loss_value": float(final_step_loss_val),
        "final_objective_loss_value": float(final_obj_val),
        "loss_delta_value": float(loss_delta),
        "loss_decreased": bool(loss_decreased),
        
        "all_step_losses_finite": bool(all_step_losses_finite),
        "all_step_gradients_finite": bool(all_step_gradients_finite),
        "all_step_gradients_present": bool(all_step_gradients_present),
        "all_parameter_deltas_finite": bool(all_parameter_deltas_finite),
        "any_parameter_changed": bool(any_parameter_changed),
        "all_expected_groups_changed": bool(all_expected_groups_changed),
        "final_objective_finite": bool(final_objective_finite),
        
        "global_parameter_delta_stats": global_deltas,
        "group_parameter_delta_stats": group_deltas,
        "step_summaries": step_summaries,
        "passed": bool(passed),
    }


def run_fc_vae_bounded_micro_training_harness_probe() -> Dict[str, Any]:
    try:
        torch = load_torch_for_p58_micro_training()
    except Exception:
        return {
            "contract_version": FC_VAE_BOUNDED_MICRO_TRAINING_CONTRACT_VERSION,
            "status": FC_VAE_BOUNDED_MICRO_TRAINING_STATUS_BLOCKED,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    res = run_p58_bounded_micro_training_once()
    
    summary = {
        "contract_version": FC_VAE_BOUNDED_MICRO_TRAINING_CONTRACT_VERSION,
        "kind": FC_VAE_BOUNDED_MICRO_TRAINING_KIND,
        "status": FC_VAE_BOUNDED_MICRO_TRAINING_STATUS_AVAILABLE,
        "torch_available": True,
        "source_phase": "P58",
        
        "target_count": res["target_count"],
        "target_ids": res["target_ids"],
        "seed_value": res["seed_value"],
        "optimizer_name": res["optimizer_name"],
        "learning_rate": res["learning_rate"],
        "requested_step_count": res["requested_step_count"],
        "completed_step_count": res["completed_step_count"],
        "all_optimizer_steps_completed": res["all_optimizer_steps_completed"],
        
        "initial_loss_value": res["initial_loss_value"],
        "final_step_loss_value": res["final_step_loss_value"],
        "final_objective_loss_value": res["final_objective_loss_value"],
        "loss_delta_value": res["loss_delta_value"],
        "loss_decreased": res["loss_decreased"],
        
        "all_step_losses_finite": res["all_step_losses_finite"],
        "all_step_gradients_finite": res["all_step_gradients_finite"],
        "all_step_gradients_present": res["all_step_gradients_present"],
        "all_parameter_deltas_finite": res["all_parameter_deltas_finite"],
        "any_parameter_changed": res["any_parameter_changed"],
        "all_expected_groups_changed": res["all_expected_groups_changed"],
        "final_objective_finite": res["final_objective_finite"],
        
        "global_parameter_delta_stats": res["global_parameter_delta_stats"],
        "group_parameter_delta_stats": res["group_parameter_delta_stats"],
        "step_summaries": res["step_summaries"],
        
        # Boundaries
        "no_dataset": True,
        "no_dataloader": True,
        "no_epoch_loop": True,
        "no_batch_loop": True,
        "no_scheduler": True,
        "no_checkpointing": True,
        
        "no_generalization_claim": True,
        "no_generation_claim": True,
        "no_gsb_claim": True,
        "no_scientific_conclusion": True,
        "no_latent_learning_claim": True,
        "no_vae_success_claim": True,
        "no_convergence_claim": True,
        
        "realizability_claim": "bounded_micro_training_harness_only_no_dataset_no_generalization",
        "reason": "fc_vae_bounded_micro_training_harness_probe_success" if res["passed"] else "fc_vae_bounded_micro_training_harness_probe_failure",
        "verdict": "PASS" if res["passed"] else "FAIL",
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def fc_vae_bounded_micro_training_harness_probe_to_json_dict(probe_res: Dict[str, Any]) -> Dict[str, Any]:
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
            # step_summaries is a list of dicts, target_ids is a list of strings
            d[k] = v
        elif isinstance(v, dict):
            d[k] = v
            
    return d


def compact_fc_vae_bounded_micro_training_harness_json(probe_res: Dict[str, Any]) -> str:
    d = fc_vae_bounded_micro_training_harness_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
