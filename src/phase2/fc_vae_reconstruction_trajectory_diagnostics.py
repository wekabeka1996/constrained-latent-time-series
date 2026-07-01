# src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py

import json
import math
from typing import Any, Dict, List

from src.phase2.torch_boundary import (
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
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
)
from src.phase2.fc_vae_bounded_micro_training_harness import (
    build_p58_deterministic_target_set,
    compute_p58_micro_training_objective,
    P58_SEED,
    P58_BETA,
    P58_LEARNING_RATE,
    P58_EPS_MODE,
    P58_STEP_COUNT,
    P58_TARGET_COUNT,
)

# Constants
FC_VAE_RECONSTRUCTION_TRAJECTORY_CONTRACT_VERSION = "phase2_p59_fc_vae_reconstruction_trajectory_diagnostics_contract_v1"
FC_VAE_RECONSTRUCTION_TRAJECTORY_KIND = "fc_vae_reconstruction_trajectory_diagnostics_no_dataset_no_generalization"
FC_VAE_RECONSTRUCTION_TRAJECTORY_MODULE_NAME = "src.phase2.fc_vae_reconstruction_trajectory_diagnostics"
FC_VAE_RECONSTRUCTION_TRAJECTORY_STATUS_AVAILABLE = "fc_vae_reconstruction_trajectory_diagnostics_available_no_dataset_no_generalization"
FC_VAE_RECONSTRUCTION_TRAJECTORY_STATUS_BLOCKED = "blocked_torch_unavailable"

P59_SEED = 58058
P59_BETA = 0.001
P59_LEARNING_RATE = 1e-4
P59_STEP_COUNT = 5


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
    forbidden_substrings = [
        "scientific " + "success", "gsb " + "solved", "vae " + "works",
        "latent " + "space learned", "semantic " + "geometry proven",
        "production " + "ready", "state-of-the-" + "art",
        "schrodinger " + "bridge solved", "c " + "generated",
        "posterior " + "collapse solved", "synthetic " + "state proven",
        "trained " + "vae", "vae " + "solved",
        "model " + "converged", "model " + "trained", "model " + "generalized",
        "generalization " + "claim", "vae " + "success claim", "gsb " + "claim",
        "semantic " + "geometry proof claim", "scientific " + "conclusion",
    ]
    for item in forbidden_substrings:
        if item in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}: {item}")


def load_torch_for_p59_trajectory_diagnostics() -> Any:
    return load_torch_for_p55_encoder_kl()


def run_p59_reconstruction_trajectory_diagnostics_once() -> Dict[str, Any]:
    torch = load_torch_for_p59_trajectory_diagnostics()
    
    torch.manual_seed(P59_SEED)
    targets = build_p58_deterministic_target_set()
    model = build_fc_vae_encoder_posterior_model()
    model.train()
    
    before_snapshot = collect_parameter_snapshot(model)
    
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=P59_LEARNING_RATE)
    
    step_summaries = []
    per_target_trajectories = {
        t["target_id"]: {
            "total_loss_trajectory": [],
            "reconstruction_loss_trajectory": [],
            "kl_loss_trajectory": [],
        }
        for t in targets
    }
    
    all_step_losses_finite = True
    all_step_gradients_finite = True
    all_step_gradients_present = True
    
    # Run bounded steps loop
    for step_idx in range(P59_STEP_COUNT):
        clear_model_grads(model)
        
        obj = compute_p58_micro_training_objective(model, targets, beta=P59_BETA)
        
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
            
        # Record target-specific metrics at this step
        for i, t in enumerate(targets):
            tid = t["target_id"]
            tot_val = obj["per_target_total_losses"][i].item()
            rec_val = obj["per_target_reconstruction_losses"][i].item()
            kl_val_t = obj["per_target_kl_losses"][i].item()
            
            per_target_trajectories[tid]["total_loss_trajectory"].append(float(tot_val))
            per_target_trajectories[tid]["reconstruction_loss_trajectory"].append(float(rec_val))
            per_target_trajectories[tid]["kl_loss_trajectory"].append(float(kl_val_t))
            
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
            
    # Final forward objective evaluation
    fwd_final = compute_p58_micro_training_objective(model, targets, beta=P59_BETA)
    final_obj_loss = fwd_final["objective_loss"]
    final_recon_mean = fwd_final["reconstruction_mean"]
    final_kl_mean = fwd_final["kl_mean"]
    
    final_obj_val = final_obj_loss.item()
    final_recon_val = final_recon_mean.item()
    final_kl_val = final_kl_mean.item()
    
    final_objective_finite = math.isfinite(final_obj_val) and math.isfinite(final_recon_val) and math.isfinite(final_kl_val)
    
    # Collect final target values to compute deltas
    per_target_diagnostics = {}
    for i, t in enumerate(targets):
        tid = t["target_id"]
        
        # Initial values (step 0 trajectory entry)
        init_tot = per_target_trajectories[tid]["total_loss_trajectory"][0]
        init_rec = per_target_trajectories[tid]["reconstruction_loss_trajectory"][0]
        init_kl = per_target_trajectories[tid]["kl_loss_trajectory"][0]
        
        # Final values after 5 steps updates
        fin_tot = fwd_final["per_target_total_losses"][i].item()
        fin_rec = fwd_final["per_target_reconstruction_losses"][i].item()
        fin_kl = fwd_final["per_target_kl_losses"][i].item()
        
        per_target_diagnostics[tid] = {
            "initial_total_loss": float(init_tot),
            "final_total_loss": float(fin_tot),
            "total_loss_delta": float(fin_tot - init_tot),
            
            "initial_reconstruction_loss": float(init_rec),
            "final_reconstruction_loss": float(fin_rec),
            "reconstruction_loss_delta": float(fin_rec - init_rec),
            
            "initial_kl_loss": float(init_kl),
            "final_kl_loss": float(fin_kl),
            "kl_loss_delta": float(fin_kl - init_kl),
        }
        
    # Best/Worst Diagnostics
    best_recon_tid = worst_recon_tid = targets[0]["target_id"]
    best_recon_val = worst_recon_val = per_target_diagnostics[best_recon_tid]["final_reconstruction_loss"]
    
    best_delta_tid = worst_delta_tid = targets[0]["target_id"]
    best_delta_val = worst_delta_val = per_target_diagnostics[best_delta_tid]["reconstruction_loss_delta"]
    
    for t in targets:
        tid = t["target_id"]
        fin_rec = per_target_diagnostics[tid]["final_reconstruction_loss"]
        rec_delta = per_target_diagnostics[tid]["reconstruction_loss_delta"]
        
        if fin_rec < best_recon_val:
            best_recon_val = fin_rec
            best_recon_tid = tid
        if fin_rec > worst_recon_val:
            worst_recon_val = fin_rec
            worst_recon_tid = tid
            
        if rec_delta < best_delta_val:  # larger decrease means more negative delta
            best_delta_val = rec_delta
            best_delta_tid = tid
        if rec_delta > worst_delta_val:
            worst_delta_val = rec_delta
            worst_delta_tid = tid
            
    best_worst_diagnostics = {
        "best_target_by_final_reconstruction": best_recon_tid,
        "worst_target_by_final_reconstruction": worst_recon_tid,
        "best_target_by_reconstruction_delta": best_delta_tid,
        "worst_target_by_reconstruction_delta": worst_delta_tid,
        "best_final_reconstruction_value": float(best_recon_val),
        "worst_final_reconstruction_value": float(worst_recon_val),
        "best_reconstruction_delta_value": float(best_delta_val),
        "worst_reconstruction_delta_value": float(worst_delta_val),
    }
    
    # Dominance Diagnostics
    initial_total_losses = [per_target_diagnostics[t["target_id"]]["initial_total_loss"] for t in targets]
    initial_sum = sum(initial_total_losses)
    
    final_total_losses = [per_target_diagnostics[t["target_id"]]["final_total_loss"] for t in targets]
    final_sum = sum(final_total_losses)
    
    dom_init_tid = targets[0]["target_id"]
    dom_init_val = per_target_diagnostics[dom_init_tid]["initial_total_loss"]
    
    dom_fin_tid = targets[0]["target_id"]
    dom_fin_val = per_target_diagnostics[dom_fin_tid]["final_total_loss"]
    
    for t in targets:
        tid = t["target_id"]
        init_val = per_target_diagnostics[tid]["initial_total_loss"]
        fin_val = per_target_diagnostics[tid]["final_total_loss"]
        
        if init_val > dom_init_val:
            dom_init_val = init_val
            dom_init_tid = tid
        if fin_val > dom_fin_val:
            dom_fin_val = fin_val
            dom_fin_tid = tid
            
    initial_dominant_share = dom_init_val / initial_sum if initial_sum > 0.0 else 0.0
    final_dominant_share = dom_fin_val / final_sum if final_sum > 0.0 else 0.0
    
    dominance_diagnostics = {
        "initial_dominant_target": dom_init_tid,
        "initial_dominant_value": float(dom_init_val),
        "initial_dominant_share": float(initial_dominant_share),
        "final_dominant_target": dom_fin_tid,
        "final_dominant_value": float(dom_fin_val),
        "final_dominant_share": float(final_dominant_share),
    }
    
    initial_loss_val = step_summaries[0]["loss_value"]
    final_step_loss_val = step_summaries[-1]["loss_value"]
    
    loss_delta = final_step_loss_val - initial_loss_val
    loss_decreased = (final_step_loss_val < initial_loss_val)
    
    step_count_completed = len(step_summaries)
    
    passed = (
        step_count_completed == P59_STEP_COUNT
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
        "seed_value": P59_SEED,
        "optimizer_name": "SGD",
        "learning_rate": P59_LEARNING_RATE,
        "requested_step_count": P59_STEP_COUNT,
        "completed_step_count": step_count_completed,
        "all_optimizer_steps_completed": (step_count_completed == P59_STEP_COUNT),
        
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
        "per_target_trajectories": per_target_trajectories,
        "per_target_diagnostics": per_target_diagnostics,
        "best_worst_diagnostics": best_worst_diagnostics,
        "dominance_diagnostics": dominance_diagnostics,
        
        "passed": bool(passed),
    }


def run_fc_vae_reconstruction_trajectory_diagnostics_probe() -> Dict[str, Any]:
    try:
        torch = load_torch_for_p59_trajectory_diagnostics()
    except Exception:
        return {
            "contract_version": FC_VAE_RECONSTRUCTION_TRAJECTORY_CONTRACT_VERSION,
            "status": FC_VAE_RECONSTRUCTION_TRAJECTORY_STATUS_BLOCKED,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    
    summary = {
        "contract_version": FC_VAE_RECONSTRUCTION_TRAJECTORY_CONTRACT_VERSION,
        "kind": FC_VAE_RECONSTRUCTION_TRAJECTORY_KIND,
        "status": FC_VAE_RECONSTRUCTION_TRAJECTORY_STATUS_AVAILABLE,
        "torch_available": True,
        "source_phase": "P59",
        
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
        
        # New Diagnostics
        "per_target_trajectories": res["per_target_trajectories"],
        "per_target_diagnostics": res["per_target_diagnostics"],
        "best_worst_diagnostics": res["best_worst_diagnostics"],
        "dominance_diagnostics": res["dominance_diagnostics"],
        
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
        "no_semantic_geometry_proof_claim": True,
        
        "realizability_claim": "trajectory_diagnostics_only_no_dataset_no_generalization",
        "reason": "fc_vae_reconstruction_trajectory_diagnostics_probe_success" if res["passed"] else "fc_vae_reconstruction_trajectory_diagnostics_probe_failure",
        "verdict": "PASS" if res["passed"] else "FAIL",
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def fc_vae_reconstruction_trajectory_diagnostics_probe_to_json_dict(probe_res: Dict[str, Any]) -> Dict[str, Any]:
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
            d[k] = v
            
    return d


def compact_fc_vae_reconstruction_trajectory_diagnostics_json(probe_res: Dict[str, Any]) -> str:
    d = fc_vae_reconstruction_trajectory_diagnostics_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
