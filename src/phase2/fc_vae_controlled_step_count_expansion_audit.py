# src/phase2/fc_vae_controlled_step_count_expansion_audit.py

import json
import math
from typing import Any, Dict, List, Tuple

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
)
from src.phase2.fc_vae_target_balance_objective_audit import (
    compute_p60_target_balance_metrics,
    compute_p60_counterfactual_weighting_diagnostics,
    derive_p60_balance_recommendation,
)

# Constants
FC_VAE_STEP_COUNT_EXPANSION_AUDIT_CONTRACT_VERSION = "phase2_p61_fc_vae_controlled_step_count_expansion_audit_contract_v1"
FC_VAE_STEP_COUNT_EXPANSION_AUDIT_KIND = "fc_vae_controlled_step_count_expansion_audit_no_dataset_no_generalization"
FC_VAE_STEP_COUNT_EXPANSION_AUDIT_MODULE_NAME = "src.phase2.fc_vae_controlled_step_count_expansion_audit"
FC_VAE_STEP_COUNT_EXPANSION_AUDIT_STATUS_AVAILABLE = "fc_vae_controlled_step_count_expansion_audit_available_no_dataset_no_generalization"
FC_VAE_STEP_COUNT_EXPANSION_AUDIT_STATUS_BLOCKED = "blocked_torch_unavailable"

P61_TARGET_COUNT = 3
P61_TARGET_IDS = ("bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75")
P61_STEP_COUNTS = (5, 10, 20)
P61_BASELINE_STEP_COUNT = 5
P61_MAX_STEP_COUNT = 20
P61_SEED = 61061
P61_BETA = 0.001
P61_LEARNING_RATE = 1e-4
P61_OPTIMIZER_NAME = "SGD"
P61_EPS_MODE = "zero"
P61_DOMINANCE_SHARE_WARNING_THRESHOLD = 0.40
P61_DOMINANCE_SHARE_BLOCK_THRESHOLD = 0.50
P61_LOSS_EXPLOSION_MULTIPLIER_BLOCK_THRESHOLD = 1.25
P61_KL_EXPLOSION_MULTIPLIER_BLOCK_THRESHOLD = 1.25


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


def load_torch_for_p61_step_count_expansion_audit() -> Any:
    return load_torch_for_p55_encoder_kl()


def run_p61_single_step_count(step_count: int) -> Dict[str, Any]:
    if step_count not in P61_STEP_COUNTS:
        raise ValueError(f"Requested step_count {step_count} is not one of P61_STEP_COUNTS: {P61_STEP_COUNTS}")
        
    torch = load_torch_for_p61_step_count_expansion_audit()
    
    torch.manual_seed(P61_SEED)
    targets = build_p58_deterministic_target_set()
    model = build_fc_vae_encoder_posterior_model()
    model.train()
    
    before_snapshot = collect_parameter_snapshot(model)
    
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=P61_LEARNING_RATE)
    
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
    
    # Bounded steps loop
    for step_idx in range(step_count):
        clear_model_grads(model)
        
        obj = compute_p58_micro_training_objective(model, targets, beta=P61_BETA)
        
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
            
        # Record target-specific metrics
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
            
    fwd_final = compute_p58_micro_training_objective(model, targets, beta=P61_BETA)
    final_obj_loss = fwd_final["objective_loss"]
    final_recon_mean = fwd_final["reconstruction_mean"]
    final_kl_mean = fwd_final["kl_mean"]
    
    final_obj_val = final_obj_loss.item()
    final_recon_val = final_recon_mean.item()
    final_kl_val = final_kl_mean.item()
    
    final_objective_finite = math.isfinite(final_obj_val) and math.isfinite(final_recon_val) and math.isfinite(final_kl_val)
    
    # Build per_target_diagnostics
    per_target_diagnostics = {}
    for i, t in enumerate(targets):
        tid = t["target_id"]
        init_tot = per_target_trajectories[tid]["total_loss_trajectory"][0]
        init_rec = per_target_trajectories[tid]["reconstruction_loss_trajectory"][0]
        init_kl = per_target_trajectories[tid]["kl_loss_trajectory"][0]
        
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
        
    initial_loss_val = step_summaries[0]["loss_value"]
    final_step_loss_val = step_summaries[-1]["loss_value"]
    loss_delta = final_step_loss_val - initial_loss_val
    loss_decreased = (final_step_loss_val < initial_loss_val)
    
    # Build a simulated probe output for P60 compatibility
    sub_probe = {
        "verdict": "PASS",
        "status": "fc_vae_reconstruction_trajectory_diagnostics_available_no_dataset_no_generalization",
        "target_count": len(targets),
        "target_ids": [t["target_id"] for t in targets],
        "per_target_diagnostics": per_target_diagnostics,
        "per_target_trajectories": per_target_trajectories,
        "best_worst_diagnostics": {},
        "dominance_diagnostics": {},
        "initial_loss_value": float(initial_loss_val),
        "final_step_loss_value": float(final_step_loss_val),
        "final_objective_loss_value": float(final_obj_val),
        "loss_delta_value": float(loss_delta),
        "loss_decreased": bool(loss_decreased),
        
        # boundaries
        "no_dataset": True, "no_dataloader": True, "no_epoch_loop": True, "no_batch_loop": True,
        "no_scheduler": True, "no_checkpointing": True, "no_generalization_claim": True,
        "no_generation_claim": True, "no_gsb_claim": True, "no_scientific_conclusion": True,
        "no_latent_learning_claim": True, "no_vae_success_claim": True, "no_convergence_claim": True,
        "no_semantic_geometry_proof_claim": True, "reason": "sub_probe_success"
    }
    
    # Compute balance metrics
    balance_metrics = compute_p60_target_balance_metrics(sub_probe)
    
    all_targets_total_loss_decreased = balance_metrics["all_targets_total_loss_decreased"]
    all_targets_reconstruction_loss_decreased = balance_metrics["all_targets_reconstruction_loss_decreased"]
    final_dominance_blocker = balance_metrics["dominance_blocker"]
    
    single_run_passed = (
        len(step_summaries) == step_count
        and all_step_losses_finite
        and all_step_gradients_finite
        and all_step_gradients_present
        and final_objective_finite
        and all_parameter_deltas_finite
        and any_parameter_changed
        and all_expected_groups_changed
        and all_targets_total_loss_decreased
        and all_targets_reconstruction_loss_decreased
        and final_dominance_blocker is False
    )
    
    return {
        "step_count": step_count,
        "completed_step_count": len(step_summaries),
        "initial_loss_value": float(initial_loss_val),
        "final_step_loss_value": float(final_step_loss_val),
        "final_objective_loss_value": float(final_obj_val),
        "final_reconstruction_mean_value": float(final_recon_val),
        "final_kl_mean_value": float(final_kl_val),
        "loss_delta_value": float(loss_delta),
        "loss_decreased": bool(loss_decreased),
        
        "all_step_losses_finite": bool(all_step_losses_finite),
        "all_step_gradients_finite": bool(all_step_gradients_finite),
        "all_step_gradients_present": bool(all_step_gradients_present),
        "all_parameter_deltas_finite": bool(all_parameter_deltas_finite),
        "any_parameter_changed": bool(any_parameter_changed),
        "all_expected_groups_changed": bool(all_expected_groups_changed),
        "final_objective_finite": bool(final_objective_finite),
        
        "per_target_diagnostics": per_target_diagnostics,
        "per_target_trajectories": per_target_trajectories,
        "balance_metrics": balance_metrics,
        "global_parameter_delta_stats": global_deltas,
        "group_parameter_delta_stats": group_deltas,
        "step_summaries": step_summaries,
        
        "passed": bool(single_run_passed),
    }


def compare_p61_step_count_runs(run_summaries: Dict[str, Any]) -> Dict[str, Any]:
    run5 = run_summaries["steps_5"]
    run10 = run_summaries["steps_10"]
    run20 = run_summaries["steps_20"]
    
    baseline_val = run5["final_objective_loss_value"]
    val10 = run10["final_objective_loss_value"]
    val20 = run20["final_objective_loss_value"]
    
    delta_10_vs_5 = val10 - baseline_val
    delta_20_vs_5 = val20 - baseline_val
    delta_20_vs_10 = val20 - val10
    
    is_10_less_equal_5 = (val10 <= baseline_val)
    is_20_less_equal_10 = (val20 <= val10)
    is_20_less_equal_5 = (val20 <= baseline_val)
    
    recon_5 = run5["final_reconstruction_mean_value"]
    recon_10 = run10["final_reconstruction_mean_value"]
    recon_20 = run20["final_reconstruction_mean_value"]
    
    kl_5 = run5["final_kl_mean_value"]
    kl_10 = run10["final_kl_mean_value"]
    kl_20 = run20["final_kl_mean_value"]
    
    kl_growth_ratio_20_vs_5 = kl_20 / kl_5 if kl_5 > 0.0 else 1.0
    obj_explosion_ratio_20_vs_5 = val20 / baseline_val if baseline_val > 0.0 else 1.0
    
    objective_explosion_blocker_triggered = (obj_explosion_ratio_20_vs_5 >= P61_LOSS_EXPLOSION_MULTIPLIER_BLOCK_THRESHOLD)
    kl_explosion_blocker_triggered = (kl_growth_ratio_20_vs_5 >= P61_KL_EXPLOSION_MULTIPLIER_BLOCK_THRESHOLD)
    
    share_5 = run5["balance_metrics"]["final_dominant_share"]
    share_10 = run10["balance_metrics"]["final_dominant_share"]
    share_20 = run20["balance_metrics"]["final_dominant_share"]
    
    dominance_share_max_across_runs = max(share_5, share_10, share_20)
    
    dominance_warning_across_runs = (dominance_share_max_across_runs >= P61_DOMINANCE_SHARE_WARNING_THRESHOLD)
    dominance_blocker_across_runs = (dominance_share_max_across_runs >= P61_DOMINANCE_SHARE_BLOCK_THRESHOLD)
    
    # Per-target improvement matrix
    target_ids = list(run5["per_target_diagnostics"].keys())
    per_target_matrix = []
    all_targets_improved = True
    
    for tid in target_ids:
        imp_5 = run5["balance_metrics"]["per_target_metrics"][tid]["absolute_total_improvement"]
        imp_10 = run10["balance_metrics"]["per_target_metrics"][tid]["absolute_total_improvement"]
        imp_20 = run20["balance_metrics"]["per_target_metrics"][tid]["absolute_total_improvement"]
        
        nondecreasing = (imp_20 >= imp_10 >= imp_5)
        if imp_5 <= 0.0 or imp_10 <= 0.0 or imp_20 <= 0.0:
            all_targets_improved = False
            
        per_target_matrix.append({
            "target_id": tid,
            "improvement_at_5": float(imp_5),
            "improvement_at_10": float(imp_10),
            "improvement_at_20": float(imp_20),
            "nondecreasing_improvement": bool(nondecreasing),
        })
        
    all_runs_passed = (run5["passed"] and run10["passed"] and run20["passed"])
    
    uniform_objective_stable_across_step_counts = (
        not objective_explosion_blocker_triggered
        and not kl_explosion_blocker_triggered
        and not dominance_blocker_across_runs
    )
    
    cross_run_passed = (
        all_runs_passed
        and not objective_explosion_blocker_triggered
        and not kl_explosion_blocker_triggered
        and not dominance_blocker_across_runs
        and all_targets_improved
    )
    
    return {
        "baseline_final_objective_value": float(baseline_val),
        "steps_10_final_objective_value": float(val10),
        "steps_20_final_objective_value": float(val20),
        
        "delta_10_vs_5": float(delta_10_vs_5),
        "delta_20_vs_5": float(delta_20_vs_5),
        "delta_20_vs_10": float(delta_20_vs_10),
        
        "is_10_less_equal_5": bool(is_10_less_equal_5),
        "is_20_less_equal_10": bool(is_20_less_equal_10),
        "is_20_less_equal_5": bool(is_20_less_equal_5),
        
        "final_reconstruction_mean_comparison": {
            "steps_5": float(recon_5),
            "steps_10": float(recon_10),
            "steps_20": float(recon_20),
        },
        "final_kl_mean_comparison": {
            "steps_5": float(kl_5),
            "steps_10": float(kl_10),
            "steps_20": float(kl_20),
        },
        
        "kl_growth_ratio_20_vs_5": float(kl_growth_ratio_20_vs_5),
        "objective_explosion_ratio_20_vs_5": float(obj_explosion_ratio_20_vs_5),
        "objective_explosion_blocker_triggered": bool(objective_explosion_blocker_triggered),
        "kl_explosion_blocker_triggered": bool(kl_explosion_blocker_triggered),
        
        "dominance_final_share": {
            "steps_5": float(share_5),
            "steps_10": float(share_10),
            "steps_20": float(share_20),
        },
        "dominance_share_max_across_runs": float(dominance_share_max_across_runs),
        "dominance_warning_across_runs": bool(dominance_warning_across_runs),
        "dominance_blocker_across_runs": bool(dominance_blocker_across_runs),
        
        "per_target_improvement_matrix": per_target_matrix,
        "all_targets_improved_at_all_step_counts": bool(all_targets_improved),
        "uniform_objective_stable_across_step_counts": bool(uniform_objective_stable_across_step_counts),
        
        "all_runs_passed": bool(all_runs_passed),
        "cross_run_passed": bool(cross_run_passed),
    }


def run_fc_vae_controlled_step_count_expansion_audit_probe() -> dict:
    try:
        torch = load_torch_for_p61_step_count_expansion_audit()
    except Exception:
        return {
            "contract_version": FC_VAE_STEP_COUNT_EXPANSION_AUDIT_CONTRACT_VERSION,
            "status": FC_VAE_STEP_COUNT_EXPANSION_AUDIT_STATUS_BLOCKED,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    run5 = run_p61_single_step_count(5)
    run10 = run_p61_single_step_count(10)
    run20 = run_p61_single_step_count(20)
    
    run_summaries = {
        "steps_5": run5,
        "steps_10": run10,
        "steps_20": run20,
    }
    
    comparison = compare_p61_step_count_runs(run_summaries)
    
    passed = comparison["cross_run_passed"]
    
    summary = {
        "contract_version": FC_VAE_STEP_COUNT_EXPANSION_AUDIT_CONTRACT_VERSION,
        "kind": FC_VAE_STEP_COUNT_EXPANSION_AUDIT_KIND,
        "status": FC_VAE_STEP_COUNT_EXPANSION_AUDIT_STATUS_AVAILABLE,
        "torch_available": True,
        "source_phase": "P61",
        
        "target_count": P61_TARGET_COUNT,
        "target_ids": list(P61_TARGET_IDS),
        "step_counts": list(P61_STEP_COUNTS),
        
        "seed_value": P61_SEED,
        "optimizer_name": P61_OPTIMIZER_NAME,
        "learning_rate": P61_LEARNING_RATE,
        "beta": P61_BETA,
        "eps_mode": P61_EPS_MODE,
        
        "run_summaries": run_summaries,
        "cross_run_comparison": comparison,
        
        "all_runs_passed": comparison["all_runs_passed"],
        "cross_run_passed": passed,
        
        # boundaries
        "no_dataset": True,
        "no_dataloader": True,
        "no_epoch_loop": True,
        "no_batch_loop": True,
        "no_scheduler": True,
        "no_checkpointing": True,
        "no_weighted_training": True,
        "uniform_objective_preserved": True,
        
        "no_generalization_claim": True,
        "no_generation_claim": True,
        "no_gsb_claim": True,
        "no_scientific_conclusion": True,
        "no_latent_learning_claim": True,
        "no_vae_success_claim": True,
        "no_convergence_claim": True,
        "no_semantic_geometry_proof_claim": True,
        
        "realizability_claim": "controlled_step_count_expansion_audit_only_no_dataset_no_generalization",
        "reason": "fc_vae_controlled_step_count_expansion_audit_probe_success" if passed else "fc_vae_controlled_step_count_expansion_audit_probe_failed",
        "verdict": "PASS" if passed else "FAIL",
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def fc_vae_controlled_step_count_expansion_audit_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_controlled_step_count_expansion_audit_json(probe_res: dict) -> str:
    d = fc_vae_controlled_step_count_expansion_audit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
