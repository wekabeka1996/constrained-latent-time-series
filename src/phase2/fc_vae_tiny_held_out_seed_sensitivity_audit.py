# src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py

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
from src.phase2.fc_vae_bounded_micro_training_harness import (
    build_p58_deterministic_target_set,
    compute_p58_micro_training_objective,
)

# Constants
SOURCE_PHASE = "P63"
CONTRACT_VERSION = "phase2_p63_fc_vae_tiny_held_out_seed_sensitivity_audit_contract_v1"
TARGET_IDS = ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
FOLDS = [
    {"fold_index": 0, "held_out_target_id": "bridge_lambda_0_25", "train_target_ids": ["bridge_lambda_0_5", "bridge_lambda_0_75"]},
    {"fold_index": 1, "held_out_target_id": "bridge_lambda_0_5", "train_target_ids": ["bridge_lambda_0_25", "bridge_lambda_0_75"]},
    {"fold_index": 2, "held_out_target_id": "bridge_lambda_0_75", "train_target_ids": ["bridge_lambda_0_25", "bridge_lambda_0_5"]},
]
SEED_VALUES = [62062, 62162, 62262]
SEED_COUNT = 3
STEP_COUNT = 20
LEARNING_RATE = 1e-4
BETA = 0.001
EPS_MODE = "zero"
OPTIMIZER_NAME = "SGD"


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
        "held-out " + "generalization", "transfer " + "proven",
        "model " + "generalizes to held-out target", "proof " + "of transfer",
        "seed-robust " + "generalization",
    ]
    for item in forbidden_substrings:
        if item in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}: {item}")


def _classify_delta(delta: float) -> str:
    if delta < 0.0:
        return "improved"
    elif delta > 0.0:
        return "degraded"
    else:
        return "unchanged"


def _run_one_seed_fold(seed_value: int, fold: dict) -> dict:
    torch = load_torch_for_p55_encoder_kl()
    
    # 1. Seed before model construction
    torch.manual_seed(seed_value)
    
    # 2. Build model fresh
    model = build_fc_vae_encoder_posterior_model()
    model.train()
    
    # 3. Targets and split
    all_targets = build_p58_deterministic_target_set()
    train_targets = [t for t in all_targets if t["target_id"] in fold["train_target_ids"]]
    held_out_target = [t for t in all_targets if t["target_id"] == fold["held_out_target_id"]][0]
    
    # 4. Pre-update diagnostics
    pre_train_obj = compute_p58_micro_training_objective(model, train_targets, beta=BETA)
    pre_ho_obj = compute_p58_micro_training_objective(model, [held_out_target], beta=BETA)
    
    initial_train_val = pre_train_obj["objective_loss"].item()
    initial_ho_total = pre_ho_obj["objective_loss"].item()
    initial_ho_recon = pre_ho_obj["reconstruction_mean"].item()
    initial_ho_kl = pre_ho_obj["kl_mean"].item()
    
    # 5. Local optimizer SGD
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    
    first_step_train_loss_value = None
    last_step_train_loss_value = None
    all_step_losses_finite = True
    all_step_gradients_present = True
    all_step_gradients_finite = True
    
    # Singular explicit range loop
    for step_idx in range(STEP_COUNT):
        clear_model_grads(model)
        
        obj = compute_p58_micro_training_objective(model, train_targets, beta=BETA)
        loss_t = obj["objective_loss"]
        recon_t = obj["reconstruction_mean"]
        kl_t = obj["kl_mean"]
        
        loss_val = loss_t.item()
        recon_val = recon_t.item()
        kl_val = kl_t.item()
        
        if step_idx == 0:
            first_step_train_loss_value = float(loss_val)
        if step_idx == STEP_COUNT - 1:
            last_step_train_loss_value = float(loss_val)
            
        loss_finite = math.isfinite(loss_val) and math.isfinite(recon_val) and math.isfinite(kl_val)
        if not loss_finite:
            all_step_losses_finite = False
            
        loss_t.backward()
        
        grad_stats = compute_all_gradient_stats(model)
        grad_finite = True
        grad_present = True
        for name, gs in grad_stats.items():
            if not gs["all_present_grads_finite"]:
                grad_finite = False
            if not gs["any_grad_present"] or not gs["any_nonzero_grad"]:
                grad_present = False
                
        if not grad_finite:
            all_step_gradients_finite = False
        if not grad_present:
            all_step_gradients_present = False
            
        optimizer.step()
        
    step_train_loss_delta_value = last_step_train_loss_value - first_step_train_loss_value
    
    # 6. After updates
    clear_model_grads(model)
    post_train_obj = compute_p58_micro_training_objective(model, train_targets, beta=BETA)
    post_ho_obj = compute_p58_micro_training_objective(model, [held_out_target], beta=BETA)
    
    final_train_val = post_train_obj["objective_loss"].item()
    final_ho_total = post_ho_obj["objective_loss"].item()
    final_ho_recon = post_ho_obj["reconstruction_mean"].item()
    final_ho_kl = post_ho_obj["kl_mean"].item()
    
    train_objective_delta_value = final_train_val - initial_train_val
    train_objective_decreased = (train_objective_delta_value < 0.0)
    
    held_out_total_loss_delta_value = final_ho_total - initial_ho_total
    held_out_reconstruction_loss_delta_value = final_ho_recon - initial_ho_recon
    held_out_kl_loss_delta_value = final_ho_kl - initial_ho_kl
    
    movement_class = _classify_delta(held_out_total_loss_delta_value)
    held_out_total_improved = (movement_class == "improved")
    held_out_total_degraded = (movement_class == "degraded")
    held_out_total_unchanged = (movement_class == "unchanged")
    
    all_finite = (
        math.isfinite(initial_train_val)
        and math.isfinite(final_train_val)
        and math.isfinite(initial_ho_total)
        and math.isfinite(final_ho_total)
        and all_step_losses_finite
    )
    
    fold_passed = (
        all_finite
        and all_step_gradients_finite
        and all_step_gradients_present
    )
    
    return {
        "seed_value": seed_value,
        "fold_id": fold["fold_index"],
        "held_out_target_id": fold["held_out_target_id"],
        "train_target_ids": fold["train_target_ids"],
        "train_target_count": len(fold["train_target_ids"]),
        "held_out_target_count": 1,
        
        "initial_train_objective_value": float(initial_train_val),
        "final_train_objective_value": float(final_train_val),
        "train_objective_delta_value": float(train_objective_delta_value),
        "train_objective_decreased": bool(train_objective_decreased),
        
        "initial_held_out_total_loss_value": float(initial_ho_total),
        "final_held_out_total_loss_value": float(final_ho_total),
        "held_out_total_loss_delta_value": float(held_out_total_loss_delta_value),
        "held_out_total_improved": bool(held_out_total_improved),
        "held_out_total_degraded": bool(held_out_total_degraded),
        "held_out_total_unchanged": bool(held_out_total_unchanged),
        "held_out_total_movement_class": movement_class,
        
        "initial_held_out_reconstruction_loss_value": float(initial_ho_recon),
        "final_held_out_reconstruction_loss_value": float(final_ho_recon),
        "held_out_reconstruction_loss_delta_value": float(held_out_reconstruction_loss_delta_value),
        
        "initial_held_out_kl_loss_value": float(initial_ho_kl),
        "final_held_out_kl_loss_value": float(final_ho_kl),
        "held_out_kl_loss_delta_value": float(held_out_kl_loss_delta_value),
        
        # compact per-fold step diagnostics
        "first_step_train_loss_value": float(first_step_train_loss_value),
        "last_step_train_loss_value": float(last_step_train_loss_value),
        "step_train_loss_delta_value": float(step_train_loss_delta_value),
        "all_step_losses_finite": bool(all_step_losses_finite),
        "all_step_gradients_present": bool(all_step_gradients_present),
        "all_step_gradients_finite": bool(all_step_gradients_finite),
        "all_optimizer_steps_completed": bool(True),
        "completed_step_count": STEP_COUNT,
        "fold_passed": bool(fold_passed),
    }


def _build_target_summary(seed_fold_summaries: List[dict]) -> Dict[str, dict]:
    target_sensitivity_diagnostics = {}
    for ho_id in TARGET_IDS:
        folds = [f for f in seed_fold_summaries if f["held_out_target_id"] == ho_id]
        # Sort by seed_value order
        folds_sorted = sorted(folds, key=lambda f: SEED_VALUES.index(f["seed_value"]))
        deltas = [f["held_out_total_loss_delta_value"] for f in folds_sorted]
        classes = [f["held_out_total_movement_class"] for f in folds_sorted]
        
        improved = sum(1 for c in classes if c == "improved")
        degraded = sum(1 for c in classes if c == "degraded")
        unchanged = sum(1 for c in classes if c == "unchanged")
        
        all_finite = all(math.isfinite(d) for d in deltas)
        mean_delta = sum(deltas) / len(deltas) if len(deltas) > 0 else 0.0
        min_delta = min(deltas) if len(deltas) > 0 else 0.0
        max_delta = max(deltas) if len(deltas) > 0 else 0.0
        
        consistent_improvement = (improved == len(SEED_VALUES))
        any_degradation = (degraded > 0)
        warning = any_degradation or (len(set(classes)) > 1)
        
        target_sensitivity_diagnostics[ho_id] = {
            "held_out_target_id": ho_id,
            "seed_count": len(SEED_VALUES),
            "improved_count": improved,
            "degraded_count": degraded,
            "unchanged_count": unchanged,
            "movement_classes_by_seed": classes,
            "held_out_total_delta_values_by_seed": deltas,
            "mean_held_out_total_delta_value": mean_delta,
            "min_held_out_total_delta_value": min_delta,
            "max_held_out_total_delta_value": max_delta,
            "all_seed_deltas_finite": all_finite,
            "consistent_improvement_across_seeds": consistent_improvement,
            "any_degradation_across_seeds": any_degradation,
            "seed_sensitivity_warning": warning,
            "diagnostic_interpretation": f"Target {ho_id} showed {improved} improvements, {degraded} degradations, and {unchanged} unchanged across the seeds.",
        }
    return target_sensitivity_diagnostics


def run_p63_tiny_held_out_seed_sensitivity_audit_probe() -> dict:
    try:
        torch = load_torch_for_p55_encoder_kl()
    except Exception:
        return {
            "contract_version": CONTRACT_VERSION,
            "status": "blocked_torch_unavailable",
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    all_targets = build_p58_deterministic_target_set()
    
    seed_summaries = []
    all_seed_fold_runs = []
    
    for s_val in SEED_VALUES:
        fold_summaries = []
        for f in FOLDS:
            res = _run_one_seed_fold(s_val, f)
            fold_summaries.append(res)
            all_seed_fold_runs.append(res)
            
        seed_all_passed = all(fs["fold_passed"] for fs in fold_summaries)
        seed_all_finite = all(fs["all_step_losses_finite"] for fs in fold_summaries)
        seed_all_grads_present = all(fs["all_step_gradients_present"] for fs in fold_summaries)
        seed_all_grads_finite = all(fs["all_step_gradients_finite"] for fs in fold_summaries)
        seed_all_completed = all(fs["all_optimizer_steps_completed"] for fs in fold_summaries)
        
        seed_ho_improved = sum(1 for fs in fold_summaries if fs["held_out_total_improved"])
        seed_ho_degraded = sum(1 for fs in fold_summaries if fs["held_out_total_degraded"])
        seed_ho_unchanged = sum(1 for fs in fold_summaries if fs["held_out_total_unchanged"])
        seed_train_obj_dec = sum(1 for fs in fold_summaries if fs["train_objective_decreased"])
        
        seed_summaries.append({
            "seed_value": s_val,
            "fold_count": len(FOLDS),
            "fold_summaries": fold_summaries,
            "seed_all_folds_passed": seed_all_passed,
            "seed_all_folds_finite": seed_all_finite,
            "seed_all_gradients_present": seed_all_grads_present,
            "seed_all_gradients_finite": seed_all_grads_finite,
            "seed_all_optimizer_steps_completed": seed_all_completed,
            "seed_held_out_improved_count": seed_ho_improved,
            "seed_held_out_degraded_count": seed_ho_degraded,
            "seed_held_out_unchanged_count": seed_ho_unchanged,
            "seed_train_objective_decreased_count": seed_train_obj_dec,
        })
        
    target_sensitivity_diagnostics = _build_target_summary(all_seed_fold_runs)
    
    # Aggregate diagnostics
    all_runs_passed = all(r["fold_passed"] for r in all_seed_fold_runs)
    all_runs_finite = all(r["all_step_losses_finite"] for r in all_seed_fold_runs)
    all_gradients_present = all(r["all_step_gradients_present"] for r in all_seed_fold_runs)
    all_gradients_finite = all(r["all_step_gradients_finite"] for r in all_seed_fold_runs)
    all_optimizer_steps_completed = all(r["all_optimizer_steps_completed"] for r in all_seed_fold_runs)
    
    total_ho_improved = sum(1 for r in all_seed_fold_runs if r["held_out_total_improved"])
    total_ho_degraded = sum(1 for r in all_seed_fold_runs if r["held_out_total_degraded"])
    total_ho_unchanged = sum(1 for r in all_seed_fold_runs if r["held_out_total_unchanged"])
    total_train_obj_dec = sum(1 for r in all_seed_fold_runs if r["train_objective_decreased"])
    
    targets_consistently_improved = [
        ho_id for ho_id, d in target_sensitivity_diagnostics.items() if d["consistent_improvement_across_seeds"]
    ]
    targets_with_any_degradation = [
        ho_id for ho_id, d in target_sensitivity_diagnostics.items() if d["any_degradation_across_seeds"]
    ]
    targets_with_seed_sensitivity_warning = [
        ho_id for ho_id, d in target_sensitivity_diagnostics.items() if d["seed_sensitivity_warning"]
    ]
    
    bridge_lambda_0_75_seed_sensitivity_summary = target_sensitivity_diagnostics["bridge_lambda_0_75"]
    
    aggregate_diagnostics = {
        "all_runs_passed": all_runs_passed,
        "all_runs_finite": all_runs_finite,
        "all_gradients_present": all_gradients_present,
        "all_gradients_finite": all_gradients_finite,
        "all_optimizer_steps_completed": all_optimizer_steps_completed,
        
        "total_held_out_improved_count": total_ho_improved,
        "total_held_out_degraded_count": total_ho_degraded,
        "total_held_out_unchanged_count": total_ho_unchanged,
        "total_train_objective_decreased_count": total_train_obj_dec,
        
        "targets_consistently_improved": targets_consistently_improved,
        "targets_with_any_degradation": targets_with_any_degradation,
        "targets_with_seed_sensitivity_warning": targets_with_seed_sensitivity_warning,
        
        "bridge_lambda_0_75_seed_sensitivity_summary": bridge_lambda_0_75_seed_sensitivity_summary,
        
        "seed_sensitivity_claim": "tiny_deterministic_seed_sensitivity_audit_only_no_dataset_no_generalization",
        "diagnostic_only_no_generalization": True,
    }
    
    passed = all_runs_passed
    
    summary = {
        "kind": "fc_vae_tiny_held_out_seed_sensitivity_audit_no_dataset_no_generalization",
        "source_phase": SOURCE_PHASE,
        "contract_version": CONTRACT_VERSION,
        "status": "fc_vae_tiny_held_out_seed_sensitivity_audit_available_no_dataset_no_generalization",
        "reason": "fc_vae_tiny_held_out_seed_sensitivity_audit_probe_success" if passed else "fc_vae_tiny_held_out_seed_sensitivity_audit_probe_failed",
        "verdict": "PASS" if passed else "FAIL",
        "torch_available": True,
        
        "seed_values": SEED_VALUES,
        "seed_count": SEED_COUNT,
        "optimizer_name": OPTIMIZER_NAME,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "eps_mode": EPS_MODE,
        "step_count": STEP_COUNT,
        
        "target_count": len(all_targets),
        "target_ids": TARGET_IDS,
        "fold_count_per_seed": len(FOLDS),
        "total_seed_fold_runs": len(all_seed_fold_runs),
        
        "seed_summaries": seed_summaries,
        "target_sensitivity_diagnostics": target_sensitivity_diagnostics,
        "aggregate_diagnostics": aggregate_diagnostics,
        
        # boundary flags
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
        "no_transfer_proof_claim": True,
        "no_seed_robustness_claim": True,
        
        "no_weighted_training": True,
        "uniform_objective_preserved": True,
        "held_out_diagnostic_only": True,
        "seed_sensitivity_diagnostic_only": True,
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def fc_vae_tiny_held_out_seed_sensitivity_audit_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_tiny_held_out_seed_sensitivity_audit_json(probe_res: dict) -> str:
    d = fc_vae_tiny_held_out_seed_sensitivity_audit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
