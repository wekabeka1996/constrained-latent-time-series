# src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py

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
from src.phase2.fc_vae_bounded_micro_training_harness import (
    build_p58_deterministic_target_set,
    compute_p58_micro_training_objective,
)

# Constants
SOURCE_PHASE = "P62"
CONTRACT_VERSION = "phase2_p62_fc_vae_tiny_held_out_deterministic_bridge_target_split_contract_v1"
TARGET_IDS = ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
FOLDS = [
    {"fold_index": 0, "held_out_target_id": "bridge_lambda_0_25", "train_target_ids": ["bridge_lambda_0_5", "bridge_lambda_0_75"]},
    {"fold_index": 1, "held_out_target_id": "bridge_lambda_0_5", "train_target_ids": ["bridge_lambda_0_25", "bridge_lambda_0_75"]},
    {"fold_index": 2, "held_out_target_id": "bridge_lambda_0_75", "train_target_ids": ["bridge_lambda_0_25", "bridge_lambda_0_5"]},
]
STEP_COUNT = 20
LEARNING_RATE = 1e-4
BETA = 0.001
SEED_VALUE = 62062
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
        "model " + "generalizes to held-out target",
    ]
    for item in forbidden_substrings:
        if item in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}: {item}")


def load_torch_for_p62_tiny_held_out_diagnostic() -> Any:
    return load_torch_for_p55_encoder_kl()


def run_p62_tiny_held_out_deterministic_bridge_target_split_probe() -> dict:
    try:
        torch = load_torch_for_p62_tiny_held_out_diagnostic()
    except Exception:
        return {
            "contract_version": CONTRACT_VERSION,
            "status": "blocked_torch_unavailable",
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    all_targets = build_p58_deterministic_target_set()
    
    fold_summaries = []
    
    # Process folds
    for f in FOLDS:
        fold_idx = f["fold_index"]
        held_out_target_id = f["held_out_target_id"]
        train_target_ids = f["train_target_ids"]
        
        # 1. Seed before model construction
        torch.manual_seed(SEED_VALUE)
        
        # 2. Build a fresh model
        model = build_fc_vae_encoder_posterior_model()
        model.train()
        
        # 3. Split targets
        train_targets = [t for t in all_targets if t["target_id"] in train_target_ids]
        held_out_target = [t for t in all_targets if t["target_id"] == held_out_target_id][0]
        
        # 4. Compute pre-update diagnostics
        pre_train_obj = compute_p58_micro_training_objective(model, train_targets, beta=BETA)
        pre_held_out_obj = compute_p58_micro_training_objective(model, [held_out_target], beta=BETA)
        
        initial_train_val = pre_train_obj["objective_loss"].item()
        initial_ho_total = pre_held_out_obj["objective_loss"].item()
        initial_ho_recon = pre_held_out_obj["reconstruction_mean"].item()
        initial_ho_kl = pre_held_out_obj["kl_mean"].item()
        
        # 5. Run SGD optimizer steps
        from torch.optim import SGD
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
        
        step_summaries = []
        all_step_losses_finite = True
        all_step_gradients_finite = True
        all_step_gradients_present = True
        
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
            
            step_summaries.append({
                "step_index": step_idx,
                "train_loss_value": float(loss_val),
                "train_loss_finite": bool(loss_finite),
                "train_reconstruction_mean_value": float(recon_val),
                "train_reconstruction_finite": bool(math.isfinite(recon_val)),
                "train_kl_mean_value": float(kl_val),
                "train_kl_finite": bool(math.isfinite(kl_val)),
                "gradients_present": bool(grad_present),
                "gradients_finite": bool(grad_finite),
                "optimizer_step_completed": True,
            })
            
        # 6. After updates
        clear_model_grads(model)
        
        post_train_obj = compute_p58_micro_training_objective(model, train_targets, beta=BETA)
        post_held_out_obj = compute_p58_micro_training_objective(model, [held_out_target], beta=BETA)
        
        final_train_val = post_train_obj["objective_loss"].item()
        final_ho_total = post_held_out_obj["objective_loss"].item()
        final_ho_recon = post_held_out_obj["reconstruction_mean"].item()
        final_ho_kl = post_held_out_obj["kl_mean"].item()
        
        train_objective_delta_value = final_train_val - initial_train_val
        train_objective_decreased = (train_objective_delta_value < 0.0)
        
        held_out_total_loss_delta_value = final_ho_total - initial_ho_total
        held_out_reconstruction_loss_delta_value = final_ho_recon - initial_ho_recon
        held_out_kl_loss_delta_value = final_ho_kl - initial_ho_kl
        
        if held_out_total_loss_delta_value < 0.0:
            movement_class = "improved"
        elif held_out_total_loss_delta_value > 0.0:
            movement_class = "degraded"
        else:
            movement_class = "unchanged"
            
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
            and len(step_summaries) == STEP_COUNT
        )
        
        fold_summaries.append({
            "fold_id": fold_idx,
            "held_out_target_id": held_out_target_id,
            "train_target_ids": train_target_ids,
            "train_target_count": len(train_target_ids),
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
            
            "completed_step_count": len(step_summaries),
            "all_step_losses_finite": bool(all_step_losses_finite),
            "all_step_gradients_present": bool(all_step_gradients_present),
            "all_step_gradients_finite": bool(all_step_gradients_finite),
            "all_optimizer_steps_completed": bool(len(step_summaries) == STEP_COUNT),
            "fold_passed": bool(fold_passed),
            "step_summaries": step_summaries,
        })
        
    # Aggregate diagnostics
    all_folds_passed = all(f["fold_passed"] for f in fold_summaries)
    all_folds_finite = all(f["all_step_losses_finite"] for f in fold_summaries)
    all_fold_gradients_present = all(f["all_step_gradients_present"] for f in fold_summaries)
    all_fold_gradients_finite = all(f["all_step_gradients_finite"] for f in fold_summaries)
    all_optimizer_steps_completed = all(f["all_optimizer_steps_completed"] for f in fold_summaries)
    
    train_objective_decreased_count = sum(1 for f in fold_summaries if f["train_objective_decreased"])
    all_train_objectives_decreased = (train_objective_decreased_count == len(FOLDS))
    
    held_out_improved_count = sum(1 for f in fold_summaries if f["held_out_total_improved"])
    held_out_degraded_count = sum(1 for f in fold_summaries if f["held_out_total_degraded"])
    held_out_unchanged_count = sum(1 for f in fold_summaries if f["held_out_total_unchanged"])
    
    held_out_improved_target_ids = [f["held_out_target_id"] for f in fold_summaries if f["held_out_total_improved"]]
    held_out_degraded_target_ids = [f["held_out_target_id"] for f in fold_summaries if f["held_out_total_degraded"]]
    held_out_unchanged_target_ids = [f["held_out_target_id"] for f in fold_summaries if f["held_out_total_unchanged"]]
    
    # hardest held-out target by final total loss
    hardest_ho = max(fold_summaries, key=lambda f: f["final_held_out_total_loss_value"])
    hardest_held_out_target_by_final_total_loss = hardest_ho["held_out_target_id"]
    hardest_held_out_final_total_loss_value = hardest_ho["final_held_out_total_loss_value"]
    
    # worst held-out delta target (most positive delta)
    worst_ho_delta = max(fold_summaries, key=lambda f: f["held_out_total_loss_delta_value"])
    worst_held_out_delta_target_id = worst_ho_delta["held_out_target_id"]
    worst_held_out_delta_value = worst_ho_delta["held_out_total_loss_delta_value"]
    
    # bridge_lambda_0_75_held_out_summary
    ho_75_f = [f for f in fold_summaries if f["held_out_target_id"] == "bridge_lambda_0_75"][0]
    # Keep it clean
    bridge_lambda_0_75_held_out_summary = {
        "held_out_target_id": ho_75_f["held_out_target_id"],
        "train_target_ids": ho_75_f["train_target_ids"],
        "initial_held_out_total_loss_value": ho_75_f["initial_held_out_total_loss_value"],
        "final_held_out_total_loss_value": ho_75_f["final_held_out_total_loss_value"],
        "held_out_total_loss_delta_value": ho_75_f["held_out_total_loss_delta_value"],
        "held_out_total_movement_class": ho_75_f["held_out_total_movement_class"],
    }
    
    aggregate_diagnostics = {
        "all_folds_passed": all_folds_passed,
        "all_folds_finite": all_folds_finite,
        "all_fold_gradients_present": all_fold_gradients_present,
        "all_fold_gradients_finite": all_fold_gradients_finite,
        "all_optimizer_steps_completed": all_optimizer_steps_completed,
        
        "train_objective_decreased_count": train_objective_decreased_count,
        "all_train_objectives_decreased": all_train_objectives_decreased,
        
        "held_out_improved_count": held_out_improved_count,
        "held_out_degraded_count": held_out_degraded_count,
        "held_out_unchanged_count": held_out_unchanged_count,
        
        "held_out_improved_target_ids": held_out_improved_target_ids,
        "held_out_degraded_target_ids": held_out_degraded_target_ids,
        "held_out_unchanged_target_ids": held_out_unchanged_target_ids,
        
        "hardest_held_out_target_by_final_total_loss": hardest_held_out_target_by_final_total_loss,
        "hardest_held_out_final_total_loss_value": hardest_held_out_final_total_loss_value,
        
        "worst_held_out_delta_target_id": worst_held_out_delta_target_id,
        "worst_held_out_delta_value": worst_held_out_delta_value,
        
        "bridge_lambda_0_75_held_out_summary": bridge_lambda_0_75_held_out_summary,
        
        "held_out_transfer_claim": "tiny_deterministic_bridge_target_transfer_diagnostic_only_no_dataset_no_generalization",
        "diagnostic_only_no_generalization": True,
    }
    
    passed = all_folds_passed
    
    summary = {
        "kind": "fc_vae_tiny_held_out_deterministic_bridge_target_split_no_dataset_no_generalization",
        "source_phase": SOURCE_PHASE,
        "contract_version": CONTRACT_VERSION,
        "status": "fc_vae_tiny_held_out_deterministic_bridge_target_split_available_no_dataset_no_generalization",
        "reason": "fc_vae_tiny_held_out_deterministic_bridge_target_split_probe_success" if passed else "fc_vae_tiny_held_out_deterministic_bridge_target_split_probe_failed",
        "verdict": "PASS" if passed else "FAIL",
        "torch_available": True,
        
        "seed_value": SEED_VALUE,
        "optimizer_name": OPTIMIZER_NAME,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "eps_mode": EPS_MODE,
        "step_count": STEP_COUNT,
        
        "target_count": len(all_targets),
        "target_ids": TARGET_IDS,
        "fold_count": len(FOLDS),
        "fold_summaries": fold_summaries,
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
        
        "no_weighted_training": True,
        "uniform_objective_preserved": True,
        "held_out_diagnostic_only": True,
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def fc_vae_tiny_held_out_deterministic_bridge_target_split_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_tiny_held_out_deterministic_bridge_target_split_json(probe_res: dict) -> str:
    d = fc_vae_tiny_held_out_deterministic_bridge_target_split_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
