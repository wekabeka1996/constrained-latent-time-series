# src/phase2/fc_vae_held_out_component_attribution_audit.py

import json
import math
from typing import Any, Dict, List

from src.phase2.fc_vae_tiny_held_out_seed_sensitivity_audit import (
    run_p63_tiny_held_out_seed_sensitivity_audit_probe,
)

# Constants
SOURCE_PHASE = "P64"
CONTRACT_VERSION = "phase2_p64_fc_vae_held_out_component_attribution_audit_contract_v1"
SOURCE_EVIDENCE_PHASE = "P63"
COMPONENT_RESIDUAL_TOLERANCE = 1e-5
TARGET_IDS = ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
SEED_VALUES = [62062, 62162, 62262]
EXPECTED_SEED_FOLD_RUNS = 9
BETA = 0.001


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
        "seed-robust " + "generalization", "component " + "proof",
    ]
    for item in forbidden_substrings:
        if item in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}: {item}")


def _classify_component_attribution(
    total_improved: bool,
    reconstruction_improved: bool,
    reconstruction_dominates: bool,
    beta_weighted_kl_helped_total: bool,
    beta_weighted_kl_hurt_total: bool,
) -> str:
    if total_improved and reconstruction_improved and reconstruction_dominates:
        return "reconstruction_supported"
    elif total_improved and reconstruction_improved and beta_weighted_kl_hurt_total:
        return "mixed_reconstruction_supported_kl_opposed"
    elif total_improved and (not reconstruction_improved) and beta_weighted_kl_helped_total:
        return "kl_supported"
    else:
        return "mixed_or_degraded"


def run_p64_held_out_component_attribution_audit_probe() -> dict:
    # 1. Fetch P63 evidence
    p63_res = run_p63_tiny_held_out_seed_sensitivity_audit_probe()
    
    # 2. Validate P63 evidence
    p63_valid = True
    if not isinstance(p63_res, dict):
        p63_valid = False
    elif p63_res.get("source_phase") != "P63":
        p63_valid = False
    elif p63_res.get("verdict") != "PASS":
        p63_valid = False
    elif p63_res.get("seed_count") != 3:
        p63_valid = False
    elif p63_res.get("fold_count_per_seed") != 3:
        p63_valid = False
    elif p63_res.get("total_seed_fold_runs") != 9:
        p63_valid = False
    elif p63_res.get("target_ids") != TARGET_IDS:
        p63_valid = False
    elif p63_res.get("seed_values") != SEED_VALUES:
        p63_valid = False
    elif p63_res.get("no_dataset") is not True:
        p63_valid = False
        
    if not p63_valid:
        return {
            "kind": "fc_vae_held_out_component_attribution_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p63_verdict": p63_res.get("verdict") if isinstance(p63_res, dict) else "ERROR",
            "source_p63_status": p63_res.get("status") if isinstance(p63_res, dict) else "ERROR",
            "status": "blocked_by_invalid_p63_evidence",
            "reason": "p63_evidence_validation_failed",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_seed_fold_runs": EXPECTED_SEED_FOLD_RUNS,
            "observed_seed_fold_runs": 0,
            "beta": BETA,
            "component_residual_tolerance": COMPONENT_RESIDUAL_TOLERANCE,
            "component_fold_summaries": [],
            "target_component_diagnostics": {},
            "aggregate_diagnostics": {},
            
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
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
            "no_component_proof_claim": True,
            
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "component_attribution_diagnostic_only": True,
        }
        
    # 3. Process P63 fold summaries
    component_fold_summaries = []
    all_runs_finite = True
    all_residuals_within_tolerance = True
    max_residual_abs = 0.0
    
    # Flatten summaries
    for s_summary in p63_res["seed_summaries"]:
        s_val = s_summary["seed_value"]
        for f_summary in s_summary["fold_summaries"]:
            tot_delta = f_summary["held_out_total_loss_delta_value"]
            rec_delta = f_summary["held_out_reconstruction_loss_delta_value"]
            kl_delta = f_summary["held_out_kl_loss_delta_value"]
            
            beta_kl_delta = BETA * kl_delta
            reconstructed_total_delta = rec_delta + beta_kl_delta
            residual = tot_delta - reconstructed_total_delta
            residual_abs = abs(residual)
            
            if residual_abs > max_residual_abs:
                max_residual_abs = residual_abs
                
            residual_within_tolerance = (residual_abs <= COMPONENT_RESIDUAL_TOLERANCE)
            if not residual_within_tolerance:
                all_residuals_within_tolerance = False
                
            finite_val = (
                math.isfinite(tot_delta)
                and math.isfinite(rec_delta)
                and math.isfinite(kl_delta)
                and math.isfinite(residual)
            )
            if not finite_val:
                all_runs_finite = False
                
            total_improved = (tot_delta < 0.0)
            total_degraded = (tot_delta > 0.0)
            reconstruction_improved = (rec_delta < 0.0)
            reconstruction_degraded = (rec_delta > 0.0)
            kl_decreased = (kl_delta < 0.0)
            kl_increased = (kl_delta > 0.0)
            
            beta_weighted_kl_helped = (beta_kl_delta < 0.0)
            beta_weighted_kl_hurt = (beta_kl_delta > 0.0)
            
            total_improvement_rec_supported = (total_improved and reconstruction_improved)
            total_improvement_kl_only = (total_improved and (not reconstruction_improved) and beta_weighted_kl_helped)
            
            rec_dominates = (abs(rec_delta) >= abs(beta_kl_delta))
            kl_dominates = (abs(beta_kl_delta) > abs(rec_delta))
            kl_opposes_rec = (reconstruction_improved and beta_weighted_kl_hurt)
            
            attribution_class = _classify_component_attribution(
                total_improved=total_improved,
                reconstruction_improved=reconstruction_improved,
                reconstruction_dominates=rec_dominates,
                beta_weighted_kl_helped_total=beta_weighted_kl_helped,
                beta_weighted_kl_hurt_total=beta_weighted_kl_hurt,
            )
            
            fold_passed = finite_val and residual_within_tolerance
            
            component_fold_summaries.append({
                "seed_value": s_val,
                "fold_id": f_summary["fold_id"],
                "held_out_target_id": f_summary["held_out_target_id"],
                "train_target_ids": f_summary["train_target_ids"],
                
                "total_delta_value": float(tot_delta),
                "reconstruction_delta_value": float(rec_delta),
                "kl_delta_value": float(kl_delta),
                "beta_weighted_kl_delta_value": float(beta_kl_delta),
                "reconstructed_total_delta_from_components": float(reconstructed_total_delta),
                "component_residual_value": float(residual),
                "component_residual_abs_value": float(residual_abs),
                "component_residual_within_tolerance": bool(residual_within_tolerance),
                
                "total_improved": bool(total_improved),
                "total_degraded": bool(total_degraded),
                "reconstruction_improved": bool(reconstruction_improved),
                "reconstruction_degraded": bool(reconstruction_degraded),
                "kl_decreased": bool(kl_decreased),
                "kl_increased": bool(kl_increased),
                "beta_weighted_kl_helped_total": bool(beta_weighted_kl_helped),
                "beta_weighted_kl_hurt_total": bool(beta_weighted_kl_hurt),
                
                "total_improvement_reconstruction_supported": bool(total_improvement_rec_supported),
                "total_improvement_kl_only": bool(total_improvement_kl_only),
                "reconstruction_dominates_total_delta": bool(rec_dominates),
                "kl_dominates_total_delta": bool(kl_dominates),
                "kl_opposes_reconstruction": bool(kl_opposes_rec),
                "component_attribution_class": attribution_class,
                "component_fold_passed": bool(fold_passed),
            })
            
    # 4. Target component diagnostics
    target_component_diagnostics = {}
    for ho_id in TARGET_IDS:
        target_runs = [r for r in component_fold_summaries if r["held_out_target_id"] == ho_id]
        # Sort by seed_value order
        target_runs_sorted = sorted(target_runs, key=lambda r: SEED_VALUES.index(r["seed_value"]))
        
        improved_tot = sum(1 for r in target_runs if r["total_improved"])
        improved_rec = sum(1 for r in target_runs if r["reconstruction_improved"])
        dec_kl = sum(1 for r in target_runs if r["kl_decreased"])
        inc_kl = sum(1 for r in target_runs if r["kl_increased"])
        beta_kl_hurt = sum(1 for r in target_runs if r["beta_weighted_kl_hurt_total"])
        beta_kl_helped = sum(1 for r in target_runs if r["beta_weighted_kl_helped_total"])
        
        rec_supported = sum(1 for r in target_runs if r["component_attribution_class"] in ["reconstruction_supported", "mixed_reconstruction_supported_kl_opposed"])
        kl_only = sum(1 for r in target_runs if r["component_attribution_class"] == "kl_supported")
        mixed_or_deg = sum(1 for r in target_runs if r["component_attribution_class"] == "mixed_or_degraded")
        
        residuals_within_tol = all(r["component_residual_within_tolerance"] for r in target_runs)
        max_abs_res = max(r["component_residual_abs_value"] for r in target_runs)
        
        classes = [r["component_attribution_class"] for r in target_runs_sorted]
        tot_deltas = [r["total_delta_value"] for r in target_runs_sorted]
        rec_deltas = [r["reconstruction_delta_value"] for r in target_runs_sorted]
        beta_kl_deltas = [r["beta_weighted_kl_delta_value"] for r in target_runs_sorted]
        
        target_component_diagnostics[ho_id] = {
            "held_out_target_id": ho_id,
            "seed_count": len(SEED_VALUES),
            "run_count": len(target_runs),
            "total_improved_count": improved_tot,
            "reconstruction_improved_count": improved_rec,
            "kl_increased_count": inc_kl,
            "kl_decreased_count": dec_kl,
            "beta_weighted_kl_hurt_count": beta_kl_hurt,
            "beta_weighted_kl_helped_count": beta_kl_helped,
            "reconstruction_supported_count": rec_supported,
            "kl_only_count": kl_only,
            "mixed_or_degraded_count": mixed_or_deg,
            
            "component_residual_max_abs": float(max_abs_res),
            "all_component_residuals_within_tolerance": bool(residuals_within_tol),
            "component_attribution_classes_by_seed": classes,
            "total_delta_values_by_seed": tot_deltas,
            "reconstruction_delta_values_by_seed": rec_deltas,
            "beta_weighted_kl_delta_values_by_seed": beta_kl_deltas,
            "diagnostic_interpretation": f"Target {ho_id} showed total improvement of {improved_tot} with {rec_supported} reconstruction-supported runs.",
        }
        
    # 5. Aggregate diagnostics
    all_component_folds_passed = all(r["component_fold_passed"] for r in component_fold_summaries)
    
    total_improved_count = sum(1 for r in component_fold_summaries if r["total_improved"])
    total_degraded_count = sum(1 for r in component_fold_summaries if r["total_degraded"])
    reconstruction_improved_count = sum(1 for r in component_fold_summaries if r["reconstruction_improved"])
    kl_increased_count = sum(1 for r in component_fold_summaries if r["kl_increased"])
    kl_decreased_count = sum(1 for r in component_fold_summaries if r["kl_decreased"])
    beta_weighted_kl_hurt_count = sum(1 for r in component_fold_summaries if r["beta_weighted_kl_hurt_total"])
    beta_weighted_kl_helped_count = sum(1 for r in component_fold_summaries if r["beta_weighted_kl_helped_total"])
    
    reconstruction_supported_count = sum(1 for r in component_fold_summaries if r["component_attribution_class"] in ["reconstruction_supported", "mixed_reconstruction_supported_kl_opposed"])
    kl_only_count = sum(1 for r in component_fold_summaries if r["component_attribution_class"] == "kl_supported")
    mixed_or_degraded_count = sum(1 for r in component_fold_summaries if r["component_attribution_class"] == "mixed_or_degraded")
    
    targets_reconstruction_supported_all_seeds = [
        ho_id for ho_id, d in target_component_diagnostics.items() if d["reconstruction_supported_count"] == len(SEED_VALUES)
    ]
    targets_with_kl_only_any_seed = [
        ho_id for ho_id, d in target_component_diagnostics.items() if d["kl_only_count"] > 0
    ]
    targets_with_kl_opposition_any_seed = [
        ho_id for ho_id in TARGET_IDS if any(r["kl_opposes_reconstruction"] for r in component_fold_summaries if r["held_out_target_id"] == ho_id)
    ]
    
    bridge_lambda_0_75_component_summary = target_component_diagnostics["bridge_lambda_0_75"]
    
    aggregate_diagnostics = {
        "source_p63_evidence_valid": bool(p63_valid),
        "all_component_folds_passed": bool(all_component_folds_passed),
        "all_component_values_finite": bool(all_runs_finite),
        "all_component_residuals_within_tolerance": bool(all_residuals_within_tolerance),
        "max_component_residual_abs": float(max_residual_abs),
        
        "total_improved_count": total_improved_count,
        "total_degraded_count": total_degraded_count,
        "reconstruction_improved_count": reconstruction_improved_count,
        "kl_increased_count": kl_increased_count,
        "kl_decreased_count": kl_decreased_count,
        "beta_weighted_kl_hurt_count": beta_weighted_kl_hurt_count,
        "beta_weighted_kl_helped_count": beta_weighted_kl_helped_count,
        "reconstruction_supported_count": reconstruction_supported_count,
        "kl_only_count": kl_only_count,
        "mixed_or_degraded_count": mixed_or_degraded_count,
        
        "targets_reconstruction_supported_all_seeds": targets_reconstruction_supported_all_seeds,
        "targets_with_kl_only_any_seed": targets_with_kl_only_any_seed,
        "targets_with_kl_opposition_any_seed": targets_with_kl_opposition_any_seed,
        "bridge_lambda_0_75_component_summary": bridge_lambda_0_75_component_summary,
        
        "component_attribution_claim": "held_out_component_attribution_audit_only_no_new_optimization_no_dataset_no_generalization",
        "diagnostic_only_no_generalization": True,
    }
    
    # Verdict is PASS only if:
    # 1. P63 evidence was valid and PASS
    # 2. All 9 fold summaries processed successfully
    # 3. All residuals are within tolerance
    # 4. Values are finite
    passed = (
        p63_valid
        and len(component_fold_summaries) == EXPECTED_SEED_FOLD_RUNS
        and all_residuals_within_tolerance
        and all_runs_finite
    )
    
    summary = {
        "kind": "fc_vae_held_out_component_attribution_audit_no_dataset_no_generalization",
        "source_phase": SOURCE_PHASE,
        "contract_version": CONTRACT_VERSION,
        "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
        "source_p63_verdict": p63_res["verdict"],
        "source_p63_status": p63_res["status"],
        "status": "fc_vae_held_out_component_attribution_audit_available_no_dataset_no_generalization" if passed else "fc_vae_held_out_component_attribution_audit_probe_failed",
        "reason": "fc_vae_held_out_component_attribution_audit_probe_success" if passed else "fc_vae_held_out_component_attribution_audit_probe_failed_due_to_residuals_or_validity",
        "verdict": "PASS" if passed else "FAIL",
        
        "target_ids": TARGET_IDS,
        "seed_values": SEED_VALUES,
        "expected_seed_fold_runs": EXPECTED_SEED_FOLD_RUNS,
        "observed_seed_fold_runs": len(component_fold_summaries),
        "beta": BETA,
        "component_residual_tolerance": COMPONENT_RESIDUAL_TOLERANCE,
        "component_fold_summaries": component_fold_summaries,
        "target_component_diagnostics": target_component_diagnostics,
        "aggregate_diagnostics": aggregate_diagnostics,
        
        # P63 source metrics info
        "source_p63_step_count": p63_res.get("step_count", 20),
        "source_p63_learning_rate": p63_res.get("learning_rate", 1e-4),
        
        # boundary flags
        "no_new_optimization": True,
        "no_direct_optimizer_created": True,
        "no_direct_model_created": True,
        "no_direct_torch_import": True,
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
        "no_component_proof_claim": True,
        
        "no_weighted_training": True,
        "uniform_objective_preserved": True,
        "held_out_diagnostic_only": True,
        "component_attribution_diagnostic_only": True,
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def fc_vae_held_out_component_attribution_audit_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_held_out_component_attribution_audit_json(probe_res: dict) -> str:
    d = fc_vae_held_out_component_attribution_audit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
