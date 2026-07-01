# src/phase2/fc_vae_target_balance_objective_audit.py

import json
import math
from typing import Any, Dict, List, Tuple

from src.phase2.torch_boundary import (
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)
from src.phase2.fc_vae_reconstruction_trajectory_diagnostics import (
    load_torch_for_p59_trajectory_diagnostics,
    run_p59_reconstruction_trajectory_diagnostics_once,
    run_fc_vae_reconstruction_trajectory_diagnostics_probe,
)



# Constants
FC_VAE_TARGET_BALANCE_AUDIT_CONTRACT_VERSION = "phase2_p60_fc_vae_target_balance_objective_audit_contract_v1"
FC_VAE_TARGET_BALANCE_AUDIT_KIND = "fc_vae_target_balance_objective_audit_no_training_change_no_dataset"
FC_VAE_TARGET_BALANCE_AUDIT_MODULE_NAME = "src.phase2.fc_vae_target_balance_objective_audit"
FC_VAE_TARGET_BALANCE_AUDIT_STATUS_AVAILABLE = "fc_vae_target_balance_objective_audit_available_no_training_change_no_dataset"
FC_VAE_TARGET_BALANCE_AUDIT_STATUS_BLOCKED = "blocked_torch_unavailable"

P60_TARGET_COUNT = 3
P60_EXPECTED_TARGET_IDS = ("bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75")
P60_DOMINANCE_SHARE_WARNING_THRESHOLD = 0.40
P60_DOMINANCE_SHARE_BLOCK_THRESHOLD = 0.50
P60_DELTA_SPREAD_WARNING_THRESHOLD = 0.002
P60_IMPROVEMENT_RATIO_WARNING_THRESHOLD = 2.0

P60_RECOMMENDATION_STATUS_NO_WEIGHTING_REQUIRED = "no_weighting_required_for_next_phase"
P60_RECOMMENDATION_STATUS_WEIGHTING_WATCH = "weighting_watch_recommended"
P60_RECOMMENDATION_STATUS_WEIGHTING_REQUIRED_BEFORE_EXPANSION = "weighting_required_before_expansion"


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


def load_torch_for_p60_balance_audit() -> Any:
    return load_torch_for_p59_trajectory_diagnostics()


def extract_p60_balance_inputs_from_p59_probe(p59_probe: dict) -> dict:
    if p59_probe.get("verdict") != "PASS":
        raise ValueError(f"P59 evidence verdict is not PASS: {p59_probe.get('verdict')}")
    if p59_probe.get("status") != "fc_vae_reconstruction_trajectory_diagnostics_available_no_dataset_no_generalization":
        raise ValueError(f"Unexpected P59 status: {p59_probe.get('status')}")
        
    required_keys = [
        "target_count", "target_ids", "per_target_diagnostics", "per_target_trajectories",
        "best_worst_diagnostics", "dominance_diagnostics", "initial_loss_value",
        "final_step_loss_value", "final_objective_loss_value", "loss_delta_value",
        "loss_decreased"
    ]
    for k in required_keys:
        if k not in p59_probe:
            raise KeyError(f"Missing required P59 probe key: {k}")
            
    if p59_probe["target_count"] != P60_TARGET_COUNT:
        raise ValueError(f"target_count {p59_probe['target_count']} != {P60_TARGET_COUNT}")
    if list(p59_probe["target_ids"]) != list(P60_EXPECTED_TARGET_IDS):
        raise ValueError(f"Unexpected target_ids: {p59_probe['target_ids']}")
        
    boundary_keys = [
        "no_dataset", "no_dataloader", "no_epoch_loop", "no_batch_loop",
        "no_scheduler", "no_checkpointing", "no_generalization_claim",
        "no_generation_claim", "no_gsb_claim", "no_scientific_conclusion",
        "no_latent_learning_claim", "no_vae_success_claim", "no_convergence_claim",
        "no_semantic_geometry_proof_claim"
    ]
    for bk in boundary_keys:
        if not p59_probe.get(bk, False):
            raise ValueError(f"Boundary violation: {bk} is not True in P59 probe")
            
    def recursive_validate(obj: Any, path: str = ""):
        if hasattr(obj, "backward") or hasattr(obj, "shape"):
            raise TypeError(f"Tensor detected at {path}")
        if isinstance(obj, float) and not math.isfinite(obj):
            raise ValueError(f"Non-finite float at {path}: {obj}")
        if isinstance(obj, str):
            assert_no_local_path_leakage(obj, path)
            assert_no_forbidden_claims(obj, path)
        if isinstance(obj, dict):
            for k, v in obj.items():
                recursive_validate(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                recursive_validate(item, f"{path}[{i}]")
                
    recursive_validate(p59_probe, "p59_probe")
    
    return p59_probe


def compute_p60_target_balance_metrics(balance_inputs: dict) -> dict:
    per_target_diag = balance_inputs["per_target_diagnostics"]
    target_ids = balance_inputs["target_ids"]
    
    initial_total_loss_sum = sum(per_target_diag[tid]["initial_total_loss"] for tid in target_ids)
    final_total_loss_sum = sum(per_target_diag[tid]["final_total_loss"] for tid in target_ids)
    
    per_target_metrics = {}
    total_improvements = []
    recon_improvements = []
    kl_improvements = []
    
    total_decreased_count = 0
    recon_decreased_count = 0
    kl_decreased_count = 0
    
    for tid in target_ids:
        diag = per_target_diag[tid]
        
        init_tot = diag["initial_total_loss"]
        fin_tot = diag["final_total_loss"]
        tot_delta = diag["total_loss_delta"]
        
        init_rec = diag["initial_reconstruction_loss"]
        fin_rec = diag["final_reconstruction_loss"]
        rec_delta = diag["reconstruction_loss_delta"]
        
        init_kl = diag["initial_kl_loss"]
        fin_kl = diag["final_kl_loss"]
        kl_delta = diag["kl_loss_delta"]
        
        init_share = init_tot / initial_total_loss_sum if initial_total_loss_sum > 0.0 else 0.0
        fin_share = fin_tot / final_total_loss_sum if final_total_loss_sum > 0.0 else 0.0
        share_delta = fin_share - init_share
        
        abs_tot_imp = init_tot - fin_tot
        abs_rec_imp = init_rec - fin_rec
        abs_kl_imp = init_kl - fin_kl
        
        total_improvements.append(abs_tot_imp)
        recon_improvements.append(abs_rec_imp)
        kl_improvements.append(abs_kl_imp)
        
        if tot_delta < 0.0:
            total_decreased_count += 1
        if rec_delta < 0.0:
            recon_decreased_count += 1
        if kl_delta < 0.0:
            kl_decreased_count += 1
            
        per_target_metrics[tid] = {
            "initial_total_loss": float(init_tot),
            "final_total_loss": float(fin_tot),
            "total_loss_delta": float(tot_delta),
            
            "initial_reconstruction_loss": float(init_rec),
            "final_reconstruction_loss": float(fin_rec),
            "reconstruction_loss_delta": float(rec_delta),
            
            "initial_kl_loss": float(init_kl),
            "final_kl_loss": float(fin_kl),
            "kl_loss_delta": float(kl_delta),
            
            "initial_total_share": float(init_share),
            "final_total_share": float(fin_share),
            "share_delta": float(share_delta),
            
            "absolute_total_improvement": float(abs_tot_imp),
            "absolute_reconstruction_improvement": float(abs_rec_imp),
            "absolute_kl_improvement": float(abs_kl_imp),
        }
        
    all_targets_total_loss_decreased = (total_decreased_count == len(target_ids))
    all_targets_reconstruction_loss_decreased = (recon_decreased_count == len(target_ids))
    all_targets_kl_loss_decreased = (kl_decreased_count == len(target_ids))
    
    best_tot_idx = total_improvements.index(max(total_improvements))
    worst_tot_idx = total_improvements.index(min(total_improvements))
    
    best_tot_tid = target_ids[best_tot_idx]
    worst_tot_tid = target_ids[worst_tot_idx]
    
    best_tot_val = total_improvements[best_tot_idx]
    worst_tot_val = total_improvements[worst_tot_idx]
    
    total_improvement_spread = best_tot_val - worst_tot_val
    
    epsilon = 1e-8
    if worst_tot_val <= epsilon:
        total_improvement_ratio = None
        total_improvement_ratio_available = False
    else:
        total_improvement_ratio = best_tot_val / worst_tot_val
        total_improvement_ratio_available = True
        
    best_rec_idx = recon_improvements.index(max(recon_improvements))
    worst_rec_idx = recon_improvements.index(min(recon_improvements))
    
    best_rec_val = recon_improvements[best_rec_idx]
    worst_rec_val = recon_improvements[worst_rec_idx]
    
    reconstruction_improvement_spread = best_rec_val - worst_rec_val
    if worst_rec_val <= epsilon:
        reconstruction_improvement_ratio = None
        reconstruction_improvement_ratio_available = False
    else:
        reconstruction_improvement_ratio = best_rec_val / worst_rec_val
        reconstruction_improvement_ratio_available = True
        
    initial_shares = [per_target_metrics[tid]["initial_total_share"] for tid in target_ids]
    final_shares = [per_target_metrics[tid]["final_total_share"] for tid in target_ids]
    
    init_dom_idx = initial_shares.index(max(initial_shares))
    fin_dom_idx = final_shares.index(max(final_shares))
    
    init_dom_tid = target_ids[init_dom_idx]
    fin_dom_tid = target_ids[fin_dom_idx]
    
    init_dom_share = initial_shares[init_dom_idx]
    fin_dom_share = final_shares[fin_dom_idx]
    
    dominant_share_delta = fin_dom_share - init_dom_share
    
    dominance_warning = fin_dom_share >= P60_DOMINANCE_SHARE_WARNING_THRESHOLD
    dominance_blocker = fin_dom_share >= P60_DOMINANCE_SHARE_BLOCK_THRESHOLD
    
    spread_warning = total_improvement_spread >= P60_DELTA_SPREAD_WARNING_THRESHOLD
    
    ratio_warning = (
        total_improvement_ratio_available 
        and total_improvement_ratio is not None 
        and total_improvement_ratio >= P60_IMPROVEMENT_RATIO_WARNING_THRESHOLD
    )
    
    return {
        "per_target_metrics": per_target_metrics,
        "initial_total_loss_sum": float(initial_total_loss_sum),
        "final_total_loss_sum": float(final_total_loss_sum),
        
        "initial_dominant_target": init_dom_tid,
        "final_dominant_target": fin_dom_tid,
        "initial_dominant_share": float(init_dom_share),
        "final_dominant_share": float(fin_dom_share),
        "dominant_share_delta": float(dominant_share_delta),
        
        "dominance_warning": bool(dominance_warning),
        "dominance_blocker": bool(dominance_blocker),
        "spread_warning": bool(spread_warning),
        "ratio_warning": bool(ratio_warning),
        
        "best_target_by_total_improvement": best_tot_tid,
        "worst_target_by_total_improvement": worst_tot_tid,
        "best_total_improvement_value": float(best_tot_val),
        "worst_total_improvement_value": float(worst_tot_val),
        "total_improvement_spread": float(total_improvement_spread),
        "total_improvement_ratio": total_improvement_ratio,
        "total_improvement_ratio_available": bool(total_improvement_ratio_available),
        "ratio_available": bool(total_improvement_ratio_available),
        
        "reconstruction_improvement_spread": float(reconstruction_improvement_spread),
        "reconstruction_improvement_ratio": reconstruction_improvement_ratio,
        "reconstruction_improvement_ratio_available": bool(reconstruction_improvement_ratio_available),
        
        "all_targets_total_loss_decreased": bool(all_targets_total_loss_decreased),
        "all_targets_reconstruction_loss_decreased": bool(all_targets_reconstruction_loss_decreased),
        "all_targets_kl_loss_decreased": bool(all_targets_kl_loss_decreased),
        
        "target_total_loss_decreased_count": int(total_decreased_count),
        "target_reconstruction_loss_decreased_count": int(recon_decreased_count),
        "target_kl_loss_decreased_count": int(kl_decreased_count),
        
        "target_count": int(len(target_ids)),
    }


def compute_p60_counterfactual_weighting_diagnostics(balance_metrics: dict) -> dict:
    per_target = balance_metrics["per_target_metrics"]
    target_ids = list(per_target.keys())
    
    schemes = {}
    
    def build_scheme_dict(weights: Dict[str, float]) -> dict:
        weighted_init = sum(weights[tid] * per_target[tid]["initial_total_loss"] for tid in target_ids)
        weighted_final = sum(weights[tid] * per_target[tid]["final_total_loss"] for tid in target_ids)
        delta = weighted_final - weighted_init
        
        weight_values = list(weights.values())
        max_w = max(weight_values)
        min_w = min(weight_values)
        
        return {
            "weights_by_target": weights,
            "max_weight": float(max_w),
            "min_weight": float(min_w),
            "weight_spread": float(max_w - min_w),
            "weighted_initial_objective": float(weighted_init),
            "weighted_final_objective": float(weighted_final),
            "weighted_delta": float(delta),
            "weighted_loss_decreased": bool(weighted_final < weighted_init),
        }
        
    uniform_weights = {tid: 1.0 / len(target_ids) for tid in target_ids}
    schemes["uniform_current"] = build_scheme_dict(uniform_weights)
    
    raw_inv_init = {tid: 1.0 / max(per_target[tid]["initial_total_loss"], 1e-8) for tid in target_ids}
    sum_inv_init = sum(raw_inv_init.values())
    inv_init_weights = {tid: raw_inv_init[tid] / sum_inv_init for tid in target_ids}
    schemes["inverse_initial_loss_balanced"] = build_scheme_dict(inv_init_weights)
    
    raw_prop_init = {tid: per_target[tid]["initial_total_loss"] for tid in target_ids}
    sum_prop_init = sum(raw_prop_init.values())
    prop_init_weights = {tid: raw_prop_init[tid] / sum_prop_init for tid in target_ids}
    schemes["proportional_initial_loss_hard_target_emphasis"] = build_scheme_dict(prop_init_weights)
    
    raw_prop_final = {tid: per_target[tid]["final_total_loss"] for tid in target_ids}
    sum_prop_final = sum(raw_prop_final.values())
    prop_final_weights = {tid: raw_prop_final[tid] / sum_prop_final for tid in target_ids}
    schemes["proportional_final_loss_hard_target_emphasis"] = build_scheme_dict(prop_final_weights)
    
    return schemes


def derive_p60_balance_recommendation(balance_metrics: dict, weighting_diagnostics: dict) -> dict:
    per_target = balance_metrics["per_target_metrics"]
    target_ids = list(per_target.keys())
    
    dominance_blocker = balance_metrics["dominance_blocker"]
    dominance_warning = balance_metrics["dominance_warning"]
    spread_warning = balance_metrics["spread_warning"]
    ratio_warning = balance_metrics["ratio_warning"]
    
    all_tot_decreased = balance_metrics["all_targets_total_loss_decreased"]
    all_rec_decreased = balance_metrics["all_targets_reconstruction_loss_decreased"]
    worst_tot_imp = balance_metrics["worst_total_improvement_value"]
    
    epsilon = 1e-8
    
    blocking_issue_found = False
    weighting_required_now = False
    
    if (
        dominance_blocker
        or not all_tot_decreased
        or not all_rec_decreased
        or worst_tot_imp <= epsilon
    ):
        status = P60_RECOMMENDATION_STATUS_WEIGHTING_REQUIRED_BEFORE_EXPANSION
        blocking_issue_found = True
        weighting_required_now = True
        reason = "Objective dominance blocker triggered, or one or more targets failed to show improvement."
    elif (
        dominance_warning
        or spread_warning
        or ratio_warning
    ):
        status = P60_RECOMMENDATION_STATUS_WEIGHTING_WATCH
        reason = "All targets improved and no blocker triggered, but warning thresholds for share dominance, improvement spread, or ratio were exceeded."
    else:
        status = P60_RECOMMENDATION_STATUS_NO_WEIGHTING_REQUIRED
        reason = "Uniform objective weighting remains balanced and all targets show stable reconstruction improvement."
        
    safe_to_continue = not weighting_required_now
    
    watch_targets = [
        tid for tid in target_ids 
        if per_target[tid]["final_total_share"] >= P60_DOMINANCE_SHARE_WARNING_THRESHOLD
    ]
    
    hardest_target_id = balance_metrics["final_dominant_target"]
    most_improved_target_id = balance_metrics["best_target_by_total_improvement"]
    least_improved_target_id = balance_metrics["worst_target_by_total_improvement"]
    
    return {
        "recommendation_status": status,
        "recommendation_reason": reason,
        "blocking_issue_found": bool(blocking_issue_found),
        "weighting_required_now": bool(weighting_required_now),
        "safe_to_continue_with_uniform_objective_next_phase": bool(safe_to_continue),
        "watch_targets": watch_targets,
        "hardest_target_id": hardest_target_id,
        "most_improved_target_id": most_improved_target_id,
        "least_improved_target_id": least_improved_target_id,
    }


def run_fc_vae_target_balance_objective_audit_probe() -> dict:
    try:
        torch = load_torch_for_p60_balance_audit()
    except Exception:
        return {
            "contract_version": FC_VAE_TARGET_BALANCE_AUDIT_CONTRACT_VERSION,
            "status": FC_VAE_TARGET_BALANCE_AUDIT_STATUS_BLOCKED,
            "torch_available": False,
            "reason": "torch_unavailable",
        }
        
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    
    try:
        balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    except Exception as e:
        return {
            "contract_version": FC_VAE_TARGET_BALANCE_AUDIT_CONTRACT_VERSION,
            "status": FC_VAE_TARGET_BALANCE_AUDIT_STATUS_BLOCKED,
            "torch_available": True,
            "reason": f"p59_evidence_validation_failed: {str(e)}",
            "verdict": "FAIL",
        }
        
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    weighting_diagnostics = compute_p60_counterfactual_weighting_diagnostics(metrics)
    recommendation = derive_p60_balance_recommendation(metrics, weighting_diagnostics)
    
    passed = (
        p59_probe.get("verdict") == "PASS"
        and len(metrics["per_target_metrics"]) == 3
        and recommendation is not None
    )
    
    summary = {
        "contract_version": FC_VAE_TARGET_BALANCE_AUDIT_CONTRACT_VERSION,
        "kind": FC_VAE_TARGET_BALANCE_AUDIT_KIND,
        "status": FC_VAE_TARGET_BALANCE_AUDIT_STATUS_AVAILABLE,
        "torch_available": True,
        "source_phase": "P60",
        "source_evidence_phase": "P59",
        "source_p59_status": p59_probe.get("status"),
        "source_p59_verdict": p59_probe.get("verdict"),
        
        "target_count": P60_TARGET_COUNT,
        "target_ids": list(P60_EXPECTED_TARGET_IDS),
        
        "balance_metrics": metrics,
        "counterfactual_weighting_diagnostics": weighting_diagnostics,
        "recommendation": recommendation,
        
        "dominance_warning_threshold": P60_DOMINANCE_SHARE_WARNING_THRESHOLD,
        "dominance_block_threshold": P60_DOMINANCE_SHARE_BLOCK_THRESHOLD,
        "delta_spread_warning_threshold": P60_DELTA_SPREAD_WARNING_THRESHOLD,
        "improvement_ratio_warning_threshold": P60_IMPROVEMENT_RATIO_WARNING_THRESHOLD,
        
        # Boundaries
        "no_training_change": True,
        "no_optimizer_created_in_p60": True,
        "no_weighted_training_applied": True,
        
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
        
        "realizability_claim": "target_balance_objective_audit_only_no_training_change_no_dataset",
        "reason": "fc_vae_target_balance_objective_audit_probe_success" if passed else "fc_vae_target_balance_objective_audit_probe_failed",
        "verdict": "PASS" if passed else "FAIL",
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    
    return summary


def fc_vae_target_balance_objective_audit_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_target_balance_objective_audit_json(probe_res: dict) -> str:
    d = fc_vae_target_balance_objective_audit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
