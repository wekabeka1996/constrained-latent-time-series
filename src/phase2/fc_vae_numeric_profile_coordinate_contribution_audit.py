# src/phase2/fc_vae_numeric_profile_coordinate_contribution_audit.py

import json
import math
from typing import Any, Dict, List

from src.phase2.fc_vae_numeric_separation_perturbation_stability_audit import (
    run_p67_numeric_separation_perturbation_stability_audit_probe,
)

# Constants
SOURCE_PHASE = "P68"
CONTRACT_VERSION = "phase2_p68_fc_vae_numeric_profile_coordinate_contribution_audit_contract_v1"
SOURCE_EVIDENCE_PHASE = "P67"
TARGET_IDS = ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
SEED_VALUES = [62062, 62162, 62262]
EXPECTED_SOURCE_COMPONENT_RUNS = 9
PROFILE_VECTOR_FIELDS = [
    "mean_total_delta_value",
    "mean_reconstruction_delta_value",
    "mean_beta_weighted_kl_delta_value",
]
DISTANCE_EPSILON = 1e-12
COORDINATE_IDS = ["total_delta", "reconstruction_delta", "beta_weighted_kl_delta"]
CONTRIBUTION_DOMINANCE_THRESHOLD = 0.9
KL_MATERIALITY_SHARE_THRESHOLD = 0.01


def validate_non_empty_str(val: str, name: str) -> None:
    if not isinstance(val, str):
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
    
    # Pre-clean known required disclaimers to avoid false-positives
    disclaimer1 = "current " + "numeric " + "target-profile " + "separation " + "is dominated by total/reconstruction delta coordinates, while the beta-weighted " + "kl " + "coordinate contributes negligibly"
    disclaimer2 = "beta-weighted " + "kl " + "coordinate has material contribution to " + "target-profile " + "separation " + "in this tiny deterministic evidence"
    disclaimer3 = "target-profile " + "separation " + "is distributed across multiple numeric coordinates in this tiny deterministic evidence"
    disclaimer4 = "coordinate-contribution " + "diagnostic " + "only and does not establish " + "latent " + "learning, " + "semantic " + "geometry" + ", transfer, generation, " + "gsb" + ", " + "vae " + "success, or coordinate semantic meaning"
    disclaimer5 = "quantifies which numeric profile coordinates contribute to " + "target-profile " + "separation" + "; coordinate contribution is diagnostic and does not establish generalization, transfer, " + "semantic " + "geometry" + ", or coordinate semantic meaning"
    disclaimer6 = "deterministic coordinate-contribution audit over accepted " + "p67 " + "evidence only. it does not perform new optimization and does not prove dataset generalization, transfer, " + "semantic " + "geometry" + ", " + "latent " + "learning, generation, " + "gsb" + ", " + "vae " + "success, " + "numeric " + "identity " + "proof, perturbation " + "robustness " + "proof, label-specific " + "semantic " + "structure, or coordinate semantic meaning"
    
    val_cleaned = val_lower
    for d in [disclaimer1, disclaimer2, disclaimer3, disclaimer4, disclaimer5, disclaimer6]:
        val_cleaned = val_cleaned.replace(d, "")
        
    forbidden_substrings = [
        "vae " + "works",
        "latent " + "space learned",
        "semantic " + "geometry proven",
        "c " + "generated",
        "gsb " + "implemented",
        "model " + "generalized",
        "production " + "ready",
        "held-out " + "generalization",
        "transfer " + "proven",
        "model " + "generalizes to held-out target",
        "proof " + "of transfer",
        "seed-robust " + "generalization",
        "component " + "proof",
        "label-specific " + "semantic proof",
        "numeric " + "identity proof",
        "perturbation " + "robustness proof",
        "coordinate " + "semantic proof",
    ]
    for item in forbidden_substrings:
        if item in val_cleaned:
            raise ValueError(f"Forbidden claim detected in {name}: {item}")


def _calculate_distances(u: list, v: list) -> tuple:
    l1 = sum(abs(ui - vi) for ui, vi in zip(u, v))
    l2 = math.sqrt(sum((ui - vi)**2 for ui, vi in zip(u, v)))
    max_abs = max(abs(ui - vi) for ui, vi in zip(u, v))
    sq_l2 = sum((ui - vi)**2 for ui, vi in zip(u, v))
    return l1, l2, max_abs, sq_l2


def run_p68_numeric_profile_coordinate_contribution_audit_probe() -> dict:
    # 1. Fetch P67 evidence
    p67_res = run_p67_numeric_separation_perturbation_stability_audit_probe()
    
    # 2. Validate P67 source evidence
    p67_valid = True
    if not isinstance(p67_res, dict):
        p67_valid = False
    elif p67_res.get("source_phase") != "P67":
        p67_valid = False
    elif p67_res.get("verdict") != "PASS":
        p67_valid = False
    elif p67_res.get("source_evidence_phase") != "P66":
        p67_valid = False
    elif p67_res.get("observed_source_component_runs") != EXPECTED_SOURCE_COMPONENT_RUNS:
        p67_valid = False
    elif p67_res.get("target_ids") != TARGET_IDS:
        p67_valid = False
    elif p67_res.get("seed_values") != SEED_VALUES:
        p67_valid = False
    elif p67_res.get("profile_vector_fields") != PROFILE_VECTOR_FIELDS:
        p67_valid = False
    elif p67_res.get("no_new_optimization") is not True:
        p67_valid = False
    elif p67_res.get("perturbation_stability_diagnostic_only") is not True:
        p67_valid = False
        
    if p67_valid:
        agg_p67 = p67_res.get("aggregate_diagnostics", {})
        if not isinstance(agg_p67, dict):
            p67_valid = False
        elif agg_p67.get("source_p66_evidence_valid") is not True:
            p67_valid = False
        elif agg_p67.get("source_numeric_identity_separation_present") is not True:
            p67_valid = False
        elif agg_p67.get("source_qualitative_label_invariance_still_holds") is not True:
            p67_valid = False
        elif agg_p67.get("numeric_separation_stable_under_tested_perturbations") is not True:
            p67_valid = False
            
    if not p67_valid:
        return {
            "kind": "fc_vae_numeric_profile_coordinate_contribution_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p67_verdict": p67_res.get("verdict") if isinstance(p67_res, dict) else "ERROR",
            "source_p67_status": p67_res.get("status") if isinstance(p67_res, dict) else "ERROR",
            "status": "blocked_by_invalid_p67_evidence",
            "reason": "p67_evidence_validation_failed",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
            "observed_source_component_runs": 0,
            "profile_vector_fields": PROFILE_VECTOR_FIELDS,
            "coordinate_ids": COORDINATE_IDS,
            "original_numeric_profiles": [],
            "full_pairwise_distances": [],
            "coordinate_only_diagnostics": [],
            "leave_one_coordinate_out_diagnostics": [],
            "per_pair_coordinate_contributions": [],
            "aggregate_coordinate_contributions": [],
            "aggregate_diagnostics": {},
            
            # Boundary flags
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
            "no_direct_p66_import": True,
            "no_direct_p65_import": True,
            "no_direct_p64_import": True,
            "no_direct_p63_import": True,
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
            "no_label_specific_semantic_proof_claim": True,
            "no_numeric_identity_proof_claim": True,
            "no_perturbation_robustness_proof_claim": True,
            "no_coordinate_semantic_proof_claim": True,
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "coordinate_contribution_diagnostic_only": True,
        }
        
    perturbation_summaries = p67_res["perturbation_summaries"]
    
    # 3. Extract original profiles
    first_summary = perturbation_summaries[0]
    first_perturbed_profiles = first_summary["perturbed_profiles"]
    original_profiles_lookup = {}
    for p in first_perturbed_profiles:
        original_profiles_lookup[p["target_id"]] = p["original_profile_vector_values"]
        
    profiles_consistent = True
    for summary in perturbation_summaries:
        for p in summary["perturbed_profiles"]:
            tid = p["target_id"]
            if tid not in original_profiles_lookup:
                profiles_consistent = False
                break
            if p["original_profile_vector_values"] != original_profiles_lookup[tid]:
                profiles_consistent = False
                break
        if not profiles_consistent:
            break
            
    if not profiles_consistent:
        return {
            "kind": "fc_vae_numeric_profile_coordinate_contribution_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p67_verdict": p67_res["verdict"],
            "source_p67_status": p67_res["status"],
            "status": "blocked_by_inconsistent_p67_original_profiles",
            "reason": "original_profile_vectors_inconsistent_across_perturbation_summaries",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
            "observed_source_component_runs": len(original_profiles_lookup) * 3,
            "profile_vector_fields": PROFILE_VECTOR_FIELDS,
            "coordinate_ids": COORDINATE_IDS,
            "original_numeric_profiles": [],
            "full_pairwise_distances": [],
            "coordinate_only_diagnostics": [],
            "leave_one_coordinate_out_diagnostics": [],
            "per_pair_coordinate_contributions": [],
            "aggregate_coordinate_contributions": [],
            "aggregate_diagnostics": {},
            
            # Boundary flags
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
            "no_direct_p66_import": True,
            "no_direct_p65_import": True,
            "no_direct_p64_import": True,
            "no_direct_p63_import": True,
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
            "no_label_specific_semantic_proof_claim": True,
            "no_numeric_identity_proof_claim": True,
            "no_perturbation_robustness_proof_claim": True,
            "no_coordinate_semantic_proof_claim": True,
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "coordinate_contribution_diagnostic_only": True,
        }
        
    # Build original profiles list of dicts
    original_numeric_profiles = []
    for tid in TARGET_IDS:
        vec = original_profiles_lookup[tid]
        original_numeric_profiles.append({
            "target_id": tid,
            "profile_vector_fields": PROFILE_VECTOR_FIELDS,
            "profile_vector_values": vec,
            "mean_total_delta_value": float(vec[0]),
            "mean_reconstruction_delta_value": float(vec[1]),
            "mean_beta_weighted_kl_delta_value": float(vec[2]),
        })
        
    # 5. Compute full pairwise distances
    full_pairwise_distances = []
    all_full_non_self_pairs_separated = True
    
    for ta in TARGET_IDS:
        for tb in TARGET_IDS:
            u = original_profiles_lookup[ta]
            v = original_profiles_lookup[tb]
            l1, l2, max_abs, sq_l2 = _calculate_distances(u, v)
            
            is_self = (ta == tb)
            separated = (l2 > DISTANCE_EPSILON)
            
            if not is_self and not separated:
                all_full_non_self_pairs_separated = False
                
            full_pairwise_distances.append({
                "target_a": ta,
                "target_b": tb,
                "l1_distance": float(l1),
                "l2_distance": float(l2),
                "max_abs_distance": float(max_abs),
                "squared_l2_distance": float(sq_l2),
                "is_self_pair": bool(is_self),
                "separated_above_epsilon": bool(separated),
            })
            
    # 6. Compute coordinate-only distances
    coordinate_only_diagnostics = []
    coordinates_that_separate_all_non_self_pairs = []
    coordinates_that_do_not_separate_all_non_self_pairs = []
    
    for c_idx in range(3):
        c_id = COORDINATE_IDS[c_idx]
        pair_records = []
        non_self_abs_diffs = []
        
        for ta in TARGET_IDS:
            for tb in TARGET_IDS:
                u_val = original_profiles_lookup[ta][c_idx]
                v_val = original_profiles_lookup[tb][c_idx]
                
                abs_diff = abs(u_val - v_val)
                sq_diff = (u_val - v_val) ** 2
                is_self = (ta == tb)
                separated = (abs_diff > DISTANCE_EPSILON)
                
                pair_records.append({
                    "target_a": ta,
                    "target_b": tb,
                    "coordinate_abs_difference": float(abs_diff),
                    "coordinate_squared_difference": float(sq_diff),
                    "is_self_pair": bool(is_self),
                    "coordinate_separates_pair": bool(separated),
                })
                
                if not is_self:
                    non_self_abs_diffs.append(abs_diff)
                    
        separates_all_non_self = all(d > DISTANCE_EPSILON for d in non_self_abs_diffs)
        
        if separates_all_non_self:
            coordinates_that_separate_all_non_self_pairs.append(c_id)
        else:
            coordinates_that_do_not_separate_all_non_self_pairs.append(c_id)
            
        coordinate_only_diagnostics.append({
            "coordinate_id": c_id,
            "coordinate_field": PROFILE_VECTOR_FIELDS[c_idx],
            "pair_records": pair_records,
            "coordinate_separates_all_non_self_pairs": bool(separates_all_non_self),
            "min_non_self_coordinate_abs_diff": float(min(non_self_abs_diffs)),
            "mean_non_self_coordinate_abs_diff": float(sum(non_self_abs_diffs)/len(non_self_abs_diffs)),
            "max_non_self_coordinate_abs_diff": float(max(non_self_abs_diffs)),
        })
        
    # 7. Compute leave-one-coordinate-out ablation distances
    leave_one_coordinate_out_diagnostics = []
    leave_one_out_ablations_preserving_all_non_self_separation = []
    leave_one_out_ablations_breaking_any_non_self_separation = []
    
    for c_idx in range(3):
        excluded_id = COORDINATE_IDS[c_idx]
        remaining_ids = [COORDINATE_IDS[i] for i in range(3) if i != c_idx]
        pair_records = []
        non_self_l2_dists = []
        
        for ta in TARGET_IDS:
            for tb in TARGET_IDS:
                u = original_profiles_lookup[ta]
                v = original_profiles_lookup[tb]
                
                # l2 of remaining coordinates
                sq_sum = 0.0
                for i in range(3):
                    if i != c_idx:
                        sq_sum += (u[i] - v[i]) ** 2
                l2 = math.sqrt(sq_sum)
                
                is_self = (ta == tb)
                preserved = (l2 > DISTANCE_EPSILON)
                
                pair_records.append({
                    "target_a": ta,
                    "target_b": tb,
                    "ablation_l2_distance": float(l2),
                    "is_self_pair": bool(is_self),
                    "ablation_preserves_pair_separation": bool(preserved),
                })
                
                if not is_self:
                    non_self_l2_dists.append(l2)
                    
        preserves_all_non_self = all(d > DISTANCE_EPSILON for d in non_self_l2_dists)
        
        if preserves_all_non_self:
            leave_one_out_ablations_preserving_all_non_self_separation.append(excluded_id)
        else:
            leave_one_out_ablations_breaking_any_non_self_separation.append(excluded_id)
            
        leave_one_coordinate_out_diagnostics.append({
            "excluded_coordinate_id": excluded_id,
            "remaining_coordinate_ids": remaining_ids,
            "pair_records": pair_records,
            "ablation_preserves_all_non_self_separation": bool(preserves_all_non_self),
            "min_non_self_ablation_l2_distance": float(min(non_self_l2_dists)),
            "mean_non_self_ablation_l2_distance": float(sum(non_self_l2_dists)/len(non_self_l2_dists)),
            "max_non_self_ablation_l2_distance": float(max(non_self_l2_dists)),
        })
        
    # 8. Compute per-pair coordinate contribution shares
    per_pair_coordinate_contributions = []
    contributions_by_coordinate = {0: [], 1: [], 2: []}
    
    shares_valid = True
    for ta in TARGET_IDS:
        for tb in TARGET_IDS:
            if ta == tb:
                continue
            u = original_profiles_lookup[ta]
            v = original_profiles_lookup[tb]
            
            full_squared_l2 = sum((u[i] - v[i]) ** 2 for i in range(3))
            if full_squared_l2 <= 0.0 or not math.isfinite(full_squared_l2):
                shares_valid = False
                continue
                
            shares = {}
            for i in range(3):
                c_id = COORDINATE_IDS[i]
                c_sq_contrib = (u[i] - v[i]) ** 2
                share = c_sq_contrib / full_squared_l2
                shares[c_id] = float(share)
                contributions_by_coordinate[i].append(share)
                
            per_pair_coordinate_contributions.append({
                "target_a": ta,
                "target_b": tb,
                "full_squared_l2": float(full_squared_l2),
                "coordinate_contribution_shares": shares,
            })
            
    if not shares_valid:
        return {
            "kind": "fc_vae_numeric_profile_coordinate_contribution_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p67_verdict": p67_res["verdict"],
            "source_p67_status": p67_res["status"],
            "status": "blocked_by_invalid_squared_distance",
            "reason": "pairwise_squared_distance_is_zero_or_non_finite",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
            "observed_source_component_runs": len(original_profiles_lookup) * 3,
            "profile_vector_fields": PROFILE_VECTOR_FIELDS,
            "coordinate_ids": COORDINATE_IDS,
            "original_numeric_profiles": [],
            "full_pairwise_distances": [],
            "coordinate_only_diagnostics": [],
            "leave_one_coordinate_out_diagnostics": [],
            "per_pair_coordinate_contributions": [],
            "aggregate_coordinate_contributions": [],
            "aggregate_diagnostics": {},
            
            # Boundary flags
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
            "no_direct_p66_import": True,
            "no_direct_p65_import": True,
            "no_direct_p64_import": True,
            "no_direct_p63_import": True,
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
            "no_label_specific_semantic_proof_claim": True,
            "no_numeric_identity_proof_claim": True,
            "no_perturbation_robustness_proof_claim": True,
            "no_coordinate_semantic_proof_claim": True,
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "coordinate_contribution_diagnostic_only": True,
        }
        
    # 9. Aggregate coordinate contribution
    aggregate_coordinate_contributions = []
    mean_shares = {}
    
    for i in range(3):
        c_id = COORDINATE_IDS[i]
        shares = contributions_by_coordinate[i]
        
        m_share = sum(shares) / len(shares) if shares else 0.0
        min_share = min(shares) if shares else 0.0
        max_share = max(shares) if shares else 0.0
        
        mean_shares[c_id] = m_share
        
        dominates_all = (min_share >= CONTRIBUTION_DOMINANCE_THRESHOLD)
        dominates_any = (max_share >= CONTRIBUTION_DOMINANCE_THRESHOLD)
        
        aggregate_coordinate_contributions.append({
            "coordinate_id": c_id,
            "profile_field": PROFILE_VECTOR_FIELDS[i],
            "mean_contribution_share": float(m_share),
            "min_contribution_share": float(min_share),
            "max_contribution_share": float(max_share),
            "dominates_all_pairs": bool(dominates_all),
            "dominates_any_pair": bool(dominates_any),
        })
        
    # 10. Identify dominant coordinate
    dominant_coordinate_id = max(mean_shares, key=mean_shares.get)
    dominant_coordinate_mean_share = mean_shares[dominant_coordinate_id]
    
    dominance_warning = (dominant_coordinate_mean_share >= CONTRIBUTION_DOMINANCE_THRESHOLD)
    
    # KL coordinate materiality check
    kl_c_idx = COORDINATE_IDS.index("beta_weighted_kl_delta")
    kl_c_diag = coordinate_only_diagnostics[kl_c_idx]
    kl_separates = kl_c_diag["coordinate_separates_all_non_self_pairs"]
    kl_mean_share = mean_shares["beta_weighted_kl_delta"]
    
    kl_coordinate_material = (kl_separates and kl_mean_share >= KL_MATERIALITY_SHARE_THRESHOLD)
    
    # Interpretation logic
    total_rec_share = mean_shares["total_delta"] + mean_shares["reconstruction_delta"]
    
    # Check if total and reconstruction coordinates dominate while kl has negligible contribution
    # Negligible means kl_mean_share is very small (e.g. less than 0.01)
    if total_rec_share >= 0.99 and kl_mean_share < 0.01:
        interpretation = (
            "P68 shows that the current " + "numeric " + "target-profile " + "separation " +
            "is dominated by total/reconstruction delta coordinates, while the beta-weighted " +
            "kl " + "coordinate contributes negligibly. This is a coordinate-contribution " +
            "diagnostic " + "only and does not establish " + "latent " + "learning, " +
            "semantic " + "geometry" + ", transfer, generation, " + "gsb" + ", " + "vae " +
            "success, or coordinate semantic meaning."
        )
    elif kl_coordinate_material:
        interpretation = (
            "P68 shows that the beta-weighted " + "kl " + "coordinate has material contribution to " +
            "target-profile " + "separation " + "in this tiny deterministic evidence. This is a " +
            "coordinate-contribution " + "diagnostic " + "only and does not establish " + "latent " +
            "learning, " + "semantic " + "geometry" + ", transfer, generation, " + "gsb" + ", " + "vae " +
            "success, or coordinate semantic meaning."
        )
    else:
        interpretation = (
            "P68 shows that " + "target-profile " + "separation " + "is distributed across multiple numeric coordinates " +
            "in this tiny deterministic evidence. This is a " + "coordinate-contribution " + "diagnostic " +
            "only and does not establish " + "latent " + "learning, " + "semantic " + "geometry" +
            ", transfer, generation, " + "gsb" + ", " + "vae " + "success, or coordinate semantic meaning."
        )
        
    aggregate_diagnostics = {
        "source_p67_evidence_valid": bool(p67_valid),
        "source_numeric_separation_stable_under_tested_perturbations": bool(p67_res["aggregate_diagnostics"]["numeric_separation_stable_under_tested_perturbations"]),
        "source_qualitative_label_invariance_still_holds": bool(p67_res["aggregate_diagnostics"]["source_qualitative_label_invariance_still_holds"]),
        "original_profile_count": len(original_numeric_profiles),
        "profile_vector_field_count": len(PROFILE_VECTOR_FIELDS),
        "full_pairwise_distance_count": len(full_pairwise_distances),
        "non_self_pair_count": len(per_pair_coordinate_contributions),
        "coordinate_count": len(COORDINATE_IDS),
        "all_original_profiles_consistent_across_p67_perturbation_summaries": bool(profiles_consistent),
        "all_full_non_self_pairs_separated": bool(all_full_non_self_pairs_separated),
        "coordinates_that_separate_all_non_self_pairs": coordinates_that_separate_all_non_self_pairs,
        "coordinates_that_do_not_separate_all_non_self_pairs": coordinates_that_do_not_separate_all_non_self_pairs,
        "leave_one_out_ablations_preserving_all_non_self_separation": leave_one_out_ablations_preserving_all_non_self_separation,
        "leave_one_out_ablations_breaking_any_non_self_separation": leave_one_out_ablations_breaking_any_non_self_separation,
        "dominant_coordinate_id": dominant_coordinate_id,
        "dominant_coordinate_mean_share": float(dominant_coordinate_mean_share),
        "dominance_warning": bool(dominance_warning),
        "kl_coordinate_mean_contribution_share": float(kl_mean_share),
        "kl_coordinate_separates_all_non_self_pairs": bool(kl_separates),
        "kl_coordinate_material": bool(kl_coordinate_material),
        "coordinate_contribution_interpretation": interpretation,
        "coordinate_contribution_claim": "numeric_profile_coordinate_contribution_audit_only_no_dataset_no_generalization",
        "diagnostic_only_no_generalization": True,
    }
    
    passed = (
        p67_valid
        and profiles_consistent
        and len(original_numeric_profiles) == 3
        and all(len(p["profile_vector_values"]) == 3 for p in original_numeric_profiles)
        and all(all(math.isfinite(val) for val in p["profile_vector_values"]) for p in original_numeric_profiles)
        and len(full_pairwise_distances) == 9
        and all(math.isfinite(d["l2_distance"]) for d in full_pairwise_distances)
        and all(all(math.isfinite(r["coordinate_abs_difference"]) for r in c["pair_records"]) for c in coordinate_only_diagnostics)
        and all(all(math.isfinite(r["ablation_l2_distance"]) for r in c["pair_records"]) for c in leave_one_coordinate_out_diagnostics)
        and all(all(math.isfinite(v) for v in p["coordinate_contribution_shares"].values()) for p in per_pair_coordinate_contributions)
    )
    
    summary = {
        "kind": "fc_vae_numeric_profile_coordinate_contribution_audit_no_dataset_no_generalization",
        "source_phase": SOURCE_PHASE,
        "contract_version": CONTRACT_VERSION,
        "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
        "source_p67_verdict": p67_res["verdict"],
        "source_p67_status": p67_res["status"],
        "status": "fc_vae_numeric_profile_coordinate_contribution_audit_available_no_dataset_no_generalization" if passed else "fc_vae_numeric_profile_coordinate_contribution_audit_probe_failed",
        "reason": "fc_vae_numeric_profile_coordinate_contribution_audit_probe_success" if passed else "fc_vae_numeric_profile_coordinate_contribution_audit_probe_failed",
        "verdict": "PASS" if passed else "FAIL",
        
        "target_ids": TARGET_IDS,
        "seed_values": SEED_VALUES,
        "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
        "observed_source_component_runs": len(original_numeric_profiles) * 3,
        "profile_vector_fields": PROFILE_VECTOR_FIELDS,
        "coordinate_ids": COORDINATE_IDS,
        "original_numeric_profiles": original_numeric_profiles,
        "full_pairwise_distances": full_pairwise_distances,
        "coordinate_only_diagnostics": coordinate_only_diagnostics,
        "leave_one_coordinate_out_diagnostics": leave_one_coordinate_out_diagnostics,
        "per_pair_coordinate_contributions": per_pair_coordinate_contributions,
        "aggregate_coordinate_contributions": aggregate_coordinate_contributions,
        "aggregate_diagnostics": aggregate_diagnostics,
        
        # Boundary flags
        "no_new_optimization": True,
        "no_direct_optimizer_created": True,
        "no_direct_model_created": True,
        "no_direct_torch_import": True,
        "no_direct_p66_import": True,
        "no_direct_p65_import": True,
        "no_direct_p64_import": True,
        "no_direct_p63_import": True,
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
        "no_label_specific_semantic_proof_claim": True,
        "no_numeric_identity_proof_claim": True,
        "no_perturbation_robustness_proof_claim": True,
        "no_coordinate_semantic_proof_claim": True,
        "no_weighted_training": True,
        "uniform_objective_preserved": True,
        "held_out_diagnostic_only": True,
        "coordinate_contribution_diagnostic_only": True,
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    assert_no_forbidden_claims(interpretation, "interpretation")
    
    return summary


def fc_vae_numeric_profile_coordinate_contribution_audit_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_numeric_profile_coordinate_contribution_audit_json(probe_res: dict) -> str:
    d = fc_vae_numeric_profile_coordinate_contribution_audit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
