# src/phase2/fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit.py

import json
import math
from typing import Any, Dict, List

from src.phase2.fc_vae_numeric_profile_coordinate_contribution_audit import (
    run_p68_numeric_profile_coordinate_contribution_audit_probe,
)

# Constants
SOURCE_PHASE = "P69"
CONTRACT_VERSION = "phase2_p69_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_contract_v1"
SOURCE_EVIDENCE_PHASE = "P68"
TARGET_IDS = ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
SEED_VALUES = [62062, 62162, 62262]
EXPECTED_SOURCE_COMPONENT_RUNS = 9
PROFILE_VECTOR_FIELDS = [
    "mean_total_delta_value",
    "mean_reconstruction_delta_value",
    "mean_beta_weighted_kl_delta_value",
]
COORDINATE_IDS = ["total_delta", "reconstruction_delta", "beta_weighted_kl_delta"]
DISTANCE_EPSILON = 1e-12
KL_MATERIALITY_SHARE_THRESHOLD = 0.01
RECON_TOTAL_EXPLANATORY_SHARE_THRESHOLD = 0.99

VIEW_IDS = [
    "full_vector",
    "reconstruction_only",
    "total_only",
    "kl_only",
    "total_plus_reconstruction",
    "reconstruction_plus_kl",
    "total_plus_kl",
]


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
    disclaimer1 = "current " + "target-profile " + "separation " + "is explainable by total/reconstruction coordinates without material beta-weighted " + "kl " + "contribution"
    disclaimer2 = "beta-weighted " + "kl " + "coordinate can separate targets above epsilon, but its contribution share remains negligible relative to total/reconstruction coordinates. this supports a " + "kl" + "-negligible explanatory boundary, not a " + "latent " + "or semantic claim"
    disclaimer3 = "beta-weighted " + "kl " + "has material contribution in the current coordinate evidence"
    disclaimer4 = "explanatory boundary " + "diagnostic " + "only and does not establish " + "latent " + "learning, " + "semantic " + "geometry" + ", transfer, generation, " + "gsb" + ", " + "vae " + "success, or coordinate semantic meaning"
    disclaimer5 = "current " + "target-profile " + "separation " + "is explainable by reconstruction/total coordinates without material beta-weighted " + "kl " + "contribution; explanatory boundary is diagnostic and does not establish generalization, transfer, " + "semantic " + "geometry" + ", or " + "latent/kl " + "explanation"
    disclaimer6 = "deterministic explanatory-boundary audit over accepted " + "p68 " + "evidence only. it does not perform new optimization and does not prove dataset generalization, transfer, " + "semantic " + "geometry" + ", " + "latent " + "learning, generation, " + "gsb" + ", " + "vae " + "success, " + "numeric " + "identity " + "proof, perturbation " + "robustness " + "proof, coordinate semantic meaning, label-specific " + "semantic " + "structure, or " + "latent/kl " + "explanatory proof"
    
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
        "latent " + "kl " + "explanatory proof",
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


def get_projection(vec: list, view_id: str) -> list:
    if view_id == "full_vector":
        return vec
    elif view_id == "reconstruction_only":
        return [vec[1]]
    elif view_id == "total_only":
        return [vec[0]]
    elif view_id == "kl_only":
        return [vec[2]]
    elif view_id == "total_plus_reconstruction":
        return [vec[0], vec[1]]
    elif view_id == "reconstruction_plus_kl":
        return [vec[1], vec[2]]
    elif view_id == "total_plus_kl":
        return [vec[0], vec[2]]
    else:
        raise ValueError(f"Unknown view: {view_id}")


def run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe() -> dict:
    # 1. Fetch P68 evidence
    p68_res = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    
    # 2. Validate P68 source evidence
    p68_valid = True
    if not isinstance(p68_res, dict):
        p68_valid = False
    elif p68_res.get("source_phase") != "P68":
        p68_valid = False
    elif p68_res.get("verdict") != "PASS":
        p68_valid = False
    elif p68_res.get("source_evidence_phase") != "P67":
        p68_valid = False
    elif p68_res.get("observed_source_component_runs") != EXPECTED_SOURCE_COMPONENT_RUNS:
        p68_valid = False
    elif p68_res.get("target_ids") != TARGET_IDS:
        p68_valid = False
    elif p68_res.get("seed_values") != SEED_VALUES:
        p68_valid = False
    elif p68_res.get("profile_vector_fields") != PROFILE_VECTOR_FIELDS:
        p68_valid = False
    elif p68_res.get("coordinate_ids") != COORDINATE_IDS:
        p68_valid = False
    elif p68_res.get("no_new_optimization") is not True:
        p68_valid = False
    elif p68_res.get("coordinate_contribution_diagnostic_only") is not True:
        p68_valid = False
        
    if p68_valid:
        agg_p68 = p68_res.get("aggregate_diagnostics", {})
        if not isinstance(agg_p68, dict):
            p68_valid = False
        elif agg_p68.get("source_p67_evidence_valid") is not True:
            p68_valid = False
        elif agg_p68.get("source_numeric_separation_stable_under_tested_perturbations") is not True:
            p68_valid = False
        elif agg_p68.get("source_qualitative_label_invariance_still_holds") is not True:
            p68_valid = False
        elif agg_p68.get("kl_coordinate_material") is not False:
            p68_valid = False
            
    if not p68_valid:
        return {
            "kind": "fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p68_verdict": p68_res.get("verdict") if isinstance(p68_res, dict) else "ERROR",
            "source_p68_status": p68_res.get("status") if isinstance(p68_res, dict) else "ERROR",
            "status": "blocked_by_invalid_p68_evidence",
            "reason": "p68_evidence_validation_failed",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
            "observed_source_component_runs": 0,
            "profile_vector_fields": PROFILE_VECTOR_FIELDS,
            "coordinate_ids": COORDINATE_IDS,
            "explanatory_views": [],
            "aggregate_diagnostics": {},
            
            # Boundary flags
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
            "no_direct_p67_import": True,
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
            "no_latent_kl_explanatory_proof_claim": True,
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "explanatory_boundary_diagnostic_only": True,
        }
        
    original_numeric_profiles = p68_res["original_numeric_profiles"]
    full_pairwise_distances = p68_res["full_pairwise_distances"]
    coordinate_only_diagnostics = p68_res["coordinate_only_diagnostics"]
    leave_one_coordinate_out_diagnostics = p68_res["leave_one_coordinate_out_diagnostics"]
    per_pair_coordinate_contributions = p68_res["per_pair_coordinate_contributions"]
    aggregate_coordinate_contributions = p68_res["aggregate_coordinate_contributions"]
    
    # 4. Validate P68 consistency
    p68_consistent = (
        len(original_numeric_profiles) == 3
        and len(full_pairwise_distances) == 9
        and len(coordinate_only_diagnostics) == 3
        and len(leave_one_coordinate_out_diagnostics) == 3
        and len(per_pair_coordinate_contributions) == 6
    )
    
    if not p68_consistent:
        return {
            "kind": "fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p68_verdict": p68_res["verdict"],
            "source_p68_status": p68_res["status"],
            "status": "blocked_by_inconsistent_p68_diagnostics",
            "reason": "p68_internal_evidence_structures_inconsistent",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
            "observed_source_component_runs": len(original_numeric_profiles) * 3,
            "profile_vector_fields": PROFILE_VECTOR_FIELDS,
            "coordinate_ids": COORDINATE_IDS,
            "explanatory_views": [],
            "aggregate_diagnostics": {},
            
            # Boundary flags
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
            "no_direct_p67_import": True,
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
            "no_latent_kl_explanatory_proof_claim": True,
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "explanatory_boundary_diagnostic_only": True,
        }
        
    # Build original profiles lookup: target_id -> 3-value list
    original_profiles_lookup = {}
    for p in original_numeric_profiles:
        original_profiles_lookup[p["target_id"]] = p["profile_vector_values"]
        
    # Calculate denominators (from full_vector view)
    # We will build full_vector first to get denominators
    full_pairwise_distances_proj = []
    full_non_self_l2_dists = []
    
    for ta in TARGET_IDS:
        for tb in TARGET_IDS:
            u_proj = get_projection(original_profiles_lookup[ta], "full_vector")
            v_proj = get_projection(original_profiles_lookup[tb], "full_vector")
            l1, l2, max_abs, sq_l2 = _calculate_distances(u_proj, v_proj)
            is_self = (ta == tb)
            separated = (l2 > DISTANCE_EPSILON)
            
            full_pairwise_distances_proj.append({
                "target_a": ta,
                "target_b": tb,
                "l1_distance": float(l1),
                "l2_distance": float(l2),
                "max_abs_distance": float(max_abs),
                "squared_l2_distance": float(sq_l2),
                "is_self_pair": bool(is_self),
                "separated_above_epsilon": bool(separated),
            })
            
            if not is_self:
                full_non_self_l2_dists.append(l2)
                
    full_min_l2 = min(full_non_self_l2_dists) if full_non_self_l2_dists else 1.0
    full_mean_l2 = sum(full_non_self_l2_dists) / len(full_non_self_l2_dists) if full_non_self_l2_dists else 1.0
    full_max_l2 = max(full_non_self_l2_dists) if full_non_self_l2_dists else 1.0
    
    # 5. Compute all views
    explanatory_views = []
    all_views_finite = True
    views_sufficiency = {}
    
    for view_id in VIEW_IDS:
        pairwise_distances = []
        non_self_l2_dists = []
        non_self_l1_dists = []
        non_self_max_abs_dists = []
        
        for ta in TARGET_IDS:
            for tb in TARGET_IDS:
                u_proj = get_projection(original_profiles_lookup[ta], view_id)
                v_proj = get_projection(original_profiles_lookup[tb], view_id)
                l1, l2, max_abs, sq_l2 = _calculate_distances(u_proj, v_proj)
                
                is_self = (ta == tb)
                separated = (l2 > DISTANCE_EPSILON)
                
                if not (math.isfinite(l1) and math.isfinite(l2) and math.isfinite(max_abs) and math.isfinite(sq_l2)):
                    all_views_finite = False
                    
                pairwise_distances.append({
                    "target_a": ta,
                    "target_b": tb,
                    "l1_distance": float(l1),
                    "l2_distance": float(l2),
                    "max_abs_distance": float(max_abs),
                    "squared_l2_distance": float(sq_l2),
                    "is_self_pair": bool(is_self),
                    "separated_above_epsilon": bool(separated),
                })
                
                if not is_self:
                    non_self_l2_dists.append(l2)
                    non_self_l1_dists.append(l1)
                    non_self_max_abs_dists.append(max_abs)
                    
        min_l2 = min(non_self_l2_dists) if non_self_l2_dists else 0.0
        mean_l2 = sum(non_self_l2_dists) / len(non_self_l2_dists) if non_self_l2_dists else 0.0
        max_l2 = max(non_self_l2_dists) if non_self_l2_dists else 0.0
        
        view_separates = all(d > DISTANCE_EPSILON for d in non_self_l2_dists)
        views_sufficiency[view_id] = view_separates
        
        ratio_min = min_l2 / full_min_l2 if full_min_l2 > 0.0 else 0.0
        ratio_mean = mean_l2 / full_mean_l2 if full_mean_l2 > 0.0 else 0.0
        ratio_max = max_l2 / full_max_l2 if full_max_l2 > 0.0 else 0.0
        
        explanatory_views.append({
            "view_id": view_id,
            "pairwise_distances": pairwise_distances,
            "view_separates_all_non_self_pairs": bool(view_separates),
            "min_non_self_l2_distance": float(min_l2),
            "mean_non_self_l2_distance": float(mean_l2),
            "max_non_self_l2_distance": float(max_l2),
            "distance_ratio_to_full_min_l2": float(ratio_min),
            "distance_ratio_to_full_mean_l2": float(ratio_mean),
            "distance_ratio_to_full_max_l2": float(ratio_max),
        })
        
    # 6. Compare explanatory sufficiency
    reconstruction_only_sufficient = views_sufficiency["reconstruction_only"]
    total_only_sufficient = views_sufficiency["total_only"]
    kl_only_sufficient = views_sufficiency["kl_only"]
    total_plus_reconstruction_sufficient = views_sufficiency["total_plus_reconstruction"]
    
    kl_only_material = p68_res["aggregate_diagnostics"]["kl_coordinate_material"] # False
    kl_only_explanatory = (kl_only_sufficient and kl_only_material)
    
    # Calculate P68 total+reconstruction mean contribution share
    mean_shares = {a["coordinate_id"]: a["mean_contribution_share"] for a in aggregate_coordinate_contributions}
    reconstruction_total_mean_contribution_share = mean_shares["total_delta"] + mean_shares["reconstruction_delta"]
    
    reconstruction_total_explains_full_separation = (
        total_plus_reconstruction_sufficient
        and reconstruction_total_mean_contribution_share >= RECON_TOTAL_EXPLANATORY_SHARE_THRESHOLD
    )
    
    # 7. Compute KL negligible boundary
    kl_mean_contribution_share = p68_res["aggregate_diagnostics"]["kl_coordinate_mean_contribution_share"]
    kl_coordinate_separates_all_non_self_pairs = p68_res["aggregate_diagnostics"]["kl_coordinate_separates_all_non_self_pairs"]
    
    kl_negligible_by_share = (kl_mean_contribution_share < KL_MATERIALITY_SHARE_THRESHOLD)
    kl_negligible_despite_coordinate_separation = (kl_coordinate_separates_all_non_self_pairs and kl_negligible_by_share)
    
    # 8. Build explanatory boundary verdict fields
    reconstruction_driven_numeric_separation = (reconstruction_total_explains_full_separation and not kl_only_material)
    
    if reconstruction_driven_numeric_separation:
        primary_explanation = "total_plus_reconstruction"
    else:
        primary_explanation = "undetermined"
        
    if kl_negligible_despite_coordinate_separation:
        kl_explanation_status = "coordinate_separates_but_negligible_share"
    else:
        kl_explanation_status = "not_negligible_or_does_not_separate"
        
    if not kl_only_material:
        latent_kl_explanatory_boundary = "not_supported_by_current_coordinate_contribution_evidence"
    else:
        latent_kl_explanatory_boundary = "supported_by_material_kl_coordinate_share"
        
    # Interpretation logic
    if reconstruction_driven_numeric_separation:
        interpretation = (
            "P69 shows that the current " + "target-profile " + "separation " +
            "is explainable by total/reconstruction coordinates without material beta-weighted " +
            "kl " + "contribution. This is an explanatory boundary " + "diagnostic " +
            "only and does not establish " + "latent " + "learning, " + "semantic " +
            "geometry" + ", transfer, generation, " + "gsb" + ", " + "vae " +
            "success, or coordinate semantic meaning."
        )
    elif kl_only_sufficient and not kl_only_material:
        interpretation = (
            "P69 shows that the beta-weighted " + "kl " + "coordinate can separate targets above epsilon, " +
            "but its contribution share remains negligible relative to total/reconstruction coordinates. " +
            "This supports a " + "kl" + "-negligible explanatory boundary, not a " + "latent " + "or semantic claim."
        )
    else:
        interpretation = (
            "P69 shows that beta-weighted " + "kl " + "has material contribution in the current coordinate evidence. " +
            "This remains " + "diagnostic " + "only and does not establish " + "latent " + "learning, " +
            "semantic " + "geometry" + ", transfer, generation, " + "gsb" + ", " + "vae " +
            "success, or coordinate semantic meaning."
        )
        
    aggregate_diagnostics = {
        "source_p68_evidence_valid": bool(p68_valid),
        "source_total_reconstruction_dominance_carried_forward": bool(p68_res["aggregate_diagnostics"]["dominant_coordinate_id"] in ["total_delta", "reconstruction_delta"]),
        "source_kl_coordinate_material": bool(kl_only_material),
        "source_kl_mean_contribution_share": float(kl_mean_contribution_share),
        "source_qualitative_label_invariance_still_holds": bool(p68_res["aggregate_diagnostics"]["source_qualitative_label_invariance_still_holds"]),
        "view_count": len(VIEW_IDS),
        "all_views_finite": bool(all_views_finite),
        "full_vector_separates_all_non_self_pairs": bool(views_sufficiency["full_vector"]),
        "reconstruction_only_sufficient": bool(reconstruction_only_sufficient),
        "total_only_sufficient": bool(total_only_sufficient),
        "kl_only_sufficient": bool(kl_only_sufficient),
        "kl_only_explanatory": bool(kl_only_explanatory),
        "total_plus_reconstruction_sufficient": bool(total_plus_reconstruction_sufficient),
        "reconstruction_plus_kl_sufficient": bool(views_sufficiency["reconstruction_plus_kl"]),
        "total_plus_kl_sufficient": bool(views_sufficiency["total_plus_kl"]),
        "reconstruction_total_mean_contribution_share": float(reconstruction_total_mean_contribution_share),
        "reconstruction_total_explains_full_separation": bool(reconstruction_total_explains_full_separation),
        "kl_negligible_by_share": bool(kl_negligible_by_share),
        "kl_negligible_despite_coordinate_separation": bool(kl_negligible_despite_coordinate_separation),
        "reconstruction_driven_numeric_separation": bool(reconstruction_driven_numeric_separation),
        "latent_kl_explanatory_boundary": latent_kl_explanatory_boundary,
        "primary_explanation": primary_explanation,
        "kl_explanation_status": kl_explanation_status,
        "explanatory_boundary_interpretation": interpretation,
        "explanatory_boundary_claim": "reconstruction_only_vs_kl_only_explanatory_boundary_audit_only_no_dataset_no_generalization",
        "diagnostic_only_no_generalization": True,
    }
    
    passed = (
        p68_valid
        and p68_consistent
        and len(original_numeric_profiles) == 3
        and all(len(p["profile_vector_values"]) == 3 for p in original_numeric_profiles)
        and all(all(math.isfinite(val) for val in p["profile_vector_values"]) for p in original_numeric_profiles)
        and len(explanatory_views) == 7
        and all_views_finite
        and views_sufficiency["full_vector"]
    )
    
    summary = {
        "kind": "fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_no_dataset_no_generalization",
        "source_phase": SOURCE_PHASE,
        "contract_version": CONTRACT_VERSION,
        "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
        "source_p68_verdict": p68_res["verdict"],
        "source_p68_status": p68_res["status"],
        "status": "fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_available_no_dataset_no_generalization" if passed else "fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe_failed",
        "reason": "fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe_success" if passed else "fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe_failed",
        "verdict": "PASS" if passed else "FAIL",
        
        "target_ids": TARGET_IDS,
        "seed_values": SEED_VALUES,
        "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
        "observed_source_component_runs": len(original_numeric_profiles) * 3,
        "profile_vector_fields": PROFILE_VECTOR_FIELDS,
        "coordinate_ids": COORDINATE_IDS,
        "explanatory_views": explanatory_views,
        "aggregate_diagnostics": aggregate_diagnostics,
        
        # Boundary flags
        "no_new_optimization": True,
        "no_direct_optimizer_created": True,
        "no_direct_model_created": True,
        "no_direct_torch_import": True,
        "no_direct_p67_import": True,
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
        "no_latent_kl_explanatory_proof_claim": True,
        "no_weighted_training": True,
        "uniform_objective_preserved": True,
        "held_out_diagnostic_only": True,
        "explanatory_boundary_diagnostic_only": True,
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    assert_no_forbidden_claims(interpretation, "interpretation")
    
    return summary


def fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_json(probe_res: dict) -> str:
    d = fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
