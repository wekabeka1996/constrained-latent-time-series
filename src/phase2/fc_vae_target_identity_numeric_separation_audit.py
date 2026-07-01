# src/phase2/fc_vae_target_identity_numeric_separation_audit.py

import json
import math
from typing import Any, Dict, List

from src.phase2.fc_vae_held_out_label_permutation_control_audit import (
    run_p65_held_out_label_permutation_control_audit_probe,
)

# Constants
SOURCE_PHASE = "P66"
CONTRACT_VERSION = "phase2_p66_fc_vae_target_identity_numeric_separation_audit_contract_v1"
SOURCE_EVIDENCE_PHASE = "P65"
TARGET_IDS = ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
SEED_VALUES = [62062, 62162, 62262]
EXPECTED_SOURCE_COMPONENT_RUNS = 9
EXPECTED_PERMUTATION_IDS = ["cyclic_forward", "cyclic_backward"]
DISTANCE_EPSILON = 1e-12

CYCLIC_FORWARD_MAP = {
    "bridge_lambda_0_25": "bridge_lambda_0_5",
    "bridge_lambda_0_5": "bridge_lambda_0_75",
    "bridge_lambda_0_75": "bridge_lambda_0_25",
}

CYCLIC_BACKWARD_MAP = {
    "bridge_lambda_0_25": "bridge_lambda_0_75",
    "bridge_lambda_0_5": "bridge_lambda_0_25",
    "bridge_lambda_0_75": "bridge_lambda_0_5",
}

PERMUTATION_MAPS = {
    "cyclic_forward": CYCLIC_FORWARD_MAP,
    "cyclic_backward": CYCLIC_BACKWARD_MAP,
}

# Permutation inverses to map permuted target ID back to its source original target ID
CYCLIC_FORWARD_INVERSE = {v: k for k, v in CYCLIC_FORWARD_MAP.items()}
CYCLIC_BACKWARD_INVERSE = {v: k for k, v in CYCLIC_BACKWARD_MAP.items()}

PERMUTATION_INVERSES = {
    "cyclic_forward": CYCLIC_FORWARD_INVERSE,
    "cyclic_backward": CYCLIC_BACKWARD_INVERSE,
}


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
    ]
    for item in forbidden_substrings:
        if item in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}: {item}")


def _calculate_distances(u: list, v: list) -> tuple:
    l1 = sum(abs(ui - vi) for ui, vi in zip(u, v))
    l2 = math.sqrt(sum((ui - vi)**2 for ui, vi in zip(u, v)))
    max_abs = max(abs(ui - vi) for ui, vi in zip(u, v))
    return l1, l2, max_abs


def run_p66_target_identity_numeric_separation_audit_probe() -> dict:
    # 1. Fetch P65 evidence
    p65_res = run_p65_held_out_label_permutation_control_audit_probe()
    
    # 2. Validate P65 source evidence
    p65_valid = True
    if not isinstance(p65_res, dict):
        p65_valid = False
    elif p65_res.get("source_phase") != "P65":
        p65_valid = False
    elif p65_res.get("verdict") != "PASS":
        p65_valid = False
    elif p65_res.get("source_evidence_phase") != "P64":
        p65_valid = False
    elif p65_res.get("observed_source_component_runs") != EXPECTED_SOURCE_COMPONENT_RUNS:
        p65_valid = False
    elif p65_res.get("target_ids") != TARGET_IDS:
        p65_valid = False
    elif p65_res.get("seed_values") != SEED_VALUES:
        p65_valid = False
    elif p65_res.get("permutation_ids") != EXPECTED_PERMUTATION_IDS:
        p65_valid = False
    elif p65_res.get("no_new_optimization") is not True:
        p65_valid = False
    elif p65_res.get("label_permutation_control_diagnostic_only") is not True:
        p65_valid = False
        
    if p65_valid:
        agg_p65 = p65_res.get("aggregate_diagnostics", {})
        if not isinstance(agg_p65, dict):
            p65_valid = False
        elif agg_p65.get("source_p64_evidence_valid") is not True:
            p65_valid = False
        elif agg_p65.get("label_specificity_warning") is not True:
            p65_valid = False
            
    if not p65_valid:
        return {
            "kind": "fc_vae_target_identity_numeric_separation_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p65_verdict": p65_res.get("verdict") if isinstance(p65_res, dict) else "ERROR",
            "source_p65_status": p65_res.get("status") if isinstance(p65_res, dict) else "ERROR",
            "status": "blocked_by_invalid_p65_evidence",
            "reason": "p65_evidence_validation_failed",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
            "observed_source_component_runs": 0,
            "profile_vector_fields": [
                "mean_total_delta_value",
                "mean_reconstruction_delta_value",
                "mean_beta_weighted_kl_delta_value",
            ],
            "original_numeric_profiles": [],
            "original_pairwise_distances": [],
            "nearest_neighbor_diagnostics": [],
            "permutation_mismatch_diagnostics": [],
            "aggregate_diagnostics": {},
            
            # Boundary flags
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
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
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "numeric_identity_separation_diagnostic_only": True,
        }
        
    original_target_summary = p65_res["original_target_summary"]
    permutation_summaries = p65_res["permutation_summaries"]
    
    # 4. Build original numeric profile vector per target
    original_numeric_profiles = []
    profiles = {}
    
    for tid in TARGET_IDS:
        tgt_orig = original_target_summary[tid]
        mean_tot = tgt_orig["mean_total_delta_value"]
        mean_rec = tgt_orig["mean_reconstruction_delta_value"]
        mean_beta_kl = tgt_orig["mean_beta_weighted_kl_delta_value"]
        
        vec = [mean_tot, mean_rec, mean_beta_kl]
        profiles[tid] = vec
        
        original_numeric_profiles.append({
            "target_id": tid,
            "profile_vector_fields": [
                "mean_total_delta_value",
                "mean_reconstruction_delta_value",
                "mean_beta_weighted_kl_delta_value",
            ],
            "profile_vector_values": vec,
            "mean_total_delta_value": float(mean_tot),
            "mean_reconstruction_delta_value": float(mean_rec),
            "mean_beta_weighted_kl_delta_value": float(mean_beta_kl),
        })
        
    # 5. Compute original pairwise target distances
    original_pairwise_distances = []
    all_distances_finite = True
    
    for ta in TARGET_IDS:
        for tb in TARGET_IDS:
            u = profiles[ta]
            v = profiles[tb]
            l1, l2, max_abs = _calculate_distances(u, v)
            
            is_self = (ta == tb)
            separated = (l2 > DISTANCE_EPSILON)
            
            if not (math.isfinite(l1) and math.isfinite(l2) and math.isfinite(max_abs)):
                all_distances_finite = False
                
            original_pairwise_distances.append({
                "target_a": ta,
                "target_b": tb,
                "l1_distance": float(l1),
                "l2_distance": float(l2),
                "max_abs_distance": float(max_abs),
                "is_self_pair": bool(is_self),
                "separated_above_epsilon": bool(separated),
            })
            
    # 6. Compute non-self minimum pairwise distance and nearest neighbor
    nearest_neighbor_diagnostics = []
    non_self_distances_list_l2 = []
    non_self_distances_list_l1 = []
    non_self_distances_list_max_abs = []
    
    for ta in TARGET_IDS:
        non_self_pairs = []
        for tb in TARGET_IDS:
            if ta == tb:
                continue
            u = profiles[ta]
            v = profiles[tb]
            l1, l2, max_abs = _calculate_distances(u, v)
            non_self_pairs.append((tb, l1, l2, max_abs))
            
            non_self_distances_list_l2.append(l2)
            non_self_distances_list_l1.append(l1)
            non_self_distances_list_max_abs.append(max_abs)
            
        # Find nearest non-self
        nearest_tb, nearest_l1, nearest_l2, nearest_max_abs = min(non_self_pairs, key=lambda x: x[2])
        self_separated = (nearest_l2 > DISTANCE_EPSILON)
        
        nearest_neighbor_diagnostics.append({
            "target_id": ta,
            "self_distance_l2": 0.0,
            "nearest_non_self_target_id": nearest_tb,
            "nearest_non_self_l2_distance": float(nearest_l2),
            "nearest_non_self_l1_distance": float(nearest_l1),
            "nearest_non_self_max_abs_distance": float(nearest_max_abs),
            "self_separated_from_non_self": bool(self_separated),
        })
        
    min_non_self_l2_distance = min(non_self_distances_list_l2) if non_self_distances_list_l2 else 0.0
    min_non_self_l1_distance = min(non_self_distances_list_l1) if non_self_distances_list_l1 else 0.0
    min_non_self_max_abs_distance = min(non_self_distances_list_max_abs) if non_self_distances_list_max_abs else 0.0
    
    # 7. Compute permutation mismatch diagnostics
    permutation_mismatch_diagnostics = []
    permutation_mismatch_separated_count = 0
    all_permutation_mismatches_finite = True
    
    for pid in EXPECTED_PERMUTATION_IDS:
        p_sum_p65 = permutation_summaries[pid]
        permuted_target_summaries = p_sum_p65["permuted_target_summaries"]
        
        for tid in TARGET_IDS:
            # Under permutation pid, the permuted target summary for tid mapped from original target T_source
            T_source = PERMUTATION_INVERSES[pid][tid]
            
            orig_profile = profiles[tid]
            
            # Retrieve the permuted profile values assigned to tid
            perm_sum = permuted_target_summaries[tid]
            v_perm = [
                perm_sum["mean_total_delta_value"],
                perm_sum["mean_reconstruction_delta_value"],
                perm_sum["mean_beta_weighted_kl_delta_value"]
            ]
            
            l1, l2, max_abs = _calculate_distances(orig_profile, v_perm)
            separated = (l2 > DISTANCE_EPSILON)
            
            if separated:
                permutation_mismatch_separated_count += 1
                
            if not (math.isfinite(l1) and math.isfinite(l2) and math.isfinite(max_abs)):
                all_permutation_mismatches_finite = False
                
            permutation_mismatch_diagnostics.append({
                "permutation_id": pid,
                "target_id": tid,
                "original_profile_vector_values": orig_profile,
                "permuted_profile_vector_values": v_perm,
                "mismatched_profile_source_target_id": T_source,
                "l1_distance": float(l1),
                "l2_distance": float(l2),
                "max_abs_distance": float(max_abs),
                "mismatch_separated_above_epsilon": bool(separated),
            })
            
    # 9. Aggregate diagnostics
    all_original_non_self_profiles_separated_above_epsilon = all(
        d["self_separated_from_non_self"] for d in nearest_neighbor_diagnostics
    )
    
    all_targets_self_nearest_when_self_allowed = True
    all_targets_have_non_self_distance_above_epsilon = all_original_non_self_profiles_separated_above_epsilon
    
    all_permutation_mismatches_separated_above_epsilon = (
        permutation_mismatch_separated_count == len(permutation_mismatch_diagnostics)
    )
    
    numeric_identity_separation_present = (
        all_original_non_self_profiles_separated_above_epsilon
        and all_permutation_mismatches_separated_above_epsilon
    )
    
    qualitative_label_invariance_still_holds = p65_res["aggregate_diagnostics"]["label_specificity_warning"]
    
    if numeric_identity_separation_present and qualitative_label_invariance_still_holds:
        interpretation = (
            "Numeric target profiles are separable in this tiny deterministic evidence, "
            "but qualitative all-improved / reconstruction-supported labels remain invariant. "
            "Therefore, numeric profile identity is present as a diagnostic signal, "
            "while semantic or generalization claims remain unsupported."
        )
    else:
        interpretation = (
            "Numeric target profiles are not sufficiently separated in this tiny deterministic evidence. "
            "Target identity remains unsupported beyond global qualitative improvement."
        )
        
    aggregate_diagnostics = {
        "source_p65_evidence_valid": bool(p65_valid),
        "source_label_specificity_warning": bool(qualitative_label_invariance_still_holds),
        "original_profile_count": len(original_numeric_profiles),
        "pairwise_distance_count": len(original_pairwise_distances),
        "non_self_pairwise_distance_count": len(non_self_distances_list_l2),
        "min_non_self_l2_distance": float(min_non_self_l2_distance),
        "min_non_self_l1_distance": float(min_non_self_l1_distance),
        "min_non_self_max_abs_distance": float(min_non_self_max_abs_distance),
        "all_original_non_self_profiles_separated_above_epsilon": bool(all_original_non_self_profiles_separated_above_epsilon),
        "all_targets_self_nearest_when_self_allowed": bool(all_targets_self_nearest_when_self_allowed),
        "all_targets_have_non_self_distance_above_epsilon": bool(all_targets_have_non_self_distance_above_epsilon),
        "permutation_mismatch_count": len(permutation_mismatch_diagnostics),
        "permutation_mismatch_separated_count": permutation_mismatch_separated_count,
        "all_permutation_mismatches_separated_above_epsilon": bool(all_permutation_mismatches_separated_above_epsilon),
        "numeric_identity_separation_present": bool(numeric_identity_separation_present),
        "qualitative_label_invariance_still_holds": bool(qualitative_label_invariance_still_holds),
        "numeric_identity_vs_qualitative_invariance_interpretation": interpretation,
        "numeric_identity_separation_claim": "target_identity_numeric_separation_audit_only_no_new_optimization_no_dataset_no_generalization",
        "diagnostic_only_no_generalization": True,
    }
    
    # Verdict passes if:
    # 1. P65 source evidence is valid and PASS
    # 2. Original target summary has exactly 3 targets
    # 3. Each original numeric profile has exactly 3 finite values
    # 4. Pairwise distances are finite
    # 5. Permutation mismatch distances are finite
    # Note: verdict does not require numeric separation to be present.
    passed = (
        p65_valid
        and len(original_numeric_profiles) == 3
        and all(len(p["profile_vector_values"]) == 3 for p in original_numeric_profiles)
        and all(all(math.isfinite(val) for val in p["profile_vector_values"]) for p in original_numeric_profiles)
        and all_distances_finite
        and all_permutation_mismatches_finite
    )
    
    summary = {
        "kind": "fc_vae_target_identity_numeric_separation_audit_no_dataset_no_generalization",
        "source_phase": SOURCE_PHASE,
        "contract_version": CONTRACT_VERSION,
        "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
        "source_p65_verdict": p65_res["verdict"],
        "source_p65_status": p65_res["status"],
        "status": "fc_vae_target_identity_numeric_separation_audit_available_no_dataset_no_generalization" if passed else "fc_vae_target_identity_numeric_separation_audit_probe_failed",
        "reason": "fc_vae_target_identity_numeric_separation_audit_probe_success" if passed else "fc_vae_target_identity_numeric_separation_audit_probe_failed",
        "verdict": "PASS" if passed else "FAIL",
        
        "target_ids": TARGET_IDS,
        "seed_values": SEED_VALUES,
        "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
        "observed_source_component_runs": len(original_numeric_profiles) * 3, # 3 targets * 3 seeds = 9
        "profile_vector_fields": [
            "mean_total_delta_value",
            "mean_reconstruction_delta_value",
            "mean_beta_weighted_kl_delta_value",
        ],
        "original_numeric_profiles": original_numeric_profiles,
        "original_pairwise_distances": original_pairwise_distances,
        "nearest_neighbor_diagnostics": nearest_neighbor_diagnostics,
        "permutation_mismatch_diagnostics": permutation_mismatch_diagnostics,
        "aggregate_diagnostics": aggregate_diagnostics,
        
        # Boundary flags
        "no_new_optimization": True,
        "no_direct_optimizer_created": True,
        "no_direct_model_created": True,
        "no_direct_torch_import": True,
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
        "no_weighted_training": True,
        "uniform_objective_preserved": True,
        "held_out_diagnostic_only": True,
        "numeric_identity_separation_diagnostic_only": True,
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    assert_no_forbidden_claims(interpretation, "interpretation")
    
    return summary


def fc_vae_target_identity_numeric_separation_audit_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_target_identity_numeric_separation_audit_json(probe_res: dict) -> str:
    d = fc_vae_target_identity_numeric_separation_audit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
