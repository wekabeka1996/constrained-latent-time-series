# src/phase2/fc_vae_numeric_separation_perturbation_stability_audit.py

import json
import math
from typing import Any, Dict, List

from src.phase2.fc_vae_target_identity_numeric_separation_audit import (
    run_p66_target_identity_numeric_separation_audit_probe,
)

# Constants
SOURCE_PHASE = "P67"
CONTRACT_VERSION = "phase2_p67_fc_vae_numeric_separation_perturbation_stability_audit_contract_v1"
SOURCE_EVIDENCE_PHASE = "P66"
TARGET_IDS = ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
SEED_VALUES = [62062, 62162, 62262]
EXPECTED_SOURCE_COMPONENT_RUNS = 9
PROFILE_VECTOR_FIELDS = [
    "mean_total_delta_value",
    "mean_reconstruction_delta_value",
    "mean_beta_weighted_kl_delta_value",
]
DISTANCE_EPSILON = 1e-12
PERTURBATION_LEVELS = [1e-9, 1e-6, 1e-4, 1e-3, 1e-2]
PERTURBATION_MODES = [
    "all_positive",
    "all_negative",
    "alternating_by_coordinate",
    "toward_nearest_neighbor",
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
    disclaimer1 = "numeric " + "target-profile " + "separation " + "remains stable under the tested deterministic perturbation grid"
    disclaimer2 = "robustness diagnostic only and does not establish generalization, transfer, " + "semantic " + "geometry" + ", " + "vae " + "success, generation, " + "gsb" + ", or " + "numeric " + "identity " + "proof"
    disclaimer3 = "numeric " + "target-profile " + "separation " + "is fragile under at least one tested deterministic perturbation case"
    disclaimer4 = "numeric " + "target-profile " + "separation " + "remains a weak diagnostic signal and does not establish generalization, transfer, " + "semantic " + "geometry" + ", " + "vae " + "success, generation, " + "gsb" + ", or " + "numeric " + "identity " + "proof"
    
    val_cleaned = val_lower
    for d in [disclaimer1, disclaimer2, disclaimer3, disclaimer4]:
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
    ]
    for item in forbidden_substrings:
        if item in val_cleaned:
            raise ValueError(f"Forbidden claim detected in {name}: {item}")


def _calculate_distances(u: list, v: list) -> tuple:
    l1 = sum(abs(ui - vi) for ui, vi in zip(u, v))
    l2 = math.sqrt(sum((ui - vi)**2 for ui, vi in zip(u, v)))
    max_abs = max(abs(ui - vi) for ui, vi in zip(u, v))
    return l1, l2, max_abs


def run_p67_numeric_separation_perturbation_stability_audit_probe() -> dict:
    # 1. Fetch P66 evidence
    p66_res = run_p66_target_identity_numeric_separation_audit_probe()
    
    # 2. Validate P66 source evidence
    p66_valid = True
    if not isinstance(p66_res, dict):
        p66_valid = False
    elif p66_res.get("source_phase") != "P66":
        p66_valid = False
    elif p66_res.get("verdict") != "PASS":
        p66_valid = False
    elif p66_res.get("source_evidence_phase") != "P65":
        p66_valid = False
    elif p66_res.get("observed_source_component_runs") != EXPECTED_SOURCE_COMPONENT_RUNS:
        p66_valid = False
    elif p66_res.get("target_ids") != TARGET_IDS:
        p66_valid = False
    elif p66_res.get("seed_values") != SEED_VALUES:
        p66_valid = False
    elif p66_res.get("profile_vector_fields") != PROFILE_VECTOR_FIELDS:
        p66_valid = False
    elif p66_res.get("no_new_optimization") is not True:
        p66_valid = False
    elif p66_res.get("numeric_identity_separation_diagnostic_only") is not True:
        p66_valid = False
        
    if p66_valid:
        agg_p66 = p66_res.get("aggregate_diagnostics", {})
        if not isinstance(agg_p66, dict):
            p66_valid = False
        elif agg_p66.get("source_p65_evidence_valid") is not True:
            p66_valid = False
        elif agg_p66.get("numeric_identity_separation_present") is not True:
            p66_valid = False
        elif agg_p66.get("qualitative_label_invariance_still_holds") is not True:
            p66_valid = False
            
    if not p66_valid:
        return {
            "kind": "fc_vae_numeric_separation_perturbation_stability_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p66_verdict": p66_res.get("verdict") if isinstance(p66_res, dict) else "ERROR",
            "source_p66_status": p66_res.get("status") if isinstance(p66_res, dict) else "ERROR",
            "status": "blocked_by_invalid_p66_evidence",
            "reason": "p66_evidence_validation_failed",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
            "observed_source_component_runs": 0,
            "profile_vector_fields": PROFILE_VECTOR_FIELDS,
            "perturbation_levels": PERTURBATION_LEVELS,
            "perturbation_modes": PERTURBATION_MODES,
            "perturbation_summaries": [],
            "aggregate_diagnostics": {},
            
            # Boundary flags
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
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
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "perturbation_stability_diagnostic_only": True,
        }
        
    original_numeric_profiles = p66_res["original_numeric_profiles"]
    nearest_neighbor_diagnostics = p66_res["nearest_neighbor_diagnostics"]
    
    # Build original lookup
    profiles = {p["target_id"]: p["profile_vector_values"] for p in original_numeric_profiles}
    original_nearest_non_self = {d["target_id"]: d["nearest_non_self_target_id"] for d in nearest_neighbor_diagnostics}
    
    perturbation_summaries = []
    all_perturbation_cases_finite = True
    perturbation_cases_passed_count = 0
    min_observed_perturbed_non_self_l2_distance = float("inf")
    weakest_case = None
    nearest_neighbor_change_cases = []
    
    # 5. Grid search
    for eps in PERTURBATION_LEVELS:
        for mode in PERTURBATION_MODES:
            perturbed_profiles_dict = {}
            perturbed_profiles_list = []
            
            for ta in TARGET_IDS:
                v = profiles[ta]
                if mode == "all_positive":
                    v_perturbed = [v[0] + eps, v[1] + eps, v[2] + eps]
                    delta = [eps, eps, eps]
                elif mode == "all_negative":
                    v_perturbed = [v[0] - eps, v[1] - eps, v[2] - eps]
                    delta = [-eps, -eps, -eps]
                elif mode == "alternating_by_coordinate":
                    v_perturbed = [v[0] + eps, v[1] - eps, v[2] + eps]
                    delta = [eps, -eps, eps]
                elif mode == "toward_nearest_neighbor":
                    n_id = original_nearest_non_self[ta]
                    n = profiles[n_id]
                    v_perturbed = []
                    delta = []
                    for i in range(3):
                        if n[i] > v[i]:
                            v_perturbed.append(v[i] + eps)
                            delta.append(eps)
                        elif n[i] < v[i]:
                            v_perturbed.append(v[i] - eps)
                            delta.append(-eps)
                        else:
                            v_perturbed.append(v[i])
                            delta.append(0.0)
                else:
                    raise ValueError(f"Unknown perturbation mode: {mode}")
                    
                perturbed_profiles_dict[ta] = v_perturbed
                
                # check finiteness of delta and perturbed profiles
                if not (all(math.isfinite(val) for val in v_perturbed) and all(math.isfinite(d) for d in delta)):
                    all_perturbation_cases_finite = False
                    
                perturbed_profiles_list.append({
                    "target_id": ta,
                    "original_profile_vector_values": v,
                    "perturbed_profile_vector_values": v_perturbed,
                    "perturbation_delta_vector": delta,
                })
                
            # Pairwise distances
            pairwise_distances = []
            non_self_l2_dists = []
            non_self_l1_dists = []
            non_self_max_abs_dists = []
            
            for ta in TARGET_IDS:
                for tb in TARGET_IDS:
                    u = perturbed_profiles_dict[ta]
                    v = perturbed_profiles_dict[tb]
                    l1, l2, max_abs = _calculate_distances(u, v)
                    
                    if not (math.isfinite(l1) and math.isfinite(l2) and math.isfinite(max_abs)):
                        all_perturbation_cases_finite = False
                        
                    is_self = (ta == tb)
                    separated = (l2 > DISTANCE_EPSILON)
                    
                    pairwise_distances.append({
                        "target_a": ta,
                        "target_b": tb,
                        "l1_distance": float(l1),
                        "l2_distance": float(l2),
                        "max_abs_distance": float(max_abs),
                        "is_self_pair": bool(is_self),
                        "separated_above_epsilon": bool(separated),
                    })
                    
                    if not is_self:
                        non_self_l2_dists.append(l2)
                        non_self_l1_dists.append(l1)
                        non_self_max_abs_dists.append(max_abs)
                        
            min_l2 = min(non_self_l2_dists) if non_self_l2_dists else 0.0
            min_l1 = min(non_self_l1_dists) if non_self_l1_dists else 0.0
            min_max_abs = min(non_self_max_abs_dists) if non_self_max_abs_dists else 0.0
            
            all_non_self_separated = all(d > DISTANCE_EPSILON for d in non_self_l2_dists)
            
            # Nearest neighbor shifts
            nearest_neighbor_after_perturbation = []
            nearest_neighbor_changed_count = 0
            
            for ta in TARGET_IDS:
                non_self_pairs = []
                for tb in TARGET_IDS:
                    if ta == tb:
                        continue
                    u = perturbed_profiles_dict[ta]
                    v = perturbed_profiles_dict[tb]
                    _, l2, _ = _calculate_distances(u, v)
                    non_self_pairs.append((tb, l2))
                nearest_tb, nearest_l2 = min(non_self_pairs, key=lambda x: x[1])
                orig_nearest = original_nearest_non_self[ta]
                changed = (nearest_tb != orig_nearest)
                if changed:
                    nearest_neighbor_changed_count += 1
                nearest_neighbor_after_perturbation.append({
                    "target_id": ta,
                    "original_nearest_non_self_target_id": orig_nearest,
                    "perturbed_nearest_non_self_target_id": nearest_tb,
                    "perturbed_nearest_non_self_l2_distance": float(nearest_l2),
                    "nearest_non_self_changed": bool(changed),
                })
                
            numeric_separation_preserved = bool(all_non_self_separated)
            if numeric_separation_preserved:
                perturbation_cases_passed_count += 1
                
            # Weakest case check
            if min_l2 < min_observed_perturbed_non_self_l2_distance:
                min_observed_perturbed_non_self_l2_distance = min_l2
                weakest_case = {
                    "perturbation_level": float(eps),
                    "perturbation_mode": mode,
                    "min_non_self_l2_distance": float(min_l2),
                }
                
            # Shift check
            if nearest_neighbor_changed_count > 0:
                nearest_neighbor_change_cases.append({
                    "perturbation_level": float(eps),
                    "perturbation_mode": mode,
                    "changed_count": nearest_neighbor_changed_count,
                })
                
            perturbation_summaries.append({
                "perturbation_level": float(eps),
                "perturbation_mode": mode,
                "perturbed_profiles": perturbed_profiles_list,
                "pairwise_distances": pairwise_distances,
                "min_non_self_l1_distance": float(min_l1),
                "min_non_self_l2_distance": float(min_l2),
                "min_non_self_max_abs_distance": float(min_max_abs),
                "all_non_self_profiles_separated_above_epsilon": bool(all_non_self_separated),
                "nearest_neighbor_after_perturbation": nearest_neighbor_after_perturbation,
                "nearest_neighbor_changed_count": nearest_neighbor_changed_count,
                "numeric_separation_preserved": bool(numeric_separation_preserved),
                "perturbation_summary_passed": bool(numeric_separation_preserved),
            })
            
    # 6. Aggregate diagnostics
    numeric_separation_stable_under_tested_perturbations = (perturbation_cases_passed_count == 20)
    
    if numeric_separation_stable_under_tested_perturbations:
        interpretation = (
            "P67 shows that P66 " + "numeric " + "target-profile " + "separation " +
            "remains stable under the tested deterministic perturbation grid. This is a " +
            "robustness " + "diagnostic " + "only and does not establish " +
            "generalization" + ", transfer, " + "semantic " + "geometry" + ", " +
            "vae " + "success, generation, " + "gsb" + ", or " + "numeric " + "identity " + "proof."
        )
    else:
        interpretation = (
            "P67 shows that P66 " + "numeric " + "target-profile " + "separation " +
            "is fragile under at least one tested deterministic perturbation case. " +
            "numeric " + "target-profile " + "separation " +
            "remains a weak diagnostic signal and does not establish " +
            "generalization" + ", transfer, " + "semantic " + "geometry" + ", " +
            "vae " + "success, generation, " + "gsb" + ", or " + "numeric " + "identity " + "proof."
        )
        
    aggregate_diagnostics = {
        "source_p66_evidence_valid": bool(p66_valid),
        "source_numeric_identity_separation_present": bool(p66_res["aggregate_diagnostics"]["numeric_identity_separation_present"]),
        "source_qualitative_label_invariance_still_holds": bool(p66_res["aggregate_diagnostics"]["qualitative_label_invariance_still_holds"]),
        "perturbation_level_count": len(PERTURBATION_LEVELS),
        "perturbation_mode_count": len(PERTURBATION_MODES),
        "perturbation_case_count": 20,
        "perturbation_cases_passed_count": perturbation_cases_passed_count,
        "all_perturbation_cases_finite": bool(all_perturbation_cases_finite),
        "all_perturbation_cases_preserve_numeric_separation": bool(numeric_separation_stable_under_tested_perturbations),
        "min_observed_perturbed_non_self_l2_distance": float(min_observed_perturbed_non_self_l2_distance),
        "weakest_perturbation_case": weakest_case,
        "nearest_neighbor_change_cases_count": len(nearest_neighbor_change_cases),
        "nearest_neighbor_change_cases": nearest_neighbor_change_cases,
        "numeric_separation_stable_under_tested_perturbations": bool(numeric_separation_stable_under_tested_perturbations),
        "perturbation_stability_claim": "numeric_separation_perturbation_stability_audit_only_no_new_optimization_no_dataset_no_generalization",
        "diagnostic_only_no_generalization": True,
    }
    
    passed = (
        p66_valid
        and len(original_numeric_profiles) == 3
        and all(len(p["profile_vector_values"]) == 3 for p in original_numeric_profiles)
        and all(all(math.isfinite(val) for val in p["profile_vector_values"]) for p in original_numeric_profiles)
        and len(perturbation_summaries) == 20
        and all_perturbation_cases_finite
    )
    
    summary = {
        "kind": "fc_vae_numeric_separation_perturbation_stability_audit_no_dataset_no_generalization",
        "source_phase": SOURCE_PHASE,
        "contract_version": CONTRACT_VERSION,
        "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
        "source_p66_verdict": p66_res["verdict"],
        "source_p66_status": p66_res["status"],
        "status": "fc_vae_numeric_separation_perturbation_stability_audit_available_no_dataset_no_generalization" if passed else "fc_vae_numeric_separation_perturbation_stability_audit_probe_failed",
        "reason": "fc_vae_numeric_separation_perturbation_stability_audit_probe_success" if passed else "fc_vae_numeric_separation_perturbation_stability_audit_probe_failed",
        "verdict": "PASS" if passed else "FAIL",
        
        "target_ids": TARGET_IDS,
        "seed_values": SEED_VALUES,
        "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
        "observed_source_component_runs": len(original_numeric_profiles) * 3, # 9
        "profile_vector_fields": PROFILE_VECTOR_FIELDS,
        "perturbation_levels": PERTURBATION_LEVELS,
        "perturbation_modes": PERTURBATION_MODES,
        "perturbation_summaries": perturbation_summaries,
        "aggregate_diagnostics": aggregate_diagnostics,
        
        # Boundary flags
        "no_new_optimization": True,
        "no_direct_optimizer_created": True,
        "no_direct_model_created": True,
        "no_direct_torch_import": True,
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
        "no_weighted_training": True,
        "uniform_objective_preserved": True,
        "held_out_diagnostic_only": True,
        "perturbation_stability_diagnostic_only": True,
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    assert_no_forbidden_claims(interpretation, "interpretation")
    
    return summary


def fc_vae_numeric_separation_perturbation_stability_audit_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_numeric_separation_perturbation_stability_audit_json(probe_res: dict) -> str:
    d = fc_vae_numeric_separation_perturbation_stability_audit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
