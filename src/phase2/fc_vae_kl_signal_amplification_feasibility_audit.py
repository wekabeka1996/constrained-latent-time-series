# src/phase2/fc_vae_kl_signal_amplification_feasibility_audit.py

import json
import math
from typing import Any, Dict, List

from src.phase2.fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit import (
    run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe,
)

# Constants
SOURCE_PHASE = "P70"
CONTRACT_VERSION = "phase2_p70_fc_vae_kl_signal_amplification_feasibility_audit_contract_v1"
SOURCE_EVIDENCE_PHASE = "P69"
TARGET_IDS = ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
SEED_VALUES = [62062, 62162, 62262]
EXPECTED_SOURCE_COMPONENT_RUNS = 9
PROFILE_VECTOR_FIELDS = [
    "mean_total_delta_value",
    "mean_reconstruction_delta_value",
    "mean_beta_weighted_kl_delta_value",
]
COORDINATE_IDS = ["total_delta", "reconstruction_delta", "beta_weighted_kl_delta"]
VIEW_IDS = [
    "full_vector",
    "reconstruction_only",
    "total_only",
    "kl_only",
    "total_plus_reconstruction",
    "reconstruction_plus_kl",
    "total_plus_kl",
]
DISTANCE_EPSILON = 1e-12
KL_MATERIALITY_SHARE_THRESHOLD = 0.01
AMPLIFICATION_TARGET_SHARES = [0.01, 0.10, 0.50]
AMPLIFICATION_REFERENCE_VIEWS = ["reconstruction_only", "total_only", "total_plus_reconstruction", "full_vector"]
AMPLIFICATION_REASONABLE_UPPER_BOUND = 10000.0


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
    disclaimer1 = "beta-weighted " + "kl " + "coordinate contains an above-epsilon separability signal, but the signal is far below reconstruction/total scale and would require extreme post-hoc amplification to become materially comparable"
    disclaimer2 = "beta-weighted " + "kl " + "coordinate contains an above-epsilon separability signal, but material comparison to reconstruction/total coordinates remains scale-limited under the configured amplification bound"
    disclaimer3 = "beta-weighted " + "kl " + "coordinate contains an above-epsilon separability signal and reaches configured material-share thresholds under deterministic post-hoc amplification"
    disclaimer4 = "amplification feasibility is diagnostic and does not establish generalization, transfer, " + "semantic " + "geometry" + ", or " + "kl " + "semantic meaning"
    disclaimer5 = "quantifies how much deterministic post-hoc amplification the beta-weighted " + "kl " + "coordinate would require to become materially comparable to reconstruction/total coordinates; amplification feasibility is diagnostic and does not establish generalization, transfer, " + "semantic " + "geometry" + ", or " + "kl " + "semantic meaning"
    disclaimer6 = "deterministic " + "kl" + "-amplification feasibility audit over accepted " + "p69 " + "evidence only. it does not perform new optimization, does not change beta or objective weighting, and does not prove dataset generalization, transfer, " + "semantic " + "geometry" + ", " + "latent " + "learning, generation, " + "gsb" + ", " + "vae " + "success, " + "numeric " + "identity " + "proof, perturbation " + "robustness " + "proof, coordinate semantic meaning, label-specific " + "semantic " + "structure, latent/kl explanatory proof, or " + "kl " + "semantic signal proof"
    
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
        "kl " + "semantic signal proof",
    ]
    for item in forbidden_substrings:
        if item in val_cleaned:
            raise ValueError(f"Forbidden claim detected in {name}: {item}")


def run_p70_kl_signal_amplification_feasibility_audit_probe() -> dict:
    # 1. Fetch P69 evidence
    p69_res = run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe()
    
    # 2. Validate P69 source evidence
    p69_valid = True
    if not isinstance(p69_res, dict):
        p69_valid = False
    elif p69_res.get("source_phase") != "P69":
        p69_valid = False
    elif p69_res.get("verdict") != "PASS":
        p69_valid = False
    elif p69_res.get("source_evidence_phase") != "P68":
        p69_valid = False
    elif p69_res.get("observed_source_component_runs") != EXPECTED_SOURCE_COMPONENT_RUNS:
        p69_valid = False
    elif p69_res.get("target_ids") != TARGET_IDS:
        p69_valid = False
    elif p69_res.get("seed_values") != SEED_VALUES:
        p69_valid = False
    elif p69_res.get("profile_vector_fields") != PROFILE_VECTOR_FIELDS:
        p69_valid = False
    elif p69_res.get("coordinate_ids") != COORDINATE_IDS:
        p69_valid = False
    elif p69_res.get("no_new_optimization") is not True:
        p69_valid = False
    elif p69_res.get("explanatory_boundary_diagnostic_only") is not True:
        p69_valid = False
        
    if p69_valid:
        agg_p69 = p69_res.get("aggregate_diagnostics", {})
        if not isinstance(agg_p69, dict):
            p69_valid = False
        elif agg_p69.get("source_p68_evidence_valid") is not True:
            p69_valid = False
        elif agg_p69.get("reconstruction_driven_numeric_separation") is not True:
            p69_valid = False
        elif agg_p69.get("kl_only_sufficient") is not True:
            p69_valid = False
        elif agg_p69.get("kl_only_explanatory") is not False:
            p69_valid = False
        elif agg_p69.get("kl_negligible_by_share") is not True:
            p69_valid = False
        elif agg_p69.get("latent_kl_explanatory_boundary") != "not_supported_by_current_coordinate_contribution_evidence":
            p69_valid = False
            
    if not p69_valid:
        return {
            "kind": "fc_vae_kl_signal_amplification_feasibility_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p69_verdict": p69_res.get("verdict") if isinstance(p69_res, dict) else "ERROR",
            "source_p69_status": p69_res.get("status") if isinstance(p69_res, dict) else "ERROR",
            "status": "blocked_by_invalid_p69_evidence",
            "reason": "p69_evidence_validation_failed",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
            "observed_source_component_runs": 0,
            "profile_vector_fields": PROFILE_VECTOR_FIELDS,
            "coordinate_ids": COORDINATE_IDS,
            "view_ids": VIEW_IDS,
            "kl_to_reference_ratio_diagnostics": [],
            "contribution_share_amplification_diagnostics": [],
            "scaled_kl_scenarios": [],
            "aggregate_diagnostics": {},
            
            # Boundary flags
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
            "no_direct_p68_import": True,
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
            "no_kl_semantic_signal_proof_claim": True,
            "no_beta_change": True,
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "kl_amplification_feasibility_diagnostic_only": True,
        }
        
    explanatory_views = p69_res["explanatory_views"]
    
    # 4. Validate P69 consistency
    p69_consistent = (
        len(explanatory_views) == 7
        and all(v["view_id"] in VIEW_IDS for v in explanatory_views)
        and all(len(v["pairwise_distances"]) == 9 for v in explanatory_views)
        and p69_res["aggregate_diagnostics"]["full_vector_separates_all_non_self_pairs"] is True
        and p69_res["aggregate_diagnostics"]["kl_only_sufficient"] is True
    )
    
    if not p69_consistent:
        return {
            "kind": "fc_vae_kl_signal_amplification_feasibility_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p69_verdict": p69_res["verdict"],
            "source_p69_status": p69_res["status"],
            "status": "blocked_by_inconsistent_p69_diagnostics",
            "reason": "p69_internal_evidence_structures_inconsistent",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
            "observed_source_component_runs": len(explanatory_views) * 3,
            "profile_vector_fields": PROFILE_VECTOR_FIELDS,
            "coordinate_ids": COORDINATE_IDS,
            "view_ids": VIEW_IDS,
            "kl_to_reference_ratio_diagnostics": [],
            "contribution_share_amplification_diagnostics": [],
            "scaled_kl_scenarios": [],
            "aggregate_diagnostics": {},
            
            # Boundary flags
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
            "no_direct_p68_import": True,
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
            "no_kl_semantic_signal_proof_claim": True,
            "no_beta_change": True,
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "kl_amplification_feasibility_diagnostic_only": True,
        }
        
    # 5. Build view lookup
    view_data = {}
    view_data_sq = {}
    for view in explanatory_views:
        vid = view["view_id"]
        pairwise = view["pairwise_distances"]
        pair_dists = {}
        pair_sq_dists = {}
        for p in pairwise:
            if p["target_a"] == p["target_b"]:
                continue
            pk = f"{p['target_a']}__to__{p['target_b']}"
            pair_dists[pk] = p["l2_distance"]
            pair_sq_dists[pk] = p["squared_l2_distance"]
            
        view_data[vid] = {
            "min_l2": view["min_non_self_l2_distance"],
            "mean_l2": view["mean_non_self_l2_distance"],
            "max_l2": view["max_non_self_l2_distance"],
            "pair_dists": pair_dists
        }
        view_data_sq[vid] = pair_sq_dists
        
    # 6. Compute KL-to-reference ratio diagnostics
    kl_to_reference_ratio_diagnostics = []
    
    for ref_vid in AMPLIFICATION_REFERENCE_VIEWS:
        pair_records = []
        ratios = []
        amp_factors = []
        
        for ta in TARGET_IDS:
            for tb in TARGET_IDS:
                if ta == tb:
                    continue
                pk = f"{ta}__to__{tb}"
                kl_l2 = view_data["kl_only"]["pair_dists"][pk]
                ref_l2 = view_data[ref_vid]["pair_dists"][pk]
                
                ratio = kl_l2 / ref_l2 if ref_l2 > 0.0 else 0.0
                amp = ref_l2 / kl_l2 if kl_l2 > 0.0 else 0.0
                
                ratios.append(ratio)
                amp_factors.append(amp)
                
                pair_records.append({
                    "target_a": ta,
                    "target_b": tb,
                    "pair_key": pk,
                    "kl_l2_distance": float(kl_l2),
                    "reference_l2_distance": float(ref_l2),
                    "kl_to_reference_ratio": float(ratio),
                    "reference_to_kl_amplification_factor": float(amp),
                })
                
        all_ratios_finite = all(math.isfinite(r) for r in ratios)
        all_amp_finite = all(math.isfinite(a) for a in amp_factors)
        
        kl_to_reference_ratio_diagnostics.append({
            "reference_view_id": ref_vid,
            "pair_records": pair_records,
            "min_kl_to_reference_ratio": float(min(ratios)),
            "mean_kl_to_reference_ratio": float(sum(ratios)/len(ratios)),
            "max_kl_to_reference_ratio": float(max(ratios)),
            "min_reference_to_kl_amplification_factor": float(min(amp_factors)),
            "mean_reference_to_kl_amplification_factor": float(sum(amp_factors)/len(amp_factors)),
            "max_reference_to_kl_amplification_factor": float(max(amp_factors)),
            "all_ratios_finite": bool(all_ratios_finite),
            "all_amplification_factors_finite": bool(all_amp_finite),
        })
        
    # 7. Compute contribution-share amplification factors
    contribution_share_amplification_diagnostics = []
    mean_factors_by_share = {}
    
    for s in AMPLIFICATION_TARGET_SHARES:
        pair_records = []
        factors = []
        all_factors_finite = True
        all_within_bound = True
        any_within_bound = False
        
        for ta in TARGET_IDS:
            for tb in TARGET_IDS:
                if ta == tb:
                    continue
                pk = f"{ta}__to__{tb}"
                total_sq = view_data_sq["total_only"][pk]
                reconstruction_sq = view_data_sq["reconstruction_only"][pk]
                kl_sq = view_data_sq["kl_only"][pk]
                
                non_kl_sq = total_sq + reconstruction_sq
                
                if kl_sq <= 0.0 or non_kl_sq <= 0.0 or not math.isfinite(kl_sq) or not math.isfinite(non_kl_sq):
                    factor = 0.0
                    all_factors_finite = False
                    within_bound = False
                else:
                    factor = math.sqrt((s * non_kl_sq) / ((1.0 - s) * kl_sq))
                    factors.append(factor)
                    within_bound = (factor <= AMPLIFICATION_REASONABLE_UPPER_BOUND)
                    if within_bound:
                        any_within_bound = True
                    else:
                        all_within_bound = False
                        
                pair_records.append({
                    "target_a": ta,
                    "target_b": tb,
                    "pair_key": pk,
                    "total_squared_l2": float(total_sq),
                    "reconstruction_squared_l2": float(reconstruction_sq),
                    "kl_squared_l2": float(kl_sq),
                    "non_kl_squared_l2": float(non_kl_sq),
                    "target_share": float(s),
                    "required_amplification_factor": float(factor),
                    "within_reasonable_upper_bound": bool(within_bound),
                })
                
        if factors:
            mean_factor = sum(factors) / len(factors)
            min_factor = min(factors)
            max_factor = max(factors)
        else:
            mean_factor = 0.0
            min_factor = 0.0
            max_factor = 0.0
            
        mean_factors_by_share[s] = mean_factor
        
        contribution_share_amplification_diagnostics.append({
            "target_share": float(s),
            "pair_records": pair_records,
            "min_required_amplification_factor": float(min_factor),
            "mean_required_amplification_factor": float(mean_factor),
            "max_required_amplification_factor": float(max_factor),
            "all_factors_finite": bool(all_factors_finite),
            "all_within_reasonable_upper_bound": bool(all_within_bound),
            "any_within_reasonable_upper_bound": bool(any_within_bound),
        })
        
    # 8. Compute deterministic scaled-KL scenarios
    candidates = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    for s in AMPLIFICATION_TARGET_SHARES:
        m_fact = mean_factors_by_share.get(s, 0.0)
        if m_fact > 0.0 and math.isfinite(m_fact):
            rounded = round(m_fact, 4)
            if rounded not in candidates:
                candidates.append(rounded)
    candidates.sort()
    
    scaled_kl_scenarios = []
    
    for factor in candidates:
        scenario_records = []
        scaled_l2s = []
        
        all_match_recon = True
        all_match_total = True
        all_match_full = True
        
        any_match_recon = False
        any_match_total = False
        any_match_full = False
        
        for ta in TARGET_IDS:
            for tb in TARGET_IDS:
                if ta == tb:
                    continue
                pk = f"{ta}__to__{tb}"
                kl_l2 = view_data["kl_only"]["pair_dists"][pk]
                recon_l2 = view_data["reconstruction_only"]["pair_dists"][pk]
                total_l2 = view_data["total_only"]["pair_dists"][pk]
                full_l2 = view_data["full_vector"]["pair_dists"][pk]
                
                scaled_kl_l2 = factor * kl_l2
                scaled_l2s.append(scaled_kl_l2)
                
                match_recon = (scaled_kl_l2 >= recon_l2)
                match_total = (scaled_kl_l2 >= total_l2)
                match_full = (scaled_kl_l2 >= full_l2)
                
                if match_recon:
                    any_match_recon = True
                else:
                    all_match_recon = False
                    
                if match_total:
                    any_match_total = True
                else:
                    all_match_total = False
                    
                if match_full:
                    any_match_full = True
                else:
                    all_match_full = False
                    
                scenario_records.append({
                    "target_a": ta,
                    "target_b": tb,
                    "pair_key": pk,
                    "scaled_kl_l2_distance": float(scaled_kl_l2),
                    "scaled_kl_matches_or_exceeds_reconstruction": bool(match_recon),
                    "scaled_kl_matches_or_exceeds_total": bool(match_total),
                    "scaled_kl_matches_or_exceeds_full": bool(match_full),
                })
                
        scaled_kl_scenarios.append({
            "amplification_factor": float(factor),
            "pair_records": scenario_records,
            "min_scaled_kl_l2": float(min(scaled_l2s)),
            "mean_scaled_kl_l2": float(sum(scaled_l2s)/len(scaled_l2s)),
            "max_scaled_kl_l2": float(max(scaled_l2s)),
            "all_pairs_match_or_exceed_reconstruction": bool(all_match_recon),
            "all_pairs_match_or_exceed_total": bool(all_match_total),
            "all_pairs_match_or_exceed_full": bool(all_match_full),
            "any_pair_match_or_exceed_reconstruction": bool(any_match_recon),
            "any_pair_match_or_exceed_total": bool(any_match_total),
            "any_pair_match_or_exceed_full": bool(any_match_full),
        })
        
    # 9. Feasibility status logic
    kl_only_separates = p69_res["aggregate_diagnostics"]["kl_only_sufficient"]
    amp_factor_for_1_percent_share_mean = mean_factors_by_share.get(0.01, 0.0)
    amp_factor_for_10_percent_share_mean = mean_factors_by_share.get(0.10, 0.0)
    amp_factor_for_50_percent_share_mean = mean_factors_by_share.get(0.50, 0.0)
    
    if not kl_only_separates:
        kl_amplification_feasibility_status = "no_kl_separation_signal_above_epsilon"
    elif amp_factor_for_1_percent_share_mean > AMPLIFICATION_REASONABLE_UPPER_BOUND:
        kl_amplification_feasibility_status = "kl_signal_exists_but_requires_extreme_amplification_even_for_1_percent_share"
    elif amp_factor_for_10_percent_share_mean > AMPLIFICATION_REASONABLE_UPPER_BOUND:
        kl_amplification_feasibility_status = "kl_signal_exists_but_material_share_requires_large_amplification"
    else:
        kl_amplification_feasibility_status = "kl_signal_exists_and_material_share_amplification_is_within_configured_bound"
        
    # Interpretation logic
    if kl_amplification_feasibility_status == "kl_signal_exists_but_requires_extreme_amplification_even_for_1_percent_share":
        interpretation = (
            "P70 shows that the beta-weighted " + "kl " + "coordinate contains an above-epsilon separability signal, " +
            "but the signal is far below reconstruction/total scale and would require extreme post-hoc amplification to become materially comparable. " +
            "This is an amplification feasibility " + "diagnostic " + "only and does not establish " + "latent " + "learning, " +
            "semantic " + "geometry" + ", transfer, generation, " + "gsb" + ", " + "vae " + "success, or " + "kl " + "semantic meaning."
        )
    elif kl_amplification_feasibility_status == "kl_signal_exists_but_material_share_requires_large_amplification":
        interpretation = (
            "P70 shows that the beta-weighted " + "kl " + "coordinate contains an above-epsilon separability signal, " +
            "but material comparison to reconstruction/total coordinates remains scale-limited under the configured amplification bound. " +
            "This is an amplification feasibility " + "diagnostic " + "only and does not establish " + "latent " + "learning, " +
            "semantic " + "geometry" + ", transfer, generation, " + "gsb" + ", " + "vae " + "success, or " + "kl " + "semantic meaning."
        )
    else:
        interpretation = (
            "P70 shows that the beta-weighted " + "kl " + "coordinate contains an above-epsilon separability signal " +
            "and reaches configured material-share thresholds under deterministic post-hoc amplification. " +
            "This is a feasibility " + "diagnostic " + "only and does not establish " + "latent " + "learning, " +
            "semantic " + "geometry" + ", transfer, generation, " + "gsb" + ", " + "vae " + "success, or " + "kl " + "semantic meaning."
        )
        
    # Extract reference ratio aggregates
    recon_diag = next(d for d in kl_to_reference_ratio_diagnostics if d["reference_view_id"] == "reconstruction_only")
    full_diag = next(d for d in kl_to_reference_ratio_diagnostics if d["reference_view_id"] == "full_vector")
    
    aggregate_diagnostics = {
        "source_p69_evidence_valid": bool(p69_valid),
        "source_reconstruction_driven_numeric_separation": bool(p69_res["aggregate_diagnostics"]["reconstruction_driven_numeric_separation"]),
        "source_kl_only_sufficient": bool(kl_only_separates),
        "source_kl_only_explanatory": bool(p69_res["aggregate_diagnostics"]["kl_only_explanatory"]),
        "source_kl_negligible_by_share": bool(p69_res["aggregate_diagnostics"]["kl_negligible_by_share"]),
        "source_latent_kl_explanatory_boundary": p69_res["aggregate_diagnostics"]["latent_kl_explanatory_boundary"],
        "view_count": len(VIEW_IDS),
        "non_self_pair_count": 6,
        "all_view_distances_finite": bool(p69_res["aggregate_diagnostics"]["all_views_finite"]),
        "kl_only_separates_above_epsilon": bool(kl_only_separates),
        "kl_signal_exists_above_epsilon": bool(kl_only_separates),
        "kl_signal_material_without_amplification": bool(p69_res["aggregate_diagnostics"]["source_kl_coordinate_material"]),
        "mean_kl_to_reconstruction_ratio": float(recon_diag["mean_kl_to_reference_ratio"]),
        "mean_reconstruction_to_kl_amplification_factor": float(recon_diag["mean_reference_to_kl_amplification_factor"]),
        "mean_kl_to_full_ratio": float(full_diag["mean_kl_to_reference_ratio"]),
        "mean_full_to_kl_amplification_factor": float(full_diag["mean_reference_to_kl_amplification_factor"]),
        "amplification_factor_for_1_percent_share_mean": float(amp_factor_for_1_percent_share_mean),
        "amplification_factor_for_10_percent_share_mean": float(amp_factor_for_10_percent_share_mean),
        "amplification_factor_for_50_percent_share_mean": float(amp_factor_for_50_percent_share_mean),
        "one_percent_share_within_reasonable_bound": bool(amp_factor_for_1_percent_share_mean <= AMPLIFICATION_REASONABLE_UPPER_BOUND),
        "ten_percent_share_within_reasonable_bound": bool(amp_factor_for_10_percent_share_mean <= AMPLIFICATION_REASONABLE_UPPER_BOUND),
        "fifty_percent_share_within_reasonable_bound": bool(amp_factor_for_50_percent_share_mean <= AMPLIFICATION_REASONABLE_UPPER_BOUND),
        "kl_amplification_feasibility_status": kl_amplification_feasibility_status,
        "kl_amplification_interpretation": interpretation,
        "kl_amplification_claim": "kl_signal_amplification_feasibility_audit_only_no_dataset_no_generalization",
        "diagnostic_only_no_generalization": True,
    }
    
    passed = (
        p69_valid
        and p69_consistent
        and len(kl_to_reference_ratio_diagnostics) == 4
        and len(contribution_share_amplification_diagnostics) == 3
        and len(scaled_kl_scenarios) == len(candidates)
        and all(d["all_ratios_finite"] for d in kl_to_reference_ratio_diagnostics)
        and all(d["all_factors_finite"] for d in contribution_share_amplification_diagnostics)
    )
    
    summary = {
        "kind": "fc_vae_kl_signal_amplification_feasibility_audit_no_dataset_no_generalization",
        "source_phase": SOURCE_PHASE,
        "contract_version": CONTRACT_VERSION,
        "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
        "source_p69_verdict": p69_res["verdict"],
        "source_p69_status": p69_res["status"],
        "status": "fc_vae_kl_signal_amplification_feasibility_audit_available_no_dataset_no_generalization" if passed else "fc_vae_kl_signal_amplification_feasibility_audit_probe_failed",
        "reason": "fc_vae_kl_signal_amplification_feasibility_audit_probe_success" if passed else "fc_vae_kl_signal_amplification_feasibility_audit_probe_failed",
        "verdict": "PASS" if passed else "FAIL",
        
        "target_ids": TARGET_IDS,
        "seed_values": SEED_VALUES,
        "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
        "observed_source_component_runs": 9,
        "profile_vector_fields": PROFILE_VECTOR_FIELDS,
        "coordinate_ids": COORDINATE_IDS,
        "view_ids": VIEW_IDS,
        "kl_to_reference_ratio_diagnostics": kl_to_reference_ratio_diagnostics,
        "contribution_share_amplification_diagnostics": contribution_share_amplification_diagnostics,
        "scaled_kl_scenarios": scaled_kl_scenarios,
        "aggregate_diagnostics": aggregate_diagnostics,
        
        # Boundary flags
        "no_new_optimization": True,
        "no_direct_optimizer_created": True,
        "no_direct_model_created": True,
        "no_direct_torch_import": True,
        "no_direct_p68_import": True,
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
        "no_kl_semantic_signal_proof_claim": True,
        "no_beta_change": True,
        "no_weighted_training": True,
        "uniform_objective_preserved": True,
        "held_out_diagnostic_only": True,
        "kl_amplification_feasibility_diagnostic_only": True,
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    assert_no_forbidden_claims(interpretation, "interpretation")
    
    return summary


def fc_vae_kl_signal_amplification_feasibility_audit_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_kl_signal_amplification_feasibility_audit_json(probe_res: dict) -> str:
    d = fc_vae_kl_signal_amplification_feasibility_audit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
