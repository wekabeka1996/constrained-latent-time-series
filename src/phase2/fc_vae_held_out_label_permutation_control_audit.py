# src/phase2/fc_vae_held_out_label_permutation_control_audit.py

import json
import math
from typing import Any, Dict, List

from src.phase2.fc_vae_held_out_component_attribution_audit import (
    run_p64_held_out_component_attribution_audit_probe,
)

# Constants
SOURCE_PHASE = "P65"
CONTRACT_VERSION = "phase2_p65_fc_vae_held_out_label_permutation_control_audit_contract_v1"
SOURCE_EVIDENCE_PHASE = "P64"
TARGET_IDS = ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
SEED_VALUES = [62062, 62162, 62262]
EXPECTED_SOURCE_COMPONENT_RUNS = 9
PERMUTATION_IDS = ["cyclic_forward", "cyclic_backward"]
NUMERIC_COMPARISON_TOLERANCE = 1e-12

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
        "label-specific " + "semantic proof",
    ]
    for item in forbidden_substrings:
        if item in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}: {item}")


def _is_bijective(m: dict, domain: list) -> bool:
    if set(m.keys()) != set(domain):
        return False
    if set(m.values()) != set(domain):
        return False
    return len(set(m.values())) == len(domain)


def _has_no_fixed_points(m: dict) -> bool:
    return all(k != v for k, v in m.items())


def _build_target_summary_from_records(records: list, target_id_field: str) -> dict:
    summary = {}
    for tid in TARGET_IDS:
        target_runs = [r for r in records if r[target_id_field] == tid]
        # Sort by seed order
        target_runs_sorted = sorted(target_runs, key=lambda r: SEED_VALUES.index(r["seed_value"]))
        
        run_count = len(target_runs)
        if run_count > 0:
            mean_tot = sum(r["total_delta_value"] for r in target_runs) / run_count
            mean_rec = sum(r["reconstruction_delta_value"] for r in target_runs) / run_count
            mean_beta_kl = sum(r["beta_weighted_kl_delta_value"] for r in target_runs) / run_count
        else:
            mean_tot = 0.0
            mean_rec = 0.0
            mean_beta_kl = 0.0
            
        tot_improved = sum(1 for r in target_runs if r["total_improved"])
        rec_supported = sum(
            1 for r in target_runs
            if r["component_attribution_class"] in ["reconstruction_supported", "mixed_reconstruction_supported_kl_opposed"]
        )
        kl_only = sum(1 for r in target_runs if r["component_attribution_class"] == "kl_supported")
        mixed_or_deg = sum(1 for r in target_runs if r["component_attribution_class"] == "mixed_or_degraded")
        kl_opp = sum(1 for r in target_runs if r["kl_opposes_reconstruction"])
        
        classes = [r["component_attribution_class"] for r in target_runs_sorted]
        tot_deltas = [r["total_delta_value"] for r in target_runs_sorted]
        rec_deltas = [r["reconstruction_delta_value"] for r in target_runs_sorted]
        beta_kl_deltas = [r["beta_weighted_kl_delta_value"] for r in target_runs_sorted]
        
        summary[tid] = {
            "held_out_target_id": tid,
            "run_count": run_count,
            "mean_total_delta_value": float(mean_tot),
            "mean_reconstruction_delta_value": float(mean_rec),
            "mean_beta_weighted_kl_delta_value": float(mean_beta_kl),
            "total_improved_count": tot_improved,
            "reconstruction_supported_count": rec_supported,
            "kl_only_count": kl_only,
            "mixed_or_degraded_count": mixed_or_deg,
            "kl_opposition_count": kl_opp,
            "component_attribution_classes_by_seed": classes,
            "total_delta_values_by_seed": tot_deltas,
            "reconstruction_delta_values_by_seed": rec_deltas,
            "beta_weighted_kl_delta_values_by_seed": beta_kl_deltas,
        }
    return summary


def run_p65_held_out_label_permutation_control_audit_probe() -> dict:
    # 1. Call P64 evidence
    p64_res = run_p64_held_out_component_attribution_audit_probe()
    
    # 2. Validate P64 source evidence
    p64_valid = True
    if not isinstance(p64_res, dict):
        p64_valid = False
    elif p64_res.get("source_phase") != "P64":
        p64_valid = False
    elif p64_res.get("verdict") != "PASS":
        p64_valid = False
    elif p64_res.get("source_evidence_phase") != "P63":
        p64_valid = False
    elif p64_res.get("observed_seed_fold_runs") != 9:
        p64_valid = False
    elif p64_res.get("target_ids") != TARGET_IDS:
        p64_valid = False
    elif p64_res.get("seed_values") != SEED_VALUES:
        p64_valid = False
    elif p64_res.get("no_new_optimization") is not True:
        p64_valid = False
    elif p64_res.get("component_attribution_diagnostic_only") is not True:
        p64_valid = False
        
    if p64_valid:
        agg_p64 = p64_res.get("aggregate_diagnostics", {})
        if not isinstance(agg_p64, dict):
            p64_valid = False
        elif agg_p64.get("source_p63_evidence_valid") is not True:
            p64_valid = False
        elif agg_p64.get("all_component_residuals_within_tolerance") is not True:
            p64_valid = False
            
    if not p64_valid:
        return {
            "kind": "fc_vae_held_out_label_permutation_control_audit_no_dataset_no_generalization",
            "source_phase": SOURCE_PHASE,
            "contract_version": CONTRACT_VERSION,
            "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
            "source_p64_verdict": p64_res.get("verdict") if isinstance(p64_res, dict) else "ERROR",
            "source_p64_status": p64_res.get("status") if isinstance(p64_res, dict) else "ERROR",
            "status": "blocked_by_invalid_p64_evidence",
            "reason": "p64_evidence_validation_failed",
            "verdict": "FAIL",
            
            "target_ids": TARGET_IDS,
            "seed_values": SEED_VALUES,
            "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
            "observed_source_component_runs": 0,
            "permutation_ids": PERMUTATION_IDS,
            "original_target_summary": {},
            "permutation_summaries": {},
            "aggregate_diagnostics": {},
            
            # Boundary flags
            "no_new_optimization": True,
            "no_direct_optimizer_created": True,
            "no_direct_model_created": True,
            "no_direct_torch_import": True,
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
            "no_weighted_training": True,
            "uniform_objective_preserved": True,
            "held_out_diagnostic_only": True,
            "label_permutation_control_diagnostic_only": True,
        }
        
    # 3. Extract original component summaries and build target summary
    original_component_summaries = p64_res["component_fold_summaries"]
    original_target_summary = _build_target_summary_from_records(
        original_component_summaries, "held_out_target_id"
    )
    
    # Check permutations properties
    all_permutation_maps_bijective = True
    all_permutation_maps_have_no_fixed_points = True
    for pid in PERMUTATION_IDS:
        pmap = PERMUTATION_MAPS[pid]
        if not _is_bijective(pmap, TARGET_IDS):
            all_permutation_maps_bijective = False
        if not _has_no_fixed_points(pmap):
            all_permutation_maps_have_no_fixed_points = False
            
    permutation_summaries = {}
    all_permutation_records_present = True
    all_labels_changed_under_permutation = True
    all_permutation_summaries_finite = True
    
    # 4. For each deterministic permutation map
    for pid in PERMUTATION_IDS:
        pmap = PERMUTATION_MAPS[pid]
        control_records = []
        
        for orig in original_component_summaries:
            orig_target = orig["held_out_target_id"]
            perm_target = pmap[orig_target]
            label_changed = (orig_target != perm_target)
            if not label_changed:
                all_labels_changed_under_permutation = False
                
            control_records.append({
                "permutation_id": pid,
                "seed_value": orig["seed_value"],
                "fold_id": orig["fold_id"],
                "original_held_out_target_id": orig_target,
                "permuted_held_out_target_id": perm_target,
                "train_target_ids": orig["train_target_ids"],
                "label_changed": bool(label_changed),
                "total_delta_value": float(orig["total_delta_value"]),
                "reconstruction_delta_value": float(orig["reconstruction_delta_value"]),
                "beta_weighted_kl_delta_value": float(orig["beta_weighted_kl_delta_value"]),
                "component_attribution_class": orig["component_attribution_class"],
                "total_improved": bool(orig["total_improved"]),
                "reconstruction_improved": bool(orig["reconstruction_improved"]),
                "kl_opposes_reconstruction": bool(orig["kl_opposes_reconstruction"]),
                "component_fold_passed": bool(orig["component_fold_passed"]),
            })
            
        if len(control_records) != EXPECTED_SOURCE_COMPONENT_RUNS:
            all_permutation_records_present = False
            
        # Aggregate permuted summaries by permuted_held_out_target_id
        permuted_target_summaries = _build_target_summary_from_records(
            control_records, "permuted_held_out_target_id"
        )
        
        # Compare original target summary vs permuted target summary
        comparison_to_original = {}
        for tid in TARGET_IDS:
            orig_sum = original_target_summary[tid]
            perm_sum = permuted_target_summaries[tid]
            
            orig_tot = orig_sum["mean_total_delta_value"]
            perm_tot = perm_sum["mean_total_delta_value"]
            tot_abs_diff = abs(orig_tot - perm_tot)
            
            orig_rec = orig_sum["mean_reconstruction_delta_value"]
            perm_rec = perm_sum["mean_reconstruction_delta_value"]
            rec_abs_diff = abs(orig_rec - perm_rec)
            
            orig_rec_supported_count = orig_sum["reconstruction_supported_count"]
            perm_rec_supported_count = perm_sum["reconstruction_supported_count"]
            rec_supported_count_changed = (orig_rec_supported_count != perm_rec_supported_count)
            
            orig_kl_opp_count = orig_sum["kl_opposition_count"]
            perm_kl_opp_count = perm_sum["kl_opposition_count"]
            kl_opp_count_changed = (orig_kl_opp_count != perm_kl_opp_count)
            
            numeric_profile_changed = (
                (tot_abs_diff > NUMERIC_COMPARISON_TOLERANCE)
                or (rec_abs_diff > NUMERIC_COMPARISON_TOLERANCE)
            )
            qualitative_profile_changed = rec_supported_count_changed or kl_opp_count_changed
            
            # check finiteness
            finite_vals = (
                math.isfinite(tot_abs_diff)
                and math.isfinite(rec_abs_diff)
                and math.isfinite(orig_tot)
                and math.isfinite(perm_tot)
                and math.isfinite(orig_rec)
                and math.isfinite(perm_rec)
            )
            if not finite_vals:
                all_permutation_summaries_finite = False
                
            comparison_to_original[tid] = {
                "target_id": tid,
                "original_mean_total_delta_value": float(orig_tot),
                "permuted_mean_total_delta_value": float(perm_tot),
                "mean_total_delta_abs_diff": float(tot_abs_diff),
                "original_mean_reconstruction_delta_value": float(orig_rec),
                "permuted_mean_reconstruction_delta_value": float(perm_rec),
                "mean_reconstruction_delta_abs_diff": float(rec_abs_diff),
                "original_reconstruction_supported_count": orig_rec_supported_count,
                "permuted_reconstruction_supported_count": perm_rec_supported_count,
                "reconstruction_supported_count_changed": bool(rec_supported_count_changed),
                "original_kl_opposition_count": orig_kl_opp_count,
                "permuted_kl_opposition_count": perm_kl_opp_count,
                "kl_opposition_count_changed": bool(kl_opp_count_changed),
                "numeric_profile_changed": bool(numeric_profile_changed),
                "qualitative_profile_changed": bool(qualitative_profile_changed),
            }
            
        permutation_passed = (
            _is_bijective(pmap, TARGET_IDS)
            and _has_no_fixed_points(pmap)
            and len(control_records) == EXPECTED_SOURCE_COMPONENT_RUNS
            and all(r["label_changed"] for r in control_records)
            and all_permutation_summaries_finite
        )
        
        permutation_summaries[pid] = {
            "permutation_id": pid,
            "permutation_map": pmap,
            "mapping_is_bijective": bool(_is_bijective(pmap, TARGET_IDS)),
            "mapping_has_no_fixed_points": bool(_has_no_fixed_points(pmap)),
            "control_record_count": len(control_records),
            "label_assignment_changed_count": sum(1 for r in control_records if r["label_changed"]),
            "permuted_target_summaries": permuted_target_summaries,
            "comparison_to_original": comparison_to_original,
            "permutation_passed": bool(permutation_passed),
        }
        
    # 5. Check qualitative invariants and warnings
    original_all_targets_reconstruction_supported = all(
        original_target_summary[tid]["reconstruction_supported_count"] == original_target_summary[tid]["run_count"]
        for tid in TARGET_IDS
    )
    
    permutation_all_targets_reconstruction_supported = all(
        all(
            perm_sum["permuted_target_summaries"][tid]["reconstruction_supported_count"]
            == perm_sum["permuted_target_summaries"][tid]["run_count"]
            for tid in TARGET_IDS
        )
        for perm_sum in permutation_summaries.values()
    )
    
    qualitative_reconstruction_supported_signal_label_invariant = (
        original_all_targets_reconstruction_supported
        and permutation_all_targets_reconstruction_supported
    )
    
    original_all_targets_improved = all(
        original_target_summary[tid]["total_improved_count"] == original_target_summary[tid]["run_count"]
        for tid in TARGET_IDS
    )
    
    permutation_all_targets_improved = all(
        all(
            perm_sum["permuted_target_summaries"][tid]["total_improved_count"]
            == perm_sum["permuted_target_summaries"][tid]["run_count"]
            for tid in TARGET_IDS
        )
        for perm_sum in permutation_summaries.values()
    )
    
    qualitative_all_improved_signal_label_invariant = (
        original_all_targets_improved
        and permutation_all_targets_improved
    )
    
    label_specificity_warning = (
        qualitative_all_improved_signal_label_invariant
        or qualitative_reconstruction_supported_signal_label_invariant
    )
    
    # Numeric profile changes
    targets_with_numeric_profile_change = []
    targets_with_qualitative_profile_change = []
    for tid in TARGET_IDS:
        has_num = False
        has_qual = False
        for pid in PERMUTATION_IDS:
            comp = permutation_summaries[pid]["comparison_to_original"][tid]
            if comp["numeric_profile_changed"]:
                has_num = True
            if comp["qualitative_profile_changed"]:
                has_qual = True
        if has_num:
            targets_with_numeric_profile_change.append(tid)
        if has_qual:
            targets_with_qualitative_profile_change.append(tid)
            
    numeric_target_profiles_changed_under_permutation = len(targets_with_numeric_profile_change) > 0
    
    if label_specificity_warning:
        negative_control_interpretation = (
            "WARNING: The qualitative target-level diagnostic signal remains invariant "
            "under label permutation. The diagnostic all-improved or reconstruction-supported "
            "results do not establish label-specific semantic structure."
        )
    else:
        negative_control_interpretation = (
            "PASS: The qualitative diagnostic signal is label-specific and changes under label permutation."
        )
        
    aggregate_diagnostics = {
        "source_p64_evidence_valid": bool(p64_valid),
        "source_component_runs_count": len(original_component_summaries),
        "all_permutation_maps_bijective": bool(all_permutation_maps_bijective),
        "all_permutation_maps_have_no_fixed_points": bool(all_permutation_maps_have_no_fixed_points),
        "all_permutation_records_present": bool(all_permutation_records_present),
        "all_labels_changed_under_permutation": bool(all_labels_changed_under_permutation),
        "all_permutation_summaries_finite": bool(all_permutation_summaries_finite),
        "original_all_targets_reconstruction_supported": bool(original_all_targets_reconstruction_supported),
        "permutation_all_targets_reconstruction_supported": bool(permutation_all_targets_reconstruction_supported),
        "qualitative_all_improved_signal_label_invariant": bool(qualitative_all_improved_signal_label_invariant),
        "qualitative_reconstruction_supported_signal_label_invariant": bool(qualitative_reconstruction_supported_signal_label_invariant),
        "numeric_target_profiles_changed_under_permutation": bool(numeric_target_profiles_changed_under_permutation),
        "targets_with_numeric_profile_change": targets_with_numeric_profile_change,
        "targets_with_qualitative_profile_change": targets_with_qualitative_profile_change,
        "label_specificity_warning": bool(label_specificity_warning),
        "negative_control_interpretation": negative_control_interpretation,
        "label_permutation_control_claim": "label_permutation_control_audit_only_no_new_optimization_no_dataset_no_generalization",
        "diagnostic_only_no_generalization": True,
    }
    
    # Verdict passes if:
    # 1. P64 source evidence is valid and PASS
    # 2. Exactly 9 original component records are present
    # 3. Both permutation maps are bijective
    # 4. Both permutation maps have no fixed points
    # 5. Each permutation produces exactly 9 control records
    # 6. All control records change labels
    # 7. All numeric summaries are finite
    # Note: label_specificity_warning does NOT fail P65.
    passed = (
        p64_valid
        and len(original_component_summaries) == EXPECTED_SOURCE_COMPONENT_RUNS
        and all_permutation_maps_bijective
        and all_permutation_maps_have_no_fixed_points
        and all_permutation_records_present
        and all_labels_changed_under_permutation
        and all_permutation_summaries_finite
    )
    
    summary = {
        "kind": "fc_vae_held_out_label_permutation_control_audit_no_dataset_no_generalization",
        "source_phase": SOURCE_PHASE,
        "contract_version": CONTRACT_VERSION,
        "source_evidence_phase": SOURCE_EVIDENCE_PHASE,
        "source_p64_verdict": p64_res["verdict"],
        "source_p64_status": p64_res["status"],
        "status": "fc_vae_held_out_label_permutation_control_audit_available_no_dataset_no_generalization" if passed else "fc_vae_held_out_label_permutation_control_audit_probe_failed",
        "reason": "fc_vae_held_out_label_permutation_control_audit_probe_success" if passed else "fc_vae_held_out_label_permutation_control_audit_probe_failed",
        "verdict": "PASS" if passed else "FAIL",
        
        "target_ids": TARGET_IDS,
        "seed_values": SEED_VALUES,
        "expected_source_component_runs": EXPECTED_SOURCE_COMPONENT_RUNS,
        "observed_source_component_runs": len(original_component_summaries),
        "permutation_ids": PERMUTATION_IDS,
        "original_target_summary": original_target_summary,
        "permutation_summaries": permutation_summaries,
        "aggregate_diagnostics": aggregate_diagnostics,
        
        # Boundary flags
        "no_new_optimization": True,
        "no_direct_optimizer_created": True,
        "no_direct_model_created": True,
        "no_direct_torch_import": True,
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
        "no_weighted_training": True,
        "uniform_objective_preserved": True,
        "held_out_diagnostic_only": True,
        "label_permutation_control_diagnostic_only": True,
    }
    
    validate_non_empty_str(summary["contract_version"], "contract_version")
    validate_non_empty_str(summary["status"], "status")
    validate_non_empty_str(summary["reason"], "reason")
    assert_no_local_path_leakage(summary["reason"], "reason")
    assert_no_forbidden_claims(summary["reason"], "reason")
    assert_no_forbidden_claims(negative_control_interpretation, "negative_control_interpretation")
    
    return summary


def fc_vae_held_out_label_permutation_control_audit_probe_to_json_dict(probe_res: dict) -> dict:
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


def compact_fc_vae_held_out_label_permutation_control_audit_json(probe_res: dict) -> str:
    d = fc_vae_held_out_label_permutation_control_audit_probe_to_json_dict(probe_res)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
