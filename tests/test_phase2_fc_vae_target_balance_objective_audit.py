# tests/test_phase2_fc_vae_target_balance_objective_audit.py

import json
import math
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.fc_vae_target_balance_objective_audit import (
    FC_VAE_TARGET_BALANCE_AUDIT_CONTRACT_VERSION,
    FC_VAE_TARGET_BALANCE_AUDIT_KIND,
    FC_VAE_TARGET_BALANCE_AUDIT_MODULE_NAME,
    FC_VAE_TARGET_BALANCE_AUDIT_STATUS_AVAILABLE,
    P60_TARGET_COUNT,
    P60_EXPECTED_TARGET_IDS,
    P60_DOMINANCE_SHARE_WARNING_THRESHOLD,
    P60_DOMINANCE_SHARE_BLOCK_THRESHOLD,
    P60_DELTA_SPREAD_WARNING_THRESHOLD,
    P60_IMPROVEMENT_RATIO_WARNING_THRESHOLD,
    P60_RECOMMENDATION_STATUS_NO_WEIGHTING_REQUIRED,
    P60_RECOMMENDATION_STATUS_WEIGHTING_WATCH,
    P60_RECOMMENDATION_STATUS_WEIGHTING_REQUIRED_BEFORE_EXPANSION,
    load_torch_for_p60_balance_audit,
    extract_p60_balance_inputs_from_p59_probe,
    compute_p60_target_balance_metrics,
    compute_p60_counterfactual_weighting_diagnostics,
    derive_p60_balance_recommendation,
    run_fc_vae_target_balance_objective_audit_probe,
    fc_vae_target_balance_objective_audit_probe_to_json_dict,
)
from src.phase2.fc_vae_reconstruction_trajectory_diagnostics import (
    run_fc_vae_reconstruction_trajectory_diagnostics_probe,
)


def test_p60_01_constants_exact():
    assert FC_VAE_TARGET_BALANCE_AUDIT_CONTRACT_VERSION == "phase2_p60_fc_vae_target_balance_objective_audit_contract_v1"
    assert FC_VAE_TARGET_BALANCE_AUDIT_KIND == "fc_vae_target_balance_objective_audit_no_training_change_no_dataset"
    assert FC_VAE_TARGET_BALANCE_AUDIT_MODULE_NAME == "src.phase2.fc_vae_target_balance_objective_audit"
    assert FC_VAE_TARGET_BALANCE_AUDIT_STATUS_AVAILABLE == "fc_vae_target_balance_objective_audit_available_no_training_change_no_dataset"
    assert P60_TARGET_COUNT == 3
    assert P60_EXPECTED_TARGET_IDS == ("bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75")
    assert P60_DOMINANCE_SHARE_WARNING_THRESHOLD == 0.40
    assert P60_DOMINANCE_SHARE_BLOCK_THRESHOLD == 0.50
    assert P60_DELTA_SPREAD_WARNING_THRESHOLD == 0.002
    assert P60_IMPROVEMENT_RATIO_WARNING_THRESHOLD == 2.0
    
    assert P60_RECOMMENDATION_STATUS_NO_WEIGHTING_REQUIRED == "no_weighting_required_for_next_phase"
    assert P60_RECOMMENDATION_STATUS_WEIGHTING_WATCH == "weighting_watch_recommended"
    assert P60_RECOMMENDATION_STATUS_WEIGHTING_REQUIRED_BEFORE_EXPANSION == "weighting_required_before_expansion"
 
 
def test_p60_02_no_top_level_torch_import():
    filepath = "src/phase2/fc_vae_target_balance_objective_audit.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"
 
 
def test_p60_03_no_optimizer_import_in_p60_source():
    p = pathlib.Path("src/phase2/fc_vae_target_balance_objective_audit.py").read_text(encoding="utf-8")
    assert "torch.optim" not in p
    assert "SGD" not in p
    assert "Adam" not in p


def test_p60_04_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_target_balance_objective_audit.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p60_05_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_target_balance_objective_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p60_06_no_optimizer_or_scheduler_creation():
    p = pathlib.Path("src/phase2/fc_vae_target_balance_objective_audit.py").read_text(encoding="utf-8")
    assert "optimizer" not in p.lower() or "no_optimizer" in p.lower()
    assert "scheduler" not in p.lower() or "no_scheduler" in p.lower()


def test_p60_07_no_dataset_dataloader_imports():
    p = pathlib.Path("src/phase2/fc_vae_target_balance_objective_audit.py").read_text(encoding="utf-8")
    assert "DataLoader" not in p
    assert "Dataset" not in p


def test_p60_08_no_epoch_batch_loop_strings():
    p = pathlib.Path("src/phase2/fc_vae_target_balance_objective_audit.py").read_text(encoding="utf-8")
    assert "for epoch" not in p.lower()
    assert "for batch" not in p.lower()


def test_p60_09_allowed_p59_imports_present():
    p = pathlib.Path("src/phase2/fc_vae_target_balance_objective_audit.py").read_text(encoding="utf-8")
    assert "load_torch_for_p59_trajectory_diagnostics" in p
    assert "run_p59_reconstruction_trajectory_diagnostics_once" in p
    assert "run_fc_vae_reconstruction_trajectory_diagnostics_probe" in p


def test_p60_10_extract_inputs_validates_p59_pass():
    # Pass mock
    mock_pass = {
        "verdict": "PASS",
        "status": "fc_vae_reconstruction_trajectory_diagnostics_available_no_dataset_no_generalization",
        "target_count": 3,
        "target_ids": ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"],
        "per_target_diagnostics": {
            "bridge_lambda_0_25": {"initial_total_loss": 0.88, "final_total_loss": 0.87, "total_loss_delta": -0.01,
                                   "initial_reconstruction_loss": 0.88, "final_reconstruction_loss": 0.87, "reconstruction_loss_delta": -0.01,
                                   "initial_kl_loss": 0.2, "final_kl_loss": 0.2, "kl_loss_delta": 0.0},
            "bridge_lambda_0_5": {"initial_total_loss": 0.82, "final_total_loss": 0.81, "total_loss_delta": -0.01,
                                  "initial_reconstruction_loss": 0.82, "final_reconstruction_loss": 0.81, "reconstruction_loss_delta": -0.01,
                                  "initial_kl_loss": 0.2, "final_kl_loss": 0.2, "kl_loss_delta": 0.0},
            "bridge_lambda_0_75": {"initial_total_loss": 0.91, "final_total_loss": 0.90, "total_loss_delta": -0.01,
                                   "initial_reconstruction_loss": 0.91, "final_reconstruction_loss": 0.90, "reconstruction_loss_delta": -0.01,
                                   "initial_kl_loss": 0.2, "final_kl_loss": 0.2, "kl_loss_delta": 0.0},
        },
        "per_target_trajectories": {},
        "best_worst_diagnostics": {},
        "dominance_diagnostics": {},
        "initial_loss_value": 0.87,
        "final_step_loss_value": 0.86,
        "final_objective_loss_value": 0.86,
        "loss_delta_value": -0.01,
        "loss_decreased": True,
        
        "no_dataset": True, "no_dataloader": True, "no_epoch_loop": True, "no_batch_loop": True,
        "no_scheduler": True, "no_checkpointing": True, "no_generalization_claim": True,
        "no_generation_claim": True, "no_gsb_claim": True, "no_scientific_conclusion": True,
        "no_latent_learning_claim": True, "no_vae_success_claim": True, "no_convergence_claim": True,
        "no_semantic_geometry_proof_claim": True, "reason": "probe_success"
    }
    extracted = extract_p60_balance_inputs_from_p59_probe(mock_pass)
    assert extracted["target_count"] == 3
    
    # Check invalid verdict raises ValueError
    mock_fail = mock_pass.copy()
    mock_fail["verdict"] = "FAIL"
    with pytest.raises(ValueError):
        extract_p60_balance_inputs_from_p59_probe(mock_fail)


def test_p60_11_target_count_exactly_three():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    assert p59_probe["target_count"] == 3


def test_p60_12_target_ids_exact_and_ordered():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    assert p59_probe["target_ids"] == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]


def test_p60_13_balance_metrics_finite():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    
    assert metrics["initial_total_loss_sum"] > 0.0
    assert metrics["final_total_loss_sum"] > 0.0
    assert math.isfinite(metrics["initial_total_loss_sum"])
    assert math.isfinite(metrics["final_total_loss_sum"])


def test_p60_14_all_per_target_improvement_values_present():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    
    for tid in p59_probe["target_ids"]:
        m = metrics["per_target_metrics"][tid]
        assert m["absolute_total_improvement"] > 0.0
        assert m["absolute_reconstruction_improvement"] > 0.0
        assert m["absolute_kl_improvement"] > 0.0


def test_p60_15_all_target_shares_sum_approximately_one():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    
    initial_shares_sum = sum(metrics["per_target_metrics"][tid]["initial_total_share"] for tid in p59_probe["target_ids"])
    final_shares_sum = sum(metrics["per_target_metrics"][tid]["final_total_share"] for tid in p59_probe["target_ids"])
    
    assert abs(initial_shares_sum - 1.0) < 1e-5
    assert abs(final_shares_sum - 1.0) < 1e-5


def test_p60_16_dominant_target_present():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    
    assert metrics["initial_dominant_target"] in p59_probe["target_ids"]
    assert metrics["final_dominant_target"] in p59_probe["target_ids"]
    assert metrics["initial_dominant_share"] > 0.0
    assert metrics["final_dominant_share"] > 0.0


def test_p60_17_dominance_warning_blocker_booleans_present():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    
    assert isinstance(metrics["dominance_warning"], bool)
    assert isinstance(metrics["dominance_blocker"], bool)
    assert isinstance(metrics["spread_warning"], bool)
    assert isinstance(metrics["ratio_warning"], bool)


def test_p60_18_all_target_improved_booleans_present():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    
    assert metrics["all_targets_total_loss_decreased"] is True
    assert metrics["all_targets_reconstruction_loss_decreased"] is True
    assert metrics["all_targets_kl_loss_decreased"] is True


def test_p60_19_counterfactual_weighting_schemes_present():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    weighting = compute_p60_counterfactual_weighting_diagnostics(metrics)
    
    expected_schemes = [
        "uniform_current",
        "inverse_initial_loss_balanced",
        "proportional_initial_loss_hard_target_emphasis",
        "proportional_final_loss_hard_target_emphasis"
    ]
    for sch in expected_schemes:
        assert sch in weighting


def test_p60_20_each_weighting_scheme_weights_sum_to_one():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    weighting = compute_p60_counterfactual_weighting_diagnostics(metrics)
    
    for sch, info in weighting.items():
        w_sum = sum(info["weights_by_target"].values())
        assert abs(w_sum - 1.0) < 1e-5


def test_p60_21_each_weighting_scheme_has_finite_weighted_objectives():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    weighting = compute_p60_counterfactual_weighting_diagnostics(metrics)
    
    for sch, info in weighting.items():
        assert math.isfinite(info["weighted_initial_objective"])
        assert math.isfinite(info["weighted_final_objective"])
        assert math.isfinite(info["weighted_delta"])
        assert isinstance(info["weighted_loss_decreased"], bool)


def test_p60_22_recommendation_status_in_allowed_enum():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    weighting = compute_p60_counterfactual_weighting_diagnostics(metrics)
    rec = derive_p60_balance_recommendation(metrics, weighting)
    
    allowed = [
        P60_RECOMMENDATION_STATUS_NO_WEIGHTING_REQUIRED,
        P60_RECOMMENDATION_STATUS_WEIGHTING_WATCH,
        P60_RECOMMENDATION_STATUS_WEIGHTING_REQUIRED_BEFORE_EXPANSION
    ]
    assert rec["recommendation_status"] in allowed


def test_p60_23_recommendation_has_watch_hardest_best_worst_ids():
    p59_probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    balance_inputs = extract_p60_balance_inputs_from_p59_probe(p59_probe)
    metrics = compute_p60_target_balance_metrics(balance_inputs)
    weighting = compute_p60_counterfactual_weighting_diagnostics(metrics)
    rec = derive_p60_balance_recommendation(metrics, weighting)
    
    assert rec["hardest_target_id"] in p59_probe["target_ids"]
    assert rec["most_improved_target_id"] in p59_probe["target_ids"]
    assert rec["least_improved_target_id"] in p59_probe["target_ids"]
    assert isinstance(rec["watch_targets"], list)


def test_p60_24_aggregate_probe_pass():
    probe = run_fc_vae_target_balance_objective_audit_probe()
    assert probe["verdict"] == "PASS"
    assert probe["status"] == "fc_vae_target_balance_objective_audit_available_no_training_change_no_dataset"


def test_p60_25_serialization_excludes_tensors():
    probe = run_fc_vae_target_balance_objective_audit_probe()
    json_dict = fc_vae_target_balance_objective_audit_probe_to_json_dict(probe)
    
    for k, v in json_dict.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, dict):
                    for k3, v3 in sub_v.items():
                        assert not hasattr(v3, "shape")
                elif isinstance(sub_v, list):
                    for item in sub_v:
                        assert not hasattr(item, "shape")
                else:
                    assert not hasattr(sub_v, "shape")
        assert not hasattr(v, "shape"), f"Tensor leaked in JSON: {k}"


def test_p60_26_no_training_change_flags():
    probe = run_fc_vae_target_balance_objective_audit_probe()
    assert probe["no_training_change"] is True
    assert probe["no_optimizer_created_in_p60"] is True
    assert probe["no_weighted_training_applied"] is True


def test_p60_27_no_dataset_dataloader_flags():
    probe = run_fc_vae_target_balance_objective_audit_probe()
    assert probe["no_dataset"] is True
    assert probe["no_dataloader"] is True
    assert probe["no_epoch_loop"] is True
    assert probe["no_batch_loop"] is True
    assert probe["no_scheduler"] is True
    assert probe["no_checkpointing"] is True


def test_p60_28_no_forbidden_claims_in_probe():
    probe = run_fc_vae_target_balance_objective_audit_probe()
    assert probe["no_generalization_claim"] is True
    assert probe["no_generation_claim"] is True
    assert probe["no_gsb_claim"] is True
    assert probe["no_scientific_conclusion"] is True
    assert probe["no_latent_learning_claim"] is True
    assert probe["no_vae_success_claim"] is True
    assert probe["no_convergence_claim"] is True
    assert probe["no_semantic_geometry_proof_claim"] is True


def test_p60_29_no_forbidden_claims_in_source():
    p = pathlib.Path("src/phase2/fc_vae_target_balance_objective_audit.py").read_text(encoding="utf-8")
    p_lower = p.lower()
    
    assert "vae works" not in p_lower
    assert "latent space learned" not in p_lower
    assert "semantic geometry proven" not in p_lower
    assert "c generated" not in p_lower
    assert "gsb implemented" not in p_lower
    assert "posterior collapse solved" not in p_lower
    assert "scientific success" not in p_lower
    assert "vae solved" not in p_lower
    assert "model converged" not in p_lower
    assert "model trained" not in p_lower
    assert "model generalized" not in p_lower
    assert "generalization claim" not in p_lower
    
    rpt = pathlib.Path("reports/PHASE_2_P60_FC_VAE_TARGET_BALANCE_OBJECTIVE_AUDIT_NO_TRAINING_CHANGE_NO_DATASET_REPORT.md")
    if rpt.exists():
        text = rpt.read_text(encoding="utf-8").lower()
        assert "vae works" not in text
        assert "latent space learned" not in text
        assert "scientific success" not in text
        assert "model converged" not in text
        assert "model generalized" not in text
        assert "vae success" not in text


def test_p60_30_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_target_balance_objective_audit.py",
        "tools/phase2/run_p60_fc_vae_target_balance_objective_audit_smoke.py",
        "tests/test_phase2_fc_vae_target_balance_objective_audit.py",
        "tests/test_phase2_p60_fc_vae_target_balance_objective_audit_smoke.py",
        "reports/PHASE_2_P60_FC_VAE_TARGET_BALANCE_OBJECTIVE_AUDIT_NO_TRAINING_CHANGE_NO_DATASET_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p60-fc-vae-target-balance-objective-audit-no-training-change-no-dataset",
        base_commit="933d8b9f2f7644969677ae99c0c27598c2f7fa39",
        allowed_files=allowed,
        phase_label="P60",
    )
