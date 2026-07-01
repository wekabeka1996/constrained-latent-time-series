# tests/test_phase2_fc_vae_reconstruction_trajectory_diagnostics.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.fc_vae_reconstruction_trajectory_diagnostics import (
    FC_VAE_RECONSTRUCTION_TRAJECTORY_CONTRACT_VERSION,
    FC_VAE_RECONSTRUCTION_TRAJECTORY_KIND,
    FC_VAE_RECONSTRUCTION_TRAJECTORY_MODULE_NAME,
    FC_VAE_RECONSTRUCTION_TRAJECTORY_STATUS_AVAILABLE,
    P59_SEED,
    P59_BETA,
    P59_LEARNING_RATE,
    P59_STEP_COUNT,
    load_torch_for_p59_trajectory_diagnostics,
    run_p59_reconstruction_trajectory_diagnostics_once,
    run_fc_vae_reconstruction_trajectory_diagnostics_probe,
    fc_vae_reconstruction_trajectory_diagnostics_probe_to_json_dict,
)
from src.phase2.fc_vae_encoder_posterior_kl_boundary import (
    build_fc_vae_encoder_posterior_model,
)
from src.phase2.fc_vae_bounded_micro_training_harness import (
    build_p58_deterministic_target_set,
    compute_p58_micro_training_objective,
)


def test_p59_01_constants_exact():
    assert FC_VAE_RECONSTRUCTION_TRAJECTORY_CONTRACT_VERSION == "phase2_p59_fc_vae_reconstruction_trajectory_diagnostics_contract_v1"
    assert FC_VAE_RECONSTRUCTION_TRAJECTORY_KIND == "fc_vae_reconstruction_trajectory_diagnostics_no_dataset_no_generalization"
    assert FC_VAE_RECONSTRUCTION_TRAJECTORY_MODULE_NAME == "src.phase2.fc_vae_reconstruction_trajectory_diagnostics"
    assert FC_VAE_RECONSTRUCTION_TRAJECTORY_STATUS_AVAILABLE == "fc_vae_reconstruction_trajectory_diagnostics_available_no_dataset_no_generalization"
    assert P59_SEED == 58058
    assert P59_BETA == 0.001
    assert P59_LEARNING_RATE == 1e-4
    assert P59_STEP_COUNT == 5


def test_p59_02_no_top_level_torch_import():
    filepath = "src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p59_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p59_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p59_05_optimizer_use_limited():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py").read_text(encoding="utf-8")
    for line in p.splitlines():
        if line.startswith("from torch.optim") or line.startswith("import torch.optim"):
            assert False, f"Forbidden top-level optimizer import: {line}"
            
    assert "from torch.optim import SGD" in p


def test_p59_06_no_forbidden_optimizers_or_schedulers():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py").read_text(encoding="utf-8")
    assert "Adam" not in p
    assert "AdamW" not in p
    assert "RMSprop" not in p
    assert "lr_scheduler" not in p


def test_p59_07_no_dataset_dataloader_imports():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py").read_text(encoding="utf-8")
    assert "DataLoader" not in p
    assert "Dataset" not in p


def test_p59_08_no_epoch_batch_loop_strings():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py").read_text(encoding="utf-8")
    assert "for epoch" not in p.lower()
    assert "for batch" not in p.lower()


def test_p59_09_exactly_one_bounded_step_loop():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py").read_text(encoding="utf-8")
    matches = re.findall(r"for\s+\w+\s+in\s+range\(P59_STEP_COUNT\):", p)
    assert len(matches) == 1


def test_p59_10_allowed_imports_present():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py").read_text(encoding="utf-8")
    assert "load_torch_for_p55_encoder_kl" in p
    assert "build_fc_vae_encoder_posterior_model" in p
    assert "compute_all_gradient_stats" in p
    assert "clear_model_grads" in p
    assert "collect_parameter_snapshot" in p
    assert "compare_parameter_snapshots" in p
    assert "compare_parameter_group_deltas" in p
    assert "build_p58_deterministic_target_set" in p
    assert "compute_p58_micro_training_objective" in p


def test_p59_11_target_set_has_exactly_three():
    targets = build_p58_deterministic_target_set()
    assert len(targets) == 3


def test_p59_12_target_ids_exact_and_ordered():
    targets = build_p58_deterministic_target_set()
    ids = [t["target_id"] for t in targets]
    assert ids == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]


def test_p59_13_objective_returns_tensors():
    torch = load_torch_for_p59_trajectory_diagnostics()
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p58_deterministic_target_set()
    
    obj = compute_p58_micro_training_objective(model, targets)
    assert torch.is_tensor(obj["objective_loss"])
    assert torch.is_tensor(obj["reconstruction_mean"])
    assert torch.is_tensor(obj["kl_mean"])


def test_p59_14_objective_losses_finite():
    torch = load_torch_for_p59_trajectory_diagnostics()
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p58_deterministic_target_set()
    
    obj = compute_p58_micro_training_objective(model, targets)
    assert torch.isfinite(obj["objective_loss"]).item() is True
    assert torch.isfinite(obj["reconstruction_mean"]).item() is True
    assert torch.isfinite(obj["kl_mean"]).item() is True


def test_p59_15_bounded_harness_completes():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert res["all_optimizer_steps_completed"] is True
    assert res["passed"] is True


def test_p59_16_completed_step_count_exact():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert res["completed_step_count"] == 5


def test_p59_17_all_optimizer_steps_completed():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert res["all_optimizer_steps_completed"] is True


def test_p59_18_all_step_losses_finite():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert res["all_step_losses_finite"] is True


def test_p59_19_all_step_gradients_finite():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert res["all_step_gradients_finite"] is True


def test_p59_20_all_step_gradients_present():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert res["all_step_gradients_present"] is True


def test_p59_21_final_objective_finite():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert res["final_objective_finite"] is True


def test_p59_22_global_parameter_deltas_finite():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert res["all_parameter_deltas_finite"] is True


def test_p59_23_any_parameter_changed():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert res["any_parameter_changed"] is True


def test_p59_24_all_expected_groups_changed():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert res["all_expected_groups_changed"] is True


def test_p59_25_group_delta_stats_finite():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    group_stats = res["group_parameter_delta_stats"]
    for gname, gs in group_stats.items():
        assert gs["parameter_count"] > 0
        assert gs["all_deltas_finite"] is True
        assert gs["any_parameter_changed"] is True


def test_p59_26_loss_decrease_reported():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    assert "loss_decreased" in res
    assert isinstance(res["loss_decreased"], bool)


def test_p59_27_trajectories_contain_five_entries():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    trajectories = res["per_target_trajectories"]
    for tid, traj in trajectories.items():
        assert len(traj["total_loss_trajectory"]) == 5
        assert len(traj["reconstruction_loss_trajectory"]) == 5
        assert len(traj["kl_loss_trajectory"]) == 5


def test_p59_28_diagnostics_delta_calculations_correct():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    diagnostics = res["per_target_diagnostics"]
    for tid, diag in diagnostics.items():
        import math
        assert math.isfinite(diag["initial_total_loss"])
        assert math.isfinite(diag["final_total_loss"])
        assert math.isfinite(diag["total_loss_delta"])
        
        assert math.isfinite(diag["initial_reconstruction_loss"])
        assert math.isfinite(diag["final_reconstruction_loss"])
        assert math.isfinite(diag["reconstruction_loss_delta"])
        
        assert math.isfinite(diag["initial_kl_loss"])
        assert math.isfinite(diag["final_kl_loss"])
        assert math.isfinite(diag["kl_loss_delta"])
        
        assert abs(diag["total_loss_delta"] - (diag["final_total_loss"] - diag["initial_total_loss"])) < 1e-6


def test_p59_29_best_worst_diagnostics_correct():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    bw = res["best_worst_diagnostics"]
    assert bw["best_target_by_final_reconstruction"] in res["target_ids"]
    assert bw["worst_target_by_final_reconstruction"] in res["target_ids"]
    assert bw["best_target_by_reconstruction_delta"] in res["target_ids"]
    assert bw["worst_target_by_reconstruction_delta"] in res["target_ids"]
    assert bw["best_final_reconstruction_value"] <= bw["worst_final_reconstruction_value"]


def test_p59_30_target_dominance_results_correct():
    res = run_p59_reconstruction_trajectory_diagnostics_once()
    dom = res["dominance_diagnostics"]
    assert dom["initial_dominant_target"] in res["target_ids"]
    assert dom["final_dominant_target"] in res["target_ids"]
    assert 0.0 < dom["initial_dominant_share"] <= 1.0
    assert 0.0 < dom["final_dominant_share"] <= 1.0


def test_p59_31_aggregate_probe_pass():
    probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    assert probe["verdict"] == "PASS"
    assert probe["status"] == "fc_vae_reconstruction_trajectory_diagnostics_available_no_dataset_no_generalization"


def test_p59_32_serialization_excludes_tensors():
    probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    json_dict = fc_vae_reconstruction_trajectory_diagnostics_probe_to_json_dict(probe)
    
    # We inspect the nested structures
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


def test_p59_33_no_dataset_dataloader_flags():
    probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    assert probe["no_dataset"] is True
    assert probe["no_dataloader"] is True
    assert probe["no_epoch_loop"] is True
    assert probe["no_batch_loop"] is True
    assert probe["no_scheduler"] is True
    assert probe["no_checkpointing"] is True


def test_p59_34_no_forbidden_claims_in_probe():
    probe = run_fc_vae_reconstruction_trajectory_diagnostics_probe()
    assert probe["no_generalization_claim"] is True
    assert probe["no_generation_claim"] is True
    assert probe["no_gsb_claim"] is True
    assert probe["no_scientific_conclusion"] is True
    assert probe["no_latent_learning_claim"] is True
    assert probe["no_vae_success_claim"] is True
    assert probe["no_convergence_claim"] is True
    assert probe["no_semantic_geometry_proof_claim"] is True


def test_p59_35_no_forbidden_claims_in_source():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py").read_text(encoding="utf-8")
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
    
    rpt = pathlib.Path("reports/PHASE_2_P59_FC_VAE_RECONSTRUCTION_TRAJECTORY_DIAGNOSTICS_NO_DATASET_NO_GENERALIZATION_REPORT.md")
    if rpt.exists():
        text = rpt.read_text(encoding="utf-8").lower()
        assert "vae works" not in text
        assert "latent space learned" not in text
        assert "scientific success" not in text
        assert "model converged" not in text
        assert "model generalized" not in text
        assert "vae success" not in text


def test_p59_36_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_reconstruction_trajectory_diagnostics.py",
        "tools/phase2/run_p59_fc_vae_reconstruction_trajectory_diagnostics_smoke.py",
        "tests/test_phase2_fc_vae_reconstruction_trajectory_diagnostics.py",
        "tests/test_phase2_p59_fc_vae_reconstruction_trajectory_diagnostics_smoke.py",
        "reports/PHASE_2_P59_FC_VAE_RECONSTRUCTION_TRAJECTORY_DIAGNOSTICS_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p59-fc-vae-reconstruction-trajectory-diagnostics-no-dataset-no-generalization",
        base_commit="c086f2c3c03160193db331f77345fb2b474cdafb",
        allowed_files=allowed,
        phase_label="P59",
    )
