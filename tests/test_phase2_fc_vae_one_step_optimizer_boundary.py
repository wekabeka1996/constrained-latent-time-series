# tests/test_phase2_fc_vae_one_step_optimizer_boundary.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.fc_vae_one_step_optimizer_boundary import (
    FC_VAE_ONE_STEP_OPTIMIZER_CONTRACT_VERSION,
    FC_VAE_ONE_STEP_OPTIMIZER_KIND,
    FC_VAE_ONE_STEP_OPTIMIZER_MODULE_NAME,
    FC_VAE_ONE_STEP_OPTIMIZER_STATUS_AVAILABLE,
    P57_TARGET_ID,
    P57_EPS_MODE,
    P57_BETA,
    P57_LEARNING_RATE,
    P57_OPTIMIZER_NAME,
    P57_STEP_COUNT,
    P57_SEED,
    load_torch_for_p57_optimizer_boundary,
    collect_parameter_snapshot,
    compare_parameter_snapshots,
    collect_group_named_parameters,
    compare_parameter_group_deltas,
    run_p57_one_step_optimizer_boundary_once,
    run_fc_vae_one_step_optimizer_boundary_probe,
    fc_vae_one_step_optimizer_boundary_probe_to_json_dict,
)
from src.phase2.fc_vae_encoder_posterior_kl_boundary import (
    build_fc_vae_encoder_posterior_model,
)


def test_p57_01_constants_exact():
    assert FC_VAE_ONE_STEP_OPTIMIZER_CONTRACT_VERSION == "phase2_p57_fc_vae_one_step_optimizer_boundary_contract_v1"
    assert FC_VAE_ONE_STEP_OPTIMIZER_KIND == "fc_vae_one_step_optimizer_boundary_no_training_loop_no_dataset"
    assert FC_VAE_ONE_STEP_OPTIMIZER_MODULE_NAME == "src.phase2.fc_vae_one_step_optimizer_boundary"
    assert FC_VAE_ONE_STEP_OPTIMIZER_STATUS_AVAILABLE == "fc_vae_one_step_optimizer_boundary_available_no_training_loop_no_dataset"
    assert P57_TARGET_ID == "bridge_lambda_0_25"
    assert P57_EPS_MODE == "zero"
    assert P57_BETA == 0.001
    assert P57_LEARNING_RATE == 1e-4
    assert P57_OPTIMIZER_NAME == "SGD"
    assert P57_STEP_COUNT == 1
    assert P57_SEED == 57057


def test_p57_02_no_top_level_torch_import():
    filepath = "src/phase2/fc_vae_one_step_optimizer_boundary.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p57_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_one_step_optimizer_boundary.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p57_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_one_step_optimizer_boundary.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p57_05_optimizer_use_limited():
    p = pathlib.Path("src/phase2/fc_vae_one_step_optimizer_boundary.py").read_text(encoding="utf-8")
    # SGD must only be imported inside functions
    for line in p.splitlines():
        if line.startswith("from torch.optim") or line.startswith("import torch.optim"):
            assert False, f"Forbidden top-level optimizer import: {line}"
            
    assert "from torch.optim import SGD" in p


def test_p57_06_no_forbidden_optimizers_or_schedulers():
    p = pathlib.Path("src/phase2/fc_vae_one_step_optimizer_boundary.py").read_text(encoding="utf-8")
    assert "Adam" not in p
    assert "AdamW" not in p
    assert "RMSprop" not in p
    assert "lr_scheduler" not in p
    assert "StepLR" not in p


def test_p57_07_no_dataset_dataloader_imports():
    p = pathlib.Path("src/phase2/fc_vae_one_step_optimizer_boundary.py").read_text(encoding="utf-8")
    assert "DataLoader" not in p
    assert "Dataset" not in p


def test_p57_08_no_training_loop_strings():
    p = pathlib.Path("src/phase2/fc_vae_one_step_optimizer_boundary.py").read_text(encoding="utf-8")
    assert "for epoch" not in p.lower()
    assert "for batch" not in p.lower()


def test_p57_09_allowed_imports_present():
    p = pathlib.Path("src/phase2/fc_vae_one_step_optimizer_boundary.py").read_text(encoding="utf-8")
    assert "load_torch_for_p55_encoder_kl" in p
    assert "build_fc_vae_encoder_posterior_model" in p
    assert "collect_p56_parameter_groups" in p
    assert "compute_all_gradient_stats" in p
    assert "clear_model_grads" in p


def test_p57_10_parameter_snapshot_collects_all():
    torch = load_torch_for_p57_optimizer_boundary()
    model = build_fc_vae_encoder_posterior_model()
    snapshot = collect_parameter_snapshot(model)
    
    assert len(snapshot) > 0
    all_params = list(model.parameters())
    assert len(snapshot) == len(all_params)
    
    for name, p in model.named_parameters():
        assert name in snapshot
        assert torch.all(snapshot[name] == p).item()


def test_p57_11_parameter_snapshot_no_tensor_leakage():
    model = build_fc_vae_encoder_posterior_model()
    snapshot = collect_parameter_snapshot(model)
    
    # Snapshot is internal dict mapping strings to PyTorch tensors.
    # It must not leak into any JSON serialization.
    # We will verify this inside serialization tests.
    pass


def test_p57_12_parameter_snapshot_compare_identical():
    model = build_fc_vae_encoder_posterior_model()
    snapshot1 = collect_parameter_snapshot(model)
    snapshot2 = collect_parameter_snapshot(model)
    
    res = compare_parameter_snapshots(snapshot1, snapshot2)
    assert res["parameter_count"] == len(snapshot1)
    assert res["parameters_changed"] == 0
    assert res["parameters_unchanged"] == len(snapshot1)
    assert res["any_parameter_changed"] is False
    assert res["max_abs_parameter_delta_value"] == 0.0
    assert res["mean_abs_parameter_delta_value"] == 0.0


def test_p57_13_one_step_run_completes():
    res = run_p57_one_step_optimizer_boundary_once()
    assert res["one_step_completed"] is True
    assert res["passed"] is True


def test_p57_14_optimizer_step_count_exact():
    res = run_p57_one_step_optimizer_boundary_once()
    assert res["optimizer_step_count"] == 1


def test_p57_15_loss_before_finite():
    res = run_p57_one_step_optimizer_boundary_once()
    assert res["loss_before_finite"] is True
    assert res["loss_before_value"] > 0.0


def test_p57_16_loss_after_finite():
    res = run_p57_one_step_optimizer_boundary_once()
    assert res["loss_after_finite"] is True
    assert res["loss_after_value"] > 0.0


def test_p57_17_after_forward_finite():
    res = run_p57_one_step_optimizer_boundary_once()
    assert res["after_forward_finite"] is True


def test_p57_18_gradients_before_step_finite():
    res = run_p57_one_step_optimizer_boundary_once()
    assert res["gradients_before_step_finite"] is True


def test_p57_19_gradients_before_step_present():
    res = run_p57_one_step_optimizer_boundary_once()
    assert res["gradients_before_step_present"] is True


def test_p57_20_global_parameter_deltas_finite():
    res = run_p57_one_step_optimizer_boundary_once()
    assert res["all_parameter_deltas_finite"] is True


def test_p57_21_any_parameter_changed():
    res = run_p57_one_step_optimizer_boundary_once()
    assert res["any_parameter_changed"] is True


def test_p57_22_all_expected_groups_changed():
    res = run_p57_one_step_optimizer_boundary_once()
    assert res["all_expected_groups_changed"] is True


def test_p57_23_group_delta_stats_finite():
    res = run_p57_one_step_optimizer_boundary_once()
    group_stats = res["group_parameter_delta_stats"]
    
    for gname, gs in group_stats.items():
        assert gs["parameter_count"] > 0
        assert gs["all_deltas_finite"] is True
        assert gs["any_parameter_changed"] is True
        assert gs["max_abs_parameter_delta_value"] > 0.0
        assert gs["mean_abs_parameter_delta_value"] > 0.0


def test_p57_24_loss_decrease_reported():
    res = run_p57_one_step_optimizer_boundary_once()
    # It must be reported, but it is not required for the pass verdict
    assert "loss_decreased" in res
    assert isinstance(res["loss_decreased"], bool)


def test_p57_25_aggregate_probe_pass():
    probe = run_fc_vae_one_step_optimizer_boundary_probe()
    assert probe["verdict"] == "PASS"
    assert probe["status"] == "fc_vae_one_step_optimizer_boundary_available_no_training_loop_no_dataset"


def test_p57_26_serialization_excludes_tensors():
    probe = run_fc_vae_one_step_optimizer_boundary_probe()
    json_dict = fc_vae_one_step_optimizer_boundary_probe_to_json_dict(probe)
    
    for k, v in json_dict.items():
        if isinstance(v, dict):
            # Safe nested dict check
            for sub_k, sub_v in v.items():
                assert not hasattr(sub_v, "shape")
        assert not hasattr(v, "shape"), f"Tensor leaked in JSON: {k}"


def test_p57_27_no_training_loop_dataloader_flags():
    probe = run_fc_vae_one_step_optimizer_boundary_probe()
    assert probe["no_training_loop"] is True
    assert probe["no_dataset"] is True
    assert probe["no_dataloader"] is True
    assert probe["no_epoch_loop"] is True
    assert probe["no_batch_loop"] is True
    assert probe["no_scheduler"] is True
    assert probe["no_checkpointing"] is True


def test_p57_28_no_forbidden_claims_in_probe():
    probe = run_fc_vae_one_step_optimizer_boundary_probe()
    assert probe["no_gsb_claim"] is True
    assert probe["no_generation_claim"] is True
    assert probe["no_scientific_conclusion"] is True
    assert probe["no_latent_learning_claim"] is True
    assert probe["no_vae_success_claim"] is True
    assert probe["no_convergence_claim"] is True


def test_p57_29_no_forbidden_claims_in_source():
    p = pathlib.Path("src/phase2/fc_vae_one_step_optimizer_boundary.py").read_text(encoding="utf-8")
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
    
    rpt = pathlib.Path("reports/PHASE_2_P57_FC_VAE_ONE_STEP_OPTIMIZER_BOUNDARY_NO_TRAINING_LOOP_NO_DATASET_REPORT.md")
    if rpt.exists():
        text = rpt.read_text(encoding="utf-8").lower()
        assert "vae works" not in text
        assert "latent space learned" not in text
        assert "scientific success" not in text
        assert "model converged" not in text


def test_p57_30_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_one_step_optimizer_boundary.py",
        "tools/phase2/run_p57_fc_vae_one_step_optimizer_boundary_smoke.py",
        "tests/test_phase2_fc_vae_one_step_optimizer_boundary.py",
        "tests/test_phase2_p57_fc_vae_one_step_optimizer_boundary_smoke.py",
        "reports/PHASE_2_P57_FC_VAE_ONE_STEP_OPTIMIZER_BOUNDARY_NO_TRAINING_LOOP_NO_DATASET_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p57-fc-vae-one-step-optimizer-boundary-no-training-loop-no-dataset",
        base_commit="cd68cc2b0de253ddf834571a59dda22156ebb692",
        allowed_files=allowed,
        phase_label="P57",
    )
