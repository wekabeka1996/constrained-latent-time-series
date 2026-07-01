# tests/test_phase2_fc_vae_bounded_micro_training_harness.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.fc_vae_bounded_micro_training_harness import (
    FC_VAE_BOUNDED_MICRO_TRAINING_CONTRACT_VERSION,
    FC_VAE_BOUNDED_MICRO_TRAINING_KIND,
    FC_VAE_BOUNDED_MICRO_TRAINING_MODULE_NAME,
    FC_VAE_BOUNDED_MICRO_TRAINING_STATUS_AVAILABLE,
    P58_TARGET_COUNT,
    P58_TARGET_IDS,
    P58_EPS_MODE,
    P58_BETA,
    P58_LEARNING_RATE,
    P58_OPTIMIZER_NAME,
    P58_STEP_COUNT,
    P58_SEED,
    load_torch_for_p58_micro_training,
    build_p58_deterministic_target_set,
    compute_p58_micro_training_objective,
    run_p58_bounded_micro_training_once,
    run_fc_vae_bounded_micro_training_harness_probe,
    fc_vae_bounded_micro_training_harness_probe_to_json_dict,
)
from src.phase2.fc_vae_encoder_posterior_kl_boundary import (
    build_fc_vae_encoder_posterior_model,
)


def test_p58_01_constants_exact():
    assert FC_VAE_BOUNDED_MICRO_TRAINING_CONTRACT_VERSION == "phase2_p58_fc_vae_bounded_micro_training_harness_contract_v1"
    assert FC_VAE_BOUNDED_MICRO_TRAINING_KIND == "fc_vae_bounded_micro_training_harness_no_dataset_no_generalization"
    assert FC_VAE_BOUNDED_MICRO_TRAINING_MODULE_NAME == "src.phase2.fc_vae_bounded_micro_training_harness"
    assert FC_VAE_BOUNDED_MICRO_TRAINING_STATUS_AVAILABLE == "fc_vae_bounded_micro_training_harness_available_no_dataset_no_generalization"
    assert P58_TARGET_COUNT == 3
    assert P58_TARGET_IDS == ("bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75")
    assert P58_EPS_MODE == "zero"
    assert P58_BETA == 0.001
    assert P58_LEARNING_RATE == 1e-4
    assert P58_OPTIMIZER_NAME == "SGD"
    assert P58_STEP_COUNT == 5
    assert P58_SEED == 58058


def test_p58_02_no_top_level_torch_import():
    filepath = "src/phase2/fc_vae_bounded_micro_training_harness.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p58_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_bounded_micro_training_harness.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p58_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_bounded_micro_training_harness.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p58_05_optimizer_use_limited():
    p = pathlib.Path("src/phase2/fc_vae_bounded_micro_training_harness.py").read_text(encoding="utf-8")
    for line in p.splitlines():
        if line.startswith("from torch.optim") or line.startswith("import torch.optim"):
            assert False, f"Forbidden top-level optimizer import: {line}"
            
    assert "from torch.optim import SGD" in p


def test_p58_06_no_forbidden_optimizers_or_schedulers():
    p = pathlib.Path("src/phase2/fc_vae_bounded_micro_training_harness.py").read_text(encoding="utf-8")
    assert "Adam" not in p
    assert "AdamW" not in p
    assert "RMSprop" not in p
    assert "lr_scheduler" not in p


def test_p58_07_no_dataset_dataloader_imports():
    p = pathlib.Path("src/phase2/fc_vae_bounded_micro_training_harness.py").read_text(encoding="utf-8")
    assert "DataLoader" not in p
    assert "Dataset" not in p


def test_p58_08_no_epoch_batch_loop_strings():
    p = pathlib.Path("src/phase2/fc_vae_bounded_micro_training_harness.py").read_text(encoding="utf-8")
    assert "for epoch" not in p.lower()
    assert "for batch" not in p.lower()


def test_p58_09_exactly_one_bounded_step_loop():
    p = pathlib.Path("src/phase2/fc_vae_bounded_micro_training_harness.py").read_text(encoding="utf-8")
    matches = re.findall(r"for\s+\w+\s+in\s+range\(P58_STEP_COUNT\):", p)
    assert len(matches) == 1


def test_p58_10_allowed_imports_present():
    p = pathlib.Path("src/phase2/fc_vae_bounded_micro_training_harness.py").read_text(encoding="utf-8")
    assert "load_torch_for_p55_encoder_kl" in p
    assert "build_fc_vae_encoder_posterior_model" in p
    assert "compute_all_gradient_stats" in p
    assert "clear_model_grads" in p
    assert "collect_parameter_snapshot" in p
    assert "compare_parameter_snapshots" in p
    assert "compare_parameter_group_deltas" in p


def test_p58_11_target_set_has_exactly_three():
    targets = build_p58_deterministic_target_set()
    assert len(targets) == 3


def test_p58_12_target_ids_exact_and_ordered():
    targets = build_p58_deterministic_target_set()
    ids = [t["target_id"] for t in targets]
    assert ids == list(P58_TARGET_IDS)


def test_p58_13_objective_returns_tensors():
    torch = load_torch_for_p58_micro_training()
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p58_deterministic_target_set()
    
    obj = compute_p58_micro_training_objective(model, targets)
    assert torch.is_tensor(obj["objective_loss"])
    assert torch.is_tensor(obj["reconstruction_mean"])
    assert torch.is_tensor(obj["kl_mean"])


def test_p58_14_objective_losses_finite():
    torch = load_torch_for_p58_micro_training()
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p58_deterministic_target_set()
    
    obj = compute_p58_micro_training_objective(model, targets)
    assert torch.isfinite(obj["objective_loss"]).item() is True
    assert torch.isfinite(obj["reconstruction_mean"]).item() is True
    assert torch.isfinite(obj["kl_mean"]).item() is True


def test_p58_15_bounded_harness_completes():
    res = run_p58_bounded_micro_training_once()
    assert res["all_optimizer_steps_completed"] is True
    assert res["passed"] is True


def test_p58_16_completed_step_count_exact():
    res = run_p58_bounded_micro_training_once()
    assert res["completed_step_count"] == 5


def test_p58_17_all_optimizer_steps_completed():
    res = run_p58_bounded_micro_training_once()
    assert res["all_optimizer_steps_completed"] is True


def test_p58_18_all_step_losses_finite():
    res = run_p58_bounded_micro_training_once()
    assert res["all_step_losses_finite"] is True


def test_p58_19_all_step_gradients_finite():
    res = run_p58_bounded_micro_training_once()
    assert res["all_step_gradients_finite"] is True


def test_p58_20_all_step_gradients_present():
    res = run_p58_bounded_micro_training_once()
    assert res["all_step_gradients_present"] is True


def test_p58_21_final_objective_finite():
    res = run_p58_bounded_micro_training_once()
    assert res["final_objective_finite"] is True


def test_p58_22_global_parameter_deltas_finite():
    res = run_p58_bounded_micro_training_once()
    assert res["all_parameter_deltas_finite"] is True


def test_p58_23_any_parameter_changed():
    res = run_p58_bounded_micro_training_once()
    assert res["any_parameter_changed"] is True


def test_p58_24_all_expected_groups_changed():
    res = run_p58_bounded_micro_training_once()
    assert res["all_expected_groups_changed"] is True


def test_p58_25_group_delta_stats_finite():
    res = run_p58_bounded_micro_training_once()
    group_stats = res["group_parameter_delta_stats"]
    for gname, gs in group_stats.items():
        assert gs["parameter_count"] > 0
        assert gs["all_deltas_finite"] is True
        assert gs["any_parameter_changed"] is True


def test_p58_26_loss_decrease_reported():
    res = run_p58_bounded_micro_training_once()
    assert "loss_decreased" in res
    assert isinstance(res["loss_decreased"], bool)


def test_p58_27_aggregate_probe_pass():
    probe = run_fc_vae_bounded_micro_training_harness_probe()
    assert probe["verdict"] == "PASS"
    assert probe["status"] == "fc_vae_bounded_micro_training_harness_available_no_dataset_no_generalization"


def test_p58_28_serialization_excludes_tensors():
    probe = run_fc_vae_bounded_micro_training_harness_probe()
    json_dict = fc_vae_bounded_micro_training_harness_probe_to_json_dict(probe)
    
    for k, v in json_dict.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                assert not hasattr(sub_v, "shape")
        elif isinstance(v, list) and k == "step_summaries":
            for step_obj in v:
                for sub_k, sub_v in step_obj.items():
                    assert not hasattr(sub_v, "shape")
        assert not hasattr(v, "shape"), f"Tensor leaked in JSON: {k}"


def test_p58_29_no_dataset_dataloader_flags():
    probe = run_fc_vae_bounded_micro_training_harness_probe()
    assert probe["no_dataset"] is True
    assert probe["no_dataloader"] is True
    assert probe["no_epoch_loop"] is True
    assert probe["no_batch_loop"] is True
    assert probe["no_scheduler"] is True
    assert probe["no_checkpointing"] is True


def test_p58_30_no_generalization_generation_science_claims():
    probe = run_fc_vae_bounded_micro_training_harness_probe()
    assert probe["no_generalization_claim"] is True
    assert probe["no_generation_claim"] is True
    assert probe["no_gsb_claim"] is True
    assert probe["no_scientific_conclusion"] is True
    assert probe["no_latent_learning_claim"] is True
    assert probe["no_vae_success_claim"] is True
    assert probe["no_convergence_claim"] is True


def test_p58_31_no_forbidden_claims_in_source():
    p = pathlib.Path("src/phase2/fc_vae_bounded_micro_training_harness.py").read_text(encoding="utf-8")
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
    assert "generalize claim" not in p_lower
    
    rpt = pathlib.Path("reports/PHASE_2_P58_FC_VAE_BOUNDED_MICRO_TRAINING_HARNESS_NO_DATASET_NO_GENERALIZATION_REPORT.md")
    if rpt.exists():
        text = rpt.read_text(encoding="utf-8").lower()
        assert "vae works" not in text
        assert "latent space learned" not in text
        assert "scientific success" not in text
        assert "model converged" not in text
        assert "model generalized" not in text


def test_p58_32_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_bounded_micro_training_harness.py",
        "tools/phase2/run_p58_fc_vae_bounded_micro_training_harness_smoke.py",
        "tests/test_phase2_fc_vae_bounded_micro_training_harness.py",
        "tests/test_phase2_p58_fc_vae_bounded_micro_training_harness_smoke.py",
        "reports/PHASE_2_P58_FC_VAE_BOUNDED_MICRO_TRAINING_HARNESS_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p58-fc-vae-bounded-micro-training-harness-no-dataset-no-generalization",
        base_commit="ea8c838d45f2a9d79725a490ec9f9c8d57af42f9",
        allowed_files=allowed,
        phase_label="P58",
    )
