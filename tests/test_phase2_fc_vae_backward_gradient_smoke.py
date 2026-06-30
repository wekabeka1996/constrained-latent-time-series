# tests/test_phase2_fc_vae_backward_gradient_smoke.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.fc_vae_backward_gradient_smoke import (
    FC_VAE_BACKWARD_GRADIENT_SMOKE_CONTRACT_VERSION,
    FC_VAE_BACKWARD_GRADIENT_SMOKE_KIND,
    FC_VAE_BACKWARD_GRADIENT_SMOKE_MODULE_NAME,
    FC_VAE_BACKWARD_GRADIENT_SMOKE_STATUS_AVAILABLE,
    P56_EPS_MODE,
    P56_BETA_ZERO,
    P56_BETA_SMALL,
    P56_TARGET_ID,
    load_torch_for_p56_backward_smoke,
    collect_p56_parameter_groups,
    compute_gradient_stats_for_group,
    compute_all_gradient_stats,
    clear_model_grads,
    run_p56_reconstruction_backward_pass,
    run_p56_kl_backward_pass,
    run_p56_combined_backward_pass,
    run_fc_vae_backward_gradient_smoke_probe,
    fc_vae_backward_gradient_smoke_probe_to_json_dict,
)
from src.phase2.fc_vae_encoder_posterior_kl_boundary import (
    build_fc_vae_encoder_posterior_model,
)
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    build_p50_bridge_targets,
)


def test_p56_01_constants_exact():
    assert FC_VAE_BACKWARD_GRADIENT_SMOKE_CONTRACT_VERSION == "phase2_p56_fc_vae_backward_gradient_smoke_contract_v1"
    assert FC_VAE_BACKWARD_GRADIENT_SMOKE_KIND == "fc_vae_backward_gradient_smoke_no_optimizer_no_training"
    assert FC_VAE_BACKWARD_GRADIENT_SMOKE_MODULE_NAME == "src.phase2.fc_vae_backward_gradient_smoke"
    assert FC_VAE_BACKWARD_GRADIENT_SMOKE_STATUS_AVAILABLE == "fc_vae_backward_gradient_smoke_available_no_optimizer_no_training"
    assert P56_EPS_MODE == "zero"
    assert P56_BETA_ZERO == 0.0
    assert P56_BETA_SMALL == 0.001
    assert P56_TARGET_ID == "bridge_lambda_0_25"


def test_p56_02_no_top_level_torch_import():
    filepath = "src/phase2/fc_vae_backward_gradient_smoke.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p56_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_backward_gradient_smoke.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p56_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_backward_gradient_smoke.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p56_05_no_torch_optim():
    p = pathlib.Path("src/phase2/fc_vae_backward_gradient_smoke.py").read_text(encoding="utf-8")
    assert "torch.optim" not in p
    assert "optimizer =" not in p
    assert not re.search(r"\bOptimizer\b", p)


def test_p56_06_no_optimizer_classes_or_step():
    p = pathlib.Path("src/phase2/fc_vae_backward_gradient_smoke.py").read_text(encoding="utf-8")
    assert ".step()" not in p
    assert "SGD" not in p
    assert "Adam" not in p
    assert "RMSprop" not in p


def test_p56_07_no_dataset_dataloader_imports():
    p = pathlib.Path("src/phase2/fc_vae_backward_gradient_smoke.py").read_text(encoding="utf-8")
    assert "DataLoader" not in p
    assert "Dataset" not in p


def test_p56_08_allowed_p55_p50_imports_present():
    p = pathlib.Path("src/phase2/fc_vae_backward_gradient_smoke.py").read_text(encoding="utf-8")
    assert "load_torch_for_p55_encoder_kl" in p
    assert "build_fc_vae_encoder_posterior_model" in p
    assert "build_p50_bridge_targets" in p


def test_p56_09_parameter_groups_exist_and_non_empty():
    torch = load_torch_for_p56_backward_smoke()
    model = build_fc_vae_encoder_posterior_model()
    groups = collect_p56_parameter_groups(model)
    
    assert "encoder_trunk" in groups
    assert "posterior_mean_mu_heads" in groups
    assert "posterior_mean_logvar_heads" in groups
    assert "posterior_volatility_mu_heads" in groups
    assert "posterior_volatility_logvar_heads" in groups
    assert "posterior_shared_mu_heads" in groups
    assert "posterior_shared_logvar_heads" in groups
    assert "decoder" in groups
    
    for k, v in groups.items():
        assert len(v) > 0, f"Group {k} is empty"


def test_p56_10_gradient_stats_helper_json_safe():
    torch = load_torch_for_p56_backward_smoke()
    model = build_fc_vae_encoder_posterior_model()
    groups = collect_p56_parameter_groups(model)
    
    # Try stats before gradients exist
    stats = compute_gradient_stats_for_group(groups["encoder_trunk"])
    assert stats["parameter_count"] > 0
    assert stats["parameters_with_grad"] == 0
    assert stats["all_present_grads_finite"] is True
    assert stats["any_grad_present"] is False
    assert stats["any_nonzero_grad"] is False
    assert stats["max_abs_grad_value"] == 0.0
    assert stats["mean_abs_grad_value"] == 0.0


def test_p56_11_reconstruction_backward_completes():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    target = targets[0]
    
    res = run_p56_reconstruction_backward_pass(model, target)
    assert res["backward_completed"] is True
    assert res["passed"] is True


def test_p56_12_reconstruction_loss_finite():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_reconstruction_backward_pass(model, targets[0])
    assert res["loss_finite"] is True
    assert res["loss_value"] > 0.0


def test_p56_13_reconstruction_backward_finite_encoder_trunk_grads():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_reconstruction_backward_pass(model, targets[0])
    stats = res["gradient_stats"]["encoder_trunk"]
    assert stats["parameters_with_grad"] > 0
    assert stats["all_present_grads_finite"] is True
    assert stats["any_nonzero_grad"] is True


def test_p56_14_reconstruction_backward_finite_posterior_mu_head_grads():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_reconstruction_backward_pass(model, targets[0])
    
    for group_name in ["posterior_mean_mu_heads", "posterior_volatility_mu_heads", "posterior_shared_mu_heads"]:
        stats = res["gradient_stats"][group_name]
        assert stats["parameters_with_grad"] > 0
        assert stats["all_present_grads_finite"] is True
        assert stats["any_nonzero_grad"] is True


def test_p56_15_reconstruction_backward_finite_decoder_grads():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_reconstruction_backward_pass(model, targets[0])
    stats = res["gradient_stats"]["decoder"]
    assert stats["parameters_with_grad"] > 0
    assert stats["all_present_grads_finite"] is True
    assert stats["any_nonzero_grad"] is True


def test_p56_16_reconstruction_backward_does_not_require_logvar_gradients():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_reconstruction_backward_pass(model, targets[0])
    
    # We explicitly verify they can be 0 or without grad
    for group_name in ["posterior_mean_logvar_heads", "posterior_volatility_logvar_heads", "posterior_shared_logvar_heads"]:
        stats = res["gradient_stats"][group_name]
        assert stats["all_present_grads_finite"] is True


def test_p56_17_kl_backward_completes():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_kl_backward_pass(model, targets[0])
    assert res["backward_completed"] is True
    assert res["passed"] is True


def test_p56_18_kl_loss_finite_and_non_negative():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_kl_backward_pass(model, targets[0])
    assert res["loss_finite"] is True
    assert res["loss_non_negative"] is True
    assert res["loss_value"] > 0.0


def test_p56_19_kl_backward_finite_encoder_trunk_grads():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_kl_backward_pass(model, targets[0])
    stats = res["gradient_stats"]["encoder_trunk"]
    assert stats["parameters_with_grad"] > 0
    assert stats["all_present_grads_finite"] is True
    assert stats["any_nonzero_grad"] is True


def test_p56_20_kl_backward_finite_posterior_mu_head_grads():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_kl_backward_pass(model, targets[0])
    
    for group_name in ["posterior_mean_mu_heads", "posterior_volatility_mu_heads", "posterior_shared_mu_heads"]:
        stats = res["gradient_stats"][group_name]
        assert stats["parameters_with_grad"] > 0
        assert stats["all_present_grads_finite"] is True
        assert stats["any_nonzero_grad"] is True


def test_p56_21_kl_backward_finite_posterior_logvar_head_grads():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_kl_backward_pass(model, targets[0])
    
    for group_name in ["posterior_mean_logvar_heads", "posterior_volatility_logvar_heads", "posterior_shared_logvar_heads"]:
        stats = res["gradient_stats"][group_name]
        assert stats["parameters_with_grad"] > 0
        assert stats["all_present_grads_finite"] is True
        assert stats["any_nonzero_grad"] is True


def test_p56_22_kl_backward_does_not_require_decoder_grads():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_kl_backward_pass(model, targets[0])
    stats = res["gradient_stats"]["decoder"]
    # Decoder parameters shouldn't have grads from KL
    assert stats["all_present_grads_finite"] is True
    assert stats["parameters_with_grad"] == 0


def test_p56_23_combined_backward_completes():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_combined_backward_pass(model, targets[0])
    assert res["backward_completed"] is True
    assert res["passed"] is True


def test_p56_24_combined_loss_finite():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_combined_backward_pass(model, targets[0])
    assert res["loss_finite"] is True
    assert res["loss_value"] > 0.0


def test_p56_25_combined_backward_finite_encoder_trunk_grads():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_combined_backward_pass(model, targets[0])
    stats = res["gradient_stats"]["encoder_trunk"]
    assert stats["parameters_with_grad"] > 0
    assert stats["all_present_grads_finite"] is True
    assert stats["any_nonzero_grad"] is True


def test_p56_26_combined_backward_finite_all_posterior_head_grads():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_combined_backward_pass(model, targets[0])
    
    posterior_groups = [
        "posterior_mean_mu_heads", "posterior_mean_logvar_heads",
        "posterior_volatility_mu_heads", "posterior_volatility_logvar_heads",
        "posterior_shared_mu_heads", "posterior_shared_logvar_heads"
    ]
    for group_name in posterior_groups:
        stats = res["gradient_stats"][group_name]
        assert stats["parameters_with_grad"] > 0
        assert stats["all_present_grads_finite"] is True
        assert stats["any_nonzero_grad"] is True


def test_p56_27_combined_backward_finite_decoder_grads():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    res = run_p56_combined_backward_pass(model, targets[0])
    stats = res["gradient_stats"]["decoder"]
    assert stats["parameters_with_grad"] > 0
    assert stats["all_present_grads_finite"] is True
    assert stats["any_nonzero_grad"] is True


def test_p56_28_aggregate_probe_pass():
    probe = run_fc_vae_backward_gradient_smoke_probe()
    assert probe["verdict"] == "PASS"
    assert probe["status"] == "fc_vae_backward_gradient_smoke_available_no_optimizer_no_training"
    assert probe["backward_pass_count"] == 3


def test_p56_29_serialization_excludes_tensors():
    probe = run_fc_vae_backward_gradient_smoke_probe()
    json_dict = fc_vae_backward_gradient_smoke_probe_to_json_dict(probe)
    
    for k, v in json_dict.items():
        if isinstance(v, list):
            # Only list of pass objects is allowed
            assert k == "passes"
            for pass_obj in v:
                assert "grad" not in pass_obj
        assert not hasattr(v, "grad"), f"Tensor leaked in JSON: {k}"


def test_p56_30_no_optimizer_training_flags():
    probe = run_fc_vae_backward_gradient_smoke_probe()
    assert probe["no_optimizer"] is True
    assert probe["no_optimizer_step"] is True
    assert probe["no_parameter_update"] is True
    assert probe["no_training_loop"] is True
    assert probe["no_dataset"] is True
    assert probe["no_dataloader"] is True


def test_p56_31_no_forbidden_claims_in_probe():
    probe = run_fc_vae_backward_gradient_smoke_probe()
    assert probe["no_gsb_claim"] is True
    assert probe["no_generation_claim"] is True
    assert probe["no_scientific_conclusion"] is True
    assert probe["no_latent_learning_claim"] is True
    assert probe["no_vae_success_claim"] is True


def test_p56_32_no_forbidden_claims_in_source():
    p = pathlib.Path("src/phase2/fc_vae_backward_gradient_smoke.py").read_text(encoding="utf-8")
    p_lower = p.lower()
    assert "vae works" not in p_lower
    assert "latent space learned" not in p_lower
    assert "semantic geometry proven" not in p_lower
    assert "c generated" not in p_lower
    assert "gsb implemented" not in p_lower
    assert "posterior collapse solved" not in p_lower
    assert "scientific success" not in p_lower
    assert "trained vae" not in p_lower
    assert "optimizer step" not in p_lower
    assert "vae solved" not in p_lower
    
    rpt = pathlib.Path("reports/PHASE_2_P56_FC_VAE_BACKWARD_GRADIENT_SMOKE_NO_OPTIMIZER_NO_TRAINING_REPORT.md")
    if rpt.exists():
        text = rpt.read_text(encoding="utf-8").lower()
        assert "vae works" not in text
        assert "latent space learned" not in text
        assert "scientific success" not in text


def test_p56_33_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_backward_gradient_smoke.py",
        "tools/phase2/run_p56_fc_vae_backward_gradient_smoke.py",
        "tests/test_phase2_fc_vae_backward_gradient_smoke.py",
        "tests/test_phase2_p56_fc_vae_backward_gradient_smoke.py",
        "reports/PHASE_2_P56_FC_VAE_BACKWARD_GRADIENT_SMOKE_NO_OPTIMIZER_NO_TRAINING_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p56-fc-vae-backward-gradient-smoke-no-optimizer-no-training",
        base_commit="75feed27880becee61fd5a5e875441212c6d99f7",
        allowed_files=allowed,
        phase_label="P56",
    )
