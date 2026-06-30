# tests/test_phase2_fc_vae_encoder_posterior_kl_boundary.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.fc_vae_encoder_posterior_kl_boundary import (
    FC_VAE_ENCODER_POSTERIOR_KL_CONTRACT_VERSION,
    FC_VAE_ENCODER_POSTERIOR_KL_KIND,
    FC_VAE_ENCODER_POSTERIOR_KL_MODULE_NAME,
    FC_VAE_ENCODER_POSTERIOR_KL_STATUS_AVAILABLE,
    P55_LATENT_MEAN_DIM,
    P55_LATENT_VOLATILITY_DIM,
    P55_LATENT_SHARED_DIM,
    P55_ENCODER_HIDDEN_DIM,
    P55_SIGNATURE_VECTOR_PER_SAMPLE_LENGTH,
    P55_SIGNATURE_VECTOR_TOTAL_LENGTH,
    P55_EXPECTED_BATCH_SIZE,
    P55_TARGET_COUNT,
    P55_EPS_MODE,
    P55_BETA,
    P55_BETA_STATUS,
    P55_SIGNATURE_TENSOR_FIELD_COUNT,
    load_torch_for_p55_encoder_kl,
    build_p55_signature_vector_from_target,
    reparameterize_latent,
    compute_diag_gaussian_kl_to_standard_normal,
    compute_factorised_kl_dict,
    build_fc_vae_encoder_posterior_model,
    run_fc_vae_encoder_posterior_kl_probe,
    fc_vae_encoder_posterior_kl_probe_to_json_dict,
)
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    build_p50_bridge_targets,
)


def test_p55_01_constants_exact():
    assert FC_VAE_ENCODER_POSTERIOR_KL_CONTRACT_VERSION == "phase2_p55_fc_vae_encoder_posterior_kl_boundary_contract_v1"
    assert FC_VAE_ENCODER_POSTERIOR_KL_KIND == "fc_vae_encoder_posterior_kl_boundary_no_training"
    assert FC_VAE_ENCODER_POSTERIOR_KL_MODULE_NAME == "src.phase2.fc_vae_encoder_posterior_kl_boundary"
    assert P55_LATENT_MEAN_DIM == 4
    assert P55_LATENT_VOLATILITY_DIM == 4
    assert P55_LATENT_SHARED_DIM == 4
    assert P55_ENCODER_HIDDEN_DIM == 16
    assert P55_SIGNATURE_VECTOR_PER_SAMPLE_LENGTH == 48
    assert P55_SIGNATURE_VECTOR_TOTAL_LENGTH == 96
    assert P55_EXPECTED_BATCH_SIZE == 2
    assert P55_TARGET_COUNT == 3
    assert P55_EPS_MODE == "zero"
    assert P55_BETA == 0.0
    assert P55_BETA_STATUS == "declared_not_trained_not_tuned"
    assert P55_SIGNATURE_TENSOR_FIELD_COUNT == 11


def test_p55_02_no_top_level_torch_import():
    filepath = "src/phase2/fc_vae_encoder_posterior_kl_boundary.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p55_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_encoder_posterior_kl_boundary.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p55_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_encoder_posterior_kl_boundary.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p55_05_no_torch_optim():
    p = pathlib.Path("src/phase2/fc_vae_encoder_posterior_kl_boundary.py").read_text(encoding="utf-8")
    assert "torch.optim" not in p
    assert "optimizer =" not in p
    assert not re.search(r"\bOptimizer\b", p)


def test_p55_06_no_dataset_dataloader_imports():
    p = pathlib.Path("src/phase2/fc_vae_encoder_posterior_kl_boundary.py").read_text(encoding="utf-8")
    assert "DataLoader" not in p
    assert "Dataset" not in p


def test_p55_07_allowed_p54_p50_p52_imports_present():
    p = pathlib.Path("src/phase2/fc_vae_encoder_posterior_kl_boundary.py").read_text(encoding="utf-8")
    assert "build_fc_vae_model" in p
    assert "build_p50_bridge_targets" in p
    assert "signature_tensor_field_names" in p


def test_p55_08_signature_vector_contract_deterministic():
    targets = build_p50_bridge_targets()
    target = targets[0]
    contract = build_p55_signature_vector_from_target(target)
    assert contract["signature_tensor_field_count"] == 11
    assert contract["signature_vector_per_sample_length"] == 48
    assert contract["signature_vector_total_length"] == 96


def test_p55_09_signature_vector_batch_shape():
    targets = build_p50_bridge_targets()
    target = targets[0]
    contract = build_p55_signature_vector_from_target(target)
    assert contract["signature_vector_batch_shape"] == [2, 48]


def test_p55_10_encoder_model_is_nn_module():
    torch = load_torch_for_p55_encoder_kl()
    model = build_fc_vae_encoder_posterior_model()
    assert isinstance(model, torch.nn.Module)


def test_p55_11_encoder_trunk_and_posterior_heads_exist():
    model = build_fc_vae_encoder_posterior_model()
    assert hasattr(model, "encoder_trunk")
    assert hasattr(model, "z_mean_mu_head")
    assert hasattr(model, "z_mean_logvar_head")
    assert hasattr(model, "z_volatility_mu_head")
    assert hasattr(model, "z_volatility_logvar_head")
    assert hasattr(model, "z_shared_mu_head")
    assert hasattr(model, "z_shared_logvar_head")
    assert hasattr(model, "decoder")


def test_p55_12_posterior_mu_logvar_shapes():
    torch = load_torch_for_p55_encoder_kl()
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    target = targets[0]
    contract = build_p55_signature_vector_from_target(target)
    posterior = model.encode(contract["signature_vector"])
    
    assert list(posterior["z_mean_mu"].shape) == [2, 4]
    assert list(posterior["z_mean_logvar"].shape) == [2, 4]
    assert list(posterior["z_volatility_mu"].shape) == [2, 4]
    assert list(posterior["z_volatility_logvar"].shape) == [2, 4]
    assert list(posterior["z_shared_mu"].shape) == [2, 4]
    assert list(posterior["z_shared_logvar"].shape) == [2, 4]


def test_p55_13_reparameterize_zero_eps_returns_mu():
    torch = load_torch_for_p55_encoder_kl()
    mu = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    logvar = torch.tensor([[0.5, -0.5, 0.1, -0.1], [0.2, -0.2, 0.3, -0.3]])
    
    result = reparameterize_latent(mu, logvar, eps_mode="zero")
    assert torch.allclose(result["z"], mu, atol=1e-7)
    assert torch.all(result["eps"] == 0.0).item()


def test_p55_14_reparameterize_ones_eps_shifts_by_std():
    torch = load_torch_for_p55_encoder_kl()
    mu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    logvar = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    
    result = reparameterize_latent(mu, logvar, eps_mode="ones")
    expected_std = (0.5 * logvar).exp()
    expected_z = mu + expected_std
    assert torch.allclose(result["z"], expected_z, atol=1e-6)


def test_p55_15_kl_per_group_finite():
    torch = load_torch_for_p55_encoder_kl()
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    fwd = model(targets[0], eps_mode="zero")
    kl = fwd["kl_dict"]
    
    assert torch.all(torch.isfinite(kl["kl_mean_per_sample"])).item()
    assert torch.all(torch.isfinite(kl["kl_volatility_per_sample"])).item()
    assert torch.all(torch.isfinite(kl["kl_shared_per_sample"])).item()
    assert torch.all(torch.isfinite(kl["kl_total_per_sample"])).item()


def test_p55_16_kl_per_group_non_negative():
    torch = load_torch_for_p55_encoder_kl()
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    fwd = model(targets[0], eps_mode="zero")
    kl = fwd["kl_dict"]
    
    assert (kl["kl_mean_batch_mean"].item() >= 0.0)
    assert (kl["kl_volatility_batch_mean"].item() >= 0.0)
    assert (kl["kl_shared_batch_mean"].item() >= 0.0)
    assert (kl["kl_total_batch_mean"].item() >= 0.0)


def test_p55_17_decoder_raw_shapes_from_p54():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    fwd = model(targets[0], eps_mode="zero")
    raw = fwd["decoder_forward"]["raw_tensors"]
    
    assert list(raw["raw_kappa"].shape) == [2, 5]
    assert list(raw["raw_omega"].shape) == [2, 1]
    assert list(raw["raw_total_mass"].shape) == [2, 1]
    assert list(raw["raw_allocation_logits"].shape) == [2, 2]


def test_p55_18_decoder_signature_finite():
    torch = load_torch_for_p55_encoder_kl()
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    fwd = model(targets[0], eps_mode="zero")
    sig = fwd["decoder_forward"]["combined_signature"]
    
    for k, v in sig.items():
        if torch.is_tensor(v):
            assert torch.all(torch.isfinite(v)).item(), f"Non-finite in sig field: {k}"


def test_p55_19_reconstruction_loss_finite():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    fwd = model(targets[0], eps_mode="zero")
    loss_dict = fwd["loss_dict"]
    
    assert loss_dict is not None
    assert "loss_total" in loss_dict
    import math
    assert math.isfinite(loss_dict["loss_total"].item())


def test_p55_20_total_loss_finite_with_beta_zero():
    model = build_fc_vae_encoder_posterior_model()
    targets = build_p50_bridge_targets()
    fwd = model(targets[0], eps_mode="zero", beta=0.0)
    
    assert fwd["total_loss"] is not None
    import math
    assert math.isfinite(fwd["total_loss"].item())


def test_p55_21_beta_placeholder_exact():
    probe = run_fc_vae_encoder_posterior_kl_probe()
    assert probe["beta_value"] == 0.0
    assert probe["beta_status"] == "declared_not_trained_not_tuned"


def test_p55_22_serialization_excludes_tensors():
    probe = run_fc_vae_encoder_posterior_kl_probe()
    json_dict = fc_vae_encoder_posterior_kl_probe_to_json_dict(probe)
    
    for k, v in json_dict.items():
        if isinstance(v, list):
            # Only shape lists are allowed
            assert k.endswith("_shape") or k == "signature_vector_batch_shape", \
                f"Unexpected list in JSON: {k}"
        # No tensor objects
        assert not hasattr(v, "shape"), f"Tensor leaked in JSON: {k}"


def test_p55_23_no_training_optimizer_claims():
    probe = run_fc_vae_encoder_posterior_kl_probe()
    assert probe["no_training"] is True
    assert probe["no_optimizer"] is True
    assert probe["no_dataset"] is True
    assert probe["no_dataloader"] is True


def test_p55_24_no_gsb_generation_science_claims():
    probe = run_fc_vae_encoder_posterior_kl_probe()
    assert probe["no_gsb_claim"] is True
    assert probe["no_generation_claim"] is True
    assert probe["no_scientific_conclusion"] is True
    assert probe["no_latent_learning_claim"] is True
    assert probe["no_vae_success_claim"] is True


def test_p55_25_no_forbidden_claims_in_source():
    p = pathlib.Path("src/phase2/fc_vae_encoder_posterior_kl_boundary.py").read_text(encoding="utf-8")
    p_lower = p.lower()
    assert "vae works" not in p_lower
    assert "latent space learned" not in p_lower
    assert "semantic geometry proven" not in p_lower
    assert "c generated" not in p_lower
    assert "gsb implemented" not in p_lower
    assert "scientific success" not in p_lower
    assert "posterior collapse solved" not in p_lower
    
    rpt = pathlib.Path("reports/PHASE_2_P55_FC_VAE_ENCODER_POSTERIOR_KL_BOUNDARY_NO_TRAINING_REPORT.md")
    if rpt.exists():
        text = rpt.read_text(encoding="utf-8").lower()
        assert "vae works" not in text
        assert "latent space learned" not in text
        assert "scientific success" not in text


def test_p55_26_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_encoder_posterior_kl_boundary.py",
        "tools/phase2/run_p55_fc_vae_encoder_posterior_kl_boundary_smoke.py",
        "tests/test_phase2_fc_vae_encoder_posterior_kl_boundary.py",
        "tests/test_phase2_p55_fc_vae_encoder_posterior_kl_boundary_smoke.py",
        "reports/PHASE_2_P55_FC_VAE_ENCODER_POSTERIOR_KL_BOUNDARY_NO_TRAINING_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p55-fc-vae-encoder-posterior-kl-boundary-no-training",
        base_commit="e065c3b27ad36dca5c03a160b453626075db3715",
        allowed_files=allowed,
        phase_label="P55",
    )
