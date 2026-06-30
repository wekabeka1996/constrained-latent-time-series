# tests/test_phase2_fc_vae_architecture_forward.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.fc_vae_architecture_forward import (
    FC_VAE_ARCHITECTURE_FORWARD_CONTRACT_VERSION,
    FC_VAE_ARCHITECTURE_FORWARD_KIND,
    FC_VAE_ARCHITECTURE_FORWARD_MODULE_NAME,
    DEFAULT_BETA,
    BETA_STATUS,
    load_torch_for_p54_fc_vae,
    build_fc_vae_model,
    run_fc_vae_architecture_forward_probe,
    fc_vae_architecture_forward_probe_to_json_dict,
    compact_fc_vae_architecture_forward_json,
)

# P50 allowed imports to check target
from src.phase2.direct_raw_fit_to_endpoint_bridge_targets import (
    build_p50_bridge_targets,
)


def test_p54_01_constants_exact():
    assert FC_VAE_ARCHITECTURE_FORWARD_CONTRACT_VERSION == "phase2_p54_fc_vae_architecture_forward_contract_v1"
    assert FC_VAE_ARCHITECTURE_FORWARD_KIND == "fc_vae_architecture_and_forward_pass_no_training"
    assert FC_VAE_ARCHITECTURE_FORWARD_MODULE_NAME == "src.phase2.fc_vae_architecture_forward"
    assert DEFAULT_BETA == 0.0
    assert BETA_STATUS == "declared_not_trained_not_tuned"


def test_p54_02_no_top_level_torch_import():
    filepath = "src/phase2/fc_vae_architecture_forward.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p54_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_architecture_forward.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p54_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_architecture_forward.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p54_05_no_torch_optim():
    p = pathlib.Path("src/phase2/fc_vae_architecture_forward.py").read_text(encoding="utf-8")
    assert "torch.optim" not in p
    assert "optimizer =" not in p
    assert ".step()" not in p
    assert not re.search(r"\bOptimizer\b", p)


def test_p54_06_model_instantiation():
    torch = load_torch_for_p54_fc_vae()
    model = build_fc_vae_model(
        latent_mean_dim=4,
        latent_volatility_dim=4,
        latent_shared_dim=4,
        hidden_dim=16,
        ar_order=5,
    )
    assert isinstance(model, torch.nn.Module)


def test_p54_07_decoder_architecture_structure():
    torch = load_torch_for_p54_fc_vae()
    model = build_fc_vae_model(
        latent_mean_dim=4,
        latent_volatility_dim=4,
        latent_shared_dim=4,
        hidden_dim=16,
        ar_order=5,
    )
    
    # Assert trunk exists
    assert hasattr(model, "shared_trunk")
    assert isinstance(model.shared_trunk, torch.nn.Sequential)
    
    # Assert heads exist
    assert hasattr(model, "mean_head")
    assert hasattr(model, "volatility_trunk")
    assert hasattr(model, "omega_head")
    assert hasattr(model, "mass_head")
    assert hasattr(model, "logits_head")


def test_p54_08_raw_parameters_returned():
    torch = load_torch_for_p54_fc_vae()
    model = build_fc_vae_model(
        latent_mean_dim=4,
        latent_volatility_dim=4,
        latent_shared_dim=4,
        hidden_dim=16,
        ar_order=5,
    )
    
    # Batch size = 3
    z_mean = torch.randn(3, 4)
    z_vol = torch.randn(3, 4)
    z_shared = torch.randn(3, 4)
    
    res = model(z_mean, z_vol, z_shared)
    
    raw = res["raw_tensors"]
    assert list(raw["raw_kappa"].shape) == [3, 5]
    assert list(raw["raw_omega"].shape) == [3, 1]
    assert list(raw["raw_total_mass"].shape) == [3, 1]
    assert list(raw["raw_allocation_logits"].shape) == [3, 2]
    
    assert torch.all(torch.isfinite(raw["raw_kappa"])).item() is True
    assert torch.all(torch.isfinite(raw["raw_omega"])).item() is True


def test_p54_09_constrained_parameters_returned():
    torch = load_torch_for_p54_fc_vae()
    model = build_fc_vae_model(
        latent_mean_dim=4,
        latent_volatility_dim=4,
        latent_shared_dim=4,
        hidden_dim=16,
        ar_order=5,
    )
    
    z_mean = torch.randn(3, 4)
    z_vol = torch.randn(3, 4)
    z_shared = torch.randn(3, 4)
    
    res = model(z_mean, z_vol, z_shared)
    
    diag = res["constrained_diagnostics"]
    assert list(diag["ar_coefficients"].shape) == [3, 5]
    assert list(diag["omega"].shape) == [3, 1]
    assert list(diag["alpha"].shape) == [3, 1]
    assert list(diag["beta"].shape) == [3, 1]
    assert list(diag["persistence"].shape) == [3, 1]
    assert list(diag["stationarity_margin"].shape) == [3, 1]
    
    assert torch.all(torch.isfinite(diag["ar_coefficients"])).item() is True
    assert torch.all(torch.isfinite(diag["persistence"])).item() is True


def test_p54_10_signatures_constructed():
    torch = load_torch_for_p54_fc_vae()
    model = build_fc_vae_model(
        latent_mean_dim=4,
        latent_volatility_dim=4,
        latent_shared_dim=4,
        hidden_dim=16,
        ar_order=5,
    )
    
    z_mean = torch.randn(3, 4)
    z_vol = torch.randn(3, 4)
    z_shared = torch.randn(3, 4)
    
    res = model(z_mean, z_vol, z_shared)
    sig = res["combined_signature"]
    
    assert "ar_spectrum" in sig
    assert "garch_persistence" in sig
    assert "garch_stationarity_margin" in sig
    
    for k, v in sig.items():
        if torch.is_tensor(v):
            assert torch.all(torch.isfinite(v)).item() is True


def test_p54_11_matching_loss_computed():
    torch = load_torch_for_p54_fc_vae()
    model = build_fc_vae_model(
        latent_mean_dim=4,
        latent_volatility_dim=4,
        latent_shared_dim=4,
        hidden_dim=16,
        ar_order=5,
    )
    
    # Target signature from P50
    targets = build_p50_bridge_targets()
    target_sig = targets[0]
    
    z_mean = torch.randn(2, 4)
    z_vol = torch.randn(2, 4)
    z_shared = torch.randn(2, 4)
    
    res = model(z_mean, z_vol, z_shared, target_signature=target_sig)
    loss_dict = res["loss_dict"]
    
    assert loss_dict is not None
    assert "loss_total" in loss_dict
    assert torch.all(torch.isfinite(loss_dict["loss_total"])).item() is True
    assert float(loss_dict["loss_total"].item()) >= 0.0


def test_p54_12_no_optimizer_training_loop_or_dataloader():
    probe_res = run_fc_vae_architecture_forward_probe()
    assert probe_res["no_model_claim"] is True
    assert probe_res["no_training"] is True
    assert probe_res["no_optimizer"] is True
    assert probe_res["no_dataset"] is True
    assert probe_res["no_dataloader"] is True


def test_p54_13_default_beta_and_status():
    probe_res = run_fc_vae_architecture_forward_probe()
    assert probe_res["default_beta"] == 0.0
    assert probe_res["beta_status"] == "declared_not_trained_not_tuned"


def test_p54_14_no_forbidden_claims_in_code_or_reason():
    probe_res = run_fc_vae_architecture_forward_probe()
    assert probe_res["no_gsb_claim"] is True
    assert probe_res["no_generation_claim"] is True
    assert probe_res["no_scientific_conclusion"] is True


def test_p54_15_no_scientific_success_claims():
    p1 = pathlib.Path("src/phase2/fc_vae_architecture_forward.py").read_text(encoding="utf-8")
    assert "scientific success" not in p1.lower()
    assert "gsb solved" not in p1.lower()
    assert "optimal transport" not in p1.lower()
    
    p2 = pathlib.Path("reports/PHASE_2_P54_FC_VAE_ARCHITECTURE_AND_FORWARD_PASS_NO_TRAINING_REPORT.md")
    if p2.exists():
        text = p2.read_text(encoding="utf-8")
        assert "scientific success" not in text.lower()
        assert "optimal transport" not in text.lower()


def test_p54_16_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_architecture_forward.py",
        "tools/phase2/run_p54_fc_vae_architecture_forward_smoke.py",
        "tests/test_phase2_fc_vae_architecture_forward.py",
        "tests/test_phase2_p54_fc_vae_architecture_forward_smoke.py",
        "reports/PHASE_2_P54_FC_VAE_ARCHITECTURE_AND_FORWARD_PASS_NO_TRAINING_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p54-fc-vae-architecture-and-forward-pass-no-training",
        base_commit="af33932f50bb270edc0e018e460dae98412bd284",
        allowed_files=allowed,
        phase_label="P54",
    )
