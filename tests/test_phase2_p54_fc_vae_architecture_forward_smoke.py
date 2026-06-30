# tests/test_phase2_p54_fc_vae_architecture_forward_smoke.py

import json
import os
import subprocess
import sys

def test_p54_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p54_fc_vae_architecture_forward_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P54"
    assert data["status"] == "fc_vae_architecture_and_forward_pass_available_no_training"
    
    # Verify shapes and counts
    assert data["z_mean_shape"] == [2, 4]
    assert data["z_volatility_shape"] == [2, 4]
    assert data["z_shared_shape"] == [2, 4]
    
    assert data["raw_kappa_shape"] == [2, 5]
    assert data["raw_kappa_finite"] is True
    assert data["raw_omega_shape"] == [2, 1]
    assert data["raw_omega_finite"] is True
    assert data["raw_total_mass_shape"] == [2, 1]
    assert data["raw_total_mass_finite"] is True
    assert data["raw_allocation_logits_shape"] == [2, 2]
    assert data["raw_allocation_logits_finite"] is True
    
    assert data["ar_coefficients_shape"] == [2, 5]
    assert data["ar_coefficients_finite"] is True
    assert data["garch_omega_shape"] == [2, 1]
    assert data["garch_omega_finite"] is True
    assert data["garch_alpha_shape"] == [2, 1]
    assert data["garch_alpha_finite"] is True
    assert data["garch_beta_shape"] == [2, 1]
    assert data["garch_beta_finite"] is True
    assert data["garch_persistence_shape"] == [2, 1]
    assert data["garch_persistence_finite"] is True
    assert data["garch_margin_shape"] == [2, 1]
    assert data["garch_margin_finite"] is True
    
    assert data["signature_tensor_field_count"] == 11
    assert data["signature_all_tensor_fields_finite"] is True
    
    # Verify loss
    assert data["loss_total_finite"] is True
    assert data["loss_total_value"] > 0.0
    
    # Verify beta
    assert data["default_beta"] == 0.0
    assert data["beta_status"] == "declared_not_trained_not_tuned"
    
    # Verify boundaries and no claims
    assert data["no_model_claim"] is True
    assert data["no_training"] is True
    assert data["no_optimizer"] is True
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    assert data["no_gsb_claim"] is True
    assert data["no_generation_claim"] is True
    assert data["no_scientific_conclusion"] is True
    assert data["realizability_claim"] == "fc_vae_architecture_and_forward_pass_only_no_training"
    
    # Verify no lists/arrays are leaked in diagnostics (excluding shapes or lambda lists)
    for k, v in data.items():
        if isinstance(v, list) and not k.endswith("_shape") and not k.startswith("lambda_"):
            assert False, f"Unexpected list leaked in smoke JSON: {k}"


def test_p54_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p54_fc_vae_architecture_forward_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
