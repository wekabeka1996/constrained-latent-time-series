# tests/test_phase2_p55_fc_vae_encoder_posterior_kl_boundary_smoke.py

import json
import os
import subprocess
import sys


def test_p55_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p55_fc_vae_encoder_posterior_kl_boundary_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    data = json.loads(res.stdout.strip())
    
    # Verdict and status
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P55"
    assert data["status"] == "fc_vae_encoder_posterior_kl_boundary_available_no_training"
    
    # Target
    assert "target_id" in data
    assert isinstance(data["target_id"], str)
    assert len(data["target_id"]) > 0
    
    # Signature vector
    assert data["signature_tensor_field_count"] == 11
    assert data["signature_vector_batch_shape"] == [2, 48]
    assert data["signature_vector_per_sample_length"] == 48
    assert data["signature_vector_total_length"] == 96
    
    # Posterior shapes
    assert data["z_mean_mu_shape"] == [2, 4]
    assert data["z_mean_logvar_shape"] == [2, 4]
    assert data["z_volatility_mu_shape"] == [2, 4]
    assert data["z_volatility_logvar_shape"] == [2, 4]
    assert data["z_shared_mu_shape"] == [2, 4]
    assert data["z_shared_logvar_shape"] == [2, 4]
    
    # Latent sample shapes
    assert data["z_mean_shape"] == [2, 4]
    assert data["z_volatility_shape"] == [2, 4]
    assert data["z_shared_shape"] == [2, 4]
    
    # Posterior finiteness
    assert data["posterior_all_mu_finite"] is True
    assert data["posterior_all_logvar_finite"] is True
    
    # Latent finiteness
    assert data["latent_samples_finite"] is True
    
    # KL values
    assert data["kl_values_finite"] is True
    assert data["kl_values_non_negative"] is True
    assert isinstance(data["kl_mean_batch_mean_value"], float)
    assert isinstance(data["kl_volatility_batch_mean_value"], float)
    assert isinstance(data["kl_shared_batch_mean_value"], float)
    assert isinstance(data["kl_total_batch_mean_value"], float)
    
    # Decoder raw shapes
    assert data["raw_kappa_shape"] == [2, 5]
    assert data["raw_omega_shape"] == [2, 1]
    assert data["raw_total_mass_shape"] == [2, 1]
    assert data["raw_allocation_logits_shape"] == [2, 2]
    
    # Decoder signature
    assert data["decoder_signature_tensor_field_count"] == 11
    assert data["decoder_signature_all_tensor_fields_finite"] is True
    
    # Total loss
    if data.get("total_loss_available"):
        assert data["total_loss_finite"] is True
    
    # Beta
    assert data["beta_value"] == 0.0
    assert data["beta_status"] == "declared_not_trained_not_tuned"
    
    # Boundary flags
    assert data["no_training"] is True
    assert data["no_optimizer"] is True
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    assert data["no_gsb_claim"] is True
    assert data["no_generation_claim"] is True
    assert data["no_scientific_conclusion"] is True
    assert data["no_latent_learning_claim"] is True
    assert data["no_vae_success_claim"] is True
    assert data["realizability_claim"] == "encoder_posterior_kl_boundary_only_no_training"
    
    # Verify no raw/posterior/latent tensors in JSON
    for k, v in data.items():
        if isinstance(v, list):
            assert k.endswith("_shape") or k == "signature_vector_batch_shape", \
                f"Unexpected list leaked in smoke JSON: {k}"


def test_p55_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p55_fc_vae_encoder_posterior_kl_boundary_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
