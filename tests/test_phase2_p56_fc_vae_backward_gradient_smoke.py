# tests/test_phase2_p56_fc_vae_backward_gradient_smoke.py

import json
import os
import subprocess
import sys


def test_p56_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p56_fc_vae_backward_gradient_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P56"
    assert data["status"] == "fc_vae_backward_gradient_smoke_available_no_optimizer_no_training"
    
    # Verify shapes/counts/stats
    assert data["backward_pass_count"] == 3
    assert data["reconstruction_backward_passed"] is True
    assert data["kl_backward_passed"] is True
    assert data["combined_backward_passed"] is True
    
    assert data["all_backward_passes_completed"] is True
    assert data["all_required_gradients_finite"] is True
    assert data["all_required_gradients_present"] is True
    
    # Verify boundaries and no claims
    assert data["no_optimizer"] is True
    assert data["no_optimizer_step"] is True
    assert data["no_parameter_update"] is True
    assert data["no_training_loop"] is True
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    
    assert data["no_gsb_claim"] is True
    assert data["no_generation_claim"] is True
    assert data["no_scientific_conclusion"] is True
    assert data["no_latent_learning_claim"] is True
    assert data["no_vae_success_claim"] is True
    assert data["realizability_claim"] == "backward_gradient_smoke_only_no_optimizer_no_training"
    
    # Verify passes list
    assert len(data["passes"]) == 3
    for p in data["passes"]:
        assert "pass_name" in p
        assert "loss_value" in p
        assert p["passed"] is True
        
    # Verify no raw/posterior/latent/gradient tensors in JSON
    for k, v in data.items():
        if isinstance(v, list) and k != "passes":
            assert k.endswith("_shape") or k.startswith("lambda_"), \
                f"Unexpected list leaked in smoke JSON: {k}"


def test_p56_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p56_fc_vae_backward_gradient_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
