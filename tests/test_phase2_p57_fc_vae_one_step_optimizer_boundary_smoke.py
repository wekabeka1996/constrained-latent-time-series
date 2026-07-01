# tests/test_phase2_p57_fc_vae_one_step_optimizer_boundary_smoke.py

import json
import os
import subprocess
import sys


def test_p57_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p57_fc_vae_one_step_optimizer_boundary_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P57"
    assert data["status"] == "fc_vae_one_step_optimizer_boundary_available_no_training_loop_no_dataset"
    
    # Verify shapes/counts/stats
    assert data["optimizer_name"] == "SGD"
    assert data["optimizer_step_count"] == 1
    assert data["loss_before_finite"] is True
    assert data["loss_after_finite"] is True
    assert data["after_forward_finite"] is True
    
    assert data["gradients_before_step_finite"] is True
    assert data["gradients_before_step_present"] is True
    assert data["all_parameter_deltas_finite"] is True
    assert data["any_parameter_changed"] is True
    assert data["all_expected_groups_changed"] is True
    
    # Verify boundaries and no claims
    assert data["no_training_loop"] is True
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    assert data["no_epoch_loop"] is True
    assert data["no_batch_loop"] is True
    assert data["no_scheduler"] is True
    assert data["no_checkpointing"] is True
    
    assert data["no_gsb_claim"] is True
    assert data["no_generation_claim"] is True
    assert data["no_scientific_conclusion"] is True
    assert data["no_latent_learning_claim"] is True
    assert data["no_vae_success_claim"] is True
    assert data["no_convergence_claim"] is True
    assert data["realizability_claim"] == "one_step_optimizer_boundary_only_no_training_loop_no_dataset"
    
    # Verify no raw/posterior/latent/gradient/snapshot tensors in JSON
    for k, v in data.items():
        if isinstance(v, list):
            assert k.endswith("_shape") or k.startswith("lambda_"), \
                f"Unexpected list leaked in smoke JSON: {k}"


def test_p57_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p57_fc_vae_one_step_optimizer_boundary_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
