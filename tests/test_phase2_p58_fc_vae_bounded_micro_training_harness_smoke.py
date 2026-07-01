# tests/test_phase2_p58_fc_vae_bounded_micro_training_harness_smoke.py

import json
import os
import subprocess
import sys


def test_p58_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p58_fc_vae_bounded_micro_training_harness_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P58"
    assert data["status"] == "fc_vae_bounded_micro_training_harness_available_no_dataset_no_generalization"
    
    # Verify shapes/counts/stats
    assert data["target_count"] == 3
    assert data["target_ids"] == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert data["optimizer_name"] == "SGD"
    assert data["requested_step_count"] == 5
    assert data["completed_step_count"] == 5
    assert data["all_optimizer_steps_completed"] is True
    
    assert data["all_step_losses_finite"] is True
    assert data["all_step_gradients_finite"] is True
    assert data["all_step_gradients_present"] is True
    assert data["final_objective_finite"] is True
    assert data["all_parameter_deltas_finite"] is True
    assert data["any_parameter_changed"] is True
    assert data["all_expected_groups_changed"] is True
    
    # Verify boundaries and no claims
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    assert data["no_epoch_loop"] is True
    assert data["no_batch_loop"] is True
    assert data["no_scheduler"] is True
    assert data["no_checkpointing"] is True
    
    assert data["no_generalization_claim"] is True
    assert data["no_generation_claim"] is True
    assert data["no_gsb_claim"] is True
    assert data["no_scientific_conclusion"] is True
    assert data["no_latent_learning_claim"] is True
    assert data["no_vae_success_claim"] is True
    assert data["no_convergence_claim"] is True
    assert data["realizability_claim"] == "bounded_micro_training_harness_only_no_dataset_no_generalization"
    
    # Verify step_summaries list
    assert len(data["step_summaries"]) == 5
    for step in data["step_summaries"]:
        assert "step_index" in step
        assert "loss_value" in step
        assert "reconstruction_mean_value" in step
        assert "kl_mean_value" in step
        assert step["loss_finite"] is True
        assert step["reconstruction_finite"] is True
        assert step["kl_finite"] is True
        assert step["gradients_finite"] is True
        assert step["gradients_present"] is True
        assert step["optimizer_step_completed"] is True
        
    # Verify no raw/posterior/latent/gradient/snapshot tensors in JSON
    for k, v in data.items():
        if isinstance(v, list) and k != "step_summaries" and k != "target_ids":
            assert k.endswith("_shape") or k.startswith("lambda_"), \
                f"Unexpected list leaked in smoke JSON: {k}"


def test_p58_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p58_fc_vae_bounded_micro_training_harness_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
