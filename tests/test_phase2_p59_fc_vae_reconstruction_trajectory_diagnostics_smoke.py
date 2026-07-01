# tests/test_phase2_p59_fc_vae_reconstruction_trajectory_diagnostics_smoke.py

import json
import os
import subprocess
import sys


def test_p59_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p59_fc_vae_reconstruction_trajectory_diagnostics_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P59"
    assert data["status"] == "fc_vae_reconstruction_trajectory_diagnostics_available_no_dataset_no_generalization"
    
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
    
    # Verify Trajectory Diagnostics structure
    assert "per_target_trajectories" in data
    assert "per_target_diagnostics" in data
    assert "best_worst_diagnostics" in data
    assert "dominance_diagnostics" in data
    
    # Check best/worst details
    bw = data["best_worst_diagnostics"]
    assert "best_target_by_final_reconstruction" in bw
    assert "worst_target_by_final_reconstruction" in bw
    assert "best_target_by_reconstruction_delta" in bw
    assert "worst_target_by_reconstruction_delta" in bw
    assert bw["best_final_reconstruction_value"] <= bw["worst_final_reconstruction_value"]
    
    # Check dominance details
    dom = data["dominance_diagnostics"]
    assert "initial_dominant_target" in dom
    assert "final_dominant_target" in dom
    assert 0.0 < dom["initial_dominant_share"] <= 1.0
    
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
    assert data["no_semantic_geometry_proof_claim"] is True
    assert data["realizability_claim"] == "trajectory_diagnostics_only_no_dataset_no_generalization"
    
    # Verify no raw/posterior/latent/gradient/snapshot tensors in JSON
    for k, v in data.items():
        if isinstance(v, list) and k not in ["step_summaries", "target_ids"]:
            assert k.endswith("_shape") or k.startswith("lambda_"), \
                f"Unexpected list leaked in smoke JSON: {k}"


def test_p59_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p59_fc_vae_reconstruction_trajectory_diagnostics_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
