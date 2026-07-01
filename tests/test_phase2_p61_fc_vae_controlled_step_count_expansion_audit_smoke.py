# tests/test_phase2_p61_fc_vae_controlled_step_count_expansion_audit_smoke.py

import json
import os
import subprocess
import sys


def test_p61_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p61_fc_vae_controlled_step_count_expansion_audit_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P61"
    assert data["status"] == "fc_vae_controlled_step_count_expansion_audit_available_no_dataset_no_generalization"
    
    # Verify shapes/counts/stats
    assert data["target_count"] == 3
    assert data["target_ids"] == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert data["step_counts"] == [5, 10, 20]
    
    # Verify structures
    assert "run_summaries" in data
    assert "cross_run_comparison" in data
    
    # Check runs
    runs = data["run_summaries"]
    for sc in ["steps_5", "steps_10", "steps_20"]:
        assert sc in runs
        r = runs[sc]
        assert r["passed"] is True
        assert r["completed_step_count"] == int(sc.split("_")[1])
        assert r["final_objective_finite"] is True
        
    # Check cross-run comparison
    comp = data["cross_run_comparison"]
    assert comp["cross_run_passed"] is True
    assert comp["all_runs_passed"] is True
    assert comp["objective_explosion_blocker_triggered"] is False
    assert comp["kl_explosion_blocker_triggered"] is False
    assert comp["dominance_blocker_across_runs"] is False
    assert comp["all_targets_improved_at_all_step_counts"] is True
    
    # Verify boundaries and no claims
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    assert data["no_epoch_loop"] is True
    assert data["no_batch_loop"] is True
    assert data["no_scheduler"] is True
    assert data["no_checkpointing"] is True
    assert data["no_weighted_training"] is True
    assert data["uniform_objective_preserved"] is True
    
    assert data["no_generalization_claim"] is True
    assert data["no_generation_claim"] is True
    assert data["no_gsb_claim"] is True
    assert data["no_scientific_conclusion"] is True
    assert data["no_latent_learning_claim"] is True
    assert data["no_vae_success_claim"] is True
    assert data["no_convergence_claim"] is True
    assert data["no_semantic_geometry_proof_claim"] is True
    assert data["realizability_claim"] == "controlled_step_count_expansion_audit_only_no_dataset_no_generalization"
    
    # Verify no raw/posterior/latent/gradient/snapshot tensors in JSON
    for k, v in data.items():
        if isinstance(v, list) and k not in ["target_ids", "step_counts"]:
            assert False, f"Unexpected list leaked in smoke JSON: {k}"


def test_p61_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p61_fc_vae_controlled_step_count_expansion_audit_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
