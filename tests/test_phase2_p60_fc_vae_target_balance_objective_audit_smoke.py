# tests/test_phase2_p60_fc_vae_target_balance_objective_audit_smoke.py

import json
import os
import subprocess
import sys


def test_p60_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p60_fc_vae_target_balance_objective_audit_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P60"
    assert data["source_evidence_phase"] == "P59"
    assert data["status"] == "fc_vae_target_balance_objective_audit_available_no_training_change_no_dataset"
    assert data["source_p59_verdict"] == "PASS"
    
    # Verify shapes/counts/stats
    assert data["target_count"] == 3
    assert data["target_ids"] == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    
    # Verify Audit Diagnostics structures
    assert "balance_metrics" in data
    assert "counterfactual_weighting_diagnostics" in data
    assert "recommendation" in data
    
    # Check recommendation details
    rec = data["recommendation"]
    assert "recommendation_status" in rec
    assert "recommendation_reason" in rec
    assert "hardest_target_id" in rec
    assert "most_improved_target_id" in rec
    assert "least_improved_target_id" in rec
    assert isinstance(rec["watch_targets"], list)
    assert isinstance(rec["blocking_issue_found"], bool)
    assert isinstance(rec["weighting_required_now"], bool)
    
    # Check balance metrics details
    bm = data["balance_metrics"]
    assert "initial_dominant_target" in bm
    assert "final_dominant_target" in bm
    assert "initial_dominant_share" in bm
    assert "final_dominant_share" in bm
    assert "total_improvement_spread" in bm
    assert "total_improvement_ratio" in bm
    assert bm["all_targets_total_loss_decreased"] is True
    assert bm["all_targets_reconstruction_loss_decreased"] is True
    
    # Verify counterfactual weighting schemes
    cwd = data["counterfactual_weighting_diagnostics"]
    expected_schemes = [
        "uniform_current",
        "inverse_initial_loss_balanced",
        "proportional_initial_loss_hard_target_emphasis",
        "proportional_final_loss_hard_target_emphasis"
    ]
    for sch in expected_schemes:
        assert sch in cwd
        assert "weighted_initial_objective" in cwd[sch]
        assert "weighted_final_objective" in cwd[sch]
        assert "weighted_delta" in cwd[sch]
        assert "weighted_loss_decreased" in cwd[sch]
        
    # Verify boundaries and no claims
    assert data["no_training_change"] is True
    assert data["no_optimizer_created_in_p60"] is True
    assert data["no_weighted_training_applied"] is True
    
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
    assert data["realizability_claim"] == "target_balance_objective_audit_only_no_training_change_no_dataset"
    
    # Verify no raw/posterior/latent/gradient/snapshot tensors in JSON
    for k, v in data.items():
        if isinstance(v, list) and k not in ["target_ids"]:
            assert k.endswith("_shape") or k.startswith("lambda_"), \
                f"Unexpected list leaked in smoke JSON: {k}"


def test_p60_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p60_fc_vae_target_balance_objective_audit_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
