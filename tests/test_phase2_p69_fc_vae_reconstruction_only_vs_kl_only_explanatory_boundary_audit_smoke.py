# tests/test_phase2_p69_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_smoke.py

import json
import os
import subprocess
import sys


def test_p69_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p69_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P69"
    assert data["source_evidence_phase"] == "P68"
    assert data["status"] == "fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_available_no_dataset_no_generalization"
    assert data["contract_version"] == "phase2_p69_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_contract_v1"
    
    assert data["expected_source_component_runs"] == 9
    assert data["observed_source_component_runs"] == 9
    assert len(data["explanatory_views"]) == 7
    
    # boundary flags
    assert data["no_new_optimization"] is True
    assert data["no_direct_optimizer_created"] is True
    assert data["no_direct_model_created"] is True
    assert data["no_direct_torch_import"] is True
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    assert data["no_epoch_loop"] is True
    assert data["no_batch_loop"] is True
    assert data["no_scheduler"] is True
    assert data["no_checkpointing"] is True
    assert data["no_component_proof_claim"] is True
    assert data["no_numeric_identity_proof_claim"] is True
    assert data["no_perturbation_robustness_proof_claim"] is True
    assert data["no_coordinate_semantic_proof_claim"] is True
    assert data["no_latent_kl_explanatory_proof_claim"] is True


def test_p69_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p69_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
