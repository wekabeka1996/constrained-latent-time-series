# tests/test_phase2_p66_fc_vae_target_identity_numeric_separation_audit_smoke.py

import json
import os
import subprocess
import sys


def test_p66_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p66_fc_vae_target_identity_numeric_separation_audit_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P66"
    assert data["source_evidence_phase"] == "P65"
    assert data["status"] == "fc_vae_target_identity_numeric_separation_audit_available_no_dataset_no_generalization"
    assert data["contract_version"] == "phase2_p66_fc_vae_target_identity_numeric_separation_audit_contract_v1"
    
    assert data["expected_source_component_runs"] == 9
    assert data["observed_source_component_runs"] == 9
    assert data["aggregate_diagnostics"]["original_profile_count"] == 3
    assert data["aggregate_diagnostics"]["permutation_mismatch_count"] == 6
    
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


def test_p66_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p66_fc_vae_target_identity_numeric_separation_audit_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
