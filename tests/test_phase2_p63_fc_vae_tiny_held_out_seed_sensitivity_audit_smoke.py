# tests/test_phase2_p63_fc_vae_tiny_held_out_seed_sensitivity_audit_smoke.py

import json
import os
import subprocess
import sys


def test_p63_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p63_fc_vae_tiny_held_out_seed_sensitivity_audit_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P63"
    assert data["status"] == "fc_vae_tiny_held_out_seed_sensitivity_audit_available_no_dataset_no_generalization"
    assert data["contract_version"] == "phase2_p63_fc_vae_tiny_held_out_seed_sensitivity_audit_contract_v1"
    
    assert data["seed_count"] == 3
    assert data["fold_count_per_seed"] == 3
    assert data["total_seed_fold_runs"] == 9
    assert len(data["seed_summaries"]) == 3
    
    # boundary flags
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    assert data["no_epoch_loop"] is True
    assert data["no_batch_loop"] is True
    assert data["no_scheduler"] is True
    assert data["no_checkpointing"] is True
    assert data["no_transfer_proof_claim"] is True
    assert data["no_seed_robustness_claim"] is True


def test_p63_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p63_fc_vae_tiny_held_out_seed_sensitivity_audit_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
