# tests/test_phase2_p62_fc_vae_tiny_held_out_deterministic_bridge_target_split_smoke.py

import json
import os
import subprocess
import sys


def test_p62_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p62_fc_vae_tiny_held_out_deterministic_bridge_target_split_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P62"
    assert data["status"] == "fc_vae_tiny_held_out_deterministic_bridge_target_split_available_no_dataset_no_generalization"
    assert data["contract_version"] == "phase2_p62_fc_vae_tiny_held_out_deterministic_bridge_target_split_contract_v1"
    
    assert data["fold_count"] == 3
    assert len(data["fold_summaries"]) == 3
    
    # boundary flags
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    assert data["no_epoch_loop"] is True
    assert data["no_batch_loop"] is True
    assert data["no_scheduler"] is True
    assert data["no_checkpointing"] is True
    
    # Verify no raw/posterior/latent/gradient/snapshot tensors in JSON
    for k, v in data.items():
        if isinstance(v, list) and k not in ["target_ids"]:
            assert k in ["fold_summaries", "step_counts", "train_target_ids"], f"Unexpected list leaked in smoke JSON: {k}"


def test_p62_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p62_fc_vae_tiny_held_out_deterministic_bridge_target_split_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
