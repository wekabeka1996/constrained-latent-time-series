# tests/test_phase2_p52_bridge_path_consistency_diagnostics_smoke.py

import json
import os
import subprocess
import sys

def test_p52_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p52_bridge_path_consistency_diagnostics_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P52"
    assert data["status"] == "bridge_path_consistency_diagnostics_available_no_model_no_vae_no_gsb_no_science"
    assert data["bridge_target_count"] == 3
    assert data["lambda_values"] == [0.25, 0.5, 0.75]
    assert data["projection_positions_finite"] is True
    assert data["projection_positions_strictly_increasing"] is True
    assert data["own_target_distances_finite"] is True
    assert data["all_targets_loss_decreased"] is True
    assert data["all_losses_finite"] is True
    assert data["path_consistency_passed"] is True
    
    # Verify boundaries and no claims
    assert data["no_model"] is True
    assert data["no_vae"] is True
    assert data["no_dataset"] is True
    assert data["no_torch_optimizer"] is True
    assert data["no_gsb_claim"] is True
    assert data["no_generation_claim"] is True
    assert data["no_scientific_conclusion"] is True
    assert data["realizability_claim"] == "path_consistency_diagnostics_only_not_state_validity"
    
    # Verify no lists/arrays (e.g. the 100-long AR spectrum log) are leaked in diagnostics
    for r in data["path_points"]:
        for k, v in r.items():
            if isinstance(v, list):
                assert False, f"Unexpected list leaked in path point: {k}"


def test_p52_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p52_bridge_path_consistency_diagnostics_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
