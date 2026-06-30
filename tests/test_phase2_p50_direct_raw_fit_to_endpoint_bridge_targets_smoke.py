# tests/test_phase2_p50_direct_raw_fit_to_endpoint_bridge_targets_smoke.py

import json
import os
import subprocess
import sys

def test_p50_smoke_01_run_success():
    # Set PYTHONPATH so the subprocess can import src
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p50_direct_raw_fit_to_endpoint_bridge_targets_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P50"
    assert data["status"] == "direct_raw_fit_to_bridge_targets_available_no_model_no_vae_no_gsb_no_science"
    assert data["bridge_target_count"] == 3
    assert data["targets_passed"] == 3
    assert data["lambda_values"] == [0.25, 0.5, 0.75]
    assert data["all_targets_loss_decreased"] is True
    assert data["all_losses_finite"] is True
    
    # Verify boundaries and no claims
    assert data["no_model"] is True
    assert data["no_vae"] is True
    assert data["no_dataset"] is True
    assert data["no_torch_optimizer"] is True
    assert data["no_gsb_claim"] is True
    assert data["no_generation_claim"] is True
    assert data["no_scientific_conclusion"] is True
    assert data["realizability_claim"] == "optimization_approachability_only_not_state_validity"
    
    # Exclude raw tensors, full signatures, target tensors
    for k in data:
        assert "raw" not in k.lower()
        assert "sig" not in k.lower()


def test_p50_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p50_direct_raw_fit_to_endpoint_bridge_targets_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
