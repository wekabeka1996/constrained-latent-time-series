# tests/test_phase2_p49_deterministic_endpoint_bridge_targets_smoke.py

import json
import subprocess
import sys
import os


def test_p49_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p49_deterministic_endpoint_bridge_targets_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    assert res.returncode == 0
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P49"
    assert data["status"] == "endpoint_bridge_targets_available_no_fit_no_model_no_science"
    assert data["endpoint_count"] == 2
    assert data["bridge_target_count"] == 3
    assert data["lambda_values"] == [0.25, 0.50, 0.75]
    
    assert data["all_targets_finite"] is True
    assert data["all_endpoint_tensors_no_grad"] is True
    
    assert data["no_loss"] is True
    assert data["no_fit"] is True
    assert data["no_optimization"] is True
    assert data["no_model"] is True
    assert data["no_vae"] is True
    assert data["no_encoder"] is True
    assert data["no_decoder"] is True
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    assert data["no_torch_optimizer"] is True
    assert data["no_scientific_conclusion"] is True
    assert data["no_gsb_claim"] is True
    assert data["no_generation_claim"] is True
    
    # Check target summaries
    assert len(data["target_summaries"]) == 3
    for summary in data["target_summaries"]:
        assert summary["all_tensor_fields_finite"] is True
        assert summary["construction_method"] == "linear_interpolation_in_p45_combined_signature_space"
        assert summary["realizability_claim"] == "not_claimed"
        assert summary["tensor_field_count"] == 11
        
    # Check that no raw tensors are in stdout
    assert "raw_kappa" not in res.stdout
    assert "ar_spectrum" not in res.stdout


def test_p49_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p49_deterministic_endpoint_bridge_targets_smoke.py", "--invalid"],
        capture_output=True,
        text=True,
        env=env
    )
    assert res.returncode != 0
    assert "does not accept command line arguments" in res.stderr
