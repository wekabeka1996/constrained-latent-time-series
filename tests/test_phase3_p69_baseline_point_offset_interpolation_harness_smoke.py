# tests/test_phase3_p69_baseline_point_offset_interpolation_harness_smoke.py

import json
import os
import subprocess
import sys


def test_p69_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p69_baseline_point_offset_interpolation_harness_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "P69_READY_FOR_REVIEW"
    assert data["phase"] == "P69"
    assert data["phase_group"] == "PHASE_3"
    assert data["contract_version"] == "phase3_p69_baseline_point_offset_interpolation_harness_contract_v1"
    
    # Check baseline outputs presence
    out = data["baseline_outputs"]
    assert "linear_latent_interpolation" in out
    assert "z_b_minus_z_a_offset_transfer" in out
    assert "mean_offset_per_relation_type" in out
    assert "no_relation_apply_or_decoder_baseline" in out
    assert "random_relation_vector_baseline" in out
    
    assert data["training_allowed"] is False
    assert data["dataset_generation_allowed"] is False
    assert data["model_implementation_allowed"] is False
    assert data["optimization_allowed"] is False
    assert data["torch_allowed"] is False
    assert data["numpy_allowed"] is False
    assert data["bridge_implementation_allowed"] is False


def test_p69_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p69_baseline_point_offset_interpolation_harness_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
