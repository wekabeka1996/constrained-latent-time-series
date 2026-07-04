# tests/test_phase4_p83_clean_input_identifiability_collision_audit_smoke.py

import json
import os
import sys
import subprocess


def test_p83_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p83_clean_input_identifiability_collision_audit_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    assert res.returncode == 0
    
    # Verify stdout is valid JSON
    data = json.loads(res.stdout.strip())
    assert isinstance(data, dict)
    
    # Verify outcomes
    assert data["verdict"] == "P83_READY_FOR_REVIEW"
    assert data["clean_input_identifiability_audit_performed"] is True
    assert data["clean_input_label_collision_audit_performed"] is True
    assert data["deterministic_oracle_upper_bound_computed"] is True
    assert data["model_training_performed"] is False
    assert data["bridge_ready"] is False


def test_p83_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p83_clean_input_identifiability_collision_audit_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
