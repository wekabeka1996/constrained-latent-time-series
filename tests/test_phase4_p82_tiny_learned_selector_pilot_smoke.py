# tests/test_phase4_p82_tiny_learned_selector_pilot_smoke.py

import json
import os
import sys
import subprocess


def test_p82_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p82_tiny_learned_selector_pilot_smoke.py"],
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
    assert data["verdict"] == "P82_READY_FOR_REVIEW"
    assert data["tiny_learned_selector_trained"] is True
    assert data["clean_selector_input_used"] is True
    assert data["hint_passthrough_used_as_model_input"] is False
    assert data["model_input_audit"]["selector_input_leakage_detected"] is False
    assert data["bridge_ready"] is False


def test_p82_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p82_tiny_learned_selector_pilot_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
