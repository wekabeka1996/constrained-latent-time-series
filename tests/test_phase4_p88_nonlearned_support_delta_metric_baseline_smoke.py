# tests/test_phase4_p88_nonlearned_support_delta_metric_baseline_smoke.py

import json
import os
import sys
import subprocess


def test_p88_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p88_nonlearned_support_delta_metric_baseline_smoke.py"],
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
    assert data["verdict"] == "P88_READY_FOR_REVIEW"
    assert data["nonlearned_metric_signal_present"] is True
    assert data["primary_support_split_audit"]["same_split_diagonal_only"] is True
    assert data["primary_support_split_audit"]["off_diagonal_support_count"] == 0
    assert data["metric_input_leakage_audit"]["diagnostic_pass"] is True
    assert "mode" in data["best_metric_result"]
    assert "distance_family" in data["best_metric_result"]
    assert data["bridge_ready"] is False


def test_p88_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p88_nonlearned_support_delta_metric_baseline_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
