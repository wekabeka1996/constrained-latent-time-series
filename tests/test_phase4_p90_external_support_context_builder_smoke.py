# tests/test_phase4_p90_external_support_context_builder_smoke.py

import json
import os
import sys
import subprocess


def test_p90_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p90_external_support_context_builder_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    assert res.returncode == 0
    data = json.loads(res.stdout.strip())
    assert isinstance(data, dict)
    
    assert data["verdict"] == "P90_READY_FOR_REVIEW"
    assert data["external_context_builder_ready"] is True
    assert data["non_label_selected_support_ready"] is True
    assert data["nonlearned_metric_signal_present"] is True
    assert data["hard_generalization_supported"] is True
    assert data["bridge_ready"] is False


def test_p90_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p90_external_support_context_builder_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error" in res.stderr
