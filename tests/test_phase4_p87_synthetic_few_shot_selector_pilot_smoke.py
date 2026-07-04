# tests/test_phase4_p87_synthetic_few_shot_selector_pilot_smoke.py

import json
import os
import sys
import subprocess


def test_p87_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p87_synthetic_few_shot_selector_pilot_smoke.py"],
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
    assert data["verdict"] == "P87_READY_FOR_REVIEW"
    assert data["support_split_policy_audit_defined"] is True
    assert isinstance(data["learned_selector_evidence_present"], bool)
    assert data["model_input_leakage_audit"]["diagnostic_pass"] is True
    assert data["primary_support_split_audit"]["same_split_support_strict_pass"] is True
    assert data["diagnostic_train_bank_support_split_audit"]["train_bank_support_diagnostic_pass"] is True
    assert data["semantic_geometry_claims_allowed"] is False
    assert data["bridge_ready"] is False


def test_p87_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p87_synthetic_few_shot_selector_pilot_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
