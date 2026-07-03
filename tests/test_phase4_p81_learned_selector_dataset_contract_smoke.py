# tests/test_phase4_p81_learned_selector_dataset_contract_smoke.py

import json
import os
import sys
import subprocess


def test_p81_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p81_learned_selector_dataset_contract_smoke.py"],
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
    assert data["verdict"] == "P81_READY_FOR_REVIEW"
    assert data["phase_group"] == "PHASE_4"
    assert data["learned_selector_dataset_contract_present"] is True
    assert data["selector_train_val_test_split_built"] is True
    assert data["dataset_contract_audit"]["selector_input_leakage_detected"] is False
    assert data["model_training_performed"] is False
    assert data["bridge_ready"] is False


def test_p81_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p81_learned_selector_dataset_contract_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
