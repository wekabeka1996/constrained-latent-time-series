# tests/test_phase4_p80_learned_experiment_authority_contract_smoke.py

import json
import os
import sys
import subprocess


def test_p80_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p80_learned_experiment_authority_contract_smoke.py"],
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
    assert data["verdict"] == "P80_READY_FOR_REVIEW"
    assert data["phase_group"] == "PHASE_4"
    assert data["phase4_learned_experiments_allowed"] is True
    assert data["bridge_implementation_allowed"] is False
    assert data["bridge_ready"] is False


def test_p80_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p80_learned_experiment_authority_contract_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
