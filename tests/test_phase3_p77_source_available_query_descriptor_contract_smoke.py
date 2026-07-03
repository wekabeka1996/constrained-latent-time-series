# tests/test_phase3_p77_source_available_query_descriptor_contract_smoke.py

import json
import os
import sys
import subprocess


def test_p77_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p77_source_available_query_descriptor_contract_smoke.py"],
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
    assert data["verdict"] == "P77_READY_FOR_REVIEW"
    assert data["source_contracts_validated"] is True
    assert data["source_available_query_contract_audit"]["source_available_query_contract_present"] is True
    assert data["source_available_query_contract_audit"]["endpoint_leakage_detected"] is False
    assert data["source_available_query_contract_audit"]["target_delta_leakage_detected"] is False
    assert data["source_available_query_contract_audit"]["exact_label_pass_through_detected"] is False
    assert data["valid_for_future_selector_experiment"] is True
    assert data["learned_selector_evidence_present"] is False
    assert data["bridge_ready"] is False


def test_p77_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p77_source_available_query_descriptor_contract_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
