# tests/test_phase3_p78_selector_candidate_under_contract_smoke.py

import json
import os
import sys
import subprocess


def test_p78_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p78_selector_candidate_under_contract_smoke.py"],
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
    assert data["verdict"] == "P78_READY_FOR_REVIEW"
    assert data["source_contracts_validated"] is True
    assert data["selector_candidate_audit"]["rule_based_selector_candidate_evaluated"] is True
    assert data["selector_candidate_audit"]["selector_contract_record_audit"]["stripped_records_free_of_target_endpoint"] is True
    assert data["selector_candidate_audit"]["selector_contract_record_audit"]["stripped_records_free_of_target_delta"] is True
    assert data["selector_candidate_audit"]["selector_contract_record_audit"]["stripped_records_free_of_exact_relation_label"] is True
    assert data["selector_candidate_audit"]["selector_contract_record_audit"]["stripped_records_free_of_audit_metadata"] is True
    assert data["selector_candidate_audit"]["family_hint_pass_through_risk_present"] is True
    assert data["learned_selector_evidence_present"] is False
    assert data["bridge_ready"] is False


def test_p78_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p78_selector_candidate_under_contract_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
