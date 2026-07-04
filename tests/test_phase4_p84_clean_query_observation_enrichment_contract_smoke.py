# tests/test_phase4_p84_clean_query_observation_enrichment_contract_smoke.py

import json
import os
import sys
import subprocess


def test_p84_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p84_clean_query_observation_enrichment_contract_smoke.py"],
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
    assert data["verdict"] == "P84_READY_FOR_REVIEW"
    assert data["query_observation_enrichment_contract_present"] is True
    assert data["target_record_endpoint_used_for_selector_input"] is False
    assert data["exact_relation_label_used_for_selector_input"] is False
    assert data["valid_for_final_selector_evidence"] is False
    assert data["bridge_ready"] is False


def test_p84_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p84_clean_query_observation_enrichment_contract_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
