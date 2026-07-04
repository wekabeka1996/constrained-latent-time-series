# tests/test_phase4_p85_externalized_query_context_support_episode_contract_smoke.py

import json
import os
import sys
import subprocess


def test_p85_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p85_externalized_query_context_support_episode_contract_smoke.py"],
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
    assert data["verdict"] == "P85_READY_FOR_REVIEW"
    assert data["externalized_query_context_source_defined"] is True
    assert data["support_episode_context_defined"] is True
    assert data["support_selection_label_visible_to_selector"] is False
    assert data["evaluated_target_endpoint_used_for_selector_input"] is False
    assert data["exact_relation_label_used_for_selector_input"] is False
    assert data["valid_for_final_semantic_geometry_evidence"] is False
    assert data["bridge_ready"] is False


def test_p85_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p85_externalized_query_context_support_episode_contract_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
