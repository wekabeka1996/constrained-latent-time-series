# tests/test_phase3_p76_relation_metric_selector_prebridge_audit_smoke.py

import json
import os
import sys
import subprocess


def test_p76_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p76_relation_metric_selector_prebridge_audit_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    assert res.returncode == 0
    
    # Verify stdout is valid JSON
    data = json.loads(res.stdout.strip())
    assert isinstance(data, dict)
    
    # Verify exact keys and outcomes
    assert data["verdict"] == "P76_READY_FOR_REVIEW"
    assert data["source_contracts_validated"] is True
    assert data["descriptor_dependency_audit"]["all_current_strong_descriptors_target_dependent"] is True
    assert data["descriptor_dependency_audit"]["target_dependent_descriptors_blocked_for_prediction"] is True
    assert data["valid_pre_prediction_selector_available"] is False
    assert data["semantic_metric_ready"] is False
    assert data["bridge_ready"] is False
    assert data["sanity_summary"]["predictive_selector_claims_made"] is False


def test_p76_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p76_relation_metric_selector_prebridge_audit_smoke.py", "--some-arg"],
        capture_output=True,
        text=True,
        env=env
    )
    
    # Must exit non-zero
    assert res.returncode != 0
    assert "Error:" in res.stderr
