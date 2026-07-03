# tests/test_phase3_p79_hard_ablated_selector_evidence_gate_smoke.py

import json
import os
import sys
import subprocess


def test_p79_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with no arguments
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p79_hard_ablated_selector_evidence_gate_smoke.py"],
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
    assert data["verdict"] == "P79_READY_FOR_REVIEW"
    assert data["source_contracts_validated"] is True
    assert data["hard_ablated_selector_evidence_gate_audit"]["hard_ablated_selector_evidence_gate_evaluated"] is True
    assert data["hard_ablated_selector_evidence_gate_audit"]["relation_specific_hints_removed"] is True
    
    audit_a = data["hard_ablated_selector_evidence_gate_audit"]["p70a_hard_ablated_input_audit"]
    audit_b = data["hard_ablated_selector_evidence_gate_audit"]["p70b_hard_ablated_input_audit"]
    
    for domain_audit in [audit_a, audit_b]:
        assert domain_audit["audit_metadata_present_count"] == 0
        assert domain_audit["exact_label_present_count"] == 0
        assert domain_audit["target_endpoint_present_count"] == 0
        assert domain_audit["target_delta_present_count"] == 0
        assert domain_audit["relation_family_hint_present_count"] == 0
        assert domain_audit["transformation_class_hint_present_count"] == 0
        assert domain_audit["axis_or_group_hint_present_count"] == 0
        
    assert data["learned_selector_evidence_present"] is False
    assert data["bridge_ready"] is False


def test_p79_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool with an extra argument
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p79_hard_ablated_selector_evidence_gate_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
