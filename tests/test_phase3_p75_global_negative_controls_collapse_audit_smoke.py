# tests/test_phase3_p75_global_negative_controls_collapse_audit_smoke.py

import json
import os
import subprocess
import sys


def test_p75_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p75_global_negative_controls_collapse_audit_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "P75_READY_FOR_REVIEW"
    assert data["phase"] == "P75"
    assert data["phase_group"] == "PHASE_3"
    assert data["contract_version"] == "phase3_p75_global_negative_controls_collapse_audit_contract_v1"
    
    assert data["source_contracts_deep_validated"] is True
    assert data["p74_source_validation_depth_note_addressed"] is True
    assert data["p74_target_midpoint_note_addressed"] is True
    assert data["target_endpoint_used_for_prediction"] is False
    assert data["target_endpoint_used_for_evaluation_only"] is True
    assert data["target_midpoint_used_for_prediction"] is False
    assert data["target_midpoint_used_for_evaluation_only"] is True
    
    assert data["training_allowed"] is False
    assert data["model_implementation_allowed"] is False
    assert data["optimization_allowed"] is False
    assert data["torch_allowed"] is False
    assert data["numpy_allowed"] is False
    assert data["bridge_implementation_allowed"] is False
    
    # Check sanity summary
    sanity = data["sanity_summary"]
    assert sanity["source_contracts_deep_validated"] is True
    assert sanity["p74_source_validation_depth_note_addressed"] is True
    assert sanity["p74_target_midpoint_note_addressed"] is True
    assert sanity["target_endpoint_used_for_prediction"] is False
    assert sanity["target_midpoint_used_for_prediction"] is False
    
    assert sanity["p70a_positive_oracle_control_exact"] is True
    assert sanity["p70b_positive_oracle_control_exact"] is True
    assert sanity["p70a_negative_controls_diagnostic_pass"] is True
    assert sanity["p70b_negative_controls_diagnostic_pass"] is True
    assert sanity["structural_negative_controls_ready"] is True
    
    assert sanity["p70a_known_ambiguity_reported"] is True
    assert sanity["p70a_unexpected_false_passes_zero"] is True
    assert sanity["p70b_unexpected_false_passes_zero"] is True
    
    assert sanity["wrong_operator_controls_fail_as_expected"] is True
    assert sanity["no_op_collapse_controls_fail_as_expected"] is True
    assert sanity["constant_output_collapse_controls_fail_as_expected"] is True
    assert sanity["source_agnostic_controls_fail_as_expected"] is True
    assert sanity["failure_rate_measured_only_when_executed"] is True
    
    assert sanity["learned_negative_control_claims_made"] is False
    assert sanity["training_or_model_added"] is False


def test_p75_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p75_global_negative_controls_collapse_audit_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
