# tests/test_phase3_p73_transfer_invariant_preservation_audit_smoke.py

import json
import os
import subprocess
import sys


def test_p73_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p73_transfer_invariant_preservation_audit_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "P73_READY_FOR_REVIEW"
    assert data["phase"] == "P73"
    assert data["phase_group"] == "PHASE_3"
    assert data["contract_version"] == "phase3_p73_transfer_invariant_preservation_audit_contract_v1"
    
    assert data["source_contracts_complete_validated"] is True
    assert data["p72_validation_completeness_note_addressed"] is True
    assert data["target_endpoint_used_for_prediction"] is False
    assert data["target_endpoint_used_for_evaluation_only"] is True
    assert data["training_allowed"] is False
    assert data["model_implementation_allowed"] is False
    assert data["optimization_allowed"] is False
    assert data["torch_allowed"] is False
    assert data["numpy_allowed"] is False
    assert data["bridge_implementation_allowed"] is False
    
    # Check sanity summary
    sanity = data["sanity_summary"]
    assert sanity["source_contracts_complete_validated"] is True
    assert sanity["p72_validation_completeness_note_addressed"] is True
    assert sanity["target_endpoint_used_for_prediction"] is False
    assert sanity["target_endpoint_used_for_evaluation_only"] is True
    
    assert sanity["p70a_transfer_diagnostic_pass"] is True
    assert sanity["p70b_transfer_diagnostic_pass"] is True
    assert sanity["negative_control_diagnostic_pass"] is True
    assert sanity["p70a_all_splits_exact"] is True
    assert sanity["p70b_all_splits_exact"] is True
    assert sanity["p70a_invariants_preserved"] is True
    assert sanity["p70b_invariants_preserved"] is True
    assert sanity["p70a_transfer_gaps_zero"] is True
    assert sanity["p70b_transfer_gaps_zero"] is True
    assert sanity["heldout_base_audited"] is True
    assert sanity["heldout_magnitude_audited"] is True
    assert sanity["learned_transfer_claims_made"] is False
    assert sanity["training_or_model_added"] is False


def test_p73_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p73_transfer_invariant_preservation_audit_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
