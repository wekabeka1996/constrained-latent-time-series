# tests/test_phase3_p72_oracle_sparse_operator_bank_mvp_smoke.py

import json
import os
import subprocess
import sys


def test_p72_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p72_oracle_sparse_operator_bank_mvp_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "P72_READY_FOR_REVIEW"
    assert data["phase"] == "P72"
    assert data["phase_group"] == "PHASE_3"
    assert data["contract_version"] == "phase3_p72_oracle_sparse_operator_bank_mvp_contract_v1"
    
    assert data["source_contracts_deep_validated"] is True
    assert data["p71_validation_depth_note_addressed"] is True
    assert data["training_allowed"] is False
    assert data["model_implementation_allowed"] is False
    assert data["optimization_allowed"] is False
    assert data["torch_allowed"] is False
    assert data["numpy_allowed"] is False
    assert data["bridge_implementation_allowed"] is False
    
    # Check specs presence
    p70b_specs = data["operator_bank"]["p70b_time_series_operator_specs"]
    assert "change_frequency" in p70b_specs
    assert "scale_amplitude" in p70b_specs
    
    # Check sanity summary
    sanity = data["sanity_summary"]
    assert sanity["source_contracts_deep_validated"] is True
    assert sanity["p71_validation_depth_note_addressed"] is True
    assert sanity["operator_bank_present"] is True
    assert sanity["operator_bank_learned"] is False
    assert sanity["oracle_selection_used"] is True
    assert sanity["learned_selection_used"] is False
    
    assert sanity["p70a_operator_diagnostic_pass"] is True
    assert sanity["p70b_operator_diagnostic_pass"] is True
    assert sanity["p70a_oracle_endpoint_l2_mean_zero"] is True
    assert sanity["p70a_oracle_invariant_violation_max_zero"] is True
    assert sanity["p70b_oracle_parameter_l2_mean_zero"] is True
    assert sanity["p70b_oracle_series_l2_mean_zero"] is True
    assert sanity["p70b_oracle_invariant_violation_max_zero"] is True
    assert sanity["heldout_base_evaluated"] is True
    assert sanity["heldout_magnitude_evaluated"] is True
    assert sanity["oracle_beats_no_relation_baseline"] is True
    assert sanity["training_or_model_added"] is False


def test_p72_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p72_oracle_sparse_operator_bank_mvp_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
