# tests/test_phase3_p71_relation_contrastive_signal_smoke.py

import json
import os
import subprocess
import sys


def test_p71_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p71_relation_contrastive_signal_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "P71_READY_FOR_REVIEW"
    assert data["phase"] == "P71"
    assert data["phase_group"] == "PHASE_3"
    assert data["contract_version"] == "phase3_p71_relation_descriptor_contrastive_signal_smoke_contract_v1"
    
    assert data["source_contracts_validated"] is True
    assert data["training_allowed"] is False
    assert data["model_implementation_allowed"] is False
    assert data["optimization_allowed"] is False
    assert data["torch_allowed"] is False
    assert data["numpy_allowed"] is False
    assert data["bridge_implementation_allowed"] is False
    
    # Check evaluations
    evals = data["descriptor_evaluations"]
    assert "p70a_vector_delta_descriptor" in evals
    assert "p70b_parameter_delta_descriptor" in evals
    assert "p70b_series_summary_delta_descriptor" in evals
    
    # Check sanity summary
    sanity = data["sanity_summary"]
    assert sanity["source_contracts_validated"] is True
    assert sanity["all_descriptor_views_present"] is True
    assert sanity["p70b_parameter_descriptor_strong_pass"] is True
    assert sanity["label_permutation_records_ready"] is True
    assert sanity["mismatched_pair_records_ready"] is True
    assert sanity["labels_used_for_descriptor_construction"] is False
    assert sanity["labels_used_for_evaluation_only"] is True


def test_p71_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p71_relation_contrastive_signal_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
