# tests/test_phase3_p68_latent_operator_genesis_research_contract_smoke.py

import json
import os
import subprocess
import sys


def test_p68_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p68_latent_operator_genesis_research_contract_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "P68_READY_FOR_REVIEW"
    assert data["phase"] == "P68"
    assert data["phase_group"] == "PHASE_3"
    assert data["contract_version"] == "phase3_p68_latent_operator_genesis_research_contract_v1"
    
    assert data["p70a_required_before_p70b"] is True
    assert data["contrastive_signal_required_for_p71"] is True
    assert data["relation_labels_decoder_forbidden"] is True
    assert data["permutation_control_required_per_module"] is True
    assert data["bridge_deferred_until_metric_and_operator_evidence"] is True
    
    assert data["training_allowed"] is False
    assert data["dataset_generation_allowed"] is False
    assert data["model_implementation_allowed"] is False
    assert data["optimization_allowed"] is False
    assert data["bridge_implementation_allowed"] is False


def test_p68_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p68_latent_operator_genesis_research_contract_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
