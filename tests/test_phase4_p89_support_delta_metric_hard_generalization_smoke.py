# tests/test_phase4_p89_support_delta_metric_hard_generalization_smoke.py

import json
import os
import sys
import subprocess


def test_p89_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p89_support_delta_metric_hard_generalization_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    assert res.returncode == 0
    data = json.loads(res.stdout.strip())
    assert isinstance(data, dict)
    
    assert data["verdict"] == "P89_READY_FOR_REVIEW"
    assert data["hard_generalization_supported"] is True
    assert data["nonlearned_metric_signal_present"] is True
    assert data["external_context_contract"]["external_context_contract_defined"] is True
    assert data["negative_control_results"]["shuffled_support_delta_control_implemented"] is True
    assert data["hard_policy_controls_aligned"] is True
    assert data["hard_policy_fallback_controls_used"] is False
    assert data["best_raw_hard_policy"] == "train_to_heldout_base_state"
    assert data["best_evidence_passing_hard_policy"] == "train_to_heldout_base_state"
    assert data["bridge_ready"] is False


def test_p89_smoke_02_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    res = subprocess.run(
        [sys.executable, "tools/phase4/run_p89_support_delta_metric_hard_generalization_smoke.py", "--invalid-flag"],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert res.returncode != 0
    assert "Error:" in res.stderr
