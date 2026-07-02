# tests/test_phase3_p70b_synthetic_time_series_relation_testbed_smoke.py

import json
import os
import subprocess
import sys


def test_p70b_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p70b_synthetic_time_series_relation_testbed_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "P70B_READY_FOR_REVIEW"
    assert data["phase"] == "P70B"
    assert data["phase_group"] == "PHASE_3"
    assert data["contract_version"] == "phase3_p70b_synthetic_time_series_relation_testbed_contract_v1"
    
    assert data["p70a_contract_validated"] is True
    assert data["training_allowed"] is False
    assert data["model_implementation_allowed"] is False
    assert data["optimization_allowed"] is False
    assert data["torch_allowed"] is False
    assert data["numpy_allowed"] is False
    assert data["bridge_implementation_allowed"] is False
    
    # Check specs presence
    specs = data["testbed"]["relation_specs"]
    assert "change_frequency" in specs
    assert "scale_amplitude" in specs
    assert "shift_phase" in specs
    assert "scale_volatility_envelope" in specs
    assert "shift_trend" in specs
    
    # Check sanity summary
    sanity = data["sanity_summary"]
    assert sanity["all_relation_types_present"] is True
    assert sanity["all_relation_types_have_repeated_instances"] is True
    assert sanity["negative_controls_nondegenerate"] is True
    assert sanity["negative_control_endpoint_collision_count"] == 0
    assert sanity["max_parameter_invariant_violation"] == 0.0
    assert sanity["all_series_lengths_match"] is True
    assert sanity["all_series_values_finite"] is True


def test_p70b_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p70b_synthetic_time_series_relation_testbed_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
