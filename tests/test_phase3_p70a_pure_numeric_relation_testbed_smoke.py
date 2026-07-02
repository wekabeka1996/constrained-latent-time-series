# tests/test_phase3_p70a_pure_numeric_relation_testbed_smoke.py

import json
import os
import subprocess
import sys


def test_p70a_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p70a_pure_numeric_relation_testbed_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "P70A_READY_FOR_REVIEW"
    assert data["phase"] == "P70A"
    assert data["phase_group"] == "PHASE_3"
    assert data["contract_version"] == "phase3_p70a_pure_numeric_relation_testbed_contract_v1"
    
    assert data["p69_contract_validated"] is True
    assert data["training_allowed"] is False
    assert data["model_implementation_allowed"] is False
    assert data["optimization_allowed"] is False
    assert data["torch_allowed"] is False
    assert data["numpy_allowed"] is False
    assert data["bridge_implementation_allowed"] is False
    
    # Check specs presence
    specs = data["testbed"]["relation_specs"]
    assert "translate_x" in specs
    assert "translate_y" in specs
    assert "scale_s" in specs
    assert "reflect_x" in specs
    assert "nonlinear_x_from_y" in specs
    
    # Check sanity summary
    sanity = data["sanity_summary"]
    assert sanity["all_relation_types_present"] is True
    assert sanity["all_relation_types_have_repeated_instances"] is True
    assert sanity["max_invariant_violation"] == 0.0
    assert sanity["negative_control_records_present"] is True


def test_p70a_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase3/run_p70a_pure_numeric_relation_testbed_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
