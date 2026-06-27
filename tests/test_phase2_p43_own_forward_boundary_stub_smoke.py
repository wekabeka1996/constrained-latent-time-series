# tests/test_phase2_p43_own_forward_boundary_stub_smoke.py

import json
import os
import subprocess
import sys


def test_p43_smoke_runner_execution():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p43_own_forward_boundary_stub_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    assert res.returncode == 0
    stdout = res.stdout.strip()
    
    # Assert single line JSON
    assert "\n" not in stdout
    
    data = json.loads(stdout)
    
    # Assert public fields
    assert data["verdict"] == "PASS"
    assert data["contract"] == "phase2_p43_own_forward_boundary_stub_contract_v1"
    assert data["source_phase"] == "P43"
    assert data["status"] == "forward_boundary_declared_no_execution_no_output_no_training"
    assert data["torch_available"] is True
    assert data["class_declared"] is True
    assert data["module_instance_created"] is True
    assert data["module_object_returned"] is False
    assert data["is_torch_nn_module"] is True
    assert data["own_forward_declared"] is True
    assert data["forward_signature_available"] is True
    assert data["forward_execution_attempted"] is False
    assert data["forward_execution_available_in_p43"] is False
    assert data["forward_executed_in_p43"] is False
    assert data["output_generation_available_in_p43"] is False
    assert data["output_generated_in_p43"] is False
    assert data["training_available_in_p43"] is False
    assert data["training_executed_in_p43"] is False
    
    # Assert all no_* are True
    for key in (
        "no_forward_execution",
        "no_output_generation",
        "no_training_loop",
        "no_optimizer",
        "no_checkpointing",
        "no_artifact_generation",
        "no_final_comparison",
        "no_scientific_conclusion"
    ):
        assert data[key] is True, f"{key} must be True"
        
    # Assert no local path leakage or forbidden success claims
    def recursive_check(item):
        if isinstance(item, str):
            for path_part in (":\\", "Users", "home", "/Users", "/home", ".git"):
                assert path_part.lower() not in item.lower(), f"Leakage detected: {item}"
            for claim in ("scientific success", "solved", "best", "winner", "production ready", "state of the art"):
                assert claim.lower() not in item.lower(), f"Forbidden claim detected: {item}"
        elif isinstance(item, dict):
            for k, v in item.items():
                recursive_check(k)
                recursive_check(v)
        elif isinstance(item, list):
            for x in item:
                recursive_check(x)

    recursive_check(data)
