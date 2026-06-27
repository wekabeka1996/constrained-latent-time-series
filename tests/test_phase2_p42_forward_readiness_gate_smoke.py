# tests/test_phase2_p42_forward_readiness_gate_smoke.py

import json
import subprocess
import sys


def test_p42_smoke_runner_execution():
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p42_forward_readiness_gate_smoke.py"],
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
    assert data["contract"] == "phase2_p42_forward_readiness_gate_after_module_shell_contract_v1"
    assert data["source_phase"] == "P42"
    assert data["status"] == "blocked_by_forward_implementation_unavailable"
    assert data["torch_available"] is True
    assert data["tensor_materialized_in_p39"] is True
    assert data["module_shell_created_in_p41"] is True
    assert data["implementation_available_in_p41"] is True
    assert data["module_shell_ready_in_p42"] is True
    assert data["forward_ready_in_p42"] is False
    assert data["forward_implementation_available_in_p42"] is False
    assert data["forward_available_in_p42"] is False
    
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
        
    # Assert no local path leakage in keys/values
    def recursive_check(item):
        if isinstance(item, str):
            for path_part in (":\\", "Users", "home", "/Users", "/home", ".git"):
                assert path_part.lower() not in item.lower(), f"Leakage detected: {item}"
        elif isinstance(item, dict):
            for k, v in item.items():
                recursive_check(k)
                recursive_check(v)
        elif isinstance(item, list):
            for x in item:
                recursive_check(x)

    recursive_check(data)
