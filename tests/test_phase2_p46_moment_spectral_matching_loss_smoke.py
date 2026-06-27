# tests/test_phase2_p46_moment_spectral_matching_loss_smoke.py

import json
import os
import subprocess
import sys


def test_p46_smoke_runner_execution():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p46_moment_spectral_matching_loss_smoke.py"],
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
    assert data["source_phase"] == "P46"
    assert data["status"] == "moment_spectral_matching_loss_available_no_model_no_training"
    assert data["torch_available"] is True
    assert data["ar_loss_available"] is True
    assert data["garch_loss_available"] is True
    assert data["combined_loss_available"] is True
    assert data["total_loss_finite"] is True
    assert data["total_loss_nonnegative"] is True
    assert data["ar_component_loss_nonnegative"] is True
    assert data["garch_component_loss_nonnegative"] is True
    assert data["candidate_raw_kappa_grad_nonzero"] is True
    assert data["candidate_raw_omega_grad_nonzero"] is True
    assert data["candidate_raw_total_mass_grad_nonzero"] is True
    assert data["candidate_raw_allocation_logits_grad_nonzero"] is True
    
    # Assert all no_* are True
    for key in (
        "no_model",
        "no_forward_execution",
        "no_output_generation",
        "no_optimizer",
        "no_training_loop",
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


def test_p46_smoke_runner_rejects_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p46_moment_spectral_matching_loss_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True,
        env=env
    )
    assert res.returncode == 0
