# tests/test_phase2_p48_direct_raw_parameter_fit_robustness_smoke.py
#
# P48 Smoke Script integration tests.

import json
import os
import subprocess
import sys


def test_p48_smoke_script_execution():
    """Verify smoke script executes and outputs PASS JSON with 4/4 scenarios."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p48_direct_raw_parameter_fit_robustness_smoke.py"],
        capture_output=True, text=True, check=True,
        env=env,
    )
    stdout_clean = res.stdout.strip()
    assert "\n" not in stdout_clean, "Smoke output must be a single JSON line"

    parsed = json.loads(stdout_clean)
    assert parsed["verdict"] == "PASS"
    assert parsed["source_phase"] == "P48"
    assert parsed["status"] == "direct_fit_robustness_smoke_available_no_model_no_vae_no_science"
    assert parsed["torch_available"] is True
    assert parsed["scenario_count"] == 4
    assert parsed["scenarios_passed"] == 4
    assert parsed["all_scenarios_loss_decreased"] is True
    assert parsed["all_losses_finite"] is True
    assert parsed["no_model"] is True
    assert parsed["no_vae"] is True
    assert parsed["no_encoder"] is True
    assert parsed["no_decoder"] is True
    assert parsed["no_dataset"] is True
    assert parsed["no_dataloader"] is True
    assert parsed["no_torch_optimizer"] is True
    assert parsed["no_scientific_conclusion"] is True

    # Verify no raw tensors serialized
    def check_no_tensors(obj, path=""):
        if hasattr(obj, "backward"):
            raise AssertionError(f"Tensor at path '{path}'!")
        if isinstance(obj, dict):
            for k, v in obj.items():
                check_no_tensors(v, path=f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                check_no_tensors(v, path=f"{path}[{i}]")
    check_no_tensors(parsed)

    # Verify scenario_results array
    assert isinstance(parsed["scenario_results"], list)
    assert len(parsed["scenario_results"]) == 4
    for sr in parsed["scenario_results"]:
        assert sr["loss_decreased"] is True
        assert sr["initial_loss_finite"] is True
        assert sr["final_loss_finite"] is True
        assert sr["initial_loss_value"] > sr["final_loss_value"]


def test_p48_smoke_script_rejects_arguments():
    """Verify smoke script rejects command-line arguments."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p48_direct_raw_parameter_fit_robustness_smoke.py", "--invalid"],
        capture_output=True, text=True,
        env=env,
    )
    assert res.returncode != 0
    parsed = json.loads(res.stdout.strip())
    assert parsed["verdict"] == "REJECTED"
    assert "takes no arguments" in parsed["reason"]
