# tests/test_phase2_p47_direct_raw_parameter_fit_smoke.py
#
# P47 Smoke Script integration tests.

import json
import subprocess
import sys


import os

def test_p47_smoke_script_execution():
    """Verify run_p47_direct_raw_parameter_fit_smoke.py executes and outputs valid PASS JSON."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p47_direct_raw_parameter_fit_smoke.py"],
        capture_output=True, text=True, check=True,
        env=env
    )
    
    # Must print a single compact JSON line
    stdout_clean = res.stdout.strip()
    assert "\n" not in stdout_clean, "Smoke script output must be a single line"
    
    parsed = json.loads(stdout_clean)
    assert parsed["verdict"] == "PASS"
    assert parsed["source_phase"] == "P47"
    assert parsed["status"] == "direct_raw_parameter_fit_smoke_available_no_model_no_vae_no_science"
    assert parsed["torch_available"] is True
    
    # Verify loss and non-model assertions
    assert parsed["loss_decreased"] is True
    assert parsed["loss_decrease_positive"] is True
    assert parsed["initial_loss_value"] > parsed["final_loss_value"]
    assert parsed["step_count"] == 20
    assert parsed["no_model"] is True
    assert parsed["no_vae"] is True
    assert parsed["no_encoder"] is True
    assert parsed["no_decoder"] is True
    assert parsed["no_dataset"] is True
    assert parsed["no_dataloader"] is True
    assert parsed["no_torch_optimizer"] is True
    assert parsed["no_scientific_conclusion"] is True

    # Verify no raw tensors or parameter arrays are serialized
    for key in parsed.keys():
        assert "raw_" not in key, f"Forbidden serialization key found: {key}"
        assert not isinstance(parsed[key], list), f"Arrays should not be serialized: key={key}"


def test_p47_smoke_script_rejects_arguments():
    """Verify smoke script rejects command-line arguments."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p47_direct_raw_parameter_fit_smoke.py", "--invalid-arg"],
        capture_output=True, text=True,
        env=env
    )
    assert res.returncode != 0
    parsed = json.loads(res.stdout.strip())
    assert parsed["verdict"] == "REJECTED"
    assert "takes no arguments" in parsed["reason"]
