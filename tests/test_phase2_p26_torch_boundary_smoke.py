# tests/test_phase2_p26_torch_boundary_smoke.py

import json
import pathlib
import subprocess
import sys
import pytest


def test_p26_smoke_01_smoke_script_exits_zero():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p26_torch_boundary_smoke.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode == 0, f"Smoke script failed: {result.stderr}"


def test_p26_smoke_02_smoke_output_valid_json():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p26_torch_boundary_smoke.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert type(data) is dict


def test_p26_smoke_03_smoke_verdict_pass():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p26_torch_boundary_smoke.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["verdict"] == "PASS"


def test_p26_smoke_04_smoke_contract_version():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p26_torch_boundary_smoke.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["contract"] == "phase2_p26_torch_boundary_contract_v1"


def test_p26_smoke_05_smoke_no_flags():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p26_torch_boundary_smoke.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["no_model_implementation"] is True
    assert data["no_training_loop"] is True
    assert data["no_optimizer"] is True
    assert data["no_checkpointing"] is True
    assert data["no_artifact_generation"] is True
    assert data["no_final_comparison"] is True
    assert data["no_scientific_conclusion"] is True


def test_p26_smoke_06_smoke_torch_boundary_nested():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p26_torch_boundary_smoke.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    tb = data["torch_boundary"]
    assert type(tb) is dict
    assert "torch_status" in tb
    assert "future_model_boundary" in tb


def test_p26_smoke_07_smoke_future_model_not_allowed():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p26_torch_boundary_smoke.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["future_model_allowed_in_p26"] is False


def test_p26_smoke_08_smoke_torch_required_false():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p26_torch_boundary_smoke.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["torch_required_for_p26"] is False


def test_p26_smoke_09_no_local_paths_in_output():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p26_torch_boundary_smoke.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode == 0
    for forbidden in ["file:///", "C:/", "C:\\", "/home/", "/Users/"]:
        assert forbidden not in result.stdout


def test_p26_smoke_10_smoke_script_rejects_args():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p26_torch_boundary_smoke.py", "--unexpected"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode != 0
