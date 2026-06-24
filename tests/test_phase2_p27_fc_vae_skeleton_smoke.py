# tests/test_phase2_p27_fc_vae_skeleton_smoke.py

import json
import pathlib
import subprocess
import sys
import pytest


def _run_smoke():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p27_fc_vae_skeleton_smoke.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    return result


def test_p27_smoke_01_compact_json_sorts():
    from tools.phase2.run_p27_fc_vae_skeleton_smoke import compact_json
    d = {"b": 2, "a": 1}
    js = compact_json(d)
    assert js.index('"a"') < js.index('"b"')


def test_p27_smoke_02_main_rejects_args():
    result = subprocess.run(
        [sys.executable, "tools/phase2/run_p27_fc_vae_skeleton_smoke.py", "--unexpected"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode != 0


def test_p27_smoke_03_run_returns_pass():
    result = _run_smoke()
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["verdict"] == "PASS"


def test_p27_smoke_04_output_contract_matches():
    result = _run_smoke()
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["contract"] == "phase2_p27_fc_vae_model_skeleton_contract_v1"


def test_p27_smoke_05_output_source_phase():
    result = _run_smoke()
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["source_phase"] == "P27"


def test_p27_smoke_06_torch_required_for_p27_false():
    result = _run_smoke()
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["torch_required_for_p27"] is False


def test_p27_smoke_07_torch_required_for_future_execution_true():
    result = _run_smoke()
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["torch_required_for_future_execution"] is True


def test_p27_smoke_08_all_no_flags_true():
    result = _run_smoke()
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert data["no_model_implementation"] is True
    assert data["no_training_loop"] is True
    assert data["no_optimizer"] is True
    assert data["no_checkpointing"] is True
    assert data["no_artifact_generation"] is True
    assert data["no_final_comparison"] is True
    assert data["no_scientific_conclusion"] is True


def test_p27_smoke_09_output_json_compact_parseable():
    result = _run_smoke()
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    assert type(data) is dict


def test_p27_smoke_10_script_does_not_import_torch():
    p = pathlib.Path("tools/phase2/run_p27_fc_vae_skeleton_smoke.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p


def test_p27_smoke_11_script_does_not_import_forbidden():
    p = pathlib.Path("tools/phase2/run_p27_fc_vae_skeleton_smoke.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn", "yaml"]
    for f in forbidden:
        assert f"import {f}" not in p
        assert f"from {f}" not in p


def test_p27_smoke_12_script_does_not_call_artifact_generation():
    p = pathlib.Path("tools/phase2/run_p27_fc_vae_skeleton_smoke.py").read_text(encoding="utf-8")
    forbidden = ["write_dataset_artifacts", "run_phase2_artifact_generation"]
    for f in forbidden:
        assert f not in p


def test_p27_smoke_13_script_no_training_definitions():
    p = pathlib.Path("tools/phase2/run_p27_fc_vae_skeleton_smoke.py").read_text(encoding="utf-8")
    forbidden = ["def train(", "def fit(", "def train_step(", "class Trainer",
                 "class Optimizer", "class Checkpoint"]
    for f in forbidden:
        assert f not in p


def test_p27_smoke_14_script_does_not_use_argparse():
    p = pathlib.Path("tools/phase2/run_p27_fc_vae_skeleton_smoke.py").read_text(encoding="utf-8")
    assert "argparse" not in p


def test_p27_smoke_15_script_does_not_use_subprocess():
    p = pathlib.Path("tools/phase2/run_p27_fc_vae_skeleton_smoke.py").read_text(encoding="utf-8")
    assert "subprocess" not in p


def test_p27_smoke_16_output_no_forbidden_claims():
    result = _run_smoke()
    assert result.returncode == 0
    stdout = result.stdout.lower()
    forbidden = ["final comparison", "scientific conclusion", "scientifically proven",
                 "outperforms", "state of the art"]
    for f in forbidden:
        assert f not in stdout
