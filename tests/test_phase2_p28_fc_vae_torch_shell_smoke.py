# tests/test_phase2_p28_fc_vae_torch_shell_smoke.py

import json
import pathlib
import pytest

from tools.phase2.run_p28_fc_vae_torch_shell_smoke import (
    run_p28_fc_vae_torch_shell_smoke,
    compact_json,
    main,
)


def test_p28_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'


def test_p28_smoke_02_main_rejects_args():
    import sys
    orig_argv = sys.argv
    try:
        sys.argv = ["script.py", "extra_arg"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig_argv


def test_p28_smoke_03_run_smoke_verdict_pass():
    res = run_p28_fc_vae_torch_shell_smoke()
    assert res["verdict"] == "PASS"


def test_p28_smoke_04_output_contract_matches():
    res = run_p28_fc_vae_torch_shell_smoke()
    assert res["contract"] == "phase2_p28_fc_vae_torch_shell_contract_v1"


def test_p28_smoke_05_output_source_phase():
    res = run_p28_fc_vae_torch_shell_smoke()
    assert res["source_phase"] == "P28"


def test_p28_smoke_06_torch_required_for_p28_false():
    res = run_p28_fc_vae_torch_shell_smoke()
    assert res["torch_required_for_p28"] is False


def test_p28_smoke_07_torch_required_for_future_execution_true():
    res = run_p28_fc_vae_torch_shell_smoke()
    assert res["torch_required_for_future_execution"] is True


def test_p28_smoke_08_module_created_false():
    res = run_p28_fc_vae_torch_shell_smoke()
    assert res["module_created"] is False


def test_p28_smoke_09_implementation_available_in_p28_false():
    res = run_p28_fc_vae_torch_shell_smoke()
    assert res["implementation_available_in_p28"] is False


def test_p28_smoke_10_all_no_flags_true():
    res = run_p28_fc_vae_torch_shell_smoke()
    assert res["no_model_implementation"] is True
    assert res["no_training_loop"] is True
    assert res["no_optimizer"] is True
    assert res["no_checkpointing"] is True
    assert res["no_artifact_generation"] is True


def test_p28_smoke_11_output_json_compact_parseable():
    res = run_p28_fc_vae_torch_shell_smoke()
    js = compact_json(res)
    assert " " not in js
    parsed = json.loads(js)
    assert parsed["verdict"] == "PASS"


def test_p28_smoke_12_does_not_import_torch():
    p = pathlib.Path("tools/phase2/run_p28_fc_vae_torch_shell_smoke.py").read_text(encoding="utf-8")
    for line in p.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("import torch"), f"Forbidden torch import: {stripped}"
        assert not stripped.startswith("from torch"), f"Forbidden torch import: {stripped}"


def test_p28_smoke_13_does_not_import_forbidden_libs():
    p = pathlib.Path("tools/phase2/run_p28_fc_vae_torch_shell_smoke.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn", "yaml", "argparse", "subprocess"]
    for f in forbidden:
        assert f"import {f}" not in p
        assert f"from {f}" not in p


def test_p28_smoke_14_no_artifact_generation_calls():
    p = pathlib.Path("tools/phase2/run_p28_fc_vae_torch_shell_smoke.py").read_text(encoding="utf-8")
    assert "write_dataset_artifacts" not in p
    assert "run_phase2_artifact_generation" not in p


def test_p28_smoke_15_no_training_class_or_fn():
    p = pathlib.Path("tools/phase2/run_p28_fc_vae_torch_shell_smoke.py").read_text(encoding="utf-8")
    forbidden = ["class ", "def train", "def fit", "def optimizer", "def checkpoint"]
    for f in forbidden:
        if f == "class ":
            assert f not in p
        else:
            assert f not in p


def test_p28_smoke_16_no_argparse():
    p = pathlib.Path("tools/phase2/run_p28_fc_vae_torch_shell_smoke.py").read_text(encoding="utf-8")
    assert "argparse" not in p
    assert "ArgumentParser" not in p


def test_p28_smoke_17_no_subprocess():
    p = pathlib.Path("tools/phase2/run_p28_fc_vae_torch_shell_smoke.py").read_text(encoding="utf-8")
    assert "subprocess" not in p


def test_p28_smoke_18_no_claims_in_output():
    res = run_p28_fc_vae_torch_shell_smoke()
    js = compact_json(res).lower()
    assert "model works" not in js
    assert "scientific success" not in js
    assert "solved" not in js
    assert "best" not in js
    assert "winner" not in js
