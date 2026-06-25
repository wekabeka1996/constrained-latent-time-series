# tests/test_phase2_p30_constructor_binding_smoke.py

import json
import pathlib
import pytest

from tools.phase2.run_p30_constructor_binding_smoke import (
    run_p30_constructor_binding_smoke,
    compact_json,
    main,
)


def test_p30_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'


def test_p30_smoke_02_main_rejects_args():
    import sys
    orig = sys.argv
    try:
        sys.argv = ["script.py", "extra"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig


def test_p30_smoke_03_verdict_pass():
    res = run_p30_constructor_binding_smoke()
    assert res["verdict"] == "PASS"


def test_p30_smoke_04_contract_matches():
    res = run_p30_constructor_binding_smoke()
    assert res["contract"] == "phase2_p30_constructor_binding_contract_v1"


def test_p30_smoke_05_source_phase():
    res = run_p30_constructor_binding_smoke()
    assert res["source_phase"] == "P30"


def test_p30_smoke_06_torch_required_for_p30_false():
    res = run_p30_constructor_binding_smoke()
    assert res["torch_required_for_p30"] is False


def test_p30_smoke_07_torch_required_for_future_execution_true():
    res = run_p30_constructor_binding_smoke()
    assert res["torch_required_for_future_execution"] is True


def test_p30_smoke_08_module_object_returned_false():
    res = run_p30_constructor_binding_smoke()
    assert res["module_object_returned"] is False


def test_p30_smoke_09_constructor_binding_available_in_p30_false():
    res = run_p30_constructor_binding_smoke()
    assert res["constructor_binding_available_in_p30"] == res["binding_created"]


def test_p30_smoke_10_gating_flags_false():
    res = run_p30_constructor_binding_smoke()
    assert res["forward_available_in_p30"] is False
    assert res["layers_available_in_p30"] is False
    assert res["training_available_in_p30"] is False


def test_p30_smoke_11_all_no_flags_true():
    res = run_p30_constructor_binding_smoke()
    assert res["no_training_loop"] is True
    assert res["no_optimizer"] is True
    assert res["no_checkpointing"] is True
    assert res["no_artifact_generation"] is True
    assert res["no_final_comparison"] is True
    assert res["no_scientific_conclusion"] is True


def test_p30_smoke_12_json_parseable():
    res = run_p30_constructor_binding_smoke()
    js = compact_json(res)
    d = json.loads(js)
    assert d["verdict"] == "PASS"


def test_p30_smoke_13_no_torch_import():
    p = pathlib.Path("tools/phase2/run_p30_constructor_binding_smoke.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p


def test_p30_smoke_14_no_forbidden_libs():
    p = pathlib.Path("tools/phase2/run_p30_constructor_binding_smoke.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn", "yaml", "argparse", "subprocess"]
    for f in forbidden:
        assert f"import {f}" not in p
        assert f"from {f}" not in p


def test_p30_smoke_15_no_training_code():
    p = pathlib.Path("tools/phase2/run_p30_constructor_binding_smoke.py").read_text(encoding="utf-8")
    forbidden = ["class ", "def train", "def fit", "def optimizer", "def checkpoint"]
    for f in forbidden:
        assert f not in p


def test_p30_smoke_16_no_forbidden_claims_in_output():
    res = run_p30_constructor_binding_smoke()
    js = compact_json(res).lower()
    # Replace allowed fields before check
    cleaned = (js
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("constructor_binding_available_in_p30=false", ""))
    forbidden = ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]
    for claim in forbidden:
        assert claim not in cleaned
