# tests/test_phase2_p31_forward_boundary_smoke.py

import json
import pathlib
import pytest

from tools.phase2.run_p31_forward_boundary_smoke import (
    run_p31_forward_boundary_smoke,
    compact_json,
    main,
)


# 1. compact_json sorts keys
def test_p31_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'


# 2. main rejects args
def test_p31_smoke_02_main_rejects_args():
    import sys
    orig = sys.argv
    try:
        sys.argv = ["script.py", "extra"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig


# 3. smoke returns PASS
def test_p31_smoke_03_verdict_pass():
    res = run_p31_forward_boundary_smoke()
    assert res["verdict"] == "PASS"


# 4. contract matches
def test_p31_smoke_04_contract_matches():
    res = run_p31_forward_boundary_smoke()
    assert res["contract"] == "phase2_p31_noop_forward_boundary_contract_v1"


# 5. source_phase P31
def test_p31_smoke_05_source_phase():
    res = run_p31_forward_boundary_smoke()
    assert res["source_phase"] == "P31"


# 6. torch_required_for_p31 False
def test_p31_smoke_06_torch_required_p31_false():
    res = run_p31_forward_boundary_smoke()
    assert res["torch_required_for_p31"] is False


# 7. torch_required_for_future_execution True
def test_p31_smoke_07_torch_required_future_true():
    res = run_p31_forward_boundary_smoke()
    assert res["torch_required_for_future_execution"] is True


# 8. execution_attempted False
def test_p31_smoke_08_execution_attempted_false():
    res = run_p31_forward_boundary_smoke()
    assert res["execution_attempted"] is False


# 9. tensor_allocation_attempted False
def test_p31_smoke_09_tensor_attempted_false():
    res = run_p31_forward_boundary_smoke()
    assert res["tensor_allocation_attempted"] is False


# 10. output_generation_attempted False
def test_p31_smoke_10_output_attempted_false():
    res = run_p31_forward_boundary_smoke()
    assert res["output_generation_attempted"] is False


# 11. module_object_returned False
def test_p31_smoke_11_module_object_returned_false():
    res = run_p31_forward_boundary_smoke()
    assert res["module_object_returned"] is False


# 12. generated_output False
def test_p31_smoke_12_generated_output_false():
    res = run_p31_forward_boundary_smoke()
    assert res["generated_output"] is False


# 13. forward/tensor/output/training/loss/optimizer/checkpoint/artifact unavailable
def test_p31_smoke_13_all_unavailable():
    res = run_p31_forward_boundary_smoke()
    assert res["forward_execution_available_in_p31"] is False
    assert res["tensor_allocation_available_in_p31"] is False
    assert res["output_generation_available_in_p31"] is False
    assert res["training_available_in_p31"] is False
    assert res["loss_available_in_p31"] is False
    assert res["optimizer_available_in_p31"] is False
    assert res["checkpointing_available_in_p31"] is False
    assert res["artifact_generation_available_in_p31"] is False


# 14. declared_total_output_dim equals 13
def test_p31_smoke_14_declared_total_output_dim():
    res = run_p31_forward_boundary_smoke()
    assert res["declared_total_output_dim"] == 13


# 15. JSON parseable
def test_p31_smoke_15_json_parseable():
    res = run_p31_forward_boundary_smoke()
    js = compact_json(res)
    d = json.loads(js)
    assert d["verdict"] == "PASS"


# 16. smoke script no torch import
def test_p31_smoke_16_smoke_script_no_torch():
    p = pathlib.Path("tools/phase2/run_p31_forward_boundary_smoke.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p


# 17. smoke script no argparse
def test_p31_smoke_17_smoke_script_no_argparse():
    p = pathlib.Path("tools/phase2/run_p31_forward_boundary_smoke.py").read_text(encoding="utf-8")
    assert "argparse" not in p


# 18. smoke script no subprocess
def test_p31_smoke_18_smoke_script_no_subprocess():
    p = pathlib.Path("tools/phase2/run_p31_forward_boundary_smoke.py").read_text(encoding="utf-8")
    assert "subprocess" not in p


# 19. smoke output no success/scientific claims
def test_p31_smoke_19_no_claims_in_output():
    res = run_p31_forward_boundary_smoke()
    js = compact_json(res).lower()
    cleaned = (js
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("noop_forward_blocked_in_p31", "")
               .replace("forward_execution_available_in_p31=false", ""))
    for claim in ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]:
        assert claim not in cleaned
