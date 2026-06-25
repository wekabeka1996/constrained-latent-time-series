# tests/test_phase2_p32_forward_input_batch_smoke.py

import json
import pathlib
import pytest

from tools.phase2.run_p32_forward_input_batch_smoke import (
    run_p32_forward_input_batch_smoke,
    compact_json,
    main,
)


# 1. compact_json sorts keys
def test_p32_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'


# 2. main rejects args
def test_p32_smoke_02_main_rejects_args():
    import sys
    orig = sys.argv
    try:
        sys.argv = ["script.py", "extra"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig


# 3. smoke returns PASS
def test_p32_smoke_03_verdict_pass():
    res = run_p32_forward_input_batch_smoke()
    assert res["verdict"] == "PASS"


# 4. contract matches
def test_p32_smoke_04_contract_matches():
    res = run_p32_forward_input_batch_smoke()
    assert res["contract"] == "phase2_p32_forward_input_batch_contract_v1"


# 5. source_phase P32
def test_p32_smoke_05_source_phase():
    res = run_p32_forward_input_batch_smoke()
    assert res["source_phase"] == "P32"


# 6. batch_size == 2
def test_p32_smoke_06_batch_size():
    res = run_p32_forward_input_batch_smoke()
    assert res["batch_size"] == 2


# 7. input_flat_dim == 32
def test_p32_smoke_07_input_flat_dim():
    res = run_p32_forward_input_batch_smoke()
    assert res["input_flat_dim"] == 32


# 8. rank == 2
def test_p32_smoke_08_rank():
    res = run_p32_forward_input_batch_smoke()
    assert res["rank"] == 2


# 9. shape_tuple == [2, 32]
def test_p32_smoke_09_shape_tuple():
    res = run_p32_forward_input_batch_smoke()
    assert res["shape_tuple"] == [2, 32]


# 10. batch_shape_declared True
def test_p32_smoke_10_batch_shape_declared():
    res = run_p32_forward_input_batch_smoke()
    assert res["batch_shape_declared"] is True


# 11. input_batch_available_in_p32 False
def test_p32_smoke_11_input_batch_available():
    res = run_p32_forward_input_batch_smoke()
    assert res["input_batch_available_in_p32"] is False


# 12. tensor_materialization_attempted False
def test_p32_smoke_12_tensor_attempted_false():
    res = run_p32_forward_input_batch_smoke()
    assert res["tensor_materialization_attempted"] is False


# 13. array_materialization_attempted False
def test_p32_smoke_13_array_attempted_false():
    res = run_p32_forward_input_batch_smoke()
    assert res["array_materialization_attempted"] is False


# 14. values_materialization_attempted False
def test_p32_smoke_14_values_attempted_false():
    res = run_p32_forward_input_batch_smoke()
    assert res["values_materialization_attempted"] is False


# 15. forward_execution_attempted False
def test_p32_smoke_15_forward_attempted_false():
    res = run_p32_forward_input_batch_smoke()
    assert res["forward_execution_attempted"] is False


# 16. output_generation_attempted False
def test_p32_smoke_16_output_attempted_false():
    res = run_p32_forward_input_batch_smoke()
    assert res["output_generation_attempted"] is False


# 17. tensor/array/values materialized False
def test_p32_smoke_17_materialized_false():
    res = run_p32_forward_input_batch_smoke()
    assert res["tensor_materialized"] is False
    assert res["array_materialized"] is False
    assert res["values_materialized"] is False


# 18. all no_* flags True
def test_p32_smoke_18_no_flags_true():
    res = run_p32_forward_input_batch_smoke()
    assert res["no_tensor_created"] is True
    assert res["no_array_created"] is True
    assert res["no_values_materialized"] is True
    assert res["no_forward_execution"] is True
    assert res["no_output_generation"] is True
    assert res["no_training_loop"] is True
    assert res["no_optimizer"] is True
    assert res["no_checkpointing"] is True
    assert res["no_artifact_generation"] is True
    assert res["no_final_comparison"] is True
    assert res["no_scientific_conclusion"] is True


# 19. JSON parseable
def test_p32_smoke_19_json_parseable():
    res = run_p32_forward_input_batch_smoke()
    js = compact_json(res)
    d = json.loads(js)
    assert d["verdict"] == "PASS"


# 20. smoke script no torch import
def test_p32_smoke_20_no_torch_import():
    p = pathlib.Path("tools/phase2/run_p32_forward_input_batch_smoke.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p


# 21. smoke script no numpy import
def test_p32_smoke_21_no_numpy_import():
    p = pathlib.Path("tools/phase2/run_p32_forward_input_batch_smoke.py").read_text(encoding="utf-8")
    assert "import numpy" not in p
    assert "from numpy" not in p


# 22. smoke script no argparse
def test_p32_smoke_22_no_argparse():
    p = pathlib.Path("tools/phase2/run_p32_forward_input_batch_smoke.py").read_text(encoding="utf-8")
    assert "argparse" not in p


# 23. smoke script no subprocess
def test_p32_smoke_23_no_subprocess():
    p = pathlib.Path("tools/phase2/run_p32_forward_input_batch_smoke.py").read_text(encoding="utf-8")
    assert "subprocess" not in p


# 24. smoke output no success/scientific claims
def test_p32_smoke_24_no_claims_in_output():
    res = run_p32_forward_input_batch_smoke()
    js = compact_json(res).lower()
    cleaned = (js
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("input_batch_materialization_blocked_in_p32", ""))
    for claim in ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]:
        assert claim not in cleaned
