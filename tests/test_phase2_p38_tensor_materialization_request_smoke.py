# tests/test_phase2_p38_tensor_materialization_request_smoke.py

import json
import pathlib
import pytest
import sys

from tools.phase2.run_p38_tensor_materialization_request_smoke import (
    run_p38_tensor_materialization_request_smoke,
    compact_json,
    main,
)

# 1. compact_json sorts keys
def test_p38_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'

# 2. main rejects args
def test_p38_smoke_02_main_rejects_args():
    orig = sys.argv
    try:
        sys.argv = ["script.py", "extra"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig

# 3. smoke returns PASS
def test_p38_smoke_03_verdict_pass():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["verdict"] == "PASS"

# 4. contract exact
def test_p38_smoke_04_contract_exact():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["contract"] == "phase2_p38_tensor_materialization_request_contract_v1"

# 5. source_phase P38
def test_p38_smoke_05_source_phase():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["source_phase"] == "P38"

# 6. request_kind exact
def test_p38_smoke_06_request_kind():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["request_kind"] == "tensor_materialization_request_metadata_no_tensor_allocation"

# 7. target_framework torch
def test_p38_smoke_07_target_framework():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["target_framework"] == "torch"

# 8. target_dtype_intent float32
def test_p38_smoke_08_target_dtype_intent():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["target_dtype_intent"] == "float32"

# 9. target_device_intent cpu
def test_p38_smoke_09_target_device_intent():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["target_device_intent"] == "cpu"

# 10. target_layout_kind row_major_2d_batch_tensor
def test_p38_smoke_10_target_layout_kind():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["target_layout_kind"] == "row_major_2d_batch_tensor"

# 11. shape_tuple [2, 32]
def test_p38_smoke_11_shape_tuple():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["shape_tuple"] == [2, 32]

# 12. batch_size 2
def test_p38_smoke_12_batch_size():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["batch_size"] == 2

# 13. input_flat_dim 32
def test_p38_smoke_13_input_flat_dim():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["input_flat_dim"] == 32

# 14. full_vector_length 64
def test_p38_smoke_14_full_vector_length():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["full_vector_length"] == 64

# 15. source_nested_row_count 2
def test_p38_smoke_15_source_nested_row_count():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["source_nested_row_count"] == 2

# 16. source_nested_row_lengths [32, 32]
def test_p38_smoke_16_source_nested_row_lengths():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["source_nested_row_lengths"] == [32, 32]

# 17. source_total_scalar_count 64
def test_p38_smoke_17_source_total_scalar_count():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["source_total_scalar_count"] == 64

# 18. source_flattened_matches_p35 True
def test_p38_smoke_18_source_flattened_matches_p35():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["source_flattened_matches_p35"] is True

# 19. requires_grad False
def test_p38_smoke_19_requires_grad():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["requires_grad"] is False

# 20. torch_backend_name
def test_p38_smoke_20_torch_backend_name():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["torch_backend_name"] == "torch"

# 21. torch_policy
def test_p38_smoke_21_torch_policy():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["torch_policy"] == "optional"

# 22. torch_available
def test_p38_smoke_22_torch_available():
    res = run_p38_tensor_materialization_request_smoke()
    assert "torch_available" in res
    assert isinstance(res["torch_available"], bool)

# 23. torch_import_safe
def test_p38_smoke_23_torch_import_safe():
    res = run_p38_tensor_materialization_request_smoke()
    assert "torch_import_safe" in res
    assert isinstance(res["torch_import_safe"], bool)

# 24. top_level_torch_import_required False
def test_p38_smoke_24_top_level_torch_import_required():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["top_level_torch_import_required"] is False

# 25. nested_values_available_in_p37 True
def test_p38_smoke_25_nested_values_available_in_p37():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["nested_values_available_in_p37"] is True

# 26. nested_values_materialized_in_p37 True
def test_p38_smoke_26_nested_values_materialized_in_p37():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["nested_values_materialized_in_p37"] is True

# 27. nested_values_status
def test_p38_smoke_27_nested_values_status():
    res = run_p38_tensor_materialization_request_smoke()
    assert isinstance(res["nested_values_status"], str)

# 28. status
def test_p38_smoke_28_status():
    res = run_p38_tensor_materialization_request_smoke()
    assert isinstance(res["status"], str)

# 29. tensor_request_available_in_p38 True
def test_p38_smoke_29_tensor_request_available_in_p38():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["tensor_request_available_in_p38"] is True

# 30. tensor_materialization_available_in_p38 False
def test_p38_smoke_30_tensor_materialization_available_in_p38():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["tensor_materialization_available_in_p38"] is False

# 31. array_materialization_available_in_p38 False
def test_p38_smoke_31_array_materialization_available_in_p38():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["array_materialization_available_in_p38"] is False

# 32. forward_execution_available_in_p38 False
def test_p38_smoke_32_forward_execution_available_in_p38():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["forward_execution_available_in_p38"] is False

# 33. output_generation_available_in_p38 False
def test_p38_smoke_33_output_generation_available_in_p38():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["output_generation_available_in_p38"] is False

# 34. training_available_in_p38 False
def test_p38_smoke_34_training_available_in_p38():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["training_available_in_p38"] is False

# 35. attempted flags all false
def test_p38_smoke_35_attempted_flags():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["tensor_materialization_attempted"] is False
    assert res["array_materialization_attempted"] is False
    assert res["forward_execution_attempted"] is False
    assert res["output_generation_attempted"] is False
    assert res["training_attempted"] is False

# 36. no flags all true
def test_p38_smoke_36_no_flags():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["no_tensor_created"] is True
    assert res["no_array_created"] is True
    assert res["no_forward_execution"] is True
    assert res["no_output_generation"] is True
    assert res["no_training_loop"] is True
    assert res["no_optimizer"] is True
    assert res["no_checkpointing"] is True
    assert res["no_artifact_generation"] is True
    assert res["no_final_comparison"] is True
    assert res["no_scientific_conclusion"] is True

# 37. JSON parseable
def test_p38_smoke_37_json_parseable():
    res = run_p38_tensor_materialization_request_smoke()
    js = compact_json(res)
    parsed = json.loads(js)
    assert parsed["verdict"] == "PASS"

# 38. smoke output has no nested_batch_values
def test_p38_smoke_38_no_nested_batch_values():
    res = run_p38_tensor_materialization_request_smoke()
    # Check key absence in the dictionary and its nested blocks
    assert "nested_batch_values" not in res
    assert "nested_batch_values" not in res["result"]
    assert "nested_batch_values" not in res["result"]["request"]
    assert "nested_batch_values" not in res["result"]["spec"]
    assert "nested_batch_values" not in res["result"]["metadata"]

# 39. smoke output has no flat_values
def test_p38_smoke_39_no_flat_values():
    res = run_p38_tensor_materialization_request_smoke()
    assert "flat_values" not in res
    assert "flat_values" not in res["result"]
    assert "flat_values" not in res["result"]["request"]
    assert "flat_values" not in res["result"]["spec"]
    assert "flat_values" not in res["result"]["metadata"]

# 40. smoke output has no key with ": bool"
def test_p38_smoke_40_no_colon_bool():
    res = run_p38_tensor_materialization_request_smoke()
    js = compact_json(res)
    assert ": bool" not in js

# 41. smoke output has no success/scientific claims
def test_p38_smoke_41_no_forbidden_claims():
    res = run_p38_tensor_materialization_request_smoke()
    js = compact_json(res)
    # the words solved, best, winner, state of the art, etc. should not be in the serialized JSON
    forbidden = ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]
    for word in forbidden:
        assert word not in js.lower()

# 42. smoke script imports clean
def test_p38_smoke_42_imports_clean():
    content = pathlib.Path("tools/phase2/run_p38_tensor_materialization_request_smoke.py").read_text(encoding="utf-8")
    for pattern in (
        "import torch", "from torch",
        "import numpy", "from numpy",
        "import random", "from random",
        "import secrets", "from secrets",
        "import array", "from array",
        "import argparse", "from argparse",
        "import subprocess", "from subprocess",
    ):
        assert pattern not in content

# 43. result serialized valid
def test_p38_smoke_43_result_serialized():
    res = run_p38_tensor_materialization_request_smoke()
    assert "result" in res
    assert isinstance(res["result"], dict)

# 44. reason exact
def test_p38_smoke_44_reason_exact():
    res = run_p38_tensor_materialization_request_smoke()
    assert res["reason"] == "p38_tensor_materialization_request_smoke_completed"

# 45. main no args return code
def test_p38_smoke_45_main_no_args():
    orig = sys.argv
    try:
        sys.argv = ["script.py"]
        code = main()
        assert code == 0
    finally:
        sys.argv = orig
