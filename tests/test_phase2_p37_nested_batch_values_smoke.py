# tests/test_phase2_p37_nested_batch_values_smoke.py

import json
import pathlib
import pytest

from tools.phase2.run_p37_nested_batch_values_smoke import (
    run_p37_nested_batch_values_smoke,
    compact_json,
    main,
)

# 1. compact_json sorts keys
def test_p37_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'

# 2. main rejects args
def test_p37_smoke_02_main_rejects_args():
    import sys
    orig = sys.argv
    try:
        sys.argv = ["script.py", "extra"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig

# 3. smoke returns PASS
def test_p37_smoke_03_verdict_pass():
    res = run_p37_nested_batch_values_smoke()
    assert res["verdict"] == "PASS"

# 4. contract exact
def test_p37_smoke_04_contract_exact():
    res = run_p37_nested_batch_values_smoke()
    assert res["contract"] == "phase2_p37_nested_batch_values_contract_v1"

# 5. source_phase P37
def test_p37_smoke_05_source_phase():
    res = run_p37_nested_batch_values_smoke()
    assert res["source_phase"] == "P37"

# 6. nested_values_kind exact
def test_p37_smoke_06_nested_values_kind():
    res = run_p37_nested_batch_values_smoke()
    assert res["nested_values_kind"] == "controlled_nested_python_tuple_batch_values_no_tensor_no_array"

# 7. order_kind exact
def test_p37_smoke_07_order_kind():
    res = run_p37_nested_batch_values_smoke()
    assert res["order_kind"] == "row_major_flat_to_nested_tuple"

# 8. shape_tuple [2, 32]
def test_p37_smoke_08_shape_tuple():
    res = run_p37_nested_batch_values_smoke()
    assert res["shape_tuple"] == [2, 32]

# 9. batch_size 2
def test_p37_smoke_09_batch_size():
    res = run_p37_nested_batch_values_smoke()
    assert res["batch_size"] == 2

# 10. input_flat_dim 32
def test_p37_smoke_10_input_flat_dim():
    res = run_p37_nested_batch_values_smoke()
    assert res["input_flat_dim"] == 32

# 11. full_vector_length 64
def test_p37_smoke_11_full_vector_length():
    res = run_p37_nested_batch_values_smoke()
    assert res["full_vector_length"] == 64

# 12. row_count 2
def test_p37_smoke_12_row_count():
    res = run_p37_nested_batch_values_smoke()
    assert res["row_count"] == 2

# 13. row_lengths [32, 32]
def test_p37_smoke_13_row_lengths():
    res = run_p37_nested_batch_values_smoke()
    assert res["row_lengths"] == [32, 32]

# 14. total_scalar_count 64
def test_p37_smoke_14_total_scalar_count():
    res = run_p37_nested_batch_values_smoke()
    assert res["total_scalar_count"] == 64

# 15. first row first 4 values match P35/P34 preview values
def test_p37_smoke_15_first_row_first_4_values():
    res = run_p37_nested_batch_values_smoke()
    expected = [-0.24297189, -0.1686747, -0.09437751, -0.02008032]
    assert res["first_row_first_4_values"] == expected

# 16. second row first 4 values match expected P35 flat values positions 32–35
def test_p37_smoke_16_second_row_first_4_values():
    res = run_p37_nested_batch_values_smoke()
    expected = [0.13253012, 0.20682731, 0.2811245, 0.35542169]
    assert res["second_row_first_4_values"] == expected

# 17. flattened_matches_source_flat_vector True
def test_p37_smoke_17_flattened_matches():
    res = run_p37_nested_batch_values_smoke()
    assert res["flattened_matches_source_flat_vector"] is True

# 18. all_values_are_exact_float True
def test_p37_smoke_18_all_values_are_exact_float():
    res = run_p37_nested_batch_values_smoke()
    assert res["all_values_are_exact_float"] is True

# 19. all_values_within_source_range True
def test_p37_smoke_19_all_values_within_source_range():
    res = run_p37_nested_batch_values_smoke()
    assert res["all_values_within_source_range"] is True

# 20. flat_vector_status matches P35 status
def test_p37_smoke_20_flat_vector_status():
    res = run_p37_nested_batch_values_smoke()
    assert res["flat_vector_status"] in ("blocked_torch_unavailable", "flat_vector_only_no_2d_batch_in_p35")

# 21. batch_view_status matches P36 status
def test_p37_smoke_21_batch_view_status():
    res = run_p37_nested_batch_values_smoke()
    assert res["batch_view_status"] in ("blocked_torch_unavailable", "view_metadata_only_no_2d_values_in_p36")

# 22. torch_available is bool
def test_p37_smoke_22_torch_available():
    res = run_p37_nested_batch_values_smoke()
    assert isinstance(res["torch_available"], bool)

# 23. status matches expectation
def test_p37_smoke_23_status():
    res = run_p37_nested_batch_values_smoke()
    assert res["status"] in ("blocked_torch_unavailable", "nested_values_only_no_tensor_no_forward_in_p37")

# 24. nested_values_available_in_p37 True
def test_p37_smoke_24_nested_values_available():
    res = run_p37_nested_batch_values_smoke()
    assert res["nested_values_available_in_p37"] is True

# 25. array_materialization_available_in_p37 False
def test_p37_smoke_25_array_avail():
    res = run_p37_nested_batch_values_smoke()
    assert res["array_materialization_available_in_p37"] is False

# 26. tensor_materialization_available_in_p37 False
def test_p37_smoke_26_tensor_avail():
    res = run_p37_nested_batch_values_smoke()
    assert res["tensor_materialization_available_in_p37"] is False

# 27. forward_execution_available_in_p37 False
def test_p37_smoke_27_forward_avail():
    res = run_p37_nested_batch_values_smoke()
    assert res["forward_execution_available_in_p37"] is False

# 28. output_generation_available_in_p37 False
def test_p37_smoke_28_output_avail():
    res = run_p37_nested_batch_values_smoke()
    assert res["output_generation_available_in_p37"] is False

# 29. nested_values_materialized True
def test_p37_smoke_29_nested_values_materialized():
    res = run_p37_nested_batch_values_smoke()
    assert res["nested_values_materialized"] is True

# 30. array_materialization_attempted False
def test_p37_smoke_30_array_attempted():
    res = run_p37_nested_batch_values_smoke()
    assert res["array_materialization_attempted"] is False

# 31. tensor_materialization_attempted False
def test_p37_smoke_31_tensor_attempted():
    res = run_p37_nested_batch_values_smoke()
    assert res["tensor_materialization_attempted"] is False

# 32. forward_execution_attempted False
def test_p37_smoke_32_forward_attempted():
    res = run_p37_nested_batch_values_smoke()
    assert res["forward_execution_attempted"] is False

# 33. output_generation_attempted False
def test_p37_smoke_33_output_attempted():
    res = run_p37_nested_batch_values_smoke()
    assert res["output_generation_attempted"] is False

# 34. no_array_created and no_tensor_created are True
def test_p37_smoke_34_no_array_no_tensor():
    res = run_p37_nested_batch_values_smoke()
    assert res["no_array_created"] is True
    assert res["no_tensor_created"] is True

# 35. no_forward_execution and no_output_generation are True
def test_p37_smoke_35_no_forward_no_output():
    res = run_p37_nested_batch_values_smoke()
    assert res["no_forward_execution"] is True
    assert res["no_output_generation"] is True

# 36. no_training_loop and no_optimizer are True
def test_p37_smoke_36_no_train_no_opt():
    res = run_p37_nested_batch_values_smoke()
    assert res["no_training_loop"] is True
    assert res["no_optimizer"] is True

# 37. remaining no_* flags are True
def test_p37_smoke_37_no_checkpoint_no_artifact():
    res = run_p37_nested_batch_values_smoke()
    assert res["no_checkpointing"] is True
    assert res["no_artifact_generation"] is True
    assert res["no_final_comparison"] is True
    assert res["no_scientific_conclusion"] is True

# 38. JSON parseable
def test_p37_smoke_38_json_parseable():
    res = run_p37_nested_batch_values_smoke()
    js = compact_json(res)
    d = json.loads(js)
    assert d["verdict"] == "PASS"

# 39. smoke script has no forbidden imports
def test_p37_smoke_39_smoke_imports():
    content = pathlib.Path("tools/phase2/run_p37_nested_batch_values_smoke.py").read_text(encoding="utf-8")
    for pattern in ("import torch", "import numpy", "import random", "import secrets", "import array", "import argparse", "import subprocess"):
        assert pattern not in content

# 40. smoke output has no key with ": bool"
def test_p37_smoke_40_no_bool_key_pollution():
    res = run_p37_nested_batch_values_smoke()
    js = compact_json(res)
    assert ": bool" not in js

# 41. smoke output has no success/scientific claims
def test_p37_smoke_41_no_forbidden_claims():
    res = run_p37_nested_batch_values_smoke()
    js = compact_json(res)
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        assert claim not in js.lower()
