# tests/test_phase2_p36_flat_vector_batch_view_smoke.py

import json
import pathlib
import pytest

from tools.phase2.run_p36_flat_vector_batch_view_smoke import (
    run_p36_flat_vector_batch_view_smoke,
    compact_json,
    main,
)

# 1. compact_json sorts keys
def test_p36_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'

# 2. main rejects args
def test_p36_smoke_02_main_rejects_args():
    import sys
    orig = sys.argv
    try:
        sys.argv = ["script.py", "extra"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig

# 3. smoke returns PASS
def test_p36_smoke_03_verdict_pass():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["verdict"] == "PASS"

# 4. contract matches
def test_p36_smoke_04_contract_matches():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["contract"] == "phase2_p36_flat_vector_2d_batch_view_contract_v1"

# 5. source_phase P36
def test_p36_smoke_05_source_phase():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["source_phase"] == "P36"

# 6. view_kind matches
def test_p36_smoke_06_view_kind():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["view_kind"] == "flat_vector_2d_batch_view_metadata_no_nested_values"

# 7. view_order_kind matches
def test_p36_smoke_07_view_order_kind():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["view_order_kind"] == "row_major_flat_index_view"

# 8. view_rank matches
def test_p36_smoke_08_view_rank():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["view_rank"] == 2

# 9. shape_tuple matches
def test_p36_smoke_09_shape_tuple():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["shape_tuple"] == [2, 32]

# 10. batch_size matches
def test_p36_smoke_10_batch_size():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["batch_size"] == 2

# 11. input_flat_dim matches
def test_p36_smoke_11_input_flat_dim():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["input_flat_dim"] == 32

# 12. full_vector_length matches
def test_p36_smoke_12_full_vector_length():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["full_vector_length"] == 64

# 13. row_major_formula matches
def test_p36_smoke_13_row_major_formula():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["row_major_formula"] == "flat_index = batch_index * input_flat_dim + feature_index"

# 14. example_flat_index values
def test_p36_smoke_14_example_indices():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["example_flat_index_0_0"] == 0
    assert res["example_flat_index_0_31"] == 31
    assert res["example_flat_index_1_0"] == 32
    assert res["example_flat_index_1_31"] == 63

# 15. flat_vector_status matches P35 expectations
def test_p36_smoke_15_flat_vector_status():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["flat_vector_status"] in ("blocked_torch_unavailable", "flat_vector_only_no_2d_batch_in_p35")

# 16. torch_available is boolean
def test_p36_smoke_16_torch_available():
    res = run_p36_flat_vector_batch_view_smoke()
    assert isinstance(res["torch_available"], bool)

# 17. status matches
def test_p36_smoke_17_status():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["status"] in ("blocked_torch_unavailable", "view_metadata_only_no_2d_values_in_p36")

# 18. batch_view_available_in_p36 is True
def test_p36_smoke_18_batch_view_available():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["batch_view_available_in_p36"] is True

# 19. nested_values_available_in_p36 is False
def test_p36_smoke_19_nested_values_available():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["nested_values_available_in_p36"] is False

# 20. two_d_batch_available_in_p36 is False
def test_p36_smoke_20_two_d_batch_available():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["two_d_batch_available_in_p36"] is False

# 21. array_materialization_available_in_p36 is False
def test_p36_smoke_21_array_materialization_available():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["array_materialization_available_in_p36"] is False

# 22. tensor_materialization_available_in_p36 is False
def test_p36_smoke_22_tensor_materialization_available():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["tensor_materialization_available_in_p36"] is False

# 23. forward_execution_available_in_p36 is False
def test_p36_smoke_23_forward_execution_available():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["forward_execution_available_in_p36"] is False

# 24. output_generation_available_in_p36 is False
def test_p36_smoke_24_output_generation_available():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["output_generation_available_in_p36"] is False

# 25. flat_vector_reused_without_copy is True
def test_p36_smoke_25_flat_vector_reused():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["flat_vector_reused_without_copy"] is True

# 26. nested_values_attempted is False
def test_p36_smoke_26_nested_values_attempted():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["nested_values_attempted"] is False

# 27. two_d_batch_materialization_attempted is False
def test_p36_smoke_27_two_d_batch_attempted():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["two_d_batch_materialization_attempted"] is False

# 28. array_materialization_attempted is False
def test_p36_smoke_28_array_attempted():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["array_materialization_attempted"] is False

# 29. tensor_materialization_attempted is False
def test_p36_smoke_29_tensor_attempted():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["tensor_materialization_attempted"] is False

# 30. forward_execution_attempted is False
def test_p36_smoke_30_forward_attempted():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["forward_execution_attempted"] is False

# 31. output_generation_attempted is False
def test_p36_smoke_31_output_attempted():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["output_generation_attempted"] is False

# 32. no_nested_values is True
def test_p36_smoke_32_no_nested_values():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["no_nested_values"] is True

# 33. no_2d_batch_materialized is True
def test_p36_smoke_33_no_2d_batch_materialized():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["no_2d_batch_materialized"] is True

# 34. no_array_created is True
def test_p36_smoke_34_no_array_created():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["no_array_created"] is True

# 35. no_tensor_created is True
def test_p36_smoke_35_no_tensor_created():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["no_tensor_created"] is True

# 36. no_forward_execution is True
def test_p36_smoke_36_no_forward_execution():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["no_forward_execution"] is True

# 37. no_output_generation is True
def test_p36_smoke_37_no_output_generation():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["no_output_generation"] is True

# 38. no_training_loop and no_optimizer are True
def test_p36_smoke_38_no_training_and_optimizer():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["no_training_loop"] is True
    assert res["no_optimizer"] is True

# 39. no_checkpointing, no_artifact_generation, no_final_comparison, no_scientific_conclusion are True
def test_p36_smoke_39_no_checkpointing_and_conclusions():
    res = run_p36_flat_vector_batch_view_smoke()
    assert res["no_checkpointing"] is True
    assert res["no_artifact_generation"] is True
    assert res["no_final_comparison"] is True
    assert res["no_scientific_conclusion"] is True

# 40. serialized result matches shape dict
def test_p36_smoke_40_serialized_result_structure():
    res = run_p36_flat_vector_batch_view_smoke()
    assert isinstance(res["result"], dict)
    assert res["result"]["contract_version"] == "phase2_p36_flat_vector_2d_batch_view_contract_v1"

# 41. smoke output has no nested 2D values or flat_values key
def test_p36_smoke_41_no_nested_2d_or_flat_values():
    res = run_p36_flat_vector_batch_view_smoke()
    js = compact_json(res)
    assert "[[" not in js
    assert "flat_values" not in js
