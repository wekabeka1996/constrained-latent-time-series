# tests/test_phase2_p35_fake_input_flat_vector_smoke.py

import json
import pathlib
import pytest

from tools.phase2.run_p35_fake_input_flat_vector_smoke import (
    run_p35_fake_input_flat_vector_smoke,
    compact_json,
    main,
)

# 1. compact_json sorts keys
def test_p35_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'

# 2. main rejects args
def test_p35_smoke_02_main_rejects_args():
    import sys
    orig = sys.argv
    try:
        sys.argv = ["script.py", "extra"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig

# 3. smoke returns PASS
def test_p35_smoke_03_verdict_pass():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["verdict"] == "PASS"

# 4. contract matches
def test_p35_smoke_04_contract_matches():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["contract"] == "phase2_p35_fake_input_flat_vector_contract_v1"

# 5. source_phase P35
def test_p35_smoke_05_source_phase():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["source_phase"] == "P35"

# 6. descriptor_seed == 1337
def test_p35_smoke_06_descriptor_seed():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["descriptor_seed"] == 1337

# 7. min_value == -1.0
def test_p35_smoke_07_min_value():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["min_value"] == -1.0

# 8. max_value == 1.0
def test_p35_smoke_08_max_value():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["max_value"] == 1.0

# 9. batch_size == 2
def test_p35_smoke_09_batch_size():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["batch_size"] == 2

# 10. input_flat_dim == 32
def test_p35_smoke_10_input_flat_dim():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["input_flat_dim"] == 32

# 11. full_vector_length == 64
def test_p35_smoke_11_full_vector_length():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["full_vector_length"] == 64

# 12. flat_values_count == 64
def test_p35_smoke_12_flat_values_count():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["flat_values_count"] == 64

# 13. flat_values is a list of 64 floats
def test_p35_smoke_13_flat_values_format():
    res = run_p35_fake_input_flat_vector_smoke()
    assert isinstance(res["flat_values"], list)
    assert len(res["flat_values"]) == 64
    for val in res["flat_values"]:
        assert isinstance(val, float)

# 14. every flat value is within range
def test_p35_smoke_14_flat_values_range():
    res = run_p35_fake_input_flat_vector_smoke()
    for val in res["flat_values"]:
        assert val >= -1.0
        assert val <= 1.0

# 15. first 4 flat values match P34 preview
def test_p35_smoke_15_first_four_values_match_p34():
    res = run_p35_fake_input_flat_vector_smoke()
    expected = [-0.24297189, -0.1686747, -0.09437751, -0.02008032]
    assert res["flat_values"][:4] == expected

# 16. is_flat_vector is True
def test_p35_smoke_16_is_flat_vector():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["is_flat_vector"] is True

# 17. is_2d_batch is False
def test_p35_smoke_17_is_2d_batch():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["is_2d_batch"] is False

# 18. all_values_within_range is True
def test_p35_smoke_18_all_values_within_range():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["all_values_within_range"] is True

# 19. torch_available is a bool
def test_p35_smoke_19_torch_available():
    res = run_p35_fake_input_flat_vector_smoke()
    assert isinstance(res["torch_available"], bool)

# 20. preview_status is a non-empty string
def test_p35_smoke_20_preview_status():
    res = run_p35_fake_input_flat_vector_smoke()
    assert isinstance(res["preview_status"], str)
    assert res["preview_status"] != ""

# 21. status matches expectations
def test_p35_smoke_21_status():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["status"] in ("blocked_torch_unavailable", "flat_vector_only_no_2d_batch_in_p35")

# 22. flat_vector_available_in_p35 is True
def test_p35_smoke_22_flat_vector_available():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["flat_vector_available_in_p35"] is True

# 23. two_d_batch_available_in_p35 is False
def test_p35_smoke_23_two_d_batch_available():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["two_d_batch_available_in_p35"] is False

# 24. rng_execution_attempted is False
def test_p35_smoke_24_rng_execution_attempted():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["rng_execution_attempted"] is False

# 25. array_materialization_attempted is False
def test_p35_smoke_25_array_materialization_attempted():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["array_materialization_attempted"] is False

# 26. tensor_materialization_attempted is False
def test_p35_smoke_26_tensor_materialization_attempted():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["tensor_materialization_attempted"] is False

# 27. forward_execution_attempted is False
def test_p35_smoke_27_forward_execution_attempted():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["forward_execution_attempted"] is False

# 28. output_generation_attempted is False
def test_p35_smoke_28_output_generation_attempted():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["output_generation_attempted"] is False

# 29. rng_executed is False
def test_p35_smoke_29_rng_executed():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["rng_executed"] is False

# 30. array_materialized is False
def test_p35_smoke_30_array_materialized():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["array_materialized"] is False

# 31. tensor_materialized is False
def test_p35_smoke_31_tensor_materialized():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["tensor_materialized"] is False

# 32. forward_executed is False
def test_p35_smoke_32_forward_executed():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["forward_executed"] is False

# 33. no_rng_execution is True
def test_p35_smoke_33_no_rng_execution():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["no_rng_execution"] is True

# 34. no_2d_batch_materialized is True
def test_p35_smoke_34_no_2d_batch_materialized():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["no_2d_batch_materialized"] is True

# 35. no_array_created is True
def test_p35_smoke_35_no_array_created():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["no_array_created"] is True

# 36. no_tensor_created is True
def test_p35_smoke_36_no_tensor_created():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["no_tensor_created"] is True

# 37. no_forward_execution is True
def test_p35_smoke_37_no_forward_execution():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["no_forward_execution"] is True

# 38. no_output_generation is True
def test_p35_smoke_38_no_output_generation():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["no_output_generation"] is True

# 39. no_training_loop and no_optimizer is True
def test_p35_smoke_39_no_training_and_optimizer():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["no_training_loop"] is True
    assert res["no_optimizer"] is True

# 40. no_checkpointing, no_artifact_generation, no_final_comparison, no_scientific_conclusion are True
def test_p35_smoke_40_no_checkpointing_and_conclusions():
    res = run_p35_fake_input_flat_vector_smoke()
    assert res["no_checkpointing"] is True
    assert res["no_artifact_generation"] is True
    assert res["no_final_comparison"] is True
    assert res["no_scientific_conclusion"] is True

# 41. smoke output has no nested 2D values
def test_p35_smoke_41_no_nested_2d():
    res = run_p35_fake_input_flat_vector_smoke()
    js = compact_json(res)
    assert "[[" not in js
